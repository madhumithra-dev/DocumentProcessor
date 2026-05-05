"""
Document Intelligence Processor
================================
Pipeline: OCR → Reading Order → Layout Detection → Agent (VLM tools)
"""

from __future__ import annotations

import base64
import json
import logging
import torch  # Ensure torch is imported first to lock DLL versions on Windows
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import openai
import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("doc_processor")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OCRRegion:
    text: str
    bbox: list[list[int]]      # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    confidence: float

    @property
    def bbox_xyxy(self) -> list[int]:
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return [min(xs), min(ys), max(xs), max(ys)]


@dataclass
class LayoutRegion:
    region_id: int
    region_type: str           # "table" | "figure" | "text" | "title" …
    bbox: list[int]            # [x1, y1, x2, y2]
    confidence: float


@dataclass
class OrderedTextEntry:
    position: int
    text: str
    confidence: float
    bbox: list[int]


@dataclass
class DocumentResult:
    ordered_text: list[OrderedTextEntry] = field(default_factory=list)
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    region_analyses: dict[int, dict[str, Any]] = field(default_factory=dict)

    def as_markdown(self) -> str:
        lines: list[str] = ["# Document Analysis Result\n"]

        lines.append("## Extracted Text (reading order)\n")
        for entry in self.ordered_text:
            lines.append(f"{entry.position:3}. {entry.text}")

        lines.append("\n## Layout Regions\n")
        for r in self.layout_regions:
            lines.append(
                f"- [{r.region_id}] **{r.region_type}** "
                f"(conf={r.confidence:.2f}) bbox={r.bbox}"
            )

        if self.region_analyses:
            lines.append("\n## Region Analyses (VLM)\n")
            for rid, analysis in self.region_analyses.items():
                lines.append(f"### Region {rid} – {analysis.get('type', '?')}")
                lines.append(json.dumps(analysis, indent=2, ensure_ascii=False))

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Text extraction via PaddleOCR
# ─────────────────────────────────────────────────────────────────────────────
class OCRExtractor:
    def __init__(self, lang: str = "en"):
        log.info("Initialising PaddleOCR (lang=%s)…", lang)
        from paddleocr import PaddleOCR  # lazy import
        self._ocr = PaddleOCR(lang=lang)

    def extract(self, image_path: str | Path) -> list[OCRRegion]:
        log.info("Running OCR on %s…", image_path)
        # PaddleOCR.ocr returns a list of pages, each being a list of [box, (text, score)]
        result = self._ocr.ocr(str(image_path), cls=True)
        
        if not result or result[0] is None:
            log.warning("OCR returned no results.")
            return []
            
        page = result[0]
        regions: list[OCRRegion] = []
        for line in page:
            box = line[0]
            text, score = line[1]
            regions.append(
                OCRRegion(
                    text=text,
                    bbox=box,
                    confidence=float(score),
                )
            )

        log.info("OCR extracted %d regions", len(regions))
        return regions


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Reading-order detection via LayoutReader
# ─────────────────────────────────────────────────────────────────────────────
class ReadingOrderDetector:
    MODEL_SLUG = "hantian/layoutreader"

    def __init__(self):
        log.info("Loading LayoutReader (%s)…", self.MODEL_SLUG)
        from transformers import LayoutLMv3ForTokenClassification  # lazy import
        self._model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.MODEL_SLUG
        )
        log.info("LayoutReader ready.")

    def order(self, ocr_regions: list[OCRRegion]) -> list[OrderedTextEntry]:
        from layoutreader_helpers import boxes2inputs, parse_logits, prepare_inputs

        if not ocr_regions:
            return []

        # Compute canvas dimensions with 10 % padding
        max_x = max_y = 0
        for r in ocr_regions:
            x1, y1, x2, y2 = r.bbox_xyxy
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

        W, H = max_x * 1.1, max_y * 1.1

        # Normalise to [0, 1000]
        norm_boxes: list[list[int]] = []
        for r in ocr_regions:
            x1, y1, x2, y2 = r.bbox_xyxy
            norm_boxes.append(
                [
                    int(x1 / W * 1000),
                    int(y1 / H * 1000),
                    int(x2 / W * 1000),
                    int(y2 / H * 1000),
                ]
            )

        inputs = boxes2inputs(norm_boxes)
        inputs = prepare_inputs(inputs, self._model)
        logits = self._model(**inputs).logits.cpu().squeeze(0)
        reading_order: list[int] = parse_logits(logits, len(norm_boxes))

        # Sort by reading position
        paired = sorted(
            zip(reading_order, range(len(ocr_regions))),
            key=lambda t: t[0],
        )

        return [
            OrderedTextEntry(
                position=pos,
                text=ocr_regions[idx].text,
                confidence=ocr_regions[idx].confidence,
                bbox=ocr_regions[idx].bbox_xyxy,
            )
            for pos, idx in paired
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Layout / region detection via PaddleOCR LayoutDetection
# ─────────────────────────────────────────────────────────────────────────────
class LayoutDetector:
    def __init__(self):
        log.info("Initialising PaddleOCR LayoutDetection…")
        
        from paddleocr import PPStructure
        self._engine = PPStructure(table=False, ocr=False, show_log=False)

    def detect(self, image_path: str | Path) -> list[LayoutRegion]:
        log.info("Detecting layout regions in %s…", image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            log.error("Could not read image: %s", image_path)
            return []

        result = self._engine(img)   # 2.8.x returns a list, not a tuple

        regions: list[LayoutRegion] = []
        for i, region in enumerate(result):
            regions.append(
                LayoutRegion(
                    region_id=i,
                    region_type=region["type"].lower(),
                    bbox=[int(c) for c in region["bbox"]],
                    confidence=float(region.get("score", 1.0)),
                )
            )

        regions.sort(key=lambda r: r.confidence, reverse=True)
        log.info("Detected %d layout regions", len(regions))
        return regions
# ─────────────────────────────────────────────────────────────────────────────
# Helpers – image utilities
# ─────────────────────────────────────────────────────────────────────────────
def crop_region(pil_img: Image.Image, bbox: list[int], padding: int = 10) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(pil_img.width, x2 + padding)
    y2 = min(pil_img.height, y2 + padding)
    return pil_img.crop((x1, y1, x2, y2))


def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def draw_regions(
    image_path: str | Path,
    layout_regions: list[LayoutRegion],
    output_path: str | Path | None = None,
) -> Image.Image:
    """Draw coloured bounding boxes for each detected region."""
    COLOR_MAP = {
        "table": (255, 165, 0),
        "figure": (0, 128, 255),
        "text": (0, 200, 0),
        "title": (220, 50, 50),
    }
    img = cv2.imread(str(image_path))
    for r in layout_regions:
        x1, y1, x2, y2 = r.bbox
        color = COLOR_MAP.get(r.region_type, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"[{r.region_id}] {r.region_type} {r.confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if output_path:
        pil.save(str(output_path))
    return pil


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Document Agent (Anthropic + tool use)
# ─────────────────────────────────────────────────────────────────────────────

# Tool schemas passed to the model
AGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "AnalyzeChart",
            "description": (
                "Analyse a chart or figure region. "
                "Pass the region_id of a figure/chart layout region. "
                "Returns chart_type, axes labels, data points, and trends."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {
                        "type": "integer",
                        "description": "The layout region ID to analyse.",
                    }
                },
                "required": ["region_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AnalyzeTable",
            "description": (
                "Analyse a table region and return its structured data. "
                "Pass the region_id of a table layout region. "
                "Returns headers, rows, values, and any notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {
                        "type": "integer",
                        "description": "The layout region ID to analyse.",
                    }
                },
                "required": ["region_id"],
            },
        },
    },
]


class DocumentAgent:
    """
    Agentic loop: Claude decides which tools to call; each tool call sends
    the cropped region image to Claude Vision and returns structured data.
    """

    def __init__(
        self,
        ordered_text: list[OrderedTextEntry],
        layout_regions: list[LayoutRegion],
        region_images: dict[int, dict],  # {region_id: {base64, type, bbox}}
        model: str = "gpt-4o",
    ):
        self._client = openai.OpenAI()
        self._model = model
        self._ordered_text = ordered_text
        self._layout_regions = layout_regions
        self._region_images = region_images

        self._system_prompt = self._build_system_prompt()
        self._analyses: dict[int, dict] = {}

    # ── prompt helpers ────────────────────────────────────────────────────────
    def _build_system_prompt(self) -> str:
        text_block = "\n".join(
            f"{e.position:3}. {e.text}" for e in self._ordered_text
        )
        layout_block = "\n".join(
            f"  - id={r.region_id}  type={r.region_type}  "
            f"conf={r.confidence:.2f}  bbox={r.bbox}"
            for r in self._layout_regions
        )
        return (
            "You are a Document Intelligence Agent.\n"
            "You analyse documents by combining OCR text with visual tool calls.\n\n"
            "## Document Text (reading order)\n"
            f"{text_block}\n\n"
            "## Detected Layout Regions\n"
            f"{layout_block}\n\n"
            "## Instructions\n"
            "1. For TEXT regions: rely on the OCR text above.\n"
            "2. For TABLE regions: call AnalyzeTable with the region_id.\n"
            "3. For CHART/FIGURE regions: call AnalyzeChart with the region_id.\n"
            "Answer questions accurately, using tools when visual data is needed."
        )

    # ── VLM tool executors ────────────────────────────────────────────────────
    def _vlm_analyze_chart(self, region_id: int) -> dict:
        """Send cropped chart image to OpenAI Vision; return structured dict."""
        img_data = self._region_images.get(region_id)
        if not img_data:
            return {"error": f"No image for region {region_id}"}

        log.info("  AnalyzeChart → region %d", region_id)
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_data['base64']}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyse this chart. "
                                "Respond ONLY with a JSON object (no markdown fences) "
                                "with keys: chart_type, x_axis, y_axis, data_points "
                                "(list of {label, value}), trends (list of strings)."
                            ),
                        },
                    ],
                }
            ],
        )
        raw = resp.choices[0].message.content.strip().lstrip("```json").rstrip("```").strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"raw_analysis": raw}
        result["type"] = "chart"
        result["region_id"] = region_id
        self._analyses[region_id] = result
        return result

    def _vlm_analyze_table(self, region_id: int) -> dict:
        """Send cropped table image to OpenAI Vision; return structured dict."""
        img_data = self._region_images.get(region_id)
        if not img_data:
            return {"error": f"No image for region {region_id}"}

        log.info("  AnalyzeTable → region %d", region_id)
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_data['base64']}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract this table. "
                                "Respond ONLY with a JSON object (no markdown fences) "
                                "with keys: headers (list), rows (list of lists), notes (string)."
                            ),
                        },
                    ],
                }
            ],
        )
        raw = resp.choices[0].message.content.strip().lstrip("```json").rstrip("```").strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"raw_analysis": raw}
        result["type"] = "table"
        result["region_id"] = region_id
        self._analyses[region_id] = result
        return result

    # ── agentic loop ──────────────────────────────────────────────────────────
    def run(self, user_query: str, max_turns: int = 10) -> tuple[str, dict[int, dict]]:
        """
        Run the agentic loop for a user query.
        Returns (final_answer, region_analyses).
        """
        log.info("Agent query: %s", user_query)
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_query},
        ]

        for turn in range(max_turns):
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=2048,
                tools=AGENT_TOOLS,
                messages=messages,
            )

            assistant_msg = resp.choices[0].message
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                log.info("Agent finished after %d turns.", turn + 1)
                return assistant_msg.content or "", self._analyses

            # Process tool calls
            for tool_call in assistant_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                log.info("Tool call: %s(%s)", func_name, func_args)

                if func_name == "AnalyzeChart":
                    result = self._vlm_analyze_chart(func_args["region_id"])
                elif func_name == "AnalyzeTable":
                    result = self._vlm_analyze_table(func_args["region_id"])
                else:
                    result = {"error": f"Unknown tool: {func_name}"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        return "Agent loop exhausted.", self._analyses


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator – ties all stages together
# ─────────────────────────────────────────────────────────────────────────────
class DocumentProcessor:
    """
    Full pipeline:
      image → OCR → ReadingOrder → LayoutDetection → Agent(VLM tools)
    """

    def __init__(self, ocr_lang: str = "en"):
        # Initialize ReadingOrderDetector (Torch) before OCRExtractor (Paddle)
        # to avoid WinError 127 DLL conflicts on Windows.
        self._order = ReadingOrderDetector()
        self._ocr = OCRExtractor(lang=ocr_lang)
        self._layout = LayoutDetector()

    def process(
        self,
        image_path: str | Path,
        query: str = "Summarise this document, analysing all tables and charts.",
        save_annotated: str | Path | None = None,
    ) -> DocumentResult:
        image_path = Path(image_path)

        # ── Stage 1: OCR ──────────────────────────────────────────────────────
        ocr_regions = self._ocr.extract(image_path)

        # ── Stage 2: Reading order ────────────────────────────────────────────
        ordered_text = self._order.order(ocr_regions)

        # ── Stage 3: Layout detection ─────────────────────────────────────────
        layout_regions = self._layout.detect(image_path)

        # ── Crop regions for the agent ────────────────────────────────────────
        pil_img = Image.open(image_path).convert("RGB")
        region_images: dict[int, dict] = {}
        for r in layout_regions:
            cropped = crop_region(pil_img, r.bbox)
            region_images[r.region_id] = {
                "base64": pil_to_base64(cropped),
                "type": r.region_type,
                "bbox": r.bbox,
            }

        # Optional: save annotated image
        if save_annotated:
            draw_regions(image_path, layout_regions, save_annotated)
            log.info("Annotated image saved to %s", save_annotated)

        # ── Stage 4: Agent ────────────────────────────────────────────────────
        agent = DocumentAgent(
            ordered_text=ordered_text,
            layout_regions=layout_regions,
            region_images=region_images,
        )
        answer, region_analyses = agent.run(query)

        result = DocumentResult(
            ordered_text=ordered_text,
            layout_regions=layout_regions,
            region_analyses=region_analyses,
        )

        log.info("\n%s\n\nAgent Answer:\n%s", result.as_markdown(), answer)
        return result
