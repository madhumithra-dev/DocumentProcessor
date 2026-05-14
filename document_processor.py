"""
Document Intelligence Processor
================================
Pipeline:
  PDF  → per-page classify (text vs image-based)
         text-based  → direct PDF text extraction
         image-based → layout detection → reading-order regions
                         text/title  → PaddleOCR crop
                         table       → VLM (structured JSON)
                         figure      → VLM (structured JSON)
  Image → treated as single image-based page (same as above)
"""

import base64
import json
import logging
import torch  # Import first to lock DLL versions on Windows (prevents WinError 127)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, Type, TypeVar

import cv2
import numpy as np
import openai
import requests
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
class LayoutRegion:
    region_id: int
    region_type: str    # "text" | "title" | "table" | "figure"
    bbox: list[int]     # [x1, y1, x2, y2]
    confidence: float


@dataclass
class PageElement:
    """One content element extracted from a page, in reading order."""
    position: int       # 0-indexed order within the page
    element_type: str   # "text" | "title" | "table" | "figure"
    content: Any        # str for text/title; dict for table/figure VLM result
    bbox: list[int]     # [x1, y1, x2, y2] in page-image pixels ([] for text-pages)
    confidence: float


@dataclass
class PageResult:
    """All extracted elements for one page, in reading order."""
    page_number: int    # 1-indexed
    page_type: str      # "text" | "image"
    elements: list[PageElement] = field(default_factory=list)

    def as_markdown(self) -> str:
        lines = [f"## Page {self.page_number}  _(type: {self.page_type}-based)_\n"]
        for el in self.elements:
            if el.element_type == "title":
                lines.append(f"### {el.content}")
            elif el.element_type == "text":
                lines.append(str(el.content))
            else:
                lines.append(f"\n**[{el.element_type.upper()} – region {el.position}]**")
                if isinstance(el.content, dict):
                    lines.append(
                        "```json\n"
                        + json.dumps(el.content, indent=2, ensure_ascii=False)
                        + "\n```"
                    )
                else:
                    lines.append(str(el.content))
        return "\n".join(lines)


@dataclass
class DocumentResult:
    """Full document result across all pages."""
    pages: list[PageResult] = field(default_factory=list)
    source: str = ""

    def as_markdown(self) -> str:
        lines = [f"# Document Analysis: {self.source}\n"]
        for page in self.pages:
            lines.append(page.as_markdown())
            lines.append("\n---\n")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return asdict(self)

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Structured Data Schemas (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class TableSchema(BaseModel):
    headers: list[str] = Field(description="List of column headers")
    rows: list[list[str | int | float | None]] = Field(description="List of rows, each containing a list of cell values (strings or numbers)")
    notes: str = Field(default="", description="Any additional notes or captions found near the table")

class FigureSchema(BaseModel):
    figure_type: str = Field(description="Type of figure (e.g., bar chart, flowchart, line graph)")
    description: str = Field(description="Comprehensive textual description of the visual content")
    data_points: list[dict[str, str | int | float | None]] = Field(default_factory=list, description="Extracted data points as label-value pairs")
    trends: list[str] = Field(default_factory=list, description="Key trends or insights observed in the figure")

T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def crop_region(pil_img: Image.Image, bbox: list[int], padding: int = 8) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(pil_img.width, x2 + padding)
    y2 = min(pil_img.height, y2 + padding)
    return pil_img.crop((x1, y1, x2, y2))


def sort_reading_order(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    """Sort layout regions top-to-bottom, left-to-right (row-based grouping)."""
    if not regions:
        return regions
    by_y = sorted(regions, key=lambda r: r.bbox[1])
    rows: list[list[LayoutRegion]] = []
    current: list[LayoutRegion] = [by_y[0]]
    for r in by_y[1:]:
        # same row if top edge within 40 px of the row's first element
        if abs(r.bbox[1] - current[0].bbox[1]) < 40:
            current.append(r)
        else:
            rows.append(sorted(current, key=lambda rr: rr.bbox[0]))
            current = [r]
    rows.append(sorted(current, key=lambda rr: rr.bbox[0]))
    result: list[LayoutRegion] = []
    for row in rows:
        result.extend(row)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage A – Layout detection (PaddleOCR PPStructure)
# ─────────────────────────────────────────────────────────────────────────────

class LayoutDetector:
    def __init__(self):
        log.info("Initialising PPStructure for layout detection…")
        from paddleocr import PPStructure
        self._engine = PPStructure(table=False, ocr=False, show_log=False)

    def detect(self, pil_image: Image.Image) -> list[LayoutRegion]:
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        raw = self._engine(img_cv)
        regions: list[LayoutRegion] = []
        for i, region in enumerate(raw):
            regions.append(LayoutRegion(
                region_id=i,
                region_type=region["type"].lower(),
                bbox=[int(c) for c in region["bbox"]],
                confidence=float(region.get("score", 1.0)),
            ))
        return regions


# ─────────────────────────────────────────────────────────────────────────────
# Stage B – OCR on text/title region crops (PaddleOCR)
# ─────────────────────────────────────────────────────────────────────────────

class OCRExtractor:
    def __init__(self, lang: str = "en"):
        log.info("Initialising PaddleOCR (lang=%s)…", lang)
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(lang=lang)

    def extract_from_crop(self, pil_crop: Image.Image) -> str:
        """Run OCR on a PIL image crop and return joined text string."""
        img_array = np.array(pil_crop)
        result = self._ocr.ocr(img_array, cls=True)
        if not result or result[0] is None:
            return ""
        return " ".join(line[1][0] for line in result[0])


# ─────────────────────────────────────────────────────────────────────────────
# Stage C – Agentic VLM analysis (VCoT + Self-Correction)
# ─────────────────────────────────────────────────────────────────────────────

class VLMProvider(ABC):
    @abstractmethod
    def call(self, pil_image: Image.Image, prompt: str) -> str:
        pass

class OpenAIProvider(VLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        self._model = model
        self._llm = ChatOpenAI(model=model, max_tokens=2048)

    def call(self, pil_image: Image.Image, prompt: str) -> str:
        b64 = pil_to_base64(pil_image)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ]
        )
        resp = self._llm.invoke([message])
        return str(resp.content).strip()

class OllamaProvider(VLMProvider):
    def __init__(self, model: str = "llava", host: str = "http://localhost:11434", timeout: int = 300):
        self._model = model
        self._host = host
        self._timeout = timeout

    def call(self, pil_image: Image.Image, prompt: str) -> str:
        b64 = pil_to_base64(pil_image)
        url = f"{self._host}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [b64],
            "stream": False
        }
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            log.error("Ollama call failed (timeout=%ds): %s", self._timeout, e)
            return f"Error: {str(e)}"

class DocumentAgent:
    """
    Agentic wrapper for VLM extraction.
    Uses Visual Chain-of-Thought (Observe -> Extract) and Self-Correction.
    """
    def __init__(self, provider: VLMProvider, use_vcot: bool = True):
        self._provider = provider
        self._use_vcot = use_vcot

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def analyze_region(self, pil_image: Image.Image, schema: Type[T], region_type: str) -> dict:
        log.info("    [LangChain Agent] Analyzing %s...", region_type)
        
        # Use LangChain's structured output capability
        if hasattr(self._provider, "_llm"):
            llm = self._provider._llm.with_structured_output(schema)
            
            b64 = pil_to_base64(pil_image)
            prompt = f"Extract the data from this {region_type} crop according to the schema."
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            )
            
            try:
                # This automatically handles JSON parsing and validation against the Pydantic schema
                result_obj = llm.invoke([message])
                result = result_obj.model_dump()
                result["region_type"] = region_type
                return result
            except Exception as e:
                log.warning("    [LangChain Agent] Structured output failed, falling back to manual: %s", e)

        # Fallback to manual flow if not using LangChain-compatible provider or if it failed
        observation = ""
        if self._use_vcot:
            obs_prompt = (
                f"Observe this {region_type} crop carefully. Describe its layout, structure, "
                "and all visible data points in detail. Do not output JSON yet."
            )
            observation = self._provider.call(pil_image, obs_prompt)

        extract_prompt = (
            f"Extract the data from this {region_type} into a JSON object matching the schema.\n"
            f"Observation: {observation}\n"
            f"Schema: {json.dumps(schema.model_json_schema())}"
        )
        
        raw_response = self._provider.call(pil_image, extract_prompt)
        json_str = self._extract_json(raw_response)
        
        try:
            validated = schema.model_validate_json(json_str)
            result = validated.model_dump()
            result["region_type"] = region_type
            return result
        except Exception as e:
            return {"raw": raw_response, "region_type": region_type, "error": str(e)}

    def analyze_table(self, pil_image: Image.Image) -> dict:
        return self.analyze_region(pil_image, TableSchema, "table")

    def analyze_figure(self, pil_image: Image.Image, region_type: str = "figure") -> dict:
        return self.analyze_region(pil_image, FigureSchema, region_type)


# ─────────────────────────────────────────────────────────────────────────────
# Annotated image helper
# ─────────────────────────────────────────────────────────────────────────────

_REGION_COLORS = {
    "table":  (255, 165,   0),
    "figure": (  0, 128, 255),
    "text":   (  0, 200,   0),
    "title":  (220,  50,  50),
}


def save_annotated_image(
    pil_img: Image.Image,
    regions: list[LayoutRegion],
    output_path: str | Path,
) -> None:
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r.bbox
        color = _REGION_COLORS.get(r.region_type, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"[{i}] {r.region_type} {r.confidence:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(str(output_path))
    log.info("Annotated image saved → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Routing constants
# ─────────────────────────────────────────────────────────────────────────────

# Region types that require visual (VLM) analysis.
# Everything NOT in this set goes to PaddleOCR.
_VLM_TYPES: frozenset[str] = frozenset({"table", "figure", "chart"})

# ─────────────────────────────────────────────────────────────────────────────
# Main processor
# ─────────────────────────────────────────────────────────────────────────────

# A page is considered "text-based" if it has more than this many characters
# of extractable text in the PDF layer.
_TEXT_THRESHOLD = 50


class DocumentProcessor:
    """
    Unified processor for both PDFs and images.

    PDF pages
    ---------
    Each page is inspected independently:
    • If extractable text > _TEXT_THRESHOLD chars  → text-based page
      Text is taken directly from the PDF layer (no OCR needed).
    • Otherwise                                    → image-based page
      The page is rendered to an image and processed with the
      layout-first pipeline below.

    Image files / image-based PDF pages / Hybrid PDF pages
    -----------------------------------------------------
    1. LayoutDetector (PPStructure) finds all regions.
    2. Regions are sorted into reading order (top→bottom, left→right).
    Unified processor for both PDFs and images using an Agentic VLM flow.
    """

    def __init__(self, ocr_lang: str = "en", vlm_provider: VLMProvider | None = None, use_vcot: bool = True):
        self._layout = LayoutDetector()
        self._ocr = OCRExtractor(lang=ocr_lang)
        
        # Default to OpenAI if no provider given
        provider = vlm_provider or OpenAIProvider()
        self._agent = DocumentAgent(provider, use_vcot=use_vcot)

    # ── public API ────────────────────────────────────────────────────────────

    def process(
        self,
        input_path: str | Path,
        save_annotated: str | Path | None = None,
    ) -> DocumentResult:
        input_path = Path(input_path)
        result = DocumentResult(source=input_path.name)

        if input_path.suffix.lower() == ".pdf":
            result.pages = self._process_pdf(input_path, save_annotated)
        else:
            pil_img = Image.open(input_path).convert("RGB")
            page = self._process_image_page(pil_img, page_number=1,
                                            save_annotated=save_annotated)
            result.pages = [page]

        return result

    # ── PDF handling ──────────────────────────────────────────────────────────

    def _process_pdf(
        self,
        pdf_path: Path,
        save_annotated: str | Path | None,
    ) -> list[PageResult]:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        pages: list[PageResult] = []

        for i in range(len(doc)):
            page_num = i + 1
            fitz_page = doc[i]

            log.info("Page %d: hybrid processing (layout detection + PDF text layer)", page_num)
            
            # Render page to image for layout detection (150 DPI)
            zoom = 150 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Only save annotated image for first page or if explicitly requested
            ann = save_annotated if (i == 0) else None
            pages.append(self._process_image_page(
                pil_img, 
                page_num, 
                ann, 
                fitz_page=fitz_page,
                scale_factor=zoom
            ))

        doc.close()
        return pages

    # ── Page processors ───────────────────────────────────────────────────────

    def _process_text_page(self, text: str, page_number: int) -> PageResult:
        """Wrap PDF-extracted text as a single text element."""
        element = PageElement(
            position=0,
            element_type="text",
            content=text,
            bbox=[],
            confidence=1.0,
        )
        return PageResult(page_number=page_number, page_type="text", elements=[element])

    def _process_image_page(
        self,
        pil_img: Image.Image,
        page_number: int,
        save_annotated: str | Path | None = None,
        fitz_page: Any = None,
        scale_factor: float = 1.0,
    ) -> PageResult:
        """
        Layout-first pipeline for one image page:
          detect regions → sort reading order → OCR or VLM or PDF-clip per region.
        """
        log.info("Page %d: running layout detection…", page_number)
        regions = self._layout.detect(pil_img)
        regions = sort_reading_order(regions)
        log.info("Page %d: %d regions in reading order", page_number, len(regions))

        if save_annotated:
            save_annotated_image(pil_img, regions, save_annotated)

        elements: list[PageElement] = []
        for pos, region in enumerate(regions):
            crop = crop_region(pil_img, region.bbox)
            rtype = region.region_type

            log.info(
                "  Region %d/%d  type=%-7s  bbox=%s",
                pos + 1, len(regions), rtype, region.bbox,
            )

            if rtype == "table" and rtype in _VLM_TYPES:
                content: Any = self._agent.analyze_table(crop)
            elif rtype in _VLM_TYPES:  # figure, chart
                content = self._agent.analyze_figure(crop, region_type=rtype)
            else:  # text, title, caption, reference, equation, etc.
                # Try PDF text extraction first if fitz_page is available
                extracted = ""
                if fitz_page:
                    # Convert bbox from pixels (at scale_factor) to PDF points
                    x1, y1, x2, y2 = [c / scale_factor for c in region.bbox]
                    extracted = fitz_page.get_text("text", clip=(x1, y1, x2, y2)).strip()
                
                # If PDF extraction yielded nothing, fallback to OCR
                if not extracted:
                    content = self._ocr.extract_from_crop(crop)
                else:
                    content = extracted

            elements.append(PageElement(
                position=pos,
                element_type=rtype,
                content=content,
                bbox=region.bbox,
                confidence=region.confidence,
            ))

        return PageResult(page_number=page_number, page_type="hybrid" if fitz_page else "image", elements=elements)
