# Document Intelligence Processor

A four-stage pipeline that turns a document image into structured, 
queryable data by chaining OCR, reading-order detection, layout 
detection, and an AI agent with vision tools.

---

## Architecture

```
Input Image
    │
    ├──► Stage 1 – OCR (PaddleOCR)
    │         text strings, bounding boxes, confidence scores
    │              │
    │              ▼
    │    Stage 2 – Reading Order (LayoutReader / LayoutLMv3)
    │         orders OCR regions left-to-right, top-to-bottom
    │
    └──► Stage 3 – Layout Detection (PaddleOCR LayoutDetection)
              tables, figures, text blocks, titles
                   │
                   ▼
         Stage 4 – Document Agent (Claude + tool use)
         ┌─────────────────────────────────────────────┐
         │  System Prompt                               │
         │  • ordered OCR text                          │
         │  • region IDs & types                        │
         │  • tool descriptions                         │
         │                                              │
         │  AnalyzeChart tool                           │
         │  → crops region → Claude Vision              │
         │  ← chart_type, axes, data_points, trends     │
         │                                              │
         │  AnalyzeTable tool                           │
         │  → crops region → Claude Vision              │
         │  ← headers, rows, values, notes              │
         └─────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run

```bash
# Default: full summary + analyse all tables/charts
python run.py report.png

# Custom query
python run.py report.png --query "What are the Q3 revenue figures?"

# Save annotated layout image and Markdown result
python run.py report.png \
    --annotated annotated.png \
    --output result.md \
    --verbose
```

---

## Python API

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(ocr_lang="en")

result = processor.process(
    image_path="report.png",
    query="Summarise the key metrics from all tables.",
    save_annotated="annotated.png",   # optional
)

# Markdown report
print(result.as_markdown())

# Programmatic access
for entry in result.ordered_text:
    print(entry.position, entry.text)

for region in result.layout_regions:
    print(region.region_id, region.region_type, region.bbox)

for rid, analysis in result.region_analyses.items():
    print(rid, analysis)
```

---

## Module Reference

| Class / File            | Responsibility                                                |
|-------------------------|---------------------------------------------------------------|
| `OCRExtractor`          | Wraps PaddleOCR; returns `List[OCRRegion]`                    |
| `ReadingOrderDetector`  | Wraps LayoutReader; returns `List[OrderedTextEntry]`          |
| `LayoutDetector`        | Wraps PaddleOCR LayoutDetection; returns `List[LayoutRegion]` |
| `DocumentAgent`         | Agentic loop; calls `AnalyzeChart` / `AnalyzeTable` via VLM  |
| `DocumentProcessor`     | Orchestrator: chains all four stages                          |
| `run.py`                | CLI entry point                                               |

---

## Output Shape

```python
@dataclass
class DocumentResult:
    ordered_text:     List[OrderedTextEntry]   # reading-order OCR
    layout_regions:   List[LayoutRegion]       # detected regions
    region_analyses:  Dict[int, dict]          # VLM tool results
```

### `AnalyzeChart` returns
```json
{
  "type": "chart",
  "region_id": 2,
  "chart_type": "bar",
  "x_axis": "Quarter",
  "y_axis": "Revenue ($M)",
  "data_points": [{"label": "Q1", "value": 12.4}, ...],
  "trends": ["Revenue grew 18% YoY", "Q3 peak"]
}
```

### `AnalyzeTable` returns
```json
{
  "type": "table",
  "region_id": 5,
  "headers": ["Product", "Units", "Revenue"],
  "rows": [["Widget A", "1200", "$48K"], ...],
  "notes": "Figures are unaudited estimates."
}
```
