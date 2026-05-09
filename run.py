"""
CLI entry point for the Document Intelligence Processor.

Usage
-----
  python run.py path/to/document.pdf
  python run.py path/to/document.png
  python run.py path/to/document.pdf --output result.md
  python run.py path/to/document.png --annotated annotated.png --output result.md
"""

import argparse
import logging
import torch  # Early import to prevent DLL conflicts on Windows
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from document_processor import DocumentProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run")

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(
        description="Document Intelligence Processor – PDF and image support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "document",
        type=Path,
        help="Path to input document (PDF or image: png/jpg/tiff/bmp/webp)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="OCR language code passed to PaddleOCR (default: en)",
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        default=None,
        help="Save layout-annotated image to this path (image inputs / first PDF page)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the Markdown result to this file (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.document.exists():
        parser.error(f"File not found: {args.document}")

    ext = args.document.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        parser.error(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    processor = DocumentProcessor(ocr_lang=args.lang)
    result = processor.process(
        input_path=args.document,
        save_annotated=args.annotated,
    )

    md = result.as_markdown()

    if args.output:
        args.output.write_text(md, encoding="utf-8")
        log.info("Result written to %s", args.output)
    else:
        print("\n" + "=" * 72)
        print(md)
        print("=" * 72)


if __name__ == "__main__":
    main()
