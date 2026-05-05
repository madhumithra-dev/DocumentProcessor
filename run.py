"""
CLI entry point for the Document Intelligence Processor.

Usage
-----
  python run.py path/to/document.png
  python run.py path/to/document.png --query "What does the revenue table show?"
  python run.py path/to/document.png --annotated annotated.png --output result.md
"""

import argparse
import logging
import torch  # Early import to prevent DLL conflicts
from pathlib import Path

from document_processor import DocumentProcessor
from dotenv import load_dotenv
load_dotenv()
log = logging.getLogger("run")


def main():
    parser = argparse.ArgumentParser(
        description="Document Intelligence Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="Path to input document image")
    parser.add_argument(
        "--query",
        default="Summarise this document. Analyse all tables and charts.",
        help="Natural-language query for the agent (default: full summary)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="OCR language code (default: en)",
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        default=None,
        help="Optional path to save layout-annotated image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the Markdown result",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.image.exists():
        parser.error(f"Image not found: {args.image}")

    processor = DocumentProcessor(ocr_lang=args.lang)
    result = processor.process(
        image_path=args.image,
        query=args.query,
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
