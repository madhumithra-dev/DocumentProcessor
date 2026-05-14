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
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="VLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--vlm-model",
        default="gpt-4o",
        help="Model name for the VLM provider (default: gpt-4o for openai, llava for ollama)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Host URL for Ollama (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for Ollama requests (default: 300)",
    )
    parser.add_argument(
        "--skip-vcot",
        action="store_true",
        help="Skip the 'Observation' step in the agentic flow (saves time, but may reduce accuracy)",
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
        help="Save the Markdown result to this file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save the structured JSON result to this file",
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

    # Initialise VLM provider
    from document_processor import OpenAIProvider, OllamaProvider
    if args.provider == "openai":
        vlm_provider = OpenAIProvider(model=args.vlm_model)
    elif args.provider == "ollama":
        # Default to llava if model not specified for ollama
        m = args.vlm_model if args.vlm_model != "gpt-4o" else "llava"
        vlm_provider = OllamaProvider(model=m, host=args.ollama_host, timeout=args.ollama_timeout)
    else:
        vlm_provider = OpenAIProvider(model=args.vlm_model)

    processor = DocumentProcessor(
        ocr_lang=args.lang, 
        vlm_provider=vlm_provider,
        use_vcot=not args.skip_vcot
    )
    result = processor.process(
        input_path=args.document,
        save_annotated=args.annotated,
    )

    # Save Markdown
    md = result.as_markdown()
    if args.output:
        args.output.write_text(md, encoding="utf-8")
        log.info("Markdown result written to %s", args.output)
    
    # Save JSON
    if args.output_json:
        args.output_json.write_text(result.as_json(), encoding="utf-8")
        log.info("JSON result written to %s", args.output_json)

    # If no output files specified, print markdown to stdout
    if not args.output and not args.output_json:
        print("\n" + "=" * 72)
        print(md)
        print("=" * 72)


if __name__ == "__main__":
    main()
