"""
docling_parser.py
──────────────────
PDF and image text extraction using Docling (primary parser).

Docling replaces the legacy PyMuPDF + pytesseract pipeline with
deep-learning-based document understanding, handling:
  - Text-based PDFs
  - Scanned/image-only PDFs (built-in OCR via EasyOCR)
  - Tables and complex layouts
  - Standalone image files (JPG/PNG)

Output is plain text, matching the format the downstream chunker
and embedding pipeline expect.
"""

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def extract_text_docling(pdf_path: Path) -> str | None:
    """
    Extract text from a PDF using Docling.

    Returns plain text, or None if extraction fails
    (caller falls back to PyMuPDF + Tesseract).
    """
    try:
        converter = _get_converter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_text()
    except Exception as e:
        print(f"  [docling_parser] Docling PDF extraction failed: {e}")
        return None


def extract_from_image_docling(image_path: Path) -> str | None:
    """
    Extract text from an image file (JPG/PNG) using Docling.

    Returns plain OCR text, or None if extraction fails
    (caller falls back to Tesseract).
    """
    try:
        converter = _get_converter()
        result = converter.convert(str(image_path))
        return result.document.export_to_text()
    except Exception as e:
        print(f"  [docling_parser] Docling image extraction failed: {e}")
        return None


def convert_to_docling_document(path: Path) -> DoclingDocument | None:
    """
    Convert a PDF or image file to a DoclingDocument object.

    Returns the raw DoclingDocument (for downstream structure-aware chunking),
    or None if conversion fails.
    """
    try:
        converter = _get_converter()
        result = converter.convert(str(path))
        return result.document
    except Exception as e:
        print(f"  [docling_parser] Docling document conversion failed: {e}")
        return None
