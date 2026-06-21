"""
pdf_extractor.py
─────────────────
Extract text from PDFs using Docling (primary), with
PyMuPDF + pytesseract OCR fallback for edge cases.
"""

from pathlib import Path

from ingestion.docling_parser import (
    extract_text_docling,
    extract_from_image_docling,
)

import fitz  # PyMuPDF (fallback)
from PIL import Image
import pytesseract


def extract_text(pdf_path: Path, ocr_fallback: bool = True) -> str:
    """
    Extract text from a PDF.

    Priority:
      1. Docling (deep-learning document parsing, handles both
         text-based and scanned PDFs with built-in OCR).
      2. PyMuPDF direct extraction + pytesseract OCR fallback
         per-page (if Docling fails).

    Returns the full document text.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # --- Primary: Docling ---------------------------------------------------
    docling_text = extract_text_docling(pdf_path)
    if docling_text is not None:
        print(f"  [pdf_extractor] Used Docling for {pdf_path.name}")
        return docling_text

    # --- Fallback: PyMuPDF + per-page Tesseract OCR -------------------------
    print(f"  [pdf_extractor] Docling failed — falling back to PyMuPDF for {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if ocr_fallback and len(text) < 50:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                text = ocr_text
                print(f"  [pdf_extractor] Page {page_num}: used OCR ({len(ocr_text)} chars)")

        pages_text.append(text)

    doc.close()
    return "\n\n".join(pages_text)


def extract_from_image(image_path: Path) -> str:
    """
    Extract text from an image file (JPG/PNG).

    Priority:
      1. Docling (built-in OCR).
      2. pytesseract (fallback).

    Used for standalone advertisement images in the corpus.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- Primary: Docling ---------------------------------------------------
    docling_text = extract_from_image_docling(image_path)
    if docling_text is not None:
        print(f"  [pdf_extractor] Used Docling for image {image_path.name}")
        return docling_text

    # --- Fallback: plain Tesseract ------------------------------------------
    print(f"  [pdf_extractor] Docling failed — falling back to Tesseract for image {image_path.name}")
    img = Image.open(str(image_path))
    text = pytesseract.image_to_string(img).strip()
    return text
