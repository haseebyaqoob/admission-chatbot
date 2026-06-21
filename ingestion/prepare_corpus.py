"""
prepare_corpus.py
──────────────────
Copies .txt files from the corpus directory into the extracted directory,
then runs pdf_extractor on all PDFs and images (recursively).
"""

import shutil
from pathlib import Path

from config_loader import cfg
from ingestion.pdf_extractor import extract_text, extract_from_image
from ingestion.docling_parser import convert_to_docling_document

_CORPUS_DIR     = Path(cfg["corpus_dir"])
_EXTRACTED_DIR  = Path(cfg["extracted_dir"])
_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

# Image file extensions to OCR
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}


def copy_text_files():
    """Copy all .txt files from corpus to extracted dir (recursively)."""
    for txt_path in sorted(_CORPUS_DIR.rglob("*.txt")):
        dest = _EXTRACTED_DIR / txt_path.name
        if dest.exists() and dest.stat().st_mtime >= txt_path.stat().st_mtime:
            print(f"  [prepare] Skipping (up-to-date): {txt_path.name}")
            continue
        shutil.copy2(txt_path, dest)
        print(f"  [prepare] Copied: {txt_path.name}")


def extract_pdfs():
    """Extract text from all PDFs in the corpus directory (recursively)."""
    pdf_files = sorted(_CORPUS_DIR.rglob("*.pdf"))
    print(f"  [prepare] Found {len(pdf_files)} PDFs to process")

    for pdf_path in pdf_files:
        out_name = pdf_path.stem + "_extracted.txt"
        out_path = _EXTRACTED_DIR / out_name

        if out_path.exists() and out_path.stat().st_mtime >= pdf_path.stat().st_mtime:
            print(f"  [prepare] Skipping (up-to-date): {pdf_path.name}")
            continue

        print(f"  [prepare] Extracting: {pdf_path.name} ...")
        text = extract_text(pdf_path, ocr_fallback=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"  [prepare]   → {len(text)} chars written to {out_name}")

        # Also serialize the Docling document for structure-aware chunking
        docling_doc = convert_to_docling_document(pdf_path)
        if docling_doc is not None:
            docling_json_path = out_path.with_suffix(".docling.json")
            docling_doc.save_as_json(str(docling_json_path))
            print(f"  [prepare]   → .docling.json saved")


def extract_images():
    """OCR all standalone image files in the corpus directory (recursively)."""
    for img_path in sorted(_CORPUS_DIR.rglob("*")):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue

        out_name = img_path.stem + "_ocr.txt"
        out_path = _EXTRACTED_DIR / out_name

        if out_path.exists() and out_path.stat().st_mtime >= img_path.stat().st_mtime:
            print(f"  [prepare] Skipping (up-to-date): {img_path.name}")
            continue

        print(f"  [prepare] OCR: {img_path.name} ...")
        text = extract_from_image(img_path)
        if text:
            out_path.write_text(text, encoding="utf-8")
            print(f"  [prepare]   → {len(text)} chars written to {out_name}")

            # Also serialize the Docling document for structure-aware chunking
            docling_doc = convert_to_docling_document(img_path)
            if docling_doc is not None:
                docling_json_path = out_path.with_suffix(".docling.json")
                docling_doc.save_as_json(str(docling_json_path))
                print(f"  [prepare]   → .docling.json saved")
        else:
            print(f"  [prepare]   → (empty/no text found)")


def run():
    """Run the full extraction pipeline."""
    print("=" * 60)
    print("Step 1: Copying text files...")
    copy_text_files()

    print("\nStep 2: Extracting PDFs...")
    extract_pdfs()

    print("\nStep 3: OCR images...")
    extract_images()

    print("\n✅ Corpus preparation complete.")
    print(f"   Extracted files → {_EXTRACTED_DIR}")
