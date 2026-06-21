"""
build.py
─────────
One-command orchestrator for the full ingestion pipeline.

Usage: python -m ingestion.build

Runs:
  1. prepare_corpus()  — copy .txt files, extract PDFs, OCR images
  2. csv_builder.build()  — construct programs.csv
  3. shuttle_builder.build() — construct shuttle_routes.csv
  4. corpus_index.build() — chunk all extracted text → build FAISS index
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ingestion import prepare_corpus, csv_builder, shuttle_builder
from index import corpus_index


def run():
    print("\n" + "=" * 60)
    print("  ADMISSION BOT — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Extract all text from corpus
    print("\n📄 Step 1/4: Extract text from corpus...")
    prepare_corpus.run()

    # Step 2: Build structured CSV (programs)
    print("\n📊 Step 2/4: Build programs.csv...")
    csv_builder.build()

    # Step 3: Build structured CSV (shuttle routes)
    print("\n🚌 Step 3/4: Build shuttle_routes.csv...")
    shuttle_builder.build()

    # Step 4: Build FAISS index (force=True because embed model changed)
    print("\n🔍 Step 4/4: Build vector index...")
    corpus_index.build(force=True)

    print("\n" + "=" * 60)
    print("✅ Ingestion complete! Run 'chainlit run app.py' to start.")
    print("=" * 60)


if __name__ == "__main__":
    run()