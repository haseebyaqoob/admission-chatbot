"""
corpus_index.py
────────────────
Builds the FAISS index + BM25 index by orchestrating chunker, vector_store,
and hybrid_searcher over extracted text files.

Two chunking paths per .txt file:
  1. .docling.json exists → structure-aware Docling HybridChunker
  2. No .docling.json      → plain-text fallback chunker (chunk_file)

After chunking, every chunk gets contextual header enrichment and token
limit enforcement via the chunker pipeline.
"""

from pathlib import Path

from config_loader import cfg
from index.chunker import (
    Chunk,
    chunk_file,
    chunk_docling_document,
)
from index.vector_store import VectorStore
from index.hybrid_searcher import HybridSearcher
from docling_core.types.doc.document import DoclingDocument

_EXTRACTED_DIR = Path(cfg["extracted_dir"])
_INDEX_DIR     = Path(cfg["extracted_dir"]).parent / "index"
_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build(force: bool = False):
    """
    Chunk all extracted .txt files and build/update FAISS + BM25 index.

    For each .txt file in extracted_dir:
      1. If a matching .docling.json exists → structure-aware Docling chunking
      2. Otherwise → plain-text fallback chunking via chunk_file()

    The chunker automatically applies:
      - Token enforcement (splitting oversized chunks at sentence boundaries)
      - Contextual header enrichment (prepending [Source | Section | Topic])
      - Single-chunk fallback (re-chunking if Docling produced only 1 chunk)

    If force=False and index already exists, skip (idempotent).
    """
    index_path = _INDEX_DIR / "corpus_index"

    if not force and index_path.with_suffix(".faiss").exists():
        print(f"  [corpus_index] Index already exists at {index_path}.faiss")
        print(f"  [corpus_index] Pass force=True or delete the index to rebuild.")
        return

    print(f"  [corpus_index] Chunking files from {_EXTRACTED_DIR} …")

    all_chunks: list[Chunk] = []
    txt_files = sorted(_EXTRACTED_DIR.glob("*.txt"))

    docling_count = 0
    plaintext_count = 0

    for txt_path in txt_files:
        docling_json_path = txt_path.with_suffix(".docling.json")

        if docling_json_path.exists():
            doc = DoclingDocument.load_from_json(str(docling_json_path))
            file_chunks = chunk_docling_document(doc, source=txt_path.stem)
            docling_count += 1
        else:
            file_chunks = chunk_file(txt_path)
            if file_chunks:
                print(f"  [corpus_index] Plain-text fallback: {txt_path.name} → {len(file_chunks)} chunks")
            plaintext_count += 1

        all_chunks.extend(file_chunks)

    print(f"  [corpus_index] Files: {docling_count} Docling + {plaintext_count} plain-text = "
          f"{docling_count + plaintext_count} total")

    if not all_chunks:
        print("  [corpus_index] No chunks created — nothing to index.")
        return

    print(f"  [corpus_index] Total: {len(all_chunks)} chunks across all files")

    vs = VectorStore()
    vs.build(all_chunks)
    vs.save(index_path)

    print(f"  [corpus_index] Index saved to {index_path}.faiss")
    print(f"  [corpus_index] {len(all_chunks)} chunks indexed")


def load() -> VectorStore:
    """Load the pre-built FAISS index from disk."""
    index_path = _INDEX_DIR / "corpus_index"
    vs = VectorStore()
    vs.load(index_path)
    return vs


def load_hybrid() -> HybridSearcher:
    """Load the hybrid (FAISS + BM25) searcher from disk."""
    index_path = _INDEX_DIR / "corpus_index"
    hs = HybridSearcher()
    hs.load(index_path)
    return hs
