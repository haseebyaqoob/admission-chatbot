

import logging
import sys
import time
from pathlib import Path

from config_loader import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ingestion.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("build")


def _banner(step: str, n: int, total: int) -> None:
    log.info("")
    log.info("=" * 60)
    log.info(f"  STEP {n}/{total} — {step}")
    log.info("=" * 60)


def run(force: bool = False) -> None:
    corpus_dir    = cfg["corpus_dir"]
    extracted_dir = cfg["extracted_dir"]
    index_dir     = cfg["index_dir"]
    embed_model   = cfg["embed_model"]
    reranker_model= cfg.get("reranker_model", "")

    t_total = time.time()

    # ── Step 1: Extract ───────────────────────────────────────────────────────
    _banner("EXTRACT  (PDF page routing + Docling + TXT)", 1, 4)

    if force:
        # Wipe previous extraction outputs so everything is re-run
        import shutil
        ext_path = Path(extracted_dir)
        if ext_path.exists():
            shutil.rmtree(ext_path)
            log.info(f"Cleared {extracted_dir} (--force)")

    from ingestion.extractor import run as extract_run
    manifest = extract_run(corpus_dir)

    n_errors = sum(1 for e in manifest if e.get("error"))
    if n_errors:
        log.warning(f"{n_errors} file(s) failed during extraction — check ingestion.log")

    # ── Step 2: Post-process ──────────────────────────────────────────────────
    _banner("POSTPROCESS  (Docling JSON → clean RAG-ready JSON)", 2, 4)

    from ingestion.postprocessor import run as postprocess_run
    postprocess_run()

    # Verify we have at least some _clean.json files
    clean_files = list(Path(extracted_dir).glob("*_clean.json"))
    clean_files = [f for f in clean_files if "manifest" not in f.name]
    if not clean_files:
        log.error("No *_clean.json files produced. Aborting.")
        sys.exit(1)
    log.info(f"Clean JSON files ready: {len(clean_files)}")

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    _banner("CHUNK  (structure-aware, atomic tables)", 3, 4)

    from ingestion.chunker import chunk_all_clean_files
    chunks = chunk_all_clean_files(extracted_dir)

    if not chunks:
        log.error("No chunks produced. Check extracted_dir contents.")
        sys.exit(1)

    text_chunks  = [c for c in chunks if c.chunk_type == "text"]
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    log.info(f"Chunks: {len(chunks)} total  "
             f"({len(text_chunks)} text, {len(table_chunks)} table)")

    # ── Step 4: Build index ───────────────────────────────────────────────────
    _banner("INDEX  (BGE embed → FAISS + BM25)", 4, 4)

    from index.hybrid_searcher import HybridSearcher
    searcher = HybridSearcher(
        embed_model_name    = embed_model,
        reranker_model_name = reranker_model,
    )
    t0 = time.time()
    searcher.build(chunks)
    searcher.save(index_dir)
    log.info(f"Index built in {time.time()-t0:.1f}s → {index_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info(f"BUILD COMPLETE in {time.time()-t_total:.1f}s")
    log.info(f"  Clean JSON files : {len(clean_files)}")
    log.info(f"  Total chunks     : {len(chunks)}")
    log.info(f"    Text chunks    : {len(text_chunks)}")
    log.info(f"    Table chunks   : {len(table_chunks)}")
    log.info(f"  Index            : {index_dir}")
    log.info("=" * 60)
    log.info("")
    log.info("Next: chainlit run app.py   or   python scripts/chat_cli.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the admissions RAG index")
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe extracted/ and rebuild from scratch"
    )
    args = parser.parse_args()
    run(force=args.force)
