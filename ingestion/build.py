import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ingestion import (
    prepare_corpus,
    csv_builder,
    shuttle_builder,
    supervisor_builder,
    scholarship_builder,
)
from index import corpus_index


def run():
    print("\n" + "=" * 60)
    print("  ADMISSION BOT — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Extract all text from corpus
    print("\nStep 1/6: Extract text from corpus...")
    prepare_corpus.run()

    # Step 2: Build structured CSV (programs)
    print("\n Step 2/6: Build programs.csv...")
    csv_builder.build()

    # Step 3: Build structured CSV (shuttle routes)
    print("\n Step 3/6: Build shuttle_routes.csv...")
    shuttle_builder.build()

    # Step 4: Build structured CSV (PhD supervisors)
    print("\n Step 4/6: Build supervisors.csv...")
    supervisor_builder.build()

    # Step 5: Build structured CSV (scholarships)
    print("\n Step 5/6: Build scholarships.csv...")
    scholarship_builder.build()

    # Step 6: Build FAISS index (force=True to rebuild with fresh chunks)
    print("\n Step 6/6: Build vector index...")
    corpus_index.build(force=True)

    # Invalidate the BM25 cache so the next app startup rebuilds it from
    # the freshly chunked corpus rather than a stale pickle.
    bm25_cache = _PROJECT_ROOT / "data" / "index" / "bm25_index.pkl"
    if bm25_cache.exists():
        bm25_cache.unlink()
        print(f"[build] BM25 cache invalidated: {bm25_cache}")

    print("\n" + "=" * 60)
    print("Ingestion complete! Run 'chainlit run app.py' to start.")
    print("=" * 60)


if __name__ == "__main__":
    run()
