"""
vector_store.py
────────────────
FAISS index wrapper: embedding + search over chunked corpus text.

Uses sentence-transformers for embeddings and FAISS IndexFlatIP
(inner product = cosine similarity with L2-normalized embeddings).
"""

from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config_loader import cfg
from index.chunker import Chunk

_EMBED_MODEL     = cfg["vector_store"]["embedding_model"]
_EMBED_QUERY_PFX = cfg["vector_store"].get("embed_query_prefix", "")
_RAG_TOP_K       = int(cfg.get("retrieval", {}).get("rag_top_k", 4))


class VectorStore:
    """
    Wraps a sentence-transformer encoder and a FAISS flat IP index.

    Usage:
        vs = VectorStore()
        vs.build(chunks)
        results = vs.search("some query", top_k=4)
    """

    def __init__(self, model_name: str = _EMBED_MODEL):
        print(f"[vector_store] Loading embedding model '{model_name}' …")
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[Chunk] = []
        self.index_path: Optional[Path] = None

    def build(self, chunks: list[Chunk]):
        """Build the FAISS index from a list of Chunks."""
        if not chunks:
            print("[vector_store] No chunks to index.")
            return

        print(f"[vector_store] Embedding {len(chunks)} chunks …")
        texts      = [c.content for c in chunks]
        embeddings = self.model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        print(f"[vector_store] FAISS index built — {len(chunks)} chunks, dim={dim}")

    def search(self, query: str, top_k: int = _RAG_TOP_K) -> list[dict]:
        """
        Return top_k most relevant chunks for the query.

        Returns list of dicts:
          {chunk_id, source, topic, score, content}
        """
        if self.index is None or not self.chunks:
            return []

        prefixed_query = _EMBED_QUERY_PFX + query if _EMBED_QUERY_PFX else query
        query_vec = np.array(
            self.model.encode([prefixed_query], normalize_embeddings=True,
                               show_progress_bar=False), dtype="float32"
        )

        search_k = min(max(top_k * 6, 10), len(self.chunks))
        scores, indices = self.index.search(query_vec, search_k)

        seen_ids: set[str] = set()
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            results.append({
                "chunk_id": chunk.chunk_id,
                "source":   chunk.source,
                "topic":    chunk.topic,
                "score":    float(score),
                "content":  chunk.content,
            })
            if len(results) >= top_k:
                break

        return results

    def save(self, path: Path):
        """Persist the FAISS index and chunk metadata to disk."""
        if self.index is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        # Save chunk metadata (simple text-based format for portability)
        meta_path = path.with_suffix(".meta")
        with open(meta_path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(f"{c.chunk_id}\t{c.source}\t{c.topic}\t{c.content}\n")
        print(f"[vector_store] Saved index to {path}.faiss + .meta")

    def load(self, path: Path):
        """Load a previously saved FAISS index and chunk metadata."""
        self.index = faiss.read_index(str(path.with_suffix(".faiss")))
        meta_path = path.with_suffix(".meta")
        self.chunks = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 3)
                if len(parts) == 4:
                    self.chunks.append(Chunk(parts[0], parts[1], parts[2], parts[3]))
        print(f"[vector_store] Loaded index — {len(self.chunks)} chunks, dim={self.index.d}")
