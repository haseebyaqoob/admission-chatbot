"""
index/hybrid_searcher.py — Hybrid BM25 + FAISS retrieval with cross-encoder reranking.

Architecture:
  Phase 1:  BM25 keyword search  (top initial_k)
            + FAISS vector search (top initial_k)
            → RRF fusion → ranked candidate list
  Phase 2:  Optional table-boost for numerical queries
  Phase 3:  BGE cross-encoder reranker → final top_n
  Output:   List of result dicts with content, metadata, score

Key design choices:
  - NO source-file routing. All chunks are searched uniformly.
  - Table chunks get a score boost (not hard filter) when query needs numbers.
  - BGE reranker v2-m3 is multilingual — handles Urdu-influenced queries.
  - Graceful degradation: reranker failure → RRF scores used directly.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from ingestion.chunker import Chunk
from config_loader import cfg


class HybridSearcher:
    """
    Hybrid retriever. Combines BM25 and dense FAISS search, fuses via RRF,
    then optionally reranks with a cross-encoder.
    """

    RRF_K = cfg.get("rrf_k", 60)

    def __init__(
        self,
        embed_model_name:    str = None,
        reranker_model_name: str = None,
    ):
        embed_model_name    = embed_model_name    or cfg["embed_model"]
        reranker_model_name = reranker_model_name or cfg.get("reranker_model", "")

        print(f"Loading embedding model: {embed_model_name} ...")
        self.embedder = SentenceTransformer(embed_model_name)
        self._embed_model_name = embed_model_name

        self.reranker: Optional[CrossEncoder] = None
        if reranker_model_name:
            print(f"Loading reranker: {reranker_model_name} ...")
            try:
                self.reranker = CrossEncoder(reranker_model_name)
            except Exception as e:
                print(f"WARNING: Could not load reranker ({e}). Scores from RRF will be used.")

        self.chunks: List[Chunk] = []
        self.faiss_index = None
        self.bm25: Optional[BM25Okapi] = None

    # ─── Index building ───────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and build FAISS + BM25 indices."""
        self.chunks = chunks
        print(f"Building indices for {len(chunks)} chunks ...")

        # BM25
        print("  Building BM25 ...")
        tokenized = [c.content.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        # FAISS
        print("  Embedding chunks for FAISS (this may take a while) ...")
        texts = [c.content for c in chunks]

        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        print(f"  FAISS index: {self.faiss_index.ntotal} vectors, dim={dim}")

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, index_dir: str) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(path / "index.faiss"))

        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        print(f"Index saved → {index_dir}")

    def load(self, index_dir: str) -> None:
        path = Path(index_dir)

        self.faiss_index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        with open(path / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        print(f"Index loaded: {len(self.chunks)} chunks, "
              f"{self.faiss_index.ntotal} vectors")

    # ─── Internal: BM25 search ────────────────────────────────────────────────

    def _bm25_search(
        self,
        query:             str,
        top_k:             int,
        chunk_type_filter: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """
        BM25 keyword search.
        Returns [(chunk_index, bm25_score)] sorted by score desc.
        """
        scores = self.bm25.get_scores(query.lower().split())

        if chunk_type_filter:
            for i, c in enumerate(self.chunks):
                if c.chunk_type != chunk_type_filter:
                    scores[i] = 0.0

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    # ─── Internal: FAISS search ───────────────────────────────────────────────

    def _faiss_search(
        self,
        query:             str,
        top_k:             int,
        chunk_type_filter: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """
        Dense vector search.
        BGE models want a prefix on the QUERY side only (not during indexing).
        Returns [(chunk_index, cosine_score)] sorted by score desc.
        """
        if "bge" in self._embed_model_name.lower():
            query_text = f"Represent this sentence for searching relevant passages: {query}"
        else:
            query_text = query

        q_emb = self.embedder.encode(
            [query_text], normalize_embeddings=True
        ).astype(np.float32)

        search_k = top_k * 4 if chunk_type_filter else top_k * 2
        search_k = min(search_k, len(self.chunks))

        scores, indices = self.faiss_index.search(q_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if chunk_type_filter and chunk.chunk_type != chunk_type_filter:
                continue
            results.append((int(idx), float(score)))
            if len(results) >= top_k:
                break

        return results

    # ─── Internal: RRF fusion ─────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        *ranked_lists: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion over any number of ranked lists.
        Returns [(chunk_index, rrf_score)] sorted desc.
        """
        scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _) in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.RRF_K + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ─── Internal: Cross-encoder reranking ───────────────────────────────────

    def _rerank(
        self,
        query:      str,
        candidates: List[Chunk],
        top_n:      int,
    ) -> List[Tuple[Chunk, float]]:
        """
        Cross-encoder reranking of candidate chunks.
        Returns [(chunk, score)] sorted by score desc.
        Falls back to index order if reranker unavailable.
        """
        if not self.reranker or not candidates:
            return [(c, 0.5 - i * 0.01) for i, c in enumerate(candidates[:top_n])]

        # Truncate content to avoid overloading the cross-encoder
        pairs = [(query, c.content[:600]) for c in candidates]

        try:
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_n]
        except Exception as e:
            print(f"WARNING: Reranker failed ({e}). Using RRF order.")
            return [(c, 0.5 - i * 0.01) for i, c in enumerate(candidates[:top_n])]

    # ─── Public search API ────────────────────────────────────────────────────

    def search(
        self,
        query:          str,
        key_terms:      List[str] = None,
        initial_k:      int = None,
        final_k:        int = None,
        prefer_tables:  bool = False,
    ) -> List[Dict]:
        """
        Main search method.

        Args:
            query:         Original or rewritten query string.
            key_terms:     Important terms from query analysis (boost in BM25).
            initial_k:     Number of candidates for reranking (default from config).
            final_k:       Number of results to return (default from config).
            prefer_tables: Boost table chunk scores (for numerical queries).

        Returns:
            List of result dicts, sorted by relevance.
        """
        initial_k = initial_k or cfg.get("initial_top_k", 20)
        final_k   = final_k   or cfg.get("final_top_k", 5)
        table_boost = cfg.get("table_boost", 1.5)

        # Augment BM25 query with key terms
        bm25_query = f"{query} {' '.join(key_terms)}" if key_terms else query

        # ── Phase 1: Broad hybrid search ─────────────────────────────────────
        bm25_results  = self._bm25_search(bm25_query, top_k=initial_k)
        faiss_results = self._faiss_search(query, top_k=initial_k)
        fused = dict(self._rrf_fuse(bm25_results, faiss_results))

        # ── Phase 2: Table boost ──────────────────────────────────────────────
        if prefer_tables:
            bm25_tables  = self._bm25_search(bm25_query, top_k=10, chunk_type_filter="table")
            faiss_tables = self._faiss_search(query,      top_k=10, chunk_type_filter="table")
            table_fused  = dict(self._rrf_fuse(bm25_tables, faiss_tables))

            for idx, score in table_fused.items():
                fused[idx] = fused.get(idx, 0.0) + score * (table_boost - 1.0)

        # Sort and take top initial_k
        sorted_fused = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:initial_k]

        # ── Phase 3: Rerank ───────────────────────────────────────────────────
        candidates = [self.chunks[idx] for idx, _ in sorted_fused if idx < len(self.chunks)]
        reranked   = self._rerank(query, candidates, top_n=final_k)

        # ── Phase 4: Format output ────────────────────────────────────────────
        results = []
        for chunk, score in reranked:
            results.append({
                "chunk_id":      chunk.chunk_id,
                "source_file":   chunk.source_file,
                "document_name": chunk.document_name,
                "document_type": chunk.document_type,
                "chunk_type":    chunk.chunk_type,
                "heading_path":  chunk.heading_path,
                "section_title": chunk.section_title,
                "academic_year": chunk.academic_year,
                "content":       chunk.content,
                "score":         float(score),
            })

        return results
