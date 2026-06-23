"""
hybrid_searcher.py
──────────────────
Hybrid search using Reciprocal Rank Fusion (RRF) to combine
BM25 keyword search + FAISS vector search, with cross-encoder
reranking over the top pool of candidates.

No re-embedding on search — FAISS index is pre-built.

FIXES IN THIS VERSION
─────────────────────
FIX 1 — source_boost parameter:
  search() now accepts an optional source_boost keyword. After RRF fusion
  but before sorting, RRF scores for any chunk whose source stem contains
  source_boost (case-insensitive substring) are multiplied by
  _SOURCE_BOOST_FACTOR (default 3.0). This is a SOFT boost — it raises
  matching chunks in the ranking without completely suppressing other
  sources, so if the boosted source genuinely has no relevant content the
  fall-through to other sources still works. Used by pipeline.py to
  surface specific notice/scheme documents (e.g. PM Laptop Notice) whose
  source stem contains a distinctive keyword from the query (e.g. "laptop").

FIX 2 — expand_pool parameter:
  search() now accepts expand_pool: bool (default False). When True, the
  cross-encoder rerank pool is doubled from _RERANK_POOL_SIZE to
  _RERANK_POOL_SIZE * 2. Used for ELIGIBILITY queries where the correct
  program-eligibility chunk may rank just outside the default pool due to
  the large number of similar chunks in prospectus PDFs. Doubling the pool
  costs one extra batch of cross-encoder inference but prevents the right
  chunk from being silently dropped before the expensive reranker runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config_loader import cfg
from index.chunker import Chunk
from index.vector_store import VectorStore

_RAG_TOP_K        = int(cfg.get("retrieval", {}).get("rag_top_k", 4))
_RERANKER_MODEL   = cfg["reranker_model"]          # still top-level, no change
_RERANK_POOL_SIZE = int(cfg["rerank_pool_size"])    # still top-level, no change
_EMBED_QUERY_PFX  = cfg.get("vector_store", {}).get("embed_query_prefix", "")


# RRF constant — chunks ranked below this by one method still get some score
_RRF_K = 60

# FIX 1: Multiplier applied to RRF scores for chunks whose source matches
# the source_boost hint. Set high enough to consistently surface the target
# document, but not so high that a completely irrelevant chunk from the
# target source beats a highly relevant chunk from another source.
_SOURCE_BOOST_FACTOR = 3.0


class HybridSearcher:
    """
    Hybrid search: BM25 + FAISS via RRF fusion, then cross-encoder rerank.

    Usage:
        hs = HybridSearcher()
        hs.load(index_path)
        results = hs.search("PhD supervisor in AI", top_k=4)
        results = hs.search("BE Civil", source_filter="academic_programmes")
        # FIX 1: boost a specific source document
        results = hs.search("pm laptop notice", top_k=4, source_boost="laptop")
        # FIX 2: larger rerank pool for eligibility
        results = hs.search("MS Data Science eligibility", top_k=4, expand_pool=True)
    """

    def __init__(self):
        self.vs: Optional[VectorStore] = None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self.reranker: Optional[CrossEncoder] = None

    def load(self, path: Path):
        """Load the FAISS index + chunk metadata, then build BM25 index."""
        self.vs = VectorStore()
        self.vs.load(path)
        self.chunks = self.vs.chunks

        print(f"[hybrid] Building BM25 index ({len(self.chunks)} chunks) …")
        self._tokenized = [
            c.content.lower().split() for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self._tokenized)
        print(f"[hybrid] BM25 index ready ({len(self._tokenized)} docs)")

        print(f"[hybrid] Loading reranker '{_RERANKER_MODEL}' …")
        self.reranker = CrossEncoder(_RERANKER_MODEL)
        print(f"[hybrid] Reranker ready")

    @property
    def index(self):
        """Expose the underlying FAISS index for direct vector search."""
        return self.vs.index if self.vs else None

    def search(
        self,
        query: str,
        top_k: int = _RAG_TOP_K,
        source_filter: str | None = None,
        source_boost: str | None = None,   # FIX 1
        expand_pool: bool = False,          # FIX 2
    ) -> list[dict]:
        """
        Hybrid search via Reciprocal Rank Fusion.

        Parameters
        ----------
        query         : the user query text
        top_k         : number of results to return
        source_filter : hard filter — only consider chunks from sources
                        whose stem contains this substring (unchanged from
                        the original implementation)
        source_boost  : FIX 1 — soft boost — multiply RRF scores for
                        chunks from sources containing this keyword before
                        reranking. None = no boost (default behaviour).
        expand_pool   : FIX 2 — when True, use 2× rerank pool size so the
                        cross-encoder sees more candidates. Useful when the
                        correct chunk is expected to rank just outside the
                        default pool. None = normal pool (default behaviour).

        Phase 1 — BM25: score all chunks by keyword match.
        Phase 2 — FAISS: score all chunks by semantic similarity.
        Fusion  — RRF: combine ranks from both methods.
        FIX 1   — Source boost applied to RRF scores (if requested).
        Phase 3 — Cross-encoder reranking over top pool (FIX 2: pool may be 2×).
        """
        if self.vs is None or self.bm25 is None or not self.chunks:
            return []

        n = len(self.chunks)

        # ── Determine which indices to consider ──────────────────────
        if source_filter:
            indices = [
                i for i, c in enumerate(self.chunks)
                if source_filter.lower() in c.source.lower()
            ]
        else:
            indices = list(range(n))

        if not indices:
            return []

        # ── Phase 1: BM25 scores ─────────────────────────────────────
        query_tokens = query.lower().split()
        bm25_raw = self.bm25.get_scores(query_tokens)

        bm25_ranked = sorted(indices, key=lambda i: bm25_raw[i], reverse=True)
        bm25_rank: dict[int, int] = {
            idx: pos for pos, idx in enumerate(bm25_ranked)
        }

        # ── Phase 2: FAISS vector search (no re-embedding) ───────────
        prefixed_query = _EMBED_QUERY_PFX + query if _EMBED_QUERY_PFX else query
        query_vec = np.array(
            self.vs.model.encode([prefixed_query], normalize_embeddings=True,
                                 show_progress_bar=False),
            dtype="float32",
        )
        search_k = min(n, 200)
        faiss_scores, faiss_indices = self.vs.index.search(query_vec, search_k)

        faiss_rank: dict[int, int] = {}
        for pos, idx in enumerate(faiss_indices[0]):
            if idx >= 0 and idx in indices:
                faiss_rank[int(idx)] = pos

        # ── Fusion: RRF ──────────────────────────────────────────────
        rrf_scores: dict[int, float] = {}
        all_candidates = set(bm25_rank.keys()) | set(faiss_rank.keys())

        for idx in all_candidates:
            score = 0.0
            if idx in bm25_rank:
                score += 1.0 / (_RRF_K + bm25_rank[idx])
            if idx in faiss_rank:
                score += 1.0 / (_RRF_K + faiss_rank[idx])
            rrf_scores[idx] = score

        # FIX 1: Source boost — multiply RRF scores for chunks from the
        # hinted source. Applied after fusion so both BM25 and vector
        # signals contribute before the boost is applied, preventing a
        # zero-relevance chunk from a matching source from being ranked #1.
        # The boost is intentionally soft (multiplicative, not additive) so
        # that a highly relevant chunk from a non-boosted source can still
        # outrank a weakly relevant chunk from the boosted source when the
        # score difference is large enough.
        if source_boost:
            boost_kw = source_boost.lower()
            n_boosted = 0
            for idx in rrf_scores:
                if boost_kw in self.chunks[idx].source.lower():
                    rrf_scores[idx] *= _SOURCE_BOOST_FACTOR
                    n_boosted += 1
            if n_boosted:
                print(f"[hybrid] source_boost='{source_boost}' → boosted {n_boosted} chunks "
                      f"(×{_SOURCE_BOOST_FACTOR})")
            else:
                print(f"[hybrid] source_boost='{source_boost}' → no matching chunks found")

        # Sort by RRF score descending (post-boost)
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # ── Phase 3: Cross-encoder reranking ───────────────────────────
        # FIX 2: expand_pool doubles the candidate pool so the reranker
        # has a better chance of seeing the correct chunk for queries where
        # it may rank just outside the default pool (e.g. eligibility
        # sections in large prospectus PDFs with many similar sections).
        pool_size = (_RERANK_POOL_SIZE * 2) if expand_pool else _RERANK_POOL_SIZE
        if expand_pool:
            print(f"[hybrid] expand_pool=True → rerank pool: {pool_size} (2× default)")

        rerank_pool = ranked[:pool_size]
        if self.reranker is not None and rerank_pool:
            pairs = [(query, self.chunks[idx].content) for idx, _ in rerank_pool]
            ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
            reranked = [
                (idx, float(score), float(ce_scores[i]))
                for i, (idx, score) in enumerate(rerank_pool)
            ]
            reranked.sort(key=lambda x: x[2], reverse=True)
        else:
            reranked = [
                (idx, float(score), float(score))
                for idx, score in rerank_pool
            ]

        # ── Log scores for diagnostics when results are sparse ──────
        if not reranked:
            print(f"[hybrid] No reranked candidates for query — BM25 had {len(bm25_rank)} docs, "
                  f"FAISS had {len(faiss_rank)} docs")
        elif len(reranked) < top_k:
            print(f"[hybrid] Only {len(reranked)} candidates after rerank (requested {top_k}) — "
                  f"top RRF score: {reranked[0][1]:.4f}, top CE score: {reranked[0][2]:.4f}")

        # ── Build results with source diversity ─────────────────────
        seen_ids: set[str] = set()
        initial: list[dict] = []
        for idx, rrf_score, ce_score in reranked:
            chunk = self.chunks[idx]
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            initial.append({
                "chunk_id":     chunk.chunk_id,
                "source":       chunk.source,
                "topic":        chunk.topic,
                "score":        rrf_score,
                "rerank_score": ce_score,
                "content":      chunk.content,
            })
            if len(initial) >= top_k:
                break

        # Source diversity pass: if top-k are dominated by one source,
        # promote the best result from each underrepresented source.
        # NOTE: When source_boost is active, the boosted source will
        # naturally dominate — skip diversity enforcement in that case so
        # we don't accidentally demote the boosted source's best results.
        if len(initial) > 1 and not source_boost:
            sources = [r["source"] for r in initial]
            unique_srcs = set(sources)
            required_diversity = min(2, top_k)
            if len(unique_srcs) < required_diversity:
                seen_ids.clear()
                for r in initial:
                    seen_ids.add(r["chunk_id"])
                extra: list[dict] = []
                for idx, rrf_score, ce_score in reranked:
                    if len(extra) + len(initial) >= top_k:
                        break
                    chunk = self.chunks[idx]
                    if chunk.chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk.chunk_id)
                    if chunk.source not in unique_srcs:
                        extra.append({
                            "chunk_id":     chunk.chunk_id,
                            "source":       chunk.source,
                            "topic":        chunk.topic,
                            "score":        rrf_score,
                            "rerank_score": ce_score,
                            "content":      chunk.content,
                        })
                        unique_srcs.add(chunk.source)
                results = initial[:1] + extra + initial[1:]
                results = results[:top_k]
            else:
                results = initial[:top_k]
        else:
            results = initial[:top_k]

        return results