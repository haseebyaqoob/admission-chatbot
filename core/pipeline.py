"""
pipeline.py
───────────
Orchestrator: pattern pre-router → config-driven tool dispatch → answer.

Dispatch is built from the tools list in config.yaml. No hardcoded source
names or if/elif intent branches in the dispatch logic — every decision
flows from tool config (source_type, filters_supported, source_file).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config_loader import cfg
from core.llm_handler import LocalLLM
from core.router import route_query
from core.query_engine import QueryEngine
from core.answer_generator import generate_answer
from core.shuttle_matcher import find_routes_by_location
from core.supervisor_matcher import find_supervisors_by_area
from db.database import DatabaseManager
from index import corpus_index
from ingestion import shuttle_builder, supervisor_builder

_pipeline_instance: Optional["AdmissionPipeline"] = None

# ── Load tool registry from config ────────────────────────────────────
_TOOLS: list[dict] = cfg.get("tools", [])
_TOOL_MAP: dict[str, dict] = {t["name"]: t for t in _TOOLS}
_FALLBACK_SOURCE = _TOOLS[-1]["name"] if _TOOLS else "corpus"

# Map config tool names → skill prompt names (skills.py SKILL_NAMES).
# Add one entry here whenever a new structured tool is added to config.yaml.
_SOURCE_TO_SKILL: dict[str, str] = {
    "programs":      "PROGRAMS",
    "supervisors":   "SUPERVISOR",
    "shuttle_routes": "SHUTTLE",
    "scholarships":  "SCHOLARSHIPS",
}


def _get_skill_name(source: str) -> str:
    return _SOURCE_TO_SKILL.get(source, "GENERAL")


def _get_tool(name: str) -> dict:
    return _TOOL_MAP.get(name, _TOOL_MAP.get(_FALLBACK_SOURCE, {}))


# ── Score threshold — read from config so it can be tuned without a
#    code change. Default 0.08 matches observed reranker score range for
#    BAAI/bge-base-en-v1.5 + ms-marco-MiniLM-L-6-v2 on this corpus.
#    (Was previously a hardcoded constant of 0.35 which discarded all
#    valid matches.)
MIN_RAG_SCORE = float(cfg.get("retrieval", {}).get("min_rag_score", 0.08))

# ── Pre-router patterns (unchanged from before) ───────────────────────

OFF_TOPIC_PATTERN = re.compile(
    r"^(how are you|what can you do|who are you|who made you|what is your name)$",
    re.IGNORECASE,
)
GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|assalam o alaikum|salam|good morning|good afternoon|good evening)$",
    re.IGNORECASE,
)
FAREWELL_PATTERN = re.compile(
    r"^(bye|goodbye|see you|take care|thanks|thank you|thankyou|thanks a lot)$",
    re.IGNORECASE,
)

NOTICE_KEYWORD_PATTERN = re.compile(
    r"\b(laptop|pmyls|pm\s+laptop|prime\s+minister.*laptop|"
    r"laptop\s+scheme|youth\s+laptop\s+scheme|laptop\s+notice)\b",
    re.IGNORECASE,
)

COUNT_PATTERN = re.compile(
    r"(how many|count|total number of|number of|"
    r"tell me the (total )?number of|what is the (total )?count of|"
    r"how many total|count all)",
    re.IGNORECASE,
)


def _tag_source(rows: list[dict], source_file: str) -> list[dict]:
    for row in rows:
        row.setdefault("_source_file", source_file)
    return rows


class AdmissionPipeline:
    """Main orchestrator — dispatch is built from config tools."""

    def __init__(self):
        print("[pipeline] Initialising Admission Pipeline...")
        self.llm = LocalLLM()
        self.hybrid = self._try_load_hybrid()
        self.qe = QueryEngine(hybrid_searcher=self.hybrid)
        self.db = DatabaseManager()

        # Load structured data sources from config tool registry.
        # Any tool with source_type: structured and a source_file is
        # loaded automatically here — no per-tool code changes needed.
        self._dataframes: dict[str, pd.DataFrame] = {}
        for t in _TOOLS:
            if t["source_type"] == "structured" and t.get("source_file"):
                sf = t["source_file"]
                df = self._load_tool_csv(sf, t["name"])
                if df is not None:
                    self._dataframes[t["name"]] = df
                    print(
                        f"[pipeline] Loaded '{t['name']}' → "
                        f"{len(df)} rows from {sf}"
                    )

        self.session_id: str = "default"
        print("[pipeline] Pipeline ready.")

    @staticmethod
    def _load_tool_csv(source_file: str, tool_name: str) -> pd.DataFrame | None:
        path = Path(source_file)
        if not path.exists():
            print(
                f"[pipeline] Source file not found for tool "
                f"'{tool_name}': {path}"
            )
            return None
        try:
            if tool_name == "shuttle_routes":
                return shuttle_builder.load()
            elif tool_name == "supervisors":
                return supervisor_builder.load()
            else:
                return pd.read_csv(path)
        except Exception as e:
            print(f"[pipeline] Could not load '{tool_name}' from {path}: {e}")
            return None

    @staticmethod
    def _try_load_hybrid():
        try:
            hs = corpus_index.load_hybrid()
            if hs.index is not None:
                print(
                    f"[pipeline] Hybrid searcher loaded "
                    f"({len(hs.chunks)} chunks)"
                )
                return hs
        except Exception as e:
            print(
                f"[pipeline] No index found ({e}). "
                "Run 'python -m ingestion.build' first."
            )
        return None

    def set_session(self, session_id: str):
        self.session_id = session_id

    def _save_messages(self, user_query: str, response: str, source: str):
        self.db.save_message(self.session_id, "user", user_query, source)
        self.db.save_message(self.session_id, "assistant", response, source)

    def _build_context_window(self, n: int = 2) -> str:
        history = self.db.get_recent_history(self.session_id, n=n)
        if not history:
            return ""
        lines: list[str] = []
        for h in history:
            role = "User" if h["role"] == "user" else "Assistant"
            msg = h["message"][:200]
            lines.append(f"{role}: {msg}")
        return "\n".join(lines)

    # ── Pattern-based pre-router ─────────────────────────────────────

    def _pre_route(self, query: str) -> dict | None:
        """Returns plan dict or None to fall through to LLM router."""
        q = query.strip().rstrip("?!.")

        if OFF_TOPIC_PATTERN.match(q):
            response = (
                "I can only answer questions related to NED University admissions.\n\n"
                "Here's what I can help with:\n"
                "- Programs and degrees offered\n"
                "- Eligibility criteria\n"
                "- Fee information\n"
                "- Admission deadlines\n"
                "- Required documents\n"
                "- Hostel and facilities\n"
                "- Shuttle / transport routes\n"
                "- PhD supervisors\n"
                "- Scholarships\n"
                "- University history and contact info"
            )
            self._save_messages(query, response, "OFF_TOPIC")
            return {"_handled": True, "response": response}

        if GREETING_PATTERN.match(q):
            response = (
                "👋 Hello! Welcome to the NED Admissions Assistant.\n\n"
                "**Try asking:**\n"
                '- "What programs does CS offer?"\n'
                '- "What is the eligibility for BE?"\n'
                '- "When is the last date for admission 2026?"\n'
                '- "Find PhD supervisors in AI"\n'
                '- "What scholarships are available?"\n'
                '- "Which shuttle route covers Defence?"'
            )
            self._save_messages(query, response, "GREETING")
            return {"_handled": True, "response": response}

        if FAREWELL_PATTERN.match(q):
            response = (
                "You're welcome! Best of luck with your NED University "
                "admissions. Feel free to come back anytime!"
            )
            self._save_messages(query, response, "FAREWELL")
            return {"_handled": True, "response": response}

        notice_match = NOTICE_KEYWORD_PATTERN.search(q)
        if notice_match:
            raw_match = notice_match.group(1)
            source_hint = raw_match.lower().split()[0]
            print(
                f"[pipeline] Notice keyword '{raw_match}' → "
                f"source_hint='{source_hint}'"
            )
            return {
                "source":      "corpus",
                "operation":   "SEARCH",
                "query_mode":  "SEARCH",
                "filters":     {},
                "source_hint": source_hint,
                "reason":      f"pattern: notice-keyword ({raw_match})",
            }

        # Counting: route directly without LLM call
        if COUNT_PATTERN.search(q):
            return {
                "source":     "programs",
                "operation":  "SEARCH",
                "query_mode": "SEARCH",
                "filters":    {},
                "reason":     "pattern: count",
                "_is_count":  True,
            }

        return None  # fall through to LLM router

    # ── Structured data handler (config-driven) ──────────────────────

    def _handle_structured(
        self,
        tool: dict,
        user_query: str,
        operation: str,
        query_mode: str,
        filters: dict,
    ) -> tuple[list[dict], list[dict]]:
        """Handle a structured source query.

        Returns (csv_rows, evidence_parts).
        """
        name = tool["name"]
        df = self._dataframes.get(name)
        if df is None or len(df) == 0:
            print(
                f"[pipeline] Structured source '{name}' has no data "
                "— falling back to search"
            )
            return [], self._search(user_query, query_mode=query_mode)

        source_file = (
            Path(str(tool.get("source_file", ""))).name or f"{name}.csv"
        )

        # LIST mode: return all rows
        if query_mode == "LIST" or operation == "LIST":
            rows = _tag_source(df.to_dict(orient="records"), source_file)
            print(f"[pipeline] {name} LIST: {len(rows)} rows")
            return rows, []

        # SEARCH mode — delegate to tool-specific matchers
        filters_supported = tool.get("filters_supported", [])

        if "research_area" in filters_supported:
            research_area = filters.get("research_area")
            rows = find_supervisors_by_area(
                user_query, df, research_area=research_area
            )
            rows = _tag_source(rows, source_file)
            return rows, []

        if "area" in filters_supported or "route_id" in filters_supported:
            rows = find_routes_by_location(user_query, df)
            if not rows:
                rows = df.to_dict(orient="records")
            rows = _tag_source(rows, source_file)
            return rows, []

        if (
            "department" in filters_supported
            or "degree_level" in filters_supported
        ):
            filtered = self._filter_dataframe(df, filters)
            if len(filtered) > 0:
                rows = _tag_source(
                    filtered.head(10).to_dict(orient="records"), source_file
                )
            else:
                rows = _tag_source(
                    df.head(10).to_dict(orient="records"), source_file
                )
            # Also return evidence from corpus for enrichment
            evidence = self._search(user_query, query_mode=query_mode)
            return rows, evidence

        # Generic fallback: filter by any matching columns (or return all
        # rows when filters is empty, as for the scholarships tool).
        filtered = self._filter_dataframe(df, filters)
        rows = _tag_source(
            filtered.to_dict(orient="records")
            if len(filtered) > 0
            else df.to_dict(orient="records"),
            source_file,
        )
        return rows, []

    @staticmethod
    def _filter_dataframe(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        result = df.copy()
        for key, val in filters.items():
            if val and key in result.columns:
                result = result[
                    result[key]
                    .astype(str)
                    .str.contains(str(val), case=False, na=False)
                ]
        return result

    # ── Main entry point ─────────────────────────────────────────────

    def process_query(self, user_query: str) -> str:
        context = self._build_context_window()

        # Step 1: Pattern-based pre-router
        plan = self._pre_route(user_query)
        if plan and plan.get("_handled"):
            return plan["response"]

        if plan is None:
            plan = route_query(
                user_query=user_query,
                llm=self.llm,
                df=self.qe.df if len(self.qe.df) > 0 else None,
                retry_once=True,
            )

        source      = plan.get("source", _FALLBACK_SOURCE)
        operation   = plan.get("operation", "SEARCH")
        query_mode  = plan.get("query_mode", "SEARCH")
        filters     = plan.get("filters", {})
        is_count    = plan.get("_is_count", False)
        source_hint: str | None = plan.get("source_hint")
        if source_hint is None and isinstance(filters, dict):
            source_hint = filters.get("source_hint")

        tool = _get_tool(source)
        print(
            f"[pipeline] Source={source}, Operation={operation}, "
            f"Mode={query_mode}"
            + (f", source_hint={source_hint}" if source_hint else "")
        )

        # Step 2: Execute via config-driven dispatch
        csv_rows: list[dict] = []
        evidence_parts: list[dict] = []

        if is_count:
            count_data = self.qe.count_programs(filters)
            if count_data["total"] > 0:
                source_file = Path(
                    str(tool.get("source_file", "programs.csv"))
                ).name
                csv_rows = _tag_source([count_data], source_file)
            else:
                evidence_parts = self._search(user_query, query_mode=query_mode)

        elif tool.get("source_type") == "structured":
            csv_rows, extra_evidence = self._handle_structured(
                tool, user_query, operation, query_mode, filters,
            )
            if not evidence_parts:
                evidence_parts = extra_evidence

        elif tool.get("source_type") == "unstructured":
            # Build augmented query for eligibility/fees within the corpus
            rag_query = user_query
            if source == "corpus":
                rag_query = self._augment_query(user_query, filters)
            evidence_parts = self._search(
                rag_query,
                source_hint=source_hint,
                query_mode=query_mode,
            )

        # Step 2.5: Grade retrieved evidence for relevance
        if evidence_parts and not csv_rows:
            evidence_parts = self._grade_evidence(user_query, evidence_parts)

        # Step 3: Generate answer
        response = generate_answer(
            query=user_query,
            llm=self.llm,
            evidence_parts=evidence_parts,
            intent=_get_skill_name(source),
            csv_rows=csv_rows if csv_rows else None,
            context_history=context,
        )

        self._save_messages(user_query, response, source)
        return response

    @staticmethod
    def _augment_query(user_query: str, filters: dict) -> str:
        """Augment query with filter context for better BM25 recall."""
        if not filters:
            return user_query
        filter_parts = [
            str(v)
            for k, v in filters.items()
            if k in ("degree_level", "department") and v
        ]
        if filter_parts:
            return (
                f"{' '.join(filter_parts)} {user_query} "
                "eligibility admission fee criteria requirements"
            )
        return user_query

    # ── Retrieval relevance grading ─────────────────────────────────

    def _grade_evidence(
        self, query: str, chunks: list[dict]
    ) -> list[dict]:
        """LLM-based relevance grading on retrieved chunks.

        Returns only chunks that pass the relevance check. Config for
        thresholds lives in config.yaml under the ``retrieval`` section.
        """
        if not chunks:
            return []

        rc = cfg.get("retrieval", {})
        max_grade = int(rc.get("max_chunks_to_grade", 6))
        threshold = float(rc.get("grade_threshold", 0.25))
        min_pass = int(rc.get("min_passing_chunks", 2))
        grading_prompt_tpl: str = rc.get("grading_prompt", "")

        if not grading_prompt_tpl:
            return chunks

        to_grade = chunks[:max_grade]
        passed: list[dict] = []

        for chunk in to_grade:
            chunk_text = chunk.get("content", "")[:800]
            prompt = (
                grading_prompt_tpl
                .replace("{question}", query)
                .replace("{chunk_text}", chunk_text)
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are a concise relevance grader. "
                               "Answer YES or NO only.",
                },
                {"role": "user", "content": prompt},
            ]
            try:
                raw = self.llm.chat(
                    messages, max_new_tokens=8, temperature=0.0
                ).strip().upper()
            except Exception as e:
                print(f"[pipeline] Grading LLM call failed: {e}")
                raw = "YES"  # pass on error (fail open)
            if raw.startswith("YES"):
                passed.append(chunk)
            else:
                print(
                    f"[pipeline] Grading: chunk from "
                    f"'{chunk.get('source', '?')}' filtered out "
                    f"(score={chunk.get('score', '?')})"
                )

        count_passed = len(passed)
        count_total = len(to_grade)
        frac = count_passed / count_total if count_total > 0 else 0
        print(
            f"[pipeline] Grading: {count_passed}/{count_total} passed "
            f"(threshold={threshold})"
        )

        if frac >= threshold and count_passed >= min_pass:
            return passed
        # Fallback: not enough passed — return original set unchanged
        print(
            f"[pipeline] Grading: below threshold — returning original "
            f"{len(chunks)} chunks unchanged"
        )
        return chunks

    # ── Hybrid search (preserved from previous versions) ─────────────

    def _search(
        self,
        query: str,
        source_hint: str | None = None,
        query_mode: str = "SEARCH",
    ) -> list[dict]:
        """Hybrid search with multi-source diversity + score threshold.

        Used by the unstructured (FAISS) dispatch path.
        """
        if self.hybrid is None:
            return self.qe.search(query, top_k=8) if self.qe else []

        expand_pool = False

        if source_hint:
            results = self.hybrid.search(
                query,
                top_k=10,
                source_boost=source_hint,
                expand_pool=expand_pool,
            )
            if source_hint:
                print(
                    f"[pipeline] Applied source_boost='{source_hint}' "
                    f"({len(results)} candidates)"
                )
        else:
            results = self.hybrid.search(
                query, top_k=10, expand_pool=expand_pool
            )

        threshold = MIN_RAG_SCORE
        if query_mode == "LIST":
            threshold = MIN_RAG_SCORE * 0.5
            print(
                f"[pipeline] LIST mode: using reduced score threshold "
                f"{threshold:.2f}"
            )

        good = [r for r in results if r["score"] >= threshold]

        if len(good) >= 2:
            diverse = [good[0]]
            seen_sources = {good[0]["source"]}
            for r in good[1:]:
                if len(diverse) >= 4:
                    break
                src = r["source"]
                if src not in seen_sources or len(seen_sources) >= 3:
                    diverse.append(r)
                    seen_sources.add(src)
            if len(diverse) < 2 and len(good) > 1:
                diverse = good[:min(4, len(good))]
            return diverse[:4]

        if good:
            return good[:4]

        if results:
            scores_str = ", ".join(
                f"{r['score']:.3f}" for r in results[:5]
            )
            print(
                f"[pipeline] All {len(results)} results below threshold "
                f"{threshold:.2f} — scores: [{scores_str}]"
            )
            return results[:2]

        print(f"[pipeline] No results found for query (mode={query_mode})")
        return []


def get_pipeline() -> AdmissionPipeline:
    """Get the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AdmissionPipeline()
    return _pipeline_instance
