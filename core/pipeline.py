"""
pipeline.py
───────────
Orchestrator: pattern pre-router (expanded) → LLM router → execute → answer.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

from config_loader import cfg
from core.llm_handler import LocalLLM
from core.router import route_query
from core.query_engine import QueryEngine
from core.answer_generator import generate_answer
from core.skills import SKILL_NAMES
from core.shuttle_matcher import (
    find_routes_by_location,
    is_generic_listing_query,
)
from db.database import DatabaseManager
from index import corpus_index
from ingestion import shuttle_builder

_pipeline_instance: Optional["AdmissionPipeline"] = None

# ── Pre-router patterns (ordered: specific → generic) ──────────────

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

# Concrete keyword matchers (run before generic WHAT_IS)
DEADLINE_PATTERN = re.compile(
    r"(deadline|last date|closing date|admission schedule|when.*(open|close|start|end))",
    re.IGNORECASE,
)
FEE_PATTERN = re.compile(
    r"(fee|fees|tuition|cost|how much|price|rs\.|pkr|self.?finance)",
    re.IGNORECASE,
)
ELIGIBILITY_PATTERN = re.compile(
    r"(eligib|required|qualification|criteria|requirement|need.*(apply|admission)|entry test)",
    re.IGNORECASE,
)
SUPERVISOR_PATTERN = re.compile(
    r"(supervisor|phd supervisor|research area|phd.*guide|thesis supervisor)",
    re.IGNORECASE,
)
HISTORY_PATTERN = re.compile(
    r"(history|established|founded|when was|background|timeline)",
    re.IGNORECASE,
)
CONTACT_PATTERN = re.compile(
    r"(phone|contact|email|address|telephone|fax|office number|helpline)",
    re.IGNORECASE,
)

# NEW: shuttle / transport pattern — previously did not exist, so any
# shuttle-related query fell through to WHAT_IS_PATTERN (wrong intent:
# PROGRAMS) or to the LLM classifier (which also had no matching intent).
SHUTTLE_PATTERN = re.compile(
    r"(shuttle|bus route|bus service|transport|pick.?up point|drop.?off point|"
    r"which (bus|shuttle)|route.*(campus|university)|commute)",
    re.IGNORECASE,
)

COUNT_PATTERN = re.compile(
    r"(how many|count|total number of|number of|"
    r"tell me the (total )?number of|what is the (total )?count of|"
    r"list all|how many total|count all)",
    re.IGNORECASE,
)
HOW_TO_PATTERN = re.compile(
    r"^(how (do I|to|can I)|what do I need|what are the requirements|steps to|process for)",
    re.IGNORECASE,
)

# NOTE: WHAT_IS_PATTERN used to hardcode intent=PROGRAMS for ANY query
# starting with "what is/are", "tell me about", "describe", "explain" —
# regardless of topic. That swallowed queries like "tell me about shuttle
# routes" into the wrong intent/route. It now ONLY fires when the query
# also contains a program/academic keyword; otherwise it falls through
# to the LLM classifier, which has full context to pick the right intent
# (including the new SHUTTLE and GENERAL intents).
WHAT_IS_PATTERN = re.compile(
    r"^(what (is|are)|tell me about|describe|explain)",
    re.IGNORECASE,
)
WHAT_IS_PROGRAM_TOPIC = re.compile(
    r"(program|degree|course|department|faculty|major|specialization|"
    r"be|bs|ms|me|phd|bachelor|master|doctorate)",
    re.IGNORECASE,
)

# Broad category query — asks about programs generically without specifying level
BROAD_PROGRAM_PATTERN = re.compile(
    r"(programs|degrees?|courses?|departments?|fields?|disciplines?)"
    r"( offered| available| does.*offer| are there| in total| tell me| can i study)",
    re.IGNORECASE,
)

MIN_RAG_SCORE = 0.35


class AdmissionPipeline:
    """Main orchestrator for the admission bot."""

    def __init__(self):
        print("[pipeline] Initialising Admission Pipeline...")
        self.llm    = LocalLLM()
        self.hybrid = self._try_load_hybrid()
        self.qe     = QueryEngine(hybrid_searcher=self.hybrid)
        self.db     = DatabaseManager()
        self.shuttle_df = self._try_load_shuttle()
        self.session_id: str = "default"
        print("[pipeline] Pipeline ready.")

    @staticmethod
    def _try_load_shuttle():
        """Load the structured shuttle-route table (built by shuttle_builder).

        Falls back to an empty DataFrame if it hasn't been built yet — the
        SHUTTLE intent will then fall back to plain RAG search with the
        'shuttle_route' source filter as a degraded-but-functional safety net
        (see _search / _intent_to_source_filter below).
        """
        try:
            df = shuttle_builder.load()
            if len(df) > 0:
                print(f"[pipeline] Shuttle route table loaded ({len(df)} route-legs)")
            else:
                print("[pipeline] Shuttle route table empty — run shuttle_builder.build() "
                      "(via ingestion/build.py) to enable structured shuttle matching.")
            return df
        except Exception as e:
            print(f"[pipeline] Could not load shuttle route table ({e}) — "
                  "SHUTTLE queries will fall back to plain RAG.")
            import pandas as pd
            return pd.DataFrame(columns=["route_id", "leg", "timing", "stops_raw", "stops_list", "notes"])

    @staticmethod
    def _try_load_hybrid():
        try:
            hs = corpus_index.load_hybrid()
            if hs.index is not None:
                print(f"[pipeline] Hybrid searcher loaded ({len(hs.chunks)} chunks)")
                return hs
        except Exception as e:
            print(f"[pipeline] No index found ({e}). Run 'python -m ingestion.build' first.")
        return None

    def set_session(self, session_id: str):
        self.session_id = session_id

    def _save_messages(self, user_query: str, response: str, intent: str):
        self.db.save_message(self.session_id, "user", user_query, intent)
        self.db.save_message(self.session_id, "assistant", response, intent)

    def _build_context_window(self, n: int = 2) -> str:
        """Format the last N QA pairs as conversational context."""
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
        """Route query by pattern. Returns plan dict or None for LLM router."""
        q = query.strip().rstrip("?!.")

        # Canned responses (no LLM call)
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
                '- "Which shuttle route covers Defence?"'
            )
            self._save_messages(query, response, "GREETING")
            return {"_handled": True, "response": response}

        if FAREWELL_PATTERN.match(q):
            response = ("You're welcome! Best of luck with your NED University admissions. "
                        "Feel free to come back anytime!")
            self._save_messages(query, response, "FAREWELL")
            return {"_handled": True, "response": response}

        # Concrete keyword matchers (specific intents, no LLM needed)
        if DEADLINE_PATTERN.search(q):
            return {"route": "CSV_AND_RAG", "intent": "DEADLINES",
                    "filters": {}, "reason": "pattern: deadline"}
        if FEE_PATTERN.search(q):
            return {"route": "RAG", "intent": "FEES",
                    "filters": {}, "reason": "pattern: fee"}
        if ELIGIBILITY_PATTERN.search(q):
            return {"route": "CSV_AND_RAG", "intent": "ELIGIBILITY",
                    "filters": {}, "reason": "pattern: eligibility"}
        if SUPERVISOR_PATTERN.search(q):
            return {"route": "RAG", "intent": "SUPERVISOR",
                    "filters": {}, "reason": "pattern: supervisor"}
        if HISTORY_PATTERN.search(q):
            return {"route": "RAG", "intent": "HISTORY",
                    "filters": {}, "reason": "pattern: history"}
        if CONTACT_PATTERN.search(q):
            return {"route": "RAG", "intent": "CONTACT",
                    "filters": {}, "reason": "pattern: contact"}
        # NEW: shuttle/transport — must run before WHAT_IS_PATTERN so that
        # "tell me about shuttle routes" is not swallowed into PROGRAMS.
        if SHUTTLE_PATTERN.search(q):
            return {"route": "RAG", "intent": "SHUTTLE",
                    "filters": {}, "reason": "pattern: shuttle"}

        # Counting queries (CSV only, falls back to RAG)
        if COUNT_PATTERN.search(q):
            return {"route": "COUNT", "intent": "COUNT",
                    "filters": {}, "reason": "pattern: count"}

        # Broad program query — no specific degree_level mentioned
        if BROAD_PROGRAM_PATTERN.search(q) and not re.search(
            r"(be|bs|b\.e\.|bachelor|ms|m\.s\.|me|m\.e\.|phd|doctorate|"
            r"undergraduate|graduate|postgraduate|master)", q, re.IGNORECASE
        ):
            return {"route": "CSV_AND_RAG", "intent": "PROGRAMS",
                    "filters": {}, "reason": "pattern: broad-program"}

        # How-to / process questions — search broadly across all admission docs
        if HOW_TO_PATTERN.match(q):
            return {"route": "CSV_AND_RAG", "intent": "DOCUMENTS",
                    "filters": {}, "reason": "pattern: how-to"}

        # Generic "what is / tell me about" — ONLY treat as PROGRAMS if the
        # query actually mentions a program/academic keyword. Otherwise fall
        # through to the LLM classifier, which can correctly route topics
        # like shuttle/facilities/hostel/etc. that also use this phrasing.
        if WHAT_IS_PATTERN.match(q) and WHAT_IS_PROGRAM_TOPIC.search(q):
            return {"route": "CSV_AND_RAG", "intent": "PROGRAMS",
                    "filters": {}, "reason": "pattern: what-is-program"}

        return None  # fall through to LLM router

    # ── Main entry point ─────────────────────────────────────────────

    def process_query(self, user_query: str) -> str:
        """Process a user query end-to-end."""
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

        intent = plan.get("intent", "GENERAL")
        route  = plan.get("route", "RAG")
        filters = plan.get("filters", {})
        print(f"[pipeline] Route={route}, Intent={intent}")

        # Step 2: Execute
        csv_rows: list[dict] = []
        evidence_parts: list[dict] = []
        rag_fallback = False

        if route == "COUNT":
            count_data = self.qe.count_programs(filters)
            if count_data["total"] > 0:
                csv_rows = [count_data]
            else:
                print("[pipeline] Count: 0 — falling back to RAG")
                evidence_parts = self._search(user_query, intent)
                intent = "RAG"
                rag_fallback = True

        elif route in ("CSV_QUERY", "CSV_AND_RAG") and len(self.qe.df) > 0:
            result_df = self.qe.query_programs(filters)
            if len(result_df) > 0:
                csv_rows = result_df.head(10).to_dict(orient="records")
            if route == "CSV_AND_RAG" or len(result_df) == 0:
                evidence_parts = self._search(user_query, intent)

        elif route == "RAG" and intent == "SHUTTLE":
            # Deterministic structured lookup instead of semantic RAG —
            # see core/shuttle_matcher.py. Shuttle-stop matching is a
            # finite-vocabulary string-matching problem, not a narrative
            # question-answering problem, so embeddings/BM25 ranking are
            # the wrong tool here.
            if len(self.shuttle_df) > 0:
                matches = find_routes_by_location(user_query, self.shuttle_df)
                if matches:
                    csv_rows = matches
                elif is_generic_listing_query(user_query):
                    # "Tell me about shuttle routes" — wants the full list,
                    # not a top-k semantic-search subset.
                    csv_rows = self.shuttle_df.to_dict(orient="records")
                else:
                    # A specific-sounding location was mentioned but no
                    # stop matched it. Leave evidence empty deliberately —
                    # the skill is instructed to say plainly that no route
                    # covers it, NOT to guess a "closest" route or fall back
                    # to unrelated semantic search results.
                    csv_rows = []
                    evidence_parts = []
            else:
                # Degraded fallback: structured table not built yet.
                print("[pipeline] Shuttle table unavailable — falling back to RAG search")
                evidence_parts = self._search(user_query, intent)

        elif route == "RAG":
            evidence_parts = self._search(user_query, intent)

        elif route == "GENERAL":
            # Genuine catch-all: unrestricted RAG, no CSV, no forced skill
            # template. Used when neither the pre-router nor the classifier
            # confidently maps the query to a known topic.
            evidence_parts = self._search(user_query, intent)

        elif route == "OFF_TOPIC":
            pass

        # Step 3: Generate answer with context memory
        response = generate_answer(
            query=user_query,
            llm=self.llm,
            evidence_parts=evidence_parts,
            intent=intent if not rag_fallback else "RAG",
            csv_rows=csv_rows if csv_rows else None,
            context_history=context,
        )

        self._save_messages(user_query, response, intent)
        return response

    def _search(self, query: str, intent: str) -> list[dict]:
        """Hybrid search with multi-source diversity + score threshold.

        Searches broadly (no source filter) and ensures the top results
        span at least 2 different source files. Only uses intent-specific
        filters for truly single-source intents (supervisor, history, etc.).
        """
        # Determine if this intent should prefer a specific source
        preferred_source = self._intent_to_source_filter(intent)

        if preferred_source is not None:
            # For single-source intents: search with filter first
            results = self.qe.search(query, top_k=8, source_filter=preferred_source)
            good = [r for r in results if r["score"] >= MIN_RAG_SCORE]
            if good:
                return good[:4]
            # Fallback: search without filter
            print(f"[pipeline] Low scores from {preferred_source} — expanding")
            results = self.qe.search(query, top_k=8)
        else:
            # For broad intents: search without filter, aim for source diversity
            results = self.qe.search(query, top_k=10)

        # Filter by score
        good = [r for r in results if r["score"] >= MIN_RAG_SCORE]

        # If we have multiple results, enforce source diversity
        if len(good) >= 2:
            # Always include the top result
            diverse = [good[0]]
            seen_sources = {good[0]["source"]}
            for r in good[1:]:
                if len(diverse) >= 4:
                    break
                # Try to include results from different source files
                src = r["source"]
                if src not in seen_sources or len(seen_sources) >= 3:
                    diverse.append(r)
                    seen_sources.add(src)
            # If diversity enforcement dropped us below 2, add back top results
            if len(diverse) < 2 and len(good) > 1:
                diverse = good[:min(4, len(good))]
            return diverse[:4]

        # If no good results, include what we have (low-score fallback)
        if good:
            return good[:4]

        # Absolute last resort: return anything that scored
        if results:
            print(f"[pipeline] All results below threshold — returning best available")
            return results[:2]

        return []

    @staticmethod
    def _intent_to_source_filter(intent: str) -> str | None:
        """Return a preferred source file for truly single-source intents.

        Only applies to intents where all relevant data lives in ONE file.
        For broad intents (PROGRAMS, DOCUMENTS, ELIGIBILITY, DEADLINES, FEES,
        GENERAL), returns None so the search is unrestricted.
        """
        mapping = {
            # Single-source intents: keep source-filtered
            "FACILITIES": "facilities_info",
            "HOSTEL":     "hostel_facilities",
            "SUPERVISOR": "phd_supervisors",
            "HISTORY":    "university_history",
            "CONTACT":    "ned_links",
            "SHUTTLE":    "shuttle_route",   # NEW — matches your shuttle source file
            # Broad intents removed from filter so multiple sources are searched
            # DOCUMENTS  → None (search FAQ + admissions_schedule + other docs)
            # ELIGIBILITY → None (search eligibility + prospectus + FAQ)
            # DEADLINES  → None (search admissions_schedule + FAQ + other docs)
            # FEES       → None (search multiple fee sources)
            # PROGRAMS   → None (search program text + CSV)
            # GENERAL    → None (unrestricted fallback)
        }
        return mapping.get(intent)


def get_pipeline() -> AdmissionPipeline:
    """Get the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AdmissionPipeline()
    return _pipeline_instance