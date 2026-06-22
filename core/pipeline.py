"""
pipeline.py
───────────
Orchestrator: pattern pre-router (expanded) → LLM router → execute → answer.

FIXES IN THIS VERSION
─────────────────────
FIX (LIST mode):
  Added query_mode support from router.py. When query_mode == "LIST",
  semantic search is skipped entirely and the full dataset for the
  detected intent is returned directly (all supervisors, all programs,
  all shuttle routes, etc.). Score thresholds are halved for LIST mode
  to prevent legitimate enumeration results from being filtered out.

FIX 1 — PM Laptop Notice retrieval (Failure 1):
  Added NOTICE_KEYWORD_PATTERN to catch "laptop", "pmyls", "pm laptop", etc.
  The matched keyword is used as a source-boost hint (not a hard filename —
  the keyword is a substring of the actual source stem, so no hardcoding).
  _search() accepts an optional source_hint and passes it to
  hybrid_searcher.search() as a soft score multiplier so the correct
  document rises to the top without suppressing other sources entirely.

FIX 2 — MS Data Science eligibility retrieval (Failure 2):
  For ELIGIBILITY intent, the RAG search query is augmented with extracted
  filter context (degree_level / department) and standard eligibility terms,
  improving BM25 recall for program-specific sections in large PDFs.
  _search() also tells hybrid_searcher to use an expanded rerank pool
  (expand_pool=True) so the correct chunk isn't dropped before the
  cross-encoder sees it.

FIX 3 — Shuttle synonym "point" (Failure 3):
  SHUTTLE_PATTERN is extended to match "which point", "boarding point",
  "point goes/passes through", and similar phrasing so these queries
  reach the SHUTTLE intent before the LLM classifier is invoked.
  (See also router.py for the LLM-prompt half of this fix, and
  shuttle_matcher.py for the stopword-list half.)

FIX 5 — Fees retrieval never reaching Stage 2 (this version):
  FEE_PATTERN previously returned a complete plan dict directly from
  _pre_route() with "filters": {} hardcoded — this bypassed route_query()
  entirely, meaning NEITHER Stage 1 NOR Stage 2 of the LLM router ever ran
  for fee queries. In particular, Stage 2's filter extractor (which now
  also extracts source_hint — see router.py) never got a chance to run,
  so fee queries had no source_hint to boost the correct prospectus
  document, and the LLM frequently answered "no fee information found"
  even when the fee table existed in the corpus.

  Fix mirrors SUPERVISOR_PATTERN's existing pattern: detect-but-don't-
  early-return, so the query falls through to the LLM router and gets
  full Stage 1 (intent confirmation) + Stage 2 (filter + source_hint
  extraction) treatment. The regex itself is left in place (currently
  unused for short-circuiting, but harmless / available for future
  telemetry or pre-classification hints) — see note at its use site below.

FIX 6 — Eligibility/Fees consistency (this version):
  ELIGIBILITY_PATTERN had the EXACT SAME early-return bug as FEE_PATTERN
  (FIX 5, above): it returned a complete plan with "filters": {}
  hardcoded directly from _pre_route(), bypassing route_query() and
  therefore Stage 2's filter extractor entirely. This meant the
  ELIGIBILITY query-augmentation logic inside _search()/process_query()
  (which only fires when `filters` is non-empty) had been unreachable
  dead code in production — Fix 2's "augmented RAG query" benefit
  (degree_level/department context for better BM25 recall on e.g.
  "MS Data Science") never actually applied. Now fixed identically to
  FEE_PATTERN: detect-but-fall-through, so ELIGIBILITY queries get full
  Stage 1 + Stage 2 treatment, same 3-LLM-call cost as FEES/PROGRAMS/
  SUPERVISOR.
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
from core.shuttle_matcher import find_routes_by_location
from core.supervisor_matcher import find_supervisors_by_area
from db.database import DatabaseManager
from index import corpus_index
from ingestion import shuttle_builder, supervisor_builder

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
# FIX (fees): FEE_PATTERN is now detect-only — it is no longer used to
# early-return a complete plan from _pre_route(). It's kept defined here
# in case a future caller wants a cheap "does this look fee-related"
# check (e.g. logging/telemetry), but it must NOT be used to bypass the
# LLM router. See _pre_route() below for where the old early-return used
# to be, and the comment there explaining why it was removed.
FEE_PATTERN = re.compile(
    r"(fee|fees|tuition|cost|how much|price|rs\.|pkr|self.?finance)",
    re.IGNORECASE,
)
# FIX (eligibility consistency): ELIGIBILITY_PATTERN is now detect-only —
# it is no longer used to early-return a complete plan from _pre_route().
# This mirrors the exact same fix already applied to FEE_PATTERN above.
# Previously this pattern early-returned {"route": "CSV_AND_RAG",
# "intent": "ELIGIBILITY", "filters": {}}, bypassing route_query()
# entirely — so Stage 2's filter extractor never ran for ELIGIBILITY
# queries either, and pipeline.py's own ELIGIBILITY query-augmentation
# block (`if intent == "ELIGIBILITY" and filters:` below) could never
# fire, because `filters` was always the hardcoded empty dict. Fix 2's
# "augmented RAG query" half (degree_level/department context improving
# BM25 recall for "MS Data Science" in large prospectus PDFs) had never
# actually been running in production, even though it was implemented —
# it was simply unreachable code.
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

# FIX 3: Extended SHUTTLE_PATTERN — adds transit-stop synonyms:
#   "point" (e.g. "which point goes through Nazimabad?")
#   "boarding point", "alighting point", "bus stop"
#   "goes/passes through" phrasing commonly used for route queries
# These are common in Pakistan English for bus/shuttle stops and were
# previously missed, causing mis-classification as SUPERVISOR or GENERAL.
SHUTTLE_PATTERN = re.compile(
    r"(shuttle|bus\s+route|bus\s+service|transport|pick.?up\s+point|drop.?off\s+point|"
    r"which\s+(bus|shuttle)|route.*(campus|university)|commute|"
    # FIX 3a: explicit stop/boarding synonyms
    r"boarding\s+point|alighting\s+point|bus\s+stop|drop\s+point|"
    # FIX 3b: "which/what point" — query asking about a specific stop
    r"which\s+point|what\s+point|"
    # FIX 3c: "point goes/passes through X" or "goes/passes through [location]"
    r"point\s+(go|goes|pass|passes|cover|covers|through)|"
    r"(go|goes|pass|passes)\s+through\b|"
    # FIX 3d: "get on/off" phrasing
    r"get\s+(on|off)\s+(at|near|the)|"
    r"where\s+(do|can)\s+i\s+(board|get\s+on|get\s+off))",
    re.IGNORECASE,
)

# FIX 1: Notice / government-scheme keyword pattern.
# Detects queries about specific notices or schemes that have a dedicated
# source document in the corpus. The matched keyword is used as a
# source-boost hint in _search() — it's a substring of the actual source
# stem (e.g. "laptop" matches "PM_LAPTOP_NOTICE_MASTER_extracted"), so no
# filename is hardcoded and any future notice document whose filename
# contains a distinctive keyword will be picked up automatically.
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
HOW_TO_PATTERN = re.compile(
    r"^(how (do I|to|can I)|what do I need|what are the requirements|steps to|process for)",
    re.IGNORECASE,
)

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

_SUPERVISOR_SOURCE_FILE = "supervisors.csv"
_SHUTTLE_SOURCE_FILE    = "shuttle_routes.csv"
_PROGRAMS_SOURCE_FILE   = "programs.csv"


def _tag_source(rows: list[dict], source_file: str) -> list[dict]:
    """Attach a `_source_file` field to every structured row, without
    overwriting one a caller may have already set."""
    for row in rows:
        row.setdefault("_source_file", source_file)
    return rows


class AdmissionPipeline:
    """Main orchestrator for the admission bot."""

    def __init__(self):
        print("[pipeline] Initialising Admission Pipeline...")
        self.llm    = LocalLLM()
        self.hybrid = self._try_load_hybrid()
        self.qe     = QueryEngine(hybrid_searcher=self.hybrid)
        self.db     = DatabaseManager()
        self.shuttle_df = self._try_load_shuttle()
        self.supervisor_df = self._try_load_supervisors()
        self.session_id: str = "default"
        print("[pipeline] Pipeline ready.")

    @staticmethod
    def _try_load_shuttle():
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
    def _try_load_supervisors():
        try:
            df = supervisor_builder.load()
            if len(df) > 0:
                print(f"[pipeline] Supervisor table loaded ({len(df)} supervisors)")
            else:
                print("[pipeline] Supervisor table empty — run supervisor_builder.build() "
                      "(via ingestion/build.py) to enable structured supervisor matching.")
            return df
        except Exception as e:
            print(f"[pipeline] Could not load supervisor table ({e}) — "
                  "SUPERVISOR queries will fall back to plain RAG.")
            import pandas as pd
            return pd.DataFrame(columns=["name", "designation", "subject"])

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

        # FIX 1: Notice/scheme keyword detection — must run BEFORE the generic
        # WHAT_IS / BROAD_PROGRAM patterns so "tell me about pm laptop notice"
        # doesn't accidentally fall into PROGRAMS routing.
        notice_match = NOTICE_KEYWORD_PATTERN.search(q)
        if notice_match:
            # Derive a source-boost hint from the matched keyword (first word,
            # lower-cased). This is a substring of the actual source stem —
            # e.g. "laptop" matches "PM_LAPTOP_NOTICE_MASTER_extracted".
            # No filename is hardcoded; any future document whose stem contains
            # this keyword will be boosted automatically.
            raw_match = notice_match.group(1)
            source_hint = raw_match.lower().split()[0]  # e.g. "laptop"
            print(f"[pipeline] Notice keyword '{raw_match}' → source_hint='{source_hint}'")
            return {
                "route":       "GENERAL",
                "intent":      "GENERAL",
                "filters":     {},
                "source_hint": source_hint,
                "reason":      f"pattern: notice-keyword ({raw_match})",
            }

        # Concrete keyword matchers (specific intents, no LLM needed)
        if DEADLINE_PATTERN.search(q):
            return {"route": "CSV_AND_RAG", "intent": "DEADLINES",
                    "filters": {}, "reason": "pattern: deadline"}
        # FIX (fees): FEE_PATTERN used to early-return here with a complete
        # plan dict ({"route": "RAG", "intent": "FEES", "filters": {}}),
        # which bypassed route_query() entirely — meaning Stage 2's filter
        # extractor (which now also infers source_hint, see router.py)
        # never ran. That's why fee queries with a perfectly good answer
        # in the corpus (e.g. "What are the fees for MS Data Science?")
        # came back with no source_hint to boost the right prospectus
        # document, and the LLM had nothing pointing it at the correct
        # source — it would frequently fall back to "no fee information
        # found" even when the evidence was retrievable.
        #
        # Fix: deliberately fall through to the LLM router, exactly like
        # SUPERVISOR_PATTERN already does just below. The intent/route
        # mapping is unaffected — Stage 1 will still classify this as
        # FEES, and _INTENT_TO_ROUTE["FEES"] = "CSV_AND_RAG" in router.py
        # already matches what this pattern used to hardcode (modulo the
        # CSV_AND_RAG vs RAG mismatch that this same fix also resolves —
        # see Issue 3 in the spec).
        if FEE_PATTERN.search(q):
            # Deliberately falls through to LLM router so Stage 2 filter
            # extraction (department/degree_level/source_hint) also runs.
            pass
        if ELIGIBILITY_PATTERN.search(q):
            # FIX (eligibility consistency): deliberately falls through to
            # LLM router so Stage 2 filter extraction (department/
            # degree_level/source_hint) also runs — mirrors FEE_PATTERN
            # above and SUPERVISOR_PATTERN below. This is what actually
            # lets the ELIGIBILITY query-augmentation block further down
            # in this file fire, since it depends on a non-empty
            # `filters` dict that only Stage 2 ever populates.
            pass
        if SUPERVISOR_PATTERN.search(q):
            # Deliberately falls through to LLM router so Stage 2
            # research_area extraction also runs — see router.py.
            pass
        if HISTORY_PATTERN.search(q):
            return {"route": "RAG", "intent": "HISTORY",
                    "filters": {}, "reason": "pattern: history"}
        if CONTACT_PATTERN.search(q):
            return {"route": "RAG", "intent": "CONTACT",
                    "filters": {}, "reason": "pattern: contact"}
        if SHUTTLE_PATTERN.search(q):
            # FIX 3: extended pattern now catches "point" synonyms
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
        # query actually mentions a program/academic keyword.
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

        intent     = plan.get("intent", "GENERAL")
        route      = plan.get("route", "RAG")
        query_mode = plan.get("query_mode", "SEARCH")
        filters    = plan.get("filters", {})
        # FIX 1: source_hint may be set by _pre_route (notice keywords) or
        # FIX (fees): now also set by router.py's Stage 2 filter extractor
        # for FEES queries (LLM-inferred from context, e.g. "prospectus").
        # default to None if neither path set it.
        source_hint: str | None = plan.get("source_hint")
        # FIX (fees): filters may also carry source_hint nested by the
        # router's filter extractor; pull it out as a fallback so callers
        # that only look at plan["source_hint"] vs plan["filters"]["source_hint"]
        # both work, without requiring router.py and pipeline.py to agree
        # on exactly one nesting shape.
        if source_hint is None and isinstance(filters, dict):
            source_hint = filters.get("source_hint")
        print(f"[pipeline] Route={route}, Intent={intent}, Mode={query_mode}"
              + (f", source_hint={source_hint}" if source_hint else ""))

        # ── LIST mode: skip semantic search, return full dataset ────────
        if query_mode == "LIST":
            csv_rows = []
            evidence_parts = []
            if intent == "SUPERVISOR" and len(self.supervisor_df) > 0:
                csv_rows = _tag_source(
                    self.supervisor_df.to_dict(orient="records"),
                    _SUPERVISOR_SOURCE_FILE,
                )
                print(f"[pipeline] LIST SUPERVISOR: {len(csv_rows)} rows")
            elif intent == "SHUTTLE" and len(self.shuttle_df) > 0:
                csv_rows = _tag_source(
                    self.shuttle_df.to_dict(orient="records"),
                    _SHUTTLE_SOURCE_FILE,
                )
                print(f"[pipeline] LIST SHUTTLE: {len(csv_rows)} rows")
            elif intent == "PROGRAMS" and len(self.qe.df) > 0:
                filtered = self.qe.query_programs(filters)
                if len(filtered) > 0:
                    csv_rows = _tag_source(
                        filtered.to_dict(orient="records"),
                        _PROGRAMS_SOURCE_FILE,
                    )
                else:
                    csv_rows = _tag_source(
                        self.qe.df.to_dict(orient="records"),
                        _PROGRAMS_SOURCE_FILE,
                    )
                print(f"[pipeline] LIST PROGRAMS: {len(csv_rows)} rows")
            elif intent in ("FACILITIES", "HOSTEL", "CONTACT", "HISTORY", "DEADLINES", "DOCUMENTS"):
                pref = self._intent_to_source_filter(intent)
                if pref and self.hybrid:
                    evidence_parts = self.hybrid.search(
                        user_query, top_k=50,
                        source_filter=pref,
                    )
                    print(f"[pipeline] LIST {intent}: {len(evidence_parts)} chunks from {pref}")
            if csv_rows or evidence_parts:
                response = generate_answer(
                    query=user_query,
                    llm=self.llm,
                    evidence_parts=evidence_parts,
                    intent=intent,
                    csv_rows=csv_rows if csv_rows else None,
                    context_history=context,
                )
                self._save_messages(user_query, response, intent)
                return response

        # FIX 2: For ELIGIBILITY, build an augmented RAG search query that
        # includes the extracted degree_level and/or department alongside the
        # original user query and standard eligibility terms.  This improves
        # BM25 recall for program-specific eligibility sections in large PDFs
        # (e.g. "MS Data Science" may appear near eligibility criteria in the
        # prospectus but not surface for the original query alone).
        rag_query = user_query
        if intent == "ELIGIBILITY" and filters:
            filter_parts = [
                str(v) for k, v in filters.items()
                if k in ("degree_level", "department") and v
            ]
            if filter_parts:
                rag_query = (
                    f"{' '.join(filter_parts)} {user_query} "
                    "eligibility admission criteria requirements"
                )
                print(f"[pipeline] ELIGIBILITY: augmented RAG query for chunk recall")

        # FIX (fees): For FEES, augment the RAG query the same way as
        # ELIGIBILITY — department/degree_level context measurably helps
        # BM25 recall in large prospectus PDFs where the fee table sits
        # under a specific program heading rather than near generic "fee"
        # language.
        if intent == "FEES" and filters:
            filter_parts = [
                str(v) for k, v in filters.items()
                if k in ("degree_level", "department") and v
            ]
            if filter_parts:
                rag_query = (
                    f"{' '.join(filter_parts)} {user_query} "
                    "fee tuition self-finance regular"
                )
                print(f"[pipeline] FEES: augmented RAG query for chunk recall")

        # Step 2: Execute
        csv_rows: list[dict] = []
        evidence_parts: list[dict] = []
        rag_fallback = False

        if route == "COUNT":
            count_data = self.qe.count_programs(filters)
            if count_data["total"] > 0:
                csv_rows = _tag_source([count_data], _PROGRAMS_SOURCE_FILE)
            else:
                print("[pipeline] Count: 0 — falling back to RAG")
                evidence_parts = self._search(rag_query, intent, source_hint=source_hint, query_mode=query_mode)
                intent = "RAG"
                rag_fallback = True

        elif route in ("CSV_QUERY", "CSV_AND_RAG") and len(self.qe.df) > 0:
            result_df = self.qe.query_programs(filters)
            if len(result_df) > 0:
                csv_rows = _tag_source(
                    result_df.head(10).to_dict(orient="records"),
                    _PROGRAMS_SOURCE_FILE,
                )
            if route == "CSV_AND_RAG" or len(result_df) == 0:
                # FIX 2 / FIX (fees): pass augmented rag_query instead of
                # raw user_query, and propagate source_hint (FEES now gets
                # one from router.py's Stage 2 extractor; ELIGIBILITY and
                # others still default to None as before).
                evidence_parts = self._search(rag_query, intent, source_hint=source_hint, query_mode=query_mode)

        elif route == "RAG" and intent == "SHUTTLE":
            if len(self.shuttle_df) > 0:
                matches = find_routes_by_location(user_query, self.shuttle_df)
                if matches:
                    csv_rows = _tag_source(matches, _SHUTTLE_SOURCE_FILE)
                else:
                    csv_rows = _tag_source(
                        self.shuttle_df.to_dict(orient="records"),
                        _SHUTTLE_SOURCE_FILE,
                    )
            else:
                print("[pipeline] Shuttle table unavailable — falling back to RAG search")
                evidence_parts = self._search(user_query, intent, query_mode=query_mode)

        elif route == "RAG" and intent == "SUPERVISOR":
            if len(self.supervisor_df) > 0:
                research_area = filters.get("research_area")
                csv_rows = find_supervisors_by_area(
                    user_query,
                    self.supervisor_df,
                    research_area=research_area,
                )
                csv_rows = _tag_source(csv_rows, _SUPERVISOR_SOURCE_FILE)
            else:
                print("[pipeline] Supervisor table unavailable — falling back to RAG search")
                evidence_parts = self._search(user_query, intent, query_mode=query_mode)

        elif route == "RAG":
            # FIX (fees): this branch now also covers FEES queries that
            # don't go through CSV_AND_RAG (e.g. if Stage 1 ever maps FEES
            # to plain RAG in the future) — source_hint flows through
            # either way since it's read from `plan`/`filters` above.
            evidence_parts = self._search(rag_query, intent, source_hint=source_hint, query_mode=query_mode)

        elif route == "GENERAL":
            # FIX 1: pass source_hint so the correct notice document is boosted
            evidence_parts = self._search(rag_query, intent, source_hint=source_hint, query_mode=query_mode)

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

    def _search(
        self,
        query: str,
        intent: str,
        source_hint: str | None = None,
        query_mode: str = "SEARCH",
    ) -> list[dict]:
        """Hybrid search with multi-source diversity + score threshold.

        FIX 1 — source_hint:
          When set (e.g. "laptop" for PM Laptop Notice queries, or now
          also an LLM-inferred hint like "prospectus" for FEES queries —
          see router.py), calls hybrid_searcher.search() directly with
          source_boost=source_hint so chunks from the matching source are
          score-multiplied before reranking. Falls back to broad search
          automatically if the boosted search yields insufficient results
          (the boost is soft, not a hard filter).

        FIX 2 — expand_pool:
          For ELIGIBILITY intent, passes expand_pool=True to
          hybrid_searcher.search() so the cross-encoder sees a 2× larger
          candidate pool. This prevents the correct program-eligibility chunk
          from being dropped before the expensive reranker runs.

        FIX (LIST mode):
          When query_mode is LIST, uses a lower score threshold so that
          enumeration queries are not filtered out by strict scoring.

        NOTE: SHUTTLE and SUPERVISOR no longer reach this method in the
        normal case — both route through their deterministic matchers.
        """
        preferred_source = self._intent_to_source_filter(intent)

        # FIX 2: ELIGIBILITY uses an expanded rerank pool so the correct
        # program-eligibility chunk isn't dropped before the cross-encoder runs.
        expand_pool = (intent == "ELIGIBILITY")

        if preferred_source is not None:
            # For single-source intents: search with filter first
            if self.hybrid is not None:
                results = self.hybrid.search(
                    query, top_k=8,
                    source_filter=preferred_source,
                    expand_pool=expand_pool,
                )
            else:
                results = self.qe.search(query, top_k=8, source_filter=preferred_source)
            good = [r for r in results if r["score"] >= MIN_RAG_SCORE]
            if good:
                return good[:4]
            # Fallback: search without filter
            print(f"[pipeline] Low scores from {preferred_source} — expanding")
            if self.hybrid is not None:
                results = self.hybrid.search(query, top_k=8, expand_pool=expand_pool)
            else:
                results = self.qe.search(query, top_k=8)
        else:
            # For broad intents: search without source filter, aim for source diversity.
            # FIX 1: When a source_hint is available, apply a soft score boost so
            # the correct document rises above generic-topic results.
            if self.hybrid is not None:
                results = self.hybrid.search(
                    query, top_k=10,
                    source_boost=source_hint,   # FIX 1: None is a no-op
                    expand_pool=expand_pool,    # FIX 2
                )
                if source_hint:
                    print(f"[pipeline] Applied source_boost='{source_hint}' "
                          f"({len(results)} candidates)")
            else:
                # Degraded path: qe.search doesn't support source_boost/expand_pool,
                # so fall back to source_filter as a harder-but-functional alternative.
                if source_hint:
                    results = self.qe.search(query, top_k=8, source_filter=source_hint)
                    good = [r for r in results if r["score"] >= MIN_RAG_SCORE * 0.7]
                    if good:
                        print(f"[pipeline] source_hint filter '{source_hint}' → {len(good)} results")
                        return good[:4]
                    print(f"[pipeline] source_hint '{source_hint}' filter missed — expanding")
                results = self.qe.search(query, top_k=10)

        # Filter by score — lower threshold for LIST mode so enumeration
        # queries with broadly relevant chunks are not silently dropped.
        threshold = MIN_RAG_SCORE
        if query_mode == "LIST":
            threshold = MIN_RAG_SCORE * 0.5
            print(f"[pipeline] LIST mode: using reduced score threshold {threshold:.2f}")

        good = [r for r in results if r["score"] >= threshold]

        # If we have multiple results, enforce source diversity
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
            scores_str = ", ".join(f"{r['score']:.3f}" for r in results[:5])
            print(f"[pipeline] All {len(results)} results below threshold {threshold:.2f} — "
                  f"scores: [{scores_str}]")
            return results[:2]

        print(f"[pipeline] No results found for query (mode={query_mode})")
        return []

    @staticmethod
    def _intent_to_source_filter(intent: str) -> str | None:
        """Return a preferred source file for truly single-source intents."""
        mapping = {
            "FACILITIES": "facilities_info",
            "HOSTEL":     "hostel_facilities",
            "SUPERVISOR": "phd_supervisors",
            "HISTORY":    "university_history",
            "CONTACT":    "ned_links",
            "SHUTTLE":    "shuttle_route",
        }
        return mapping.get(intent)


def get_pipeline() -> AdmissionPipeline:
    """Get the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AdmissionPipeline()
    return _pipeline_instance
