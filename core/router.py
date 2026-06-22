"""
router.py
─────────
2-stage router for the admission bot.

Stage 1 — Intent + mode classifier:
  Classifies user query into one of the admission-related intents AND
  determines whether the user is asking for a LIST (enumeration) or a
  SEARCH (semantic lookup).

Stage 2 — Schema extraction (conditional):
  PROGRAMS / ELIGIBILITY / FEES / DEADLINES → extract filters (department, level,
                                               and — FIX (fees) — source_hint)
  SUPERVISOR                                → extract a canonical research_area
  All other intents skip Stage 2.

FIX (LIST mode) — this version:
  Added query_mode field ("SEARCH" | "LIST") to classifier output. When
  the user asks for enumeration ("list all", "show all", "give me all",
  "what are all"), the router sets query_mode=LIST. The pipeline then
  skips semantic search and returns the full dataset for the detected
  intent source, ensuring enumeration queries never return empty results
  due to strict scoring thresholds.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd

from config_loader import cfg
from core.llm_handler import LocalLLM
from core.utils import extract_json

_FUZZY_THRESHOLD = int(cfg["fuzzy_match_threshold"])
_ROUTER_TOKENS   = int(cfg["router_max_tokens"])

VALID_INTENTS = {
    "PROGRAMS", "ELIGIBILITY", "FEES", "DEADLINES",
    "DOCUMENTS", "FACILITIES", "HOSTEL", "SUPERVISOR",
    "HISTORY", "CONTACT", "SHUTTLE", "GENERAL",
    "GREETING", "FAREWELL", "OFF_TOPIC",
}

_INTENT_TO_ROUTE: dict[str, str] = {
    "PROGRAMS":    "CSV_AND_RAG",
    "ELIGIBILITY": "CSV_AND_RAG",
    "FEES":        "CSV_AND_RAG",
    "DEADLINES":   "CSV_AND_RAG",
    "DOCUMENTS":   "RAG",
    "FACILITIES":  "RAG",
    "HOSTEL":      "RAG",
    "SUPERVISOR":  "RAG",
    "HISTORY":     "RAG",
    "CONTACT":     "RAG",
    "SHUTTLE":     "RAG",
    "GENERAL":     "GENERAL",
    "GREETING":    "GREETING",
    "FAREWELL":    "FAREWELL",
    "OFF_TOPIC":   "OFF_TOPIC",
}

# FIX (fees) / FIX (gpa cleanup): the only keys _run_filter_extractor is
# allowed to hand back to callers. Anything else the LLM puts in its JSON
# response — including legacy keys like "gpa" that used to appear in the
# hardcoded failure fallback — is stripped before the dict leaves this
# module. Centralizing the allowlist here (rather than just fixing the
# one fallback dict) means a stray key can't leak through the *success*
# path either, not just the failure path.
_FILTER_KEYS = {"department", "degree_level", "source_hint"}


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 — Intent Classifier
# ═══════════════════════════════════════════════════════════════════════

def _build_classifier_prompt() -> str:
    """Build the intent classifier system prompt."""
    today = date.today().strftime("%d %B %Y")
    return f"""You are an intent classifier for a NED University admissions chatbot. Today: {today}

Classify the user query into EXACTLY ONE intent AND EXACTLY ONE query_mode.

Intents:

PROGRAMS   — asking about departments, programs, degrees offered:
             "what programs does CS offer", "list BE degrees",
             "departments in NED", "how many degrees are there",
             "how many undergraduate programs", "tell me about textile",
             "tell me about the CS degree", "total number of departments",
             "count all programs"

ELIGIBILITY — asking about eligibility criteria, requirements, qualifications:
             "am I eligible for MS Data Science", "what is the eligibility for BE",
             "requirements for admission"

FEES       — asking about fees, tuition, self-finance, costs:
             "how much is the fee", "what is the self-finance fee",
             "tuition fee for BE"

DEADLINES  — asking about dates, deadlines, schedule, last date:
             "when is the last date", "admission schedule 2026",
             "application deadline"

DOCUMENTS  — asking about required documents, application process:
             "what documents do I need", "how to apply",
             "what to bring for admission", "how do I apply",
             "how to get admission", "steps to apply",
             "what do I need for admission"

FACILITIES — asking about university facilities, library, labs, sports:
             "does NED have a library", "what facilities are available"

HOSTEL     — asking about hostel, accommodation, boarding:
             "how to apply for hostel", "hostel fees", "room types"

SUPERVISOR — asking about PhD supervisors, research areas:
             "find PhD supervisors in AI", "supervisors in computer science",
             "list all supervisors", "show all supervisors",
             "give me all supervisors"

HISTORY    — asking about university history, establishment, background:
             "when was NED established", "history of NED"

CONTACT    — asking for contact info, phone, address, office:
             "admission office phone number", "where is the admission office"

SHUTTLE    — asking about shuttle service, bus routes, transport, commuting to/from campus.

             IMPORTANT: In Pakistan English, a bus/shuttle stop is commonly
             called a "point" (e.g. "Nazimabad point", "Defence point").
             Queries using "point", "stop", "boarding point", "alighting point",
             "get on/off", "passes through", or "goes through" in the context
             of commuting or routes are SHUTTLE, not SUPERVISOR or anything else.

             Examples:
             "shuttle route from defence"
             "is there a bus from nazimabad"
             "which shuttle covers korangi"
             "how do I commute to NED"
             "shuttle timings"
             "transport facility"
             "which point goes through Nazimabad?"       ← SHUTTLE (point = bus stop)
             "what point do I board for Gulshan?"        ← SHUTTLE
             "which bus stop covers Clifton?"            ← SHUTTLE
             "does any route pass through North Karachi?" ← SHUTTLE
             "where do I get off for NED from Saddar?"  ← SHUTTLE
             "boarding point for Defence route"          ← SHUTTLE
             "is there a pickup point near Orangi?"     ← SHUTTLE
             "which route goes through PECHS?"          ← SHUTTLE
             "list all shuttle routes"                   ← SHUTTLE + LIST

GENERAL    — admissions-adjacent but doesn't fit any category above, AND is not
             clearly off-topic. Use this rather than forcing a poor fit into
             another intent. The system will run an unrestricted search and
             answer only from whatever evidence is found, or say it isn't
             covered.

GREETING   — simple standalone greetings only:
             "hi", "hello", "hey", "assalam o alaikum", "good morning"
             (greeting + question → route to the question's intent)

FAREWELL   — goodbye expressions:
             "bye", "thanks", "thank you", "take care"

OFF_TOPIC  — clearly unrelated to NED admissions at all:
             "how are you", "who are you", "what can you do",
             "who made you", "what is your name",
             "tell me a joke", "what is the weather today"

Query modes:

SEARCH — user is looking for specific information about a particular item:
         "what is the fee for MS Data Science", "eligibility for BE",
         "which shuttle covers defence", "find supervisors in AI"
         Most queries are SEARCH.

LIST   — user is asking to enumerate ALL items in a category:
         "list all supervisors", "show all programs",
         "give me all shuttle routes", "what are all the departments",
         "list all faculty", "show me all BE programs",
         "list all PhD supervisors", "all programs offered"
         LIST queries ask for the full set, not specific information.

Decision rules:
1. "how many", "count", "total number of" → PROGRAMS + SEARCH
2. "list all", "show all", "give me all", "what are all" + category → [intent based on category] + LIST
3. "tell me about X" → classify by what X actually is, not always PROGRAMS.
   If X is a program/field/department → PROGRAMS. If X is shuttle/transport/
   bus/route/commute → SHUTTLE. If X doesn't match any specific category → GENERAL.
4. "how do I", "how to" → DOCUMENTS (unless clearly about shuttle/hostel/etc.)
5. "point" in a route/commute/travel context → SHUTTLE (it means a bus stop).
   "point" in a research/supervisor context → SUPERVISOR.
   When in doubt about "point", check whether other words suggest travel
   (goes through, route, bus, Karachi area names) vs. research (PhD, thesis).
6. When in doubt between two specific intents → pick whichever fits best.
   When nothing fits at all → GENERAL, not OFF_TOPIC and not a forced guess.
7. If the query is in a mix of Urdu and English → classify by the English keywords
8. LIST mode takes priority: if the user asks "list all <X>", set the matching
   intent and query_mode=LIST, even if the intent would normally be SEARCH.

Output ONLY this JSON object, nothing else:
{{
  "intent": "PROGRAMS|ELIGIBILITY|FEES|DEADLINES|DOCUMENTS|FACILITIES|HOSTEL|SUPERVISOR|HISTORY|CONTACT|SHUTTLE|GENERAL|GREETING|FAREWELL|OFF_TOPIC",
  "query_mode": "SEARCH|LIST",
  "reason": "one sentence explanation"
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — Filter Extractor (PROGRAMS / ELIGIBILITY / FEES / DEADLINES)
# ═══════════════════════════════════════════════════════════════════════

def _build_filter_extractor_prompt(df: pd.DataFrame) -> str:
    """Build the filter extraction prompt for structured queries.

    FIX (fees): now also asks for `source_hint` — a single lowercase word
    or short phrase guessing which KIND of document is likely to contain
    the answer, inferred purely from context (never from a fixed keyword
    → filename table). The hint is expected to be a SUBSTRING of the real
    source document's filename stem, the same contract pipeline.py's
    NOTICE_KEYWORD_PATTERN already uses for source_boost — so the LLM is
    reasoning about "what kind of document would this live in", not
    memorizing actual filenames it has never seen.
    """
    dept_list = sorted(df["department"].unique().tolist()) if len(df) > 0 else []
    dept_catalog = "\n".join(f"  - {d}" for d in dept_list)

    return f"""Extract structured filters from the user query about NED admissions.

Valid filter keys: department, degree_level, source_hint

Available departments in the catalog:
{dept_catalog}

Examples:
  "MS programs in Computer Science"
      → {{"department": "Computer Science & IT", "degree_level": "MS", "source_hint": null}}
  "BE fees"
      → {{"department": null, "degree_level": "BE", "source_hint": "prospectus"}}
  "eligibility for MS Data Science"
      → {{"department": "Computer Science & IT", "degree_level": "MS", "source_hint": "prospectus"}}
  "what is the eligibility for MS Data Science?"
      → {{"department": "Computer Science & IT", "degree_level": "MS", "source_hint": "prospectus"}}
  "what programs does Civil offer"
      → {{"department": "Civil Engineering", "degree_level": null, "source_hint": null}}
  "deadlines for undergraduate"
      → {{"department": null, "degree_level": "BE", "source_hint": "schedule"}}
  "what are the fees for MS Data Science?"
      → {{"department": "Computer Science & IT", "degree_level": "MS", "source_hint": "prospectus"}}
  "self finance fee for BE Civil"
      → {{"department": "Civil Engineering", "degree_level": "BE", "source_hint": "prospectus"}}

Rules:
- department: fuzzy-match against the catalog. Use the EXACT catalog name.
  Look for department name, field name, or keyword in the query.
  "Data Science" → Computer Science & IT (it is offered under CS dept).
- degree_level: BE, BS, ME, M.Engg, MS, PhD, or null
- source_hint: a single lowercase word (or short hyphenless phrase) guessing
  what KIND of document would contain the answer — reason about it the same
  way a human admissions-office staffer would ("fee/eligibility/program
  details for an academic degree are published in the admissions
  prospectus", "schedules and last dates are published in the admission
  schedule/timetable"). This is NOT a lookup against a fixed list of
  filenames — you have never seen the actual filenames in this corpus.
  Give your best single-word-or-short-phrase guess at the DOCUMENT TYPE,
  or null if you have no confident guess. Do not guess a department name,
  program name, or anything already captured by `department`/`degree_level`.
- If nothing can be extracted, output null values

Output ONLY this JSON, nothing else:
{{
  "department": null,
  "degree_level": null,
  "source_hint": null
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — Research-area Extractor (SUPERVISOR)
# ═══════════════════════════════════════════════════════════════════════

def _build_research_area_extractor_prompt() -> str:
    """Build the research-area extraction prompt for SUPERVISOR queries."""
    return """Extract the research area being asked about from a user query
about NED University PhD supervisors. Your ONLY job is to strip
conversational/verb phrasing and return the bare subject-matter noun
phrase, exactly as the user named it — do not rephrase it into a
synonym, do not expand it, do not guess a related field, do not correct
or normalize spelling/capitalization beyond simple cleanup.

Valid output key: research_area (string or null)

Examples:
  "Give me phd supervisors who manage engineering and technology"
      → {"research_area": "engineering and technology"}
  "find PhD supervisors in AI"
      → {"research_area": "AI"}
  "supervisors working in cyber security"
      → {"research_area": "cyber security"}
  "who specializes in computer science research"
      → {"research_area": "computer science"}
  "list all PhD supervisors"
      → {"research_area": null}
  "give me the full list of supervisors"
      → {"research_area": null}

Rules:
- If the query names ANY subject/field/topic, extract it as written
  (minus surrounding verbs/filler words like "who manage", "working
  in", "specializing in", "interested in", "research area of").
- If the query is a generic listing request with no subject named,
  output null.
- Never substitute a different word for what the user said (e.g. do
  NOT turn "AI" into "Artificial Intelligence" or "engineering and
  technology" into "Engineering & Technology" — leave casing/punctuation
  as the user wrote it; the matcher normalizes that on its own).

Output ONLY this JSON, nothing else:
{
  "research_area": null
}
"""


# ═══════════════════════════════════════════════════════════════════════
# Stage runners
# ═══════════════════════════════════════════════════════════════════════

def _run_classifier(
    user_query: str,
    llm: LocalLLM,
    retry_once: bool = True,
) -> dict | None:
    """Stage 1: classify intent. Returns dict with intent/reason or None."""
    system_prompt = _build_classifier_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"User query:\n{user_query}"},
    ]

    raw    = llm.chat(messages, max_new_tokens=_ROUTER_TOKENS, temperature=0.0, use_json_format=True)
    result = extract_json(raw)
    if result and result.get("intent") in VALID_INTENTS:
        return result

    if retry_once:
        repair = (
            "\n\nYour previous output was invalid. "
            "Return ONLY the JSON object with 'intent' and 'reason'. "
            "No explanation, no markdown, nothing else."
        )
        messages_r = [
            {"role": "system", "content": system_prompt + repair},
            {"role": "user",   "content": f"User query:\n{user_query}"},
        ]
        raw2    = llm.chat(messages_r, max_new_tokens=_ROUTER_TOKENS, temperature=0.0, use_json_format=True)
        result2 = extract_json(raw2)
        if result2 and result2.get("intent") in VALID_INTENTS:
            return result2

    return None


def _clean_filters(result: dict) -> dict:
    """FIX (fees) / FIX (gpa cleanup): allowlist the keys that are allowed
    to leave the filter extractor, regardless of which code path produced
    `result` (success, repair-retry, or failure fallback).

    This is the single place that decides what counts as a valid filter
    key, so:
      - the unused "gpa" key (previously hardcoded into the failure
        fallback dict) can never appear in a returned filters dict, and
      - any other unexpected key the LLM invents (typo, hallucinated
        field, future prompt-drift) is silently dropped rather than
        leaking into pipeline.py, which only ever reads the three keys
        it knows about anyway.
    Missing keys are filled in as None so callers can always rely on
    department/degree_level/source_hint being present (possibly None).
    """
    cleaned = {k: result.get(k) for k in _FILTER_KEYS}
    return cleaned


def _run_filter_extractor(
    user_query: str,
    llm: LocalLLM,
    df: pd.DataFrame,
    retry_once: bool = True,
) -> dict:
    """Stage 2: extract structured filters for CSV queries.

    FIX (fees): response schema now includes `source_hint` (see
    _build_filter_extractor_prompt). FIX (gpa cleanup): every return path
    — success, repair-retry success, and the final failure fallback — is
    passed through _clean_filters() so the unused "gpa" key can never
    appear, and no other stray key can leak through either.
    """
    system_prompt = _build_filter_extractor_prompt(df)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Extract filters for:\n{user_query}"},
    ]

    raw    = llm.chat(messages, max_new_tokens=200, temperature=0.0, use_json_format=True)
    result = extract_json(raw)
    if isinstance(result, dict):
        return _clean_filters(result)

    if retry_once:
        repair = (
            "\n\nReturn ONLY the JSON filter object. "
            "No explanation, no markdown fences."
        )
        messages_r = [
            {"role": "system", "content": system_prompt + repair},
            {"role": "user",   "content": f"Extract filters for:\n{user_query}"},
        ]
        raw2    = llm.chat(messages_r, max_new_tokens=200, temperature=0.0, use_json_format=True)
        result2 = extract_json(raw2)
        if isinstance(result2, dict):
            return _clean_filters(result2)

    print("[router] Filter extractor failed both attempts")
    # FIX (gpa cleanup): no more hardcoded "gpa" key — fall back through
    # the same allowlist so this path is guaranteed consistent with the
    # success paths above (department/degree_level/source_hint, all None).
    return _clean_filters({})


def _run_research_area_extractor(
    user_query: str,
    llm: LocalLLM,
    retry_once: bool = True,
) -> str | None:
    """Stage 2 (SUPERVISOR): extract a canonical research_area phrase."""
    system_prompt = _build_research_area_extractor_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Extract the research area for:\n{user_query}"},
    ]

    raw    = llm.chat(messages, max_new_tokens=100, temperature=0.0, use_json_format=True)
    result = extract_json(raw)

    if not isinstance(result, dict) and retry_once:
        repair = (
            "\n\nReturn ONLY the JSON object with key 'research_area'. "
            "No explanation, no markdown fences."
        )
        messages_r = [
            {"role": "system", "content": system_prompt + repair},
            {"role": "user",   "content": f"Extract the research area for:\n{user_query}"},
        ]
        raw2   = llm.chat(messages_r, max_new_tokens=100, temperature=0.0, use_json_format=True)
        result = extract_json(raw2)

    if isinstance(result, dict):
        area = result.get("research_area")
        if isinstance(area, str) and area.strip():
            return area.strip()

    print("[router] Research-area extractor returned no usable area — caller should use raw query")
    return None


# ═══════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════

def route_query(
    user_query: str,
    llm: LocalLLM,
    df: pd.DataFrame | None = None,
    retry_once: bool = True,
) -> dict[str, Any]:
    """
    2-stage routing pipeline for the admission bot.

    Stage 1 — intent + query_mode classification (always runs).
    Stage 2 — extraction (conditional):
      PROGRAMS/ELIGIBILITY/FEES/DEADLINES → extract structured filters
                                             (department, degree_level,
                                             and — FIX (fees) — source_hint)
      SUPERVISOR                          → extract research_area into filters
      All others                          → skip Stage 2

    Returns
    -------
    dict with keys: route, intent, query_mode, filters, reason
    query_mode is "SEARCH" (default) or "LIST" (enumeration request).
    """
    # ── Deterministic pre-router for common greetings/farewells ───────
    q_lower = user_query.strip().lower()
    if q_lower in ("hi", "hello", "hey", "assalam o alaikum", "good morning", "good afternoon", "good evening"):
        return {"route": "GREETING", "intent": "GREETING", "query_mode": "SEARCH", "filters": {}, "reason": "deterministic greeting"}
    if q_lower in ("bye", "goodbye", "see you", "take care", "thanks", "thank you"):
        return {"route": "FAREWELL", "intent": "FAREWELL", "query_mode": "SEARCH", "filters": {}, "reason": "deterministic farewell"}

    # ── Stage 1: classify intent ─────────────────────────────────────
    stage1 = _run_classifier(user_query, llm, retry_once)

    if stage1 is None:
        print("[router] Stage 1 failed — falling back to GENERAL")
        return {"route": "GENERAL", "intent": "GENERAL", "query_mode": "SEARCH", "filters": {}, "reason": "classifier fallback"}

    intent     = stage1.get("intent", "GENERAL")
    query_mode = stage1.get("query_mode", "SEARCH")
    reason     = stage1.get("reason", "")
    route      = _INTENT_TO_ROUTE.get(intent, "GENERAL")

    # ── Stage 2: extract filters (conditional) ───────────────────────
    filters: dict = {}

    if route in ("CSV_QUERY", "CSV_AND_RAG") and df is not None and len(df) > 0:
        # FIX (fees): this now also covers FEES queries, since
        # pipeline.py no longer early-returns before reaching here.
        # filters will include department/degree_level/source_hint.
        filters = _run_filter_extractor(user_query, llm, df, retry_once)

    elif intent == "SUPERVISOR":
        research_area = _run_research_area_extractor(user_query, llm, retry_once)
        filters = {"research_area": research_area}

    return {
        "route":      route,
        "intent":     intent,
        "query_mode": query_mode,
        "filters":    filters,
        "reason":     reason,
    }
