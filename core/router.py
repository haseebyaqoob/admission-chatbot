"""
router.py
─────────
2-stage router for the admission bot.

Stage 1 — Intent classifier:
  Classifies user query into one of the admission-related intents.

Stage 2 — Schema extraction (conditional):
  PROGRAMS / ELIGIBILITY / FEES / DEADLINES → extract filters (department, level)
  All other intents skip Stage 2.
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

# ── Admission intents ──────────────────────────────────────────────────
# NOTE: SHUTTLE and GENERAL were missing previously. Without SHUTTLE, the
# classifier had no correct bucket for transport queries and was forced to
# guess (usually landing on OFF_TOPIC or a poorly-fitting RAG intent with
# no source filter, which produced ungrounded/hallucinated answers).
# GENERAL is a true catch-all for topics not yet enumerated below — it
# triggers unrestricted RAG without pretending to be one of the specific
# intents, so the model isn't forced into a wrong skill template.
VALID_INTENTS = {
    "PROGRAMS", "ELIGIBILITY", "FEES", "DEADLINES",
    "DOCUMENTS", "FACILITIES", "HOSTEL", "SUPERVISOR",
    "HISTORY", "CONTACT", "SHUTTLE", "GENERAL",
    "GREETING", "FAREWELL", "OFF_TOPIC",
}

_INTENT_TO_ROUTE: dict[str, str] = {
    "PROGRAMS":    "CSV_AND_RAG",  # CSV for structured data + RAG for brochure details
    "ELIGIBILITY": "CSV_AND_RAG",
    "FEES":        "CSV_AND_RAG",  # CSV for fee data + RAG for details
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


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 — Intent Classifier
# ═══════════════════════════════════════════════════════════════════════

def _build_classifier_prompt() -> str:
    """Build the intent classifier system prompt."""
    today = date.today().strftime("%d %B %Y")
    return f"""You are an intent classifier for a NED University admissions chatbot. Today: {today}

Classify the user query into EXACTLY ONE intent:

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
             "find PhD supervisors in AI", "supervisors in computer science"

HISTORY    — asking about university history, establishment, background:
             "when was NED established", "history of NED"

CONTACT    — asking for contact info, phone, address, office:
             "admission office phone number", "where is the admission office"

SHUTTLE    — asking about shuttle service, bus routes, transport, commuting to/from campus:
             "shuttle route from defence", "is there a bus from nazimabad",
             "which shuttle covers korangi", "how do I commute to NED",
             "shuttle timings", "transport facility"

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

Decision rules:
1. "how many", "count", "total number of" → PROGRAMS
2. "tell me about X" → classify by what X actually is, not always PROGRAMS.
   If X is a program/field/department → PROGRAMS. If X is shuttle/transport →
   SHUTTLE. If X doesn't match any specific category → GENERAL.
3. "how do I", "how to" → DOCUMENTS (unless clearly about shuttle/hostel/etc.)
4. When in doubt between two specific intents → pick whichever fits best.
   When nothing fits at all → GENERAL, not OFF_TOPIC and not a forced guess.
5. If the query is in a mix of Urdu and English → classify by the English keywords

Output ONLY this JSON object, nothing else:
{{
  "intent": "PROGRAMS|ELIGIBILITY|FEES|DEADLINES|DOCUMENTS|FACILITIES|HOSTEL|SUPERVISOR|HISTORY|CONTACT|SHUTTLE|GENERAL|GREETING|FAREWELL|OFF_TOPIC",
  "reason": "one sentence explanation"
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — Filter Extractor (PROGRAMS / ELIGIBILITY / FEES / DEADLINES)
# ═══════════════════════════════════════════════════════════════════════

def _build_filter_extractor_prompt(df: pd.DataFrame) -> str:
    """Build the filter extraction prompt for structured queries."""
    dept_list = sorted(df["department"].unique().tolist()) if len(df) > 0 else []
    dept_catalog = "\n".join(f"  - {d}" for d in dept_list)

    return f"""Extract structured filters from the user query about NED admissions.

Valid filter keys: department, degree_level

Available departments in the catalog:
{dept_catalog}

Examples:
  "MS programs in Computer Science"  → {{"department": "Computer Science & IT", "degree_level": "MS"}}
  "BE fees"                           → {{"department": null, "degree_level": "BE"}}
  "eligibility for MS Data Science"   → {{"department": null, "degree_level": "MS"}}
  "what programs does Civil offer"    → {{"department": "Civil Engineering", "degree_level": null}}
  "deadlines for undergraduate"       → {{"department": null, "degree_level": "BE"}}

Rules:
- department: fuzzy-match against the catalog. Use the EXACT catalog name.
  Look for department name, field name, or keyword in the query.
- degree_level: BE, BS, ME, M.Engg, MS, PhD, or null
- If nothing can be extracted, output null values

Output ONLY this JSON, nothing else:
{{
  "department": null,
  "degree_level": null
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════

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


def _run_filter_extractor(
    user_query: str,
    llm: LocalLLM,
    df: pd.DataFrame,
    retry_once: bool = True,
) -> dict:
    """Stage 2: extract structured filters for CSV queries."""
    system_prompt = _build_filter_extractor_prompt(df)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Extract filters for:\n{user_query}"},
    ]

    raw    = llm.chat(messages, max_new_tokens=200, temperature=0.0, use_json_format=True)
    result = extract_json(raw)
    if isinstance(result, dict):
        return result

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
            return result2

    print("[router] Filter extractor failed both attempts")
    return {"department": None, "degree_level": None, "gpa": None}


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

    Stage 1 — intent classification (always runs):
      Returns intent + reason.

    Stage 2 — extraction (conditional):
      PROGRAMS/ELIGIBILITY/FEES/DEADLINES → extract structured filters
      All others → skip Stage 2
    """
    # ── Deterministic pre-router for common greetings/farewells ───────
    q_lower = user_query.strip().lower()
    if q_lower in ("hi", "hello", "hey", "assalam o alaikum", "good morning", "good afternoon", "good evening"):
        return {"route": "GREETING", "intent": "GREETING", "filters": {}, "reason": "deterministic greeting"}
    if q_lower in ("bye", "goodbye", "see you", "take care", "thanks", "thank you"):
        return {"route": "FAREWELL", "intent": "FAREWELL", "filters": {}, "reason": "deterministic farewell"}

    # ── Stage 1: classify intent ─────────────────────────────────────
    stage1 = _run_classifier(user_query, llm, retry_once)

    if stage1 is None:
        # NOTE: previously fell back to a non-existent "RAG" intent (not
        # in VALID_INTENTS, not in SKILL_MAP — silently handled by skills.py's
        # default-to-OFF_TOPIC behavior in get_skill(), which is misleading).
        # Falls back to GENERAL now: a real intent with unrestricted RAG
        # and an honest "answer from evidence or say not covered" skill.
        print("[router] Stage 1 failed — falling back to GENERAL")
        return {"route": "GENERAL", "intent": "GENERAL", "filters": {}, "reason": "classifier fallback"}

    intent    = stage1.get("intent", "GENERAL")
    reason    = stage1.get("reason", "")
    route     = _INTENT_TO_ROUTE.get(intent, "GENERAL")

    # ── Stage 2: extract filters (conditional) ───────────────────────
    filters: dict = {}

    if route in ("CSV_QUERY", "CSV_AND_RAG") and df is not None and len(df) > 0:
        filters = _run_filter_extractor(user_query, llm, df, retry_once)

    return {
        "route":          route,
        "intent":         intent,
        "filters":        filters,
        "reason":         reason,
    }