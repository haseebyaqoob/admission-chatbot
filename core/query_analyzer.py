"""
core/query_analyzer.py — LLM-based query analysis.

This module REPLACES all pattern-based routing.

Key difference from friend's architecture:
  - No regex patterns
  - No intent→source_file mapping
  - LLM understands the query and returns retrieval HINTS
  - Hints (key_terms, needs_table_data) improve retrieval quality
    WITHOUT routing to specific documents

Output fields:
  rewritten_query   — expanded, clean query for FAISS embedding
  key_terms         — specific nouns/values to boost in BM25
  needs_table_data  — should table chunks be preferred?
  is_comparison     — comparing two or more items?
  is_numerical      — expects a specific number/amount?
  is_follow_up      — references prior conversation?
"""

import json
from dataclasses import dataclass
from typing import List

from core.llm_handler import LLMHandler, extract_json
from config_loader import cfg


@dataclass
class QueryAnalysis:
    original_query: str
    rewritten_query: str
    key_terms: List[str]
    needs_table_data: bool
    is_comparison: bool
    is_numerical: bool
    is_follow_up: bool

    def get_search_query(self) -> str:
        """Best query string for retrieval."""
        return self.rewritten_query.strip() or self.original_query

    def get_final_k(self) -> int:
        """How many final chunks to retrieve (more for comparisons)."""
        base = cfg.get("final_top_k", 5)
        if self.is_comparison:
            return cfg.get("comparison_top_k", 8)
        return base


# ─── Prompt ───────────────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
You are a query analysis module for a university admissions chatbot.

Analyze the user query and return a JSON object with EXACTLY these fields:

{{
  "rewritten_query": "expanded, specific version of the query",
  "key_terms": ["list", "of", "important", "specific", "terms"],
  "needs_table_data": true or false,
  "is_comparison": true or false,
  "is_numerical": true or false,
  "is_follow_up": true or false
}}

Rules for each field:
- rewritten_query: expand abbreviations (BSCS → BS Computer Science, AI → Artificial Intelligence),
  if this is a follow-up, incorporate context from the conversation to make it self-contained.
- key_terms: specific nouns, program names, department names, amounts, academic years.
  Do NOT include generic words (what, the, is, are, how, tell, me).
- needs_table_data: true if the question asks for fees, amounts, seat counts, credit hours,
  scholarship amounts, schedules, deadlines with dates, or any structured numbers.
- is_comparison: true if the query compares two or more programs, fees, or departments.
- is_numerical: true if the expected answer is a specific number, fee, count, or amount.
- is_follow_up: true if the query uses pronouns like "it", "that", "those", "also",
  or clearly refers to the previous conversation.

Conversation context (may be empty):
{context}

User query: {query}

Return only the JSON object. No explanation."""


class QueryAnalyzer:

    def __init__(self, llm: LLMHandler):
        self.llm = llm

    def analyze(self, query: str, context: str = "") -> QueryAnalysis:
        """
        Analyze a query and return structured retrieval hints.
        Gracefully falls back on any error.
        """
        prompt = _ANALYSIS_PROMPT.format(
            query=query,
            context=context.strip() or "None",
        )

        raw = self.llm.generate(
            prompt      = prompt,
            max_tokens  = cfg.get("analysis_max_tokens", 256),
            temperature = cfg.get("analysis_temperature", 0.0),
            json_mode   = True,
        )

        data = extract_json(raw)

        if data:
            return QueryAnalysis(
                original_query  = query,
                rewritten_query = str(data.get("rewritten_query", query)),
                key_terms       = [str(t) for t in data.get("key_terms", [])],
                needs_table_data= bool(data.get("needs_table_data", False)),
                is_comparison   = bool(data.get("is_comparison", False)),
                is_numerical    = bool(data.get("is_numerical", False)),
                is_follow_up    = bool(data.get("is_follow_up", False)),
            )

        # Fallback: never crash on analysis failure
        return QueryAnalysis(
            original_query  = query,
            rewritten_query = query,
            key_terms       = [w for w in query.split() if len(w) > 3][:5],
            needs_table_data= any(kw in query.lower() for kw in
                                  ["fee", "seat", "cost", "amount", "how much", "how many",
                                   "scholarship", "deadline", "schedule"]),
            is_comparison   = any(kw in query.lower() for kw in
                                  ["compare", "difference", "versus", "vs", "between"]),
            is_numerical    = any(kw in query.lower() for kw in
                                  ["how much", "how many", "total", "amount", "cost", "fee"]),
            is_follow_up    = any(kw in query.lower() for kw in
                                  ["it", "that", "those", "also", "same", "as well"]),
        )
