"""
shuttle_matcher.py
───────────────────
Deterministic location → route matching for shuttle queries.

This is intentionally NOT semantic/RAG search. Matching a user-mentioned
area against a known, finite list of stop names is a string-matching
problem, not a question-answering problem — embeddings/BM25 ranking are
the wrong tool here.

REGRESSION FIX (Root cause #4 — see chunking-regression spec):
The previous version had a positive-pattern-matched "is this a generic
listing query?" check (is_generic_listing_query / _GENERIC_LISTING_PATTERN)
that the pipeline used to decide whether a zero-match query should fall
back to the full route table or be treated as a failed specific lookup.
That pattern list missed real phrasing ("what shuttle routes are
available?" has no literal "tell me about" / "list" / "what are the"
substring), so the query fell through to "failed specific lookup" — and
THEN the whole-string difflib fallback inside the old _partial_score
(comparing the entire raw query against each stop name) coincidentally
scored some unrelated stop high enough to count as a "match" purely
because the two strings shared common letters, with no real relationship.

Both problems are fixed by:
  1. Removing the whole-string difflib fallback entirely.
  2. Returning zero matches for any no-confident-match case; the CALLER
     (pipeline.py) shows the full route table.

FIX 3 — "point" stopword (Failure 3):
  In Pakistan English a bus stop is commonly called a "point"
  (e.g. "Nazimabad point", "which point goes through Gulshan?").
  Before this fix, "point" survived stopword-stripping as a "distinctive"
  token, so the all-tokens-must-match rule required BOTH "point" AND the
  location name to appear together as words inside a stop entry.  No real
  stop entry is named "Nazimabad Point" (they're just "Nazimabad"), so
  the all-tokens rule failed and no route was matched — even though
  "nazimabad" alone would have matched correctly.

  Fix: add "point" and "points" to _QUERY_STOPWORDS.  Like "stop"/"stops"
  (which were already there), these are transit-domain filler words that
  name no specific location, and omitting them from the distinctive-token
  set lets the actual place name ("Nazimabad") do the matching correctly.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

import pandas as pd

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# Ordinary question phrasing to strip before looking for "distinctive"
# (place-name-like) tokens in the query. This list doesn't need to be
# exhaustive to be SAFE — see _partial_score's docstring: a word that
# slips through here just means a real match is missed and the query
# falls back to the full route table (safe), never that a wrong route
# gets matched (unsafe).
_QUERY_STOPWORDS = {
    "i", "live", "in", "near", "around", "from", "at", "to", "a", "an", "the",
    "is", "are", "what", "which", "where", "does", "do", "for", "me", "my",
    "you", "your", "please", "tell", "about", "find", "there", "any", "best",
    "shuttle", "route", "routes", "bus", "service", "transport", "commute",
    "stop", "stops", "of", "on", "and", "or", "covers", "cover", "covering",
    # FIX 3: "point" / "points" — Pakistan English for "bus stop".
    # e.g. "which point goes through Nazimabad?" — "point" names no real
    # location and must be stripped so "nazimabad" alone is the distinctive
    # token. Same logic as "stop"/"stops" already present above.
    "point", "points",
    # FIX 3: "boarding" / "alighting" — transit-domain filler synonyms for
    # stop/point that also name no specific location.
    "boarding", "alighting",
    # generic "list everything" phrasing words
    "available", "give", "provide", "info", "information", "details",
    "list", "all", "existing", "current", "want", "know", "exist", "exists",
    "show", "can", "could",
    # common motion/request verbs
    "go", "goes", "going", "get", "gets", "getting", "reach", "reaches",
    "reaching", "need", "needs", "looking", "search", "searching", "take",
    "takes", "visit", "visits", "use", "uses", "check", "help", "way",
    "how", "passing", "passes", "pass", "stay", "located", "area", "place",
    # FIX 3: additional transit verbs that name no location
    "through", "via", "pass", "passes", "passing", "cross", "crosses",
}


def _normalize(s: str) -> str:
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def _partial_score(query_text: str, stop_name: str) -> float:
    """
    Score how well `stop_name` matches somewhere inside `query_text`.

    1. Exact substring containment (either direction) → 100.
    2. ALL-distinctive-words match → 95. Strips ordinary question phrasing
       (_QUERY_STOPWORDS) from the query, then requires EVERY remaining
       word to appear as an exact word in the stop name.
    3. A narrow, high-bar typo-tolerance fallback — ONLY when there is
       exactly ONE distinctive query token, of meaningful length (>=4
       chars), compared WORD-BY-WORD against the stop's individual words.
       Requires a difflib ratio >= 85 to count at all.
    Anything else scores 0.
    """
    q = _normalize(query_text)
    s = _normalize(stop_name)
    if not q or not s:
        return 0.0

    if s in q or q in s:
        return 100.0

    q_tokens = [t for t in q.split() if t not in _QUERY_STOPWORDS and len(t) >= 3]
    if not q_tokens:
        return 0.0

    s_tokens = set(s.split())
    matched = sum(1 for qt in q_tokens if qt in s_tokens)

    if matched == len(q_tokens):
        return 95.0

    if matched > 0:
        return 40.0 * (matched / len(q_tokens))

    if len(q_tokens) == 1 and len(q_tokens[0]) >= 4:
        token = q_tokens[0]
        best = max(
            (difflib.SequenceMatcher(None, token, w).ratio() * 100.0 for w in s_tokens),
            default=0.0,
        )
        return best if best >= 85.0 else 0.0

    return 0.0


def find_routes_by_location(
    query_text: str,
    df: pd.DataFrame,
    threshold: float = 70.0,
) -> list[dict[str, Any]]:
    """
    Fuzzy-match query_text against every stop name in every route/leg.

    Returns a list of matching rows (as dicts), each with an added
    'matched_stop' (the specific stop name that triggered the match) and
    'match_score' (0-100), sorted best-match first. A stop appearing in
    multiple routes returns ALL of them — never picks just one.

    An empty return means "no confident match" — the caller (pipeline.py)
    treats this as "show the full route table."
    """
    if df is None or len(df) == 0:
        return []

    matches: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        stops = row.get("stops_list") or []
        if isinstance(stops, str):
            try:
                stops = json.loads(stops)
            except (ValueError, TypeError):
                stops = []

        best_score = 0.0
        best_stop = None
        for stop in stops:
            score = _partial_score(query_text, stop)
            if score > best_score:
                best_score = score
                best_stop = stop

        if best_score >= threshold:
            match = row.to_dict()
            match["matched_stop"] = best_stop
            match["match_score"] = round(best_score, 1)
            matches.append(match)

    matches.sort(key=lambda m: (-m["match_score"], str(m.get("route_id", ""))))
    return matches
