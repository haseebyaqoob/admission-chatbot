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

Both problems are fixed here by:
  1. Removing the whole-string difflib fallback entirely. The only
     remaining fuzzy path is a much narrower, much higher-bar single-token
     typo check (see _partial_score below) — never a comparison of the
     full query string against a stop's full name.
  2. Removing the dependency on phrase-pattern detection for "is this
     generic?" altogether. find_routes_by_location() now simply returns
     zero matches whenever nothing confidently matches (whether because
     the query had no distinctive words at all, or because it named a
     specific-sounding place that genuinely isn't in the data — e.g. a
     fake stop). The CALLER (pipeline.py) is expected to treat ANY
     zero-match result as "show the full table" — see pipeline.py's
     SHUTTLE branch. This is a deliberate trade-off: a query that names a
     real-sounding but nonexistent place ("Is there a shuttle from Mars
     Colony?") will now also get the full table rather than a tailored
     "no route covers that" message — but it will NEVER again get a wrong
     single route guessed by coincidence, which is the more severe
     failure mode. skill_shuttle() in skills.py is instructed to add a
     short note covering this case when presenting the full table.

No new dependency is required — uses difflib (stdlib) for the narrow typo
fallback described above.
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
# gets matched (unsafe). It's worth keeping reasonably complete for
# precision/usefulness, but its incompleteness is no longer a correctness
# bug the way the old whole-string difflib fallback was.
_QUERY_STOPWORDS = {
    "i", "live", "in", "near", "around", "from", "at", "to", "a", "an", "the",
    "is", "are", "what", "which", "where", "does", "do", "for", "me", "my",
    "you", "your", "please", "tell", "about", "find", "there", "any", "best",
    "shuttle", "route", "routes", "bus", "service", "transport", "commute",
    "stop", "stops", "of", "on", "and", "or", "covers", "cover", "covering",
    # generic "list everything" phrasing words that aren't place names but
    # would otherwise leak through as "distinctive" tokens, e.g.
    # "available" in "what shuttle routes are available?"
    "available", "give", "provide", "info", "information", "details",
    "list", "all", "existing", "current", "want", "know", "exist", "exists",
    "show", "can", "could",
    # common motion/request verbs that name no place by themselves, e.g.
    # "goes" in "which shuttle goes to Nazimabad?" — without filtering
    # these, "goes" + "nazimabad" both become "distinctive" tokens, and
    # since "goes" never appears in any real stop name, the all-tokens-
    # must-match rule below would fail even though "nazimabad" alone
    # would have matched cleanly.
    "go", "goes", "going", "get", "gets", "getting", "reach", "reaches",
    "reaching", "need", "needs", "looking", "search", "searching", "take",
    "takes", "visit", "visits", "use", "uses", "check", "help", "way",
    "how", "passing", "passes", "pass", "stay", "located", "area", "place",
}


def _normalize(s: str) -> str:
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def _partial_score(query_text: str, stop_name: str) -> float:
    """
    Score how well `stop_name` matches somewhere inside `query_text`.

    1. Exact substring containment (either direction) → 100.
    2. ALL-distinctive-words match → 95. Strips ordinary question phrasing
       (_QUERY_STOPWORDS) from the query, then requires EVERY remaining
       word to appear as an exact word in the stop name — not just any
       single word. This is what correctly rejects a fabricated "Mars
       Colony" (needs both "mars" AND "colony" present in the same stop;
       no real stop has both) while still matching a real stop like
       "Naval Colony" against the query "naval colony" (both words
       present), and still matching a single distinctive word like
       "Defence" or "Nazimabad" on its own (trivially "all of one word").
    3. A narrow, high-bar typo-tolerance fallback — ONLY when there is
       exactly ONE distinctive query token, of meaningful length (>=4
       chars), compared WORD-BY-WORD against the stop's individual words
       (never the whole query string against the whole stop string — that
       whole-string comparison is what produced the Root-cause-#4 false
       match). Requires a difflib ratio >= 85 to count at all.
    Anything else scores 0 — there is deliberately no remaining "fall back
    to comparing the whole query against the whole stop name" path.
    """
    q = _normalize(query_text)
    s = _normalize(stop_name)
    if not q or not s:
        return 0.0

    if s in q or q in s:
        return 100.0

    q_tokens = [t for t in q.split() if t not in _QUERY_STOPWORDS and len(t) >= 3]
    if not q_tokens:
        # No distinctive tokens at all — nothing to score against this
        # stop. Returning 0 (instead of a whole-string difflib ratio) is
        # the actual fix for Root cause #4.
        return 0.0

    s_tokens = set(s.split())
    matched = sum(1 for qt in q_tokens if qt in s_tokens)

    if matched == len(q_tokens):
        return 95.0

    if matched > 0:
        # Partial match (some but not all distinctive words present) is
        # deliberately kept well below any sane threshold — this rejects
        # "Mars Colony" (only "colony" matches, "mars" doesn't) without
        # needing a hand-curated list of "generic suffix" words.
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

    An empty return means "no confident match" — this can mean either
    "the query had no distinctive location-like word at all" or "it named
    a specific-sounding place that isn't in the data." The caller
    (pipeline.py) treats both the same way: fall back to showing the full
    route table. See module docstring for why that's the safer default.
    """
    if df is None or len(df) == 0:
        return []

    matches: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        stops = row.get("stops_list") or []
        if isinstance(stops, str):
            # Defensive: in case caller passed the raw JSON-encoded column
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