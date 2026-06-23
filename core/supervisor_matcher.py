"""
supervisor_matcher.py
────────────────────────
Deterministic research-area → supervisor matching.

Mirrors shuttle_matcher.py and exists for the same underlying reason, but
addresses a more severe failure mode (Root cause #3 in the chunking-
regression spec): when asked "Find PhD supervisors in AI" against RAG
evidence that didn't actually contain a matching row, the model didn't say
"no match" — it invented two entirely fictional supervisors ("Dr. Ayesha
Khan", "Dr. Muhammad Ali") and presented them as if they came from the
real source file. Grounding (giving the model correct evidence) didn't
prevent this, because the model was still the one deciding WHICH rows
belonged in the table; under instruction-following pressure to produce a
filled table, it filled it with plausible-looking fabrications instead of
an empty one.

The fix is architectural, not another prompt instruction: the matching
decision (does any supervisor's subject match the requested area?) is made
here, deterministically, before the model ever sees the question. The
model only ever sees rows that have already been confirmed to match — it
is never put in a position where producing a table requires it to invent
a row.

UPDATE — research_area now comes pre-extracted from router.py, not from
a hand-maintained stopword list alone:
Stripping verbs out of the raw query with a fixed _QUERY_STOPWORDS set
doesn't scale — "manage" was a real production gap (the query "supervisors
who manage engineering and technology" failed to match a supervisor whose
subject literally IS "Engineering & Technology", because "manage" wasn't
in the stopword list, survived as a "distinctive" token, and the
all-tokens-must-match rule correctly found no row containing "manage").
Adding "manage" doesn't fix the underlying problem — the next verb
("handles", "deals with", "is responsible for"...) reopens it.

router.py now runs an LLM extraction step (Stage 2 for SUPERVISOR intent)
whose ONLY job is to strip that kind of verb/filler phrasing and hand back
a clean research_area noun phrase. That phrase is passed in here via the
`research_area` parameter and matched EXACTLY the same deterministic way
as before (normalize → require every distinctive token to appear in
`subject`). The LLM never participates in the match/no-match decision
itself — it only rephrases the question, never answers it. If no
research_area is supplied (extraction failed, or this is called from a
context that doesn't have router.py's Stage 2, e.g. an old caller or a
test), this function falls back to running the same stopword-stripping
extraction directly on query_text that it always has, so behavior is
unchanged for any caller that doesn't pass the new parameter.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

import pandas as pd

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# Fallback-path stopwords only (used when no pre-extracted research_area
# is supplied — see module docstring). Kept deliberately small; this is
# now a safety net for callers that skip Stage 2, not the primary
# mechanism for handling phrasing variation. New verb synonyms should NOT
# be added here as the fix for a matching gap — see module docstring for
# why that doesn't scale, and add/adjust the Stage 2 extraction prompt in
# router.py instead.
_QUERY_STOPWORDS = {
    "find", "search", "show", "me", "a", "an", "the", "in", "for", "of",
    "is", "are", "who", "which", "what", "supervisor", "supervisors",
    "phd", "research", "area", "areas", "guide", "thesis", "working",
    "on", "about", "with", "interested", "i", "want", "need", "looking",
    "to", "do", "does", "list", "all", "give", "provide",
    # Connective words. Unlike verbs ("manage", "handle", "deal with"...),
    # this is a small, genuinely closed set in English — there's no
    # open-ended list of new connectives to keep discovering, so adding
    # these doesn't reopen the scalability problem that motivated moving
    # verb/filler stripping to the LLM extractor in router.py. Without
    # this, an extracted research_area like "engineering and technology"
    # would keep "and" as a "distinctive" token, and the all-tokens-match
    # rule would then fail to match subject "Engineering & Technology"
    # (which has no literal word "and") — the multi-word equivalent of the
    # original "manage" bug, just one level downstream of the fix.
    "and", "or", "&",
}


def _normalize(s: str) -> str:
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def _distinctive_tokens(text: str) -> list[str]:
    """Tokenize and strip the fallback stopword list.

    Used both for the fallback path (raw query, no extracted research_area)
    and for tokenizing an already-clean research_area phrase (which won't
    contain much for this list to strip anyway, but running it is harmless
    and catches stray filler words the extractor might have left in).
    """
    q = _normalize(text)
    # Minimum length 2, not 3 — this domain has meaningful short acronyms
    # (AI, ML, OS, DB, VR, AR) that a length>=3 filter would silently drop,
    # which would be exactly backwards for this fix (AI is the literal
    # example from the bug report).
    return [t for t in q.split() if t not in _QUERY_STOPWORDS and len(t) >= 2]


def find_supervisors_by_area(
    query_text: str,
    df: pd.DataFrame,
    threshold: float = 70.0,
    research_area: str | None = None,
) -> list[dict[str, Any]]:
    """
    Match a requested research area against the literal `subject` field.
    Requires EVERY distinctive word from the area to appear as an exact
    word inside the subject field — not a single coincidental word — so a
    broad category like "Engineering & Technology" never gets treated as a
    match for a specific area like "AI" or "Cyber Security" just because
    the words are vaguely topically related.

    Parameters
    ----------
    query_text     : the raw user query (used as the matching source ONLY
                      if `research_area` is not supplied — see below).
    df              : supervisors DataFrame with a `subject` column.
    threshold       : difflib ratio floor for the narrow typo fallback.
    research_area   : an already-extracted, clean research-area phrase
                       (from router.py's Stage 2 LLM extraction). When
                       given, tokenization/matching runs against THIS
                       string instead of `query_text`, so the caller does
                       not need to maintain a verb/filler stopword list
                       for every way a user might phrase the request. When
                       None (extraction unavailable, or a caller that
                       predates this parameter), falls back to running the
                       same stopword-based extraction directly on
                       `query_text`, exactly as before.

    If the resulting tokens are empty (e.g. "list all PhD supervisors", or
    research_area was explicitly None from the extractor meaning "no area
    named"), returns ALL rows — a generic listing request, handled the
    same way shuttle's empty-distinctive-token case is.

    An empty (non-generic) result means no supervisor's subject matched —
    the caller should pass this through as empty evidence, NOT fall back
    to RAG search, so the model is never asked to fill a table it has no
    real rows for.
    """
    if df is None or len(df) == 0:
        return []

    match_source = research_area if research_area is not None else query_text
    tokens = _distinctive_tokens(match_source)
    if not tokens:
        return df.to_dict(orient="records")

    matches: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        subject = _normalize(str(row.get("subject", "")))
        subject_tokens = set(subject.split())

        if all(t in subject_tokens for t in tokens):
            match = row.to_dict()
            match["match_score"] = 100.0
            matches.append(match)
            continue

        # Narrow typo-tolerance fallback — ONLY for a single, meaningfully
        # long (>=4 char) distinctive token, scored word-by-word against
        # the subject's own words (never whole-query-vs-whole-subject).
        # Multi-word areas with a typo (e.g. "computer scince") are NOT
        # covered by this fallback and will fail closed (return no match)
        # rather than risk a coincidental fuzzy hit — consistent with the
        # restriction applied to shuttle_matcher's difflib path.
        if len(tokens) == 1 and len(tokens[0]) >= 4:
            token = tokens[0]
            best = max(
                (difflib.SequenceMatcher(None, token, w).ratio() * 100.0 for w in subject_tokens),
                default=0.0,
            )
            if best >= threshold + 15:  # higher bar than the base threshold
                match = row.to_dict()
                match["match_score"] = round(best, 1)
                matches.append(match)

    matches.sort(key=lambda m: (-m["match_score"], str(m.get("name", ""))))
    return matches
