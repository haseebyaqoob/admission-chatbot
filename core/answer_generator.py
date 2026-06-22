"""
answer_generator.py
────────────────────
Produces the final answer using intent-specific skill templates.
Supports conversational context history and configurable token budgets.

EVIDENCE FORMATTING:
Every evidence item (CSV row or RAG chunk) gets its own clearly fenced
block with the label directly attached to its own content, so multiple
items can never visually blend into each other regardless of how long any
individual field is (this is what fixed the earlier route-mislabeling bug,
where Route 16's stops were getting attributed to Route 17+3).

SOURCE ATTRIBUTION FOR STRUCTURED ROWS (FIX — this version):
Structured rows (SUPERVISOR/SHUTTLE/PROGRAMS, tagged by pipeline.py's
_tag_source with a `_source_file` field) previously had the source
rendered as a trailing "(cite this row as: ...)" line AFTER all the
regular fields. In practice, qwen2.5:7b-instruct frequently ignored this
trailing instruction and composed its own vague placeholder instead —
"(source: structured data)" — which is not a real filename and makes the
citation unverifiable.

The fix moves `_source_file` INTO the `[ROW i]` header itself:
    [ROW 1 | source: supervisors.csv]
      name: Dr. Abdul Ghaffar Memon
      ...
rather than appending it as a separate trailing line. This mirrors what
_format_rag_evidence() already does successfully for RAG chunks (their
source has always lived in the `[EVIDENCE i | source: ...]` header, and
that path has not shown the same fabricated-source-label problem). A
header is read first and applies to everything beneath it until the next
boundary, which is a much stronger positional cue for a small local model
than a trailing instruction the model has often already "moved past" by
the time it reaches end-of-row. skills.py's instructions have been
updated to describe this new header location explicitly.

POST-GENERATION TIME-CLAIM VERIFICATION (Root cause #2, fix #2):
Grounding (giving the model correct evidence) is necessary but is NOT
sufficient on its own to stop a specific kind of plausible-sounding
substitution: in production, Route 8's morning and evening legs both had
the literal evidence value `timing: 7:00 a.m`, and the model still wrote
"7:00 a.m" for morning and invented "7:00 p.m" for evening — not because
the evidence was wrong or ambiguous, but because the model "improved" on
a literal field value with something that seemed more plausible.

The skill-prompt instruction in skills.py (copy `timing` character-for-
character) is the first line of defense, but per the spec this needs a
second, structural check: extract any time-like token from the model's
answer and verify it appears verbatim somewhere in the evidence block. If
it doesn't, regenerate once with a targeted correction; if it's still
wrong after that, surface the issue in the answer rather than silently
shipping a value nobody confirmed.

Scope note: this check is deliberately narrowed to TIME tokens
(`\\d{1,2}:\\d{2}\\s*[ap]\\.?m`) rather than "any number," per the spec's
broader suggestion. A generic number-verifier would also flag legitimate,
intentional reformatting (e.g. the existing skill instruction that turns
"32" into "thirty-two (32)", or a date reformatted from the source's
"15/07/2026" into "July 15, 2026"), which would cause false-positive
regenerations and erode trust in the check. Time values don't have that
reformatting ambiguity in this domain (they're always presented in the
same H:MM am/pm shape), so this is the safe place to start. The same
mechanism (extract → verify verbatim → regenerate-or-flag) could be
extended to fee/PKR amounts later if a similar fabrication pattern shows
up there — see the structured-lookup fee/seats work mentioned in the
chunking-regression spec.
"""

from __future__ import annotations

import re
from typing import Any

from config_loader import cfg
from core.llm_handler import LocalLLM
from core.skills import get_skill

_MAX_TOKENS = int(cfg["answer_max_tokens"])

# Cap per RAG chunk to avoid blowing the prompt budget on pathological
# inputs, but generous enough that a normal multi-line record (e.g. a full
# shuttle route's stop list) is never truncated mid-record.
_MAX_CHUNK_CHARS = 1500

_TIME_TOKEN_RE = re.compile(r"\b\d{1,2}:\d{2}\s*[ap]\.?\s*m\.?\b", re.IGNORECASE)


def _normalize_time_token(tok: str) -> str:
    """Normalize a time token for comparison (strip spacing/punctuation
    variance like "7:00 a.m" vs "7:00a.m." vs "7:00 AM")."""
    return re.sub(r"[\s.]", "", tok).lower()


def _unverified_time_claims(answer: str, evidence_block: str) -> list[str]:
    """Return any time-like tokens in `answer` that don't appear (after
    normalizing spacing/punctuation) anywhere in `evidence_block`."""
    evidence_times = {
        _normalize_time_token(t) for t in _TIME_TOKEN_RE.findall(evidence_block)
    }
    bad: list[str] = []
    for t in _TIME_TOKEN_RE.findall(answer):
        if _normalize_time_token(t) not in evidence_times:
            bad.append(t)
    return bad


def _format_value(v: Any) -> str:
    """Render a single field value for display in the evidence block.

    Lists (e.g. stops_list) get a readable arrow-joined format instead of
    Python's default repr (['a', 'b', 'c']), which is harder for the model
    to parse correctly and easier to garble when several rows are nearby.
    """
    if isinstance(v, list):
        return " → ".join(str(x) for x in v)
    return str(v)


def _format_csv_rows(csv_rows: list[dict]) -> str:
    """Format CSV/structured rows as clearly fenced, self-labeled blocks.

    Each row gets its own [ROW i] header directly attached to its own
    content, so multiple rows can never visually bleed into each other
    regardless of how long any individual field is.

    FIX (source attribution): `_source_file` (set by pipeline.py's
    _tag_source — see its docstring) is pulled out of the regular field
    list, exactly as before, but is now rendered INSIDE the `[ROW i]`
    header itself — `[ROW 1 | source: supervisors.csv]` — instead of as a
    separate trailing "(cite this row as: ...)" line after all the other
    fields.

    Why this changed: the trailing-line approach relied on the model
    reading and obeying an instruction-shaped line positioned AFTER the
    data it was meant to apply to. qwen2.5:7b-instruct frequently failed
    to do this in practice and instead composed a vague, unverifiable
    placeholder like "(source: structured data)". Putting the same value
    in the header instead means it's the FIRST thing the model reads for
    this row, and headers-govern-content-until-next-boundary is already
    the exact convention _format_rag_evidence() below uses successfully
    for RAG chunk sources — this brings structured rows in line with that
    proven pattern instead of using a different, weaker convention for no
    real reason.

    The header is still visually distinct from ordinary `field: value`
    data rows (it's a single bracketed line, not an indented field), so
    skills.py's instruction not to display `_source_file` as a table
    column is unaffected — there's no `_source_file` field left in the
    per-row field loop to accidentally display in the first place.
    """
    lines = ["── STRUCTURED DATA ──"]
    for i, row in enumerate(csv_rows[:15], 1):
        source_file = row.get("_source_file")
        # FIX: source now lives in the row header, not a trailing line.
        header = f"[ROW {i} | source: {source_file}]" if source_file else f"[ROW {i}]"
        lines.append(header)
        for k, v in row.items():
            if k == "_source_file":
                continue
            if v is None or str(v) in ("nan", "", "None", "[]"):
                continue
            lines.append(f"  {k}: {_format_value(v)}")
        lines.append("")  # blank line = explicit boundary between rows
    return "\n".join(lines).rstrip()


def _format_rag_evidence(evidence_parts: list[dict]) -> str:
    """Format RAG chunks as clearly fenced, self-labeled blocks.

    Each chunk's source/topic label sits directly above its own content,
    separated from the next chunk by a blank line + dashed rule.
    """
    lines = ["── RETRIEVED EVIDENCE ──"]
    for i, ep in enumerate(evidence_parts[:6], 1):
        source = ep.get("source", "unknown")
        topic = ep.get("topic", "")
        content = ep.get("content", "")
        if len(content) > _MAX_CHUNK_CHARS:
            content = content[:_MAX_CHUNK_CHARS] + " …(truncated)"
        lines.append(f"[EVIDENCE {i} | source: {source}{f' | topic: {topic}' if topic else ''}]")
        lines.append(content.strip())
        lines.append("---")  # explicit boundary between chunks
    return "\n".join(lines).rstrip("- \n")


def generate_answer(
    query: str,
    llm: LocalLLM,
    evidence_parts: list[dict[str, Any]],
    intent: str,
    csv_rows: list[dict] | None = None,
    context_history: str = "",
) -> str:
    """
    Generate a final answer using the intent-specific skill template.

    Parameters
    ----------
    query            : original user query
    llm              : LocalLLM instance
    evidence_parts   : list of RAG chunks with {source, content, score}
    intent           : classified intent (used to select skill)
    csv_rows         : optional CSV/structured query results (as list of dicts)
    context_history  : recent QA history for conversational memory
    """
    system_prompt = get_skill(intent)

    # Build evidence block — each section clearly fenced and self-labeled.
    evidence_sections: list[str] = []

    if csv_rows:
        evidence_sections.append(_format_csv_rows(csv_rows))

    if evidence_parts:
        evidence_sections.append(_format_rag_evidence(evidence_parts))

    evidence_block = "\n\n".join(evidence_sections) if evidence_sections else "(no evidence available)"

    user_prompt_parts = []
    if context_history:
        user_prompt_parts.append(f"RECENT CONVERSATION:\n{context_history}\n")
    user_prompt_parts.append(f"User query: {query}")
    user_prompt_parts.append(f"Intent: {intent}")
    user_prompt_parts.append(f"EVIDENCE:\n{evidence_block}")
    user_prompt_parts.append(
        "Generate a helpful answer using the evidence and two-tier reasoning:\n\n"
        "TIER 1 — Corpus-grounded answers (EVIDENCE-REQUIRED):\n"
        "  For fees, eligibility, programs, supervisors, schedules, deadlines,\n"
        "  documents, facilities, hostel, contact info — use ONLY the evidence\n"
        "  provided above. Cite the source for each fact. If the evidence does\n"
        "  not contain the answer, say so plainly.\n\n"
        "TIER 2 — General Institutional Knowledge (NO EVIDENCE REQUIRED):\n"
        "  For basic factual questions about NED University that a prospective\n"
        "  student would reasonably expect anyone to know, you may use your own\n"
        "  general knowledge. These include:\n"
        "    - University location (e.g. \"Where is NED located?\" → Karachi)\n"
        "    - Full official name (NED University of Engineering and Technology)\n"
        "    - Founding year / establishment date (if widely known)\n"
        "    - General description of the institution\n"
        "  If you answer from general knowledge, explicitly state it:\n"
        "  \"(based on general knowledge)\" rather than citing a source.\n\n"
        "RULE: Before refusing to answer, determine whether this is a basic\n"
        "institutional fact. If yes, answer it even without retrieved evidence.\n\n"
        "Each evidence item is self-contained between its own header and the\n"
        "next boundary marker ('---' or a blank line) — do not mix content\n"
        "across items."
    )

    user_prompt = "\n\n".join(user_prompt_parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    answer = llm.chat(messages, max_new_tokens=_MAX_TOKENS, temperature=0.1)

    # ── Post-generation time-claim verification ──────────────────────
    # See module docstring for why this exists and why it's scoped to
    # time tokens specifically.
    unverified = _unverified_time_claims(answer, evidence_block)
    if unverified:
        print(f"[answer_generator] Unverified time claim(s) {unverified} — regenerating with correction")
        correction = (
            "\n\nIMPORTANT CORRECTION: your previous answer included a time "
            f"value ({', '.join(unverified)}) that does not appear anywhere "
            "in the evidence above. Every time value in your answer must be "
            "copied character-for-character from the evidence's `timing` "
            "field (or equivalent) — never inferred, adjusted, or guessed "
            "(e.g. do not assume an 'evening' leg must be PM just because "
            "it seems more plausible). If two rows show the identical "
            "timing value, your answer must show that identical value for "
            "both. Regenerate the full answer now using ONLY time values "
            "that appear verbatim in the evidence."
        )
        messages_retry = [
            {"role": "system", "content": system_prompt + correction},
            {"role": "user",   "content": user_prompt},
        ]
        answer2 = llm.chat(messages_retry, max_new_tokens=_MAX_TOKENS, temperature=0.0)
        still_unverified = _unverified_time_claims(answer2, evidence_block)
        if not still_unverified:
            answer = answer2
            print("[answer_generator] Correction succeeded")
        else:
            # Still wrong after a deterministic-temperature retry — don't
            # ship a silently-fabricated time value. Surface the issue
            # instead of hiding it.
            print(f"[answer_generator] Still unverified after retry: {still_unverified} — flagging in response")
            answer = answer2 + (
                "\n\n*(Note: a time value in this answer could not be "
                "automatically verified against the source data — please "
                "confirm exact timings with the relevant university office "
                "before relying on it.)*"
            )

    return answer
