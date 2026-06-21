"""
answer_generator.py
────────────────────
Produces the final answer using intent-specific skill templates.
Supports conversational context history and configurable token budgets.

EVIDENCE FORMATTING FIX:
The previous version joined CSV rows with "; " on one line and RAG chunks
as "(from: source) content[:500]" on one line each. Two problems with that:
  1. RAG chunk `content` can itself contain internal newlines (multi-line
     records, e.g. a shuttle route's stop list). Cramming that into a
     single "line" with no real boundary marker meant that once several
     such chunks were concatenated, there was nothing forcing the model to
     keep track of which content belonged to which header — it could (and
     did) blend a route number from one chunk with the stop list of
     another.
  2. Truncating content to 500 chars could cut a record in half (e.g. a
     long shuttle route's stop list), discarding real information instead
     of just being a length cap.
This version gives every evidence item (CSV row or RAG chunk) its own
clearly fenced block with the label directly attached to its own content,
and avoids collapsing list-valued fields into an unreadable inline repr.
"""

from __future__ import annotations

from typing import Any

from config_loader import cfg
from core.llm_handler import LocalLLM
from core.skills import get_skill

_MAX_TOKENS = int(cfg["answer_max_tokens"])

# Cap per RAG chunk to avoid blowing the prompt budget on pathological
# inputs, but generous enough that a normal multi-line record (e.g. a full
# shuttle route's stop list) is never truncated mid-record. Previously this
# was 500, which could cut a long route's stops in half.
_MAX_CHUNK_CHARS = 1500


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
    """
    lines = ["── STRUCTURED DATA ──"]
    for i, row in enumerate(csv_rows[:15], 1):
        lines.append(f"[ROW {i}]")
        for k, v in row.items():
            if v is None or str(v) in ("nan", "", "None", "[]"):
                continue
            lines.append(f"  {k}: {_format_value(v)}")
        lines.append("")  # blank line = explicit boundary between rows
    return "\n".join(lines).rstrip()


def _format_rag_evidence(evidence_parts: list[dict]) -> str:
    """Format RAG chunks as clearly fenced, self-labeled blocks.

    Each chunk's source/topic label sits directly above its own content,
    separated from the next chunk by a blank line + dashed rule — this is
    the actual fix for the route-mislabeling bug: the model no longer has
    to infer which header a multi-line block of text belongs to.
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
        "Generate a helpful answer using ONLY the evidence above. Each evidence "
        "item is self-contained between its own header and the next boundary "
        "marker ('---' or a blank line) — do not mix content across items."
    )

    user_prompt = "\n\n".join(user_prompt_parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    answer = llm.chat(messages, max_new_tokens=_MAX_TOKENS, temperature=0.1)
    return answer