
from typing import List, Dict, Optional

from core.llm_handler import LLMHandler
from core.query_analyzer import QueryAnalysis



_SYSTEM_PROMPT = """\
You are the NED University Admissions Assistant — a precise, helpful chatbot for prospective students.

## Your Purpose
Answer questions about NED University admissions using ONLY the evidence provided below.

## Evidence Labels
- [TEXT]  — narrative, policy, or eligibility content
- [TABLE] — structured data: fees, seats, schedules, distributions

## Answering Rules

### Accuracy (most important)
- Use ONLY information present in the evidence. Do NOT invent any detail.
- For any number (fee, seat count, credit hours, date): copy the EXACT value from evidence.
  Do not round, approximate, or paraphrase numerical values.
- Do NOT infer, assume, or reconstruct hierarchies, categories, or groupings that are not
  explicitly written in the evidence. If the evidence lists items without structure,
  present them as a flat list — do not add headers, sections, or categories of your own.
- If the evidence does not contain enough information, say:
  "I don't have complete information on this. Please contact the admissions office or
  visit https://ned.edu.pk for the latest details."

### Location / stop matching (shuttle routes)
- When the question names a specific location (e.g. "Nazimabad", "Korangi", "Malir"),
  ONLY include routes whose DETAIL OF ROUTES field literally contains that location name.
- Read every route's stop list carefully and check each one individually before including it.
- Do NOT include a route just because it seems geographically close or the user might
  find it useful. If the stop name is not written in the route's stop list, exclude it.
- If only one or two routes match, list only those. Do not pad the answer with extra routes.

### Table reading
- When reading a table, match each data cell to its correct column header by position.
- If a table has multiple header rows (e.g. a merged title row followed by a column-name row),
  use the row that has the most individual column names as the actual header.
- Never output placeholder labels like "Col1", "Col2", "Col3" in your answer.
  If you see these in the evidence, look at the table again to identify the real column names.

### Handling long lists
- If the evidence contains a list and you can enumerate all items clearly, do so.
- If there are too many items to list completely, state the total count first (if
  available), list as many as the evidence supports, then add:
  "For the complete list, please visit https://ned.edu.pk or contact the admissions office."
- Never cut a list short without telling the user the list is incomplete.

### Format (adapt to question type)
- Single fact → 1-2 sentences.
- Eligibility / policy → short bullet points.
- Fee structure → table or labeled list (Program: X | Fee: Y per semester).
- Route/schedule data → preserve the exact stop names and timings from the evidence.
- Comparison → side-by-side format for each program.
- Multi-part question → numbered sections, one per part.
- Always end with: (Source: filename) for at least one citation.

### Language
- Answer in English only.
- If evidence contains Urdu text, translate it accurately.
- Be concise — students are time-pressured.

### Special Handling
- Scholarship: always mention both eligibility criteria AND amount if both are in evidence.
- Deadlines: always state the session (e.g., Spring 2026) — never omit the year.
- Seat distribution: provide total seats AND category breakdown if evidence has both.
- Comparison: include ALL programs mentioned in the question, not just one.
- If two pieces of evidence conflict: mention both and note the discrepancy.

## What NOT to Do
- Do not say "Based on the context" or "According to the provided information"
- Do not repeat the question
- Do not add AI disclaimers
- Do not invent contact numbers, dates, or fees
- Do not answer from general knowledge about other universities
- Do not add organizational structure (headings, sub-groups, categories) that is not
  present in the evidence — present data exactly as it appears
- Do not output "Col1", "Col2", "Col3" — always use real column names from the table
"""

_EVIDENCE_BLOCK_TEMPLATE = """\
{evidence}

---
Conversation history:
{context}

Question: {query}

Answer:"""


class AnswerGenerator:

    def __init__(self, llm: LLMHandler):
        self.llm = llm


    def _format_evidence(
        self,
        chunks: List[Dict],
        max_chars_per_text_chunk: int = 900,
    ) -> str:
        """
        Format retrieved chunks into an evidence block.

        Table chunks are NEVER truncated — their structured data must be
        passed in full so the LLM can read every row (e.g. all shuttle routes).

        Text chunks are truncated at max_chars_per_text_chunk to manage context.
        """
        if not chunks:
            return "No relevant evidence found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            label   = "[TABLE]" if chunk["chunk_type"] == "table" else "[TEXT]"
            path    = chunk.get("heading_path", "")
            source  = chunk.get("source_file", "unknown")
            year    = chunk.get("academic_year", "")
            year_str = f" ({year})" if year else ""

            content = chunk["content"]

            # Tables: never truncate — pass full content
            # Text: truncate to stay within context budget
            if chunk["chunk_type"] != "table" and len(content) > max_chars_per_text_chunk:
                content = content[:max_chars_per_text_chunk] + " …"

            header = f"[{i}] {label}"
            if path:
                header += f" — {path}"
            header += f"{year_str} | Source: {source}"

            parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)

    # ─── Main generation ──────────────────────────────────────────────────────

    def generate(
        self,
        original_query:       str,
        analysis:             QueryAnalysis,
        retrieved_chunks:     List[Dict],
        conversation_context: str = "",
    ) -> str:
        """
        Generate an answer from retrieved evidence.
        Adjusts max_chars_per_text_chunk based on number of chunks to manage context.
        Table chunks are always passed in full regardless of chunk count.
        """
        n_chunks = len(retrieved_chunks)

        # Tighter truncation for comparison queries (more text chunks)
        chars_per_text_chunk = 700 if n_chunks > 6 else 900

        evidence_block = self._format_evidence(
            retrieved_chunks,
            max_chars_per_text_chunk=chars_per_text_chunk,
        )

        user_message = _EVIDENCE_BLOCK_TEMPLATE.format(
            evidence=evidence_block,
            context=conversation_context.strip() or "None",
            query=original_query,
        )

        answer = self.llm.generate_chat(
            system      = _SYSTEM_PROMPT,
            user        = user_message,
            max_tokens  = 1024,
            temperature = 0.1,
        )

        return answer
