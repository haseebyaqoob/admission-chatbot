"""
chunker.py
───────────
Text chunking with token-aware enforcement, contextual header enrichment,
and config-driven parameters. Every numeric value is read from config.yaml.

Key features:
  - Docling HybridChunker (structure-aware) for PDF-derived documents
  - Tabular heuristics for allowlisted sources (shuttle, supervisor, programs)
  - Plain-text sliding-window fallback for everything else
  - Token enforcement: any chunk exceeding effective_max_tokens is split at
    sentence boundaries recursively
  - Contextual enrichment: every chunk gets a compact [Source | Section | Topic]
    header prepended to its text so document identity survives embedding
  - Single-chunk fallback: if Docling produces only 1 chunk for a document
    with >600 chars, re-chunk with the sentence-window fallback

ADDITION — whole-file preservation (FIX for hostel/facilities chunks):
  Some files contain tightly coupled structured content (capacity tables,
  numbered sections) that loses meaning when split at token boundaries.
  For example, hostel_facilities.txt has a 5-column capacity table whose
  header and data rows end up in separate chunks, causing reranker scores
  near 0.033 — below any useful threshold — and forcing the model to
  hallucinate from general knowledge.

  The fix: before splitting, check the file's stem against a list of
  protected keywords in config.yaml (chunking.preserve_whole_files). If
  the stem matches AND the entire text fits within the preservation
  ceiling (default 1500 tokens), emit the file as a single chunk and
  skip all normal splitting AND token-limit enforcement for that file.
  The ceiling is intentionally generous (well above the normal 400-token
  chunk size) to handle files like hostel_facilities.txt (~600 tokens)
  that are still reasonable for embedding but must not be torn apart.

  The keyword list lives entirely in config.yaml — no file names appear
  in this file. Adding a new protected file requires only a config edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config_loader import cfg

# ── All tunable values from config ────────────────────────────────────
_CNF = cfg.get("chunking", {})
_MAX_TOKENS       = int(_CNF.get("max_tokens", 400))
_OVERLAP_TOKENS   = int(_CNF.get("overlap_tokens", 40))
_MIN_TOKENS       = int(_CNF.get("min_tokens", 30))
_FALLBACK_SIZE    = int(_CNF.get("fallback_chunk_size", 400))
_FALLBACK_OVERLAP = int(_CNF.get("fallback_overlap", 40))
_HEADER_TEMPLATE  = str(_CNF.get("context_header_template",
    "[Source: {document_name} | Section: {section_heading} | Topic: {topic}]"))

_VS_CNF = cfg.get("vector_store", {})
_EFFECTIVE_MAX_TOKENS = int(_VS_CNF.get("effective_max_tokens", 400))

# Tabular source allowlist from config
_TABULAR_SOURCE_KEYWORDS = set(
    cfg.get("tabular_source_keywords", ["shuttle", "supervisor"])
)

# Whole-file preservation: stems whose entire content must be emitted as
# one chunk (provided they fit within the ceiling below). Read from config
# so no file names are hardcoded in this module.
_PRESERVE_WHOLE_KEYWORDS: list[str] = [
    kw.lower()
    for kw in _CNF.get("preserve_whole_files", [])
]
# Maximum token count for a file to qualify for whole-file preservation.
# Files above this limit fall through to normal chunking even if their
# stem keyword matches the preserve list.
_PRESERVE_WHOLE_MAX_TOKENS = int(_CNF.get("preserve_whole_max_tokens", 1500))

# ── Shared patterns ──────────────────────────────────────────────────
_COLUMN_ROW = re.compile(r"^\d+\s{2,}\S.*\S\s{2,}\S")
_LONE_RECORD_NUMBER = re.compile(r"^\*?\d+(?:\+\d+)?\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING_LIKE = re.compile(
    r"^(#{1,6}\s+)?[A-Z][A-Z\s]{3,}$|"
    r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,5}$"
)
_MIN_TABLE_ROWS = 3
_MIN_RECORD_BLOCKS = 2

# Lazy-loaded tokenizer singleton
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        model_name = _VS_CNF.get("embedding_model", "BAAI/bge-base-en-v1.5")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


@dataclass
class Chunk:
    chunk_id:    str
    source:      str
    topic:       str
    content:     str
    token_count: int = 0


# ── Document-name cleaning ────────────────────────────────────────────

def _clean_document_name(source: str) -> str:
    name = Path(source).stem
    name = name.replace("_extracted", "").replace("_ocr", "")
    name = name.replace("_", " ").replace("-", " ").strip()
    name = re.sub(r"^\d+[a-z]+_\s*", "", name, flags=re.IGNORECASE).strip()
    return name.title()


def _infer_section_heading(text: str) -> str | None:
    lines = text.strip().split("\n")
    for line in lines[:8]:
        raw = line.strip().strip("#").strip()
        if raw and len(raw) < 100 and _HEADING_LIKE.match(raw):
            return raw
    return None


def _extract_frequent_noun_phrase(text: str) -> str:
    words = text.lower().split()
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "or", "but", "not", "no", "this",
            "that", "these", "those", "it", "its", "his", "her", "their",
            "our", "your", "my", "all", "each", "every", "both", "few",
            "more", "most", "some", "any", "such", "only", "own", "same",
            "than", "also", "very", "just", "about", "if", "then", "else",
            "when", "where", "which", "who", "whom", "what", "why", "how"}
    freq = {}
    for w in words:
        w_clean = w.strip(".,;:!?\"'()[]{}")
        if w_clean and len(w_clean) > 3 and w_clean not in stop:
            freq[w_clean] = freq.get(w_clean, 0) + 1
    if not freq:
        return ""
    return max(freq, key=freq.get)


def _build_context_header(
    source: str, section_heading: str | None, topic: str
) -> str:
    doc_name = _clean_document_name(source)
    heading = section_heading or topic or "General"
    topic_label = topic or heading
    return _HEADER_TEMPLATE.format(
        document_name=doc_name,
        section_heading=heading,
        topic=topic_label,
    )


# ── Token enforcement ─────────────────────────────────────────────────

def _enforce_token_limit(chunks: list[Chunk]) -> list[Chunk]:
    """Split chunks exceeding effective_max_tokens at sentence boundaries."""
    result: list[Chunk] = []
    for chunk in chunks:
        token_count = _count_tokens(chunk.content)
        chunk.token_count = token_count

        if token_count < _MIN_TOKENS:
            print(
                f"  [chunker] Discarding {chunk.chunk_id} "
                f"({token_count} tokens < {_MIN_TOKENS})"
            )
            continue

        if token_count <= _EFFECTIVE_MAX_TOKENS:
            result.append(chunk)
            continue

        print(
            f"  [chunker] Splitting {chunk.chunk_id} "
            f"({token_count} tokens > {_EFFECTIVE_MAX_TOKENS})"
        )
        sentences = _split_sentences(chunk.content)
        if len(sentences) <= 1:
            sentences = chunk.content.split("\n")
        sub_texts: list[str] = []
        current = []
        current_len = 0
        for sent in sentences:
            sent_tokens = _count_tokens(sent)
            if current_len + sent_tokens > _EFFECTIVE_MAX_TOKENS and current:
                sub_texts.append(" ".join(current))
                current = [sent]
                current_len = sent_tokens
            else:
                current.append(sent)
                current_len += sent_tokens
        if current:
            sub_texts.append(" ".join(current))
        for j, sub_text in enumerate(sub_texts):
            sub = Chunk(
                chunk_id=f"{chunk.chunk_id}_sub_{j}",
                source=chunk.source,
                topic=chunk.topic,
                content=sub_text,
                token_count=_count_tokens(sub_text),
            )
            result.append(sub)
    return result


# ── Contextual header enrichment ──────────────────────────────────────

def _enrich_chunk(chunk: Chunk, section_heading: str | None = None) -> Chunk:
    header = _build_context_header(chunk.source, section_heading, chunk.topic)
    chunk.content = f"{header}\n{chunk.content}"
    chunk.token_count = _count_tokens(chunk.content)
    return chunk


def _apply_header_enrichment(
    chunks: list[Chunk],
    docling_chunks_raw: list | None = None,
) -> list[Chunk]:
    enriched: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        heading = None
        if docling_chunks_raw and i < len(docling_chunks_raw):
            heading = _infer_section_heading(chunk.content)
        if heading is None:
            heading = _infer_section_heading(chunk.content)
        enriched.append(_enrich_chunk(chunk, heading))
    return enriched


# ── Single-chunk fallback ──────────────────────────────────────────────

def _apply_single_chunk_fallback(
    docling_chunks: list[Chunk],
    source: str,
    topic: str,
    raw_text: str,
) -> list[Chunk]:
    if len(docling_chunks) == 1 and len(raw_text) > 600:
        print(
            f"  [chunker] WARNING: {source} produced 1 chunk with "
            f"{len(raw_text)} chars — applying sentence-window fallback"
        )
        return _chunk_plain_text(raw_text, source, topic)
    return docling_chunks


# ── Plain-text sliding-window chunker ─────────────────────────────────

def _chunk_plain_text(text: str, source: str, topic: str) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(words):
        end = min(start + _FALLBACK_SIZE, len(words))
        chunk_txt = " ".join(words[start:end])
        chunks.append(Chunk(
            chunk_id=f"{source}_chunk_{idx}",
            source=source,
            topic=topic,
            content=chunk_txt,
        ))
        if end == len(words):
            break
        start += _FALLBACK_SIZE - _FALLBACK_OVERLAP
        idx += 1
    return chunks


# ── Tabular heuristics ────────────────────────────────────────────────

def _is_allowlisted_tabular_source(source_stem: str) -> bool:
    stem_lower = source_stem.lower()
    return any(keyword in stem_lower for keyword in _TABULAR_SOURCE_KEYWORDS)


def _try_chunk_column_table(
    lines: list[str], source: str, topic: str
) -> list[Chunk] | None:
    matching_idx = [
        i for i, ln in enumerate(lines) if _COLUMN_ROW.match(ln.strip())
    ]
    if len(matching_idx) < _MIN_TABLE_ROWS:
        return None
    header_lines = [
        ln for ln in lines[:matching_idx[0]]
        if ln.strip() and not re.match(r"^[-=\s]+$", ln.strip())
    ]
    header_text = "\n".join(header_lines).strip()
    chunks: list[Chunk] = []
    for n, i in enumerate(matching_idx):
        row_text = lines[i].strip()
        content = (
            f"{header_text}\n{row_text}".strip() if header_text else row_text
        )
        chunks.append(Chunk(
            chunk_id=f"{source}_row_{n}",
            source=source,
            topic=topic,
            content=content,
        ))
    return chunks


def _try_chunk_numbered_records(
    lines: list[str], source: str, topic: str
) -> list[Chunk] | None:
    boundary_idx = [
        i for i, ln in enumerate(lines)
        if _LONE_RECORD_NUMBER.match(ln.strip())
    ]
    if len(boundary_idx) < _MIN_RECORD_BLOCKS:
        return None
    header_lines = [ln for ln in lines[:boundary_idx[0]] if ln.strip()]
    header_text = "\n".join(header_lines).strip()
    chunks: list[Chunk] = []
    for n, start in enumerate(boundary_idx):
        end = (
            boundary_idx[n + 1] if n + 1 < len(boundary_idx) else len(lines)
        )
        record_lines = [ln for ln in lines[start:end] if ln.strip()]
        record_text = "\n".join(record_lines).strip()
        if not record_text:
            continue
        content = (
            f"{header_text}\n{record_text}".strip()
            if header_text
            else record_text
        )
        chunks.append(Chunk(
            chunk_id=f"{source}_record_{n}",
            source=source,
            topic=topic,
            content=content,
        ))
    return chunks if chunks else None


# ── Whole-file preservation check ────────────────────────────────────

def _should_preserve_whole(stem: str, token_count: int) -> bool:
    """Return True if this file stem matches a preserve keyword AND fits
    within the whole-file preservation token ceiling.

    Both checks are required:
      - Keyword match: ensures only explicitly listed files are preserved
        (the list comes from config, not hardcoded here).
      - Token ceiling: prevents accidentally preserving a large file that
        happens to have a matching stem keyword but is too long to embed
        meaningfully as a single chunk.
    """
    if not _PRESERVE_WHOLE_KEYWORDS:
        return False
    stem_lower = stem.lower()
    keyword_match = any(kw in stem_lower for kw in _PRESERVE_WHOLE_KEYWORDS)
    return keyword_match and token_count <= _PRESERVE_WHOLE_MAX_TOKENS


# ── Public API ────────────────────────────────────────────────────────

def _infer_topic(filename: str, text_prefix: str = "") -> str:
    name = Path(filename).stem
    name = name.replace("_extracted", "").replace("_ocr", "")
    topic = name.replace("_", " ").replace("-", " ").strip().title()
    if not topic:
        topic = text_prefix[:60].strip()
    return topic


def chunk_file(filepath: Path) -> list[Chunk]:
    """Read a text file, chunk it, enforce token limits, enrich headers.

    For files whose stem matches a keyword in config's
    chunking.preserve_whole_files AND whose total token count is within
    the preservation ceiling, the entire file is emitted as a single
    chunk. This bypasses both normal splitting and token-limit enforcement
    to keep tightly coupled content (e.g. capacity tables with headers)
    intact. Token limit enforcement is intentionally skipped for preserved
    files — the content cohesion benefit outweighs the slight overrun
    beyond the normal chunk budget.

    For tabular-allowlisted sources, tries column-table and numbered-record
    heuristics before falling back to plain-text sliding window.
    """
    if not filepath.exists():
        print(f"[chunker] Warning: file not found: {filepath}")
        return []

    text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    source = filepath.stem
    topic = _infer_topic(source, text[:80])

    # ── Whole-file preservation ────────────────────────────────────
    # Check before any other chunking logic. If the file qualifies, emit
    # it as a single chunk with header enrichment and return immediately.
    # Token-limit enforcement is deliberately NOT applied here — these
    # files are in the preserve list precisely because splitting them
    # would destroy the structural context that makes them useful.
    token_count = _count_tokens(text)
    if _should_preserve_whole(source, token_count):
        print(
            f"  [chunker] {source} → preserved as single chunk "
            f"({token_count} tokens, matched preserve_whole_files)"
        )
        single_chunk = Chunk(
            chunk_id=f"{source}_chunk_0",
            source=source,
            topic=topic,
            content=text,
            token_count=token_count,
        )
        return _apply_header_enrichment([single_chunk])

    # ── Normal chunking path ───────────────────────────────────────
    lines = text.split("\n")
    raw_chunks: list[Chunk] = []

    if _is_allowlisted_tabular_source(source):
        table_chunks = _try_chunk_column_table(lines, source, topic)
        if table_chunks:
            print(
                f"  [chunker] {source} → detected column table, "
                f"{len(table_chunks)} row-chunks"
            )
            raw_chunks = table_chunks
        else:
            record_chunks = _try_chunk_numbered_records(lines, source, topic)
            if record_chunks:
                print(
                    f"  [chunker] {source} → detected numbered records, "
                    f"{len(record_chunks)} record-chunks"
                )
                raw_chunks = record_chunks

    if not raw_chunks:
        raw_chunks = _chunk_plain_text(text, source, topic)

    # Post-processing pipeline
    chunks = _enforce_token_limit(raw_chunks)
    chunks = _apply_header_enrichment(chunks)
    return chunks


def chunk_docling_document(
    doc,
    source: str,
    tokenizer_model_name: str | None = None,
) -> list[Chunk]:
    """Structure-aware chunking via Docling's HybridChunker.

    Falls back to plain-text sentence-window chunking if Docling produces
    only 1 chunk for a large document. Applies token enforcement and
    contextual header enrichment to all output chunks.
    """
    from transformers import AutoTokenizer
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import (
        MarkdownTableSerializer,
        MarkdownParams,
    )

    model_name = (
        tokenizer_model_name
        or _VS_CNF.get("embedding_model", "BAAI/bge-base-en-v1.5")
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer)

    class MDTableSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(compact_tables=True),
            )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=_MAX_TOKENS,
        repeat_table_header=True,
        serializer_provider=MDTableSerializerProvider(),
    )

    topic = _infer_topic(source, "")
    raw_docling_chunks: list[Chunk] = []
    docling_objects = []

    for i, docling_chunk in enumerate(chunker.chunk(dl_doc=doc)):
        c = Chunk(
            chunk_id=f"{source}_chunk_{i}",
            source=source,
            topic=topic,
            content=docling_chunk.text,
        )
        raw_docling_chunks.append(c)
        docling_objects.append(docling_chunk)

    if not raw_docling_chunks:
        print(f"  [chunker] {source} (Docling) → 0 chunks — no content")
        return []

    raw_text = ""
    try:
        raw_text = (
            doc.export_to_text() if hasattr(doc, "export_to_text") else ""
        )
    except Exception:
        raw_text = ""

    fallback_chunks = _apply_single_chunk_fallback(
        raw_docling_chunks, source, topic, raw_text
    )
    chunks = _enforce_token_limit(fallback_chunks)
    chunks = _apply_header_enrichment(chunks)

    print(
        f"  [chunker] {source} (Docling) → {len(chunks)} chunks "
        "after enforcement + enrichment"
    )
    return chunks


def chunk_all_files(extracted_dir: Path) -> list[Chunk]:
    """Chunk all .txt files in extracted_dir using chunk_file()."""
    all_chunks: list[Chunk] = []
    for txt_file in sorted(extracted_dir.glob("*.txt")):
        file_chunks = chunk_file(txt_file)
        all_chunks.extend(file_chunks)
        if file_chunks:
            print(f"  [chunker] {txt_file.name} → {len(file_chunks)} chunks")
    print(
        f"  [chunker] Total: {len(all_chunks)} chunks across all files"
    )
    return all_chunks