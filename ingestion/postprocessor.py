"""
ingestion/postprocessor.py — Raw Docling JSON → clean RAG-ready JSON.

Reads:   extracted/*__digital.json  and  extracted/*__scanned.json
Writes:  extracted/*__digital_clean.json  and  extracted/*__scanned_clean.json

TXT files (*__txt_clean.json) are already clean — produced by extractor.py.
This module only processes PDF Docling output.

What it fixes
-------------
1. REDUNDANT grid ARRAYS — Docling stores every cell twice (table_cells +
   grid). We drop grid, keep only table_cells.

2. FLAT TABLE_CELLS LIST — table_cells uses row/col offset indices. We
   group into row records: {"row_index": 1, "cells": ["BSCS", "45,000"]}

3. SCATTERED TEXT FRAGMENTS — texts is a flat list of small fragments. We
   merge consecutive text-label elements into paragraphs and keep
   section_header/title as distinct heading entries.

4. OCR GARBAGE DETECTION — flags fragments with low vowel ratio or heavy
   digit/letter mixing as "likely_garbage": true rather than deleting them.
   The chunker decides whether to skip flagged elements.

Output schema (same for all source types)
------------------------------------------
{
  "source_name":     str,
  "source_filename": str,
  "num_pages":       int | null,
  "text_elements": [
    {
      "type":           "heading" | "text",
      "text":           str,
      "level":          int | null,   # headings only
      "page_no":        int | null,
      "likely_garbage": bool
    }
  ],
  "tables": [
    {
      "num_rows": int,
      "num_cols": int,
      "page_no":  int | null,
      "rows": [
        {
          "row_index":         int,
          "is_column_header":  bool,
          "is_section_header": bool,
          "cells":             [str, ...]
        }
      ]
    }
  ],
  "stats": {
    "num_text_elements":           int,
    "num_tables":                  int,
    "num_likely_garbage_elements": int
  }
}
"""

import json
import logging
import re
from pathlib import Path

from config_loader import cfg

log = logging.getLogger("ingest.postprocessor")

EXTRACTED_DIR = Path(cfg["extracted_dir"])


# ─── OCR garbage heuristic ────────────────────────────────────────────────────

def is_likely_garbage(text: str, min_len: int = 6) -> bool:
    """
    Flags likely OCR misreads e.g. "3m gwe 3 3m gg", "oWaseem 3m8".
    Conservative — never flags short strings, plain numbers, or real codes.
    Explicitly excluded: ordinal suffixes (nd, th, st, rd), plain integers,
    codes with separators (CE-501 — hyphen breaks fullmatch).
    """
    stripped = text.strip()
    if len(stripped) < min_len:
        return False

    letters = re.findall(r"[A-Za-z]", stripped)
    if len(letters) < min_len:
        return False

    vowels      = re.findall(r"[AEIOUaeiou]", stripped)
    vowel_ratio = len(vowels) / len(letters)

    dl_mix      = len(re.findall(r"\d[A-Za-z]|[A-Za-z]\d", stripped))
    mix_ratio   = dl_mix / max(len(stripped), 1)

    ORDINAL     = {"nd", "th", "st", "rd"}
    tokens      = stripped.split()
    short_alnum = [
        t for t in tokens
        if re.fullmatch(r"[A-Za-z0-9]{1,4}", t)
        and re.search(r"[A-Za-z]", t)
        and re.search(r"\d", t)
        and t.lower() not in ORDINAL
    ]
    repeated_alnum = (
        len(short_alnum) >= 2
        and len(short_alnum) / max(len(tokens), 1) >= 0.3
    )

    if vowel_ratio < 0.20:
        return True
    if mix_ratio > 0.15:
        return True
    if repeated_alnum:
        return True
    return False


# ─── Table flattening ─────────────────────────────────────────────────────────

def flatten_table(table: dict) -> dict:
    """Convert a Docling table object to clean row records."""
    data     = table.get("data", {})
    cells    = data.get("table_cells", [])
    num_rows = data.get("num_rows", 0)
    num_cols = data.get("num_cols", 0)

    page_no = None
    prov    = table.get("prov", [])
    if prov:
        page_no = prov[0].get("page_no")

    # Group cells by row index
    rows_map: dict = {}
    for cell in cells:
        r = cell.get("start_row_offset_idx", 0)
        rows_map.setdefault(r, []).append(cell)

    rows_out = []
    for r in sorted(rows_map.keys()):
        row_cells = sorted(rows_map[r], key=lambda c: c.get("start_col_offset_idx", 0))
        rows_out.append({
            "row_index":         r,
            "is_section_header": any(c.get("row_section")    for c in row_cells),
            "is_column_header":  any(c.get("column_header")  for c in row_cells),
            "cells":             [c.get("text", "") for c in row_cells],
        })

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "page_no":  page_no,
        "rows":     rows_out,
    }


# ─── Text reconstruction ──────────────────────────────────────────────────────

def flatten_texts(texts: list) -> list:
    """
    Merge consecutive text fragments into paragraphs.
    Headings (section_header, title) flush any buffered text first.
    """
    out: list         = []
    buffer: list      = []
    buffer_page       = None

    def flush():
        if buffer:
            merged = " ".join(buffer).strip()
            out.append({
                "type":           "text",
                "text":           merged,
                "page_no":        buffer_page,
                "likely_garbage": is_likely_garbage(merged),
            })
            buffer.clear()

    for t in texts:
        label   = t.get("label", "text")
        content = t.get("text", t.get("orig", ""))
        page_no = None
        prov    = t.get("prov", [])
        if prov:
            page_no = prov[0].get("page_no")

        if label in ("section_header", "title"):
            flush()
            out.append({
                "type":           "heading",
                "text":           content,
                "level":          t.get("level"),
                "page_no":        page_no,
                "likely_garbage": is_likely_garbage(content),
            })
        else:
            # Page break → flush before starting new buffer
            if buffer and buffer_page != page_no:
                flush()
            buffer.append(content)
            buffer_page = page_no

    flush()
    return out


# ─── Core transform ───────────────────────────────────────────────────────────

def process_docling_document(doc: dict) -> dict:
    tables_clean = [flatten_table(t) for t in doc.get("tables", [])]
    texts_clean  = flatten_texts(doc.get("texts", []))
    n_garbage    = sum(1 for t in texts_clean if t.get("likely_garbage"))

    return {
        "source_name":     doc.get("name"),
        "source_filename": doc.get("origin", {}).get("filename"),
        "num_pages":       len(doc.get("pages", {})),
        "text_elements":   texts_clean,
        "tables":          tables_clean,
        "stats": {
            "num_text_elements":           len(texts_clean),
            "num_tables":                  len(tables_clean),
            "num_likely_garbage_elements": n_garbage,
        },
    }


# ─── File-level helper ────────────────────────────────────────────────────────

def process_file(in_path: Path) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    cleaned  = process_docling_document(doc)
    out_path = in_path.with_name(in_path.stem + "_clean.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    in_sz  = in_path.stat().st_size
    out_sz = out_path.stat().st_size
    pct    = 100 * (1 - out_sz / in_sz) if in_sz else 0
    log.info(
        f"  {in_path.name} -> {out_path.name} "
        f"({in_sz:,} -> {out_sz:,} bytes, -{pct:.0f}%) "
        f"| {cleaned['stats']}"
    )


# ─── Main entry point ─────────────────────────────────────────────────────────

def run() -> None:
    """
    Post-process all unprocessed *__digital.json and *__scanned.json files
    in extracted_dir into *_clean.json files.
    """
    candidates = sorted(
        p for p in EXTRACTED_DIR.glob("*.json")
        if ("__digital" in p.stem or "__scanned" in p.stem)
        and "_clean" not in p.stem
        and "ingestion_manifest" not in p.stem
    )

    if not candidates:
        log.info("No unprocessed Docling JSONs found — nothing to post-process.")
        return

    log.info(f"Post-processing {len(candidates)} file(s) ...")
    for path in candidates:
        try:
            process_file(path)
        except Exception:
            log.exception(f"FAILED on {path.name}")
