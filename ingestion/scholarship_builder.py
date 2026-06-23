"""
scholarship_builder.py
──────────────────────
Build scholarships.csv from the extracted scholarship list text.

The source PDF ("List of Scholarship.pdf") is extracted to a .txt file
in the extracted directory. The raw text contains 52 numbered entries in
the format "N.  Scholarship Name", but rows 26–39 were collapsed by the
PDF extractor into a single pipe-delimited markdown table cell.

This parser handles both formats by replacing pipe characters with
newlines and normalising intra-line whitespace before applying the
numbered-entry regex to the full text via findall — no hardcoded row
numbers, no pre-split-at-pipe strategy.

Row 47 contains a spurious "Name of Scholarship" column header that the
PDF extractor included as an entry — it is skipped by name. No other
rows are skipped by number.

No eligibility criteria, award amounts, or deadlines are present in this
source file; the CSV therefore carries names only.

Output schema (one row per scholarship):
  number    e.g. "1", "26", "52" — stored as string, sorted numerically
  name      scholarship name exactly as printed in the source
  category  blank — not present in the source; reserved for future use
"""

import re
from pathlib import Path

import pandas as pd

from config_loader import cfg

_EXTRACTED_DIR = Path(cfg["extracted_dir"])
_SCHOLARSHIP_CSV_OUTPUT = Path(cfg["csv_output"]).parent / "scholarships.csv"

# Matches a numbered scholarship entry anywhere in the text:
#   group 1 — 1 to 3-digit number
#   group 2 — everything after "N.  " to the end of the line
# The \s{1,4} allows for 1–4 whitespace characters between the dot and
# the name, which covers the "N.  Name" (two spaces) and "N. Name" (one
# space after whitespace normalisation) shapes that appear in this source.
_NUMBERED_ENTRY = re.compile(r"(\d{1,3})\.\s{1,4}(.+)")

# Lowercase names that are structural artefacts of the PDF layout, not
# real scholarship entries. Checked after .strip().lower() on the matched
# name so leading/trailing whitespace variants are also caught.
_SKIP_NAMES = {"name of scholarship"}


def _find_scholarship_source() -> Path | None:
    """Locate the extracted scholarship text file by filename keyword.

    Globs the extracted directory for any .txt file whose stem contains
    the word "scholarship" (case-insensitive), matching the same pattern
    used by _find_supervisor_source() and _find_shuttle_source(). Returns
    the first alphabetical match, or None if no file is found.
    """
    if not _EXTRACTED_DIR.exists():
        return None
    for txt_file in sorted(_EXTRACTED_DIR.glob("*.txt")):
        if "scholarship" in txt_file.stem.lower():
            return txt_file
    return None


def parse() -> list[dict]:
    """Parse the extracted scholarship list into structured rows.

    Processing steps:
      1. Replace pipe characters with newlines so that the PDF extractor's
         collapsed table cells (rows 26–39 in the source) are split into
         individual lines rather than remaining as one continuous run.
      2. Collapse runs of spaces and tabs to a single space per line while
         preserving newlines, so "26.   Name" becomes "26. Name" and the
         regex \s{1,4} matches correctly after normalisation.
      3. Run findall with _NUMBERED_ENTRY on the whole cleaned text to
         locate every numbered entry regardless of the original line
         structure — the regex's non-dotall (.+) naturally stops at each
         newline, so each entry is captured independently.
      4. Skip entries whose name matches _SKIP_NAMES (the spurious column
         header at position 47).
      5. Deduplicate by (number, name) key to guard against any entry
         appearing in both the plain-text section and the collapsed table.
      6. Sort by the integer value of the number field before returning.
    """
    src = _find_scholarship_source()
    if src is None:
        print(
            "[scholarship_builder] Warning: no scholarship file found "
            "in extracted dir — skipping"
        )
        return []

    text = src.read_text(encoding="utf-8", errors="ignore")

    # Split collapsed markdown table cells into individual lines so that
    # entries sandwiched between '|' characters are each on their own line.
    text = text.replace("|", "\n")

    # Normalise intra-line whitespace (spaces/tabs only; newlines stay so
    # the non-dotall regex still splits on line boundaries).
    text = re.sub(r"[ \t]+", " ", text)

    matches = _NUMBERED_ENTRY.findall(text)

    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    for num_str, raw_name in matches:
        name = raw_name.strip()
        if not name or name.lower() in _SKIP_NAMES:
            continue
        key = (num_str, name)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"number": num_str, "name": name, "category": ""})

    rows.sort(key=lambda r: int(r["number"]))
    print(
        f"[scholarship_builder] Parsed {len(rows)} scholarship rows from {src.name}"
    )
    return rows


def build() -> pd.DataFrame:
    """Build scholarships.csv from the extracted corpus."""
    print("=" * 60)
    print("Building scholarships.csv...")

    rows = parse()

    if not rows:
        print(
            "[scholarship_builder] No scholarships parsed — "
            "creating empty CSV with headers only."
        )
        df = pd.DataFrame(columns=["number", "name", "category"])
    else:
        df = pd.DataFrame(rows)

    _SCHOLARSHIP_CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_SCHOLARSHIP_CSV_OUTPUT, index=False, encoding="utf-8")
    print(
        f"[scholarship_builder] CSV written → "
        f"{_SCHOLARSHIP_CSV_OUTPUT} ({len(df)} rows)"
    )

    return df


def load() -> pd.DataFrame:
    """Load scholarships.csv."""
    if not _SCHOLARSHIP_CSV_OUTPUT.exists():
        print(
            f"[scholarship_builder] {_SCHOLARSHIP_CSV_OUTPUT} not found "
            "— run build() first"
        )
        return pd.DataFrame(columns=["number", "name", "category"])

    return pd.read_csv(
        _SCHOLARSHIP_CSV_OUTPUT, encoding="utf-8", keep_default_na=False
    )
