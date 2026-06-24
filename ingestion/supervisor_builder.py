"""
supervisor_builder.py
────────────────────────
Build supervisors.csv from the extracted PhD-supervisor roster text.

Mirrors shuttle_builder.py. The roster is a single-line column table —
the same shape chunker.py's _COLUMN_ROW already detects for chunking
purposes (a leading row number, then 2+-space-delimited columns), 

This builder parses that same row shape directly into named fields
(name, designation, subject) instead of leaving it as opaque chunk text,
so core/supervisor_matcher.py can do an exact/fuzzy lookup against the
`subject` field instead of asking an LLM to read RAG chunks and decide
which rows match — see supervisor_matcher.py's module docstring for why
that distinction matters (Root cause #3: fabricated supervisor names).

Output schema (one row per supervisor):
  name          e.g. "Dr. Abdul Ghaffar Memon"
  designation   e.g. "Associate Professor"
  subject       e.g. "Engineering & Technology" (as printed in the source —
                often a broad category, not a specific research area; see
                supervisor_matcher.py for how that's handled)
"""

import re
from pathlib import Path

import pandas as pd

from config_loader import cfg

_EXTRACTED_DIR = Path(cfg["extracted_dir"])
# Derive output path from the existing programs CSV location, same
# convention as shuttle_builder.py, so no new config key is strictly
# required.
_SUPERVISOR_CSV_OUTPUT = Path(cfg["csv_output"]).parent / "supervisors.csv"

# Same row shape chunker.py's _COLUMN_ROW already detects: a leading row
# number, then 2+-space-delimited columns.
_COLUMN_ROW = re.compile(r"^\d+\s{2,}\S.*\S\s{2,}\S")
_SPLIT_COLS = re.compile(r"\s{2,}")


# supervisor_builder.py
def _find_supervisor_source():
    if not _EXTRACTED_DIR.exists():
        return None
    
    # Priority: phd_supervisors first
    for txt_file in sorted(_EXTRACTED_DIR.glob("*.txt")):
        if "phd_supervisors" in txt_file.stem.lower():
            return txt_file
    
    # Fallback: any file with "supervisor"
    for txt_file in sorted(_EXTRACTED_DIR.glob("*.txt")):
        if "supervisor" in txt_file.stem.lower():
            return txt_file
    
    return None


def parse() -> list[dict]:
    """Parse the extracted roster text file into structured supervisor rows."""
    src = _find_supervisor_source()
    if src is None:
        print("[supervisor_builder] Warning: no supervisor roster file found in extracted dir — skipping")
        return []

    lines = src.read_text(encoding="utf-8", errors="ignore").split("\n")
    rows: list[dict] = []

    for line in lines:
        stripped = line.strip()
        if not _COLUMN_ROW.match(stripped):
            continue
        cols = [c.strip() for c in _SPLIT_COLS.split(stripped) if c.strip()]
        if len(cols) < 4:
            # Not enough columns to confidently be row-number/name/
            # designation/subject — skip rather than guess at a
            # misaligned row (e.g. a wrapped continuation line that
            # coincidentally starts with a digit).
            continue
        # cols[0] = row number (not a stable id, discarded), cols[1] =
        # name, cols[2] = designation, cols[3:] = subject (rejoined in
        # case the subject text itself contained a double-space split).
        name = cols[1]
        designation = cols[2]
        subject = " ".join(cols[3:])
        rows.append({"name": name, "designation": designation, "subject": subject})

    print(f"[supervisor_builder] Parsed {len(rows)} supervisor rows from {src.name}")
    return rows


def build() -> pd.DataFrame:
    """Build supervisors.csv from the extracted corpus."""
    print("=" * 60)
    print("Building supervisors.csv...")

    rows = parse()

    if not rows:
        print("[supervisor_builder] No supervisors parsed — creating empty CSV with headers only.")
        df = pd.DataFrame(columns=["name", "designation", "subject"])
    else:
        df = pd.DataFrame(rows)

    _SUPERVISOR_CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_SUPERVISOR_CSV_OUTPUT, index=False, encoding="utf-8")
    print(f"[supervisor_builder] CSV written → {_SUPERVISOR_CSV_OUTPUT} ({len(df)} rows)")

    return df


def load() -> pd.DataFrame:
    """Load supervisors.csv."""
    if not _SUPERVISOR_CSV_OUTPUT.exists():
        print(f"[supervisor_builder] {_SUPERVISOR_CSV_OUTPUT} not found — run build() first")
        return pd.DataFrame(columns=["name", "designation", "subject"])

    return pd.read_csv(_SUPERVISOR_CSV_OUTPUT, encoding="utf-8", keep_default_na=False)
