"""
shuttle_builder.py
───────────────────
Build shuttle_routes.csv from the extracted shuttle-route text file.

Unlike programs.csv (parsed from raw corpus text), the shuttle route data
comes from a PDF/scan, so it is read from the EXTRACTED directory (after
ingestion/prepare_corpus.py has produced a plain-text version of it).

Output schema (one row per route PER LEG — a route with separate morning
and evening stop lists becomes two rows):

  route_id      e.g. "1", "4", "*14+3", "*17+3"
  leg           "morning" | "evening" | "both"
  timing        e.g. "7:40 a.m" (the timing printed at the top of the block;
                applies to the route as a whole, not per-leg, in the source data)
  stops_raw     original dash-separated stop text for this leg
  stops_list    JSON-encoded list of individual stop names (for CSV storage)
  notes         any special note (e.g. "combined due to shortage of drivers")
"""

import json
import re
from pathlib import Path

import pandas as pd

from config_loader import cfg

_EXTRACTED_DIR = Path(cfg["extracted_dir"])
# Derive output path from the existing programs CSV location so no new
# config key is strictly required. If you add a dedicated cfg key later
# (e.g. cfg["shuttle_csv_output"]), prefer that instead.
_SHUTTLE_CSV_OUTPUT = Path(cfg["csv_output"]).parent / "shuttle_routes.csv"

# A line that is ONLY a route number (optionally prefixed with * and/or
# suffixed with +N for combined routes), e.g. "1", "*14+3", "17".
_LONE_ROUTE_NUMBER = re.compile(r"^\*?\d+(?:\+\d+)?\s*$")

# A line that looks like a timing, e.g. "7:40 a.m", "7:00 a.m."
_TIMING_LINE = re.compile(r"^\d{1,2}:\d{2}\s*[ap]\.?\s*m\.?$", re.IGNORECASE)

_MORNING_MARKER = re.compile(r"ONLY\s+MORNING\s*:", re.IGNORECASE)
_EVENING_MARKER = re.compile(r"ONLY\s+EVENING\s*:", re.IGNORECASE)

_WHITESPACE = re.compile(r"\s+")

# En-dash always splits (the source's standard stop separator, regardless of
# spacing). Plain hyphen ONLY splits when flanked by whitespace on both
# sides (i.e. used as a literal separator like "Foo - Bar") — NOT when
# embedded inside a compound name/code such as "E-Complex" or "5C-4", which
# must stay intact as a single stop name.
_DASH_SPLIT = re.compile(r"\s*–\s*|\s+-\s+")


def _find_shuttle_source() -> Path | None:
    """Locate the extracted shuttle-route text file by filename hint."""
    if not _EXTRACTED_DIR.exists():
        return None
    for txt_file in sorted(_EXTRACTED_DIR.glob("*.txt")):
        if "shuttle" in txt_file.stem.lower():
            return txt_file
    return None


def _split_stops(text: str) -> list[str]:
    """Split a dash-separated stop list into clean individual stop names.

    Collapses all whitespace (including newlines from line-wrapped source
    text, e.g. "NEDUET \n(Main Campus)") into single spaces BEFORE splitting,
    so a stop name that wraps across a line in the source never ends up with
    an embedded newline in stops_list.
    """
    text = _WHITESPACE.sub(" ", text).strip().rstrip(".").strip()
    if not text:
        return []
    parts = _DASH_SPLIT.split(text)
    cleaned = [p.strip().rstrip(".").strip() for p in parts]
    return [c for c in cleaned if c]


def _parse_block(route_id: str, timing: str, body_lines: list[str]) -> list[dict]:
    """Parse one route's body text into 1-2 rows (legs)."""
    body_text = "\n".join(body_lines).strip()
    notes = ""
    if route_id.startswith("*"):
        notes = "Route combined with another due to shortage of drivers"

    has_morning = bool(_MORNING_MARKER.search(body_text))
    has_evening = bool(_EVENING_MARKER.search(body_text))

    rows: list[dict] = []

    if has_morning or has_evening:
        # Split body into segments by marker
        segments = re.split(r"ONLY\s+(?:MORNING|EVENING)\s*:", body_text, flags=re.IGNORECASE)
        markers = re.findall(r"ONLY\s+(MORNING|EVENING)\s*:", body_text, flags=re.IGNORECASE)
        # segments[0] is any preamble before the first marker (usually empty);
        # segments[1:] line up with markers
        for marker, seg in zip(markers, segments[1:]):
            leg = marker.lower()
            stops_raw = seg.strip()
            stops_list = _split_stops(stops_raw)
            if not stops_list:
                continue
            rows.append({
                "route_id":   route_id,
                "leg":        leg,
                "timing":     timing,
                "stops_raw":  stops_raw,
                "stops_list": json.dumps(stops_list),
                "notes":      notes,
            })
    else:
        # No morning/evening split — applies all day
        stops_list = _split_stops(body_text)
        if stops_list:
            rows.append({
                "route_id":   route_id,
                "leg":        "both",
                "timing":     timing,
                "stops_raw":  body_text,
                "stops_list": json.dumps(stops_list),
                "notes":      notes,
            })

    return rows


def parse() -> list[dict]:
    """Parse the extracted shuttle route text file into structured rows."""
    src = _find_shuttle_source()
    if src is None:
        print("[shuttle_builder] Warning: no shuttle route file found in extracted dir — skipping")
        return []

    lines = src.read_text(encoding="utf-8", errors="ignore").split("\n")

    # Find boundaries: lines that are ONLY a route number
    boundaries = [i for i, ln in enumerate(lines) if _LONE_ROUTE_NUMBER.match(ln.strip())]
    if not boundaries:
        print(f"[shuttle_builder] Warning: no route-number boundaries found in {src.name}")
        return []

    all_rows: list[dict] = []
    for n, start in enumerate(boundaries):
        end = boundaries[n + 1] if n + 1 < len(boundaries) else len(lines)
        route_id = lines[start].strip()
        block_lines = [ln for ln in lines[start + 1:end] if ln.strip()]

        timing = ""
        body_start = 0
        if block_lines and _TIMING_LINE.match(block_lines[0].strip()):
            timing = block_lines[0].strip()
            body_start = 1

        body_lines = block_lines[body_start:]
        rows = _parse_block(route_id, timing, body_lines)
        all_rows.extend(rows)

    print(f"[shuttle_builder] Parsed {len(all_rows)} route-leg rows from {src.name}")
    return all_rows


def build() -> pd.DataFrame:
    """Build shuttle_routes.csv from the extracted corpus."""
    print("=" * 60)
    print("Building shuttle_routes.csv...")

    rows = parse()

    if not rows:
        print("[shuttle_builder] No routes parsed — creating empty CSV with headers only.")
        df = pd.DataFrame(columns=["route_id", "leg", "timing", "stops_raw", "stops_list", "notes"])
    else:
        df = pd.DataFrame(rows)

    _SHUTTLE_CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_SHUTTLE_CSV_OUTPUT, index=False, encoding="utf-8")
    print(f"[shuttle_builder] CSV written → {_SHUTTLE_CSV_OUTPUT} ({len(df)} rows)")

    return df


def load() -> pd.DataFrame:
    """Load shuttle_routes.csv, parsing stops_list back into real lists."""
    if not _SHUTTLE_CSV_OUTPUT.exists():
        print(f"[shuttle_builder] {_SHUTTLE_CSV_OUTPUT} not found — run build() first")
        return pd.DataFrame(columns=["route_id", "leg", "timing", "stops_raw", "stops_list", "notes"])

    df = pd.read_csv(_SHUTTLE_CSV_OUTPUT, encoding="utf-8", keep_default_na=False)
    if "stops_list" in df.columns:
        df["stops_list"] = df["stops_list"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s.strip().startswith("[") else []
        )
    return df