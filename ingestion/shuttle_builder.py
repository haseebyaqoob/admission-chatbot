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

FIX — empty CSV / inline parser never matching real extracted text:
The previous _parse_inline_all() produced an EMPTY shuttle_routes.csv
against the real extracted source. Root cause, confirmed against the
actual extracted .txt (see shuttle_routes.csv screenshot — header row
only, zero data rows):

  1. The source text has NO standalone route-number lines at all — every
     route number is immediately followed by its timing and stops on the
     SAME line/paragraph (the whole table collapsed into one wrapped
     block by the PDF extractor). So `_LONE_ROUTE_NUMBER` never matches
     anything, `boundaries` is empty, and `_parse_with_standalone()`
     correctly bails out and falls through to the inline path — that
     part was working as designed.

  2. `_parse_inline_all()`'s OLD strategy split the text on literal '|'
     characters into "cells" (because the extractor renders the whole
     route table as a single-row Markdown table, with every route's data
     crammed into ONE giant cell between two '|' characters). That
     produced exactly one real data "cell" — containing ALL 16+ routes
     concatenated together — plus the leading header text ("ROUTE
     TIMINGS DETAIL OF ROUTES") still attached at the front.

  3. `_ROUTE_WITH_TIMING.match(cell)` is start-anchored (`^\\s*` + the
     pattern), so it only ever tried to match at position 0 of that one
     giant cell — which starts with literal header text, not a route
     number — and never tried matching anywhere else inside the string.
     Net result: the regex never matched anything, so zero rows were
     produced and the CSV ended up with headers only, exactly as shown.

THE FIX (this version): _parse_inline_all() no longer pre-splits into
"cells" at all. Instead it scans the ENTIRE raw text for every occurrence
of a "route-number immediately followed by a timing" boundary (via
finditer, not a single start-anchored match), then treats the text
between one boundary and the next as that route's stop block. This
correctly handles the real shape of the data: an arbitrarily long run of
`<route_id> <timing> <stops...> <route_id> <timing> <stops...> ...` all
on one logical block, regardless of how many '|' or newline characters
the extractor happened to insert around it.

A new `_TRAILING_NOISE` pattern trims each route's stop-text at the first
sign of markdown table closing syntax ('|') or a long dash run (the
markdown header-separator row, e.g. "----...----"), since real stop
names in this corpus never contain a literal '|' and never contain 3+
consecutive dash characters (compound stop names like "5C-4" or "13-D"
use a single embedded hyphen, which is preserved untouched).

IMPORTANT — per explicit instruction, the EXTRACTED .txt FILE ITSELF IS
NOT MODIFIED ANYWHERE. This fix is parser-only; it adapts the code to
read the existing extracted text as-is, whatever shape the upstream
PDF-extraction step produced.

_ROUTE_WITH_TIMING (the old start-anchored single-line pattern) and
_parse_with_standalone() / the standalone-boundary path are left in
place and unchanged — they're still tried FIRST and remain correct for
any source file that DOES have genuine standalone route-number lines
(e.g. a cleaner future extraction). Only the inline fallback's internal
strategy changed.
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

# Inline pattern: route number, timing, then stops — used by the OLD
# single-cell, start-anchored matching strategy. Kept for reference/
# potential reuse, but no longer driving _parse_inline_all() directly;
# see _ROUTE_BOUNDARY below for the pattern that replaced it there.
_ROUTE_WITH_TIMING = re.compile(
    r"""
    ^\s*
    (\*?\d+(?:\+\d+)?)          # Route number: 1, 2, *14+3, *17+3
    \s+
    (\d{1,2}:\d{2}\s*[ap]\.?m\.?)  # Timing: 7:40 a.m, 7:00 a.m.
    \s+
    (.+)                         # Stops (rest of line)
    """,
    re.IGNORECASE | re.VERBOSE
)

# FIX — scan-anywhere route boundary (replaces the old cell-split +
# start-anchored-match strategy in _parse_inline_all()).
#
# Matches a "<route_id> <timing>" boundary occurring ANYWHERE in the text
# (not just at the start of a pre-split chunk), so a single block
# containing many routes back-to-back — "1 7:40 a.m <stops> 2 7:40 a.m
# <stops> ... *17+3 7:15 a.m <stops>" — yields one match per route, in
# document order, via finditer().
#
# The leading (?:^|\s) requires the route number to be preceded by either
# the very start of the text or whitespace, so a stray digit glued onto
# the end of a stop name (e.g. "...No.7" immediately followed by more
# text with no space) can't be misread as the START of a new route
# unless it is ALSO immediately followed by a recognizable "H:MM am/pm"
# timing token. In practice no stop name in this corpus is followed by a
# time-shaped string, so this combination only ever fires at genuine
# route boundaries — confirmed against the real extracted text, including
# stop names containing embedded numbers like "Korangi (No.5)" and
# "5C-4", none of which false-trigger a boundary.
_ROUTE_BOUNDARY = re.compile(
    r"""
    (?:^|\s)                          # start of text or preceding whitespace
    (\*?\d+(?:\+\d+)?)                # route number: 1, 4, *14+3, *17+3
    \s+
    (\d{1,2}:\d{2}\s*[ap]\.?\s*m\.?)  # timing: 7:40 a.m, 7:00 a.m.
    \s+
    """,
    re.IGNORECASE | re.VERBOSE,
)

# FIX — trailing markdown/table noise that can appear after the LAST
# route's stop list (table-closing '|', or the markdown header-separator
# row of repeated dashes). Real stop names in this corpus never contain a
# literal '|' and never contain a run of 3+ consecutive dash characters
# (compound names use at most one embedded hyphen, e.g. "5C-4", "13-D"),
# so truncating at the first occurrence of either is safe and never cuts
# a legitimate stop name short.
_TRAILING_NOISE = re.compile(r"[|]|-{3,}")


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


def _parse_with_standalone(lines: list[str], boundaries: list[int]) -> list[dict]:
    """Parse using standalone route-number lines (original method)."""
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

    return all_rows


def _parse_inline_route(route_id: str, timing: str, stops_text: str) -> list[dict]:
    """Parse a single inline route row into 1-2 leg rows."""
    notes = ""
    if route_id.startswith("*"):
        notes = "Route combined with another due to shortage of drivers"

    # Clean up stops_text: collapse spaces, but keep dash separators
    stops_text = _WHITESPACE.sub(" ", stops_text).strip()

    has_morning = bool(_MORNING_MARKER.search(stops_text))
    has_evening = bool(_EVENING_MARKER.search(stops_text))

    rows: list[dict] = []

    if has_morning or has_evening:
        # Split body into segments by marker
        segments = re.split(r"ONLY\s+(?:MORNING|EVENING)\s*:", stops_text, flags=re.IGNORECASE)
        markers = re.findall(r"ONLY\s+(MORNING|EVENING)\s*:", stops_text, flags=re.IGNORECASE)
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
        stops_list = _split_stops(stops_text)
        if stops_list:
            rows.append({
                "route_id":   route_id,
                "leg":        "both",
                "timing":     timing,
                "stops_raw":  stops_text,
                "stops_list": json.dumps(stops_list),
                "notes":      notes,
            })

    return rows


def _parse_inline_all(text: str) -> list[dict]:
    """Parse the entire text by scanning for route-number+timing boundaries
    anywhere in the text, regardless of line breaks or '|' characters.

    FIX (replaces the old cell-split + start-anchored-match strategy):
    The previous version split `text` on '|' into "cells" and tried
    `_ROUTE_WITH_TIMING.match(cell)` (start-anchored) on each cell. Against
    the real extracted source — where the ENTIRE multi-route table is
    rendered as one giant Markdown table cell, with leading header text
    ("ROUTE TIMINGS DETAIL OF ROUTES") still attached at the front — that
    never matched anything, because the giant cell doesn't start with a
    route number and the regex never looked anywhere past position 0.

    This version ignores '|'/newline structure entirely and instead scans
    the raw text with `_ROUTE_BOUNDARY.finditer()` to find every
    "<route_id> <timing>" occurrence in document order. The text between
    one boundary and the next (or end-of-text, for the last route) is
    that route's stop block. Each block is trimmed at the first sign of
    markdown/table noise (a literal '|' or a long dash run — see
    `_TRAILING_NOISE`) before being handed to `_parse_inline_route()`,
    which is UNCHANGED from before — it still handles morning/evening
    splitting and stop-list parsing exactly as it did previously.
    """
    matches = list(_ROUTE_BOUNDARY.finditer(text))
    if not matches:
        return []

    all_rows: list[dict] = []
    for i, m in enumerate(matches):
        route_id = m.group(1)
        timing = m.group(2)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        stops_text = text[body_start:body_end]

        # Trim at the first sign of markdown/table noise (closing '|',
        # or the header-separator dash row) so it never leaks into the
        # last stop name of a route's block.
        noise_match = _TRAILING_NOISE.search(stops_text)
        if noise_match:
            stops_text = stops_text[:noise_match.start()]
        stops_text = stops_text.strip()

        if not stops_text:
            continue

        rows = _parse_inline_route(route_id, timing, stops_text)
        all_rows.extend(rows)

    return all_rows


def parse() -> list[dict]:
    """Parse the extracted shuttle route text file into structured rows."""
    src = _find_shuttle_source()
    if src is None:
        print("[shuttle_builder] Warning: no shuttle route file found in extracted dir — skipping")
        return []

    text = src.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Try method 1: standalone route numbers
    boundaries = [i for i, ln in enumerate(lines) if _LONE_ROUTE_NUMBER.match(ln.strip())]
    if boundaries:
        print(f"[shuttle_builder] Found {len(boundaries)} standalone route-number boundaries")
        all_rows = _parse_with_standalone(lines, boundaries)
        if all_rows:
            print(f"[shuttle_builder] Parsed {len(all_rows)} route-leg rows from {src.name} (standalone)")
            return all_rows
        # If we got boundaries but no rows, fall through to inline

    # Fallback: inline pattern (FIX: now scans the whole text for
    # route-number+timing boundaries instead of splitting on '|' and
    # start-anchored matching — see _parse_inline_all()'s docstring)
    print(f"[shuttle_builder] No standalone route numbers (or empty result); trying inline pattern...")
    all_rows = _parse_inline_all(text)
    if all_rows:
        print(f"[shuttle_builder] Parsed {len(all_rows)} route-leg rows from {src.name} (inline)")
    else:
        print(f"[shuttle_builder] Warning: no routes parsed from {src.name}")
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
