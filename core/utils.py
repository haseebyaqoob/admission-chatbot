
import json
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional


# ─── JSON extraction ──────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from LLM output.
    Tries four strategies in order; returns None if all fail.

    Strategy 1: Strip markdown code fences → json.loads()
    Strategy 2: Regex-extract first {...} block → json.loads()
    Strategy 3: Trailing-comma cleanup → json.loads()
    Strategy 4: Python literal replacement (True/False/None) → json.loads()
    """
    # Strategy 1: strip markdown fences
    clean = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    clean = clean.replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first {...} block
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

        # Strategy 3: trailing comma cleanup
        no_trailing = re.sub(r",\s*([}\]])", r"\1", m.group(0))
        try:
            return json.loads(no_trailing)
        except json.JSONDecodeError:
            pass

        # Strategy 4: Python literal → JSON
        py2json = (
            no_trailing
            .replace("True",  "true")
            .replace("False", "false")
            .replace("None",  "null")
        )
        try:
            return json.loads(py2json)
        except json.JSONDecodeError:
            pass

    return None


# ─── Text cleaning ────────────────────────────────────────────────────────────

def clean_extracted_text(text: str) -> str:
    """
    Clean up common extraction artefacts from Docling output.
    - Collapse 3+ blank lines into 2
    - Remove page markers (e.g. "--- Page 3 ---")
    - Strip null bytes
    """
    text = text.replace("\x00", "")
    text = re.sub(r"---\s*Page\s*\d+\s*---", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into a single space."""
    return re.sub(r"[ \t]+", " ", text).strip()


def truncate(text: str, max_chars: int, suffix: str = " …") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


# ─── Session ID ───────────────────────────────────────────────────────────────

def new_session_id() -> str:
    """Generate a unique session ID based on timestamp + UUID fragment."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid  = uuid.uuid4().hex[:6]
    return f"{ts}_{uid}"


# ─── Timing ───────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        tag = f"[{label}] " if label else ""
        print(f"{tag}{elapsed:.3f}s")
