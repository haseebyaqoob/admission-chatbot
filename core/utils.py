"""
utils.py
─────────
Shared utility functions for the admission bot.
"""

import json
import re


def extract_json(raw: str) -> dict | None:
    """Try several strategies to extract a JSON object from raw LLM output."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$",           "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        fragment = m.group(0)
        try:
            return json.loads(fragment)
        except Exception:
            pass
        fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
        fragment = re.sub(r"\bTrue\b",  "true",  fragment)
        fragment = re.sub(r"\bFalse\b", "false", fragment)
        fragment = re.sub(r"\bNone\b",  "null",  fragment)
        try:
            return json.loads(fragment)
        except Exception:
            pass

    return None
