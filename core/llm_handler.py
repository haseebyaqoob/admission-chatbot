"""
core/llm_handler.py — Thin wrapper around the Ollama Python client.

Provides:
  - generate()       single-turn prompt → text
  - generate_chat()  system + user messages → text
  - JSON mode support (Ollama format="json")
  - Graceful error handling
"""

import json
import re
from typing import Optional

import ollama

from config_loader import cfg


class LLMHandler:
    def __init__(self, model: str = None):
        self.model = model or cfg["ollama_model"]
        self._warm_up()

    def _warm_up(self) -> None:
        """Ping Ollama on startup to surface connection errors early."""
        try:
            ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama. Make sure Ollama is running and "
                f"'{self.model}' is pulled.\nError: {e}"
            )

    # ─── Single-turn generation ───────────────────────────────────────────────

    def generate(
        self,
        prompt:      str,
        max_tokens:  int  = 512,
        temperature: float = 0.1,
        json_mode:   bool = False,
    ) -> str:
        """Simple single-message generation."""
        try:
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            )
            if json_mode:
                kwargs["format"] = "json"

            resp = ollama.chat(**kwargs)
            return resp["message"]["content"].strip()
        except Exception as e:
            return f"[LLM ERROR] {e}"

    # ─── Chat-style generation ────────────────────────────────────────────────

    def generate_chat(
        self,
        system:      str,
        user:        str,
        max_tokens:  int  = 1024,
        temperature: float = 0.1,
        json_mode:   bool = False,
    ) -> str:
        """System + user message generation."""
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            kwargs = dict(
                model=self.model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            )
            if json_mode:
                kwargs["format"] = "json"

            resp = ollama.chat(**kwargs)
            return resp["message"]["content"].strip()
        except Exception as e:
            return f"[LLM ERROR] {e}"


# ─── JSON extraction utility ──────────────────────────────────────────────────

def extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from LLM output.
    Tries four strategies in order.
    """
    # 1. Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # 2. Extract first {...} block
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 3. Trailing-comma cleanup
    if m:
        fixed = re.sub(r',\s*([}\]])', r'\1', m.group(0))
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # 4. Python literal → JSON (True/False/None)
    if m:
        py2json = m.group(0).replace("True", "true").replace("False", "false").replace("None", "null")
        try:
            return json.loads(py2json)
        except json.JSONDecodeError:
            pass

    return None
