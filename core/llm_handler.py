"""
llm_handler.py
──────────────
Thin wrapper around the local Ollama API.
"""

import ollama

from config_loader import cfg
from core.utils import extract_json

_OLLAMA_MODEL = cfg["ollama_model"]


class LocalLLM:
    def __init__(self, model_name: str = _OLLAMA_MODEL):
        self.model_name = model_name
        print(f"[llm_init] Using Ollama model: {self.model_name}")
        # Warm-up ping so any model-load delay happens at startup, not mid-chat
        try:
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
            )
            print("[llm_init] Ollama connection verified — model ready")
        except Exception as e:
            print(f"[llm_init] Warning: Ollama warm-up failed: {e}")

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 768,
        temperature: float = 0.3,
        use_json_format: bool = False,
    ) -> str:
        """
        Call the model.

        Parameters
        ----------
        messages         : standard OpenAI-style message list
        max_new_tokens   : token budget for the response
        temperature      : 0.0 = deterministic ; 0.1–0.3 = near-deterministic
        use_json_format  : if True, passes format="json" to Ollama, forcing
                           structurally valid JSON at the inference level.
        """
        try:
            kwargs = dict(
                model=self.model_name,
                messages=messages,
                options={
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                },
            )
            if use_json_format:
                kwargs["format"] = "json"

            response = ollama.chat(**kwargs)
            return response["message"]["content"].strip()
        except Exception as e:
            err_msg = str(e).lower()
            if "connect" in err_msg:
                return (
                    "I'm having trouble connecting to the AI model. "
                    "Please make sure Ollama is running and the model is available:\n"
                    "1. Run `ollama serve`\n"
                    f"2. Run `ollama pull {self.model_name}`\n"
                    "3. Try again"
                )
            return f"I'm sorry, I encountered an error: {e}"

    def chat_json(
        self,
        messages: list[dict],
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> dict | None:
        """
        Convenience: call chat with JSON format and extract the result.

        Returns a parsed JSON dict, or None if extraction fails.
        """
        raw = self.chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_json_format=True,
        )
        return extract_json(raw)
