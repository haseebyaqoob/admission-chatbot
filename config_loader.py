"""
config_loader.py
─────────────────
Loads config.yaml once at import time and exposes a single `cfg` dict.
All modules import `cfg` from here instead of hardcoding values.
"""

from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Place config.yaml next to config_loader.py."
        )
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a YAML mapping at the top level.")
    return data


cfg: dict = _load()
