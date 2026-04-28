#!/usr/bin/env python3
"""
vix_suite_config.py
───────────────────
Read/write ~/.vix_suite/config.json — shared runtime config for VIX Suite.

Keys:
  japan_mode (bool, default False) — lower delta alert threshold to 0.35,
                                     prepend JST time to email subjects
"""
import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".vix_suite" / "config.json"
_DEFAULTS: dict = {"japan_mode": False}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return {**_DEFAULTS, **json.loads(CONFIG_PATH.read_text())}
        except Exception:
            pass
    return dict(_DEFAULTS)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def is_japan_mode() -> bool:
    return bool(load_config().get("japan_mode", False))
