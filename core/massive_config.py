"""
Massive API configuration for VIX / UVXY Suite.

This version replaces the old dict-based config with a proper dataclass.
It also fixes the API base URL and ensures MassiveConfig is used
correctly by massive_client.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# =====================================================================
# Dataclass for Massive settings
# =====================================================================

@dataclass
class MassiveConfig:
    api_key: str
    base_url: str = "https://api.massive.app"    # <--- FIXED (was .dev)
    timeout: int = 30

    # Local response cache
    cache_root: Path = Path.home() / ".massive_cache"

    # Risk / option universe
    underlying_symbol: str = "VIX"
    max_contracts_per_trade: int = 10
    target_annual_irr: float = 0.25
    target_annual_iv: float = 0.30
    max_drawdown: float = 0.50

    def ensure_cache_dir(self):
        self.cache_root.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Load configuration from environment
# =====================================================================

def load_massive_config() -> MassiveConfig:
    """
    Reads environment variables and returns a MassiveConfig object.

    Expected:
        export MASSIVE_API_KEY="your key"
        optional:
        export MASSIVE_BASE_URL="https://something"
        export MASSIVE_CACHE_ROOT="/path/to/cache"
    """

    api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MASSIVE_API_KEY not set. Export it in your shell environment."
        )

    base_url = os.environ.get("MASSIVE_BASE_URL", "").strip()
    if not base_url:
        base_url = "https://api.massive.app"      # <--- FIXED DEFAULT

    cache_root = os.environ.get("MASSIVE_CACHE_ROOT")
    if cache_root:
        cache_path = Path(cache_root).expanduser()
    else:
        cache_path = Path.home() / ".massive_cache"

    cfg = MassiveConfig(
        api_key=api_key,
        base_url=base_url,
        cache_root=cache_path,
    )

    cfg.ensure_cache_dir()
    return cfg


# =====================================================================
# Backwards compatibility for old imports
# =====================================================================

# WARNING:
# These exist only so old code doesnâ€™t crash.
# The correct usage is cfg.api_key and cfg.base_url.
API_KEY = os.environ.get("MASSIVE_API_KEY", "")
BASE_URL = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.app")


__all__ = [
    "MassiveConfig",
    "load_massive_config",
    "API_KEY",
    "BASE_URL",
]