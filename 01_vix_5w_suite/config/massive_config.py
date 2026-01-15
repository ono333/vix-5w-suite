"""
Massive API configuration for VIX / UVXY Suite.

This version replaces the old dict-based config with a proper dataclass.
It also fixes the API base URL and ensures MassiveConfig is used
correctly by massive_client.py.

Supports fetching API key from macOS Keychain.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =====================================================================
# Keychain Access Helper
# =====================================================================

def get_api_key_from_keychain(
    service: str = "MASSIVE_API_KEY",
    account: str = "MASSIVE_API_KEY"
) -> Optional[str]:
    """
    Retrieve API key from macOS Keychain.
    
    To add key to keychain:
        security add-generic-password -s MASSIVE_API_KEY -a MASSIVE_API_KEY -w 'your-api-key'
    
    Falls back to environment variable if keychain access fails.
    """
    # Try macOS Keychain first
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                [
                    "security", "find-generic-password",
                    "-s", service,
                    "-a", account,
                    "-w"  # Output password only
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass  # Fall through to env var
    
    # Fallback to environment variable
    env_key = os.environ.get("MASSIVE_API_KEY", "").strip()
    return env_key if env_key else None


# =====================================================================
# Dataclass for Massive settings
# =====================================================================

@dataclass
class MassiveConfig:
    api_key: str
    base_url: str = "https://api.massive.app"
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
# Load configuration from environment / keychain
# =====================================================================

def load_massive_config() -> MassiveConfig:
    """
    Reads API key from macOS Keychain (preferred) or environment variables.

    Keychain setup:
        security add-generic-password -s MASSIVE_API_KEY -a MASSIVE_API_KEY -w 'your-key'
    
    Environment fallback:
        export MASSIVE_API_KEY="your key"
        
    Optional:
        export MASSIVE_BASE_URL="https://something"
        export MASSIVE_CACHE_ROOT="/path/to/cache"
    """

    # Try keychain first, then environment variable
    api_key = get_api_key_from_keychain()
    
    if not api_key:
        raise RuntimeError(
            "MASSIVE_API_KEY not found. Either:\n"
            "1. Add to macOS Keychain:\n"
            "   security add-generic-password -s MASSIVE_API_KEY -a MASSIVE_API_KEY -w 'your-key'\n"
            "2. Set environment variable:\n"
            "   export MASSIVE_API_KEY='your-key'"
        )

    base_url = os.environ.get("MASSIVE_BASE_URL", "").strip()
    if not base_url:
        base_url = "https://api.massive.app"

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
# These exist only so old code doesn't crash.
# The correct usage is cfg.api_key and cfg.base_url.
def _get_compat_api_key():
    key = get_api_key_from_keychain()
    return key if key else ""

API_KEY = _get_compat_api_key()
BASE_URL = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.app")


__all__ = [
    "MassiveConfig",
    "load_massive_config",
    "get_api_key_from_keychain",
    "API_KEY",
    "BASE_URL",
]