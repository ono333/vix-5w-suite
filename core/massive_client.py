#!/usr/bin/env python3
"""
Massive API client + local caching for option chains.

- Fetches historical option chains from Massive.
- Caches each (symbol, date) chain as a pickle file under cache_dir.
- Returns a standardized pandas DataFrame:

    columns (expected):
        - trade_date : pd.Timestamp
        - expiration : pd.Timestamp
        - strike     : float
        - option_type: "C" or "P"
        - bid        : float
        - ask        : float
        - mid        : float
        - iv         : float
        - underlying : float (VIX, UVXY, etc.)
        - dte        : float (days to expiration)

You MUST adapt the `MASSIVE_OPTIONS_PATH` constant and
`parse_massive_chain_json` to match Massive's real schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import datetime as dt

import pandas as pd
import numpy as np
import requests
from requests.exceptions import RequestException


class MassiveConnectionError(RuntimeError):
    """Raised when Massive API is unreachable or returns a network error."""
    pass


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Single place to control WHICH endpoint we call for historical chains.
# Once you know the correct REST path from Massive docs, put it here
# (or override via MASSIVE_OPTIONS_PATH in your environment).
MASSIVE_OPTIONS_PATH = os.environ.get(
    "MASSIVE_OPTIONS_PATH",
    "/v1/options/historical-chains",  # <-- CHANGE THIS TO WHATEVER MASSIVE USES
)


@dataclass
class MassiveConfig:
    api_key: str
    base_url: str
    cache_dir: Path
    cache_mode: str = "full_weekly"  # "minimal", "full_weekly", "full_daily"
    timeout: int = 30                # seconds
    max_retries: int = 3
    backoff_factor: float = 0.5      # exponential backoff base


def load_massive_config() -> MassiveConfig:
    """
    Construct MassiveConfig from environment variables.

    Expected:
        export MASSIVE_API_KEY="..."
    Optional:
        export MASSIVE_BASE_URL="https://api.massive.app"
        export MASSIVE_CACHE_DIR="..."
        export MASSIVE_CACHE_MODE="full_weekly"
    """
    api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY is not set in environment.")

    base_url = os.environ.get("MASSIVE_BASE_URL", "").strip()

    # Provide a sensible default AND auto-fix old .dev host if present
    if not base_url:
        base_url = "https://api.massive.app"
    elif "massive.dev" in base_url:
        base_url = base_url.replace("massive.dev", "massive.app")

    cache_root = os.environ.get(
        "MASSIVE_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "prr_massive"),
    )

    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_mode = os.environ.get("MASSIVE_CACHE_MODE", "full_weekly")

    return MassiveConfig(
        api_key=api_key,
        base_url=base_url,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
    )


# ---------------------------------------------------------------------
# Low-level helper: call Massive and return JSON
# ---------------------------------------------------------------------


def _massive_get_json(
    config: MassiveConfig,
    path: str,
    params: dict | None = None,
    timeout_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Call Massive REST API and return parsed JSON.

    `config` is a MassiveConfig dataclass (NOT a dict).
    """
    base = config.base_url.rstrip("/")
    url = base + path

    api_key = config.api_key
    if not api_key:
        raise RuntimeError(
            "Massive API key missing. "
            "Check your ~/.zshrc or environment and MassiveConfig."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    timeout = timeout_override if timeout_override is not None else config.timeout
    max_retries = config.max_retries
    backoff = config.backoff_factor

    import time

    last_exc: Optional[RequestException] = None
    for attempt in range(max_retries):
        try:
            # Optional debug â€“ set MASSIVE_DEBUG=1 to see every call + raw text
            if os.environ.get("MASSIVE_DEBUG") == "1":
                print(
                    f"[Massive] GET {url} params={params} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

            resp = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )

            if os.environ.get("MASSIVE_DEBUG") == "1":
                print("\n===== MASSIVE RAW RESPONSE =====")
                print(f"[URL] {resp.url}")
                print(f"[STATUS] {resp.status_code}")
                print(resp.text[:2000])
                print("================================\n")

            resp.raise_for_status()
            return resp.json()

        except RequestException as e:
            last_exc = e
            if attempt < max_retries - 1:
                sleep_s = backoff * (2 ** attempt)
                time.sleep(sleep_s)
            else:
                raise MassiveConnectionError(
                    f"Massive request failed after {max_retries} attempts: {e}"
                ) from e

    if last_exc:
        raise MassiveConnectionError(str(last_exc))
    return {}


# ---------------------------------------------------------------------
# Parsing Massive response -> DataFrame
# ---------------------------------------------------------------------


def parse_massive_chain_json(raw: Any) -> pd.DataFrame:
    """
    Convert Massive's option-chain JSON into a pandas DataFrame.

    *** YOU MUST ADAPT THIS TO MATCH MASSIVE'S REAL SCHEMA. ***

    The dummy implementation expects something like:

        {
          "underlying": {"symbol": "VIX", "price": 18.23},
          "chains": [
            {
              "trade_date": "2020-01-02",
              "expiration": "2020-02-19",
              "strike": 20.0,
              "type": "call",
              "bid": 1.35,
              "ask": 1.50,
              "iv": 0.87
            },
            ...
          ]
        }
    """
    chains = raw.get("chains", [])
    if not chains:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "expiration",
                "strike",
                "option_type",
                "bid",
                "ask",
                "mid",
                "iv",
                "underlying",
                "dte",
            ]
        )

    underlying_px = float(raw.get("underlying", {}).get("price", np.nan))

    records: List[Dict[str, Any]] = []
    for row in chains:
        trade_date = pd.to_datetime(row.get("trade_date"))
        expiration = pd.to_datetime(row.get("expiration"))
        strike = float(row.get("strike"))
        opt_type_raw = str(row.get("type", "")).upper()
        option_type = "C" if "C" in opt_type_raw else "P"

        bid = float(row.get("bid", np.nan))
        ask = float(row.get("ask", np.nan))
        if np.isfinite(bid) and np.isfinite(ask) and ask >= bid:
            mid = 0.5 * (bid + ask)
        elif np.isfinite(bid):
            mid = bid
        elif np.isfinite(ask):
            mid = ask
        else:
            mid = np.nan

        iv = float(row.get("iv", np.nan))
        dte = (expiration - trade_date).days

        records.append(
            {
                "trade_date": trade_date,
                "expiration": expiration,
                "strike": strike,
                "option_type": option_type,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "iv": iv,
                "underlying": underlying_px,
                "dte": float(dte),
            }
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values(["expiration", "strike", "option_type"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------


def _cache_path_for_chain(
    config: MassiveConfig,
    symbol: str,
    trade_date: pd.Timestamp,
) -> Path:
    """
    Build a cache path for a given (symbol, trade_date).

    We use .pkl (pickle) so we don't depend on pyarrow/fastparquet.
    """
    d_str = trade_date.strftime("%Y-%m-%d")
    symbol_sanitized = symbol.replace("/", "_")
    folder = config.cache_dir / symbol_sanitized
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{symbol_sanitized}_chain_{d_str}.pkl"


def load_chain_from_cache(
    config: MassiveConfig,
    symbol: str,
    trade_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Load cached chain (if any) from a pickle file.
    Returns None if no file / load failure.
    """
    path = _cache_path_for_chain(config, symbol, trade_date)
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception as e:
        print(f"[Massive-cache] failed to load {path}: {e}")
        return None


def save_chain_to_cache(
    config: MassiveConfig,
    symbol: str,
    trade_date: pd.Timestamp,
    df: pd.DataFrame,
) -> None:
    """
    Save chain DataFrame to a pickle file.

    NOTE: we cache even empty DataFrames â€“ that still tells us
    "we already asked Massive and there was no data for this date".
    """
    path = _cache_path_for_chain(config, symbol, trade_date)
    try:
        df.to_pickle(path)
        if os.environ.get("MASSIVE_DEBUG") == "1":
            print(f"[Massive-cache] wrote {path}")
    except Exception as e:
        print(f"[Massive-cache] failed to write {path}: {e}")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def get_option_chain(
    symbol: str,
    trade_date: pd.Timestamp,
    *,
    config: Optional[MassiveConfig] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return Massive option chain for the given symbol and trade_date.

    - Checks cache first (unless force_refresh=True).
    - Otherwise hits Massive, parses, and caches.
    """
    if config is None:
        config = load_massive_config()

    trade_date = pd.to_datetime(trade_date).normalize()

    # --- CACHE CHECK ---
    if use_cache and not force_refresh:
        cached = load_chain_from_cache(config, symbol, trade_date)
        # treat even an empty DataFrame as a valid cache hit
        if cached is not None:
            return cached

    params = {
        "symbol": symbol,
        "date": trade_date.strftime("%Y-%m-%d"),
    }

    # single place where the path is used
    raw = _massive_get_json(config, MASSIVE_OPTIONS_PATH, params)
    df = parse_massive_chain_json(raw)

    if use_cache:
        save_chain_to_cache(config, symbol, trade_date, df)

    return df


def get_underlying_history(
    symbol: str,
    start: dt.date,
    end: dt.date,
) -> Optional[pd.Series]:
    """
    Request Massive underlying price history (daily).

    Returns a pandas Series indexed by date with the 'close' price,
    or None on error / no data.
    """
    config = load_massive_config()
    params = {
        "symbol": symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    try:
        raw = _massive_get_json(
            config,
            "/v1/historical/prices",  # TODO: confirm with Massive docs
            params,
            timeout_override=15,
        )
        df = pd.DataFrame(raw.get("results", []))
        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")["close"]
        return df
    except Exception as e:
        print("Massive underlying history error:", e)
        return None