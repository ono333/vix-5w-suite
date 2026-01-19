#!/usr/bin/env python3
"""
UVXY / VIX Massive auto-cache prefetcher.

Purpose
-------
Fetch Massive historical option chains for UVXY (or any symbol) across a date
range and store them in the **same local cache** your Streamlit app uses.

- Skips dates that are already cached.
- Respects Massive API key & base URL from environment:
    MASSIVE_API_KEY
    MASSIVE_BASE_URL
    MASSIVE_CACHE_DIR   (optional; defaults to ~/.cache/prr_massive)
- Shows a simple progress bar + running stats in the terminal.

Usage
-----
# Basic: UVXY weekly (Thursdays) from 2012-01-01 to today
    python prefetch_uvxy_chains.py

# Custom date range / symbol / frequency
    python prefetch_uvxy_chains.py \
        --symbol UVXY \
        --start 2012-01-01 \
        --end 2025-12-31 \
        --freq W-FRI

# Force refetch even if cache exists
    python prefetch_uvxy_chains.py --force-refresh
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import List

import numpy as np
import pandas as pd

# Adjust this import to match your project layout.
# For your repo it should be correct as written:
from core.massive_client import (
    MassiveConfig,
    load_massive_config,
    get_option_chain,
    load_chain_from_cache,
)


def _build_date_range(
    start: dt.date,
    end: dt.date,
    freq: str = "W-THU",
) -> List[pd.Timestamp]:
    """
    Build a list of dates for which we’ll request option chains.

    freq:
        - "W-THU"  -> Thursday each week
        - "W-FRI"  -> Friday each week
        - etc. (any pandas weekly alias)
    """
    idx = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        inclusive="both",
    )
    # normalize to midnight to match cache keys
    return [pd.to_datetime(d).normalize() for d in idx]


def prefetch_chains(
    symbol: str,
    start: dt.date,
    end: dt.date,
    freq: str = "W-THU",
    force_refresh: bool = False,
) -> None:
    """
    Core prefetch loop.

    For each weekly date in [start, end]:
        - if cached chain exists and not force_refresh -> skip
        - otherwise call Massive via get_option_chain (which also saves cache)
    """
    config: MassiveConfig = load_massive_config()

    dates = _build_date_range(start, end, freq=freq)
    total = len(dates)

    print(f"\n=== Massive auto-cache prefetch ===")
    print(f"Symbol          : {symbol}")
    print(f"Date range      : {start} → {end}  ({total} dates, freq={freq})")
    print(f"Cache directory : {config.cache_dir}")
    print(f"Force refresh   : {force_refresh}")
    print("====================================\n")

    if total == 0:
        print("No dates in range – nothing to do.")
        return

    hits = 0
    misses = 0
    errors = 0

    for idx, d in enumerate(dates, start=1):
        # Check cache first so we don’t hammer the API
        cached = None
        if not force_refresh:
            cached = load_chain_from_cache(config, symbol, d)

        if cached is not None and not cached.empty:
            hits += 1
            status = "cache hit"
        else:
            try:
                _ = get_option_chain(
                    symbol=symbol,
                    trade_date=d,
                    config=config,
                    use_cache=True,         # this will save to cache
                    force_refresh=force_refresh,
                )
                misses += 1
                status = "downloaded"
            except Exception as e:
                errors += 1
                status = f"ERROR: {e}"

        # simple textual progress bar
        pct = idx / total * 100.0
        sys.stdout.write(
            f"\r[{idx:4d}/{total:4d}] {pct:5.1f}%  "
            f"{d.date()}  -> {status:<11}  "
            f"(hits={hits}, misses={misses}, errors={errors})"
        )
        sys.stdout.flush()

    print("\n\nDone.")
    print(f"  Cache hits     : {hits}")
    print(f"  Downloads      : {misses}")
    print(f"  Errors         : {errors}")
    print(f"  Cache location : {config.cache_dir}")
    print("You can now run the Streamlit app; Massive chains should be served "
          "entirely from local cache for this symbol/date range.\n")


def _parse_args() -> argparse.Namespace:
    today = dt.date.today()

    parser = argparse.ArgumentParser(
        description="Prefetch Massive historical option chains into local cache."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="UVXY",
        help="Underlying symbol to fetch (default: UVXY).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2012-01-01",
        help="Start date YYYY-MM-DD (default: 2012-01-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=today.isoformat(),
        help=f"End date YYYY-MM-DD (default: {today.isoformat()}).",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="W-THU",
        help="Pandas weekly frequency alias (default: W-THU). "
             "Examples: W-WED, W-THU, W-FRI.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download even if a cached chain already exists.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
    except ValueError as e:
        print(f"Invalid date format: {e}")
        sys.exit(1)

    if start > end:
        print("Error: start date must be <= end date.")
        sys.exit(1)

    prefetch_chains(
        symbol=args.symbol.upper(),
        start=start,
        end=end,
        freq=args.freq,
        force_refresh=args.force_refresh,
    )


if __name__ == "__main__":
    main()