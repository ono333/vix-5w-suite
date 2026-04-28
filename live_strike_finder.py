#!/usr/bin/env python3
"""
live_strike_finder.py
─────────────────────
Thin wrapper around tradier_orchestrator primitives.
Fetches live short-strike recommendations for V1–V5 from the next-Friday
UVXY option chain via Tradier Sandbox.

Reuses TradierClient, find_short_strike, and next_friday from
tradier_orchestrator — no duplication.

Usage:
    from live_strike_finder import fetch_live_strikes
    results = fetch_live_strikes()
    # {"V1": {"strike": 45, "delta": 0.27, "bid": 0.65, "ask": 0.75,
    #          "mid": 0.70, "expiry": date(2026,5,8), "status": "ok"}, ...}
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

from tradier_orchestrator import TradierClient, find_short_strike, next_friday

DELTA_CEILINGS: dict[str, float] = {
    "V1": 0.28,
    "V2": 0.22,
    "V3": 0.18,
    "V4": 0.15,
    "V5": 0.22,
}

_VARIANTS = list(DELTA_CEILINGS.keys())

_BLANK = {
    "strike": None, "delta": None, "bid": None,
    "ask": None, "mid": None, "expiry": None,
}


def fetch_live_strikes(target_expiry: Optional[date] = None) -> dict[str, dict]:
    """
    Fetch best short strike per variant from Tradier Sandbox.

    Args:
        target_expiry: query this expiry; defaults to next Friday ≥ 7 DTE.

    Returns dict keyed V1–V5:
        {"strike": float, "delta": float, "bid": float, "ask": float,
         "mid": float, "expiry": date, "status": "ok"|"no_strike"|"unavailable"|...}
    """
    token = os.environ.get("TRADIER_SANDBOX_TOKEN", "")
    if not token:
        return {v: {**_BLANK, "status": "no_token"} for v in _VARIANTS}

    try:
        client = TradierClient(sandbox=True)

        if target_expiry is None:
            expirations = client.get_expirations("UVXY")
            if not expirations:
                return {v: {**_BLANK, "status": "no_expirations"} for v in _VARIANTS}
            target_expiry = next_friday(expirations, min_dte=7)

        if target_expiry is None:
            return {v: {**_BLANK, "status": "no_friday"} for v in _VARIANTS}

        chain = client.get_option_chain("UVXY", target_expiry)
        if not chain:
            return {v: {**_BLANK, "status": "empty_chain"} for v in _VARIANTS}

        results: dict[str, dict] = {}
        for vkey in _VARIANTS:
            dc   = DELTA_CEILINGS[vkey]
            best = find_short_strike(chain, dc)
            if best:
                results[vkey] = {
                    "strike": best["strike"],
                    "delta":  best["delta"],
                    "bid":    best["bid"],
                    "ask":    best["ask"],
                    "mid":    best["mid"],
                    "expiry": target_expiry,
                    "status": "ok",
                }
            else:
                results[vkey] = {**_BLANK, "expiry": target_expiry, "status": "no_strike"}

        return results

    except Exception as exc:
        err = f"error: {exc}"
        return {v: {**_BLANK, "status": err} for v in _VARIANTS}


if __name__ == "__main__":
    import json
    rows = fetch_live_strikes()
    for vk, r in rows.items():
        strike = f"${r['strike']:.0f}" if r["strike"] else "—"
        delta  = f"{r['delta']:.3f}"   if r["delta"]  else "—"
        bid    = f"${r['bid']:.2f}"    if r["bid"]    else "—"
        mid    = f"${r['mid']:.2f}"    if r["mid"]    else "—"
        expiry = str(r["expiry"])      if r["expiry"] else "—"
        print(f"{vk}  {strike:>6}  δ={delta:>5}  bid={bid:>6}  mid={mid:>6}"
              f"  exp={expiry}  [{r['status']}]")
