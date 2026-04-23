#!/usr/bin/env python3
"""Layer 4: Liquidity check before option order placement."""
import json
from datetime import datetime
from pathlib import Path

SKIPS_PATH = Path.home() / ".vix_suite" / "liquidity_skips.json"


def check_liquidity(bid: float, ask: float) -> tuple[bool, str]:
    if bid <= 0.10:
        return False, f"bid ${bid:.2f} ≤ $0.10"
    if ask > 0 and bid > 0:
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = spread / mid
        if spread_pct > 0.80 and spread > 2.00:
            return False, f"spread ${spread:.2f} ({spread_pct*100:.0f}%) > 80% and > $2.00"
    return True, "ok"


def log_skip(variant: str, strike: float, expiry: str,
             bid: float, ask: float, reason: str):
    data = []
    if SKIPS_PATH.exists():
        try:
            data = json.loads(SKIPS_PATH.read_text())
        except Exception:
            data = []
    data.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "variant": variant, "strike": strike, "expiry": expiry,
        "bid": bid, "ask": ask, "reason": reason,
    })
    SKIPS_PATH.write_text(json.dumps(data[-500:], indent=2))
