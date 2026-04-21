"""
tradier_liquidity.py
Layer 4 — Liquidity check before option entry.

Skip rules:
  bid <= $0.10                          -> price floor too low
  spread > 80% of mid AND > $2.00 abs  -> market too wide (both must be exceeded)
"""
import json
import os
from datetime import datetime
from pathlib import Path

MIN_BID: float = 0.10
MAX_SPREAD_PCT: float = 0.80
MAX_SPREAD_ABS: float = 2.00

SKIP_LOG_PATH = os.path.expanduser("~/.vix_suite/liquidity_skips.json")


def check_liquidity(bid: float, ask: float) -> tuple[bool, str | None]:
    if bid <= MIN_BID:
        return False, f"bid ${bid:.2f} <= ${MIN_BID:.2f} floor"
    if ask <= 0:
        return False, "ask is zero"
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return False, "mid is zero"
    spread_abs = ask - bid
    spread_pct = spread_abs / mid
    if spread_pct > MAX_SPREAD_PCT and spread_abs > MAX_SPREAD_ABS:
        return False, f"spread {spread_pct:.0%} / ${spread_abs:.2f} exceeds both limits"
    return True, None


def log_skip(variant: str, strike: float, expiry: str,
             bid: float, ask: float, reason: str):
    path = Path(SKIP_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        records = json.loads(path.read_text()) if path.exists() else []
    except Exception:
        records = []
    records.append({
        "ts": datetime.now().isoformat(),
        "variant": variant,
        "strike": strike,
        "expiry": expiry,
        "bid": bid,
        "ask": ask,
        "reason": reason,
    })
    path.write_text(json.dumps(records, indent=2))
