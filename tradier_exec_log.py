#!/usr/bin/env python3
"""Layer 5: Execution quality log — slippage vs mid per fill."""
import json
from datetime import datetime
from pathlib import Path

QUALITY_PATH = Path.home() / ".vix_suite" / "execution_quality.json"


def log_fill(variant: str, strike: float, expiry: str,
             action: str, mid: float, fill_price: float,
             quantity: int, order_id: str):
    data = []
    if QUALITY_PATH.exists():
        try:
            data = json.loads(QUALITY_PATH.read_text())
        except Exception:
            data = []
    data.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "variant": variant, "strike": strike, "expiry": expiry,
        "action": action, "mid": mid, "fill_price": fill_price,
        "slippage": round(fill_price - mid, 4),
        "quantity": quantity, "order_id": order_id,
    })
    QUALITY_PATH.write_text(json.dumps(data[-500:], indent=2))
