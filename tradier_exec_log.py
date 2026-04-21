"""
tradier_exec_log.py
Layer 5 — Execution quality log.
"""
import json
from datetime import datetime
from pathlib import Path

EXEC_LOG_PATH = str(Path.home() / ".vix_suite" / "execution_quality.json")


def log_fill(variant: str, strike: float, expiry: str,
             side: str, target_mid: float, fill_price: float,
             qty: int, order_id: str = "") -> dict:
    is_sell = "STO" in side.upper() or "sell" in side.lower()
    slippage = (fill_price - target_mid) if is_sell else (target_mid - fill_price)
    entry = {
        "ts": datetime.now().isoformat(),
        "variant": variant, "strike": strike, "expiry": expiry,
        "side": side, "target_mid": round(target_mid, 4),
        "fill_price": round(fill_price, 4), "slippage": round(slippage, 4),
        "qty": qty, "order_id": order_id,
    }
    path = Path(EXEC_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        records = json.loads(path.read_text()) if path.exists() else []
    except Exception:
        records = []
    records.append(entry)
    path.write_text(json.dumps(records, indent=2))
    return entry


def slippage_summary() -> dict:
    path = Path(EXEC_LOG_PATH)
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text())
    except Exception:
        return {}
    by_variant: dict = {}
    all_slip = []
    for r in records:
        v = r.get("variant","?")
        s = r.get("slippage", 0)
        all_slip.append(s)
        by_variant.setdefault(v, []).append(s)
    def stats(vals):
        if not vals: return {}
        return {"n": len(vals), "avg": round(sum(vals)/len(vals),4),
                "min": round(min(vals),4), "max": round(max(vals),4)}
    return {"overall": stats(all_slip),
            "by_variant": {k: stats(v) for k,v in by_variant.items()}}
