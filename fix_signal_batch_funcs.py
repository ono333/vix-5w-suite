#!/usr/bin/env python3
"""
fix_signal_batch_funcs.py
Injects clean standalone load_signal_batch/save_signal_batch into daily_signal.py.
Run from ~/vix_suite/
"""
import sys, re, shutil
from datetime import datetime
from pathlib import Path

DS = Path("daily_signal.py")
if not DS.exists():
    print("ERROR: run from ~/vix_suite/"); sys.exit(1)

CLEAN_FUNCS = '''
# ── Batch persistence ──────────────────────────────────────────────────────
import json as _json

def load_signal_batch():
    import os
    batch_path = os.path.expanduser("~/.vix_suite/current_signal_batch.json")
    if not os.path.exists(batch_path):
        return None
    try:
        with open(batch_path) as f:
            data = _json.load(f)
        from variant_generator import SignalBatch, VariantParams
        from regime_detector import RegimeState
        from enums import VolatilityRegime, VariantRole
        regime = VolatilityRegime(data["regime_state"]["regime"])
        regime_state = RegimeState(
            regime=regime,
            vix_level=data["regime_state"]["vix_level"],
            vix_percentile=data["regime_state"]["vix_percentile"],
            confidence=data["regime_state"].get("confidence", 0.7),
            vix_slope=data["regime_state"].get("vix_slope", 0.0),
        )
        variants = []
        for v in data.get("variants", []):
            try:
                role = VariantRole(v["role"])
                active = [VolatilityRegime(r) for r in v.get("active_in_regimes", [])]
                suppressed = [VolatilityRegime(r) for r in v.get("suppressed_in_regimes", [])]
                variants.append(VariantParams(
                    variant_id=v["variant_id"], name=v["name"], role=role,
                    entry_percentile=v.get("entry_percentile", 0.25),
                    long_dte_weeks=v.get("long_dte_weeks", 13),
                    short_dte_weeks=v.get("short_dte_weeks", 1),
                    long_strike_offset=v.get("long_strike_offset", 0.0),
                    short_strike_offset=v.get("short_strike_offset", 2.0),
                    long_strike=v.get("long_strike", 0.0),
                    short_strike=v.get("short_strike", 0.0),
                    active_in_regimes=active,
                    suppressed_in_regimes=suppressed,
                ))
            except Exception:
                continue
        return SignalBatch(
            batch_id=data["batch_id"],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            valid_until=datetime.fromisoformat(data["valid_until"]),
            regime_state=regime_state,
            variants=variants,
            frozen=data.get("frozen", True),
        )
    except Exception as e:
        print(f"   Could not load signal batch: {e}")
        return None


def save_signal_batch(batch):
    import os
    batch_path = os.path.expanduser("~/.vix_suite/current_signal_batch.json")
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)
    try:
        with open(batch_path, "w") as f:
            _json.dump(batch.to_dict(), f, indent=2, default=str)
    except Exception as e:
        print(f"   Could not save signal batch: {e}")

'''

backup = DS.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(DS, backup)
print(f"Backup: {backup}")

src = DS.read_text()
# Remove any previously injected batch block
src = re.sub(r"\n# ── Batch persistence.*?(?=\ndef main\(\):)", "\n", src, flags=re.DOTALL)

if "def main():" not in src:
    print("ERROR: def main(): not found"); sys.exit(1)

patched = src.replace("def main():", CLEAN_FUNCS + "def main():", 1)
DS.write_text(patched)
print("Done. Test: python3 daily_signal.py --dry-run")
