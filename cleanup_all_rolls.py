#!/usr/bin/env python3
"""
Clean up duplicate roll records in V2-V5 from the IB import.
Run from ~/vix_suite/
"""
import sys
sys.path.insert(0, ".")
from trade_log import get_trade_log

tl = get_trade_log()

FIXES = {
    # V2
    "V2-20260120161227": {
        "remove":  ["V2-20260120161227-R12"],   # duplicate 2026-03-21
        "fix_old": {
            "V2-20260120161227-R13": ("44.0→49.0", "old_strike", 44.0, 42.0),
            "V2-20260120161227-R14": ("49.0→56.0", "old_strike", 49.0, 54.10),
        },
    },
    # V3
    "V3-20260112123803": {
        "remove":  [
            "V3-20260112123803-R10",  # duplicate Feb 27 roll
            "V3-20260112123803-R11",  # 2026-03-21 duplicate
        ],
        "fix_old": {
            "V3-20260112123803-R12": ("fix old_strike", "old_strike", 43.0, 38.0),
            "V3-20260112123803-R13": ("fix old_strike", "old_strike", 45.0, 42.0),
            "V3-20260112123803-R14": ("fix old_strike", "old_strike", 48.0, 54.10),
        },
    },
    # V4
    "V4-20260112123937": {
        "remove":  ["V4-20260112123937-R9"],    # duplicate 2026-03-21
        "fix_old": {
            "V4-20260112123937-R10": ("fix old_strike", "old_strike", 43.0, 42.0),
            "V4-20260112123937-R11": ("fix old_strike", "old_strike", 47.0, 47.0),
            "V4-20260112123937-R12": ("fix old_strike", "old_strike", 52.0, 54.10),
        },
    },
    # V5
    "V5-20260112124054": {
        "remove":  ["V5-20260112124054-R8"],    # duplicate 2026-03-21
        "fix_old": {
            "V5-20260112124054-R9":  ("fix old_strike", "old_strike", 45.0, 42.0),
            "V5-20260112124054-R10": ("fix old_strike", "old_strike", 49.0, 54.10),
        },
    },
}

for pid, ops in FIXES.items():
    pos = tl.diagonal_positions.get(pid)
    if not pos:
        print(f"❌ {pid} not found"); continue

    print(f"\n=== {pid} ===")

    # Remove duplicates
    before = len(pos.roll_history)
    remove_ids = ops.get("remove", [])
    pos.roll_history = [r for r in pos.roll_history
                        if r.roll_id not in remove_ids]
    print(f"  Removed {before - len(pos.roll_history)} duplicate(s): {remove_ids}")

    # Fix old_strike and underlying
    for r in pos.roll_history:
        if r.roll_id in ops.get("fix_old", {}):
            label, field, new_val, underlying = ops["fix_old"][r.roll_id]
            old_val = getattr(r, field)
            setattr(r, field, new_val)
            r.underlying_price = underlying
            print(f"  Fixed {r.roll_id}: {field} {old_val}→{new_val}, "
                  f"underlying={underlying}")

    # Renumber cleanly
    prefix = pid
    roll_num = 1
    for r in pos.roll_history:
        if "-RL" not in r.roll_id:   # preserve long roll IDs
            r.roll_id = f"{prefix}-R{roll_num}"
            roll_num += 1

    # Print final
    print(f"  Final ({len(pos.roll_history)} records):")
    for r in pos.roll_history:
        print(f"    {r.roll_id}: {r.roll_date} | "
              f"old=${r.old_strike} new=${r.new_strike} "
              f"exp={r.new_expiration} credit=${r.new_credit}")

tl._save()
print("\n✅ All positions cleaned and saved.")
