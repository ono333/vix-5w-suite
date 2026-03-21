#!/usr/bin/env python3
"""Clean up V1 duplicate roll records. Run from ~/vix_suite/"""
import sys
sys.path.insert(0, ".")
from trade_log import get_trade_log

tl = get_trade_log()
pos = tl.diagonal_positions["V1-20260112111933"]

# Remove R11 (duplicate with wrong date 2026-03-21)
before = len(pos.roll_history)
pos.roll_history = [r for r in pos.roll_history if r.roll_id != "V1-20260112111933-R11"]
print(f"Removed R11: {before} → {len(pos.roll_history)} records")

# Fix R12: set old_strike=43.0 and underlying_price
for r in pos.roll_history:
    if r.roll_id == "V1-20260112111933-R12":
        r.old_strike = 43.0
        r.underlying_price = 42.0
        print(f"Fixed R12: old_strike=43.0, underlying=42.0")
    if r.roll_id == "V1-20260112111933-R13":
        r.old_strike = 47.0
        r.underlying_price = 47.0
        print(f"Fixed R13: old_strike=47.0")
    if r.roll_id == "V1-20260112111933-R14":
        r.old_strike = 52.0
        r.underlying_price = 54.10
        print(f"Fixed R14: old_strike=52.0")

# Renumber roll IDs cleanly
for i, r in enumerate(pos.roll_history, 1):
    r.roll_id = f"V1-20260112111933-R{i}"

tl._save()
print("\nFinal V1 roll history:")
for r in pos.roll_history:
    print(f"  {r.roll_id}: {r.roll_date} | "
          f"old=${r.old_strike} new=${r.new_strike} "
          f"exp={r.new_expiration} credit=${r.new_credit}")
print("\n✅ Saved.")
