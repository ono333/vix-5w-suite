#!/usr/bin/env python3
"""Run from ~/vix_suite/ to diagnose paper trade log methods and V5 ID"""
import sys, inspect
sys.path.insert(0, ".")
from trade_log import get_trade_log

tl = get_trade_log()

print("=== All paper positions ===")
for pid, pos in tl.diagonal_positions.items():
    short = pos.current_short_leg
    print(f"  {pid}: {pos.variant_name} | "
          f"short ${short.strike if short else '—'} "
          f"exp {short.expiration_date if short else '—'}")

print("\n=== Roll/short/leg methods on TradeLog ===")
methods = [m for m in dir(tl) if any(x in m.lower() for x in ['roll','short','leg','add'])]
for m in methods:
    try:
        sig = inspect.signature(getattr(tl, m))
        print(f"  {m}{sig}")
    except:
        print(f"  {m}")
