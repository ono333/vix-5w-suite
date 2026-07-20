#!/usr/bin/env python3
"""apply_long_pnl_fix.py — in-place patch: expired longs realize full loss.

Run ON THE ZBOX from ~/vix_suite:
    venv/bin/python3 apply_long_pnl_fix.py
Backs up real_trade_log.py, applies 7 edits (6 try/except copies + 1 terse),
compiles, and prints the closed-position P&L check (expect ~448).
Safe to re-run: if already patched it reports 0 changes and exits.
"""
import os
import shutil
import subprocess
import sys
from datetime import datetime

TARGET = "real_trade_log.py"

A_OLD = """    @property
    def long_pnl(self) -> float:
        try:
            cur  = float(self.long_current_price or 0)
            fill = float(self.long_fill_price or 0)
        except (TypeError, ValueError):
            return 0.0
        if cur <= 0:
            return 0.0
        return (cur - fill) * self.contracts * 100"""

A_NEW = """    @property
    def long_pnl(self) -> float:
        try:
            cur  = float(self.long_current_price or 0)
            fill = float(self.long_fill_price or 0)
        except (TypeError, ValueError):
            return 0.0
        if getattr(self, "long_status", "open") == "expired":
            return (0.0 - fill) * self.contracts * 100   # expired worthless
        if cur <= 0:
            return 0.0
        return (cur - fill) * self.contracts * 100"""

B_OLD = """    @property
    def long_pnl(self) -> float:
        if self.long_current_price <= 0:
            return 0.0
        return ((self.long_current_price - self.long_fill_price)
                * self.contracts * 100)"""

B_NEW = """    @property
    def long_pnl(self) -> float:
        if getattr(self, "long_status", "open") == "expired":
            return (0.0 - self.long_fill_price) * self.contracts * 100
        if self.long_current_price <= 0:
            return 0.0
        return ((self.long_current_price - self.long_fill_price)
                * self.contracts * 100)"""


def main():
    if not os.path.exists(TARGET):
        sys.exit(f"Run this from ~/vix_suite — {TARGET} not found here.")
    src = open(TARGET).read()
    na, nb = src.count(A_OLD), src.count(B_OLD)
    if na == 0 and nb == 0:
        if 'long_status", "open") == "expired"' in src:
            print("Already patched — no changes.")
            return
        sys.exit("Patterns not found — file differs from expected. Aborting.")
    if na != 6 or nb != 1:
        print(f"WARNING expected 6+1, found A={na} B={nb}; proceeding anyway.")

    bak = f"{TARGET}.bak_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(TARGET, bak)
    src = src.replace(A_OLD, A_NEW).replace(B_OLD, B_NEW)
    open(TARGET, "w").write(src)
    print(f"backed up -> {bak}; applied A x{na}, B x{nb}")

    r = subprocess.run([sys.executable, "-m", "py_compile", TARGET])
    if r.returncode != 0:
        shutil.copy2(bak, TARGET)
        sys.exit("compile failed — restored backup.")
    print("compile OK")

    try:
        from pathlib import Path
        from real_trade_log import RealTradeLog
        r = RealTradeLog(path=Path.home() / ".vix_suite/real_trade_log_fidelity.json")
        tot = sum(p.total_pnl for p in r.diagonal_positions.values()
                  if p.status == "closed")
        print(f"closed-position P&L check: {tot:.2f}  (expect ~448)")
    except Exception as e:
        print(f"P&L check skipped: {e}")


if __name__ == "__main__":
    main()
