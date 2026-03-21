#!/usr/bin/env python3
"""Fix syntax error in real_trade_log.py at line ~1772. Run from ~/vix_suite/"""
import shutil
from datetime import datetime
from pathlib import Path

RTL = Path("real_trade_log.py")
src = RTL.read_text()

OLD = '        notes:             str       = "",\n    ,\n        entry_date: str = "") -> RealDiagonalPosition:'
NEW = '        notes:             str       = "",\n        entry_date:        str       = "",\n    ) -> RealDiagonalPosition:'

if OLD not in src:
    print("Pattern not found — showing lines 1768-1775:")
    for i, l in enumerate(src.splitlines()[1765:1778], 1766):
        print(f"  {i}: {repr(l)}")
else:
    backup = RTL.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(RTL, backup)
    RTL.write_text(src.replace(OLD, NEW, 1))
    print("✅ Syntax error fixed in real_trade_log.py")

print("Restart Streamlit after fix.")
