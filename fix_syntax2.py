#!/usr/bin/env python3
"""Fix two syntax errors in real_trade_log.py. Run from ~/vix_suite/"""
import shutil
from datetime import datetime
from pathlib import Path

RTL = Path("real_trade_log.py")
src = RTL.read_text()
original = src

# Fix 1 — open_long_only function
OLD1 = '        notes:            str      = "",\n    ,\n        entry_date: str = "") -> "RealDiagonalPosition":'
NEW1 = '        notes:            str      = "",\n        entry_date:       str      = "",\n    ) -> "RealDiagonalPosition":'

# Fix 2 — also fix the hardcoded entry_date line inside open_long_only
OLD2 = '        entry_date  = date.today().isoformat()'
NEW2 = '        entry_date  = (entry_date if entry_date else date.today().isoformat())'

fixed = 0
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    fixed += 1
    print("✅ Fixed syntax error in open_long_only signature")
else:
    print("⚠️  Pattern 1 not found — showing lines 2043-2050:")
    for i, l in enumerate(src.splitlines()[2040:2052], 2041):
        print(f"  {i}: {repr(l)}")

if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    fixed += 1
    print("✅ Fixed hardcoded entry_date in open_long_only body")

if fixed > 0 and src != original:
    backup = RTL.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(RTL, backup)
    RTL.write_text(src)
    print(f"✅ Saved — {fixed} fix(es) applied")

# Verify no more syntax errors
import py_compile, tempfile, os
tmp = tempfile.mktemp(suffix=".py")
with open(tmp, "w") as f:
    f.write(RTL.read_text())
try:
    py_compile.compile(tmp, doraise=True)
    print("✅ Syntax check PASSED")
except py_compile.PyCompileError as e:
    print(f"❌ Still has syntax error: {e}")
finally:
    os.unlink(tmp)
