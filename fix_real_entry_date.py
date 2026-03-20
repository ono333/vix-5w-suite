#!/usr/bin/env python3
"""Fix entry_date in real_trade_log.py. Run from ~/vix_suite/"""
import sys, shutil, re
from datetime import datetime
from pathlib import Path

RTL = Path("real_trade_log.py")
if not RTL.exists():
    print("ERROR: run from ~/vix_suite/"); sys.exit(1)

src = RTL.read_text()

# Fix 1: hardcoded entry_date at line ~2159
OLD1 = "            entry_date      = datetime.date.today().isoformat(),"
NEW1 = "            entry_date      = (entry_date if entry_date else datetime.date.today().isoformat()),"
if OLD1 not in src:
    print("ERROR: entry_date pattern not found")
    for i, l in enumerate(src.splitlines(), 1):
        if "entry_date" in l and "today" in l:
            print(f"  line {i}: {repr(l)}")
    sys.exit(1)

# Fix 2: add entry_date param to the open_position function signature
# Find the function containing that line
lines = src.splitlines()
for i, line in enumerate(lines):
    if OLD1.strip() in line:
        # Walk back to find the def
        for j in range(i, max(i-50, 0), -1):
            if lines[j].strip().startswith("def "):
                func_line = j
                break
        break

# Find the closing ) of that function's params
func_src = "\n".join(lines[func_line:func_line+30])
print(f"Function found:\n{func_src[:200]}\n")

# Add entry_date param before the closing ) of the signature
OLD2 = "        notes: str = \"\",\n    ):"
NEW2 = "        notes: str = \"\",\n        entry_date: str = \"\",\n    ):"
if OLD2 not in src:
    # Try alternate without double quotes
    OLD2 = "        notes: str = '',\n    ):"
    NEW2 = "        notes: str = '',\n        entry_date: str = '',\n    ):"

backup = RTL.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(RTL, backup)
print(f"Backup: {backup}")

patched = src.replace(OLD2, NEW2, 1).replace(OLD1, NEW1, 1)
RTL.write_text(patched)
print("✅ real_trade_log.py — entry_date param added")

# Also patch app.py real position form to pass entry_date
APP = Path("app.py")
app_src = APP.read_text()

# Find where real position is opened and add entry_date field + pass it
OLD_R = "            entry_date      = datetime.date.today().isoformat(),"
# This is in real_trade_log not app.py — check if app.py has a real entry form
hits = [(i+1, l) for i, l in enumerate(app_src.splitlines()) 
        if "open_real_position\|add_real\|rtl.open" in l or 
        ("rtl" in l and "open" in l.lower())]
print(f"Real position open calls in app.py: {len(hits)}")
for lineno, l in hits[:5]:
    print(f"  line {lineno}: {l.strip()}")
