#!/usr/bin/env python3
"""
Adds vix_level and vix_percentile params to DiagonalPosition.roll_short()
so roll_diagonal_short() can pass them through.
Run from ~/vix_suite/
"""
import sys, subprocess
sys.path.insert(0, ".")
from safe_patch import patch

# First find exact signature
r = subprocess.run(["grep", "-n", "def roll_short", "trade_log.py"],
                   capture_output=True, text=True)
print(r.stdout)

src = open("trade_log.py").read()

# Find the roll_short signature in DiagonalPosition
# Try the most likely patterns
patterns = [
    # Pattern A — short signature
    ("        regime: str,\n        notes: str = \"\",\n        contracts:",
     "        regime: str,\n        vix_level: float = 0.0,\n        vix_percentile: float = 0.0,\n        notes: str = \"\",\n        contracts:"),
    # Pattern B — already has notes at end
    ("        regime: str,\n        notes: str = \"\",\n    ) ->",
     "        regime: str,\n        vix_level: float = 0.0,\n        vix_percentile: float = 0.0,\n        notes: str = \"\",\n    ) ->"),
]

patched = False
for old, new in patterns:
    if old in src:
        ok = patch("trade_log.py", old=old, new=new,
                   description="Add vix_level/vix_percentile to DiagonalPosition.roll_short()")
        if ok:
            patched = True
            break

if not patched:
    print("Pattern not found — showing roll_short context:")
    for i, line in enumerate(src.splitlines(), 1):
        if "def roll_short" in line:
            block = "\n".join(src.splitlines()[i-1:i+20])
            print(block)
            break
