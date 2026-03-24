#!/usr/bin/env python3
"""Add **kwargs to real_trade_log.roll_short() to absorb roll_date. Run from ~/vix_suite/"""
import sys, re, shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("real_trade_log.py")
src = TARGET.read_text()

# Find the closing ) of roll_short signature and add **kwargs before it
# Look for the pattern: notes: str = "" followed by ) -> 
OLD = '        notes:             str       = "",\n    ):'
NEW = '        notes:             str       = "",\n        **kwargs,\n    ):'

if OLD not in src:
    # Try alternate
    OLD = '        notes:        str = "",\n    ):'
    NEW = '        notes:        str = "",\n        **kwargs,\n    ):'

if OLD not in src:
    print("Pattern not found — showing lines 1830-1850:")
    for i, l in enumerate(src.splitlines()[1829:1850], 1830):
        print(f"  {i}: {repr(l)}")
    sys.exit(1)

from safe_patch import patch
patch("real_trade_log.py", old=OLD, new=NEW,
      description="Add **kwargs to roll_short to absorb roll_date param")
print("Done.")
