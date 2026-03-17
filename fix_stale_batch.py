#!/usr/bin/env python3
"""
fix_stale_batch.py — patches daily_signal.py stale batch check
Run from ~/vix_suite/  (Ubuntu path)
"""
import sys
import shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("daily_signal.py")

OLD = (
    "        batch_valid = (hasattr(batch, 'valid_until') and\n"
    "                       batch.valid_until.replace(tzinfo=timezone.utc) > now_utc)"
)

NEW = (
    "        batch_date  = batch.generated_at.date() if hasattr(batch, 'generated_at') else None\n"
    "        batch_valid = (\n"
    "            hasattr(batch, 'valid_until') and\n"
    "            batch.valid_until.replace(tzinfo=timezone.utc) > now_utc and\n"
    "            batch_date == date.today()   # force regen if batch is from a previous day\n"
    "        )"
)

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found — run from the repo directory")
    sys.exit(1)

src = TARGET.read_text()

if OLD not in src:
    print("ERROR: pattern not found. Showing all batch_valid lines:")
    for i, line in enumerate(src.splitlines(), 1):
        if "batch_valid" in line:
            print(f"  line {i}: {repr(line)}")
    sys.exit(1)

backup = TARGET.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(TARGET, backup)
print(f"Backup: {backup}")

patched = src.replace(OLD, NEW, 1)
TARGET.write_text(patched)
print("✅ Patched — stale batch fix applied")
