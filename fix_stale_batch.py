#!/usr/bin/env python3
"""
fix_stale_batch.py
Run this ONCE on Ubuntu to patch daily_signal.py in-place.

Usage:
    cd ~/PRR/01_vix_5w_suite
    python3 fix_stale_batch.py
"""
import sys
import shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("daily_signal.py")

OLD = (
    "                batch_valid = (hasattr(batch, 'valid_until') and\n"
    "                       batch.valid_until.replace(tzinfo=timezone.utc) > now_utc)"
)

NEW = (
    "                batch_date  = batch.generated_at.date() if hasattr(batch, 'generated_at') else None\n"
    "                batch_valid = (\n"
    "                    hasattr(batch, 'valid_until') and\n"
    "                    batch.valid_until.replace(tzinfo=timezone.utc) > now_utc and\n"
    "                    batch_date == date.today()   # force regen if batch is from a previous day\n"
    "                )"
)

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found. Run from ~/PRR/01_vix_5w_suite/")
    sys.exit(1)

src = TARGET.read_text()

if OLD not in src:
    print("ERROR: target pattern not found in daily_signal.py")
    print("The file may already be patched, or indentation differs.")
    print("Searching for nearby text...")
    for line in src.splitlines():
        if "batch_valid" in line:
            print(f"  FOUND: {repr(line)}")
    sys.exit(1)

# Backup
backup = TARGET.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(TARGET, backup)
print(f"Backup: {backup}")

# Patch
patched = src.replace(OLD, NEW, 1)
TARGET.write_text(patched)
print("✅ Patched daily_signal.py — stale batch fix applied")
print("   Batches will now regenerate if generated_at is from a previous day.")
