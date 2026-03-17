#!/usr/bin/env python3
"""
Copies load_signal_batch and save_signal_batch from app.py into daily_signal.py.
Run from ~/vix_suite/
"""
import sys
import shutil
from datetime import datetime
from pathlib import Path

DS = Path("daily_signal.py")
APP = Path("app.py")

if not DS.exists() or not APP.exists():
    print("ERROR: run from ~/vix_suite/")
    sys.exit(1)

# Extract load_signal_batch and save_signal_batch functions from app.py
app_src = APP.read_text()
lines = app_src.splitlines()

def extract_function(src_lines, func_name):
    """Extract a top-level function definition from source lines."""
    result = []
    inside = False
    for line in src_lines:
        if line.startswith(f"def {func_name}("):
            inside = True
        if inside:
            result.append(line)
            # End when we hit next top-level def/class (non-empty, non-indented)
            if result and len(result) > 1 and line and not line[0].isspace() and line != result[0]:
                result.pop()  # remove the next function's first line
                break
    return "\n".join(result)

load_fn = extract_function(lines, "load_signal_batch")
save_fn = extract_function(lines, "save_signal_batch")

if not load_fn:
    print("ERROR: could not extract load_signal_batch from app.py")
    sys.exit(1)

# Backup daily_signal.py
backup = DS.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(DS, backup)
print(f"Backup: {backup}")

# Insert the two functions just before the main() function
ds_src = DS.read_text()

insert_block = f"""
# ── Batch persistence (extracted from app.py) ─────────────────────────────
{load_fn}

{save_fn}

"""

# Insert before def main():
if "def main():" not in ds_src:
    print("ERROR: could not find 'def main():' in daily_signal.py")
    sys.exit(1)

patched = ds_src.replace("def main():", insert_block + "def main():", 1)
DS.write_text(patched)
print("✅ load_signal_batch and save_signal_batch added to daily_signal.py")
print("   Test with: python3 daily_signal.py --dry-run")
