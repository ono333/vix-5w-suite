#!/usr/bin/env python3
"""
fix_real_entry_date2.py
Fixes entry_date in real_trade_log.py at the correct locations.
Run from ~/vix_suite/
"""
import sys, shutil, re
from datetime import datetime
from pathlib import Path

RTL = Path("real_trade_log.py")
if not RTL.exists():
    print("ERROR: run from ~/vix_suite/"); sys.exit(1)

src = RTL.read_text()
lines = src.splitlines()

# First fix the bad patch from previous run — add_short_leg has broken entry_date ref
OLD_BAD = "            entry_date      = (entry_date if entry_date else datetime.date.today().isoformat()),"
NEW_BAD = "            entry_date      = datetime.date.today().isoformat(),"
if OLD_BAD in src:
    src = src.replace(OLD_BAD, NEW_BAD, 1)
    print("✅ Reverted bad patch in add_short_leg")

# Also remove the bad entry_date param from add_short_leg signature
OLD_SIG_BAD = '        entry_date: str = "",\n    ):'
# Only remove if it's in add_short_leg context — find by proximity
bad_func_region = src.find("def add_short_leg(")
if bad_func_region != -1:
    # Check if entry_date param was added to this function
    region = src[bad_func_region:bad_func_region+300]
    if 'entry_date: str = ""' in region:
        # Remove it from this specific function
        src = src[:bad_func_region] + region.replace('        entry_date: str = "",\n', '', 1) + src[bad_func_region+300:]
        print("✅ Removed bad entry_date param from add_short_leg")

# Now find the real position creation functions at lines 1796 and 2052
# Get context around each RealDiagonalPosition() call
backup = RTL.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(RTL, backup)
print(f"Backup: {backup}")

# Find the two open/create functions by looking at the defs before each instantiation
for target_line in [1796, 2052]:
    # Find the def before this line
    func_start = None
    for i in range(target_line-2, max(target_line-80, 0), -1):
        if lines[i].startswith("    def ") or lines[i].startswith("def "):
            func_start = i
            func_name = lines[i].strip()
            break
    print(f"\nRealDiagonalPosition at ~line {target_line}, in function: {func_name}")
    # Show context
    ctx = "\n".join(lines[target_line-3:target_line+5])
    print(ctx)

# Apply fixes based on the pattern found
# Pattern: entry_date = datetime.date.today().isoformat()
count = src.count("entry_date      = datetime.date.today().isoformat()")
print(f"\nFound {count} remaining hardcoded entry_date(s)")

# Replace ALL remaining hardcoded entry_dates with the conditional
# but ONLY inside RealDiagonalPosition constructors (not add_short_leg)
# We do this by finding each RealDiagonalPosition( block and patching inside it
def patch_constructor(src, constructor_start_pattern):
    result = src
    idx = 0
    patches = 0
    while True:
        pos = result.find("RealDiagonalPosition(", idx)
        if pos == -1:
            break
        end = result.find(")", pos + 500)  # rough end
        # Find entry_date in this block
        block_end = result.find("\n        )\n", pos)
        if block_end == -1:
            block_end = result.find("\n        )", pos)
        if block_end == -1:
            idx = pos + 1
            continue
        block = result[pos:block_end]
        if "entry_date      = datetime.date.today().isoformat()" in block:
            new_block = block.replace(
                "entry_date      = datetime.date.today().isoformat()",
                "entry_date      = (entry_date if entry_date else datetime.date.today().isoformat())"
            )
            result = result[:pos] + new_block + result[block_end:]
            patches += 1
        idx = pos + 1
    return result, patches

src, n = patch_constructor(src, "RealDiagonalPosition(")
print(f"✅ Patched {n} RealDiagonalPosition constructor(s)")

# Now add entry_date param to the two open functions
# Find functions that contain RealDiagonalPosition and add entry_date param
def add_param_to_func(src, func_search_str):
    idx = src.find(func_search_str)
    if idx == -1:
        return src, False
    # Find end of params (closing paren of def line or multiline)
    # Look for ): or ) -> pattern
    func_header_end = src.find("):", idx)
    func_header_end2 = src.find(") ->", idx)
    if func_header_end2 != -1 and (func_header_end == -1 or func_header_end2 < func_header_end):
        func_header_end = func_header_end2
    if func_header_end == -1:
        return src, False
    header = src[idx:func_header_end]
    if "entry_date" in header:
        return src, False  # already has it
    # Add before closing paren
    insert = ",\n        entry_date: str = \"\""
    src = src[:func_header_end] + insert + src[func_header_end:]
    return src, True

# Find the two functions by their line numbers
lines2 = src.splitlines()
for target_line in [1796, 2052]:
    for i in range(min(target_line, len(lines2))-2, max(target_line-80, 0), -1):
        if lines2[i].startswith("    def "):
            func_def = lines2[i].strip()
            func_name_only = func_def.split("(")[0].replace("def ", "").strip()
            src, ok = add_param_to_func(src, f"    def {func_name_only}(")
            if ok:
                print(f"✅ Added entry_date param to {func_name_only}()")
            else:
                print(f"⚠️  {func_name_only}() already has entry_date or not found")
            break

RTL.write_text(src)
print("\n✅ Done. Restart Streamlit.")
