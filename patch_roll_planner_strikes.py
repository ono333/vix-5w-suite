"""
Patch: daily_signal.py — roll planner strike base fix
Bug: Line 791 uses vix_level (24.6) as base for roll strikes
     Lines 795-797 pass vix_level to _estimate_roll_debit
Fix: Use uvxy_price (50.52) as base — same root cause as fresh signal bug

Result: Roll suggestions change from $61/$63/$66 (way too wide)
        to correct $56/$57/$58 range (10-15% OTM from UVXY price)

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_roll_planner_strikes.py
  sudo systemctl restart vix_daily.service vix_alert.service
"""
import shutil, pathlib, re
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "daily_signal.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix 1: Roll base anchor ───────────────────────────────────────────────────
# Old: _roll_base = max(vix_level, float(cur_strike))
# New: _roll_base = max(uvxy_price, float(cur_strike))

OLD_BASE = "_roll_base    = max(vix_level, float(cur_strike))"
NEW_BASE = "_roll_base    = max(uvxy_price, float(cur_strike))"

if OLD_BASE in src:
    src = src.replace(OLD_BASE, NEW_BASE)
    print("✅ Fix 1: _roll_base anchor → uvxy_price")
else:
    # Try without extra spaces
    OLD_BASE2 = "_roll_base = max(vix_level, float(cur_strike))"
    NEW_BASE2 = "_roll_base = max(uvxy_price, float(cur_strike))"
    if OLD_BASE2 in src:
        src = src.replace(OLD_BASE2, NEW_BASE2)
        print("✅ Fix 1: _roll_base anchor → uvxy_price (variant spacing)")
    else:
        print("⚠️  Fix 1: Pattern not found — searching by line number context")
        m = re.search(r'_roll_base\s*=\s*max\(vix_level,\s*float\(cur_strike\)\)', src)
        if m:
            src = src[:m.start()] + m.group(0).replace("vix_level", "uvxy_price") + src[m.end():]
            print("✅ Fix 1: Applied via regex")
        else:
            print("⚠️  Fix 1: Not found — check line 791 manually")

# ── Fix 2: _estimate_roll_debit calls — pass uvxy_price not vix_level ────────
# Old: rd_cons = _estimate_roll_debit(vix_level, cur_strike, roll_conservative)
#      rd_mod  = _estimate_roll_debit(vix_level, cur_strike, roll_moderate)
#      rd_agg  = _estimate_roll_debit(vix_level, cur_strike, roll_aggressive)
# New: replace vix_level with uvxy_price in these three calls only

count = 0
for old_call, new_call in [
    ("rd_cons = _estimate_roll_debit(vix_level, cur_strike, roll_conservative)",
     "rd_cons = _estimate_roll_debit(uvxy_price, cur_strike, roll_conservative)"),
    ("rd_mod  = _estimate_roll_debit(vix_level, cur_strike, roll_moderate)",
     "rd_mod  = _estimate_roll_debit(uvxy_price, cur_strike, roll_moderate)"),
    ("rd_agg  = _estimate_roll_debit(vix_level, cur_strike, roll_aggressive)",
     "rd_agg  = _estimate_roll_debit(uvxy_price, cur_strike, roll_aggressive)"),
]:
    if old_call in src:
        src = src.replace(old_call, new_call)
        count += 1
    else:
        # Try with flexible spacing via regex
        pattern = old_call.replace("  ", r'\s+')
        m = re.search(pattern, src)
        if m:
            src = src[:m.start()] + m.group(0).replace("vix_level", "uvxy_price") + src[m.end():]
            count += 1

if count == 3:
    print("✅ Fix 2: All 3 _estimate_roll_debit calls → uvxy_price")
elif count > 0:
    print(f"⚠️  Fix 2: Only {count}/3 calls patched — check lines 795-797 manually")
else:
    print("⚠️  Fix 2: No calls found — check lines 795-797 manually")
    print("    Search for: _estimate_roll_debit(vix_level")

# ── Fix 3: Same fix for real trading report section (second code path) ────────
# The real trading section has its own roll planner — patch it too
old_r = "_estimate_roll_debit(vix_level,"
new_r = "_estimate_roll_debit(uvxy_price,"
remaining = src.count(old_r)
if remaining > 0:
    src = src.replace(old_r, new_r)
    print(f"✅ Fix 3: {remaining} additional _estimate_roll_debit(vix_level) calls patched")
else:
    print("ℹ️  Fix 3: No remaining vix_level calls in _estimate_roll_debit")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Roll planning should now show ~$56/$57/$58 at UVXY $50.52 EXTREME regime")
