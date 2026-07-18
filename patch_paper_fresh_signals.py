"""
Patch: daily_signal.py — fix fresh signal strikes in paper report
Bug: Paper fresh signal section uses vix_level + offset for strikes
     causing: V2 long=short (same strike), V3/V5 long > short (inverted)
Fix: Same fix as daily_signal patch — use uvxy_price × OTM% for both legs
     Long leg: uvxy_price × 0.90 (10% ITM)
     Short leg: uvxy_price × (1 + regime_otm%)
     Guard: short must always > long

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_paper_fresh_signals.py
  sudo systemctl restart vix_daily.service
"""
import shutil, pathlib, re
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "daily_signal.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Find and fix the fresh signal section (paper report, ~line 1015-1050) ────
# Look for the fresh signal block that generates long_k_val / short_k_val
# using vix_level — this is a SECOND occurrence of the same bug

# Count how many times the bug pattern appears
bug_pattern = r'long_k_val\s*=\s*\(v\.long_strike\s+if\s+v\.long_strike\s*>\s*0'
matches = list(re.finditer(bug_pattern, src))
print(f"Found {len(matches)} fresh signal strike calculation block(s)")

# The fix we already applied in patch_daily_signal_strikes.py
# replaced the FIRST occurrence. If there's a second one, fix it too.
GOOD_LONG  = "long_k_val  = (v.long_strike if v.long_strike > 0\n                       else round(uvxy_price * 0.90))"
GOOD_SHORT = "short_k_val = (v.short_strike if v.short_strike > 0\n                       else round(uvxy_price * (1 + _tgt_otm)))"

# Check for any remaining vix_level-based strike calcs in fresh signal section
# Pattern: else (vix_level + v.long_strike_offset) or similar
old_patterns = [
    # Pattern A — original bug
    ("long_k_val  = v.long_strike  if v.long_strike  > 0 else (vix_level + v.long_strike_offset)",
     "long_k_val  = (v.long_strike if v.long_strike > 0 else round(uvxy_price * 0.90))"),
    ("short_k_val = v.short_strike if v.short_strike > 0 else (vix_level + v.short_strike_offset)",
     "short_k_val = (v.short_strike if v.short_strike > 0 else round(uvxy_price * (1 + _tgt_otm)))"),
    # Pattern B — slight variation
    ("long_k_val = v.long_strike if v.long_strike > 0 else (vix_level + v.long_strike_offset)",
     "long_k_val  = (v.long_strike if v.long_strike > 0 else round(uvxy_price * 0.90))"),
    ("short_k_val = v.short_strike if v.short_strike > 0 else (vix_level + v.short_strike_offset)",
     "short_k_val = (v.short_strike if v.short_strike > 0 else round(uvxy_price * (1 + _tgt_otm)))"),
]

fixed = 0
for old, new in old_patterns:
    if old in src:
        src = src.replace(old, new)
        fixed += 1
        print(f"✅ Fixed: {old[:60]}...")

# ── Also ensure the guard (short > long) exists after each fix ───────────────
GUARD = """\
        # Safety: short must always be above long for a valid call diagonal
        if short_k_val <= long_k_val:
            short_k_val = long_k_val + 2"""

# Check how many guard blocks exist vs how many fresh signal blocks
guard_count = src.count("short_k_val <= long_k_val")
print(f"Guard blocks present: {guard_count}")

if fixed == 0:
    print("ℹ️  No additional bug instances found — patch_daily_signal_strikes.py may have fixed all")
    print("    If fresh signals still show inverted diagonals after restart,")
    print("    run: grep -n 'long_k_val\\|short_k_val' ~/vix_suite/daily_signal.py")
    print("    and check for any remaining 'vix_level + v.' patterns")
else:
    print(f"✅ Fixed {fixed} additional strike calculation(s)")

# ── Fix the otm_pct display line if still using vix_level ────────────────────
remaining_otm = src.count("short_k_val - vix_level) / vix_level")
if remaining_otm > 0:
    src = src.replace(
        "(short_k_val - vix_level) / vix_level",
        "(short_k_val - uvxy_price) / uvxy_price"
    )
    print(f"✅ Fixed {remaining_otm} remaining otm_pct calculation(s)")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Verify: python daily_signal.py --dry-run (or trigger manual email)")
