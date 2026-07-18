"""
Patch: daily_signal.py — fresh signal strike calculator
Bugs fixed:
  1. long_k_val / short_k_val calculated from vix_level + offset
     instead of uvxy_price → strikes land ~$53 when UVXY is $50+
  2. otm_pct denominator used vix_level instead of uvxy_price
  3. _short_strike_band() fed inconsistently (UVXY in display,
     VIX in strike math — now unified: VIX selects band, UVXY sets strikes)

Deploy:
  cd ~/vix_suite
  python patch_daily_signal_strikes.py
"""
import re, shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "daily_signal.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix 1 ─────────────────────────────────────────────────────────────────────
# Old: long_k_val  = v.long_strike  if v.long_strike  > 0 else (vix_level + v.long_strike_offset)
#      short_k_val = v.short_strike if v.short_strike > 0 else (vix_level + v.short_strike_offset)
# New: use uvxy_price as base; derive target OTM% from vix_level band

OLD_CALC = (
    "        long_k_val  = v.long_strike  if v.long_strike  > 0 "
    "else (vix_level + v.long_strike_offset)\n"
    "        short_k_val = v.short_strike if v.short_strike > 0 "
    "else (vix_level + v.short_strike_offset)"
)

NEW_CALC = """\
        # OTM target derived from VIX regime, but strikes anchored to UVXY price
        if   vix_level < 17:  _tgt_otm = 0.10
        elif vix_level <= 22: _tgt_otm = 0.07
        elif vix_level <= 25: _tgt_otm = 0.05
        elif vix_level <= 35: _tgt_otm = 0.075
        elif vix_level <= 50: _tgt_otm = 0.10
        else:                  _tgt_otm = 0.125

        # Long leg: ~10% ITM (LEAP convexity anchor), always below short
        # Short leg: target OTM band applied to current UVXY price
        long_k_val  = (v.long_strike if v.long_strike > 0
                       else round(uvxy_price * 0.90))
        short_k_val = (v.short_strike if v.short_strike > 0
                       else round(uvxy_price * (1 + _tgt_otm)))

        # Safety: short must always be above long for a valid call diagonal
        if short_k_val <= long_k_val:
            short_k_val = long_k_val + 2"""

if OLD_CALC in src:
    src = src.replace(OLD_CALC, NEW_CALC)
    print("✅ Fix 1 applied: strike base → uvxy_price")
else:
    print("⚠️  Fix 1 pattern not found — check line numbers, may need manual edit")
    print("    Looking for:\n   ", OLD_CALC[:80])

# ── Fix 2 ─────────────────────────────────────────────────────────────────────
# Old: otm_pct = round((short_k_val - vix_level) / vix_level * 100, 1)
# New: otm_pct = round((short_k_val - uvxy_price) / uvxy_price * 100, 1)

OLD_OTM = "otm_pct = round((short_k_val - vix_level) / vix_level * 100, 1)"
NEW_OTM = "otm_pct = round((short_k_val - uvxy_price) / uvxy_price * 100, 1)"

if OLD_OTM in src:
    src = src.replace(OLD_OTM, NEW_OTM)
    print("✅ Fix 2 applied: otm_pct denominator → uvxy_price")
else:
    print("⚠️  Fix 2 pattern not found")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Test: python daily_signal.py --dry-run (or restart Streamlit)")
