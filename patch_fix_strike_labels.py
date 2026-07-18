"""
Patch: daily_signal.py — fix strike label ordering
Conservative = highest strike (most cushion, least premium)
Aggressive   = lowest strike (least cushion, most premium)

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_fix_strike_labels.py
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "daily_signal.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Paper path — invert so conservative=highest, aggressive=lowest ────────────
OLD_PAPER = """            # Per-variant strike targets using short_strike_offset from batch
            _v_offset = getattr(variant, 'short_strike_offset', 3.0)
            _short_itm = float(cur_strike) < uvxy_price
            _roll_base = max(uvxy_price, float(cur_strike)) if _short_itm else uvxy_price
            # Conservative = base+offset, Moderate = +1, Aggressive = +2
            roll_conservative = round(_roll_base + _v_offset)
            roll_moderate     = round(_roll_base + _v_offset + 1)
            roll_aggressive   = round(_roll_base + _v_offset + 2)"""

NEW_PAPER = """            # Per-variant strike targets using short_strike_offset from batch
            _v_offset = getattr(variant, 'short_strike_offset', 3.0)
            _short_itm = float(cur_strike) < uvxy_price
            _roll_base = max(uvxy_price, float(cur_strike)) if _short_itm else uvxy_price
            # Conservative = HIGHEST strike (most cushion, least premium)
            # Aggressive   = LOWEST strike  (least cushion, most premium)
            roll_conservative = round(_roll_base + _v_offset + 2)
            roll_moderate     = round(_roll_base + _v_offset + 1)
            roll_aggressive   = round(_roll_base + _v_offset)"""

if OLD_PAPER in src:
    src = src.replace(OLD_PAPER, NEW_PAPER)
    print("✅ Paper path: strike order corrected")
else:
    print("⚠️  Paper path pattern not found")

# ── Real path — same inversion ────────────────────────────────────────────────
OLD_REAL = """        roll_cons = round(_real_base + _v_off)
        roll_mod  = round(_real_base + _v_off + 1)
        roll_agg  = round(_real_base + _v_off + 2)"""

NEW_REAL = """        # Conservative = HIGHEST strike (most cushion, least premium)
        # Aggressive   = LOWEST strike  (least cushion, most premium)
        roll_cons = round(_real_base + _v_off + 2)
        roll_mod  = round(_real_base + _v_off + 1)
        roll_agg  = round(_real_base + _v_off)"""

if OLD_REAL in src:
    src = src.replace(OLD_REAL, NEW_REAL)
    print("✅ Real path: strike order corrected")
else:
    print("⚠️  Real path pattern not found")

# ── Also fix OTM% display label colors to match ──────────────────────────────
# Green = conservative (highest/safest), Red = aggressive (lowest/riskiest)
# These were already correct — green=conservative, red=aggressive
# But verify the HTML rendering order matches
idx = src.find("roll_conservative:.0f}")
if idx > 0:
    snippet = src[idx-100:idx+300]
    if "🟢" in snippet and "roll_conservative" in snippet:
        print("✅ HTML: 🟢 correctly assigned to conservative (highest strike)")
    else:
        print("⚠️  HTML: check roll planning table order")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Test: python daily_signal.py 2>&1 | tail -3")
