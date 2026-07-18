"""
Patch: daily_signal.py — per-variant strike offsets for roll planning
Uses each variant's short_strike_offset (dollar-based) instead of
uniform regime multipliers. Preserves the strike ladder system.

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_per_variant_strikes.py
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "daily_signal.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Paper path ────────────────────────────────────────────────────────────────
OLD_PAPER = """            # Per-variant strike targets using short_strike_offset from batch
            _v_offset = getattr(variant, 'short_strike_offset', 3.0)
            _short_itm = float(cur_strike) < uvxy_price
            _roll_base = max(uvxy_price, float(cur_strike)) if _short_itm else uvxy_price
            # Conservative = offset, Moderate = offset+1, Aggressive = offset+2
            roll_conservative = round(_roll_base + _v_offset)
            roll_moderate     = round(_roll_base + _v_offset + 1)
            roll_aggressive   = round(_roll_base + _v_offset + 2)"""

# Check if already patched
if OLD_PAPER in src:
    print("Paper path already patched — skipping")
else:
    # Find and replace the multiplier-based version
    OLD_PAPER_MULT = """            # Roll base always anchored to uvxy_price unless short is ITM
            # ITM = short_strike < uvxy_price (short is being tested)
            _short_itm = float(cur_strike) < uvxy_price
            _roll_base = max(uvxy_price, float(cur_strike)) if _short_itm else uvxy_price
            # OTM multipliers derived from regime band
            _band = _short_strike_band(vix_level)
            if   vix_level < 17:  _lo, _mid, _hi = 1.09, 1.10, 1.11
            elif vix_level <= 22: _lo, _mid, _hi = 1.06, 1.07, 1.09
            elif vix_level <= 25: _lo, _mid, _hi = 1.04, 1.05, 1.06
            elif vix_level <= 35: _lo, _mid, _hi = 1.06, 1.075, 1.09
            elif vix_level <= 50: _lo, _mid, _hi = 1.08, 1.10, 1.12
            else:                  _lo, _mid, _hi = 1.10, 1.125, 1.15
            roll_conservative = round(_roll_base * _lo)
            roll_moderate     = round(_roll_base * _mid)
            roll_aggressive   = round(_roll_base * _hi)"""

    NEW_PAPER = """            # Per-variant strike targets using short_strike_offset from batch
            _v_offset = getattr(variant, 'short_strike_offset', 3.0)
            _short_itm = float(cur_strike) < uvxy_price
            _roll_base = max(uvxy_price, float(cur_strike)) if _short_itm else uvxy_price
            # Conservative = base+offset, Moderate = +1, Aggressive = +2
            roll_conservative = round(_roll_base + _v_offset)
            roll_moderate     = round(_roll_base + _v_offset + 1)
            roll_aggressive   = round(_roll_base + _v_offset + 2)"""

    if OLD_PAPER_MULT in src:
        src = src.replace(OLD_PAPER_MULT, NEW_PAPER)
        print("✅ Paper path: per-variant offsets applied")
    else:
        print("⚠️  Paper path: multiplier pattern not found — checking what's there")
        idx = src.find("roll_conservative = round(_roll_base")
        if idx > 0:
            print(repr(src[idx-300:idx+100]))

# ── Real path ─────────────────────────────────────────────────────────────────
OLD_REAL_MULT = """        # Regime-aware OTM multipliers anchored to uvxy_price
        if   vix_level < 17:  _lo, _mid, _hi = 1.09, 1.10, 1.11
        elif vix_level <= 22: _lo, _mid, _hi = 1.06, 1.07, 1.09
        elif vix_level <= 25: _lo, _mid, _hi = 1.04, 1.05, 1.06
        elif vix_level <= 35: _lo, _mid, _hi = 1.06, 1.075, 1.09
        elif vix_level <= 50: _lo, _mid, _hi = 1.08, 1.10, 1.12
        else:                  _lo, _mid, _hi = 1.10, 1.125, 1.15
        # Roll base always anchored to uvxy_price unless short is ITM
        _real_itm = float(cur_k) < uvxy_price
        _real_base = max(uvxy_price, float(cur_k)) if _real_itm else uvxy_price
        roll_cons = round(_real_base * _lo)
        roll_mod  = round(_real_base * _mid)
        roll_agg  = round(_real_base * _hi)"""

NEW_REAL = """        # Per-variant strike offsets (dollar-based)
        _v_offset_map = {
            "v1_income_harvester": 2.0,
            "v2_mean_reversion":   3.0,
            "v3_shock_absorber":   4.0,
            "v4_tail_hunter":      5.0,
            "v5_regime_allocator": 2.5,
        }
        _v_id_key = pos.variant_id.lower() if hasattr(pos, "variant_id") else v_id.lower()
        _v_off = _v_offset_map.get(_v_id_key, 3.0)
        # Roll base anchored to uvxy_price unless short is ITM
        _real_itm = float(cur_k) < uvxy_price
        _real_base = max(uvxy_price, float(cur_k)) if _real_itm else uvxy_price
        roll_cons = round(_real_base + _v_off)
        roll_mod  = round(_real_base + _v_off + 1)
        roll_agg  = round(_real_base + _v_off + 2)"""

if OLD_REAL_MULT in src:
    src = src.replace(OLD_REAL_MULT, NEW_REAL)
    print("✅ Real path: per-variant offsets applied")
else:
    print("⚠️  Real path: multiplier pattern not found — checking what's there")
    idx = src.find("roll_cons = round(_real_base")
    if idx > 0:
        print(repr(src[idx-300:idx+100]))

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Test: python daily_signal.py 2>&1 | tail -3")
