#!/usr/bin/env python3
"""
Fix roll planning strikes to use max(uvxy_price, short_strike) as base.
ITM shorts must roll ABOVE current strike, not back below UVXY spot.
Run from ~/vix_suite/
"""
import sys
sys.path.insert(0, ".")
from safe_patch import patch

# Fix 1 — real trade email roll strikes (line ~1125)
patch("app.py",
    old='''    # Calculate suggested strikes
    suggested_strikes = {
        "conservative": round(current_price * 1.02, 0),
        "moderate": round(current_price * 1.05, 0),
        "aggressive": round(current_price * 1.10, 0),
    }''',
    new='''    # Calculate suggested strikes — anchor to max(uvxy, short_strike)
    # so ITM shorts always roll ABOVE current strike, never back below it
    def _roll_strikes(uvxy, short_k=0.0):
        base = max(uvxy, short_k) if short_k else uvxy
        return {
            "conservative": round(base * 1.02, 0),
            "moderate":     round(base * 1.05, 0),
            "aggressive":   round(base * 1.10, 0),
        }
    suggested_strikes = _roll_strikes(current_price)''',
    description="Fix roll strike base to use max(uvxy, short_strike)",
)

# Fix 2 — per-position roll strikes in real email builder
# Find where roll strikes are calculated per position
patch("app.py",
    old='''        roll_conservative = round(vix_level * 1.02)
        roll_moderate     = round(vix_level * 1.05)
        roll_aggressive   = round(vix_level * 1.10)
        rd_c = _estimate_roll_debit(vix_level, cur_k, roll_cons)
        rd_m = _estimate_roll_debit(vix_level, cur_k, roll_mod)
        rd_a = _estimate_roll_debit(vix_level, cur_k, roll_agg)''',
    new='''        # Use max(uvxy, short_strike) so ITM shorts roll above current strike
        _base = max(vix_level, float(cur_k))
        roll_conservative = round(_base * 1.02)
        roll_moderate     = round(_base * 1.05)
        roll_aggressive   = round(_base * 1.10)
        rd_c = _estimate_roll_debit(vix_level, cur_k, roll_conservative)
        rd_m = _estimate_roll_debit(vix_level, cur_k, roll_moderate)
        rd_a = _estimate_roll_debit(vix_level, cur_k, roll_aggressive)''',
    description="Fix per-position roll strikes in real email",
)

# Fix 3 — paper roll form suggested strikes
patch("app.py",
    old='''    suggested_strikes = [
        round(current_price * 1.02, 0),  # 2% OTM
        round(current_price * 1.05, 0),  # 5% OTM  
        round(current_price * 1.10, 0),  # 10% OTM
    ]''',
    new='''    # Anchor to max(uvxy, short_strike) so ITM shorts roll above current strike
    _short_k = float(pos.current_short_leg.strike) if pos.current_short_leg else 0.0
    _base_price = max(current_price, _short_k)
    suggested_strikes = [
        round(_base_price * 1.02, 0),  # 2% OTM
        round(_base_price * 1.05, 0),  # 5% OTM
        round(_base_price * 1.10, 0),  # 10% OTM
    ]''',
    description="Fix paper roll form suggested strikes",
    count=2,
)

print("Done. Restart Streamlit.")
