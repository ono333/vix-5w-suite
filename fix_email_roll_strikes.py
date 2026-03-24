#!/usr/bin/env python3
"""Fix roll planning strikes in daily_signal.py email builder. Run from ~/vix_suite/"""
import sys
sys.path.insert(0, ".")
from safe_patch import patch

patch("daily_signal.py",
    old='''            roll_conservative = round(vix_level * 1.02)
            roll_moderate     = round(vix_level * 1.05)
            roll_aggressive   = round(vix_level * 1.10)
            rd_cons = _estimate_roll_debit(vix_level, cur_strike, roll_conservative)
            rd_mod  = _estimate_roll_debit(vix_level, cur_strike, roll_moderate)
            rd_agg  = _estimate_roll_debit(vix_level, cur_strike, roll_aggressive)''',
    new='''            # Anchor to max(uvxy, short_strike) — ITM shorts must roll above current strike
            _roll_base    = max(vix_level, float(cur_strike))
            roll_conservative = round(_roll_base * 1.02)
            roll_moderate     = round(_roll_base * 1.05)
            roll_aggressive   = round(_roll_base * 1.10)
            rd_cons = _estimate_roll_debit(vix_level, cur_strike, roll_conservative)
            rd_mod  = _estimate_roll_debit(vix_level, cur_strike, roll_moderate)
            rd_agg  = _estimate_roll_debit(vix_level, cur_strike, roll_aggressive)''',
    description="Fix email roll strikes to anchor at max(uvxy, short_strike)",
)
print("Done. Restart Streamlit.")
