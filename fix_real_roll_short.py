#!/usr/bin/env python3
"""Add roll_date param to RealTradeLog.roll_short(). Run from ~/vix_suite/"""
import sys
sys.path.insert(0, ".")
from safe_patch import patch

patch("real_trade_log.py",
    old='    def roll_short(self, position_id, old_exit_price, old_fill_price,\n                   new_strike, new_expiration, new_credit, new_fill_price,\n                   underlying_price=0.0, vix_level=0.0, vix_percentile=0.0,\n                   reason="order_roll", notes="", roll_date=""):',
    new='    def roll_short(self, position_id, old_exit_price, old_fill_price,\n                   new_strike, new_expiration, new_credit, new_fill_price,\n                   underlying_price=0.0, vix_level=0.0, vix_percentile=0.0,\n                   reason="order_roll", notes="", roll_date="", **kwargs):',
    description="Add **kwargs to absorb roll_date and any future params",
)
print("Done. Restart Streamlit.")
