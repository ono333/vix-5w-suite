#!/usr/bin/env python3
"""Add **kwargs to real_trade_log.roll_short() to absorb roll_date. Run from ~/vix_suite/"""
import sys
sys.path.insert(0, ".")
from safe_patch import patch

patch("real_trade_log.py",
    old='        notes:            str       = "",\n    ) -> Optional[RealRollRecord]:',
    new='        notes:            str       = "",\n        **kwargs,\n    ) -> Optional[RealRollRecord]:',
    description="Add **kwargs to roll_short to absorb roll_date and future params",
)
print("Done. Restart Streamlit.")
