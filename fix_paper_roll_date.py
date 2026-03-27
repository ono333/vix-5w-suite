#!/usr/bin/env python3
"""
Fix paper roll form passing roll_date to roll_diagonal_short which doesn't accept it.
Run from ~/vix_suite/
"""
import sys
sys.path.insert(0, ".")
from safe_patch import patch

# Remove roll_date from the paper roll form call to roll_diagonal_short
patch("app.py",
    old='''                new_leg, roll = trade_log.roll_diagonal_short(
                    position_id=pos.position_id,
                    exit_price=exit_price,
                    new_strike=new_strike,
                    new_expiration=new_exp.isoformat(),
                    new_credit=new_credit,
                    underlying_price=underlying,
                    regime=pos.entry_regime,
                    vix_level=roll_vix_level,
                    vix_percentile=roll_vix_pct / 100,
                    contracts=contracts_to_roll,
                    roll_date=roll_date.isoformat(),
                )''',
    new='''                new_leg, roll = trade_log.roll_diagonal_short(
                    position_id=pos.position_id,
                    exit_price=exit_price,
                    new_strike=new_strike,
                    new_expiration=new_exp.isoformat(),
                    new_credit=new_credit,
                    underlying_price=underlying,
                    regime=pos.entry_regime,
                    vix_level=roll_vix_level,
                    vix_percentile=roll_vix_pct / 100,
                    contracts=contracts_to_roll,
                    notes=f"Roll date: {roll_date.isoformat()}",
                )''',
    description="Fix paper roll form - pass roll_date as notes instead",
)
print("Done. Restart Streamlit.")
