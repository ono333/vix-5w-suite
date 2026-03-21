#!/usr/bin/env python3
"""
fix_roll_date.py
Adds Roll Date field to both paper and real roll forms.
Run from ~/vix_suite/
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
from safe_patch import patch

# ── Paper roll form ────────────────────────────────────────────────────────
# Add roll_date input before the new_strike field
patch("app.py",
    old='''    with col1:
        new_strike = st.number_input(
            "New Strike",
            min_value=1.0, value=float(suggested_strikes[1]), step=0.5,
            key=f"p_roll_new_strike_{pos.position_id}"
        )
        new_exp = st.date_input("New Expiration", value=suggested_exp.date(), key=f"p_roll_new_exp_{pos.position_id}")''',
    new='''    roll_date = st.date_input(
        "Roll Date",
        value=__import__("datetime").date.today(),
        key=f"p_roll_date_{pos.position_id}",
        help="Actual date the roll was executed — can be backdated"
    )

    with col1:
        new_strike = st.number_input(
            "New Strike",
            min_value=1.0, value=float(suggested_strikes[1]), step=0.5,
            key=f"p_roll_new_strike_{pos.position_id}"
        )
        new_exp = st.date_input("New Expiration", value=suggested_exp.date(), key=f"p_roll_new_exp_{pos.position_id}")''',
    description="Add Roll Date field to paper roll form",
)

# Pass roll_date to roll_diagonal_short
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
                    roll_date=roll_date.isoformat(),
                )''',
    description="Pass roll_date to paper roll_diagonal_short",
)

# ── Real roll form ─────────────────────────────────────────────────────────
# Add roll_date before the form columns
patch("app.py",
    old='''                        with st.form(f"r_roll_{pid}"):
                            rc1, rc2, rc3 = st.columns(3)''',
    new='''                        r_roll_date = st.date_input(
                            "Roll Date",
                            value=__import__("datetime").date.today(),
                            key=f"r_roll_date_{pid}",
                            help="Actual date the roll was executed — can be backdated"
                        )
                        with st.form(f"r_roll_{pid}"):
                            rc1, rc2, rc3 = st.columns(3)''',
    description="Add Roll Date field to real roll form",
)

# Pass roll_date to real rtl.roll_short
patch("app.py",
    old='''                                rtl.roll_short(
                                    position_id      = pid,
                                    old_exit_price   = bb_mid,
                                    old_fill_price   = bb_fill,
                                    new_strike       = ns,
                                    new_expiration   = ne.isoformat(),
                                    new_credit       = nc_mid,
                                    new_fill_price   = nc_fill,
                                    underlying_price = uvxy_px,
                                    vix_level        = roll_vix,
                                    vix_percentile   = roll_pct / 100,''',
    new='''                                rtl.roll_short(
                                    position_id      = pid,
                                    old_exit_price   = bb_mid,
                                    old_fill_price   = bb_fill,
                                    new_strike       = ns,
                                    new_expiration   = ne.isoformat(),
                                    new_credit       = nc_mid,
                                    new_fill_price   = nc_fill,
                                    underlying_price = uvxy_px,
                                    vix_level        = roll_vix,
                                    vix_percentile   = roll_pct / 100,
                                    roll_date        = r_roll_date.isoformat(),''',
    description="Pass roll_date to real rtl.roll_short",
)

print("\nNow adding roll_date param to trade_log.py and real_trade_log.py...")

# trade_log.py — add roll_date to roll_diagonal_short signature
patch("trade_log.py",
    old='''    def roll_diagonal_short(
        self,
        position_id: str,
        exit_price: float,
        new_strike: float,
        new_expiration: str,
        new_credit: float,
        underlying_price: float = 0.0,
        regime: str = "",
        vix_level: float = 0.0,
        vix_percentile: float = 0.0,
        contracts: int = None,
    )''',
    new='''    def roll_diagonal_short(
        self,
        position_id: str,
        exit_price: float,
        new_strike: float,
        new_expiration: str,
        new_credit: float,
        underlying_price: float = 0.0,
        regime: str = "",
        vix_level: float = 0.0,
        vix_percentile: float = 0.0,
        contracts: int = None,
        roll_date: str = "",
    )''',
    description="Add roll_date param to trade_log.roll_diagonal_short",
)

# real_trade_log.py — add roll_date to roll_short signature
patch("real_trade_log.py",
    old='''    def roll_short(self, position_id, old_exit_price, old_fill_price,
                   new_strike, new_expiration, new_credit, new_fill_price,
                   underlying_price=0.0, vix_level=0.0, vix_percentile=0.0,
                   reason="order_roll", notes=""):''',
    new='''    def roll_short(self, position_id, old_exit_price, old_fill_price,
                   new_strike, new_expiration, new_credit, new_fill_price,
                   underlying_price=0.0, vix_level=0.0, vix_percentile=0.0,
                   reason="order_roll", notes="", roll_date=""):''',
    description="Add roll_date param to real_trade_log.roll_short",
)

print("\nAll done. Restart Streamlit.")
