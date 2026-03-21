#!/usr/bin/env python3
"""Add Assigned button to paper trade log. Run from ~/vix_suite/"""
import sys
sys.path.insert(0, ".")
from safe_patch import patch

patch("app.py",
    old='''                        if health["short_status"] in ["expired", "roll_soon", "none"]:
                            with btn_col2:
                                if st.button(f"🔄 Roll Short", key=f"p_roll_{pos.position_id}"):
                                    st.session_state[f"p_rolling_{pos.position_id}"] = True''',
    new='''                        if health["short_status"] in ["expired", "roll_soon", "none"]:
                            with btn_col2:
                                if st.button(f"🔄 Roll Short", key=f"p_roll_{pos.position_id}"):
                                    st.session_state[f"p_rolling_{pos.position_id}"] = True
                                if st.button(f"🚨 Assigned", key=f"p_assign_{pos.position_id}",
                                             help="Short expired ITM — mark as assigned"):
                                    short = pos.current_short_leg
                                    if short:
                                        st.session_state["asgn_prefill_pid"]    = pos.position_id
                                        st.session_state["asgn_prefill_strike"] = float(short.strike)
                                        st.session_state["asgn_prefill_expiry"] = short.expiration_date
                                        short.status      = "assigned"
                                        short.exit_date   = __import__("datetime").date.today().isoformat()
                                        short.exit_price  = float(short.strike)
                                        short.exit_reason = "assigned"
                                        from trade_log import get_trade_log as _gtl
                                        _gtl()._save()
                                        st.warning(f"Short ${short.strike}C marked assigned — go to Assignment Log to close shares.")
                                        st.rerun()''',
    description="Add Assigned button to paper short leg actions",
)
print("Done. Restart Streamlit.")
