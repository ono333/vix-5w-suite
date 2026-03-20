#!/usr/bin/env python3
"""
fix_real_position_save.py
Fixes the Save Changes button in _render_paper_edit_form to also
persist changes to real_trade_log and clear the cache.
Run from ~/vix_suite/
"""
import sys, shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("app.py")
if not TARGET.exists():
    print("ERROR: run from ~/vix_suite/"); sys.exit(1)

OLD = """                st.success("✅ Position updated!")
                st.session_state[f"p_editing_{pos.position_id}"] = False
                st.rerun()"""

NEW = """                # Also persist to real_trade_log if this is a real position
                try:
                    from real_trade_log import get_real_trade_log, reset_real_trade_log_cache
                    rtl = get_real_trade_log()
                    if pos.position_id in rtl.open_positions():
                        rpos = rtl.open_positions()[pos.position_id]
                        # Update short leg strike/expiry on real position
                        if short and hasattr(rpos, 'current_short_leg') and rpos.current_short_leg:
                            rpos.current_short_leg.strike = new_short_strike
                            rpos.current_short_leg.expiration_date = new_short_exp
                            rpos.current_short_leg.entry_credit = new_short_credit
                        # Update long leg
                        rpos.long_strike = new_long_strike
                        rpos.long_expiration = new_long_exp
                        rpos.long_entry_price = new_long_price
                        rpos.contracts = new_contracts
                        rtl._save()
                        reset_real_trade_log_cache()
                except Exception as _re:
                    pass  # paper-only position, no real log entry needed
                st.success("✅ Position updated!")
                st.session_state[f"p_editing_{pos.position_id}"] = False
                st.rerun()"""

src = TARGET.read_text()
if OLD not in src:
    print("ERROR: pattern not found")
    sys.exit(1)

backup = TARGET.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy(TARGET, backup)
print(f"Backup: {backup}")

TARGET.write_text(src.replace(OLD, NEW, 1))
print("✅ Fixed — real position saves now persist to real_trade_log and clear cache")
