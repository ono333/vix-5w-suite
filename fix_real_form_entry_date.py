#!/usr/bin/env python3
"""
Adds Entry Date field to the real trading form (line ~5124) in app.py.
Run from ~/vix_suite/
"""
import shutil
from datetime import datetime
from pathlib import Path

APP = Path("app.py")
src = APP.read_text()

# Add entry_date input — find the position summary block just before the button
OLD_SUMMARY = '        if st.button("📥 Record Diagonal Spread", key="manual_trade_submit"):'
NEW_SUMMARY = '        manual_entry_date = st.date_input("Entry Date", value=date.today(), key="manual_entry_date", help="Actual trade date — can be backdated")\n\n        if st.button("📥 Record Diagonal Spread", key="manual_trade_submit"):'

# Pass entry_date to open_diagonal
OLD_CALL = '''                    entry_regime="CALM",  # Default
                    entry_vix_level=0.0,  # Not specified
                    fee_per_contract=fee_per_contract,
                    notes=manual_notes,
                )'''
NEW_CALL = '''                    entry_regime="CALM",  # Default
                    entry_vix_level=0.0,  # Not specified
                    fee_per_contract=fee_per_contract,
                    notes=manual_notes,
                    entry_date=manual_entry_date.isoformat(),
                )'''

if OLD_SUMMARY not in src:
    print("ERROR: button pattern not found")
else:
    if OLD_CALL not in src:
        print("ERROR: open_diagonal call pattern not found")
    else:
        backup = APP.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy(APP, backup)
        src = src.replace(OLD_SUMMARY, NEW_SUMMARY, 1)
        src = src.replace(OLD_CALL, NEW_CALL, 1)
        APP.write_text(src)
        print("✅ Entry Date field added to real trading form")
        print("✅ entry_date passed to open_diagonal call")

print("Done. Restart Streamlit.")
