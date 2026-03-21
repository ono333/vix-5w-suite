#!/usr/bin/env python3
"""Fix missing date import in _render_paper_diagonal_entry_form. Run from ~/vix_suite/"""
import shutil
from datetime import datetime
from pathlib import Path

APP = Path("app.py")
src = APP.read_text()

OLD = '        entry_date_input = st.date_input("Entry Date", value=date.today(), key="diag_entry_date", help="Actual trade date — can be backdated")'
NEW = '        from datetime import date as _date\n        entry_date_input = st.date_input("Entry Date", value=_date.today(), key="diag_entry_date", help="Actual trade date — can be backdated")'

if OLD not in src:
    print("Pattern not found")
else:
    backup = APP.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(APP, backup)
    APP.write_text(src.replace(OLD, NEW, 1))
    print("✅ Fixed date import in paper entry form")

# Also fix real form if same issue exists
OLD_R = '        manual_entry_date = st.date_input("Entry Date", value=date.today(), key="manual_entry_date"'
NEW_R = '        from datetime import date as _date\n        manual_entry_date = st.date_input("Entry Date", value=_date.today(), key="manual_entry_date"'
src = APP.read_text()
if OLD_R in src:
    APP.write_text(src.replace(OLD_R, NEW_R, 1))
    print("✅ Fixed date import in real entry form")

print("Done. Restart Streamlit.")
