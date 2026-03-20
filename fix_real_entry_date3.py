#!/usr/bin/env python3
"""Fix remaining hardcoded entry_date in open_long_only. Run from ~/vix_suite/"""
import shutil
from datetime import datetime
from pathlib import Path

RTL = Path("real_trade_log.py")
src = RTL.read_text()

OLD = "        entry_date  = date.today().isoformat()"
NEW = "        entry_date  = (entry_date if entry_date else date.today().isoformat())"

if OLD not in src:
    print("Already patched or pattern not found")
    for i, l in enumerate(src.splitlines(), 1):
        if "entry_date" in l and "today" in l:
            print(f"  line {i}: {repr(l)}")
else:
    backup = RTL.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(RTL, backup)
    RTL.write_text(src.replace(OLD, NEW, 1))
    print("✅ Fixed open_long_only entry_date")

# Also add Entry Date field to the real position form in app.py
APP = Path("app.py")
app_src = APP.read_text()

# Find the real position entry form — look for where open_diagonal is called on rtl
OLD_APP = '''            entry_regime    = r_regime,
                    entry_vix_level = r_uvxy,
                    entry_percentile= r_pct / 100,'''
NEW_APP = '''            entry_regime    = r_regime,
                    entry_vix_level = r_uvxy,
                    entry_percentile= r_pct / 100,
                    entry_date      = r_entry_date.isoformat(),'''

# Find where the real form inputs are
OLD_FORM = '        r_regime  = st.selectbox("Entry Regime",'
NEW_FORM = '        r_entry_date = st.date_input("Entry Date", value=date.today(), key="r_entry_date", help="Actual trade date — can be backdated")\n        r_regime  = st.selectbox("Entry Regime",'

patched_app = False
if OLD_FORM in app_src:
    app_src = app_src.replace(OLD_FORM, NEW_FORM, 1)
    patched_app = True
    print("✅ Entry Date field added to real position form")
else:
    # Try alternate search
    for i, l in enumerate(app_src.splitlines(), 1):
        if 'selectbox' in l and 'Regime' in l and 'r_regime' in l:
            print(f"  Found regime selectbox at line {i}: {l.strip()}")

if OLD_APP in app_src:
    app_src = app_src.replace(OLD_APP, NEW_APP, 1)
    print("✅ entry_date passed to real open_diagonal call")
    patched_app = True
else:
    print("⚠️  real open_diagonal call pattern not found — check manually")

if patched_app:
    backup2 = APP.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(APP, backup2)
    APP.write_text(app_src)

print("\nDone. Restart Streamlit.")
