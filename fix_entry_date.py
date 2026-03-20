#!/usr/bin/env python3
"""
fix_entry_date.py
1. Adds entry_date parameter to open_diagonal() in trade_log.py and real_trade_log.py
2. Adds Entry Date field to the new position forms in app.py (paper + real)
Run from ~/vix_suite/
"""
import sys, shutil, re
from datetime import datetime, date
from pathlib import Path

def backup(p):
    b = p.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(p, b)
    print(f"Backup: {b}")

# ── 1. trade_log.py — add entry_date param to open_diagonal ───────────────
TL = Path("trade_log.py")
if TL.exists():
    src = TL.read_text()
    OLD1 = '        fee_per_contract: float = 0.65,\n        notes: str = "",\n    ) -> DiagonalPosition:'
    NEW1 = '        fee_per_contract: float = 0.65,\n        notes: str = "",\n        entry_date: str = "",\n    ) -> DiagonalPosition:'
    OLD2 = '            entry_date=datetime.now().strftime("%Y-%m-%d"),'
    NEW2 = '            entry_date=(entry_date if entry_date else datetime.now().strftime("%Y-%m-%d")),'
    if OLD1 in src and OLD2 in src:
        backup(TL)
        src = src.replace(OLD1, NEW1, 1).replace(OLD2, NEW2, 1)
        TL.write_text(src)
        print("✅ trade_log.py — entry_date param added to open_diagonal()")
    elif OLD1 not in src:
        print("⚠️  trade_log.py — param block not found, checking alternate pattern")
        # Try alternate — notes might be last without fee
        OLD1b = '        notes: str = "",\n    ) -> DiagonalPosition:'
        NEW1b = '        notes: str = "",\n        entry_date: str = "",\n    ) -> DiagonalPosition:'
        if OLD1b in src and OLD2 in src:
            backup(TL)
            src = src.replace(OLD1b, NEW1b, 1).replace(OLD2, NEW2, 1)
            TL.write_text(src)
            print("✅ trade_log.py — entry_date param added (alternate pattern)")
        else:
            print("❌ trade_log.py — could not patch, check manually")
    else:
        print("⚠️  trade_log.py — entry_date line not found")
else:
    print("⚠️  trade_log.py not found")

# ── 2. real_trade_log.py — same fix ───────────────────────────────────────
RTL = Path("real_trade_log.py")
if RTL.exists():
    src = RTL.read_text()
    patched = False
    for old_ed in [
        'entry_date=datetime.now().strftime("%Y-%m-%d"),',
        "entry_date=datetime.now().strftime('%Y-%m-%d'),",
        'entry_date=dt.datetime.now().strftime("%Y-%m-%d"),',
    ]:
        if old_ed in src:
            new_ed = old_ed.replace(
                'datetime.now().strftime("%Y-%m-%d")',
                '(entry_date if entry_date else datetime.now().strftime("%Y-%m-%d"))'
            ).replace(
                "datetime.now().strftime('%Y-%m-%d')",
                "(entry_date if entry_date else datetime.now().strftime('%Y-%m-%d'))"
            ).replace(
                'dt.datetime.now().strftime("%Y-%m-%d")',
                '(entry_date if entry_date else dt.datetime.now().strftime("%Y-%m-%d"))'
            )
            # Add entry_date param to the function signature too
            for sig_old in [
                '        notes: str = "",\n    ):',
                '        fee_per_contract: float = 0.65,\n        notes: str = "",\n    ):',
                '        notes: str = "",\n    ) -> ',
                '        fee_per_contract: float = 0.65,\n        notes: str = "",\n    ) -> ',
            ]:
                if sig_old in src:
                    sig_new = sig_old.replace('        notes: str = "",', 
                                              '        notes: str = "",\n        entry_date: str = "",')
                    src = src.replace(sig_old, sig_new, 1)
                    break
            backup(RTL)
            src = src.replace(old_ed, new_ed, 1)
            RTL.write_text(src)
            print("✅ real_trade_log.py — entry_date param added")
            patched = True
            break
    if not patched:
        print("⚠️  real_trade_log.py — entry_date hardcode pattern not found")
        for i, line in enumerate(src.splitlines(), 1):
            if "entry_date" in line and ("datetime" in line or "now()" in line):
                print(f"  line {i}: {repr(line)}")

# ── 3. app.py — add Entry Date field to paper new position form ───────────
APP = Path("app.py")
if APP.exists():
    src = APP.read_text()

    # Paper form — add date input before the Long Leg section
    OLD_P = '    st.markdown("##### Long Leg")\n    lcol1, lcol2, lcol3 = st.columns(3)\n    with lcol1:\n        long_strike = st.number_input("Long Strike", min_value=1.0, value=40.0, step=0.5, key="diag_long_strike")'
    NEW_P = '    entry_date_input = st.date_input("Entry Date", value=date.today(), key="diag_entry_date", help="Actual trade date — can be backdated")\n\n    st.markdown("##### Long Leg")\n    lcol1, lcol2, lcol3 = st.columns(3)\n    with lcol1:\n        long_strike = st.number_input("Long Strike", min_value=1.0, value=40.0, step=0.5, key="diag_long_strike")'

    # Pass entry_date to open_diagonal call
    OLD_CALL = 'entry_regime=entry_regime,\n                    entry_vix_level=entry_vix,\n                    entry_percentile=entry_pct / 100,'
    NEW_CALL = 'entry_regime=entry_regime,\n                    entry_vix_level=entry_vix,\n                    entry_percentile=entry_pct / 100,\n                    entry_date=entry_date_input.isoformat(),'

    patched_app = False
    if OLD_P in src:
        src = src.replace(OLD_P, NEW_P, 1)
        patched_app = True
        print("✅ app.py — Entry Date field added to paper form")
    else:
        print("⚠️  app.py — paper form Long Leg marker not found")

    if OLD_CALL in src:
        src = src.replace(OLD_CALL, NEW_CALL)
        print("✅ app.py — entry_date passed to open_diagonal calls")
    else:
        print("⚠️  app.py — open_diagonal call pattern not found")

    if patched_app:
        backup(APP)
        APP.write_text(src)

print("\nDone. Restart Streamlit to apply.")
print("  pkill -f streamlit && sleep 2")
print("  nohup streamlit run app.py --server.port 8501 --server.headless true >> ~/streamlit.log 2>&1 &")
