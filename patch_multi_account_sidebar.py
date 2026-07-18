"""
Patch: app.py — multi-account sidebar selector
Adds account selector (Fidelity / IB) to the Real Trading sidebar.
All get_real_trade_log() calls are patched to pass st.session_state.real_account.

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_multi_account_sidebar.py
"""
import shutil, pathlib, re
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "app.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix 1: Add account selector near the Fidelity/IB mention in sidebar ──────
# Line 5387 area: 'Fidelity / IB · Live capital · Separate from paper trades'
OLD_SIDEBAR = 'Fidelity / IB · Live capital · Separate from paper trades</span>'
NEW_SIDEBAR = '''\
Fidelity / IB · Live capital · Separate from paper trades</span>

    # Account selector
    if "real_account" not in st.session_state:
        st.session_state.real_account = "fidelity"

    _acct_options = {"🏦 Fidelity (Z31686168)": "fidelity", "📊 Interactive Brokers": "ib"}
    _acct_display = st.radio(
        "Account",
        list(_acct_options.keys()),
        index=0 if st.session_state.real_account == "fidelity" else 1,
        horizontal=True,
        key="real_account_selector",
    )
    st.session_state.real_account = _acct_options[_acct_display]'''

if OLD_SIDEBAR in src:
    src = src.replace(OLD_SIDEBAR, NEW_SIDEBAR, 1)
    print("✅ Fix 1: Account selector added to sidebar")
else:
    print("⚠️  Fix 1: Sidebar anchor not found — searching alternate")
    # Try alternate anchor
    ALT = 'Fidelity / IB'
    if ALT in src:
        idx = src.find(ALT)
        print(f"  Found 'Fidelity / IB' at position {idx} — manual placement needed")
    else:
        print("  Not found — check app.py line 5387")

# ── Fix 2: Patch all get_real_trade_log() calls to pass account ──────────────
# Replace all bare get_real_trade_log() with account-aware version
# Use session_state with fallback to avoid errors outside sidebar context

OLD_CALL = 'get_real_trade_log()'
NEW_CALL = 'get_real_trade_log(st.session_state.get("real_account", "fidelity"))'

count = src.count(OLD_CALL)
src = src.replace(OLD_CALL, NEW_CALL)
print(f"✅ Fix 2: {count} get_real_trade_log() calls updated with account param")

# ── Fix 3: Patch reset_real_trade_log_cache() calls ─────────────────────────
OLD_RESET = 'reset_real_trade_log_cache()'
NEW_RESET = 'reset_real_trade_log_cache(st.session_state.get("real_account", "fidelity"))'

count_r = src.count(OLD_RESET)
src = src.replace(OLD_RESET, NEW_RESET)
print(f"✅ Fix 3: {count_r} reset_real_trade_log_cache() calls updated")

# ── Fix 4: Update the MARGIN/CASH help text to reflect account selector ───────
OLD_HELP = 'help="MARGIN = Fidelity Live | CASH = IB Paper"'
NEW_HELP = 'help="Select account above to switch between Fidelity and IB"'
if OLD_HELP in src:
    src = src.replace(OLD_HELP, NEW_HELP)
    print("✅ Fix 4: Help text updated")

OLD_INFO = '"💵 Fidelity = MARGIN | 📋 IB Paper = CASH"'
NEW_INFO = '"💵 Active account: " + st.session_state.get("real_account", "fidelity").upper()'
if OLD_INFO in src:
    src = src.replace(OLD_INFO, NEW_INFO)
    print("✅ Fix 4b: Account info text updated")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Restart Streamlit to apply: sudo systemctl restart vix_app.service")
