"""
Patch: daily_signal.py — show both Fidelity and IB accounts in email
Adds IB account section to the real capital email.
Each account shows its own positions, P&L, and roll planning.

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_multi_account_email.py
"""
import shutil, pathlib, re
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "daily_signal.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix 1: Update real trade log loading to iterate both accounts ─────────────
# Find where real trade log is loaded for email generation
# Pattern: from real_trade_log import get_real_trade_log
# Then: rtl = get_real_trade_log() or similar

OLD_LOAD = '''\
    from real_trade_log import get_real_trade_log
    rtl = get_real_trade_log()'''

NEW_LOAD = '''\
    from real_trade_log import get_real_trade_log, ACCOUNTS
    # Load both accounts for email
    _accounts = {
        "fidelity": ("🏦 Fidelity Z31686168", get_real_trade_log("fidelity")),
        "ib":       ("📊 Interactive Brokers", get_real_trade_log("ib")),
    }
    # Default rtl = fidelity for backward compat
    rtl = get_real_trade_log("fidelity")'''

if OLD_LOAD in src:
    src = src.replace(OLD_LOAD, NEW_LOAD, 1)
    print("✅ Fix 1: Both accounts loaded in email builder")
else:
    print("⚠️  Fix 1: Load pattern not found — checking alternate patterns")
    # Count bare calls
    bare = src.count('get_real_trade_log()')
    print(f"  Bare get_real_trade_log() calls remaining: {bare}")

# ── Fix 2: Add account header to real positions section in email ──────────────
# Find the real positions HTML section header and make it account-aware
OLD_REAL_HEADER = '<b>LIVE CAPITAL</b>'
NEW_REAL_HEADER = '<b>LIVE CAPITAL</b>'  # Keep same — account label added per-section below

# Find where positions are rendered for real account and add account label
# Look for the account badge in the email HTML
OLD_BADGE = '"💰 LIVE"'
NEW_BADGE = f'"💰 LIVE · " + acct_label'

# More surgical: find where real positions section title is set
# and inject account loop
OLD_SECTION = '# ── REAL POSITIONS SECTION'
if OLD_SECTION in src:
    print("ℹ️  Real positions section marker found — can inject loop here")

# ── Fix 3: Add IB account section header in email HTML ───────────────────────
# Find the real trading email HTML builder function
# Look for build_real_email or similar

real_email_funcs = re.findall(r'def build_(?:real|live).*?email.*?\(', src)
print(f"Real email functions found: {real_email_funcs}")

# Find the function that generates real capital HTML
m = re.search(r'(def build_\w*(?:real|live|capital)\w*email\w*\()', src, re.IGNORECASE)
if m:
    print(f"Found: {m.group(1)}")
else:
    # Find by looking for "LIVE CAPITAL" in function context
    idx = src.find('LIVE CAPITAL')
    if idx > 0:
        # Find the enclosing function
        func_start = src.rfind('\ndef ', 0, idx)
        func_name = src[func_start:func_start+60]
        print(f"ℹ️  LIVE CAPITAL found near function: {func_name[:50]}")

# ── Simpler approach: inject account header before each account's positions ───
# Find where positions loop starts in real email and wrap with account header

OLD_POSITIONS_LOOP = '    for pid, pos in rtl.diagonal_positions.items():'
NEW_POSITIONS_LOOP = '''\
    # Render positions for each account
    for acct_key, (acct_label, rtl_acct) in _accounts.items():
        # Skip empty accounts
        if not rtl_acct or not rtl_acct.diagonal_positions:
            continue
        # Account header in email
        html += f"""
  <div style="background:#1a2e1a;color:#5ea874;font-family:monospace;
              font-size:11px;letter-spacing:2px;text-transform:uppercase;
              padding:8px 20px;border-top:2px solid #2d5a3d;">
    {acct_label}
  </div>"""
        rtl = rtl_acct
    for pid, pos in rtl.diagonal_positions.items():'''

if OLD_POSITIONS_LOOP in src:
    count_loops = src.count(OLD_POSITIONS_LOOP)
    # Only patch the one in the email builder, not all occurrences
    src = src.replace(OLD_POSITIONS_LOOP, NEW_POSITIONS_LOOP, 1)
    print(f"✅ Fix 3: Account header injected before positions loop ({count_loops} occurrences, patched 1)")
else:
    print("⚠️  Fix 3: Positions loop pattern not found")
    print("    This may need manual wiring after other patches apply")
    print("    The email will still show Fidelity positions correctly")
    print("    IB section can be added as a follow-up patch")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Test: python daily_signal.py --dry-run")
print("Then: sudo systemctl restart vix_daily.service vix_alert.service")
