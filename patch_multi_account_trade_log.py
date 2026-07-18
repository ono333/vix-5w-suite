"""
Patch: real_trade_log.py — multi-account support
Adds account parameter to get_real_trade_log() and reset_real_trade_log_cache()
Backward compatible — defaults to 'fidelity' if no account specified.

File locations:
  ~/.vix_suite/real_trade_log.json          → renamed to real_trade_log_fidelity.json
  ~/.vix_suite/real_trade_log_ib.json       → new IB account

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_multi_account_trade_log.py
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "real_trade_log.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix 1: Add ACCOUNTS dict and update REAL_LOG_PATH ────────────────────────
OLD_PATH = 'REAL_LOG_PATH = Path.home() / ".vix_suite" / "real_trade_log.json"'
NEW_PATH = '''\
# Multi-account support
ACCOUNTS = {
    "fidelity": Path.home() / ".vix_suite" / "real_trade_log_fidelity.json",
    "ib":       Path.home() / ".vix_suite" / "real_trade_log_ib.json",
}
# Legacy path — kept for migration
REAL_LOG_PATH = Path.home() / ".vix_suite" / "real_trade_log_fidelity.json"'''

if OLD_PATH in src:
    src = src.replace(OLD_PATH, NEW_PATH)
    print("✅ Fix 1: ACCOUNTS dict added")
else:
    print("⚠️  Fix 1: REAL_LOG_PATH pattern not found")

# ── Fix 2: Update get_real_trade_log() to accept account param ───────────────
OLD_GET = '''\
def get_real_trade_log() -> RealTradeLog:
    """Get the global trade log instance."""
    global _trade_log_instance
    if _trade_log_instance is None:
        _trade_log_instance = TradeLog()
    return _trade_log_instance'''

# Find the actual get_real_trade_log — it's in real_trade_log.py not trade_log.py
import re
m = re.search(
    r'def get_real_trade_log\(\) -> RealTradeLog:\n.*?"""Get the global.*?"""\n.*?global.*?\n.*?if.*?None.*?\n.*?return.*?\n',
    src, re.DOTALL
)
if m:
    old_func = m.group(0)
    new_func = '''\
# Per-account cache
_real_trade_log_cache: dict = {}

def get_real_trade_log(account: str = "fidelity") -> "RealTradeLog":
    """Get the real trade log for the specified account (fidelity or ib)."""
    global _real_trade_log_cache
    if account not in _real_trade_log_cache or _real_trade_log_cache[account] is None:
        log_path = ACCOUNTS.get(account, ACCOUNTS["fidelity"])
        _real_trade_log_cache[account] = RealTradeLog(log_path=log_path)
    return _real_trade_log_cache[account]

'''
    src = src.replace(old_func, new_func, 1)
    print("✅ Fix 2: get_real_trade_log() updated with account param")
else:
    # Simpler replacement
    OLD_SIMPLE = 'def get_real_trade_log() -> RealTradeLog:'
    NEW_SIMPLE = 'def get_real_trade_log(account: str = "fidelity") -> "RealTradeLog":'
    if OLD_SIMPLE in src:
        src = src.replace(OLD_SIMPLE, NEW_SIMPLE, 1)
        print("✅ Fix 2: Signature updated (simple)")
    else:
        print("⚠️  Fix 2: get_real_trade_log signature not found")

# ── Fix 3: Update reset_real_trade_log_cache() ───────────────────────────────
OLD_RESET = 'def reset_real_trade_log_cache():'
NEW_RESET = '''\
def reset_real_trade_log_cache(account: str = None):
    """Reset cache for one account or all accounts if account is None."""
    global _real_trade_log_cache
    if account:
        _real_trade_log_cache.pop(account, None)
    else:
        _real_trade_log_cache.clear()

def _reset_real_trade_log_cache_legacy():
    """Legacy reset — clears all accounts."""'''

if OLD_RESET in src:
    src = src.replace(OLD_RESET, NEW_RESET, 1)
    print("✅ Fix 3: reset_real_trade_log_cache() updated")
else:
    print("⚠️  Fix 3: reset function not found")

TARGET.write_text(src)

# ── Migrate existing JSON file ────────────────────────────────────────────────
old_json = pathlib.Path.home() / ".vix_suite" / "real_trade_log.json"
new_json = pathlib.Path.home() / ".vix_suite" / "real_trade_log_fidelity.json"
ib_json  = pathlib.Path.home() / ".vix_suite" / "real_trade_log_ib.json"

if old_json.exists() and not new_json.exists():
    shutil.copy2(old_json, new_json)
    print(f"✅ Migrated: real_trade_log.json → real_trade_log_fidelity.json")
elif new_json.exists():
    print("ℹ️  real_trade_log_fidelity.json already exists — no migration needed")

if not ib_json.exists():
    ib_json.write_text('{"positions": {}, "closed_positions": [], "assignments": []}')
    print(f"✅ Created: real_trade_log_ib.json (empty)")

print(f"\nPatched: {TARGET}")
