"""
Patch: daily_signal.py — suppress EMERGENCY ROLL on expiring worthless positions
Bug: DTE=0 + delta=0.000 triggers EMERGENCY ROLL label
Fix: If DTE=0 AND delta < 0.05 (effectively zero), show EXPIRING WORTHLESS instead

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_expiry_label.py
  sudo systemctl restart vix_daily.service vix_alert.service
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "daily_signal.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix: _roll_mode() — add expiring worthless check before emergency check ──
OLD_ROLL_MODE = '''\
def _roll_mode(short_delta: float, short_dte: int, short_strike: float,
               uvxy_price: float, debit_cap: float) -> dict:
    """
    Returns dict with mode, color, emoji, action_label, reason, target_otm_pct.
    """
    itm = short_strike <= uvxy_price'''

NEW_ROLL_MODE = '''\
def _roll_mode(short_delta: float, short_dte: int, short_strike: float,
               uvxy_price: float, debit_cap: float) -> dict:
    """
    Returns dict with mode, color, emoji, action_label, reason, target_otm_pct.
    """
    # Expiring worthless — DTE=0 and delta effectively zero
    if short_dte <= 0 and short_delta < 0.05:
        return dict(
            mode        = "EXPIRING_WORTHLESS",
            color       = "#16a34a",
            emoji       = "✅",
            badge       = "✅ Expiring Worthless",
            action_label= "No action needed",
            reason      = "Short expires worthless today — let it expire. Sell new short Monday.",
            target_otm  = "—",
        )
    itm = short_strike <= uvxy_price'''

if OLD_ROLL_MODE in src:
    src = src.replace(OLD_ROLL_MODE, NEW_ROLL_MODE)
    print("✅ Fix applied: EXPIRING WORTHLESS label for DTE=0 delta=0 positions")
else:
    print("⚠️  Pattern not found — trying flexible search")
    import re
    m = re.search(r'def _roll_mode\(short_delta.*?itm = short_strike <= uvxy_price', src, re.DOTALL)
    if m:
        old = m.group(0)
        new = old.replace(
            "    itm = short_strike <= uvxy_price",
            """\
    # Expiring worthless — DTE=0 and delta effectively zero
    if short_dte <= 0 and short_delta < 0.05:
        return dict(
            mode        = "EXPIRING_WORTHLESS",
            color       = "#16a34a",
            emoji       = "✅",
            badge       = "✅ Expiring Worthless",
            action_label= "No action needed",
            reason      = "Short expires worthless today — let it expire. Sell new short Monday.",
            target_otm  = "—",
        )
    itm = short_strike <= uvxy_price"""
        )
        src = src.replace(old, new)
        print("✅ Fix applied via flexible search")
    else:
        print("⚠️  Could not patch automatically — manual edit needed at _roll_mode()")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
