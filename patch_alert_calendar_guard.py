"""
Patch: alert_engine.py — add market calendar guard
Problem: alert_engine.py sends signal emails on holidays and weekends
         with stale/missing data, potentially causing incorrect trades.
Fix: Check is_market_open() at the top of send_alert().
     If market is closed, send a brief "market closed" notice instead
     of a full signal email, so you know the service ran but no action needed.

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_alert_calendar_guard.py
  python alert_engine.py --dry-run   # verify output
  sudo systemctl restart vix_alert.service
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "alert_engine.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix 1: Add market_calendar import at top ──────────────────────────────────
OLD_IMPORT = "from datetime import date, timedelta, datetime as dt"
NEW_IMPORT = (
    "from datetime import date, timedelta, datetime as dt\n"
    "from market_calendar import is_market_open, get_next_trading_day, format_calendar_warning"
)

if OLD_IMPORT in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT)
    print("✅ Fix 1: market_calendar import added")
elif "from market_calendar import" in src:
    print("ℹ️  Fix 1: market_calendar already imported — skipping")
else:
    # Fallback: add after the last 'from' import line
    lines = src.split("\n")
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("from ") or line.startswith("import "):
            last_import_idx = i
    lines.insert(last_import_idx + 1,
        "from market_calendar import is_market_open, get_next_trading_day, format_calendar_warning")
    src = "\n".join(lines)
    print("✅ Fix 1: market_calendar import inserted after last import")

# ── Fix 2: Add calendar guard at start of send_alert() ───────────────────────
# Find the send_alert function and insert guard after the opening lines
OLD_SEND = "def send_alert(dry_run: bool = False) -> bool:"

NEW_SEND = '''\
def send_alert(dry_run: bool = False) -> bool:
    """Send alert email — skips on market holidays with a notice email."""
    today = date.today()
    if not is_market_open(today):
        next_day = get_next_trading_day(today)
        subject  = f"[VIX Suite] Market Closed — {today.strftime('%a %b %d %Y')} · Next trading day: {next_day.strftime('%a %b %d')}"
        html     = f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;">
          <div style="background:#1E4D2B;color:white;padding:12px 20px;border-radius:6px 6px 0 0;">
            <strong>VIX 5% Weekly Suite</strong>
          </div>
          <div style="background:#f9f9f9;border:1px solid #ddd;padding:20px;border-radius:0 0 6px 6px;">
            <h3 style="color:#555;margin-top:0;">⛔ Market Closed — No Signal Today</h3>
            <p style="color:#333;">
              <strong>{today.strftime('%A, %B %d %Y')}</strong> is a market holiday.
              No position data or signals have been generated.
            </p>
            <p style="color:#333;">
              Next trading day: <strong>{next_day.strftime('%A, %B %d %Y')}</strong>
            </p>
            <p style="color:#888;font-size:12px;margin-bottom:0;">
              Auto-generated · Do not act on stale signals from previous days.
            </p>
          </div>
        </div>"""
        _send_email(subject, html, dry_run=dry_run)
        print(f"[Calendar] Market closed ({today}). Holiday notice sent. Next: {next_day}")
        return True  # successful run, just no signal'''

if OLD_SEND in src:
    src = src.replace(OLD_SEND, NEW_SEND, 1)  # only replace first occurrence
    print("✅ Fix 2: market calendar guard added to send_alert()")
else:
    print("⚠️  Fix 2: send_alert() signature not found — check function name in alert_engine.py")
    print("    Looking for: 'def send_alert(dry_run: bool = False) -> bool:'")

# ── Fix 3: Also guard the intraday alert if it has its own entry point ────────
OLD_INTRADAY = "def send_intraday_alert(dry_run: bool = False) -> bool:"
NEW_INTRADAY = '''\
def send_intraday_alert(dry_run: bool = False) -> bool:
    """Send intraday alert — skips silently on market holidays."""
    if not is_market_open(date.today()):
        print(f"[Calendar] Market closed — intraday alert suppressed.")
        return True'''

if OLD_INTRADAY in src:
    src = src.replace(OLD_INTRADAY, NEW_INTRADAY, 1)
    print("✅ Fix 3: intraday alert calendar guard added")
else:
    print("ℹ️  Fix 3: send_intraday_alert() not found — may use different name, skipping")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("\nVerify with:")
print("  python alert_engine.py --dry-run")
print("  sudo systemctl restart vix_alert.service vix_intraday.service")
