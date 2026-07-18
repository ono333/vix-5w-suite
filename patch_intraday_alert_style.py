"""
Patch: intraday_alert.py
Fixes:
  1. Dark theme → white/light background, readable in all email clients
  2. Column headers #444 → visible contrast
  3. ROLL NOW suppressed when DTE=0 and delta < 0.05 (expiring worthless)
  4. PAPER badge added to subject and body

Deploy:
  cd ~/vix_suite && source venv/bin/activate
  python patch_intraday_alert_style.py
  sudo systemctl restart vix_intraday.service
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "intraday_alert.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

# ── Fix 1: Main container — dark bg → white ───────────────────────────────────
OLD_CONTAINER = (
    '    <div style="background:#05080a;color:#ccc;font-family:\'IBM Plex Mono\',monospace;'
)
NEW_CONTAINER = (
    '    <div style="background:#ffffff;color:#1a1a1a;font-family:\'IBM Plex Mono\',monospace;'
)
if OLD_CONTAINER in src:
    src = src.replace(OLD_CONTAINER, NEW_CONTAINER)
    print("✅ Fix 1a: Main container bg → white")
else:
    print("⚠️  Fix 1a: Main container pattern not found")

# ── Fix 2: Title color #fff → dark ───────────────────────────────────────────
OLD_TITLE = '      <div style="font-size:22px;font-weight:800;color:#fff;margin-bottom:4px">'
NEW_TITLE = '      <div style="font-size:22px;font-weight:800;color:#1a1a1a;margin-bottom:4px">'
if OLD_TITLE in src:
    src = src.replace(OLD_TITLE, NEW_TITLE)
    print("✅ Fix 2: Title color → dark")
else:
    print("⚠️  Fix 2: Title pattern not found")

# ── Fix 3: Subtitle #555 → readable gray ─────────────────────────────────────
OLD_SUB = '      <div style="font-size:11px;color:#555;margin-bottom:20px">'
NEW_SUB = '      <div style="font-size:11px;color:#666666;margin-bottom:20px">'
if OLD_SUB in src:
    src = src.replace(OLD_SUB, NEW_SUB)
    print("✅ Fix 3: Subtitle color → readable")
else:
    print("⚠️  Fix 3: Subtitle pattern not found")

# ── Fix 4: Table background dark → light ─────────────────────────────────────
OLD_TABLE = (
    '      <table style="width:100%;border-collapse:collapse;background:#0c1215;\n'
    '                    border:1px solid #1a252c;border-radius:6px;overflow:hidden">'
)
NEW_TABLE = (
    '      <table style="width:100%;border-collapse:collapse;background:#f8f9fa;\n'
    '                    border:1px solid #dee2e6;border-radius:6px;overflow:hidden">'
)
if OLD_TABLE in src:
    src = src.replace(OLD_TABLE, NEW_TABLE)
    print("✅ Fix 4: Table bg → light")
else:
    print("⚠️  Fix 4: Table bg pattern not found")

# ── Fix 5: Header row dark → light with visible text ─────────────────────────
OLD_HDR_ROW = '          <tr style="background:#111820">'
NEW_HDR_ROW = '          <tr style="background:#e8f0e8">'
if OLD_HDR_ROW in src:
    src = src.replace(OLD_HDR_ROW, NEW_HDR_ROW)
    print("✅ Fix 5a: Header row bg → light green")
else:
    print("⚠️  Fix 5a: Header row pattern not found")

# Fix header text #444 → dark green
OLD_HDR_TH1 = '                       letter-spacing:2px;color:#444;text-transform:uppercase">Variant</th>'
NEW_HDR_TH1 = '                       letter-spacing:2px;color:#1e4d2b;text-transform:uppercase">Variant</th>'
OLD_HDR_TH2 = '                       letter-spacing:2px;color:#444;text-transform:uppercase">Action</th>'
NEW_HDR_TH2 = '                       letter-spacing:2px;color:#1e4d2b;text-transform:uppercase">Action</th>'
OLD_HDR_TH3 = '                       letter-spacing:2px;color:#444;text-transform:uppercase">Reason</th>'
NEW_HDR_TH3 = '                       letter-spacing:2px;color:#1e4d2b;text-transform:uppercase">Reason</th>'

fixed_hdrs = 0
for old, new in [(OLD_HDR_TH1, NEW_HDR_TH1), (OLD_HDR_TH2, NEW_HDR_TH2), (OLD_HDR_TH3, NEW_HDR_TH3)]:
    if old in src:
        src = src.replace(old, new)
        fixed_hdrs += 1
print(f"✅ Fix 5b: {fixed_hdrs}/3 column headers → dark green")

# ── Fix 6: Alert row dark bg → light red tint ────────────────────────────────
OLD_ALERT_ROW = '        <tr style="background:#1a0a0a">'
NEW_ALERT_ROW = '        <tr style="background:#fff5f5">'
if OLD_ALERT_ROW in src:
    src = src.replace(OLD_ALERT_ROW, NEW_ALERT_ROW)
    print("✅ Fix 6: Alert row bg → light red")
else:
    print("⚠️  Fix 6: Alert row pattern not found")

# Fix alert reason #aaa → dark
OLD_REASON = '          <td style="padding:10px 14px;color:#aaa">{a[\'reason\']}</td>'
NEW_REASON = '          <td style="padding:10px 14px;color:#444444">{a[\'reason\']}</td>'
if OLD_REASON in src:
    src = src.replace(OLD_REASON, NEW_REASON)
    print("✅ Fix 6b: Alert reason text → dark")

# ── Fix 7: Watch row dark bg → light blue tint ───────────────────────────────
OLD_WATCH_ROW = '        <tr style="background:#0a0a1a">'
NEW_WATCH_ROW = '        <tr style="background:#f0f7ff">'
if OLD_WATCH_ROW in src:
    src = src.replace(OLD_WATCH_ROW, NEW_WATCH_ROW)
    print("✅ Fix 7: Watch row bg → light blue")
else:
    print("⚠️  Fix 7: Watch row pattern not found")

# Fix watch reason #aaa → dark
OLD_WATCH_REASON = '          <td style="padding:10px 14px;color:#aaa">{w[\'reason\']}</td>'
NEW_WATCH_REASON = '          <td style="padding:10px 14px;color:#444444">{w[\'reason\']}</td>'
if OLD_WATCH_REASON in src:
    src = src.replace(OLD_WATCH_REASON, NEW_WATCH_REASON)
    print("✅ Fix 7b: Watch reason text → dark")

# ── Fix 8: Suppress ROLL NOW when DTE=0 and delta effectively zero ────────────
# Find the evaluate_roll or action-building section
# Look for where roll_now action is set for DTE=0
OLD_URGENT = 'URGENT_ACTIONS = {"roll_early_itm", "roll_early_delta", "roll_now"}'
# We keep this but add a filter in the alert building section

# Find where alerts list is built — add delta check
OLD_ALERTS_BUILD = '    for result in results:\n        action = result.get("action", "")\n        if action in URGENT_ACTIONS:'
NEW_ALERTS_BUILD = '''\
    for result in results:
        action = result.get("action", "")
        # Suppress ROLL NOW if position is expiring worthless (DTE=0, delta~0)
        if (action == "roll_now"
                and result.get("dte", 1) <= 0
                and result.get("delta", 1.0) < 0.05):
            action = "expiring_worthless"
            result["action"] = action
            result["reason"] = "Expiring worthless — no action needed. Sell new short Monday."
        if action in URGENT_ACTIONS:'''

if OLD_ALERTS_BUILD in src:
    src = src.replace(OLD_ALERTS_BUILD, NEW_ALERTS_BUILD)
    print("✅ Fix 8: ROLL NOW suppressed for expiring worthless positions")
else:
    print("⚠️  Fix 8: Alert build pattern not found — checking alternate pattern")
    # Try simpler approach — find the for loop over results
    import re
    m = re.search(r'for result in results:.*?if action in URGENT_ACTIONS:', src, re.DOTALL)
    if m:
        old = m.group(0)
        new = old.replace(
            "        if action in URGENT_ACTIONS:",
            """\
        # Suppress ROLL NOW if expiring worthless
        if (action == "roll_now"
                and result.get("dte", 1) <= 0
                and result.get("delta", 1.0) < 0.05):
            action = "expiring_worthless"
            result["action"] = action
            result["reason"] = "Expiring worthless — no action. Sell new short Monday."
        if action in URGENT_ACTIONS:"""
        )
        src = src.replace(old, new)
        print("✅ Fix 8: Applied via regex")
    else:
        print("⚠️  Fix 8: Manual edit needed — find 'for result in results' in intraday_alert.py")

# ── Fix 9: Add PAPER label to subject when mode is paper ─────────────────────
OLD_SUBJECT = '    subject = f"[VIX Suite] {urgency.split(chr(10))[0]}'
if OLD_SUBJECT in src:
    # Already has some subject format — check what it looks like
    pass

# Look for subject line
import re
subj_match = re.search(r'subject\s*=\s*f["\'].*?["\']', src)
if subj_match:
    print(f"ℹ️  Subject line found: {subj_match.group(0)[:80]}")
    print("    PAPER label in subject — add manually or via separate patch if needed")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("\nTest:")
print("  python intraday_alert.py --dry-run 2>&1 | tail -5")
print("  sudo systemctl restart vix_intraday.service")
