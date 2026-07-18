"""
Patch: intraday_alert.py — show all live positions in every email
Problem: Hold positions never appear in email — body is empty on Watch Alert
Fix: Add a "Live Positions" section showing all real positions with status,
     DTE, strike, and action regardless of alert level.
     Paper positions shown separately as a summary line only.
"""
import shutil, pathlib
from datetime import datetime

TARGET = pathlib.Path.home() / "vix_suite" / "intraday_alert.py"
BACKUP = TARGET.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(TARGET, BACKUP)
print(f"Backup: {BACKUP}")

src = TARGET.read_text()

OLD_FOOTER = '''\
      {"<div style='margin-top:16px;padding:12px;background:#1a0000;border:1px solid #ff3366;border-radius:4px;color:#ff6666;font-size:12px'>⚠ Roll before market close today to avoid assignment risk.</div>" if alerts else ""}
      <div style="margin-top:20px;font-size:10px;color:#333">
        VIX 5% Weekly Suite · Intraday Monitor · Do not reply
      </div>
    </div>"""
    return subject, html'''

NEW_FOOTER = '''\
      {"<div style='margin-top:16px;padding:12px;background:#fff3cd;border:1px solid #ffc107;border-radius:4px;color:#856404;font-size:12px'>⚠ Roll before market close today to avoid assignment risk.</div>" if alerts else ""}

      <!-- Live positions summary — always shown -->
      <div style="margin-top:20px;">
        <div style="font-size:11px;font-weight:700;color:#1e4d2b;letter-spacing:1px;
                    text-transform:uppercase;margin-bottom:8px;">
          💰 Live Positions
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
          <tr style="background:#e8f0e8;">
            <th style="padding:6px 10px;text-align:left;color:#1e4d2b;">Variant</th>
            <th style="padding:6px 10px;text-align:center;color:#1e4d2b;">DTE</th>
            <th style="padding:6px 10px;text-align:center;color:#1e4d2b;">Strike</th>
            <th style="padding:6px 10px;text-align:center;color:#1e4d2b;">Status</th>
            <th style="padding:6px 10px;text-align:left;color:#1e4d2b;">Action</th>
          </tr>
          {"".join([
            f"<tr style='background:{'#fff5f5' if p['action'] not in ('hold','expiring_worthless') else '#f8fff8'};border-top:1px solid #dee2e6;'>"
            f"<td style='padding:6px 10px;font-weight:700;color:#1a1a1a;'>{p['variant'].replace('💰 [REAL] ','')}</td>"
            f"<td style='padding:6px 10px;text-align:center;color:#333;'>{p['dte']}d</td>"
            f"<td style='padding:6px 10px;text-align:center;color:#333;'>${p['strike']}</td>"
            f"<td style='padding:6px 10px;text-align:center;font-weight:700;"
            f"color:{'#16a34a' if p['action'] in ('hold','expiring_worthless') else '#dc2626'};'>"
            f"{'✅ On Track' if p['action'] == 'hold' else '⚠️ ' + p['action'].upper().replace('_',' ')}</td>"
            f"<td style='padding:6px 10px;color:#555;font-size:11px;'>"
            f"{'No action needed' if p['action'] == 'hold' else p['reason'][:60]}</td>"
            f"</tr>"
            for p in (alerts + watches + [e for e in all_entries if e.get('mode_label') == 'real' and e['action'] == 'hold'])
            if p.get('mode_label') == 'real' or '💰' in p.get('variant','')
          ])}
        </table>
      </div>

      <!-- Paper summary line -->
      <div style="margin-top:12px;padding:8px 12px;background:#f0f4f1;
                  border-radius:4px;font-size:11px;color:#666;">
        📋 Paper positions: {len([e for e in (alerts+watches) if '📋' in e.get('variant','')])} flagged
        · {len([e for e in all_entries if e.get('mode_label') == 'paper'])} total paper positions tracked
      </div>

      <div style="margin-top:16px;font-size:10px;color:#888;">
        VIX 5% Weekly Suite · Intraday Monitor · UVXY ${uvxy:.2f} · Do not reply
      </div>
    </div>"""
    return subject, html'''

if OLD_FOOTER in src:
    src = src.replace(OLD_FOOTER, NEW_FOOTER)
    print("✅ Live positions summary section added")
else:
    print("⚠️  Footer pattern not found — checking...")
    # Try to find the return statement
    if "return subject, html" in src:
        print("  'return subject, html' exists in file")
    print("  Manual edit needed in build_alert_email() function")

# Also need to track all_entries — add it to the loop
OLD_LOOP = "    alerts, watches = [], []"
NEW_LOOP = "    alerts, watches, all_entries = [], [], []"
if OLD_LOOP in src:
    src = src.replace(OLD_LOOP, NEW_LOOP, 1)
    print("✅ all_entries tracker added")

# Add mode_label to entry dict and append to all_entries
OLD_ENTRY = '''        entry = dict(
            variant  = f\'{\\'📋\\' if mode_label==\\"paper\\" else \\'💰\\'} [{mode_label.upper()}] {pos.variant_name}\',
            action   = decision.action,
            reason   = decision.reason[:80],
            est_bb   = decision.expected_bb,
            dte      = dte,
            strike   = short.strike,
        )'''
NEW_ENTRY = '''        entry = dict(
            variant    = f\'{\\'📋\\' if mode_label==\\"paper\\" else \\'💰\\'} [{mode_label.upper()}] {pos.variant_name}\',
            action     = decision.action,
            reason     = decision.reason[:80],
            est_bb     = decision.expected_bb,
            dte        = dte,
            strike     = short.strike,
            mode_label = mode_label,
        )
        all_entries.append(entry)'''
if OLD_ENTRY in src:
    src = src.replace(OLD_ENTRY, NEW_ENTRY)
    print("✅ mode_label added to entry, all_entries populated")
else:
    print("⚠️  Entry pattern not found — try simpler append")
    # Simpler: find where alerts.append or watches.append is called and add all_entries.append
    OLD_APPEND = "        if decision.action in URGENT_ACTIONS:\n            alerts.append(entry)"
    NEW_APPEND = "        all_entries.append(entry)\n        if decision.action in URGENT_ACTIONS:\n            alerts.append(entry)"
    if OLD_APPEND in src:
        src = src.replace(OLD_APPEND, NEW_APPEND)
        print("✅ all_entries.append added before alerts check")

# Pass all_entries to build_alert_email
OLD_BUILD = "        subject, html = build_alert_email(alerts, watches, uvxy_price)"
NEW_BUILD = "        subject, html = build_alert_email(alerts, watches, uvxy_price, all_entries)"
if OLD_BUILD in src:
    src = src.replace(OLD_BUILD, NEW_BUILD)
    print("✅ all_entries passed to build_alert_email")

# Update build_alert_email signature
OLD_SIG = "def build_alert_email(alerts: list, watches: list, uvxy: float) -> tuple:"
NEW_SIG = "def build_alert_email(alerts: list, watches: list, uvxy: float, all_entries: list = None) -> tuple:"
if OLD_SIG in src:
    src = src.replace(OLD_SIG, NEW_SIG)
    print("✅ build_alert_email signature updated")
else:
    print("⚠️  build_alert_email signature not found — check function name")
    import re
    m = re.search(r'def build_alert_email\(.*?\).*?:', src)
    if m:
        print(f"  Found: {m.group(0)}")

TARGET.write_text(src)
print(f"\nPatched: {TARGET}")
print("Test: python intraday_alert.py --force-send 2>&1 | tail -5")
