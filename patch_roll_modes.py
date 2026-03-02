"""
Patch: Three-mode roll system + clear action labels + automated alert emails

Roll Mode Definitions (ChatGPT spec):
  ROUTINE ROLL  — DTE ≤ 2, delta < 0.40. Scheduled harvest. Roll forward, target 4-6% OTM.
  DEFENSIVE ROLL — Delta 0.40–0.49 OR DTE ≤ 4 with delta rising. Proactive, credit still available.
  EMERGENCY ROLL — Delta ≥ 0.50 OR ITM. Gamma accelerating. Roll immediately, any credit.

Automated alert email: sent when ANY position reaches DEFENSIVE or EMERGENCY roll mode.
"""
from pathlib import Path

ds = Path("daily_signal.py")
src = ds.read_text()
changes = 0

# ══════════════════════════════════════════════════════════════════════
# 1. Replace _convexity_status with three-mode roll classifier
# ══════════════════════════════════════════════════════════════════════
old_conv = '''def _convexity_status(short_delta: float) -> tuple:'''
new_conv = '''def _roll_mode(short_delta: float, short_dte: int, short_strike: float,
               uvxy_price: float, debit_cap: float = 1.50) -> dict:
    """
    Three-mode roll classifier per ChatGPT spec.
    Returns dict with mode, color, emoji, action_label, reason, target_otm_pct.
    """
    itm = short_strike <= uvxy_price

    if short_dte <= 0 or (short_dte <= 2 and itm):
        return dict(
            mode        = "EMERGENCY",
            color       = "#cc0000",
            emoji       = "🚨",
            badge       = "🚨 EMERGENCY ROLL",
            action      = "Roll immediately — position ITM or expired",
            reason      = "Short expired or ITM — gamma risk extreme",
            target_otm  = "6–9% OTM minimum",
            target_note = "Accept debit if necessary to restore structure",
            priority    = 3,
        )
    elif short_delta >= 0.50 or itm:
        return dict(
            mode        = "EMERGENCY",
            color       = "#cc0000",
            emoji       = "🚨",
            badge       = "🚨 EMERGENCY ROLL",
            action      = "Roll now — delta ≥ 0.50, gamma accelerating",
            reason      = f"Short delta {short_delta:.2f} — credit window closing",
            target_otm  = "6–9% OTM",
            target_note = "Roll before debit exceeds cap",
            priority    = 3,
        )
    elif short_delta >= 0.40 or (short_dte <= 4 and short_delta >= 0.35):
        return dict(
            mode        = "DEFENSIVE",
            color       = "#ff6600",
            emoji       = "🛡️",
            badge       = "🛡️ DEFENSIVE ROLL SUGGESTED",
            action      = "Roll forward this week — credit still available",
            reason      = f"Short delta {short_delta:.2f} approaching 0.50 — act before gamma spikes",
            target_otm  = "4–6% OTM",
            target_note = "Target strike that keeps delta < 0.40 after roll",
            priority    = 2,
        )
    elif short_dte <= 2:
        return dict(
            mode        = "ROUTINE",
            color       = "#ff9800",
            emoji       = "🔄",
            badge       = "🔄 ROUTINE ROLL",
            action      = "Roll as scheduled — DTE ≤ 2",
            reason      = f"DTE {short_dte}d — standard weekly harvest roll",
            target_otm  = "4–6% OTM",
            target_note = "Maintain net credit above debit cap",
            priority    = 1,
        )
    else:
        return dict(
            mode        = "HOLD",
            color       = "#2196F3",
            emoji       = "✅",
            badge       = "✅ On Track",
            action      = "Hold — structure healthy",
            reason      = f"Short delta {short_delta:.2f}, DTE {short_dte}d — no action needed",
            target_otm  = "—",
            target_note = "Monitor daily",
            priority    = 0,
        )


def _convexity_status(short_delta: float) -> tuple:'''

if old_conv in src:
    src = src.replace(old_conv, new_conv)
    changes += 1
    print("✓ Added _roll_mode() three-mode classifier")
else:
    print("✗ _convexity_status not found")

# ══════════════════════════════════════════════════════════════════════
# 2. Update paper email action banner to use _roll_mode
# ══════════════════════════════════════════════════════════════════════
old_paper_action = '''            # Action styling
            action_map = {
                "TAKE_PROFIT": ("#4CAF50", "🎯 TAKE PROFIT"),
                "STOP_LOSS":   ("#f44336", "🛑 STOP LOSS — Short doubled"),
                "ROLL":        ("#ff9800", "🔄 DEFENSIVE ROLL — Delta ≥ 0.50"),
                "ROLL_NOW":    ("#f44336", "🔔 ACTION REQUIRED: Short Expired"),
                "SELL_SHORT":  ("#9c27b0", "📝 SELL SHORT — No active short"),
                "HOLD":        ("#2196F3", "✅ HOLD — Monitoring Active"),
            }
            ac, at = action_map.get(action, ("#607D8B", f"ℹ️ {action}"))

            # Urgency label
            if short_dte == 0:     urgency = "🔔 ACTION REQUIRED"
            elif short_delta >= 0.50: urgency = "🚨 DEFENSIVE ROLL REQUIRED"
            elif short_delta >= 0.35: urgency = "⚠️ Active Monitoring"
            else:                     urgency = "✅ On Track"'''

new_paper_action = '''            # Three-mode roll classifier
            _rm = _roll_mode(short_delta, short_dte,
                             float(diag.short_legs[-1].strike) if diag.short_legs else 999,
                             vix_level, debit_cap)
            urgency = _rm["badge"]
            ac      = _rm["color"]
            at      = _rm["action"]

            # Legacy action map for SELL_SHORT / TAKE_PROFIT overrides
            if action == "SELL_SHORT":
                urgency = "📝 SELL SHORT — No active short"
                ac = "#9c27b0"
            elif action == "TAKE_PROFIT":
                urgency = "🎯 TAKE PROFIT"
                ac = "#4CAF50"'''

if old_paper_action in src:
    src = src.replace(old_paper_action, new_paper_action)
    changes += 1
    print("✓ Paper email: action banner uses _roll_mode()")
else:
    print("✗ Paper action map not found")

# ══════════════════════════════════════════════════════════════════════
# 3. Update paper email action footer to show precise instructions
# ══════════════════════════════════════════════════════════════════════
old_paper_footer = '''        <div style="background:{ac};color:#fff;padding:8px 12px;border-radius:4px;
                    font-size:12px;font-weight:700;margin-top:10px;">
          {urgency}
        </div>'''

new_paper_footer = '''        <div style="background:{ac};color:#fff;padding:10px 14px;border-radius:4px;
                    font-size:12px;font-weight:700;margin-top:10px;">
          {urgency}<br>
          <span style="font-weight:400;font-size:11px;opacity:0.92;">
            Action: {_rm["action"]}<br>
            Target: {_rm["target_otm"]} OTM &nbsp;|&nbsp; {_rm["target_note"]}
          </span>
        </div>'''

if old_paper_footer in src:
    src = src.replace(old_paper_footer, new_paper_footer)
    changes += 1
    print("✓ Paper email: action footer shows precise instructions")
else:
    print("✗ Paper action footer not found")

# ══════════════════════════════════════════════════════════════════════
# 4. Update real email urgency to use _roll_mode
# ══════════════════════════════════════════════════════════════════════
old_real_urgency = '''        if short_dte <= 0:    urgency, uc = "🔔 ACTION REQUIRED", "#cc0000"
        elif short_delta >= 0.50: urgency, uc = "🚨 DEFENSIVE ROLL", "#cc3300"
        elif short_delta >= 0.35: urgency, uc = "⚠️ Active Monitoring", "#cc6600"
        else:                     urgency, uc = "✅ On Track", "#2e7d32"'''

new_real_urgency = '''        _rm_real = _roll_mode(short_delta, short_dte,
                              float(short.strike) if short else 999,
                              vix_level, debit_cap)
        urgency = _rm_real["badge"]
        uc      = _rm_real["color"]'''

if old_real_urgency in src:
    src = src.replace(old_real_urgency, new_real_urgency)
    changes += 1
    print("✓ Real email: urgency uses _roll_mode()")
else:
    print("✗ Real email urgency not found")

# ══════════════════════════════════════════════════════════════════════
# 5. Add automated alert function
# ══════════════════════════════════════════════════════════════════════
alert_func = '''

def build_alert_email(positions_needing_action: list, is_real: bool = False) -> str:
    """
    Build a concise alert email for DEFENSIVE or EMERGENCY roll positions.
    positions_needing_action: list of dicts with position info + roll_mode dict.
    """
    from datetime import datetime
    fetch_time = datetime.now()
    theme_bg   = "#1a0800" if is_real else "#f0f4ff"
    theme_txt  = "#ffcc88" if is_real else "#1a1a2e"
    header_bg  = "#cc3300" if is_real else "#1565c0"
    tag        = "💵 LIVE CAPITAL" if is_real else "📋 PAPER"

    rows = ""
    for p in positions_needing_action:
        rm    = p["roll_mode"]
        rows += f"""
    <tr style="border-bottom:1px solid #333;">
      <td style="padding:8px;font-weight:700;">{p['name']}</td>
      <td style="padding:8px;color:{rm['color']};font-weight:700;">{rm['badge']}</td>
      <td style="padding:8px;">{rm['reason']}</td>
      <td style="padding:8px;font-weight:600;">{rm['action']}</td>
      <td style="padding:8px;">{rm['target_otm']} OTM</td>
      <td style="padding:8px;font-size:11px;color:#888;">{rm['target_note']}</td>
    </tr>"""

    html = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:{theme_bg};font-family:Arial,sans-serif;">
<div style="max-width:700px;margin:0 auto;padding:20px;">

  <div style="background:{header_bg};color:#fff;padding:16px 20px;border-radius:8px 8px 0 0;">
    <div style="font-size:18px;font-weight:800;">{tag} — ⚡ ROLL ALERT</div>
    <div style="font-size:12px;opacity:0.85;margin-top:4px;">
      {len(positions_needing_action)} position(s) require attention · 
      {fetch_time.strftime('%Y-%m-%d %H:%M ET')}
    </div>
  </div>

  <div style="background:#fff;border:1px solid #ddd;padding:16px;">
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#f5f5f5;font-weight:700;">
        <th style="padding:8px;text-align:left;">Variant</th>
        <th style="padding:8px;text-align:left;">Mode</th>
        <th style="padding:8px;text-align:left;">Reason</th>
        <th style="padding:8px;text-align:left;">Action</th>
        <th style="padding:8px;text-align:left;">Target</th>
        <th style="padding:8px;text-align:left;">Note</th>
      </tr>
      {rows}
    </table>
  </div>

  <div style="background:#fff3cd;border:1px solid #ffc107;padding:12px;margin-top:12px;
              border-radius:4px;font-size:12px;">
    <strong>Roll Mode Definitions:</strong><br>
    🔄 <b>Routine Roll</b> — DTE ≤ 2, delta &lt; 0.40. Scheduled harvest. Target 4–6% OTM.<br>
    🛡️ <b>Defensive Roll</b> — Delta 0.40–0.49 or DTE ≤ 4 rising. Proactive, credit available. Target 4–6% OTM.<br>
    🚨 <b>Emergency Roll</b> — Delta ≥ 0.50 or ITM. Roll immediately. Target 6–9% OTM. Accept debit if needed.
  </div>

  <div style="text-align:center;padding:12px;color:#888;font-size:11px;margin-top:8px;">
    VIX 5% Weekly Suite · Auto-generated · Do not reply
  </div>

</div></body></html>"""
    return html


def check_and_send_alerts(batch, variant_states, vix_level: float,
                           to_email: str = "") -> list:
    """
    Check all positions for DEFENSIVE or EMERGENCY roll mode.
    Sends alert email if any found. Returns list of alert dicts.
    """
    import os
    from datetime import date as _date

    to_email = to_email or os.environ.get("SMTP_TO", os.environ.get("SMTP_USER", ""))

    paper_alerts = []
    real_alerts  = []

    # ── Check paper positions ──
    try:
        from trade_log import get_trade_log
        tl = get_trade_log()
        for pos in tl.get_open_diagonals():
            short = pos.current_short_leg
            if not short:
                continue
            try:
                short_dte = max(0, (_date.fromisoformat(short.expiration_date) - _date.today()).days)
            except Exception:
                short_dte = 0
            sd = _bs_delta(vix_level, float(short.strike), short_dte/365 if short_dte > 0 else 0.001)
            debit_cap = 1.50
            rm = _roll_mode(sd, short_dte, float(short.strike), vix_level, debit_cap)
            if rm["priority"] >= 2:  # DEFENSIVE or EMERGENCY
                paper_alerts.append({
                    "name":      getattr(pos, "variant_name", pos.variant_id),
                    "roll_mode": rm,
                })
    except Exception as _e:
        print(f"⚠️ Paper alert check: {_e}")

    # ── Check real positions ──
    try:
        from real_trade_log import get_real_trade_log
        rtl = get_real_trade_log()
        for pos in rtl.open_positions().values():
            short = pos.current_short_leg
            if not short:
                continue
            try:
                short_dte = max(0, (_date.fromisoformat(short.expiration_date) - _date.today()).days)
            except Exception:
                short_dte = 0
            sd = _bs_delta(vix_level, float(short.strike), short_dte/365 if short_dte > 0 else 0.001)
            debit_cap = 1.50
            rm = _roll_mode(sd, short_dte, float(short.strike), vix_level, debit_cap)
            if rm["priority"] >= 2:
                real_alerts.append({
                    "name":      pos.variant_name,
                    "roll_mode": rm,
                })
    except Exception as _e:
        print(f"⚠️ Real alert check: {_e}")

    # ── Send alert emails ──
    if paper_alerts:
        try:
            html = build_alert_email(paper_alerts, is_real=False)
            subj = f"⚡ ROLL ALERT [PAPER] — {len(paper_alerts)} position(s) need attention"
            send_email(html, to_email, subj)
            print(f"✅ Paper alert email sent: {len(paper_alerts)} positions")
        except Exception as _e:
            print(f"⚠️ Paper alert send: {_e}")

    if real_alerts:
        try:
            html = build_alert_email(real_alerts, is_real=True)
            subj = f"🚨 ROLL ALERT [LIVE 💵] — {len(real_alerts)} position(s) need attention"
            send_email(html, to_email, subj)
            print(f"✅ Real alert email sent: {len(real_alerts)} positions")
        except Exception as _e:
            print(f"⚠️ Real alert send: {_e}")

    return paper_alerts + real_alerts

'''

# Insert before main()
marker = "\ndef main():"
if "def check_and_send_alerts" not in src:
    src = src.replace(marker, alert_func + marker)
    changes += 1
    print("✓ Added build_alert_email() + check_and_send_alerts()")
else:
    print("  check_and_send_alerts already exists")

# ══════════════════════════════════════════════════════════════════════
# 6. Call check_and_send_alerts in main() after building emails
# ══════════════════════════════════════════════════════════════════════
old_main_send = '''        # 5b. Build and send real capital email (separate, orange theme)'''
new_main_send = '''        # 5b. Check and send automated roll alerts
        try:
            alerts = check_and_send_alerts(batch, variant_states, uvxy_price,
                                           to_email=args.to)
            if alerts:
                print(f"   ⚡ Roll alerts sent for {len(alerts)} position(s)")
        except Exception as _ae:
            print(f"   ⚠️ Alert check: {_ae}")

        # 5c. Build and send real capital email (separate, orange theme)'''

if old_main_send in src:
    src = src.replace(old_main_send, new_main_send)
    changes += 1
    print("✓ main(): check_and_send_alerts() called after emails")
else:
    print("✗ main() send anchor not found")

ds.write_text(src)
print(f"\n✅ {changes} changes applied to daily_signal.py")
