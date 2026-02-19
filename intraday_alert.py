#!/usr/bin/env python3
"""
Intraday position monitor — runs every 30 min during market hours.
Sends URGENT email only when action is required.
Checks: delta_trigger, spike_guard, ITM threat, expiry-day roll.
"""
import os, sys, json, smtplib
from pathlib import Path
from datetime import date, datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, str(Path(__file__).parent))

from roll_manager import evaluate_roll, RollDecision
from trade_log import get_trade_log

SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
ALERT_SUBJECT_PREFIX = "[VIX ALERT]"

# Action levels that warrant an immediate email
URGENT_ACTIONS = {"roll_early_itm", "roll_early_delta", "roll_now"}
WATCH_ACTIONS  = {"spike_guard_hold"}

def get_uvxy_price() -> float:
    try:
        import yfinance as yf
        t = yf.Ticker("UVXY")
        data = t.history(period="1d", interval="1m")
        return float(data["Close"].iloc[-1])
    except:
        return 0.0

def build_alert_email(alerts: list, watches: list, uvxy: float) -> tuple[str,str]:
    now = datetime.now().strftime("%b %d %Y %I:%M %p ET")
    urgency = "🚨 URGENT ACTION REQUIRED" if alerts else "👁 WATCH ALERT"
    subject = f"{ALERT_SUBJECT_PREFIX} {urgency} — UVXY ${uvxy:.2f} — {now}"

    rows = ""
    for a in alerts:
        color = "#ff3366" if a["action"] == "roll_early_itm" else "#ff9800"
        rows += f"""
        <tr style="background:#1a0a0a">
          <td style="padding:10px 14px;font-weight:700;color:#fff">{a['variant']}</td>
          <td style="padding:10px 14px;color:{color};font-weight:700">{a['action'].upper().replace('_',' ')}</td>
          <td style="padding:10px 14px;color:#aaa">{a['reason']}</td>
          <td style="padding:10px 14px;color:#ff9800">est BB ${a['est_bb']:.2f}</td>
        </tr>"""

    for w in watches:
        rows += f"""
        <tr style="background:#0a0a1a">
          <td style="padding:10px 14px;font-weight:700;color:#fff">{w['variant']}</td>
          <td style="padding:10px 14px;color:#38b4ff;font-weight:700">SPIKE GUARD</td>
          <td style="padding:10px 14px;color:#aaa">{w['reason']}</td>
          <td style="padding:10px 14px;color:#38b4ff">holding</td>
        </tr>"""

    html = f"""
    <div style="background:#05080a;color:#ccc;font-family:'IBM Plex Mono',monospace;
                padding:24px;max-width:700px;margin:0 auto">
      <div style="font-size:22px;font-weight:800;color:#fff;margin-bottom:4px">
        {"🚨" if alerts else "👁"} VIX 5W Suite — {"URGENT ALERT" if alerts else "Watch Alert"}
      </div>
      <div style="font-size:11px;color:#555;margin-bottom:20px">{now} · UVXY ${uvxy:.2f}</div>
      <table style="width:100%;border-collapse:collapse;background:#0c1215;
                    border:1px solid #1a252c;border-radius:6px;overflow:hidden">
        <thead>
          <tr style="background:#111820">
            <th style="padding:8px 14px;text-align:left;font-size:10px;
                       letter-spacing:2px;color:#444;text-transform:uppercase">Variant</th>
            <th style="padding:8px 14px;text-align:left;font-size:10px;
                       letter-spacing:2px;color:#444;text-transform:uppercase">Action</th>
            <th style="padding:8px 14px;text-align:left;font-size:10px;
                       letter-spacing:2px;color:#444;text-transform:uppercase">Reason</th>
            <th style="padding:8px 14px;text-align:left;font-size:10px;
                       letter-spacing:2px;color:#444;text-transform:uppercase">Cost Est.</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      {"<div style='margin-top:16px;padding:12px;background:#1a0000;border:1px solid #ff3366;border-radius:4px;color:#ff6666;font-size:12px'>⚠ Roll before market close today to avoid assignment risk.</div>" if alerts else ""}
      <div style="margin-top:20px;font-size:10px;color:#333">
        VIX 5% Weekly Suite · Intraday Monitor · Do not reply
      </div>
    </div>"""
    return subject, html

def send_email(subject: str, html: str):
    if not SMTP_USER or not SMTP_PASS:
        print("❌ SMTP credentials missing")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = SMTP_USER
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, SMTP_USER, msg.as_string())
        print(f"✅ Alert sent: {subject[:60]}")
        return True
    except Exception as e:
        print(f"❌ Send failed: {e}")
        return False

def main():
    today = date.today()
    # Skip weekends
    if today.weekday() >= 5:
        print("Weekend — skipping")
        return

    uvxy_price = get_uvxy_price()
    if uvxy_price == 0.0:
        print("Could not fetch UVXY price — skipping")
        return

    print(f"UVXY: ${uvxy_price:.2f}  |  {datetime.now().strftime('%H:%M ET')}")

    tl = get_trade_log()
    alerts, watches = [], []

    for pid, pos in tl.diagonal_positions.items():
        if pos.status != "open":
            continue
        short = pos.current_short_leg
        if not short:
            continue

        try:
            exp_date = date.fromisoformat(short.expiration_date)
            dte = max(0, (exp_date - today).days)
        except:
            dte = 0

        decision = evaluate_roll(
            dte_remaining    = dte,
            short_delta      = getattr(short, "delta", None),
            uvxy_price       = uvxy_price,
            short_strike     = short.strike,
            variant_params   = {"roll_dte_days": 0, "delta_trigger": 0.45,
                                "spike_guard_days": 2},
            last_spike_date  = None,
            original_premium = short.entry_credit,
        )

        entry = dict(
            variant  = pos.variant_name,
            action   = decision.action,
            reason   = decision.reason[:80],
            est_bb   = decision.expected_bb,
            dte      = dte,
            strike   = short.strike,
        )

        print(f"  {pos.variant_name:<25} DTE={dte}  strike=${short.strike}  "
              f"UVXY=${uvxy_price:.2f}  → {decision.action}")

        if decision.action in URGENT_ACTIONS:
            alerts.append(entry)
        elif decision.action in WATCH_ACTIONS:
            watches.append(entry)

    if alerts or watches:
        subject, html = build_alert_email(alerts, watches, uvxy_price)
        send_email(subject, html)
    else:
        print("✅ All positions HOLD — no alert needed")

if __name__ == "__main__":
    main()
