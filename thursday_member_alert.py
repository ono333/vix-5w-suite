#!/usr/bin/env python3
"""
thursday_member_alert.py
────────────────────────
Thursday 10am member heads-up email.
Reads from app's existing data sources:
  - ~/.vix_suite/vol_snapshots.json
  - ~/.vix_suite/current_signal_batch.json
  - ~/.vix_suite/member_signal_history.json

Purpose:
  - Delta check reminder (24h before Friday signal)
  - Current regime/phase summary
  - Roll warning if delta approaching trigger
  - Notable events next week
  - "Full signal tomorrow Friday 10am ET"

Usage:
    python3 thursday_member_alert.py --test     # send to self
    python3 thursday_member_alert.py            # send to member list
"""

from __future__ import annotations
import json
import math
import os
import smtplib
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
STORAGE_DIR         = Path.home() / ".vix_suite"
VOL_SNAPSHOTS_PATH  = STORAGE_DIR / "vol_snapshots.json"
SIGNAL_BATCH_PATH   = STORAGE_DIR / "current_signal_batch.json"
SIGNAL_HISTORY_PATH = STORAGE_DIR / "member_signal_history.json"
MEMBER_LIST_PATH    = STORAGE_DIR / "member_list.json"

# ── Email config ──────────────────────────────────────────────────────────────
SMTP_HOST  = "smtp.gmail.com"
SMTP_PORT  = 465
SMTP_USER  = os.environ.get("SMTP_USER", "")
SMTP_PASS  = os.environ.get("SMTP_PASS", "")

BARCHART_URL = "https://www.barchart.com/etfs-funds/quotes/UVXY/options"
WEBSITE_URL  = "https://volatilityharvest.com"

# ── Phase colors ──────────────────────────────────────────────────────────────
PHASE_COLOR = {
    "Compression": "#16a34a",
    "Expansion":   "#d97706",
    "Late Spike":  "#ea580c",
    "Spike Peak":  "#dc2626",
    "Collapse":    "#2563eb",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_snap() -> dict:
    if not VOL_SNAPSHOTS_PATH.exists():
        return {}
    try:
        data = json.loads(VOL_SNAPSHOTS_PATH.read_text())
        snaps = data if isinstance(data, list) else data.get("snapshots", [])
        return snaps[-1] if snaps else {}
    except Exception:
        return {}


def load_batch() -> dict:
    if not SIGNAL_BATCH_PATH.exists():
        return {}
    try:
        return json.loads(SIGNAL_BATCH_PATH.read_text())
    except Exception:
        return {}


def load_last_signal() -> dict | None:
    if not SIGNAL_HISTORY_PATH.exists():
        return None
    try:
        history = json.loads(SIGNAL_HISTORY_PATH.read_text())
        today = date.today().isoformat()
        for entry in reversed(history):
            if entry.get("signal_date", "") != today:
                return entry
        return history[-1] if history else None
    except Exception:
        return None


def load_member_list() -> list[str]:
    if MEMBER_LIST_PATH.exists():
        try:
            return json.loads(MEMBER_LIST_PATH.read_text())
        except Exception:
            pass
    return []


# ── Calendar ──────────────────────────────────────────────────────────────────

def _calendar_rows() -> str:
    try:
        from market_calendar import get_market_events
        events = get_market_events(date.today(), date.today() + timedelta(days=14))
        if not events:
            return "<tr><td style='padding:4px 8px;color:#666;font-size:12px;'>No major events next 2 weeks.</td></tr>"
        rows = ""
        for e in events[:5]:
            icon = "⛔" if getattr(e, "market_closed", False) else "⚠️"
            rows += (f"<tr><td style='padding:4px 8px;color:#444;font-size:12px;'>"
                     f"{icon} <strong>{e.date.strftime('%a %b %-d')}</strong>"
                     f" — {e.name}</td></tr>")
        return rows
    except Exception:
        return ""


# ── Delta estimate ─────────────────────────────────────────────────────────────

def _est_delta(uvxy: float, strike: float, dte_days: int,
               iv: float = 1.50) -> float:
    """Rough delta estimate for a call option."""
    try:
        T  = max(dte_days, 1) / 365.0
        d1 = (math.log(uvxy / strike) + 0.5 * iv**2 * T) / (iv * math.sqrt(T))
        from scipy.stats import norm
        return round(norm.cdf(d1), 2)
    except Exception:
        return 0.0


def _next_friday() -> str:
    d = date.today()
    days = (4 - d.weekday()) % 7 or 7
    return (d + timedelta(days=days)).strftime("%b %d")


# ── Email builder ─────────────────────────────────────────────────────────────

def build_email(snap: dict, batch: dict,
                last_signal: dict | None) -> tuple[str, str]:

    uvxy      = snap.get("uvxy", 0.0)
    vix       = snap.get("vix", 0.0)
    vix_pct   = snap.get("vix_pct", 0.5)
    regime    = snap.get("regime", "NEUTRAL")
    term      = snap.get("term_structure", "Unknown")
    phase     = snap.get("spike_label", "Expansion")
    score     = snap.get("spike_score", 0)
    collapse  = snap.get("collapse_flag", False)
    decay_p   = snap.get("uvxy_decay_pressure", "")

    rc        = PHASE_COLOR.get(phase, "#6b7280")
    today_str = date.today().strftime("%B %d, %Y")
    next_fri  = _next_friday()

    # Delta estimate on last week's signal strike
    delta_note = ""
    delta_action = ""
    delta_color = "#16a34a"
    if last_signal:
        lv   = last_signal.get("variants", {})
        v1   = lv.get("V1", {})
        v1s  = v1.get("recommended_strike", 0)
        v1d  = v1.get("dte", 7)
        if v1s and uvxy:
            days_since = (date.today() - date.fromisoformat(
                last_signal.get("signal_date", date.today().isoformat()))).days
            remaining_dte = max(0, v1d - days_since)
            est_delta = _est_delta(uvxy, v1s, remaining_dte)
            iv_label = {
                "CALM": 1.2, "DECLINING": 1.4, "NEUTRAL": 1.55,
                "STRESSED": 1.75, "EXTREME": 2.2
            }.get(regime.upper(), 1.5)
            est_delta = _est_delta(uvxy, v1s, remaining_dte, iv_label)

            if est_delta >= 0.50:
                delta_action = "🚨 Roll today — delta at danger level"
                delta_color  = "#dc2626"
            elif est_delta >= 0.35:
                delta_action = "🔄 Consider rolling tomorrow (Friday)"
                delta_color  = "#ea580c"
            elif est_delta >= 0.21:
                delta_action = "👀 Monitor — approaching roll territory"
                delta_color  = "#d97706"
            else:
                delta_action = "✅ Hold — delta safe"
                delta_color  = "#16a34a"

            delta_note = (f"Last Friday's V1 reference strike: <strong>${v1s}</strong> "
                          f"(UVXY was ${last_signal.get('uvxy_price', 0):.2f})<br>"
                          f"Estimated current delta: <strong style='color:{delta_color}'>"
                          f"~{est_delta:.2f}</strong> — "
                          f"<strong style='color:{delta_color}'>{delta_action}</strong>")

    # Context flags
    flags = ""
    if collapse:
        flags += ('<span style="background:#dbeafe;color:#1d4ed8;padding:3px 10px;'
                  'border-radius:12px;font-size:11px;margin-right:6px;">'
                  '🔵 COLLAPSE WATCH</span>')
    if decay_p == "HIGH":
        flags += ('<span style="background:#dcfce7;color:#15803d;padding:3px 10px;'
                  'border-radius:12px;font-size:11px;">'
                  '📉 UVXY Decay: HIGH</span>')

    subject = (f"📅 Volatility Harvest — Thursday Update | {today_str} | "
               f"UVXY ${uvxy:.2f} · {phase}")

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0"
       style="background:#f4f4f4;padding:20px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0"
       style="background:#ffffff;border-radius:8px;
              border:1px solid #e0e0e0;overflow:hidden;">

  <!-- HEADER -->
  <tr><td style="background:{rc};padding:18px 28px;">
    <div style="font-size:11px;color:rgba(255,255,255,0.8);
                letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">
      Volatility Harvest · Thursday Update</div>
    <div style="font-size:20px;font-weight:800;color:#ffffff;">
      📅 Friday Signal Preview — {today_str}</div>
    <div style="font-size:12px;color:rgba(255,255,255,0.85);margin-top:4px;">
      Full signal tomorrow Friday {next_fri} at 10:00am ET</div>
  </td></tr>

  <!-- MARKET STATE -->
  <tr><td style="padding:14px 28px;background:#fafafa;
                 border-bottom:1px solid #e8ede8;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td style="padding:4px 12px 4px 0;text-align:center;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">UVXY</div>
          <div style="font-size:18px;font-weight:800;color:#1a1a1a;">${uvxy:.2f}</div>
        </td>
        <td style="padding:4px 12px;text-align:center;
                   border-left:1px solid #e0e0e0;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">VIX</div>
          <div style="font-size:18px;font-weight:800;color:#1a1a1a;">{vix:.2f}</div>
        </td>
        <td style="padding:4px 12px;text-align:center;
                   border-left:1px solid #e0e0e0;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">VIX Percentile</div>
          <div style="font-size:18px;font-weight:800;color:{rc};">
            {vix_pct*100:.0f}th</div>
        </td>
        <td style="padding:4px 12px;text-align:center;
                   border-left:1px solid #e0e0e0;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">Phase</div>
          <div style="font-size:13px;font-weight:700;color:{rc};">{phase}</div>
          <div style="font-size:10px;color:#888;">Score: {score:.0f}/100</div>
        </td>
        <td style="padding:4px 0 4px 12px;text-align:center;
                   border-left:1px solid #e0e0e0;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">Term Structure</div>
          <div style="font-size:12px;font-weight:700;color:#1a1a1a;">{term}</div>
        </td>
      </tr>
    </table>
    <div style="margin-top:8px;">{flags}</div>
  </td></tr>

  <!-- DELTA CHECK -->
  <tr><td style="padding:20px 28px;border-bottom:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:10px;font-weight:700;">
      ⚡ Action Required — Check Your Delta Today</div>
    {f'<div style="font-size:12px;color:#444;margin-bottom:12px;">{delta_note}</div>' if delta_note else ''}
    <div style="font-size:12px;color:#444;margin-bottom:12px;">
      Open your broker's option chain and find your short call.
      Check the delta column.
    </div>
    <table width="100%" cellpadding="0" cellspacing="0"
           style="border-collapse:collapse;font-size:12px;">
      <tr style="background:#f0f7f2;">
        <th style="padding:7px 12px;text-align:left;color:#1E4D2B;">Delta</th>
        <th style="padding:7px 12px;text-align:left;color:#1E4D2B;">Action Today</th>
      </tr>
      <tr style="border-top:1px solid #e8ede8;background:#f0fdf4;">
        <td style="padding:7px 12px;font-weight:700;color:#16a34a;">≤ 0.20</td>
        <td style="padding:7px 12px;color:#444;">
          ✅ Hold — wait for Friday signal</td>
      </tr>
      <tr style="border-top:1px solid #e8ede8;">
        <td style="padding:7px 12px;font-weight:700;color:#d97706;">0.21 – 0.34</td>
        <td style="padding:7px 12px;color:#444;">
          👀 Monitor — check again Friday morning</td>
      </tr>
      <tr style="border-top:1px solid #e8ede8;background:#fffbf0;">
        <td style="padding:7px 12px;font-weight:700;color:#ea580c;">0.35 – 0.49</td>
        <td style="padding:7px 12px;color:#444;">
          🔄 Roll today — don't wait for Friday</td>
      </tr>
      <tr style="border-top:1px solid #e8ede8;background:#fff5f5;">
        <td style="padding:7px 12px;font-weight:700;color:#dc2626;">≥ 0.50</td>
        <td style="padding:7px 12px;color:#444;">
          🚨 Roll immediately — reply for guidance</td>
      </tr>
    </table>
    <div style="margin-top:12px;text-align:center;">
      <a href="{BARCHART_URL}"
         style="background:{rc};color:#ffffff;padding:8px 20px;
                border-radius:4px;text-decoration:none;font-size:12px;
                font-weight:700;">
        📊 Check UVXY Delta on Barchart →</a>
    </div>
  </td></tr>

  <!-- FRIDAY PREVIEW -->
  <tr><td style="padding:20px 28px;border-bottom:1px solid #e8ede8;
                 background:#f0f7f2;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:10px;font-weight:700;">
      📊 Tomorrow: Friday Signal at 10:00am ET</div>
    <div style="font-size:13px;color:#444;line-height:1.8;">
      Current conditions: <strong style="color:{rc};">{phase}</strong>
      · VIX {vix_pct*100:.0f}th percentile · {term}<br>
      Tomorrow's email will include the full strike signal,
      UVXY movement table, and execution guidance.<br><br>
      <strong>If your short expires tomorrow (Friday):</strong><br>
      You can let it expire at 4:00pm ET and sell a new short
      using tomorrow's signal — or roll early today using the
      delta table above.
    </div>
  </td></tr>

  <!-- CALENDAR -->
  <tr><td style="padding:16px 28px;border-bottom:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:10px;font-weight:700;">
      Notable Events — Next 2 Weeks</div>
    <table width="100%" cellpadding="0" cellspacing="0">
      {_calendar_rows() or
       "<tr><td style='padding:4px 8px;color:#888;font-size:12px;'>"
       "No major events detected.</td></tr>"}
    </table>
  </td></tr>

  <!-- FOOTER -->
  <tr><td style="padding:14px 28px;background:#f9f9f9;">
    <div style="font-size:11px;color:#888;line-height:1.8;text-align:center;">
      <a href="{WEBSITE_URL}" style="color:{rc};font-weight:700;">
        Volatility Harvest</a>
      &nbsp;·&nbsp; Questions? Reply to this email.
      &nbsp;·&nbsp; Full signal tomorrow Friday at 10:00am ET.<br>
      <span style="color:#bbb;font-size:10px;">
        For informational purposes only. Not investment advice.
        Options trading involves significant risk of loss.</span>
    </div>
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""

    return subject, html


# ── Send ──────────────────────────────────────────────────────────────────────

def send_email(subject: str, html: str, recipients: list[str]):
    if not SMTP_USER or not SMTP_PASS:
        print("❌ SMTP credentials not set")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = ", ".join(recipients)
    msg["Reply-To"] = SMTP_USER
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(SMTP_USER, recipients, msg.as_string())
    print(f"✅ Sent to {len(recipients)} recipient(s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Send to self only")
    args = parser.parse_args()

    snap        = load_snap()
    batch       = load_batch()
    last_signal = load_last_signal()

    if not snap:
        print("❌ No vol snapshot — run the app first")
        return

    uvxy  = snap.get("uvxy", 0.0)
    phase = snap.get("spike_label", "?")
    print(f"   UVXY: ${uvxy:.2f} | Phase: {phase}")

    subject, html = build_email(snap, batch, last_signal)

    recipients = [SMTP_USER] if args.test else load_member_list()
    if not recipients:
        print("⚠️  No members — sending to self")
        recipients = [SMTP_USER]

    print(f"   Sending to {len(recipients)} recipient(s)…")
    send_email(subject, html, recipients)


if __name__ == "__main__":
    main()
