#!/usr/bin/env python3
"""
friday_member_signal.py
───────────────────────
Friday 10am member signal email.
Reads exclusively from the app's existing data sources:
  - ~/.vix_suite/current_signal_batch.json   (variant activation, strikes)
  - ~/.vix_suite/vol_snapshots.json          (spike phase, regime, UVXY)

Member email structure (three states):
  A) Short expires today       → Roll signal
  B) Short open, not expiring  → Delta check + early roll guidance
  C) No position               → Fresh entry (long first, then short)

Usage:
    python3 friday_member_signal.py --preview   # saves preview.html
    python3 friday_member_signal.py --test      # sends to SMTP_USER only
    python3 friday_member_signal.py             # sends to member list
"""

from __future__ import annotations
import argparse
import json
import math
import os
import smtplib
import sys
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
STORAGE_DIR         = Path.home() / ".vix_suite"
SIGNAL_BATCH_PATH   = STORAGE_DIR / "current_signal_batch.json"
VOL_SNAPSHOTS_PATH  = STORAGE_DIR / "vol_snapshots.json"
SIGNAL_HISTORY_PATH = STORAGE_DIR / "member_signal_history.json"
MEMBER_LIST_PATH    = STORAGE_DIR / "member_list.json"

# ── Email config ──────────────────────────────────────────────────────────────
SMTP_HOST  = "smtp.gmail.com"
SMTP_PORT  = 465
SMTP_USER  = os.environ.get("SMTP_USER", "")
SMTP_PASS  = os.environ.get("SMTP_PASS", "")
FROM_ADDR  = SMTP_USER
REPLY_TO   = SMTP_USER

BARCHART_URL = "https://www.barchart.com/etfs-funds/quotes/UVXY/options"
SIZING_URL   = "https://volatilityharvest.com/sizing"
WEBSITE_URL  = "https://volatilityharvest.com"

# ── Spike phase display ───────────────────────────────────────────────────────
PHASE_CONFIG = {
    "Compression": {
        "color": "#16a34a", "emoji": "🟢",
        "title": "Compression — Low Volatility",
        "body": ("VIX is compressing. UVXY is in structural decay. "
                 "This is the ideal environment for V1 Income Harvester. "
                 "Sell premium aggressively — contango is your tailwind."),
    },
    "Expansion": {
        "color": "#d97706", "emoji": "🟡",
        "title": "Expansion — Volatility Rising",
        "body": ("Volatility is expanding but not yet at peak. "
                 "V3 Shock Absorber is optimal. "
                 "Widen strikes to 8–12% OTM. Do not add new long legs. "
                 "Roll shorts proactively if delta > 0.40."),
    },
    "Late Spike": {
        "color": "#ea580c", "emoji": "🟠",
        "title": "Late Spike — Hold Wide",
        "body": ("Spike is maturing. V3 + V5 hold wide positions. "
                 "No new entries. Monitor delta closely. "
                 "Prepare to transition to V4 if spike peaks."),
    },
    "Spike Peak": {
        "color": "#dc2626", "emoji": "🔴",
        "title": "Spike Peak — Harvest Premium",
        "body": ("VIX is at or near peak. IV is maximum. "
                 "V4 Tail Hunter is active — sell very OTM calls. "
                 "This is the highest-premium window. "
                 "Use 21 DTE. Mean reversion is your edge from here."),
    },
    "Collapse": {
        "color": "#2563eb", "emoji": "🔵",
        "title": "Collapse — Re-enter Income",
        "body": ("VIX is collapsing. IV is compressing fast. "
                 "V1 + V2 re-enter. Collect while premium is still rich "
                 "from the preceding spike. Transition back to income mode."),
    },
}
DEFAULT_PHASE = {
    "color": "#6b7280", "emoji": "⚪",
    "title": "Mixed Conditions",
    "body": ("Market conditions are transitional. "
             "Follow the variant activation table below. "
             "Prioritize delta safety over premium."),
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_vol_snapshot() -> dict:
    if not VOL_SNAPSHOTS_PATH.exists():
        return {}
    try:
        data = json.loads(VOL_SNAPSHOTS_PATH.read_text())
        snaps = data if isinstance(data, list) else data.get("snapshots", [])
        return snaps[-1] if snaps else {}
    except Exception as e:
        print(f"⚠️  Could not load vol snapshots: {e}")
        return {}


def load_signal_batch() -> dict:
    if not SIGNAL_BATCH_PATH.exists():
        return {}
    try:
        return json.loads(SIGNAL_BATCH_PATH.read_text())
    except Exception as e:
        print(f"⚠️  Could not load signal batch: {e}")
        return {}


def get_last_signal() -> dict | None:
    """Return last signal from a PREVIOUS date (not today)."""
    if not SIGNAL_HISTORY_PATH.exists():
        return None
    try:
        history = json.loads(SIGNAL_HISTORY_PATH.read_text())
        today = date.today().isoformat()
        # Find most recent entry from a different date
        for entry in reversed(history):
            if entry.get("signal_date", "") != today:
                return entry
        # Fallback: return last entry even if same date (for testing)
        return history[-1] if history else None
    except Exception:
        return None


def save_signal_history(record: dict):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    if SIGNAL_HISTORY_PATH.exists():
        try:
            history = json.loads(SIGNAL_HISTORY_PATH.read_text())
        except Exception:
            pass
    history.append(record)
    history = history[-12:]
    SIGNAL_HISTORY_PATH.write_text(json.dumps(history, indent=2))


# ── Variant helpers ───────────────────────────────────────────────────────────

def _build_role_map(batch: dict) -> dict:
    role_map = {}
    for v in batch.get("variants", []):
        role = v.get("role", "")
        for key, tag in [("V1","v1"),("V2","v2"),("V3","v3"),
                         ("V4","v4"),("V5","v5")]:
            if tag in role:
                role_map[key] = v
    return role_map


def _activation(v: dict, regime: str, phase: str = "") -> str:
    """Returns 'full' | 'reduced' | 'off'.
    Phase takes priority over regime when Collapse detected."""
    role = v.get("role", "")
    # Collapse phase: V1+V2 re-enter, V3/V4 roll only
    if phase.lower() == "collapse":
        if "v1" in role: return "full"
        if "v2" in role: return "full"
        if "v3" in role: return "off"
        if "v4" in role: return "off"
        if "v5" in role: return "full"
    r = regime.lower()
    active     = [x.lower() for x in v.get("active_in_regimes", [])]
    suppressed = [x.lower() for x in v.get("suppressed_in_regimes", [])]
    if r in suppressed:
        return "off"
    if r in active:
        return "full"
    return "reduced"


def _delta_ceil(v: dict) -> float:
    """Approximate delta ceiling from sigma_mult."""
    sm = v.get("sigma_mult", 1.0)
    if sm <= 0.8: return 0.28
    if sm <= 0.9: return 0.25
    if sm <= 1.0: return 0.22
    if sm <= 1.2: return 0.18
    return 0.15


def _next_friday(dte: int) -> str:
    target = date.today() + timedelta(days=dte)
    days   = (4 - target.weekday()) % 7
    target += timedelta(days=days)
    return target.strftime("%b %d")


def _est_credit(uvxy: float, strike: float, dte_days: int,
                iv: float = 1.50) -> tuple[float, float]:
    """Rough BS credit estimate. Returns (lo, hi)."""
    try:
        T  = dte_days / 365.0
        d1 = (math.log(uvxy / strike) + 0.5 * iv**2 * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        from scipy.stats import norm
        cr = max(0.05, uvxy * norm.cdf(d1) - strike * norm.cdf(d2))
    except Exception:
        cr = max(0.10, 1.20)
    return round(cr * 0.85, 2), round(cr * 1.15, 2)


def _iv_for_regime(regime: str) -> float:
    return {"calm":1.20,"declining":1.40,"neutral":1.55,
            "stressed":1.75,"extreme":2.20}.get(regime.lower(), 1.50)


# ── Movement table ────────────────────────────────────────────────────────────

def _delta_ceiling_banner(batch: dict, regime: str, phase: str = "") -> str:
    """Prominent delta ceiling display above movement table."""
    role_map = _build_role_map(batch)
    labels = {"V1":"Income","V2":"Mean Rev","V3":"Shock","V4":"Tail","V5":"Regime"}
    cells = ""
    for key in ["V1","V2","V3","V4","V5"]:
        v   = role_map.get(key, {})
        act = _activation(v, regime, phase) if v else "off"
        dc  = _delta_ceil(v) if v else 0.22
        if act == "full":
            bg = "#16a34a"; fc = "#ffffff"; op = "1.0"
        elif act == "reduced":
            bg = "#d97706"; fc = "#ffffff"; op = "1.0"
        else:
            bg = "#e5e7eb"; fc = "#9ca3af"; op = "0.7"
        cells += (
            f'<td style="padding:8px 4px;text-align:center;opacity:{op};">'
            f'<div style="background:{bg};color:{fc};padding:6px 8px;'
            f'border-radius:6px;display:inline-block;min-width:58px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:1px;">{key} {labels[key]}</div>'
            f'<div style="font-size:20px;font-weight:900;margin-top:2px;">\u03b4\u2264{dc:.2f}</div>'
            f'</div></td>'
        )
    return (
        '<div style="background:#f0f7f2;border:2px solid #1E4D2B;border-radius:8px;'
        'padding:12px 16px;margin-bottom:12px;">'
        '<div style="font-size:10px;font-weight:700;color:#1E4D2B;letter-spacing:2px;'
        'text-transform:uppercase;margin-bottom:8px;text-align:center;">'
        '&#9889; Delta Ceilings &#8212; Do Not Exceed</div>'
        f'<table width="100%" cellpadding="0" cellspacing="0"><tr>{cells}</tr></table>'
        '<div style="font-size:11px;color:#555;margin-top:8px;text-align:center;">'
        'If your strike&#39;s delta exceeds ceiling &rarr; '
        '<strong>move to next strike further OTM</strong></div></div>'
    )

VARIANT_LABELS = {
    "V1": "Income Harvester", "V2": "Mean Reversion",
    "V3": "Shock Absorber",   "V4": "Tail Hunter",
    "V5": "Regime Allocator",
}

def _movement_table(uvxy: float, batch: dict, regime: str, phase: str = "") -> str:
    role_map = _build_role_map(batch)
    all_keys = ["V1","V2","V3","V4","V5"]
    iv       = _iv_for_regime(regime)

    center = round(uvxy / 2) * 2
    prices = [center + (i - 4) * 2 for i in range(9)]

    # Headers — all 5 variants, activation from app's batch
    variant_headers = ""
    for key in all_keys:
        v   = role_map.get(key, {})
        act = _activation(v, regime, phase) if v else "off"
        dc  = _delta_ceil(v) if v else 0.22
        if act == "full":
            ind = "✅"; ac = "#16a34a"
        elif act == "reduced":
            ind = "⚠️"; ac = "#d97706"
        else:
            ind = "❌"; ac = "#9ca3af"
        variant_headers += (
            f'<th style="padding:8px 4px;text-align:left;color:#1E4D2B;'
            f'font-size:11px;">{ind} {key}<br>'
            f'<span style="font-weight:400;color:{ac};font-size:10px;">'
            f'{VARIANT_LABELS[key]}<br>δ≤{dc:.2f}</span></th>'
        )

    rows = ""
    for p in prices:
        if p <= 0:
            continue
        pct_chg = (p - uvxy) / uvxy * 100
        is_sig  = abs(p - center) < 1
        bg      = "#e8f5e9" if is_sig else "#ffffff"
        fw      = "700" if is_sig else "400"

        # DTE by distance from signal price
        if p > uvxy * 1.20:
            dte_val = 21; note = "⚠️ 21d spike";  nc = "#dc2626"
        elif p > uvxy * 1.08:
            dte_val = 14; note = "⚠️ 14d";         nc = "#d97706"
        else:
            dte_val = 7;  note = "← Signal" if is_sig else ""; nc = "#1E4D2B"

        # Per-variant cells — all 5, using app's strike offsets
        variant_cells = ""
        for key in all_keys:
            v      = role_map.get(key, {})
            act    = _activation(v, regime, phase) if v else "off"
            offset = v.get("short_strike_offset", 2.0) if v else 2.0
            strike = round(p + offset)

            if p > uvxy * 1.20:
                cr_str   = "debit~ok †"
                cr_color = "#dc2626"
            else:
                cr_lo, cr_hi = _est_credit(p, strike, dte_val, iv)
                cr_str   = f"${cr_lo:.2f}–{cr_hi:.2f}"
                cr_color = "#1E4D2B" if p <= uvxy * 1.08 else "#d97706"

            if act == "off":
                variant_cells += (
                    f'<td style="padding:8px 4px;font-size:11px;color:#aaa;">'
                    f'${strike}<br>'
                    f'<span style="font-size:10px;color:{cr_color};opacity:0.7;">'
                    f'{cr_str}</span><br>'
                    f'<span style="font-size:9px;color:#bbb;">roll only</span></td>'
                )
            elif act == "reduced":
                variant_cells += (
                    f'<td style="padding:8px 4px;font-weight:{fw};'
                    f'font-size:12px;color:#d97706;">'
                    f'${strike}<br>'
                    f'<span style="font-size:10px;color:{cr_color};font-weight:400;">'
                    f'{cr_str}</span><br>'
                    f'<span style="font-size:9px;color:#d97706;">50% size</span></td>'
                )
            else:
                variant_cells += (
                    f'<td style="padding:8px 4px;font-weight:{fw};'
                    f'font-size:12px;color:#1a1a1a;">'
                    f'${strike}<br>'
                    f'<span style="font-size:10px;color:{cr_color};font-weight:400;">'
                    f'{cr_str}</span></td>'
                )

        rows += f"""
        <tr style="background:{bg};border-top:1px solid #e8ede8;">
          <td style="padding:8px 8px;font-weight:{fw};color:#1a1a1a;
                     white-space:nowrap;">${p:.0f}
            <span style="font-size:10px;color:#999;margin-left:2px;">
              ({pct_chg:+.0f}%)</span></td>
          <td style="padding:8px 6px;font-size:11px;color:#555;
                     white-space:nowrap;">{dte_val}d</td>
          {variant_cells}
          <td style="padding:8px 6px;font-size:11px;color:{nc};">{note}</td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead>
        <tr style="background:#f0f7f2;">
          <th style="padding:8px 8px;text-align:left;color:#1E4D2B;">UVXY price</th>
          <th style="padding:8px 6px;text-align:left;color:#1E4D2B;">DTE</th>
          {variant_headers}
          <th style="padding:8px 6px;text-align:left;color:#1E4D2B;">Note</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <div style="font-size:10px;color:#999;margin-top:6px;padding:0 4px;">
      Strikes from app signal batch. Credit estimates for guidance only —
      verify live chain. Default 7 DTE — extend to 14d if delta ceiling
      exceeded at 7 DTE. 21d only in spike territory (UVXY +20%+).
      † theoretical, no historical fills at this level.
    </div>"""


# ── Variant activation table ──────────────────────────────────────────────────

def _variant_table(batch: dict, uvxy: float, regime: str, phase: str = "") -> str:
    role_map = _build_role_map(batch)
    rows = ""
    for key in ["V1","V2","V3","V4","V5"]:
        v   = role_map.get(key, {})
        act = _activation(v, regime, phase) if v else "off"
        lbl = VARIANT_LABELS[key]

        if act == "full":
            bg = "#f0f7f2"; bb = "#16a34a"
            bt = "✅ ACTIVE"; sn = "Full allocation"
        elif act == "reduced":
            bg = "#fffbf0"; bb = "#d97706"
            bt = "⚠️ REDUCED"; sn = "50% of normal size"
        else:
            bg = "#fafafa"; bb = "#9ca3af"
            bt = "❌ INACTIVE"; sn = "Roll only — no new entries"

        if v:
            offset = v.get("short_strike_offset", 2.0)
            strike = round(uvxy + offset)
            dte_d  = v.get("short_dte_weeks", 1) * 7
            expiry = _next_friday(dte_d)
            detail = (f"Ref. strike: ~${strike} &nbsp;·&nbsp; "
                      f"~{dte_d} DTE (exp ~{expiry}) &nbsp;·&nbsp; "
                      f"Verify credit + delta on live chain")
        else:
            detail = "—"

        rows += f"""
        <tr style="background:{bg};border-top:1px solid #e8ede8;">
          <td style="padding:10px 12px;font-weight:600;color:#1a1a1a;width:28%;">
            {key} {lbl}</td>
          <td style="padding:10px 12px;width:20%;">
            <span style="background:{bb};color:#fff;padding:3px 8px;
                         border-radius:12px;font-size:11px;font-weight:700;">
              {bt}</span></td>
          <td style="padding:10px 12px;font-size:11px;color:#444;">{sn}</td>
          <td style="padding:10px 12px;font-size:11px;color:#666;">{detail}</td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead>
        <tr style="background:#f0f7f2;">
          <th style="padding:8px 12px;text-align:left;color:#1E4D2B;">Variant</th>
          <th style="padding:8px 12px;text-align:left;color:#1E4D2B;">Status</th>
          <th style="padding:8px 12px;text-align:left;color:#1E4D2B;">Sizing</th>
          <th style="padding:8px 12px;text-align:left;color:#1E4D2B;">This week</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


# ── Calendar ──────────────────────────────────────────────────────────────────

def _calendar_section() -> str:
    try:
        from market_calendar import get_market_events
        events = get_market_events(date.today(), date.today() + timedelta(days=21))
        if not events:
            return ""
        lines = ""
        for e in events[:6]:
            icon = "⛔" if getattr(e, "market_closed", False) else "⚠️"
            lines += (f"<tr><td style='padding:4px 8px;color:#555;font-size:12px;'>"
                      f"{icon} <strong>{e.date.strftime('%a %b %-d')}</strong>"
                      f" — {e.name}</td></tr>")
        return f"<table style='width:100%;border-collapse:collapse;'>{lines}</table>"
    except Exception:
        return ""


# ── Email builder ─────────────────────────────────────────────────────────────

def build_email(snap: dict, batch: dict,
                last_signal: dict | None) -> tuple[str, str]:

    uvxy        = snap.get("uvxy", 0.0)
    vix         = snap.get("vix", 0.0)
    vix_pct     = snap.get("vix_pct", 0.5)
    regime      = snap.get("regime", "NEUTRAL")
    term        = snap.get("term_structure", "Unknown")
    phase       = snap.get("spike_label", "Expansion")
    spike_score = snap.get("spike_score", 0)
    decay_p     = snap.get("uvxy_decay_pressure", "")
    collapse    = snap.get("collapse_flag", False)
    aftershock  = snap.get("aftershock_risk", "LOW")

    pcfg = PHASE_CONFIG.get(phase, DEFAULT_PHASE)
    rc   = pcfg["color"]

    today_str = date.today().strftime("%B %d, %Y")
    days_fri  = (4 - date.today().weekday()) % 7 or 7
    next_exp  = (date.today() + timedelta(days=days_fri)).strftime("%b %d")

    last_note = ""
    transition_html = ""
    if last_signal:
        ld   = last_signal.get("signal_date", "")
        lu   = last_signal.get("uvxy_price", 0)
        lv   = last_signal.get("variants", {})
        v1s  = lv.get("V1", {}).get("recommended_strike", 0)
        last_phase = last_signal.get("phase", "")
        if v1s:
            last_note = (f"Last week's V1 reference strike: <strong>${v1s}</strong> "
                         f"(signal {ld}, UVXY was ${lu:.2f})")

        # Detect phase transition
        transition_from = None
        if last_phase in ("Expansion", "Late Spike", "Spike Peak") and phase == "Collapse":
            transition_from = "V4"
            transition_to   = "V2"
        elif last_phase in ("Spike Peak", "Exhaustion") and phase in ("Collapse", "Compression"):
            transition_from = "V4"
            transition_to   = "V2"

        if transition_from:
            v2s = lv.get("V2", {}).get("recommended_strike", 0)
            transition_html = f"""
            <tr><td style="padding:16px 28px;border-bottom:1px solid #e8ede8;
                           background:#fffbf0;border-left:4px solid #d97706;">
              <div style="font-size:10px;letter-spacing:2px;color:#d97706;
                          text-transform:uppercase;margin-bottom:10px;font-weight:700;">
                📌 Phase Transition: {last_phase} → {phase}</div>
              <div style="font-size:13px;font-weight:700;color:#1a1a1a;margin-bottom:8px;">
                Last week's signal was {transition_from} Tail Hunter.<br>
                This week: transition your {transition_from} to {transition_to} Mean Reversion.
              </div>
              <div style="font-size:12px;color:#444;line-height:2.0;">
                <strong>How to transition:</strong><br>
                1. <strong>Keep your existing long leg</strong> — do NOT close it<br>
                2. When your current short expires →
                   Sell to Open: <strong>{transition_to} short call</strong>,
                   target credit ${lv.get("V2", {}).get("recommended_strike", "~$45")}
                   strike area, 7 DTE<br>
                3. Your position is now <strong>{transition_to} Mean Reversion</strong><br><br>
                <span style="color:#d97706;">
                  ⚠️ If you need personal guidance on this transition,
                  reply to this email.
                </span>
              </div>
            </td></tr>"""

    flags = ""
    if collapse:
        flags += ('<span style="background:#dbeafe;color:#1d4ed8;padding:3px 10px;'
                  'border-radius:12px;font-size:11px;margin-right:8px;">'
                  '🔵 COLLAPSE WATCH — VIX falling</span>')
    if decay_p == "HIGH":
        flags += ('<span style="background:#dcfce7;color:#15803d;padding:3px 10px;'
                  'border-radius:12px;font-size:11px;margin-right:8px;">'
                  '📉 UVXY Decay: HIGH</span>')
    if aftershock == "HIGH":
        flags += ('<span style="background:#fef3c7;color:#b45309;padding:3px 10px;'
                  'border-radius:12px;font-size:11px;">'
                  '⚡ Aftershock Risk: HIGH</span>')

    subject = (f"📊 Volatility Harvest Signal | {today_str} | "
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
<table width="660" cellpadding="0" cellspacing="0"
       style="background:#ffffff;border-radius:8px;
              border:1px solid #e0e0e0;overflow:hidden;">

  <!-- HEADER -->
  <tr><td style="background:{rc};padding:20px 28px;">
    <div style="font-size:11px;color:rgba(255,255,255,0.8);
                letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">
      Volatility Harvest · Weekly Signal</div>
    <div style="font-size:22px;font-weight:800;color:#ffffff;">
      {pcfg['emoji']} {pcfg['title']}</div>
    <div style="font-size:12px;color:rgba(255,255,255,0.85);margin-top:6px;">
      {today_str} &nbsp;·&nbsp; Execute today from 10:00am ET</div>
  </td></tr>

  <!-- MARKET STATE -->
  <tr><td style="padding:16px 28px;background:#fafafa;
                 border-bottom:1px solid #e8ede8;">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td style="padding:4px 12px 4px 0;text-align:center;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">UVXY</div>
          <div style="font-size:20px;font-weight:800;color:#1a1a1a;">${uvxy:.2f}</div>
        </td>
        <td style="padding:4px 12px;text-align:center;border-left:1px solid #e0e0e0;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">VIX</div>
          <div style="font-size:20px;font-weight:800;color:#1a1a1a;">{vix:.2f}</div>
        </td>
        <td style="padding:4px 12px;text-align:center;border-left:1px solid #e0e0e0;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">VIX Percentile</div>
          <div style="font-size:20px;font-weight:800;color:{rc};">
            {vix_pct*100:.0f}th</div>
        </td>
        <td style="padding:4px 12px;text-align:center;border-left:1px solid #e0e0e0;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">Phase</div>
          <div style="font-size:13px;font-weight:700;color:{rc};">{phase}</div>
          <div style="font-size:10px;color:#888;">Score: {spike_score:.0f}/100</div>
        </td>
        <td style="padding:4px 0 4px 12px;text-align:center;
                   border-left:1px solid #e0e0e0;">
          <div style="font-size:10px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">Term Structure</div>
          <div style="font-size:12px;font-weight:700;color:#1a1a1a;">{term}</div>
          <div style="font-size:10px;color:#888;">Next exp: {next_exp}</div>
        </td>
      </tr>
    </table>
    <div style="margin-top:10px;">{flags}</div>
    <div style="margin-top:10px;font-size:13px;color:#444;line-height:1.6;">
      {pcfg['body']}</div>
  </td></tr>

  <!-- SITUATION -->
  <tr><td style="padding:20px 28px;border-bottom:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:12px;font-weight:700;">
      What's Your Situation?</div>
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td style="padding:10px 12px;background:#e8f5e9;border-radius:6px;
                   border-left:3px solid #16a34a;vertical-align:top;width:31%;">
          <div style="font-weight:700;color:#1a1a1a;font-size:12px;">
            A) Short expires TODAY</div>
          <div style="font-size:11px;color:#555;margin-top:4px;line-height:1.5;">
            Let expire at 4pm ET, then sell new short using signal below.
            Or buy to close now and sell immediately.</div>
        </td>
        <td style="width:3%;"></td>
        <td style="padding:10px 12px;background:#fff8e6;border-radius:6px;
                   border-left:3px solid #d97706;vertical-align:top;width:31%;">
          <div style="font-weight:700;color:#1a1a1a;font-size:12px;">
            B) Short open, not expiring</div>
          <div style="font-size:11px;color:#555;margin-top:4px;line-height:1.5;">
            Check your delta below. If approaching 0.35+,
            use the signal below to roll early.</div>
        </td>
        <td style="width:3%;"></td>
        <td style="padding:10px 12px;background:#eff6ff;border-radius:6px;
                   border-left:3px solid #2563eb;vertical-align:top;width:31%;">
          <div style="font-weight:700;color:#1a1a1a;font-size:12px;">
            C) No open position</div>
          <div style="font-size:11px;color:#555;margin-top:4px;line-height:1.5;">
            Buy long call (LEAP) <strong>first</strong>, then sell short.
            Never sell short without owning the long.</div>
        </td>
      </tr>
    </table>
  </td></tr>

  {transition_html}
  <!-- POSITION CHECK -->
  <tr><td style="padding:20px 28px;border-bottom:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:10px;font-weight:700;">
      B) Position Check — Check Your Delta Now</div>
    {f'<div style="font-size:12px;color:#444;margin-bottom:10px;">{last_note}</div>' if last_note else ''}
    <table width="100%" cellpadding="0" cellspacing="0"
           style="border-collapse:collapse;font-size:12px;">
      <tr style="background:#f0f7f2;">
        <th style="padding:8px 12px;text-align:left;color:#1E4D2B;">Delta</th>
        <th style="padding:8px 12px;text-align:left;color:#1E4D2B;">Action</th>
      </tr>
      <tr style="border-top:1px solid #e8ede8;background:#f0fdf4;">
        <td style="padding:8px 12px;font-weight:700;color:#16a34a;">≤ 0.20</td>
        <td style="padding:8px 12px;color:#444;">✅ Hold — no action needed today</td>
      </tr>
      <tr style="border-top:1px solid #e8ede8;">
        <td style="padding:8px 12px;font-weight:700;color:#d97706;">0.21 – 0.34</td>
        <td style="padding:8px 12px;color:#444;">👀 Monitor — prepare to roll next week</td>
      </tr>
      <tr style="border-top:1px solid #e8ede8;background:#fffbf0;">
        <td style="padding:8px 12px;font-weight:700;color:#ea580c;">0.35 – 0.49</td>
        <td style="padding:8px 12px;color:#444;">🔄 Roll this week — use signal below</td>
      </tr>
      <tr style="border-top:1px solid #e8ede8;background:#fff5f5;">
        <td style="padding:8px 12px;font-weight:700;color:#dc2626;">≥ 0.50</td>
        <td style="padding:8px 12px;color:#444;">🚨 Roll today before close</td>
      </tr>
    </table>
    <div style="margin-top:12px;text-align:center;">
      <a href="{BARCHART_URL}"
         style="background:{rc};color:#ffffff;padding:8px 20px;
                border-radius:4px;text-decoration:none;font-size:12px;
                font-weight:700;">📊 Check UVXY Delta on Barchart →</a>
    </div>
  </td></tr>

  <!-- NEW ENTRY -->
  <tr><td style="padding:20px 28px;border-bottom:1px solid #e8ede8;
                 background:#eff6ff;">
    <div style="font-size:10px;letter-spacing:2px;color:#2563eb;
                text-transform:uppercase;margin-bottom:10px;font-weight:700;">
      C) New Entry — Long Leg First</div>
    <div style="font-size:12px;color:#444;line-height:1.8;">
      <strong>Step 1 — Buy to Open (Long Call / LEAP):</strong><br>
      Target: UVXY Sep 2026 call &nbsp;·&nbsp; Delta target: 0.65–0.75<br>
      Reference strike: ~${round(uvxy * 0.90)}–${round(uvxy * 0.95)}<br>
      <span style="color:#dc2626;font-weight:700;">
        ⚠️ You MUST own the long call before selling any short call.</span><br><br>
      <strong>Step 2 — After long fills:</strong> Sell short using signal below.
    </div>
  </td></tr>

  <!-- VARIANT SIZING -->
  <tr><td style="padding:20px 28px;border-bottom:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:10px;font-weight:700;">
      Position Sizing This Week</div>
    {_variant_table(batch, uvxy, regime, phase)}
    <div style="margin-top:10px;font-size:11px;color:#888;">
      Size based on your account. See:
      <a href="{SIZING_URL}" style="color:{rc};">volatilityharvest.com/sizing</a>
    </div>
  </td></tr>

  <!-- SIGNAL TABLE -->
  <tr><td style="padding:20px 28px;border-bottom:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:6px;font-weight:700;">
      This Week's Signal — UVXY Movement Table</div>
    <div style="font-size:12px;color:#666;margin-bottom:12px;">
      Centered on UVXY ${uvxy:.2f}. Find your row at execution time.
      Use the reference strike as a starting point — find the strike
      paying the target credit with delta at or below the ceiling.</div>
    {_delta_ceiling_banner(batch, regime, phase)}
    {_movement_table(uvxy, batch, regime, phase)}
  </td></tr>

  <!-- EXECUTION RULES -->
  <tr><td style="padding:20px 28px;border-bottom:1px solid #e8ede8;
                 background:#fafafa;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:10px;font-weight:700;">
      Execution Rules</div>
    <div style="font-size:12px;color:#444;line-height:2.0;">
      1. Find the strike paying the <strong>target credit</strong><br>
      2. Confirm delta is <strong>at or below the ceiling</strong><br>
      3. Confirm strike is <strong>above the OTM floor</strong><br>
      4. No strike meets all three → <strong>extend DTE and repeat</strong><br>
      5. 21 DTE still fails → <strong>accept small debit — delta safety is non-negotiable</strong><br>
      6. <strong>Skip the week</strong> if conditions don't make sense.<br>
      7. <strong>Limit orders only</strong> — never market orders on options.
    </div>
  </td></tr>

  <!-- CALENDAR -->
  <tr><td style="padding:16px 28px;border-bottom:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:{rc};
                text-transform:uppercase;margin-bottom:10px;font-weight:700;">
      Notable Events — Next 3 Weeks</div>
    {_calendar_section() or '<div style="font-size:12px;color:#888;">No major events detected.</div>'}
  </td></tr>

  <!-- FOOTER -->
  <tr><td style="padding:16px 28px;background:#f9f9f9;">
    <div style="font-size:11px;color:#888;line-height:1.8;text-align:center;">
      <a href="{WEBSITE_URL}" style="color:{rc};font-weight:700;">
        Volatility Harvest</a>
      &nbsp;·&nbsp; Questions? Reply to this email.
      &nbsp;·&nbsp; Always verify live prices before placing any order.<br>
      Signal generated from live app data. UVXY may move before you execute
      — use the table above to adjust.<br><br>
      <span style="color:#bbb;font-size:10px;">
        For informational purposes only. Not investment advice.
        Options trading involves significant risk of loss.
        <div style="font-size:10px;color:#999;line-height:1.8;margin-top:8px;
                        padding-top:8px;border-top:1px solid #e8e8e8;">
          <strong style="color:#666;">DISCLAIMER</strong><br>
          Options trading involves significant risk of loss and is not suitable
          for all investors. Past performance is not indicative of future results.
          The information provided is for educational and informational purposes
          only and does not constitute financial, investment, or trading advice.
          You are solely responsible for your own investment decisions.
          Volatility Harvest is not a registered investment adviser.
          Always consult a licensed financial advisor before making any
          investment decisions. UVXY and other leveraged volatility products
          can lose value rapidly and may not be suitable for most investors.
        </div></span>
    </div>
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""

    return subject, html


# ── Send ──────────────────────────────────────────────────────────────────────

def load_member_list() -> list[str]:
    if MEMBER_LIST_PATH.exists():
        try:
            return json.loads(MEMBER_LIST_PATH.read_text())
        except Exception:
            pass
    return []


def send_email(subject: str, html: str, recipients: list[str]):
    if not SMTP_USER or not SMTP_PASS:
        print("❌ SMTP_USER / SMTP_PASS not set")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = FROM_ADDR
    msg["To"]      = ", ".join(recipients)
    msg["Reply-To"] = REPLY_TO
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(FROM_ADDR, recipients, msg.as_string())
    print(f"✅ Sent to {len(recipients)} recipient(s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--test",    action="store_true")
    args = parser.parse_args()

    print("📊 Loading app data…")
    snap  = load_vol_snapshot()
    batch = load_signal_batch()

    if not snap:
        print("❌ No vol snapshot — run the app to generate data first")
        sys.exit(1)
    if not batch:
        print("❌ No signal batch — generate signals in the app first")
        sys.exit(1)

    uvxy   = snap.get("uvxy", 0.0)
    regime = snap.get("regime", "NEUTRAL")
    phase  = snap.get("spike_label", "Expansion")
    print(f"   UVXY: ${uvxy:.2f} | Regime: {regime} | Phase: {phase}")

    last_signal   = get_last_signal()
    subject, html = build_email(snap, batch, last_signal)

    # Save to signal history
    role_map = _build_role_map(batch)
    save_signal_history({
        "signal_date": date.today().isoformat(),
        "uvxy_price":  uvxy,
        "vix_level":   snap.get("vix", 0.0),
        "vix_pct":     snap.get("vix_pct", 0.5),
        "regime":      regime,
        "phase":       phase,
        "variants": {
            key: {
                "recommended_strike": round(
                    uvxy + v.get("short_strike_offset", 2.0)),
                "dte": v.get("short_dte_weeks", 1) * 7,
            }
            for key, v in role_map.items()
        }
    })

    if args.preview:
        out = Path("/home/shin/vix_suite/friday_signal_preview.html")
        out.write_text(html)
        print(f"✅ Preview saved: {out}")
        return

    recipients = [SMTP_USER] if args.test else load_member_list()
    if not recipients:
        print("⚠️  No members — sending to self")
        recipients = [SMTP_USER]

    print(f"   Sending to {len(recipients)} recipient(s)…")
    send_email(subject, html, recipients)


if __name__ == "__main__":
    main()
