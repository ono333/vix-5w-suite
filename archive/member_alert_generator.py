#!/usr/bin/env python3
"""
member_alert_generator.py
Volatility Harvest — Member Alert Generator

Generates the weekly member email HTML automatically from regime state.
Zero manual composition needed. Run Thursday evening after market close.

Usage:
  python member_alert_generator.py              # sends to member list
  python member_alert_generator.py --dry-run    # prints HTML only
  python member_alert_generator.py --preview    # saves preview.html

Deploy to server: ~/vix_suite/member_alert_generator.py
"""

import os
import sys
import smtplib
import argparse
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Try to import from existing vix_suite modules ─────────────────────────────
try:
    from regime_detector import detect_regime
    from market_calendar import is_market_open, get_next_trading_day, format_calendar_warning
    HAS_SUITE = True
except ImportError:
    HAS_SUITE = False

# ════════════════════════════════════════════════════════════════════════════
# MEMBER LIST — replace with real member emails before launch
# ════════════════════════════════════════════════════════════════════════════
MEMBER_LIST_SIMPLE   = []  # Tier 1 — simple workflow
MEMBER_LIST_ADVANCED = []  # Tier 2 — advanced with roll instructions

# ════════════════════════════════════════════════════════════════════════════
# CONTENT GENERATION — all auto from regime state
# ════════════════════════════════════════════════════════════════════════════

def _regime_color(regime: str) -> str:
    return {"EXTREME": "#dc2626", "HIGH": "#d97706",
            "ELEVATED": "#ca8a04", "NEUTRAL": "#16a34a",
            "CALM": "#16a34a"}.get(regime.upper(), "#666")

def _regime_emoji(regime: str) -> str:
    return {"EXTREME": "🔴", "HIGH": "🟠", "ELEVATED": "🟡",
            "NEUTRAL": "🟢", "CALM": "🟢"}.get(regime.upper(), "⚪")

def _term_color(term: str) -> str:
    return "#dc2626" if "back" in term.lower() else "#16a34a"

def _strike_range(uvxy: float, regime: str) -> tuple:
    """Returns (range_str, detail_str) based on regime."""
    if regime in ("EXTREME",):
        lo = round(uvxy * 1.10)
        hi = round(uvxy * 1.15)
        band = "10–15% OTM (Panic)"
    elif regime in ("HIGH",):
        lo = round(uvxy * 1.08)
        hi = round(uvxy * 1.12)
        band = "8–12% OTM (Crisis)"
    elif regime in ("ELEVATED",):
        lo = round(uvxy * 1.06)
        hi = round(uvxy * 1.09)
        band = "6–9% OTM (Elevated)"
    elif regime in ("NEUTRAL",):
        lo = round(uvxy * 1.04)
        hi = round(uvxy * 1.06)
        band = "4–6% OTM (Neutral)"
    else:  # CALM
        lo = round(uvxy * 1.09)
        hi = round(uvxy * 1.11)
        band = "9–11% OTM (Calm)"
    range_str = f"${lo} – ${hi}"
    detail_str = (f"{band} · Based on UVXY ${uvxy:.2f}. "
                  f"Use ${lo} for more cushion, ${hi} for more premium. "
                  f"Your actual fill depends on available bids at your broker.")
    return range_str, detail_str

def _action_banner(regime: str, vix_pct: float, term: str,
                   fomc_this_week: bool, fomc_tomorrow: bool) -> dict:
    """Generate the top action banner content."""
    if fomc_tomorrow:
        return dict(
            bg="#fff8e7", border="#f59e0b", color="#92400e",
            emoji="⏸", headline="Wait — FOMC decision tomorrow",
            detail=("The Federal Reserve announces its rate decision tomorrow at 2pm ET. "
                    "Volatility often spikes around this announcement. "
                    "Hold your current position and wait until after 3pm ET tomorrow "
                    "before opening a new short. If your short expires Thursday, "
                    "sell the new one Monday instead.")
        )
    if fomc_this_week:
        return dict(
            bg="#fff8e7", border="#f59e0b", color="#92400e",
            emoji="⚠️", headline="Roll this week — FOMC caution",
            detail=("FOMC meeting this week. Proceed with your normal Monday STO "
                    "but use the conservative (lower) end of the strike range "
                    "for extra buffer against any volatility around the announcement.")
        )
    if vix_pct >= 0.90:
        return dict(
            bg="#fef2f2", border="#dc2626", color="#7f1d1d",
            emoji="🔄", headline="Roll this week — elevated conditions",
            detail=("VIX is in the 90th+ percentile. Your short call expires Thursday — "
                    "let it expire worthless, then sell a new short Monday morning. "
                    "Use the conservative end of the strike range below.")
        )
    if regime in ("EXTREME", "HIGH"):
        return dict(
            bg="#fef2f2", border="#dc2626", color="#7f1d1d",
            emoji="🔄", headline="Roll this week — active regime",
            detail=("Volatility regime is active. Let your short expire Thursday, "
                    "sell a new short Monday at the strike range below. "
                    "One order, one minute.")
        )
    if regime in ("NEUTRAL", "CALM") and vix_pct < 0.30:
        return dict(
            bg="#f0f7f2", border="#16a34a", color="#14532d",
            emoji="🟢", headline="Good conditions — roll and consider new entries",
            detail=("Calm regime with low VIX percentile. Ideal conditions for "
                    "rolling existing positions and for new members to enter. "
                    "Sell the new short Monday at the target strike range.")
        )
    return dict(
        bg="#f0f7f2", border="#16a34a", color="#14532d",
        emoji="🔄", headline="Roll this week — normal conditions",
        detail=("Let your short expire Thursday. Sell a new short call Monday "
                "morning at the target strike range below.")
    )

def _warning_box(regime: str, vix_pct: float, term: str,
                 uvxy_spiked: bool, fomc_this_week: bool) -> str:
    """Returns warning box HTML or empty string."""
    warnings = []
    if "back" in term.lower():
        warnings.append("Term structure flipped to backwardation — near-term stress signal. "
                        "Use the lower (more conservative) end of the strike range.")
    if uvxy_spiked:
        warnings.append("UVXY spiked intraday this week before reversing. "
                        "Extra buffer recommended — sell closer to 10% OTM rather than 15%.")
    if vix_pct >= 0.92:
        warnings.append("VIX at 92nd+ percentile — extreme elevated conditions. "
                        "Consider reducing to 1 contract if you have more than 1.")
    if fomc_this_week:
        warnings.append("FOMC meeting this week — be ready for elevated volatility "
                        "Tuesday–Wednesday. Emergency roll rules apply if needed.")

    if not warnings:
        return ""

    items = "".join(f"<li style='margin-bottom:6px;'>{w}</li>" for w in warnings)
    return f"""
  <div style="background:#fff8e7;border-left:4px solid #f59e0b;padding:14px 20px;
              border-top:1px solid #fde68a;">
    <div style="font-weight:700;color:#92400e;font-size:13px;margin-bottom:8px;">
      ⚠️  This week's notes
    </div>
    <ul style="margin:0;padding-left:18px;font-size:12px;color:#78350f;line-height:1.6;">
      {items}
    </ul>
  </div>"""

def _variant_rows(regime: str, vix_pct: float, uvxy: float) -> str:
    """Generate per-variant action rows."""
    lo = round(uvxy * 1.10)
    hi = round(uvxy * 1.15)

    variants = {
        "EXTREME": [
            ("V1 Income Harvester", "ROLL", f"${lo}–${hi}", "Calm entries — use lower end"),
            ("V2 Mean Reversion",   "HOLD", "—",            "Inactive in EXTREME regime"),
            ("V3 Shock Absorber",   "ROLL", f"${lo}–${hi}", "Defensive — 10% OTM preferred"),
            ("V4 Tail Hunter",      "ROLL", f"${lo}–${hi}", "Active — regime confirmed"),
            ("V5 Regime Allocator", "ROLL", f"${lo}–${hi}", "Active — follow V4 strike"),
        ],
        "HIGH": [
            ("V1 Income Harvester", "ROLL", f"${round(uvxy*1.08)}–${round(uvxy*1.12)}", "Normal roll"),
            ("V2 Mean Reversion",   "ROLL", f"${round(uvxy*1.08)}–${round(uvxy*1.12)}", "Active"),
            ("V3 Shock Absorber",   "ROLL", f"${round(uvxy*1.08)}–${round(uvxy*1.12)}", "Active"),
            ("V4 Tail Hunter",      "ROLL", f"${round(uvxy*1.08)}–${round(uvxy*1.12)}", "Active"),
            ("V5 Regime Allocator", "ROLL", f"${round(uvxy*1.08)}–${round(uvxy*1.12)}", "Active"),
        ],
        "CALM": [
            ("V1 Income Harvester", "ROLL", f"${round(uvxy*1.09)}–${round(uvxy*1.11)}", "Primary strategy"),
            ("V2 Mean Reversion",   "ROLL", f"${round(uvxy*1.09)}–${round(uvxy*1.11)}", "Active"),
            ("V3 Shock Absorber",   "HOLD", "—",            "Inactive in calm regime"),
            ("V4 Tail Hunter",      "HOLD", "—",            "Inactive in calm regime"),
            ("V5 Regime Allocator", "ROLL", f"${round(uvxy*1.09)}–${round(uvxy*1.11)}", "Follows V1"),
        ],
    }

    rows_data = variants.get(regime.upper(),
                variants.get("HIGH"))  # default to HIGH if unknown

    rows = ""
    for variant, action, strike, note in rows_data:
        if action == "HOLD":
            bg, ac, ae = "#fffbf0", "#856404", "⏸"
        elif action == "ROLL":
            bg, ac, ae = "#f0f7f2", "#1E4D2B", "🔄"
        else:
            bg, ac, ae = "#fff5f5", "#dc2626", "🚨"

        rows += f"""
      <tr style="border-top:1px solid #e8ede8;background:{bg};">
        <td style="padding:9px 10px;font-weight:600;color:#1a1a1a;">{variant}</td>
        <td style="padding:9px 10px;text-align:center;font-weight:700;color:{ac};">{ae} {action}</td>
        <td style="padding:9px 10px;font-weight:700;color:#1E4D2B;">{strike}</td>
        <td style="padding:9px 10px;color:#666;font-size:11px;">{note}</td>
      </tr>"""
    return rows

def _calendar_box() -> str:
    """Generate calendar alerts section."""
    try:
        from market_calendar import get_market_events
        from datetime import date, timedelta
        events = get_market_events(date.today(), date.today() + timedelta(days=21))
        if not events:
            return ""
        lines = ""
        for e in events[:5]:
            icon = "⛔" if e.market_closed else "⚠️"
            lines += f"{icon} &nbsp;<strong>{e.date.strftime('%a %b %-d')}</strong> — {e.name}<br>"
        next_trading = date.today()
        while next_trading.weekday() >= 5:
            next_trading += timedelta(days=1)
        next_sto = next_trading + timedelta(days=(7 - next_trading.weekday()) % 7)
        return f"""
  <div style="background:#ffffff;padding:16px 24px;border-top:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:#5ea874;
                text-transform:uppercase;margin-bottom:10px;">Calendar</div>
    <div style="font-size:12px;color:#444;line-height:2.0;">
      {lines}
      📅 &nbsp;<strong>Next STO window:</strong> Monday {next_sto.strftime('%b %-d')}, 9:30am ET
    </div>
  </div>"""
    except Exception:
        return ""

def _advanced_roll_section() -> str:
    """Extra section for advanced tier members — manual roll instructions."""
    return """
  <div style="background:#f8f8f8;padding:16px 24px;border-top:2px solid #1E4D2B;">
    <div style="font-size:10px;letter-spacing:2px;color:#5ea874;
                text-transform:uppercase;margin-bottom:10px;">
      Advanced — Manual Roll (Desktop platforms only)
    </div>
    <div style="font-size:12px;color:#444;line-height:1.8;">
      If your short still has value and you prefer to roll rather than let expire:<br><br>
      <strong>Step 1:</strong> Buy to Close the expiring short — limit order near the ask.<br>
      <strong>Step 2:</strong> Sell to Open the new short at next week's target strike.<br>
      <strong>Net credit target:</strong> $0.30–$0.80 per contract depending on regime.<br>
      <strong>Debit cap:</strong> Never pay more than $1.50 net debit to roll.<br><br>
      <span style="color:#666;font-size:11px;">
        If the roll costs more than $1.50 debit, let the short expire and sell fresh Monday.
      </span>
    </div>
  </div>"""

# ════════════════════════════════════════════════════════════════════════════
# HTML BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_member_email(
    uvxy: float,
    vix_level: float,
    vix_pct: float,          # 0.0–1.0
    regime: str,
    term_structure: str,
    iv_ratio: float,
    uvxy_spiked: bool = False,
    fomc_this_week: bool = False,
    fomc_tomorrow: bool = False,
    tier: str = "simple",    # "simple" or "advanced"
) -> tuple:
    """Build complete member email. Returns (subject, html)."""

    today = date.today()
    date_str = today.strftime("%A, %b %-d %Y")
    pct_int = round(vix_pct * 100)

    banner     = _action_banner(regime, vix_pct, term_structure,
                                fomc_this_week, fomc_tomorrow)
    strike_range, strike_detail = _strike_range(uvxy, regime)
    warning    = _warning_box(regime, vix_pct, term_structure,
                              uvxy_spiked, fomc_this_week)
    var_rows   = _variant_rows(regime, vix_pct, uvxy)
    cal_box    = _calendar_box()
    adv_section = _advanced_roll_section() if tier == "advanced" else ""

    subject = (f"[Volatility Harvest] {banner['emoji']} "
               f"{banner['headline']} — {today.strftime('%b %-d')}")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#f0f4f1;font-family:'Georgia',serif;">
<div style="max-width:600px;margin:0 auto;padding:20px 16px;">

  <!-- Header -->
  <div style="background:#1E4D2B;border-radius:8px 8px 0 0;padding:20px 24px;">
    <div style="font-size:11px;letter-spacing:3px;color:#5ea874;
                text-transform:uppercase;margin-bottom:6px;">Volatility Harvest</div>
    <div style="font-size:22px;font-weight:700;color:#ffffff;margin-bottom:2px;">
      Weekly Signal — {date_str}
    </div>
    <div style="font-size:12px;color:#5ea874;">
      UVXY ${uvxy:.2f} &nbsp;·&nbsp; VIX {vix_level:.1f} &nbsp;·&nbsp; {pct_int}th percentile
    </div>
  </div>

  <!-- Action Banner -->
  <div style="background:{banner['bg']};padding:16px 24px;
              border-left:4px solid {banner['border']};">
    <div style="font-size:18px;font-weight:700;color:{banner['color']};margin-bottom:4px;">
      {banner['emoji']} {banner['headline']}
    </div>
    <div style="font-size:13px;color:#444;line-height:1.5;">
      {banner['detail']}
    </div>
  </div>

  <!-- Market Conditions -->
  <div style="background:#ffffff;padding:20px 24px;border-top:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:#5ea874;
                text-transform:uppercase;margin-bottom:14px;">Market Conditions</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr>
        <td style="padding:8px 0;color:#666;width:40%;">Regime</td>
        <td style="padding:8px 0;font-weight:700;color:{_regime_color(regime)};">
          {_regime_emoji(regime)} {regime}
        </td>
        <td style="padding:8px 0;color:#666;width:25%;">VIX %ile</td>
        <td style="padding:8px 0;font-weight:700;">{pct_int}%</td>
      </tr>
      <tr style="border-top:1px solid #f0f0f0;">
        <td style="padding:8px 0;color:#666;">Term Structure</td>
        <td style="padding:8px 0;font-weight:600;color:{_term_color(term_structure)};">
          {term_structure}
        </td>
        <td style="padding:8px 0;color:#666;">IV Ratio</td>
        <td style="padding:8px 0;font-weight:600;">{iv_ratio:.3f}</td>
      </tr>
      <tr style="border-top:1px solid #f0f0f0;">
        <td style="padding:8px 0;color:#666;">Strike Band</td>
        <td style="padding:8px 0;font-weight:600;" colspan="3">
          {strike_range} &nbsp;·&nbsp; {pct_int}th pct regime
        </td>
      </tr>
    </table>
  </div>

  <!-- This Week's Execution -->
  <div style="background:#ffffff;padding:20px 24px;border-top:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:#5ea874;
                text-transform:uppercase;margin-bottom:14px;">This Week's Execution</div>
    <div style="margin-bottom:12px;">
      <div style="display:flex;align-items:flex-start;margin-bottom:12px;">
        <div style="background:#1E4D2B;color:#fff;border-radius:50%;
                    min-width:22px;width:22px;height:22px;text-align:center;
                    line-height:22px;font-size:11px;font-weight:700;
                    margin-right:12px;margin-top:1px;">1</div>
        <div>
          <div style="font-weight:700;color:#1a1a1a;font-size:13px;">
            Thursday — Let short expire
          </div>
          <div style="color:#666;font-size:12px;margin-top:2px;">
            If UVXY closes below your short strike, do nothing.
            Your short call expires worthless at 4pm ET automatically.
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:flex-start;">
        <div style="background:#1E4D2B;color:#fff;border-radius:50%;
                    min-width:22px;width:22px;height:22px;text-align:center;
                    line-height:22px;font-size:11px;font-weight:700;
                    margin-right:12px;margin-top:1px;">2</div>
        <div>
          <div style="font-weight:700;color:#1a1a1a;font-size:13px;">
            Monday — Sell new short
          </div>
          <div style="color:#666;font-size:12px;margin-top:2px;">
            Open your broker app. UVXY → Options → next Friday expiry →
            find the strike below → Sell to Open. One order, one minute.
          </div>
        </div>
      </div>
    </div>
    <!-- Strike box -->
    <div style="background:#f0f7f2;border:1px solid #b8dfc4;
                border-radius:6px;padding:14px 16px;margin-top:12px;">
      <div style="font-size:10px;letter-spacing:2px;color:#2D6A3F;
                  text-transform:uppercase;margin-bottom:6px;">Monday Strike Target</div>
      <div style="font-size:26px;font-weight:700;color:#1E4D2B;margin-bottom:4px;">
        {strike_range}
      </div>
      <div style="font-size:12px;color:#555;">{strike_detail}</div>
    </div>
  </div>

  <!-- Variant table -->
  <div style="background:#ffffff;padding:20px 24px;border-top:1px solid #e8ede8;">
    <div style="font-size:10px;letter-spacing:2px;color:#5ea874;
                text-transform:uppercase;margin-bottom:14px;">By Variant</div>
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <tr style="background:#f0f4f1;">
        <th style="padding:8px 10px;text-align:left;color:#2D6A3F;">Variant</th>
        <th style="padding:8px 10px;text-align:center;color:#2D6A3F;">Action</th>
        <th style="padding:8px 10px;text-align:left;color:#2D6A3F;">Strike</th>
        <th style="padding:8px 10px;text-align:left;color:#2D6A3F;">Note</th>
      </tr>
      {var_rows}
    </table>
  </div>

  {warning}
  {cal_box}
  {adv_section}

  <!-- Footer -->
  <div style="background:#1E4D2B;border-radius:0 0 8px 8px;padding:16px 24px;">
    <div style="font-size:11px;color:#5ea874;line-height:1.6;">
      Educational signal only — not financial advice. Apply your own sizing
      per the member guide. Your strike may differ based on available
      premiums at your broker.
    </div>
    <div style="font-size:10px;color:#3d7a54;margin-top:8px;">
      Volatility Harvest · Weekly Signal · {date_str} · Do not reply
    </div>
  </div>

</div>
</body>
</html>"""

    return subject, html

# ════════════════════════════════════════════════════════════════════════════
# EMAIL SENDER
# ════════════════════════════════════════════════════════════════════════════

def send_member_alerts(dry_run: bool = False, preview: bool = False,
                       tier: str = "both") -> None:
    """Fetch market state and send member alerts."""

    # ── Fetch live market data ────────────────────────────────────────────
    try:
        import yfinance as yf
        import numpy as np

        uvxy_data = yf.download("UVXY", period="1y", interval="1d", progress=False)
        uvxy = float(uvxy_data["Close"].iloc[-1])
        uvxy_pct = float(np.percentile(uvxy_data["Close"], 100) and
                         (uvxy_data["Close"] <= uvxy).mean())

        vix_data  = yf.download("^VIX",  period="1d", interval="1m", progress=False)
        vix3m_data= yf.download("^VIX3M",period="1d", interval="1m", progress=False)
        vix_level = float(vix_data["Close"].iloc[-1])
        vix3m     = float(vix3m_data["Close"].iloc[-1])

        # VIX 52w percentile
        vix_hist  = yf.download("^VIX", period="1y", interval="1d", progress=False)
        vix_pct   = float((vix_hist["Close"] <= vix_level).mean())

        iv_ratio  = round(vix_level / vix3m, 3)
        term      = ("Backwardation" if iv_ratio > 1.05 else
                     "Mild Backwardation" if iv_ratio > 1.0 else
                     "Mild Contango" if iv_ratio > 0.95 else "Contango")

        # Simple regime from VIX percentile
        if   vix_pct >= 0.90: regime = "EXTREME"
        elif vix_pct >= 0.75: regime = "HIGH"
        elif vix_pct >= 0.50: regime = "ELEVATED"
        elif vix_pct >= 0.25: regime = "NEUTRAL"
        else:                  regime = "CALM"

        # Check if UVXY spiked today (>5% intraday)
        uvxy_intra = yf.download("UVXY", period="1d", interval="5m", progress=False)
        uvxy_high  = float(uvxy_intra["High"].max())
        uvxy_spiked = (uvxy_high / uvxy - 1) > 0.05

    except Exception as e:
        print(f"⚠️  Could not fetch live data: {e}")
        print("    Using fallback values — check yfinance connection")
        uvxy, vix_level, vix_pct = 50.52, 25.1, 0.88
        regime, term, iv_ratio   = "EXTREME", "Mild Backwardation", 1.071
        uvxy_spiked               = False

    # ── Check calendar ────────────────────────────────────────────────────
    fomc_this_week = fomc_tomorrow = False
    try:
        from market_calendar import get_market_events
        events = get_market_events(date.today(), date.today() + timedelta(days=7))
        for e in events:
            if e.event_type == "fomc":
                fomc_this_week = True
                if (e.date - date.today()).days <= 1:
                    fomc_tomorrow = True
    except Exception:
        pass

    # ── Build emails ──────────────────────────────────────────────────────
    subj_s, html_s = build_member_email(
        uvxy=uvxy, vix_level=vix_level, vix_pct=vix_pct,
        regime=regime, term_structure=term, iv_ratio=iv_ratio,
        uvxy_spiked=uvxy_spiked, fomc_this_week=fomc_this_week,
        fomc_tomorrow=fomc_tomorrow, tier="simple"
    )
    subj_a, html_a = build_member_email(
        uvxy=uvxy, vix_level=vix_level, vix_pct=vix_pct,
        regime=regime, term_structure=term, iv_ratio=iv_ratio,
        uvxy_spiked=uvxy_spiked, fomc_this_week=fomc_this_week,
        fomc_tomorrow=fomc_tomorrow, tier="advanced"
    )

    if preview:
        with open("member_alert_preview_simple.html",   "w") as f: f.write(html_s)
        with open("member_alert_preview_advanced.html", "w") as f: f.write(html_a)
        print("✅ Previews saved:")
        print("   member_alert_preview_simple.html")
        print("   member_alert_preview_advanced.html")
        return

    if dry_run:
        print(f"Subject (simple):   {subj_s}")
        print(f"Subject (advanced): {subj_a}")
        print(f"Regime: {regime} | VIX: {vix_level:.1f} | UVXY: {uvxy:.2f}")
        print(f"FOMC this week: {fomc_this_week} | Tomorrow: {fomc_tomorrow}")
        print(f"UVXY spiked: {uvxy_spiked}")
        print(f"Simple members:   {len(MEMBER_LIST_SIMPLE)}")
        print(f"Advanced members: {len(MEMBER_LIST_ADVANCED)}")
        return

    # ── Send ──────────────────────────────────────────────────────────────
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        print("⚠️  SMTP credentials not set — set SMTP_USER and SMTP_PASS")
        return

    sent = 0
    for addr in MEMBER_LIST_SIMPLE:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subj_s
            msg["From"]    = f"Volatility Harvest <{smtp_user}>"
            msg["To"]      = addr
            msg.attach(MIMEText(html_s, "html"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                s.login(smtp_user, smtp_pass)
                s.sendmail(smtp_user, addr, msg.as_string())
            sent += 1
        except Exception as e:
            print(f"  ⚠️  Failed {addr}: {e}")

    for addr in MEMBER_LIST_ADVANCED:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subj_a
            msg["From"]    = f"Volatility Harvest <{smtp_user}>"
            msg["To"]      = addr
            msg.attach(MIMEText(html_a, "html"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                s.login(smtp_user, smtp_pass)
                s.sendmail(smtp_user, addr, msg.as_string())
            sent += 1
        except Exception as e:
            print(f"  ⚠️  Failed {addr}: {e}")

    print(f"✅ Member alerts sent: {sent} emails")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datetime import timedelta
    parser = argparse.ArgumentParser(description="Volatility Harvest member alert")
    parser.add_argument("--dry-run",  action="store_true", help="Print without sending")
    parser.add_argument("--preview",  action="store_true", help="Save HTML preview files")
    parser.add_argument("--tier",     default="both",      help="simple|advanced|both")
    args = parser.parse_args()

    send_member_alerts(
        dry_run=args.dry_run,
        preview=args.preview,
        tier=args.tier,
    )
