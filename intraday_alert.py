#!/usr/bin/env python3
"""
Intraday position monitor — runs every 30 min during market hours.
Sends URGENT email only when action is required.
Checks: delta_trigger, spike_guard, ITM threat, expiry-day roll.
When roll_early_delta or roll_now fires, fetches live Tradier sandbox chain
and includes exact roll guidance in the alert email.
"""
import argparse
import os, sys, json, smtplib
import requests
from pathlib import Path
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, str(Path(__file__).parent))

from roll_manager import evaluate_roll, RollDecision
from trade_log import get_trade_log
from real_trade_log import get_real_trade_log

SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
ALERT_SUBJECT_PREFIX = "[VIX ALERT]"

URGENT_ACTIONS       = {"roll_early_itm", "roll_early_delta", "roll_now"}
WATCH_ACTIONS        = {"spike_guard_hold"}
ROLL_TRIGGER_ACTIONS = {"roll_early_delta", "roll_now"}

TRADIER_SANDBOX_BASE = "https://sandbox.tradier.com/v1"
SIGNAL_BATCH_PATH    = Path.home() / ".vix_suite" / "current_signal_batch.json"


# ── Signal batch helpers ──────────────────────────────────────────────────────

def _load_batch() -> dict:
    if SIGNAL_BATCH_PATH.exists():
        try:
            return json.loads(SIGNAL_BATCH_PATH.read_text())
        except Exception:
            pass
    return {}


def _build_role_map(batch: dict) -> dict:
    role_map = {}
    for v in batch.get("variants", []):
        role = v.get("role", "")
        for key, tag in [("V1","v1"),("V2","v2"),("V3","v3"),
                         ("V4","v4"),("V5","v5")]:
            if tag in role:
                role_map[key] = v
    return role_map


def _delta_ceil(sigma_mult: float) -> float:
    if sigma_mult <= 0.8: return 0.28
    if sigma_mult <= 0.9: return 0.25
    if sigma_mult <= 1.0: return 0.22
    if sigma_mult <= 1.2: return 0.18
    return 0.15


def _extract_key(variant_name: str) -> str:
    """'V1 Income Harvester' -> 'V1'"""
    p = variant_name.split()
    if p and p[0].startswith("V") and p[0][1:].isdigit():
        return p[0]
    return ""


# ── Roll guidance via live Tradier chain ──────────────────────────────────────

def _fetch_roll_target(short_strike: float, short_exp: date, dc: float) -> str:
    """
    Query Tradier sandbox for nearest Friday >= 7 DTE, find best roll strike
    (delta <= dc, bid >= $0.10). Returns formatted guidance or empty string.
    """
    try:
        token = os.environ.get("TRADIER_SANDBOX_TOKEN", "")
        if not token:
            return ""
        hdrs = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        r = requests.get(
            f"{TRADIER_SANDBOX_BASE}/markets/options/expirations",
            params={"symbol": "UVXY", "includeAllRoots": "true", "strikes": "false"},
            headers=hdrs, timeout=10)
        r.raise_for_status()
        raw = r.json().get("expirations", {}).get("date", [])
        if isinstance(raw, str):
            raw = [raw]
        all_exps = [date.fromisoformat(d) for d in raw]

        today   = date.today()
        target  = today + timedelta(days=7)
        fridays = [e for e in all_exps if e >= target and e.weekday() == 4]
        if not fridays:
            return ""
        roll_exp = fridays[0]

        r = requests.get(
            f"{TRADIER_SANDBOX_BASE}/markets/options/chains",
            params={"symbol": "UVXY",
                    "expiration": roll_exp.strftime("%Y-%m-%d"),
                    "greeks": "true"},
            headers=hdrs, timeout=10)
        r.raise_for_status()
        opts  = r.json().get("options", {})
        chain = opts.get("option", []) if opts else []
        if isinstance(chain, dict):
            chain = [chain]

        candidates = []
        for c in chain:
            if c.get("option_type", "").lower() != "call":
                continue
            greeks = c.get("greeks") or {}
            delta  = abs(float(greeks.get("delta", 1.0) or 1.0))
            bid    = float(c.get("bid", 0) or 0)
            if delta <= dc and bid >= 0.10:
                candidates.append({"strike": float(c.get("strike", 0)),
                                   "delta": delta, "bid": bid})
        if not candidates:
            return ""
        candidates.sort(key=lambda x: x["delta"], reverse=True)
        best = candidates[0]

        cur_str  = short_exp.strftime("%b%-d")
        roll_str = roll_exp.strftime("%b%-d")
        return (f"BTC ${short_strike:.0f}C {cur_str} -> "
                f"STO ${best['strike']:.0f}C {roll_str} "
                f"@ ~${best['bid']:.2f} bid (d={best['delta']:.2f})")
    except Exception as e:
        print(f"   Roll guidance error: {e}")
        return ""


# ── Market data ───────────────────────────────────────────────────────────────

def get_uvxy_price() -> float:
    try:
        import yfinance as yf
        t = yf.Ticker("UVXY")
        data = t.history(period="1d", interval="1m")
        return float(data["Close"].iloc[-1])
    except:
        return 0.0


# ── Email builder ─────────────────────────────────────────────────────────────

def build_alert_email(alerts: list, watches: list, uvxy: float,
                      all_entries: list = None,
                      japan_mode: bool = False) -> tuple[str, str]:
    now     = datetime.now().strftime("%b %d %Y %I:%M %p ET")
    urgency = "URGENT ACTION REQUIRED" if alerts else "WATCH ALERT"
    if japan_mode:
        from datetime import timezone
        _jst = datetime.now(tz=timezone.utc) + timedelta(hours=9)
        _jst_tag = f"🇯🇵 JST {_jst.strftime('%H:%M')} · "
    else:
        _jst_tag = ""
    subject = f"{ALERT_SUBJECT_PREFIX} {_jst_tag}{urgency} -- UVXY ${uvxy:.2f} -- {now}"

    rows = ""
    for a in alerts:
        color    = "#cc0000" if a["action"] == "roll_early_itm" else "#cc6600"
        guidance = a.get("roll_guidance", "")
        rows += (
            f'<tr style="background:#fff5f5">'
            f'<td style="padding:10px 14px;font-weight:700;color:#1a1a1a">{a["variant"]}</td>'
            f'<td style="padding:10px 14px;color:{color};font-weight:700">{a["action"].upper().replace("_"," ")}</td>'
            f'<td style="padding:10px 14px;color:#444444">{a["reason"]}</td>'
            f'<td style="padding:10px 14px;color:#cc6600">est BB ${a["est_bb"]:.2f}</td>'
            f'<td style="padding:10px 14px;color:#1e7d4d;font-size:11px;font-style:italic">{guidance}</td>'
            f'</tr>'
        )

    for w in watches:
        rows += (
            f'<tr style="background:#f0f7ff">'
            f'<td style="padding:10px 14px;font-weight:700;color:#1a1a1a">{w["variant"]}</td>'
            f'<td style="padding:10px 14px;color:#1565c0;font-weight:700">SPIKE GUARD</td>'
            f'<td style="padding:10px 14px;color:#444444">{w["reason"]}</td>'
            f'<td style="padding:10px 14px;color:#1565c0">holding</td>'
            f'<td style="padding:10px 14px"></td>'
            f'</tr>'
        )

    live_rows = "".join(
        f'<tr style="background:{"#fff5f5" if p["action"] not in ("hold","expiring_worthless") else "#f8fff8"};border-top:1px solid #dee2e6;">'
        f'<td style="padding:6px 10px;font-weight:700;color:#1a1a1a;">{p["variant"].replace("💰 [REAL] ","")}</td>'
        f'<td style="padding:6px 10px;text-align:center;color:#333;">{p["dte"]}d</td>'
        f'<td style="padding:6px 10px;text-align:center;color:#333;">${p["strike"]}</td>'
        f'<td style="padding:6px 10px;text-align:center;font-weight:700;'
        f'color:{"#16a34a" if p["action"] in ("hold","expiring_worthless") else "#dc2626"};">'
        f'{"On Track" if p["action"] == "hold" else p["action"].upper().replace("_"," ")}</td>'
        f'<td style="padding:6px 10px;color:#555;font-size:11px;">'
        f'{"No action needed" if p["action"] == "hold" else p["reason"][:60]}</td>'
        f'</tr>'
        for p in (all_entries or []) if "💰" in p.get("variant", "")
    )

    paper_flagged = len([e for e in (alerts + watches) if "📋" in e.get("variant", "")])
    paper_total   = len([e for e in (all_entries or []) if "📋" in e.get("variant", "")])

    roll_warning = (
        "<div style='margin-top:16px;padding:12px;background:#fff3cd;"
        "border:1px solid #ffc107;border-radius:4px;color:#856404;font-size:12px'>"
        "Roll before market close today to avoid assignment risk.</div>"
        if alerts else ""
    )

    html = f"""
    <div style="background:#ffffff;color:#1a1a1a;font-family:'IBM Plex Mono',monospace;
                padding:24px;max-width:800px;margin:0 auto">
      <div style="font-size:22px;font-weight:800;color:#1a1a1a;margin-bottom:4px">
        VIX 5W Suite -- {"URGENT ALERT" if alerts else "Watch Alert"}
      </div>
      <div style="font-size:11px;color:#666666;margin-bottom:20px">{now} · UVXY ${uvxy:.2f}</div>
      <table style="width:100%;border-collapse:collapse;background:#f8f9fa;
                    border:1px solid #dee2e6;border-radius:6px;overflow:hidden">
        <thead>
          <tr style="background:#e8f0e8">
            <th style="padding:8px 14px;text-align:left;font-size:10px;letter-spacing:2px;color:#1e4d2b;text-transform:uppercase">Variant</th>
            <th style="padding:8px 14px;text-align:left;font-size:10px;letter-spacing:2px;color:#1e4d2b;text-transform:uppercase">Action</th>
            <th style="padding:8px 14px;text-align:left;font-size:10px;letter-spacing:2px;color:#1e4d2b;text-transform:uppercase">Reason</th>
            <th style="padding:8px 14px;text-align:left;font-size:10px;letter-spacing:2px;color:#444;text-transform:uppercase">Cost Est.</th>
            <th style="padding:8px 14px;text-align:left;font-size:10px;letter-spacing:2px;color:#1e4d2b;text-transform:uppercase">Roll Target</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      {roll_warning}
      <div style="margin-top:20px;">
        <div style="font-size:11px;font-weight:700;color:#1e4d2b;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">
          Live Positions
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
          <tr style="background:#e8f0e8;">
            <th style="padding:6px 10px;text-align:left;color:#1e4d2b;">Variant</th>
            <th style="padding:6px 10px;text-align:center;color:#1e4d2b;">DTE</th>
            <th style="padding:6px 10px;text-align:center;color:#1e4d2b;">Strike</th>
            <th style="padding:6px 10px;text-align:center;color:#1e4d2b;">Status</th>
            <th style="padding:6px 10px;text-align:left;color:#1e4d2b;">Action</th>
          </tr>
          {live_rows}
        </table>
      </div>
      <div style="margin-top:12px;padding:8px 12px;background:#f0f4f1;border-radius:4px;font-size:11px;color:#666;">
        Paper positions: {paper_flagged} flagged · {paper_total} total tracked
      </div>
      <div style="margin-top:16px;font-size:10px;color:#888;">
        VIX 5% Weekly Suite · Intraday Monitor · UVXY ${uvxy:.2f} · Do not reply
      </div>
    </div>"""
    return subject, html


# ── Email sender ──────────────────────────────────────────────────────────────

def send_email(subject: str, html: str):
    if not SMTP_USER or not SMTP_PASS:
        print("SMTP credentials missing")
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
        print(f"Alert sent: {subject[:60]}")
        return True
    except Exception as e:
        print(f"Send failed: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force_send: bool = False):
    today = date.today()
    if today.weekday() >= 5:
        print("Weekend -- skipping")
        return

    # Japan Mode — read once at startup
    try:
        from vix_suite_config import load_config as _load_cfg
        _cfg = _load_cfg()
    except Exception:
        _cfg = {}
    _japan = bool(_cfg.get("japan_mode", False))
    _delta_trigger = 0.35 if _japan else 0.45
    if _japan:
        print(f"[intraday] Japan Mode ON — delta trigger lowered to {_delta_trigger}")
    try:
        from market_calendar import is_market_open
        if not is_market_open(today):
            print(f"Market holiday ({today}) -- intraday alert suppressed")
            return
    except ImportError:
        pass

    uvxy_price = get_uvxy_price()

    if uvxy_price > 0:
        try:
            from vol_triangle import capture_snapshot
            capture_snapshot(force=True)
            print(f"[intraday] Snapshot updated")
        except Exception as _se:
            print(f"[intraday] Snapshot update failed: {_se}")

    if uvxy_price == 0.0:
        print("Could not fetch UVXY price -- skipping")
        return

    print(f"UVXY: ${uvxy_price:.2f}  |  {datetime.now().strftime('%H:%M ET')}")

    batch    = _load_batch()
    role_map = _build_role_map(batch)

    alerts, watches, all_entries = [], [], []

    trade_sources = [
        ('paper', get_trade_log()),
        ('real',  get_real_trade_log()),
    ]

    for mode_label, tl in trade_sources:
     for pid, pos in tl.diagonal_positions.items():
        if pos.status != "open":
            continue
        short = pos.current_short_leg
        if not short:
            continue

        try:
            exp_date = date.fromisoformat(short.expiration_date)
            dte = max(0, (exp_date - today).days)
        except Exception:
            dte = 0

        decision = evaluate_roll(
            dte_remaining    = dte,
            short_delta      = getattr(short, "delta", None),
            uvxy_price       = uvxy_price,
            short_strike     = short.strike,
            variant_params   = {"roll_dte_days": 0, "delta_trigger": _delta_trigger,
                                "spike_guard_days": 2},
            last_spike_date  = None,
            original_premium = short.entry_credit,
        )

        entry = dict(
            variant      = f'{"📋" if mode_label == "paper" else "💰"} [{mode_label.upper()}] {pos.variant_name}',
            action       = decision.action,
            reason       = decision.reason[:80],
            est_bb       = decision.expected_bb,
            dte          = dte,
            strike       = short.strike,
            roll_guidance= "",
        )

        print(f"  {pos.variant_name:<25} DTE={dte}  strike=${short.strike}  "
              f"UVXY=${uvxy_price:.2f}  -> {decision.action}")

        _delta = getattr(short, 'delta', None) or 0.0
        if decision.action == 'roll_now' and dte <= 0 and _delta < 0.05:
            continue

        if decision.action in ROLL_TRIGGER_ACTIONS:
            vkey = _extract_key(pos.variant_name)
            v    = role_map.get(vkey, {})
            dc   = _delta_ceil(v.get("sigma_mult", 1.0))
            try:
                exp_date_obj = date.fromisoformat(short.expiration_date)
            except Exception:
                exp_date_obj = today
            guidance = _fetch_roll_target(short.strike, exp_date_obj, dc)
            if guidance:
                entry["roll_guidance"] = f"{vkey}: {guidance}"
                print(f"    -> {entry['roll_guidance']}")

        all_entries.append(entry)
        if decision.action in URGENT_ACTIONS:
            alerts.append(entry)
        elif decision.action in WATCH_ACTIONS:
            watches.append(entry)

    if alerts or watches or force_send:
        subject, html = build_alert_email(alerts, watches, uvxy_price, all_entries,
                                          japan_mode=_japan)
        send_email(subject, html)
        if not alerts and not watches:
            print("Force-send test email sent")
    else:
        print("All positions HOLD -- no alert needed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--force-send", action="store_true")
    args = parser.parse_args()
    main(force_send=args.force_send)
