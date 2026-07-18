#!/usr/bin/env python3
"""
patch_action_card.py
Adds a Position Action Card to each variant in build_position_aware_email.
One card, one action, no interpretation needed.
"""
from pathlib import Path
from datetime import date, timedelta

p = Path("/home/shin/vix_suite/daily_signal.py")
src = p.read_text()

old = '''      <!-- Action banner -->
      <div style="background:{ac};color:#fff;padding:10px 14px;border-radius:4px;
                  font-size:13px;font-weight:600;">{at}</div>
    </div>
"""'''

new = '''      <!-- Action banner -->
      <div style="background:{ac};color:#fff;padding:10px 14px;border-radius:4px;
                  font-size:13px;font-weight:600;">{at}</div>

      <!-- ── ACTION CARD ── -->
      {_action_card}
    </div>
"""'''

if old not in src:
    print("❌ action banner anchor not found")
    exit(1)

# Now find where variables are computed so we can inject _action_card computation
# Insert _action_card computation right before the html += f""" block
# Find the pnl_c line which is always just before html +=
old2 = '''            pnl_c = "#2e7d32" if pnl >= 0 else "#c62828"
            lpnl_c = "#2e7d32" if long_pnl >= 0 else "#c62828"
            spnl_c = "#2e7d32" if short_pnl >= 0 else "#c62828"'''

new2 = '''            pnl_c = "#2e7d32" if pnl >= 0 else "#c62828"
            lpnl_c = "#2e7d32" if long_pnl >= 0 else "#c62828"
            spnl_c = "#2e7d32" if short_pnl >= 0 else "#c62828"

            # ── Build Action Card ──────────────────────────────────────────
            try:
                from datetime import date as _acd, timedelta as _act
                # Next Friday ≥ 7 DTE for short target
                _today_ac = _acd.today()
                _days_fri = (4 - _today_ac.weekday()) % 7 or 7
                _next_fri = _today_ac + _act(days=_days_fri)
                if _days_fri < 7:
                    _next_fri = _next_fri + _act(days=7)
                _exp_str = _next_fri.strftime("%b %d")

                # Delta ceiling from sigma_mult
                _sm = getattr(variant, "sigma_mult", 1.0)
                _dc = 0.28 if _sm <= 0.8 else 0.25 if _sm <= 0.9 else \
                      0.22 if _sm <= 1.0 else 0.18 if _sm <= 1.2 else 0.15

                # Short strike target
                _offset = getattr(variant, "short_strike_offset", 2.0)
                _tgt_strike = round(uvxy_price + _offset)

                # Long leg DTE warning
                _long_dte_remaining = diag.days_to_long_expiry() if diag else 999

                # Build card based on action
                if action == "SELL_SHORT":
                    _card_bg = "#f3e5f5"; _card_bd = "#9c27b0"
                    _card_html = f"""
                <div style="margin-top:10px;padding:12px 14px;
                             background:{_card_bg};border-left:4px solid {_card_bd};
                             border-radius:4px;">
                  <div style="font-size:11px;font-weight:700;color:{_card_bd};
                               text-transform:uppercase;letter-spacing:1px;
                               margin-bottom:8px;">📋 Action — Sell Short</div>
                  <div style="font-size:13px;font-weight:700;color:#1a1a1a;
                               margin-bottom:6px;">
                    Sell to Open: <span style="color:{_card_bd};">
                    ${_tgt_strike}C {_exp_str}</span>
                  </div>
                  <div style="font-size:12px;color:#444;line-height:1.8;">
                    Target credit: <strong>$0.50–1.50</strong><br>
                    Delta ceiling: <strong>δ ≤ {_dc:.2f}</strong><br>
                    Verify live chain — use limit order at mid.
                  </div>
                  {"" if _long_dte_remaining > 14 else f\'<div style="margin-top:8px;padding:6px 10px;background:#fff3e0;border-radius:4px;font-size:11px;color:#e65100;font-weight:700;">⚠️ Long leg expires in {_long_dte_remaining}d — roll long first</div>\'}
                </div>"""
                elif action in ("ROLL_NOW", "ROLL", "EMERGENCY"):
                    _card_bg = "#fff3e0"; _card_bd = "#e65100"
                    _card_html = f"""
                <div style="margin-top:10px;padding:12px 14px;
                             background:{_card_bg};border-left:4px solid {_card_bd};
                             border-radius:4px;">
                  <div style="font-size:11px;font-weight:700;color:{_card_bd};
                               text-transform:uppercase;letter-spacing:1px;
                               margin-bottom:8px;">🔄 Action — Roll Short</div>
                  <div style="font-size:13px;color:#444;line-height:1.8;">
                    <strong>1. Buy to Close:</strong>
                    ${f"{float(short.strike):.0f}" if short else "current"}C
                    (current short)<br>
                    <strong>2. Sell to Open:</strong>
                    <span style="color:{_card_bd};font-weight:700;">
                    ${_tgt_strike}C {_exp_str}</span><br>
                    Target credit: <strong>$0.50–1.50</strong> &nbsp;·&nbsp;
                    δ ≤ <strong>{_dc:.2f}</strong>
                  </div>
                </div>"""
                elif _long_dte_remaining <= 14:
                    # Long leg expiring — urgent warning even if short is OK
                    _card_bg = "#fff8e1"; _card_bd = "#f9a825"
                    _long_strike = float(diag.long_strike) if diag else 0
                    _new_long = round(uvxy_price * 0.92)
                    _card_html = f"""
                <div style="margin-top:10px;padding:12px 14px;
                             background:{_card_bg};border-left:4px solid {_card_bd};
                             border-radius:4px;">
                  <div style="font-size:11px;font-weight:700;color:{_card_bd};
                               text-transform:uppercase;letter-spacing:1px;
                               margin-bottom:8px;">⚠️ Urgent — Long Leg Expiring</div>
                  <div style="font-size:13px;color:#444;line-height:1.8;">
                    <strong>1. Buy to Close:</strong>
                    ${_long_strike:.0f}C (long leg — near zero)<br>
                    <strong>2. Buy to Open:</strong>
                    <span style="color:{_card_bd};font-weight:700;">
                    ~${_new_long}C Sep 18</span>
                    — target δ 0.65–0.75
                  </div>
                </div>"""
                elif action == "HOLD":
                    _card_html = """
                <div style="margin-top:10px;padding:10px 14px;
                             background:#f0fdf4;border-left:4px solid #16a34a;
                             border-radius:4px;font-size:12px;color:#444;">
                  ✅ <strong>Hold</strong> — no action needed today.
                  Monitor delta Thursday morning.
                </div>"""
                elif action == "TAKE_PROFIT":
                    _card_html = f"""
                <div style="margin-top:10px;padding:10px 14px;
                             background:#e8f5e9;border-left:4px solid #4CAF50;
                             border-radius:4px;font-size:12px;color:#444;">
                  🎯 <strong>Take Profit</strong> — Buy to Close at 80%+ profit.<br>
                  Then Sell to Open: ${_tgt_strike}C {_exp_str} — δ ≤ {_dc:.2f}
                </div>"""
                else:
                    _card_html = ""
                _action_card = _card_html
            except Exception as _ace:
                _action_card = f'<div style="font-size:11px;color:#999;">Action card error: {_ace}</div>'
            # ── End Action Card ────────────────────────────────────────────'''

if old2 not in src:
    print("❌ pnl_c anchor not found")
    exit(1)

src = src.replace(old2, new2, 1)
src = src.replace(old, new, 1)
p.write_text(src)
print("✅ Action Card patch applied")
