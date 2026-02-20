"""
patch_email_real_trades.py
──────────────────────────
Adds real money position section to the daily email.
Shows real trades ABOVE paper trades with orange styling.
Run once: python3 patch_email_real_trades.py
"""
from pathlib import Path

path = Path("daily_signal.py")
src = path.read_text()

# ── 1. Add import
if "real_trade_log" not in src:
    lines = src.splitlines(keepends=True)
    for i, l in enumerate(lines):
        if "from trade_log import" in l:
            lines.insert(i + 1,
                "from real_trade_log import get_real_trade_log\n")
            break
    src = "".join(lines)
    print("✓ Added real_trade_log import")

# ── 2. Inject real positions section into build_position_aware_email
# Find the start of the email HTML body build
inject_after = '    html_parts = []'
real_section_code = '''
    # ── Real money positions (injected by patch_email_real_trades.py)
    try:
        rtl = get_real_trade_log()
        real_open = rtl.open_positions()
    except Exception:
        real_open = {}

    if real_open:
        real_rows = ""
        for pid, pos in sorted(real_open.items(),
                                key=lambda x: x[1].variant_id):
            short = pos.current_short_leg
            dte   = pos.days_to_expiry()
            pnl   = pos.total_pnl

            if dte <= 0:   dte_color = "#ff3366"
            elif dte <= 1: dte_color = "#ff9800"
            else:          dte_color = "#aaa"

            pnl_color = "#00e5a0" if pnl >= 0 else "#ff3366"

            if short:
                action_str = (
                    "⚠️ ROLL NOW — expired" if dte <= 0 else
                    "📋 PLACE ORDER — fills tomorrow" if dte == 1 else
                    "✋ HOLD"
                )
                short_info = (
                    f"Short: ${short.strike:.0f} exp {short.expiration_date} "
                    f"(fill ${short.fill_price:.2f})"
                )
            else:
                action_str = "📝 SELL SHORT — no active short"
                short_info = "No active short leg"

            real_rows += f"""
            <tr style="border-bottom:1px solid #3d1f00">
              <td style="padding:10px 14px">
                <div style="font-weight:700;color:#ff6b35">
                    {pos.variant_name}</div>
                <div style="font-size:10px;color:#664422">
                    {pos.broker} · {pos.account_id} · {pos.contracts}c</div>
              </td>
              <td style="padding:10px 14px;color:#aaa;font-size:11px">
                Long: ${pos.long_strike:.0f} exp {pos.long_expiration}
                (fill ${pos.long_fill_price:.2f})<br>
                {short_info}
              </td>
              <td style="padding:10px 14px;text-align:right">
                <span style="color:{pnl_color};font-weight:700">
                    ${pnl:+,.0f}</span><br>
                <span style="font-size:10px;color:#664422">
                    {pos.short_coverage_pct:.0f}% covered</span>
              </td>
              <td style="padding:10px 14px;text-align:right;
                         color:{dte_color};font-weight:600">
                {dte}d DTE<br>
                <span style="font-size:10px">{action_str}</span>
              </td>
            </tr>"""

        real_block = f"""
        <div style="background:#1a0a00;border:2px solid #3d1f00;
                    border-radius:8px;margin-bottom:24px;overflow:hidden">
          <div style="background:#2d1200;padding:12px 16px;
                      border-bottom:1px solid #3d1f00">
            <span style="font-size:14px;font-weight:800;color:#ff6b35">
                💵 REAL MONEY POSITIONS</span>
            <span style="font-size:10px;color:#664422;margin-left:12px">
                {len(real_open)} open · Live capital at risk</span>
          </div>
          <table style="width:100%;border-collapse:collapse">
            <thead>
              <tr style="background:#1f0a00">
                <th style="padding:8px 14px;text-align:left;font-size:9px;
                           letter-spacing:2px;color:#664422;
                           text-transform:uppercase">Position</th>
                <th style="padding:8px 14px;text-align:left;font-size:9px;
                           letter-spacing:2px;color:#664422;
                           text-transform:uppercase">Legs</th>
                <th style="padding:8px 14px;text-align:right;font-size:9px;
                           letter-spacing:2px;color:#664422;
                           text-transform:uppercase">P&L</th>
                <th style="padding:8px 14px;text-align:right;font-size:9px;
                           letter-spacing:2px;color:#664422;
                           text-transform:uppercase">DTE / Action</th>
              </tr>
            </thead>
            <tbody>{real_rows}</tbody>
          </table>
        </div>"""
        html_parts.append(real_block)
'''

if inject_after in src and "real_trade_log" in src:
    if "real_open = rtl.open_positions()" not in src:
        src = src.replace(inject_after,
                          inject_after + real_section_code)
        print("✓ Injected real positions section into email builder")
    else:
        print("  Real positions section already in email builder")

path.write_text(src)
print("\n✅ daily_signal.py patched for real trade email section")
