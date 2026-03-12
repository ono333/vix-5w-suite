"""
real_trade_ui.py
────────────────
Streamlit UI component for real money trade management.
Call render_real_trade_section() from app.py.
Separate from paper trade UI — clear visual distinction.
"""

from __future__ import annotations
import streamlit as st
from datetime import date, datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from real_trade_log import (
    get_real_trade_log, reset_real_trade_log_cache,
    RealDiagonalPosition, BROKERS
)


# ── Visual identity for real trades ─────────────────────────
REAL_COLOR   = "#ff6b35"   # orange — distinct from paper green
REAL_BG      = "#1a0a00"
REAL_BORDER  = "#3d1f00"


def _badge(text: str, color: str = REAL_COLOR) -> str:
    return (f'<span style="background:{color}22;color:{color};'
            f'border:1px solid {color}44;padding:2px 8px;'
            f'border-radius:3px;font-size:10px;font-weight:700;'
            f'letter-spacing:1px;text-transform:uppercase">{text}</span>')


def _pnl_color(v: float) -> str:
    return "#00e5a0" if v >= 0 else "#ff3366"


def render_real_trade_section():
    """Main entry point — renders the full real trade UI."""

    st.markdown(f"""
    <div style="background:{REAL_BG};border:2px solid {REAL_BORDER};
                border-radius:8px;padding:16px 20px;margin-bottom:20px">
      <div style="display:flex;align-items:center;gap:12px">
        <span style="font-size:20px">💵</span>
        <div>
          <div style="font-size:18px;font-weight:800;color:#ff6b35">
            Real Money Trades
          </div>
          <div style="font-size:11px;color:#664422">
            Live positions · Fidelity / IB · Separate from paper trades
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tl = get_real_trade_log()
    open_pos = tl.open_positions()

    # ── Summary bar
    summary = tl.summary()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Open Positions", summary["open_count"])
    with c2:
        st.metric("Total P&L",
                  f"${summary['total_pnl']:+,.0f}",
                  delta=None)
    with c3:
        st.metric("Commissions Paid",
                  f"${summary['total_commissions']:,.2f}")
    with c4:
        st.metric("Total Slippage",
                  f"${summary['total_slippage']:+,.2f}")

    st.divider()

    # ── Tabs
    tab1, tab2, tab3 = st.tabs([
        "📋 Open Positions",
        "➕ New Entry",
        "📊 History & Stats"
    ])

    with tab1:
        _render_open_positions(tl, open_pos)

    with tab2:
        _render_new_entry(tl)

    with tab3:
        _render_history(tl)


# ══════════════════════════════════════════════════════════
# OPEN POSITIONS
# ══════════════════════════════════════════════════════════

def _render_open_positions(tl, open_pos: dict):
    if not open_pos:
        st.info("No open real money positions. Use 'New Entry' to add one.")
        return

    for pid, pos in sorted(open_pos.items(),
                            key=lambda x: x[1].variant_id):
        _render_position_card(tl, pid, pos)


def _render_position_card(tl, pid: str, pos: RealDiagonalPosition):
    short    = pos.current_short_leg
    dte      = pos.days_to_expiry()
    pnl      = pos.total_pnl
    coverage = pos.short_coverage_pct

    # DTE color
    if dte <= 0:   dte_color = "#ff3366"
    elif dte <= 1: dte_color = "#ff9800"
    else:          dte_color = "#aaa"

    with st.expander(
        f"💵 {pos.variant_name}  |  "
        f"{pos.broker} · {pos.account_id}  |  "
        f"P&L: ${pnl:+,.0f}  |  DTE: {dte}d",
        expanded=(dte <= 1)
    ):
        # Header info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**Entry Date**  \n{pos.entry_date}")
            st.markdown(f"**Regime**  \n{pos.entry_regime}")
            st.markdown(f"**Contracts**  \n{pos.contracts}")
        with col2:
            st.markdown(f"**Long Strike**  \n${pos.long_strike:.0f}")
            st.markdown(f"**Long Exp**  \n{pos.long_expiration}")
            st.markdown(f"**Long Fill**  \n${pos.long_fill_price:.2f}")
        with col3:
            if short:
                st.markdown(f"**Short Strike**  \n${short.strike:.0f}")
                st.markdown(f"**Short Exp**  \n{short.expiration_date}")
                st.markdown(f"**Short Fill**  \n${short.fill_price:.2f}")
            else:
                st.warning("No active short leg")
        with col4:
            st.metric("Net Short Credits",
                      f"${pos.net_short_credits:,.0f}")
            st.metric("Coverage",
                      f"{coverage:.1f}%")
            st.metric("Slippage",
                      f"${pos.total_slippage:+.2f}")

        # P&L breakdown
        st.markdown("---")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            color = _pnl_color(pos.long_pnl)
            st.markdown(
                f"**Long P&L:** "
                f"<span style='color:{color}'>${pos.long_pnl:+,.0f}</span>",
                unsafe_allow_html=True)
        with pc2:
            color = _pnl_color(pos.net_short_credits)
            st.markdown(
                f"**Short Credits (net):** "
                f"<span style='color:{color}'>${pos.net_short_credits:+,.0f}</span>",
                unsafe_allow_html=True)
        with pc3:
            color = _pnl_color(pnl)
            st.markdown(
                f"**Total P&L:** "
                f"<span style='color:{color}'>${pnl:+,.0f}</span>",
                unsafe_allow_html=True)

        # Roll form
        if short and short.is_open():
            st.markdown("---")
            st.markdown("#### 🔄 Roll Short Leg")

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                old_exit = st.number_input(
                    "Buy-back mid (at order)",
                    value=0.10, step=0.01, min_value=0.0,
                    key=f"roll_mid_{pid}")
                old_fill = st.number_input(
                    "Buy-back actual fill",
                    value=0.10, step=0.01, min_value=0.0,
                    key=f"roll_fill_{pid}")
            with rc2:
                new_strike = st.number_input(
                    "New short strike",
                    value=float(short.strike + 1),
                    step=1.0, key=f"new_strike_{pid}")
                new_exp = st.date_input(
                    "New expiration",
                    value=date.fromisoformat(short.expiration_date)
                          if short.expiration_date else date.today(),
                    key=f"new_exp_{pid}")
            with rc3:
                new_credit = st.number_input(
                    "New credit mid",
                    value=1.50, step=0.01, min_value=0.0,
                    key=f"new_credit_{pid}")
                new_fill = st.number_input(
                    "New credit actual fill",
                    value=1.50, step=0.01, min_value=0.0,
                    key=f"new_fill_{pid}")

            roll_reason = st.selectbox(
                "Roll reason",
                ["order_roll", "delta_trigger", "spike_guard",
                 "itm_threat", "manual"],
                key=f"roll_reason_{pid}")
            roll_notes = st.text_input(
                "Notes (optional)", key=f"roll_notes_{pid}")

            col_roll, col_close = st.columns([2, 1])
            with col_roll:
                if st.button(f"✅ Confirm Roll", key=f"roll_btn_{pid}",
                             type="primary"):
                    import yfinance as yf
                    try:
                        uvxy_price = float(
                            yf.Ticker("UVXY").history(
                                period="1d", interval="1m"
                            )["Close"].iloc[-1])
                    except:
                        uvxy_price = 0.0

                    tl.roll_short(
                        position_id      = pid,
                        old_exit_price   = old_exit,
                        old_fill_price   = old_fill,
                        new_strike       = new_strike,
                        new_expiration   = new_exp.isoformat(),
                        new_credit       = new_credit,
                        new_fill_price   = new_fill,
                        underlying_price = uvxy_price,
                        roll_reason      = roll_reason,
                        notes            = roll_notes,
                    )
                    reset_real_trade_log_cache()
                    st.success(
                        f"✅ Rolled to ${new_strike:.0f} exp {new_exp} "
                        f"· net roll credit: ${new_fill - old_fill:.2f}/c "
                        f"· slippage: ${new_fill - new_credit:.2f}")
                    st.rerun()

            with col_close:
                if st.button(f"❌ Close Position",
                             key=f"close_btn_{pid}"):
                    tl.close_position(pid, reason="manual")
                    reset_real_trade_log_cache()
                    st.success("Position closed")
                    st.rerun()

        # Roll history table
        if pos.roll_history:
            st.markdown("---")
            st.markdown("**Roll History**")
            rows = []
            for r in pos.roll_history:
                rows.append({
                    "Date":      r.roll_date,
                    "Old Strike": f"${r.old_strike:.0f}",
                    "BB Fill":   f"${r.old_fill_price:.2f}",
                    "New Strike": f"${r.new_strike:.0f}",
                    "New Fill":  f"${r.new_fill_price:.2f}",
                    "Net Credit": f"${r.roll_credit:.2f}",
                    "Reason":    r.roll_reason,
                    "Slippage":  f"${r.new_fill_price - r.new_credit:.2f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)


# ══════════════════════════════════════════════════════════
# NEW ENTRY FORM
# ══════════════════════════════════════════════════════════

def _render_new_entry(tl):
    st.markdown("### ➕ Open New Real Money Position")
    st.caption("Enter actual fill prices, not mid prices — for accurate slippage tracking.")

    import yfinance as yf

    # Auto-fetch current data
    try:
        uvxy_price = float(
            yf.Ticker("UVXY").history(period="1d", interval="1m")
            ["Close"].iloc[-1])
        vix_data   = yf.download("^VIX", period="5d", progress=False,
                                 auto_adjust=True)
        vix_level  = float(vix_data["Close"].iloc[-1])
    except:
        uvxy_price = 0.0
        vix_level  = 0.0

    # Import variant list
    try:
        from variant_generator import generate_variants, get_default_batch
        batch    = get_default_batch()
        variants = {v.role.value: v for v in batch.variants}
        var_names = [f"{v.role.value} — {v.name}" for v in batch.variants]
    except:
        variants  = {}
        var_names = ["V1_INCOME_HARVESTER", "V2_MEAN_REVERSION",
                     "V3_SHOCK_ABSORBER",  "V4_TAIL_HUNTER",
                     "V5_REGIME_ALLOCATOR"]

    st.info(f"Current market: UVXY ${uvxy_price:.2f}  |  VIX ${vix_level:.2f}")

    with st.form("new_real_entry"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Position Setup**")
            variant_sel = st.selectbox("Variant", var_names)
            contracts   = st.number_input(
                "Contracts", min_value=1, max_value=100, value=1)
            broker      = st.selectbox("Broker", BROKERS)
            account_id  = st.text_input("Account ID (last 6 digits)", "")
            regime      = st.selectbox(
                "Current Regime",
                ["CALM", "DECLINING", "RISING", "STRESSED", "EXTREME"])
            vix_pct     = st.number_input(
                "VIX Percentile (%)", value=13.0, step=0.1,
                min_value=0.0, max_value=100.0)

        with col2:
            st.markdown("**Long Leg (Debit)**")
            long_strike = st.number_input(
                "Long strike", value=round(uvxy_price + 5, 0),
                step=1.0)
            long_exp    = st.date_input("Long expiration")
            long_mid    = st.number_input(
                "Long mid price (at order)", value=5.0, step=0.01)
            long_fill   = st.number_input(
                "Long actual fill price", value=5.0, step=0.01)
            long_comm   = st.number_input(
                "Long commission ($/contract)", value=0.65, step=0.01)

            st.markdown("**Short Leg (Credit)**")
            short_strike = st.number_input(
                "Short strike", value=round(uvxy_price + 2, 0),
                step=1.0)
            short_exp    = st.date_input(
                "Short expiration",
                value=date.today())
            short_mid    = st.number_input(
                "Short mid price (at order)", value=2.0, step=0.01)
            short_fill   = st.number_input(
                "Short actual fill price", value=2.0, step=0.01)
            short_comm   = st.number_input(
                "Short commission ($/contract)", value=0.65, step=0.01)

        notes = st.text_area("Notes", placeholder="e.g. First real money trade, V1 CALM entry")

        submitted = st.form_submit_button(
            "💵 Open Real Position", type="primary")

    if submitted:
        # Extract variant_id from selection
        variant_id   = variant_sel.split(" — ")[0]
        variant_name = variant_sel.split(" — ")[1] if " — " in variant_sel \
                       else variant_id

        long_slip  = long_fill  - long_mid
        short_slip = short_fill - short_mid

        pos = tl.open_diagonal(
            variant_id       = variant_id,
            variant_name     = variant_name,
            regime           = regime,
            vix_level        = vix_level,
            vix_percentile   = vix_pct / 100,
            contracts        = int(contracts),
            long_strike      = long_strike,
            long_expiration  = long_exp.isoformat(),
            long_entry_price = long_mid,
            long_fill_price  = long_fill,
            short_strike     = short_strike,
            short_expiration = short_exp.isoformat(),
            short_credit     = short_mid,
            short_fill_price = short_fill,
            broker           = broker,
            account_id       = account_id,
            long_commission  = long_comm,
            short_commission = short_comm,
            notes            = notes,
        )

        reset_real_trade_log_cache()
        st.success(
            f"✅ Position opened: {pos.position_id}  \n"
            f"Long slippage: ${long_slip:+.2f}/c  |  "
            f"Short slippage: ${short_slip:+.2f}/c  |  "
            f"Net debit: ${long_fill - short_fill:.2f}/c")
        st.rerun()


# ══════════════════════════════════════════════════════════
# HISTORY & STATS
# ══════════════════════════════════════════════════════════

def _render_history(tl):
    all_pos = tl.diagonal_positions
    if not all_pos:
        st.info("No trade history yet.")
        return

    import pandas as pd

    rows = []
    for pid, pos in sorted(all_pos.items(),
                           key=lambda x: x[1].entry_date, reverse=True):
        rows.append({
            "ID":          pid[-12:],
            "Variant":     pos.variant_name,
            "Entry":       pos.entry_date,
            "Status":      pos.status.upper(),
            "Broker":      pos.broker,
            "Contracts":   pos.contracts,
            "Long Strike": f"${pos.long_strike:.0f}",
            "Long Fill":   f"${pos.long_fill_price:.2f}",
            "Net Credits": f"${pos.net_short_credits:,.0f}",
            "Total P&L":   f"${pos.total_pnl:+,.0f}",
            "Commissions": f"${pos.total_commissions:.2f}",
            "Slippage":    f"${pos.total_slippage:+.2f}",
            "Coverage%":   f"{pos.short_coverage_pct:.1f}%",
            "Rolls":       len(pos.roll_history),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)

    # Stats
    if len(rows) > 0:
        st.markdown("---")
        st.markdown("**Aggregate Stats**")
        sc1, sc2, sc3 = st.columns(3)
        total_comm = sum(p.total_commissions for p in all_pos.values())
        total_slip = sum(p.total_slippage    for p in all_pos.values())
        total_pnl  = sum(p.total_pnl         for p in all_pos.values())
        with sc1:
            st.metric("Total P&L (all)", f"${total_pnl:+,.0f}")
        with sc2:
            st.metric("Total Commissions", f"${total_comm:.2f}")
        with sc3:
            st.metric("Total Slippage", f"${total_slip:+.2f}")
