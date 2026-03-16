"""
VIX 5% Weekly Suite  ·  Real-Trade Edition
===========================================
Run:  streamlit run app.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import streamlit as st

from engine.regime    import fetch_market_data, RegimeState, REGIME_BANDS, get_regime_description
from engine.strategies import (
    STRATEGIES, get_active_strategies, adapt_params,
    calc_contracts, entry_signal_text,
)
from engine.options   import (
    build_option_leg, price_diagonal, mark_to_market,
    get_weekly_expiries, get_monthly_expiries,
)
from pages.page_assignments import render as render_assignments
from data.store import (
    Position, load_open, load_closed, save_position,
    close_position, mark_position, delete_position, portfolio_summary,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIX 5W Suite · Real Trade",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card { background:#1e293b; border-radius:10px; padding:16px 20px;
    border-left:4px solid var(--c); margin-bottom:12px; }
.regime-banner { border-radius:12px; padding:20px 24px; margin-bottom:20px;
    font-size:1.1rem; font-weight:600; }
.pos-card { background:#1e293b; border-radius:12px; padding:18px 22px;
    margin-bottom:14px; border:1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 VIX 5W Suite")
    st.markdown("---")
    page = st.radio("Navigate",
        ["🏠 Dashboard","🎯 Signal Center","📋 Positions","🔄 Roll Advisor","⚠️ Risk Monitor","📈 Analytics","📋 Assignments"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Settings**")
    account_size = st.number_input("Account Size ($)", min_value=5000, max_value=10_000_000,
        value=st.session_state.get("account_size", 50000), step=5000, key="account_size")
    lookback = st.selectbox("VIX Lookback", [126,252,504], index=1,
        format_func=lambda x: f"{x}d ({x//21}mo)")
    st.markdown("---")
    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _get_regime(lb):
    return fetch_market_data(lb)

with st.spinner("Fetching market data…"):
    try:
        rs = _get_regime(lookback)
    except Exception as e:
        st.error(f"Market data error: {e}")
        st.stop()


# ═══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════
def page_dashboard(rs):
    st.title("🏠 Live Dashboard")
    pct   = rs.vix_percentile
    color = rs.regime_color
    st.markdown(f"""
    <div class="regime-banner" style="background:{color}22;border:2px solid {color};color:#f1f5f9;">
        {rs.regime_emoji}&nbsp;&nbsp;<strong>Regime: {rs.regime}</strong>
        &nbsp;|&nbsp; VIX {rs.vix_current:.2f} &nbsp;({pct:.1f}th pct, {lookback}d)
        &nbsp;|&nbsp; Trend: {rs.trend_arrow}&nbsp;{rs.vix_trend} ({rs.vix_1w_chg_pct:+.1f}%/1wk)
        &nbsp;|&nbsp; {rs.vix_level}
    </div>""", unsafe_allow_html=True)
    st.caption(get_regime_description(rs.regime))

    open_pos = load_open()
    summary  = portfolio_summary(open_pos)
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("VIX", f"{rs.vix_current:.2f}", delta=f"{rs.vix_1w_chg_pct:+.1f}%", delta_color="inverse")
    col2.metric("UVXY", f"${rs.uvxy_price:.2f}" if rs.uvxy_price else "N/A")
    col3.metric("VXX",  f"${rs.vxx_price:.2f}"  if rs.vxx_price  else "N/A")
    col4.metric("Open Positions", summary["open_count"])
    col5.metric("Unrealized P&L", f"${summary['total_unrealized_pnl']:,.0f}")

    st.markdown("### 📊 VIX Percentile History")
    if not rs.percentile_series.empty:
        try:
            import altair as alt
            chart_df = rs.percentile_series.rename("Percentile").reset_index()
            chart_df.columns = ["Date","Percentile"]
            base = alt.Chart(chart_df).mark_area(color=color,opacity=0.15,
                line={"color":color,"strokeWidth":1.5}).encode(
                x=alt.X("Date:T",title=""),
                y=alt.Y("Percentile:Q",scale=alt.Scale(domain=[0,100]),title="VIX Percentile"))
            rules = alt.Chart(pd.DataFrame({"y":[20,40,60,80]})).mark_rule(
                strokeDash=[4,4],color="#475569",strokeWidth=1).encode(y="y:Q")
            st.altair_chart((base+rules).properties(height=200), use_container_width=True)
        except Exception:
            st.line_chart(rs.percentile_series)

    st.markdown("### 🎯 Strategy Status")
    active = get_active_strategies(rs)
    active_codes = {s.code for s in active}
    cols = st.columns(5)
    for i,(code,strat) in enumerate(STRATEGIES.items()):
        with cols[i]:
            is_on = code in active_codes
            bc = "#22c55e" if is_on else "#475569"
            p = adapt_params(strat, rs.regime)
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:10px;padding:14px;
                border-left:4px solid {bc};margin-bottom:4px;">
                <div style="font-size:1.3rem">{strat.emoji}</div>
                <strong style="color:#f1f5f9">{code}</strong><br>
                <small style="color:#94a3b8">{strat.name}</small><br>
                <span style="background:{bc}22;color:{bc};padding:2px 8px;
                    border-radius:4px;font-size:0.7rem;font-weight:700;">
                    {"✅ ACTIVE" if is_on else "⏸ INACTIVE"}</span>
                <div style="margin-top:8px;font-size:0.78rem;color:#64748b">
                    Short {int(p.short_moneyness*100-100)}% OTM · {p.short_dte_target}d DTE<br>
                    Max {p.max_concurrent} pos · {int(p.account_pct_per*100)}% acct each</div>
            </div>""", unsafe_allow_html=True)

    if open_pos:
        alerts = [(p,p.health) for p in open_pos if p.health != "HEALTHY"]
        if alerts:
            st.markdown("### ⚠️ Action Required")
            icons = {"ROLL_NOW":"🔄","TAKE_PROFIT":"💚","STOP_LOSS":"🛑","CRITICAL":"🚨","WATCH":"👁"}
            colors = {"ROLL_NOW":"#8b5cf6","TAKE_PROFIT":"#22c55e","STOP_LOSS":"#ef4444","CRITICAL":"#ef4444","WATCH":"#f59e0b"}
            for pos,health in alerts:
                ic = icons.get(health,"ℹ️"); hc = colors.get(health,"#94a3b8")
                st.markdown(f"""
                <div style="background:{hc}15;border:1px solid {hc};border-radius:8px;
                    padding:10px 16px;margin-bottom:8px;color:#f1f5f9;">
                    {ic} <strong>{pos.strategy} #{pos.pos_id}</strong>
                    — {health.replace('_',' ')} · P&L: ${pos.unrealized_pnl:+,.0f}
                    ({pos.unrealized_pnl_pct:+.1f}%) · Short DTE: {pos.short_dte}
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No open positions. Go to **Signal Center** to find entries.")


# ═══════════════════════════════════════════════════════════════
# PAGE: SIGNAL CENTER
# ═══════════════════════════════════════════════════════════════
def page_signals(rs):
    st.title("🎯 Signal Center")
    uvxy = rs.uvxy_price
    if not uvxy:
        st.error("UVXY price unavailable."); return

    active_strats = get_active_strategies(rs)
    active_codes  = {s.code for s in active_strats}
    open_by_strat = {}
    for p in load_open():
        open_by_strat.setdefault(p.strategy,[]).append(p)

    st.markdown(f"**UVXY:** ${uvxy:.2f}  · **VIX:** {rs.vix_current:.2f} ({rs.vix_percentile:.1f}th pct) · Regime: **{rs.regime}**")
    weekly_exp  = get_weekly_expiries(8)
    monthly_exp = get_monthly_expiries(6)

    for code,strat in STRATEGIES.items():
        p = adapt_params(strat, rs.regime)
        is_on   = code in active_codes
        n_open  = len(open_by_strat.get(code,[]))
        can_enter = is_on and n_open < p.max_concurrent

        with st.expander(
            f"{strat.emoji} **{code} – {strat.name}**  "
            f"{'✅ ACTIVE' if is_on else '⏸ INACTIVE'}  "
            f"| {n_open}/{p.max_concurrent} positions",
            expanded=is_on and can_enter):

            if not is_on:
                st.caption(f"Inactive in {rs.regime}. Activates in: {', '.join(strat.active_regimes)}"); continue
            if not can_enter:
                st.warning(f"Max {p.max_concurrent} concurrent positions reached.")

            long_strike  = round(uvxy * p.long_moneyness,  2)
            short_strike = round(uvxy * p.short_moneyness, 2)
            long_expiry_cands  = [e for e in monthly_exp if (e-date.today()).days >= p.long_dte_min]
            short_expiry_cands = [e for e in weekly_exp  if (e-date.today()).days >= p.short_dte_min]
            if not long_expiry_cands or not short_expiry_cands:
                st.error("No valid expiries found."); continue

            long_expiry  = min(long_expiry_cands,  key=lambda e: abs((e-date.today()).days - p.long_dte_target))
            short_expiry = min(short_expiry_cands, key=lambda e: abs((e-date.today()).days - p.short_dte_target))
            diag = price_diagonal(uvxy, long_strike, long_expiry, short_strike, short_expiry, rs.vix_current)
            contracts = calc_contracts(account_size, diag.net_debit, p.account_pct_per)

            c1,c2 = st.columns([3,2])
            with c1:
                st.markdown(f"*{strat.description}*")
                st.markdown("---")
                mc1,mc2 = st.columns(2)
                with mc1:
                    st.markdown("**📗 Long LEAP**")
                    for lbl,val in [("Strike",f"`{long_strike:.2f}C`"),("Expiry",f"`{long_expiry.strftime('%b %d, %Y')}`"),
                        ("DTE",f"`{(long_expiry-date.today()).days}d`"),("Ask",f"`${diag.long_leg.ask:.2f}`"),
                        ("Delta",f"`{diag.long_leg.delta:.3f}`"),("IV",f"`{diag.long_leg.iv*100:.1f}%`")]:
                        st.markdown(f"{lbl}: {val}")
                with mc2:
                    st.markdown("**📕 Short Weekly**")
                    for lbl,val in [("Strike",f"`{short_strike:.2f}C`"),("Expiry",f"`{short_expiry.strftime('%b %d, %Y')}`"),
                        ("DTE",f"`{(short_expiry-date.today()).days}d`"),("Bid",f"`${diag.short_leg.bid:.2f}`"),
                        ("Delta",f"`{diag.short_leg.delta:.3f}`"),("IV",f"`{diag.short_leg.iv*100:.1f}%`")]:
                        st.markdown(f"{lbl}: {val}")

                target_net = round(diag.net_debit*(1+p.profit_target_pct),2)
                stop_net   = round(diag.net_debit*(1-p.stop_loss_pct),2)
                st.markdown(f"""
| | |
|---|---|
| Net Debit | **${diag.net_debit:.2f}** per contract |
| Contracts | **{contracts}** (${account_size:,} acct) |
| Total at Risk | **${diag.net_debit*contracts*100:,.0f}** |
| Profit Target | Net ≥ ${target_net:.2f} (+{int(p.profit_target_pct*100)}%) |
| Stop Loss | Net ≤ ${stop_net:.2f} (-{int(p.stop_loss_pct*100)}%) |
| Roll at DTE | ≤ {p.roll_at_dte} days |
| Max Profit | ${diag.max_profit:.2f} · R/R: {diag.reward_risk:.2f}× |
| Breakeven | ${diag.breakeven:.2f} at expiry |
| Net Delta | {diag.net_delta:.3f} |
| Net Daily Theta | ${diag.net_theta*contracts*100:.2f} |
""")

            with c2:
                st.markdown("**📥 Record Entry**")
                if not can_enter:
                    st.info("Max positions reached.")
                else:
                    with st.form(f"entry_{code}"):
                        f_ls = st.number_input("Long Strike",  value=long_strike,  step=0.5, format="%.2f", key=f"ls_{code}")
                        f_le = st.date_input("Long Expiry",    value=long_expiry,              key=f"le_{code}")
                        f_la = st.number_input("Long Ask (fill)", value=diag.long_leg.ask, step=0.01, format="%.2f", key=f"la_{code}")
                        f_ss = st.number_input("Short Strike", value=short_strike, step=0.5, format="%.2f", key=f"ss_{code}")
                        f_se = st.date_input("Short Expiry",  value=short_expiry,             key=f"se_{code}")
                        f_sb = st.number_input("Short Bid (fill)", value=diag.short_leg.bid, step=0.01, format="%.2f", key=f"sb_{code}")
                        f_qty = st.number_input("Contracts", min_value=1, value=contracts, step=1, key=f"qty_{code}")
                        f_notes = st.text_input("Notes", "", key=f"notes_{code}")
                        actual_debit = round(f_la - f_sb, 2)
                        st.info(f"Net debit: **${actual_debit:.2f}** × {f_qty} = **${actual_debit*f_qty*100:,.0f}**")
                        if st.form_submit_button("✅ Record Trade", type="primary", use_container_width=True):
                            pos = Position(
                                strategy=code, entry_date=date.today().isoformat(),
                                entry_regime=rs.regime, vix_at_entry=rs.vix_current,
                                uvxy_at_entry=rs.uvxy_price or 0.0, contracts=f_qty,
                                long_strike=f_ls, long_expiry=f_le.isoformat(), long_entry_ask=f_la,
                                short_strike=f_ss, short_expiry=f_se.isoformat(), short_entry_bid=f_sb,
                                net_debit=actual_debit,
                                profit_target_net=round(actual_debit*(1+p.profit_target_pct),2),
                                stop_loss_net=round(actual_debit*(1-p.stop_loss_pct),2),
                                roll_at_dte=p.roll_at_dte, current_net=actual_debit, notes=f_notes)
                            save_position(pos)
                            st.success(f"✅ {code} position #{pos.pos_id} recorded!")
                            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: POSITIONS
# ═══════════════════════════════════════════════════════════════
def page_positions(rs):
    st.title("📋 Position Manager")
    tab1,tab2,tab3 = st.tabs(["🟢 Open","🔒 Closed History","➕ Manual Add"])

    with tab1:
        open_pos = load_open()
        if not open_pos:
            st.info("No open positions."); return

        col_l,col_r = st.columns([3,1])
        with col_r:
            if st.button("🔄 Mark All to Market", use_container_width=True):
                if rs.uvxy_price:
                    for pos in open_pos:
                        m = mark_to_market(pos.long_strike, pos.long_expiry_date,
                            pos.short_strike, pos.short_expiry_date,
                            pos.net_debit, rs.uvxy_price, rs.vix_current)
                        mark_position(pos.pos_id, m["current_net"])
                    st.success("All marked."); st.rerun()

        for pos in sorted(open_pos, key=lambda p: p.health):
            health = pos.health; hc = pos.health_color
            icons  = {"HEALTHY":"✅","WATCH":"👁","CRITICAL":"🚨","ROLL_NOW":"🔄","TAKE_PROFIT":"💚","STOP_LOSS":"🛑"}
            hi     = icons.get(health,"ℹ️")
            strat  = STRATEGIES.get(pos.strategy)
            emoji  = strat.emoji if strat else "📊"
            pnl_c  = "#22c55e" if pos.unrealized_pnl >= 0 else "#ef4444"

            st.markdown(f"""
            <div class="pos-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:1.1rem;font-weight:700;color:#f1f5f9;">
                        {emoji} {pos.strategy} #{pos.pos_id} &nbsp;·&nbsp; {pos.contracts} contracts
                    </span>
                    <span style="background:{hc}22;color:{hc};padding:3px 12px;
                        border-radius:6px;font-size:0.75rem;font-weight:700;">
                        {hi} {health.replace("_"," ")}</span>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;
                    margin-top:12px;font-size:0.85rem;color:#94a3b8;">
                    <div><div style="color:#64748b;font-size:0.7rem">LONG LEG</div>
                        {pos.long_strike:.2f}C · {pos.long_expiry} · {pos.long_dte}d</div>
                    <div><div style="color:#64748b;font-size:0.7rem">SHORT LEG</div>
                        {pos.short_strike:.2f}C · {pos.short_expiry} · <strong style="color:{hc}">{pos.short_dte}d</strong></div>
                    <div><div style="color:#64748b;font-size:0.7rem">P&L</div>
                        <strong style="color:{pnl_c}">${pos.unrealized_pnl:+,.0f} ({pos.unrealized_pnl_pct:+.1f}%)</strong></div>
                    <div><div style="color:#64748b;font-size:0.7rem">ENTRY</div>
                        ${pos.net_debit:.2f} debit · {pos.entry_date}</div>
                </div>
                <div style="margin-top:8px;font-size:0.78rem;color:#475569;">
                    Entry regime: {pos.entry_regime} · VIX@entry: {pos.vix_at_entry:.1f}
                    · Held: {pos.days_held}d · Target: ${pos.profit_target_net:.2f}
                    · Stop: ${pos.stop_loss_net:.2f} · Rolls: {pos.roll_count}
                    {" · " + pos.notes if pos.notes else ""}
                </div>
            </div>""", unsafe_allow_html=True)

            ca,cb,cc,cd = st.columns(4)
            with ca:
                with st.popover("📊 Update Mark"):
                    nv = st.number_input("Current Net", value=float(pos.current_net), step=0.01, format="%.2f", key=f"nm_{pos.pos_id}")
                    if st.button("Update", key=f"updbtn_{pos.pos_id}"):
                        mark_position(pos.pos_id, nv); st.rerun()
            with cb:
                with st.popover("🔒 Close"):
                    ex = st.number_input("Exit Net", value=float(pos.current_net or pos.net_debit), step=0.01, format="%.2f", key=f"en_{pos.pos_id}")
                    er = st.selectbox("Reason", ["TP","SL","MANUAL","EXPIRY"], key=f"er_{pos.pos_id}")
                    if st.button("Confirm Close", type="primary", key=f"clsbtn_{pos.pos_id}"):
                        r = close_position(pos.pos_id, ex, er)
                        if r: st.success(f"Realized: ${r.realized_pnl:+,.0f}"); st.rerun()
            with cc:
                with st.popover("🔄 Roll Short"):
                    uvxy_now = rs.uvxy_price or pos.uvxy_at_entry
                    sp = adapt_params(STRATEGIES[pos.strategy], rs.regime) if pos.strategy in STRATEGIES else None
                    def_mono = sp.short_moneyness if sp else 1.08
                    def_ss = round(uvxy_now * def_mono, 2)
                    new_exp = get_weekly_expiries(6)
                    rss = st.number_input("New Strike", value=def_ss, step=0.5, format="%.2f", key=f"rss_{pos.pos_id}")
                    rse = st.selectbox("New Expiry", new_exp, format_func=lambda d: d.strftime("%b %d"), key=f"rse_{pos.pos_id}")
                    rbid = st.number_input("New Short Bid", value=0.15, step=0.01, format="%.2f", key=f"rbid_{pos.pos_id}")
                    bcc  = st.number_input("Buy-to-close cost", value=0.05, step=0.01, format="%.2f", key=f"bcc_{pos.pos_id}")
                    net_rc = round(rbid - bcc, 2)
                    new_d  = round(pos.net_debit - net_rc, 2)
                    st.info(f"Roll credit: ${net_rc:.2f} → new debit: ${new_d:.2f}")
                    if st.button("Execute Roll", type="primary", key=f"rollbtn_{pos.pos_id}"):
                        close_position(pos.pos_id, pos.current_net, "ROLL")
                        s2 = STRATEGIES.get(pos.strategy)
                        new_pos = Position(
                            strategy=pos.strategy, entry_date=date.today().isoformat(),
                            entry_regime=rs.regime, vix_at_entry=rs.vix_current,
                            uvxy_at_entry=uvxy_now, contracts=pos.contracts,
                            long_strike=pos.long_strike, long_expiry=pos.long_expiry,
                            long_entry_ask=pos.long_entry_ask,
                            short_strike=rss, short_expiry=rse.isoformat(),
                            short_entry_bid=rbid, net_debit=new_d,
                            profit_target_net=round(new_d*(1+(sp.profit_target_pct if sp else 0.5)),2),
                            stop_loss_net=round(new_d*(1-(sp.stop_loss_pct if sp else 0.3)),2),
                            roll_at_dte=s2.roll_at_dte if s2 else 4,
                            current_net=new_d, parent_id=pos.pos_id,
                            roll_count=pos.roll_count+1, notes=f"Rolled from #{pos.pos_id}")
                        save_position(new_pos)
                        st.success(f"Rolled to #{new_pos.pos_id}"); st.rerun()
            with cd:
                with st.popover("🗑️ Delete"):
                    st.warning("Permanently delete?")
                    if st.button("Confirm", type="primary", key=f"delbtn_{pos.pos_id}"):
                        delete_position(pos.pos_id); st.rerun()

    with tab2:
        closed = load_closed()
        if not closed:
            st.info("No closed positions yet.")
        else:
            total_pnl = sum(p.realized_pnl for p in closed)
            wins = sum(1 for p in closed if p.realized_pnl > 0)
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Realized P&L", f"${total_pnl:+,.0f}")
            k2.metric("Trades", len(closed))
            k3.metric("Win Rate", f"{wins/len(closed)*100:.1f}%")
            k4.metric("Avg P&L/Trade", f"${total_pnl/len(closed):+,.0f}")
            df = pd.DataFrame([{
                "ID":pos.pos_id,"Strategy":pos.strategy,
                "Entry":pos.entry_date,"Exit":pos.exit_date,"Held(d)":pos.days_held,
                "Qty":pos.contracts,"Debit":f"${pos.net_debit:.2f}",
                "Exit Net":f"${pos.exit_net:.2f}","P&L":f"${pos.realized_pnl:+,.0f}",
                "Reason":pos.exit_reason,"Rolls":pos.roll_count,"Regime":pos.entry_regime,
            } for p in sorted(closed,key=lambda x:x.exit_date,reverse=True)])
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("Manually record a position entered through your broker.")
        with st.form("manual_add"):
            mc1,mc2 = st.columns(2)
            with mc1:
                m_strat  = st.selectbox("Strategy", list(STRATEGIES.keys()))
                m_entry  = st.date_input("Entry Date", value=date.today())
                m_regime = st.selectbox("Regime at Entry", [r[2] for r in REGIME_BANDS], index=2)
                m_vix    = st.number_input("VIX at Entry", value=rs.vix_current, step=0.1, format="%.2f")
                m_uvxy   = st.number_input("UVXY at Entry", value=float(rs.uvxy_price or 15), step=0.01, format="%.2f")
                m_qty    = st.number_input("Contracts", min_value=1, value=1, step=1)
            with mc2:
                m_ls = st.number_input("Long Strike",  value=float(rs.uvxy_price or 15), step=0.5, format="%.2f")
                m_le = st.date_input("Long Expiry",    value=date.today()+timedelta(days=180))
                m_la = st.number_input("Long Ask Fill", value=3.00, step=0.01, format="%.2f")
                m_ss = st.number_input("Short Strike", value=float((rs.uvxy_price or 15)*1.07), step=0.5, format="%.2f")
                m_se = st.date_input("Short Expiry",  value=get_weekly_expiries(1)[0])
                m_sb = st.number_input("Short Bid Fill",value=0.15, step=0.01, format="%.2f")
            m_notes = st.text_input("Notes")
            m_debit = round(m_la - m_sb, 2)
            st.info(f"Net Debit: **${m_debit:.2f}** × {m_qty} = **${m_debit*m_qty*100:,.0f}**")
            sp = adapt_params(STRATEGIES[m_strat], m_regime)
            if st.form_submit_button("Add Position", type="primary"):
                pos = Position(
                    strategy=m_strat, entry_date=m_entry.isoformat(),
                    entry_regime=m_regime, vix_at_entry=m_vix,
                    uvxy_at_entry=m_uvxy, contracts=m_qty,
                    long_strike=m_ls, long_expiry=m_le.isoformat(), long_entry_ask=m_la,
                    short_strike=m_ss, short_expiry=m_se.isoformat(), short_entry_bid=m_sb,
                    net_debit=m_debit,
                    profit_target_net=round(m_debit*(1+sp.profit_target_pct),2),
                    stop_loss_net=round(m_debit*(1-sp.stop_loss_pct),2),
                    roll_at_dte=sp.roll_at_dte, current_net=m_debit, notes=m_notes)
                save_position(pos)
                st.success(f"Position {pos.pos_id} added!"); st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: ROLL ADVISOR
# ═══════════════════════════════════════════════════════════════
def page_roll_advisor(rs):
    st.title("🔄 Roll Advisor")
    st.caption("Smart roll recommendations based on DTE, P&L, and regime.")
    open_pos = load_open()
    uvxy = rs.uvxy_price
    if not open_pos or not uvxy:
        st.info("No open positions or UVXY unavailable."); return

    roll_items = []
    for pos in open_pos:
        sp = adapt_params(STRATEGIES[pos.strategy], rs.regime) if pos.strategy in STRATEGIES else None
        mtm = mark_to_market(pos.long_strike, pos.long_expiry_date,
            pos.short_strike, pos.short_expiry_date,
            pos.net_debit, uvxy, rs.vix_current)
        roll_at = sp.roll_at_dte if sp else 4
        dte_score = max(0, (roll_at+3 - pos.short_dte)) * 15
        pnl_score = max(0, mtm["pnl_pct"]) * 0.5
        score = min(dte_score + pnl_score, 100)
        roll_items.append((pos, mtm, sp, score))
    roll_items.sort(key=lambda x: x[3], reverse=True)

    for pos,mtm,sp,score in roll_items:
        pri = "🔴 URGENT" if score>70 else ("🟡 SOON" if score>30 else "🟢 MONITOR")
        with st.expander(f"{pri}  {pos.strategy} #{pos.pos_id}  |  Score: {score:.0f}/100  |  Short DTE: {pos.short_dte}", expanded=score>50):
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"**Current Short:** {pos.short_strike:.2f}C · {pos.short_expiry} · **{pos.short_dte} DTE**")
                st.markdown(f"**Short Mark:** ${mtm['short_mark']:.2f}")
                st.markdown(f"**P&L:** ${mtm['pnl']:+,.2f} ({mtm['pnl_pct']:+.1f}%)")
                st.markdown(f"**Roll threshold:** ≤{sp.roll_at_dte if sp else 4} DTE")
            with c2:
                next_exp = get_weekly_expiries(5)
                def_mono = sp.short_moneyness if sp else 1.08
                tgt_dte  = sp.short_dte_target if sp else 10
                new_ss   = round(uvxy * def_mono, 2)
                tgt_exp  = min(next_exp, key=lambda e: abs((e-date.today()).days - tgt_dte))
                roll_leg = build_option_leg(uvxy, new_ss, tgt_exp, rs.vix_current, "C")
                close_val= mtm["short_leg"].mid if mtm.get("short_leg") else pos.short_entry_bid*0.1
                roll_cr  = round(roll_leg.bid - close_val, 2)
                new_d    = round(pos.net_debit - roll_cr, 2)
                st.markdown(f"**Suggested Roll:**")
                st.markdown(f"→ Buy to close existing: ${close_val:.2f}")
                st.markdown(f"→ Sell {new_ss:.2f}C {tgt_exp.strftime('%b %d')}: ${roll_leg.bid:.2f}")
                st.markdown(f"→ Net roll credit: **${roll_cr:+.2f}**")
                st.markdown(f"→ New position debit: **${new_d:.2f}**")
                if roll_cr > 0:
                    st.success(f"✅ Collect additional ${roll_cr:.2f}")
                elif roll_cr > -0.10:
                    st.info("Near-even roll. Reasonable for DTE extension.")
                else:
                    st.warning(f"${roll_cr:.2f} debit roll. Wait or adjust strike.")
            if pos.entry_regime != rs.regime:
                st.warning(f"⚠️ Regime drift: opened in **{pos.entry_regime}** → now **{rs.regime}**. Reconsider strikes.")

# ═══════════════════════════════════════════════════════════════
# PAGE: RISK MONITOR
# ═══════════════════════════════════════════════════════════════
def page_risk(rs):
    st.title("⚠️ Risk Monitor")
    open_pos = load_open()
    uvxy = rs.uvxy_price
    if not open_pos:
        st.info("No open positions."); return

    total_delta=total_theta=total_vega=total_at_risk=0.0
    mtm_all = []
    for pos in open_pos:
        if uvxy:
            mtm = mark_to_market(pos.long_strike, pos.long_expiry_date,
                pos.short_strike, pos.short_expiry_date,
                pos.net_debit, uvxy, rs.vix_current)
        else:
            mtm = {"long_leg":None,"short_leg":None,"pnl":0,"pnl_pct":0,"current_net":pos.net_debit}
        mtm_all.append((pos,mtm))
        if uvxy and mtm.get("long_leg") and mtm.get("short_leg"):
            ll,sl,n = mtm["long_leg"],mtm["short_leg"],pos.contracts
            total_delta += (ll.delta - sl.delta)*n*100
            total_theta += (sl.theta - ll.theta)*n
            total_vega  += (ll.vega  - sl.vega )*n*100
        total_at_risk += pos.net_debit*pos.contracts*100

    total_pnl = sum(pos.unrealized_pnl for pos in open_pos)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total at Risk",    f"${total_at_risk:,.0f}")
    k2.metric("Unrealized P&L",   f"${total_pnl:+,.0f}")
    k3.metric("Portfolio Delta",  f"{total_delta:+.1f}")
    k4.metric("Daily Theta",      f"${total_theta:+.2f}")
    k5.metric("Vega (per 1% IV)", f"${total_vega:+.2f}")

    st.markdown("### Position Risk Table")
    rows = []
    for pos,mtm in mtm_all:
        rows.append({"ID":pos.pos_id,"Strat":pos.strategy,"Qty":pos.contracts,
            "Debit":f"${pos.net_debit:.2f}","Max Loss":f"${-pos.net_debit*pos.contracts*100:,.0f}",
            "Curr Net":f"${mtm['current_net']:.2f}","P&L":f"${mtm['pnl']:+.2f}",
            "P&L%":f"{mtm['pnl_pct']:+.1f}%","ShortDTE":pos.short_dte,
            "LongDTE":pos.long_dte,"Health":pos.health,"Regime":pos.entry_regime})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 🔥 VIX Stress Scenarios")
    uvxy_now = rs.uvxy_price or 20
    scenarios = [
        ("VIX 12 · Complacency",12, uvxy_now*0.65),
        ("VIX 18 · Normal",     18, uvxy_now*0.88),
        ("VIX 25 · Elevated",   25, uvxy_now*1.10),
        ("VIX 35 · Stress",     35, uvxy_now*1.60),
        ("VIX 50 · Crisis",     50, uvxy_now*2.40),
        ("VIX 75 · Crash",      75, uvxy_now*3.80),
    ]
    srows = []
    for lbl,sv,su in scenarios:
        tp = sum(mark_to_market(pos.long_strike,pos.long_expiry_date,
            pos.short_strike,pos.short_expiry_date,
            pos.net_debit,su,sv)["pnl"]*pos.contracts*100 for pos in open_pos)
        srows.append({"Scenario":lbl,"VIX":sv,
            "Est UVXY":f"${su:.2f}","Portfolio P&L":f"${tp:+,.0f}",
            "% at Risk":f"{tp/total_at_risk*100:+.1f}%" if total_at_risk else "N/A"})
    st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════
def page_analytics(rs):
    st.title("📈 Performance Analytics")
    closed = load_closed()
    if not closed:
        st.info("No closed trades yet."); return

    total_pnl = sum(p.realized_pnl for p in closed)
    wins = sum(1 for p in closed if p.realized_pnl > 0)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Realized P&L", f"${total_pnl:+,.0f}")
    k2.metric("Trades",len(closed))
    k3.metric("Win Rate",f"{wins/len(closed)*100:.1f}%")
    k4.metric("Avg P&L/Trade",f"${total_pnl/len(closed):+,.0f}")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### By Strategy")
        by_s = {}
        for p in closed: by_s.setdefault(p.strategy,[]).append(p.realized_pnl)
        rows = [{"Strategy":c,"Trades":len(v),"Win Rate":f"{sum(1 for x in v if x>0)/len(v)*100:.1f}%",
            "Total P&L":f"${sum(v):+,.0f}","Avg":f"${sum(v)/len(v):+,.0f}"}
            for c,v in sorted(by_s.items())]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### By Entry Regime")
        by_r = {}
        for p in closed: by_r.setdefault(p.entry_regime,[]).append(p.realized_pnl)
        rows = [{"Regime":r,"Trades":len(v),"Win Rate":f"{sum(1 for x in v if x>0)/len(v)*100:.1f}%",
            "Total P&L":f"${sum(v):+,.0f}"}
            for r,v in sorted(by_r.items())]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Cumulative P&L")
    sorted_c = sorted(closed, key=lambda p: p.exit_date)
    cum=0; chart_data=[]
    for p in sorted_c:
        cum += p.realized_pnl
        chart_data.append({"Date":p.exit_date,"Cum P&L":cum})
    if chart_data:
        cdf = pd.DataFrame(chart_data)
        cdf["Date"] = pd.to_datetime(cdf["Date"])
        try:
            import altair as alt
            line = alt.Chart(cdf).mark_line(color="#3b82f6",strokeWidth=2).encode(
                x=alt.X("Date:T"),y=alt.Y("Cum P&L:Q"))
            area = alt.Chart(cdf).mark_area(color="#3b82f6",opacity=0.1).encode(
                x="Date:T",y="Cum P&L:Q")
            st.altair_chart((area+line).properties(height=220), use_container_width=True)
        except Exception:
            st.line_chart(cdf.set_index("Date"))

    rolled = [p for p in closed if p.exit_reason=="ROLL"]
    if rolled:
        st.markdown(f"### Roll History  ({len(rolled)} rolls)")
        rd = [{"ID":p.pos_id,"Strategy":p.strategy,"Roll#":p.roll_count,
            "Exit Net":f"${p.exit_net:.2f}","Credited":f"${(p.exit_net-p.net_debit)*p.contracts*100:+,.0f}"}
            for p in rolled]
        st.dataframe(pd.DataFrame(rd), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════
if   page == "🏠 Dashboard":    page_dashboard(rs)
elif page == "🎯 Signal Center": page_signals(rs)
elif page == "📋 Positions":     page_positions(rs)
elif page == "🔄 Roll Advisor":  page_roll_advisor(rs)
elif page == "⚠️ Risk Monitor":  page_risk(rs)
elif page == "📈 Analytics":     page_analytics(rs)
elif page == "📋 Assignments":   render_assignments(uvxy_price=float(rs.uvxy_price or 0))
