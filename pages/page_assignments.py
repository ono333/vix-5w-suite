"""
Assignment Log  —  Real & Paper
================================
Full CRUD for share-assignment events.
• Real tab  : live Fidelity positions
• Paper tab : simulated / research positions
• Edit inline via popover forms
• Delete with two-step confirmation
• Add new via form
• Auto-import from legacy assignment_engine on first load
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date
from typing import List

import pandas as pd
import streamlit as st

from data.assignment_store import (
    AssignmentEvent, load_all, load_open, load_closed,
    save_event, update_event, delete_event, close_event,
    migrate_from_assignment_engine, portfolio_summary,
)

# ── Constants ─────────────────────────────────────────────────────────────────
STRATEGY_CHOICES = [
    "V1 Income Harvester",
    "V2 Mean Reversion",
    "V3 Shock Absorber",
    "V4 Spike Trade",
    "V5 Regime Allocator",
    "Manual / Other",
]
STATUS_COLORS = {
    "open":   ("#22c55e", "🟢"),
    "closed": ("#94a3b8", "⚫"),
}


# ── Main page entry ───────────────────────────────────────────────────────────

def render(uvxy_price: float = 0.0):
    """
    Call this from app.py:
        from pages.page_assignments import render
        render(uvxy_price=rs.uvxy_price or 0.0)
    """
    st.title("📋 Assignment Log")
    st.caption("Track equity assignments (short/long shares) alongside your diagonal spreads.")

    # ── One-time migration from legacy assignment_engine ──────────────────────
    if "assignment_migrated" not in st.session_state:
        n = migrate_from_assignment_engine("real")
        if n > 0:
            st.success(f"✅ Imported {n} record(s) from existing assignment_engine.")
        st.session_state["assignment_migrated"] = True

    # ── Tab selector ──────────────────────────────────────────────────────────
    tab_real, tab_paper = st.tabs(["🏦 Real Trading", "🧪 Paper Trading"])

    with tab_real:
        _render_mode("real", uvxy_price)

    with tab_paper:
        _render_mode("paper", uvxy_price)


# ── Per-mode renderer ─────────────────────────────────────────────────────────

def _render_mode(mode: str, uvxy_price: float):
    label = "Real" if mode == "real" else "Paper"
    all_events = load_all(mode)
    open_evs   = [e for e in all_events if e.status == "open"]
    closed_evs = [e for e in all_events if e.status == "closed"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    summary = portfolio_summary(mode, uvxy_price)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open Positions", summary["open_count"])
    k2.metric("Net Shares",     f"{summary['net_shares']:+,d}")
    k3.metric("Notional",       f"${summary['total_notional']:,.0f}")
    if uvxy_price:
        pnl_c = "normal" if summary["total_unrealized"] >= 0 else "inverse"
        k4.metric("Unrealized P&L", f"${summary['total_unrealized']:+,.0f}", delta_color=pnl_c)
    else:
        k4.metric("Unrealized P&L", "—", help="Enter UVXY price below to calculate")
    k5.metric("Realized P&L (all time)", f"${summary['total_realized']:+,.0f}")

    if uvxy_price:
        st.caption(f"Marked at UVXY ${uvxy_price:.2f}")
    else:
        uvxy_override = st.number_input(
            "UVXY current price (for P&L calc)",
            min_value=0.01, value=20.00, step=0.01, format="%.2f",
            key=f"uvxy_override_{mode}"
        )
        uvxy_price = uvxy_override

    st.markdown("---")

    # ── Subtabs: Open / Closed / Add ──────────────────────────────────────────
    sub_open, sub_closed, sub_add = st.tabs(
        [f"🟢 Open ({len(open_evs)})", f"⚫ Closed ({len(closed_evs)})", "➕ Add New"]
    )

    # ═══════════════════════════════
    with sub_open:
        if not open_evs:
            st.info(f"No open {label.lower()} assignments.")
        else:
            # ── Bulk mark button ──────────────────────────────────────────────
            st.caption(f"All open positions marked at UVXY ${uvxy_price:.2f}")

            for ev in sorted(open_evs, key=lambda e: e.date, reverse=True):
                _render_open_card(ev, uvxy_price, mode)

    # ═══════════════════════════════
    with sub_closed:
        if not closed_evs:
            st.info(f"No closed {label.lower()} assignments yet.")
        else:
            total_realized = sum(e.realized_pnl for e in closed_evs)
            wins = sum(1 for e in closed_evs if e.realized_pnl > 0)
            wr   = wins / len(closed_evs) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Realized", f"${total_realized:+,.0f}")
            c2.metric("Trades", len(closed_evs))
            c3.metric("Win Rate", f"{wr:.1f}%")

            # Editable closed table
            _render_closed_table(closed_evs, mode)

    # ═══════════════════════════════
    with sub_add:
        _render_add_form(mode)


# ── Open position card ────────────────────────────────────────────────────────

def _render_open_card(ev: AssignmentEvent, uvxy_price: float, mode: str):
    upnl     = ev.unrealized_pnl(uvxy_price)
    upnl_pct = ev.unrealized_pnl_pct(uvxy_price)
    pnl_color = "#22c55e" if upnl >= 0 else "#ef4444"
    dir_color = "#ef4444" if ev.direction == "SHORT" else "#3b82f6"

    st.markdown(f"""
    <div style="background:#1e293b; border-radius:12px; padding:16px 20px;
         margin-bottom:12px; border:1px solid #334155;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:1.05rem; font-weight:700; color:#f1f5f9;">
                #{ev.assignment_id}
                &nbsp;
                <span style="background:{dir_color}22; color:{dir_color}; padding:2px 8px;
                    border-radius:4px; font-size:0.72rem; font-weight:700;">{ev.direction}</span>
            </span>
            <span style="color:#94a3b8; font-size:0.82rem;">{ev.strategy_context}</span>
        </div>
        <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:8px; font-size:0.83rem; color:#94a3b8;">
            <div><div style="color:#64748b; font-size:0.7rem">DATE</div>{ev.date}</div>
            <div><div style="color:#64748b; font-size:0.7rem">SHARES</div>
                <strong style="color:{dir_color}">{ev.shares:+,d}</strong></div>
            <div><div style="color:#64748b; font-size:0.7rem">ENTRY</div>${ev.entry_price:.3f}</div>
            <div><div style="color:#64748b; font-size:0.7rem">NOTIONAL</div>${ev.notional:,.0f}</div>
            <div><div style="color:#64748b; font-size:0.7rem">UNREALIZED P&L</div>
                <strong style="color:{pnl_color}">${upnl:+,.0f} ({upnl_pct:+.1f}%)</strong></div>
        </div>
        {f'<div style="margin-top:8px; font-size:0.75rem; color:#475569;">📝 {ev.notes}</div>' if ev.notes else ""}
    </div>
    """, unsafe_allow_html=True)

    col_edit, col_close, col_delete = st.columns(3)

    # ── Edit ──────────────────────────────────────────────────────────────────
    with col_edit:
        with st.popover("✏️ Edit", use_container_width=True):
            with st.form(f"edit_{ev.assignment_id}"):
                st.markdown(f"**Edit #{ev.assignment_id}**")
                f_date    = st.date_input("Date",          value=date.fromisoformat(ev.date) if ev.date else date.today(), key=f"ed_{ev.assignment_id}")
                f_shares  = st.number_input("Shares (neg=short)", value=ev.shares,       step=100,  key=f"es_{ev.assignment_id}")
                f_price   = st.number_input("Entry Price",  value=ev.entry_price, step=0.001, format="%.3f", key=f"ep_{ev.assignment_id}")
                f_context = st.selectbox("Strategy Context",
                    options=STRATEGY_CHOICES,
                    index=_strategy_index(ev.strategy_context),
                    key=f"ec_{ev.assignment_id}")
                f_notes   = st.text_area("Notes", value=ev.notes, height=68, key=f"en_{ev.assignment_id}")

                if st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True):
                    update_event(mode, ev.assignment_id,
                        date=f_date.isoformat(),
                        shares=int(f_shares),
                        entry_price=float(f_price),
                        strategy_context=f_context,
                        notes=f_notes,
                    )
                    st.success("Saved."); st.rerun()

    # ── Close ─────────────────────────────────────────────────────────────────
    with col_close:
        with st.popover("🔒 Close Position", use_container_width=True):
            with st.form(f"close_{ev.assignment_id}"):
                st.markdown(f"**Close #{ev.assignment_id}**")
                f_exit_price = st.number_input("Exit Price",
                    value=uvxy_price if uvxy_price else ev.entry_price,
                    step=0.001, format="%.3f", key=f"cp_{ev.assignment_id}")
                f_exit_date  = st.date_input("Exit Date", value=date.today(), key=f"cd_{ev.assignment_id}")
                est_pnl = int(f_shares) * (ev.entry_price - f_exit_price) if 'f_shares' in dir() else ev.shares * (ev.entry_price - f_exit_price)
                st.info(f"Estimated P&L: **${ev.shares * (ev.entry_price - f_exit_price):+,.0f}**")

                if st.form_submit_button("Confirm Close", type="primary", use_container_width=True):
                    closed = close_event(mode, ev.assignment_id, f_exit_price, f_exit_date.isoformat())
                    if closed:
                        st.success(f"Closed. Realized: ${closed.realized_pnl:+,.0f}")
                        st.rerun()

    # ── Delete ────────────────────────────────────────────────────────────────
    with col_delete:
        with st.popover("🗑️ Delete", use_container_width=True):
            st.warning(f"Permanently delete record **#{ev.assignment_id}**?")
            st.caption("This cannot be undone.")
            col_y, col_n = st.columns(2)
            with col_y:
                if st.button("Yes, Delete", type="primary", key=f"del_yes_{ev.assignment_id}", use_container_width=True):
                    delete_event(mode, ev.assignment_id)
                    st.rerun()
            with col_n:
                if st.button("Cancel", key=f"del_no_{ev.assignment_id}", use_container_width=True):
                    st.rerun()


# ── Closed table with inline edit ─────────────────────────────────────────────

def _render_closed_table(closed_evs: List[AssignmentEvent], mode: str):
    st.markdown("**Closed Assignments**")

    # Main read-only table
    rows = []
    for e in sorted(closed_evs, key=lambda x: x.exit_date or x.date, reverse=True):
        rows.append({
            "ID":       e.assignment_id,
            "Strategy": e.strategy_context,
            "Date":     e.date,
            "Exit":     e.exit_date,
            "Held(d)":  e.days_held,
            "Shares":   e.shares,
            "Entry $":  f"${e.entry_price:.3f}",
            "Exit $":   f"${e.exit_price:.3f}" if e.exit_price else "—",
            "Realized": f"${e.realized_pnl:+,.0f}",
            "Notes":    e.notes,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-row edit/delete
    st.markdown("**Edit or delete a closed record:**")
    sel_id = st.selectbox(
        "Select record",
        options=[e.assignment_id for e in closed_evs],
        format_func=lambda x: next(
            (f"#{x} · {e.strategy_context} · {e.date} · ${e.realized_pnl:+,.0f}"
             for e in closed_evs if e.assignment_id == x), x),
        key=f"sel_closed_{mode}",
    )
    if sel_id:
        ev = next((e for e in closed_evs if e.assignment_id == sel_id), None)
        if ev:
            c1, c2 = st.columns(2)
            with c1:
                with st.popover("✏️ Edit Closed Record", use_container_width=True):
                    with st.form(f"edit_closed_{ev.assignment_id}"):
                        f_ep  = st.number_input("Entry Price",  value=ev.entry_price, step=0.001, format="%.3f", key=f"cep_{ev.assignment_id}")
                        f_xp  = st.number_input("Exit Price",   value=ev.exit_price,  step=0.001, format="%.3f", key=f"cxp_{ev.assignment_id}")
                        f_xd  = st.date_input("Exit Date", value=date.fromisoformat(ev.exit_date) if ev.exit_date else date.today(), key=f"cxd_{ev.assignment_id}")
                        f_ctx = st.text_input("Strategy Context", value=ev.strategy_context, key=f"cctx_{ev.assignment_id}")
                        f_nt  = st.text_area("Notes", value=ev.notes, height=60, key=f"cnt_{ev.assignment_id}")
                        new_pnl = ev.shares * (float(f_ep) - float(f_xp))
                        st.info(f"Recalculated P&L: **${new_pnl:+,.0f}**")
                        if st.form_submit_button("Save", type="primary", use_container_width=True):
                            update_event(mode, ev.assignment_id,
                                entry_price=float(f_ep),
                                exit_price=float(f_xp),
                                exit_date=f_xd.isoformat(),
                                strategy_context=f_ctx,
                                notes=f_nt,
                                realized_pnl=new_pnl,
                            )
                            st.success("Updated."); st.rerun()
            with c2:
                with st.popover("🗑️ Delete Record", use_container_width=True):
                    st.warning(f"Delete **#{ev.assignment_id}**?")
                    if st.button("Confirm Delete", type="primary", key=f"del_closed_{ev.assignment_id}", use_container_width=True):
                        delete_event(mode, ev.assignment_id)
                        st.rerun()


# ── Add new form ──────────────────────────────────────────────────────────────

def _render_add_form(mode: str):
    label = "Real" if mode == "real" else "Paper"
    st.markdown(f"**Add new {label.lower()} assignment**")
    st.caption("Record a share position (short or long) alongside your diagonal spreads.")

    with st.form(f"add_{mode}"):
        c1, c2 = st.columns(2)
        with c1:
            f_date    = st.date_input("Entry Date",  value=date.today(), key=f"add_date_{mode}")
            f_shares  = st.number_input("Shares (negative = short)",
                value=-100, step=100, key=f"add_shares_{mode}")
            f_price   = st.number_input("Entry Price", value=20.00, step=0.001, format="%.3f", key=f"add_price_{mode}")
        with c2:
            f_context = st.selectbox("Strategy Context", STRATEGY_CHOICES, key=f"add_ctx_{mode}")
            f_status  = st.selectbox("Status", ["open", "closed"], key=f"add_status_{mode}")
            f_notes   = st.text_area("Notes", height=68, key=f"add_notes_{mode}")

        notional = abs(int(f_shares)) * float(f_price)
        direction = "SHORT" if int(f_shares) < 0 else "LONG"
        st.info(f"{direction} {abs(int(f_shares)):,} shares @ ${float(f_price):.3f} = **${notional:,.0f}** notional")

        if st.form_submit_button(f"✅ Add {label} Assignment", type="primary", use_container_width=True):
            ev = AssignmentEvent(
                date             = f_date.isoformat(),
                shares           = int(f_shares),
                entry_price      = float(f_price),
                status           = f_status,
                strategy_context = f_context,
                mode             = mode,
                notes            = f_notes,
            )
            save_event(ev)
            st.success(f"✅ Added #{ev.assignment_id} — {direction} {abs(ev.shares):,} shares of UVXY @ ${ev.entry_price:.3f}")
            st.rerun()


# ── Helper ────────────────────────────────────────────────────────────────────

def _strategy_index(ctx: str) -> int:
    for i, s in enumerate(STRATEGY_CHOICES):
        if s.lower() in ctx.lower() or ctx.lower() in s.lower():
            return i
    return len(STRATEGY_CHOICES) - 1  # "Manual / Other"
