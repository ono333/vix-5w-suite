#!/usr/bin/env python3
"""
fix_assignment_link.py
Links trade log positions to assignment log:
1. Adds position_id field to assignment records
2. Adds "Mark as Assigned" button to short leg management in trade log
3. Adds "Close Assignment" button that writes P&L back to position

Run from ~/vix_suite/
"""
import sys
sys.path.insert(0, ".")
from safe_patch import patch

# ── 1. Add position_id to the new assignment record ───────────────────────
patch("app.py",
    old='''                new_rec = {
                    "id":               str(uuid.uuid4()),
                    "date":             f_date.isoformat(),
                    "shares":           int(f_shares),
                    "entry_price":      float(f_price),
                    "status":           f_status,
                    "strategy_context": f_ctx,
                    "mode":             mode,
                    "notes":            f_notes,
                    "exit_price":       0.0,
                    "exit_date":        "",
                    "realized_pnl":     0.0,
                    "created_at":       datetime.now().isoformat(),
                }''',
    new='''                new_rec = {
                    "id":               str(uuid.uuid4()),
                    "date":             f_date.isoformat(),
                    "shares":           int(f_shares),
                    "entry_price":      float(f_price),
                    "status":           f_status,
                    "strategy_context": f_ctx,
                    "mode":             mode,
                    "notes":            f_notes,
                    "exit_price":       0.0,
                    "exit_date":        "",
                    "realized_pnl":     0.0,
                    "created_at":       datetime.now().isoformat(),
                    "position_id":      st.session_state.get("asgn_prefill_pid", ""),
                    "short_strike":     st.session_state.get("asgn_prefill_strike", 0.0),
                    "short_expiry":     st.session_state.get("asgn_prefill_expiry", ""),
                }''',
    description="Add position_id/strike/expiry to assignment record",
)

# ── 2. Add "Mark as Assigned" button to paper short leg actions ────────────
# Find the Roll Short button area in paper position management
patch("app.py",
    old='                if st.button(f"🔄 Roll Short", key=f"p_roll_{pos.position_id}"):',
    new='''                if st.button(f"🚨 Assigned", key=f"p_assign_{pos.position_id}",
                               help="Short expired ITM — mark as assigned"):
                    short = pos.current_short_leg
                    if short:
                        # Pre-fill assignment form
                        st.session_state["asgn_prefill_pid"]    = pos.position_id
                        st.session_state["asgn_prefill_strike"] = float(short.strike)
                        st.session_state["asgn_prefill_expiry"] = short.expiration_date
                        # Close short leg as assigned
                        short.status      = "assigned"
                        short.exit_date   = __import__("datetime").date.today().isoformat()
                        short.exit_price  = float(short.strike)
                        short.exit_reason = "assigned"
                        tl = get_trade_log() if 'get_trade_log' in dir() else None
                        if tl: tl._save()
                        st.warning(f"⚠️ {pos.variant_name} short ${short.strike}C marked as assigned. "
                                   f"Go to Assignment Log to record shares and close.")
                        st.rerun()
                if st.button(f"🔄 Roll Short", key=f"p_roll_{pos.position_id}"):''',
    description="Add Assigned button to paper short leg actions",
)

# ── 3. Add "Close Assignment" button with P&L writeback ───────────────────
# Find the close assignment section in render_assignment_log
patch("app.py",
    old='''            with st.expander("🗑️ Delete an assignment"):''',
    new='''            # Close assignment with P&L writeback to position
            open_asgns = [r for r in all_recs if r.get("status") == "open"]
            if open_asgns:
                st.markdown("#### 💰 Close Assignment (Buy Back Shares)")
                for asgn in open_asgns:
                    pid = asgn.get("position_id", "")
                    shares = abs(int(asgn.get("shares", 0)))
                    entry_p = float(asgn.get("entry_price", 0))
                    with st.expander(
                        f"{'['+pid[:12]+'] ' if pid else ''}"
                        f"{asgn.get('strategy_context','')} | "
                        f"{shares} shares @ ${entry_p:.3f} | "
                        f"{asgn.get('date','')}",
                        expanded=True,
                    ):
                        xc1, xc2 = st.columns(2)
                        exit_p = xc1.number_input(
                            "Buyback Price",
                            value=float(uvxy_price),
                            step=0.01,
                            key=f"asgn_exit_{asgn['id']}",
                        )
                        exit_dt = xc2.date_input(
                            "Close Date",
                            value=__import__("datetime").date.today(),
                            key=f"asgn_edt_{asgn['id']}",
                        )
                        direction = "SHORT" if int(asgn.get("shares", 0)) < 0 else "LONG"
                        if direction == "SHORT":
                            pnl = (entry_p - exit_p) * shares
                        else:
                            pnl = (exit_p - entry_p) * shares
                        pnl_color = "green" if pnl >= 0 else "red"
                        st.markdown(
                            f'Est. P&L: <span style="color:{pnl_color};font-weight:700;">'
                            f'${pnl:+,.2f}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption("ℹ️ Close immediately — UVXY decays daily, holding assignments is costly.")
                        if st.button(f"✅ Close Assignment", key=f"asgn_close_{asgn['id']}"):
                            asgn["status"]       = "closed"
                            asgn["exit_price"]   = float(exit_p)
                            asgn["exit_date"]    = exit_dt.isoformat()
                            asgn["realized_pnl"] = round(pnl, 2)
                            _save(ASGN_FILE, all_recs)
                            # Write P&L back to linked position if available
                            if pid:
                                try:
                                    from trade_log import get_trade_log as _gtl
                                    _tl = _gtl()
                                    _pos = _tl.diagonal_positions.get(pid)
                                    if _pos:
                                        _pos.notes = (_pos.notes or "") + (
                                            f" | Assignment closed {exit_dt}: "
                                            f"${pnl:+,.2f}"
                                        )
                                        _tl._save()
                                except Exception:
                                    pass
                            st.success(f"✅ Assignment closed | P&L ${pnl:+,.2f}")
                            st.rerun()

            with st.expander("🗑️ Delete an assignment"):''',
    description="Add Close Assignment button with P&L writeback",
)

print("\nAll patches applied. Restart Streamlit.")
