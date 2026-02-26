from pathlib import Path

path = Path("app.py")
src = path.read_text()
changes = 0

# ── 1. Replace static roll history table with spreadsheet + Edit Rolls button
old_roll_hist = '''                    # ── Roll history ──
                    if pos.roll_history:
                        st.markdown("---")'''

new_roll_hist = '''                    # ── Roll history ──
                    if pos.roll_history:
                        st.markdown("---")
                        rh_col1, rh_col2, rh_col3 = st.columns([4, 1, 1])
                        with rh_col1:
                            st.markdown("**🔄 Roll History**")
                        with rh_col2:
                            if st.button("✏️ Edit Rolls", key=f"redit_rolls_{pid}"):
                                st.session_state[f"rediting_rolls_{pid}"] = not st.session_state.get(f"rediting_rolls_{pid}", False)
                        with rh_col3:
                            if st.button("🔄 Recalc", key=f"rrecalc_{pid}"):
                                try:
                                    pos.recalc_roll_totals()
                                    rtl._save()
                                    reset_real_trade_log_cache()
                                    st.success("✅ Recalculated")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"{e}")
                        # Spreadsheet view
                        import pandas as pd
                        roll_rows = []
                        for r in pos.roll_history:
                            roll_rows.append({
                                "Date":       getattr(r, "roll_date", ""),
                                "Old Strike": f"${getattr(r, 'old_strike', 0):.0f}",
                                "Old Exp":    getattr(r, "old_expiration", ""),
                                "Old Exit":   f"${getattr(r, 'old_exit_price', 0):.2f}",
                                "Old Fill":   f"${getattr(r, 'old_fill_price', getattr(r, 'old_exit_price', 0)):.2f}",
                                "New Strike": f"${getattr(r, 'new_strike', 0):.0f}",
                                "New Exp":    getattr(r, "new_expiration", ""),
                                "New Credit": f"${getattr(r, 'new_credit', 0):.2f}",
                                "New Fill":   f"${getattr(r, 'new_fill_price', getattr(r, 'new_credit', 0)):.2f}",
                                "Net Credit": f"${getattr(r, 'roll_credit', 0):.2f}",
                                "BB Slip":    f"${getattr(r,'old_fill_price',getattr(r,'old_exit_price',0)) - getattr(r,'old_exit_price',0):+.2f}",
                                "Cr Slip":    f"${getattr(r,'new_fill_price',getattr(r,'new_credit',0)) - getattr(r,'new_credit',0):+.2f}",
                                "UVXY":       f"${getattr(r, 'underlying_price', 0):.2f}",
                                "Reason":     getattr(r, "roll_reason", ""),
                                "Notes":      getattr(r, "notes", ""),
                            })
                        st.dataframe(pd.DataFrame(roll_rows),
                                     use_container_width=True, hide_index=True)
                        # Totals
                        total_net = sum(getattr(r, "roll_credit", 0) for r in pos.roll_history)
                        total_bb_slip = sum(
                            getattr(r,'old_fill_price',getattr(r,'old_exit_price',0)) - getattr(r,'old_exit_price',0)
                            for r in pos.roll_history)
                        total_cr_slip = sum(
                            getattr(r,'new_fill_price',getattr(r,'new_credit',0)) - getattr(r,'new_credit',0)
                            for r in pos.roll_history)
                        ts1, ts2, ts3, ts4 = st.columns(4)
                        ts1.metric("Total Net Credits", f"${total_net:,.2f}")
                        ts2.metric("Total BB Slippage", f"${total_bb_slip:+.2f}")
                        ts3.metric("Total Cr Slippage", f"${total_cr_slip:+.2f}")
                        ts4.metric("Total Rolls", len(pos.roll_history))

                        # Edit mode
                        if st.session_state.get(f"rediting_rolls_{pid}"):
                            _render_real_roll_edit_form(rtl, pos)
                    if pos.roll_history:
                        pass  # already handled above
                    if False:  # placeholder to avoid empty block
                        st.markdown("---")'''

if old_roll_hist in src:
    src = src.replace(old_roll_hist, new_roll_hist)
    changes += 1
    print("✓ Roll history section replaced with spreadsheet + Edit button")
else:
    print("✗ Roll history anchor not found")

# ── 2. Add _render_real_roll_edit_form function before render_real_trade_log_page
real_edit_func = '''
def _render_real_roll_edit_form(rtl, pos):
    """Spreadsheet editor for real trade roll history."""
    import pandas as pd
    st.markdown("##### ✏️ Edit Roll History (Spreadsheet Mode)")

    if not pos.roll_history:
        st.info("No roll history to edit.")
        return

    roll_data = []
    for r in pos.roll_history:
        roll_data.append({
            "roll_id":        getattr(r, "roll_id", ""),
            "roll_date":      getattr(r, "roll_date", ""),
            "old_strike":     float(getattr(r, "old_strike", 0)),
            "old_expiration": getattr(r, "old_expiration", ""),
            "old_exit_price": float(getattr(r, "old_exit_price", 0)),
            "old_fill_price": float(getattr(r, "old_fill_price", getattr(r, "old_exit_price", 0))),
            "new_strike":     float(getattr(r, "new_strike", 0)),
            "new_expiration": getattr(r, "new_expiration", ""),
            "new_credit":     float(getattr(r, "new_credit", 0)),
            "new_fill_price": float(getattr(r, "new_fill_price", getattr(r, "new_credit", 0))),
            "roll_credit":    float(getattr(r, "roll_credit", 0)),
            "underlying_price": float(getattr(r, "underlying_price", 0)),
            "roll_reason":    getattr(r, "roll_reason", ""),
            "notes":          getattr(r, "notes", ""),
        })

    df = pd.DataFrame(roll_data)

    col_cfg = {
        "roll_id":          st.column_config.TextColumn("Roll ID", disabled=True, width="small"),
        "roll_date":        st.column_config.TextColumn("Date", width="small"),
        "old_strike":       st.column_config.NumberColumn("Old K", format="$%.1f", width="small"),
        "old_expiration":   st.column_config.TextColumn("Old Exp", width="small"),
        "old_exit_price":   st.column_config.NumberColumn("BB Mid", format="$%.2f", width="small",
                             help="Buy-back mid price"),
        "old_fill_price":   st.column_config.NumberColumn("BB Fill", format="$%.2f", width="small",
                             help="Actual buy-back fill"),
        "new_strike":       st.column_config.NumberColumn("New K", format="$%.1f", width="small"),
        "new_expiration":   st.column_config.TextColumn("New Exp", width="small"),
        "new_credit":       st.column_config.NumberColumn("Cr Mid", format="$%.2f", width="small",
                             help="New credit mid price"),
        "new_fill_price":   st.column_config.NumberColumn("Cr Fill", format="$%.2f", width="small",
                             help="Actual fill for new short"),
        "roll_credit":      st.column_config.NumberColumn("Net", format="$%.2f", disabled=True,
                             width="small", help="new_fill - bb_fill (auto)"),
        "underlying_price": st.column_config.NumberColumn("UVXY", format="$%.2f", width="small"),
        "roll_reason":      st.column_config.SelectboxColumn("Reason", width="small",
                             options=["order_roll","delta_trigger","itm_threat","manual","expired_worthless"]),
        "notes":            st.column_config.TextColumn("Notes", width="medium"),
    }

    edited_df = st.data_editor(df, column_config=col_cfg,
                               use_container_width=True, hide_index=True,
                               num_rows="fixed",
                               key=f"rroll_editor_{pos.position_id}")

    # Auto-compute roll_credit
    edited_df["roll_credit"] = (edited_df["new_fill_price"]
                                - edited_df["old_fill_price"])

    # Totals
    if len(edited_df) > 0:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rolls", len(edited_df))
        c2.metric("Net Credits", f"${edited_df['roll_credit'].sum():,.2f}")
        c3.metric("BB Slippage", f"${(edited_df['old_fill_price']-edited_df['old_exit_price']).sum():+.2f}")
        c4.metric("Cr Slippage", f"${(edited_df['new_fill_price']-edited_df['new_credit']).sum():+.2f}")

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("💾 Save Roll Edits",
                     key=f"rsave_rolls_{pos.position_id}", type="primary"):
            try:
                for i, row in edited_df.iterrows():
                    r = pos.roll_history[i]
                    r.roll_date        = str(row["roll_date"])
                    r.old_strike       = float(row["old_strike"])
                    r.old_expiration   = str(row["old_expiration"])
                    r.old_exit_price   = float(row["old_exit_price"])
                    r.old_fill_price   = float(row["old_fill_price"])
                    r.new_strike       = float(row["new_strike"])
                    r.new_expiration   = str(row["new_expiration"])
                    r.new_credit       = float(row["new_credit"])
                    r.new_fill_price   = float(row["new_fill_price"])
                    r.roll_credit      = float(row["roll_credit"])
                    r.underlying_price = float(row["underlying_price"])
                    r.roll_reason      = str(row["roll_reason"])
                    r.notes            = str(row["notes"])
                rtl._save()
                from real_trade_log import reset_real_trade_log_cache
                reset_real_trade_log_cache()
                st.session_state[f"rediting_rolls_{pos.position_id}"] = False
                st.success("✅ Roll history saved.")
                st.rerun()
            except Exception as e:
                import traceback
                st.error(f"Save failed: {e}")
                st.code(traceback.format_exc())
    with sc2:
        if st.button("❌ Cancel", key=f"rcancel_rolls_{pos.position_id}"):
            st.session_state[f"rediting_rolls_{pos.position_id}"] = False
            st.rerun()

    # Delete individual roll
    st.markdown("---")
    st.markdown("**Delete a Roll Record**")
    roll_opts = [f"{r.roll_id} ({getattr(r,'roll_date','')})"
                 for r in pos.roll_history]
    sel = st.selectbox("Select roll to delete", roll_opts,
                       key=f"rdel_roll_sel_{pos.position_id}")
    dc1, dc2 = st.columns([1, 3])
    with dc1:
        del_confirm = st.checkbox("Confirm", key=f"rdel_roll_confirm_{pos.position_id}")
    with dc2:
        if st.button("🗑️ Delete Roll", key=f"rdel_roll_btn_{pos.position_id}",
                     disabled=not del_confirm):
            idx = roll_opts.index(sel)
            pos.roll_history.pop(idx)
            rtl._save()
            from real_trade_log import reset_real_trade_log_cache
            reset_real_trade_log_cache()
            st.session_state[f"rediting_rolls_{pos.position_id}"] = False
            st.success("✅ Roll deleted.")
            st.rerun()

'''

marker = "def render_real_trade_log_page():"
if "_render_real_roll_edit_form" not in src:
    src = src.replace(marker, real_edit_func + marker)
    changes += 1
    print("✓ Added _render_real_roll_edit_form function")
else:
    print("  _render_real_roll_edit_form already exists")

Path("app.py").write_text(src)
print(f"\n✅ {changes} changes applied")
