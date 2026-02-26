from pathlib import Path

path = Path("app.py")
src = path.read_text()

# Find the end of the roll form section in real trade cards and add buttons
old_card_end = '''                    # Roll history
                    if pos.roll_history:
                        st.markdown("---")'''

new_card_end = '''                    # ── Action buttons ─────────────────────────────
                    st.markdown("---")
                    has_active_short = short and short.is_open()

                    btn1, btn2, btn3, btn4, btn5, btn6 = st.columns(6)

                    with btn1:
                        if has_active_short:
                            _expire = st.button("🎉 Expire Worthless",
                                                key=f"rexpire_{pid}",
                                                help="Mark short expired at $0")
                        else:
                            _expire = False
                            st.caption("No active short")

                    with btn2:
                        _add_short = st.button("📈 Add Short Leg",
                                               key=f"radd_short_{pid}",
                                               help="Sell new short after expiry")

                    with btn3:
                        _refresh = st.button("💰 Refresh Prices",
                                             key=f"rpx_{pid}")

                    with btn4:
                        _edit = st.button("✏️ Edit Long",
                                          key=f"redit_{pid}",
                                          help="Update long leg price/strike")

                    with btn5:
                        _close_all = st.button("🚪 Close All",
                                               key=f"rclose_{pid}",
                                               help="Close entire position")

                    with btn6:
                        _delete = st.button("🗑️ Delete",
                                            key=f"rdel_{pid}",
                                            help="Permanently delete position")

                    # ── Expire worthless ──
                    if _expire:
                        try:
                            rtl.close_short_leg(pid, exit_price=0.0,
                                                exit_reason="expired_worthless")
                            reset_real_trade_log_cache()
                            st.success("✅ Short marked expired worthless.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                    # ── Add new short leg ──
                    if _add_short or st.session_state.get(f"rshowing_add_short_{pid}"):
                        st.session_state[f"rshowing_add_short_{pid}"] = True
                        st.markdown("##### 📈 Add New Short Leg")
                        from datetime import datetime, timedelta
                        today_dt = datetime.now()
                        days_fri = (4 - today_dt.weekday()) % 7 or 7
                        next_fri = (today_dt + timedelta(days=days_fri)).date()
                        with st.form(f"radd_short_form_{pid}"):
                            as1, as2, as3 = st.columns(3)
                            with as1:
                                as_strike = st.number_input("Strike", value=float(short.strike if short else pos.long_strike), step=1.0, key=f"ras_k_{pid}")
                            with as2:
                                as_exp = st.date_input("Expiry", value=next_fri, key=f"ras_exp_{pid}")
                            with as3:
                                as_mid  = st.number_input("Mid price", value=1.50, step=0.01, key=f"ras_mid_{pid}")
                                as_fill = st.number_input("Fill price", value=1.50, step=0.01, key=f"ras_fill_{pid}")
                                as_comm = st.number_input("Commission", value=0.65, step=0.01, key=f"ras_comm_{pid}")
                            as_sub = st.form_submit_button("✅ Add Short Leg", type="primary")
                        if as_sub:
                            try:
                                leg = rtl.add_short_leg(pid, as_strike, as_exp.isoformat(), as_fill)
                                if leg:
                                    leg.entry_credit = as_mid
                                    leg.commission = as_comm
                                    leg.slippage = round(as_fill - as_mid, 4)
                                    rtl._save()
                                    reset_real_trade_log_cache()
                                    st.session_state[f"rshowing_add_short_{pid}"] = False
                                    st.success(f"✅ Added short ${as_strike:.0f} exp {as_exp}")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # ── Refresh prices ──
                    if _refresh:
                        try:
                            update_diagonal_live_prices(rtl, symbol="UVXY")
                            reset_real_trade_log_cache()
                            st.success("✅ Prices refreshed.")
                            st.rerun()
                        except Exception as e:
                            st.warning(f"Price fetch: {e}")

                    # ── Edit long leg ──
                    if _edit or st.session_state.get(f"rshowing_edit_{pid}"):
                        st.session_state[f"rshowing_edit_{pid}"] = True
                        st.markdown("##### ✏️ Edit Long Leg")
                        with st.form(f"redit_form_{pid}"):
                            e1, e2 = st.columns(2)
                            with e1:
                                e_strike = st.number_input("Strike", value=float(pos.long_strike), step=0.5, key=f"re_k_{pid}")
                                e_exp    = st.date_input("Expiry", value=__import__('datetime').date.fromisoformat(pos.long_expiration), key=f"re_exp_{pid}")
                            with e2:
                                e_fill    = st.number_input("Fill price", value=float(pos.long_fill_price), step=0.05, key=f"re_fill_{pid}")
                                e_current = st.number_input("Current price", value=float(pos.long_current_price or 0), step=0.05, key=f"re_cur_{pid}")
                            e_sub = st.form_submit_button("💾 Save", type="primary")
                        if e_sub:
                            try:
                                pos.long_strike = e_strike
                                pos.long_expiration = e_exp.isoformat()
                                pos.long_fill_price = e_fill
                                pos.long_current_price = e_current
                                rtl._save()
                                reset_real_trade_log_cache()
                                st.session_state[f"rshowing_edit_{pid}"] = False
                                st.success("✅ Long leg updated.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # ── Close all ──
                    if _close_all or st.session_state.get(f"rshowing_close_{pid}"):
                        st.session_state[f"rshowing_close_{pid}"] = True
                        st.markdown("##### 🚪 Close Entire Position")
                        with st.form(f"rclose_form_{pid}"):
                            cl1, cl2 = st.columns(2)
                            with cl1:
                                cl_long_exit  = st.number_input("Long exit price", value=float(pos.long_current_price or 0), step=0.05, key=f"rcl_long_{pid}")
                            with cl2:
                                if short and short.is_open():
                                    cl_short_exit = st.number_input("Short buy-back price", value=0.05, step=0.01, key=f"rcl_short_{pid}")
                                cl_reason = st.selectbox("Reason", ["take_profit","stop_loss","manual","long_expiring"], key=f"rcl_reason_{pid}")
                            cl_sub = st.form_submit_button("🚪 Confirm Close", type="primary")
                        if cl_sub:
                            try:
                                if short and short.is_open():
                                    rtl.close_short_leg(pid, exit_price=cl_short_exit, exit_reason=cl_reason)
                                pos.status = "closed"
                                import datetime
                                pos.close_date = datetime.date.today().isoformat()
                                pos.close_reason = cl_reason
                                pos.long_current_price = cl_long_exit
                                rtl._save()
                                reset_real_trade_log_cache()
                                st.session_state[f"rshowing_close_{pid}"] = False
                                st.success(f"✅ Position closed. Final P&L: ${pos.total_pnl:+,.0f}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # ── Delete ──
                    if _delete:
                        st.session_state[f"rconfirm_del_{pid}"] = True
                    if st.session_state.get(f"rconfirm_del_{pid}"):
                        st.warning(f"⚠️ Delete **{pos.variant_name}** permanently?")
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            if st.button("✅ Yes, Delete", key=f"rdelconfirm_{pid}"):
                                del rtl.diagonal_positions[pid]
                                rtl._save()
                                reset_real_trade_log_cache()
                                st.session_state.pop(f"rconfirm_del_{pid}", None)
                                st.success("Deleted.")
                                st.rerun()
                        with dc2:
                            if st.button("❌ Cancel", key=f"rdelcancel_{pid}"):
                                st.session_state.pop(f"rconfirm_del_{pid}", None)
                                st.rerun()

                    # ── Roll history ──
                    if pos.roll_history:
                        st.markdown("---")'''

if old_card_end in src:
    src = src.replace(old_card_end, new_card_end)
    Path("app.py").write_text(src)
    print("✓ Action buttons added to real trade position cards")
else:
    print("✗ Anchor not found")
