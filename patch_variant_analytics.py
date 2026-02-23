from pathlib import Path

path = Path("app.py")
src = path.read_text()

old_func = '''def render_variant_analytics(trade_log=None):
    """Variant Analytics - Paper trading learning metrics."""
    st.title("📊 Variant Analytics")
    
    if not PAPER_TRADING_AVAILABLE:
        st.error("Paper trading modules not available")
        return
    
    trade_log = trade_log or get_trade_log()

    
    st.markdown("""
    Track operational metrics to decide which variants survive the paper trading period.
    
    **Focus on:** Which variants create work? Which simplify decisions? Which break under stress?
    """)
    
    st.markdown("---")
    st.subheader("Metrics by Variant Role")
    
    # Group diagonal positions by variant (case-insensitive)
    positions_by_variant = {}
    for pos in trade_log.diagonal_positions.values():
        variant_id = pos.variant_id.lower()
        if variant_id not in positions_by_variant:
            positions_by_variant[variant_id] = []
        positions_by_variant[variant_id].append(pos)
    
    for role in VariantRole:
        positions = positions_by_variant.get(role.value.lower(), [])
        
        if not positions:
            continue
        
        with st.expander(f"{get_variant_display_name(role)} ({len(positions)} positions)"):
            open_count = sum(1 for p in positions if p.status == "open")
            closed_count = sum(1 for p in positions if p.status == "closed")
            
            total_pnl = sum(p.total_pnl for p in positions)
            total_rolls = sum(p.total_rolls for p in positions)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Open / Closed", f"{open_count} / {closed_count}")
            col2.metric("Total P&L", f"${total_pnl:,.0f}")
            col3.metric("Total Rolls", total_rolls)
            col4.metric("Avg Rolls/Pos", f"{total_rolls/len(positions):.1f}" if positions else "0")
    
    # Summary
    st.markdown("---")
    all_positions = list(trade_log.diagonal_positions.values())
    open_positions = [p for p in all_positions if p.status == "open"]
    total_pnl = sum(p.total_pnl for p in all_positions)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", len(all_positions))
    col2.metric("Net P&L", f"${total_pnl:,.0f}")
    col3.metric("Open Positions", len(open_positions))'''

new_func = '''def render_variant_analytics(trade_log=None):
    """Variant Analytics — full spreadsheet with long duration & P&L analysis."""
    st.title("📊 Variant Analytics")
    trade_log = trade_log or get_trade_log()

    import pandas as pd
    from datetime import date, datetime

    all_pos   = list(trade_log.diagonal_positions.values())
    open_pos  = [p for p in all_pos if p.status == "open"]
    closed_pos = [p for p in all_pos if p.status == "closed"]

    if not all_pos:
        st.info("No positions yet.")
        return

    today = date.today()

    # ═══════════════════════════════════════════════════════
    # SUMMARY BAR
    # ═══════════════════════════════════════════════════════
    total_pnl   = sum(p.total_pnl for p in all_pos)
    total_long  = sum(p.long_pnl  for p in all_pos)
    total_short = sum(p.net_short_credits for p in all_pos)
    total_comm  = sum(p.total_commissions for p in all_pos)
    total_rolls = sum(p.total_rolls for p in all_pos)
    win_rate    = (sum(1 for p in closed_pos if p.total_pnl > 0)
                   / len(closed_pos) * 100) if closed_pos else 0.0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Positions", len(all_pos))
    c2.metric("Open / Closed",   f"{len(open_pos)} / {len(closed_pos)}")
    c3.metric("Total P&L",       f"${total_pnl:+,.0f}")
    c4.metric("Short Credits",   f"${total_short:+,.0f}")
    c5.metric("Total Rolls",     total_rolls)
    c6.metric("Win Rate",        f"{win_rate:.0f}%")

    st.divider()

    # ═══════════════════════════════════════════════════════
    # TAB LAYOUT
    # ═══════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Position Scorecard",
        "📈 Long Duration Analysis",
        "💰 P&L Breakdown",
        "🔄 Roll Efficiency",
    ])

    # ── Build core analytics rows ────────────────────────
    rows = []
    for pos in sorted(all_pos, key=lambda p: p.variant_id):
        try:
            long_entry  = date.fromisoformat(pos.entry_date)
            long_expiry = date.fromisoformat(pos.long_expiration)
            long_total  = max(1, (long_expiry - long_entry).days)
            long_elapsed = (today - long_entry).days
            long_remaining = max(0, (long_expiry - today).days)
            long_pct_used = long_elapsed / long_total * 100
        except Exception:
            long_total = long_elapsed = long_remaining = 1
            long_pct_used = 0.0

        cpd = pos.net_short_credits / long_elapsed if long_elapsed > 0 else 0
        proj_total = cpd * long_total
        be_days = int(pos.long_cost / cpd) if cpd > 0 else 9999
        be_date = (long_entry + __import__("datetime").timedelta(days=be_days)
                   ).isoformat() if be_days < 9999 else "N/A"
        bb_drag = (pos.total_buybacks / pos.gross_short_credits * 100
                   if pos.gross_short_credits > 0 else 0)

        short = pos.current_short_leg
        rows.append({
            # Identity
            "_pos":          pos,
            "Variant":       pos.variant_name,
            "Status":        pos.status.upper(),
            "Entry":         pos.entry_date,
            "Regime":        getattr(pos, "entry_regime", ""),
            # Long leg
            "Long Strike":   f"${pos.long_strike:.0f}",
            "Long Exp":      pos.long_expiration,
            "Long DTE":      long_remaining,
            "Long % Used":   f"{long_pct_used:.0f}%",
            "Long Cost":     pos.long_cost,
            "Long P&L":      pos.long_pnl,
            "Long Fill":     f"${pos.long_fill_price:.2f}",
            # Short leg
            "Short Strike":  f"${short.strike:.0f}" if short else "—",
            "Short Exp":     short.expiration_date if short else "—",
            "Short DTE":     pos.days_to_expiry(),
            "Short Fill":    f"${short.fill_price:.2f}" if short else "—",
            # Credits
            "Gross Credits": pos.gross_short_credits,
            "Buy-backs":     pos.total_buybacks,
            "Net Credits":   pos.net_short_credits,
            "BB Drag%":      f"{bb_drag:.0f}%",
            "Coverage%":     pos.short_coverage_pct,
            # P&L
            "Long P&L $":    pos.long_pnl,
            "Short P&L $":   pos.net_short_credits,
            "Total P&L $":   pos.total_pnl,
            "Return%":       (pos.total_pnl / pos.long_cost * 100
                              if pos.long_cost > 0 else 0),
            # Efficiency
            "$/day":         cpd,
            "Proj Total":    proj_total,
            "BE Date":       be_date,
            "Days Open":     long_elapsed,
            "Contracts":     pos.contracts,
            "Rolls":         pos.total_rolls,
            "Commission":    pos.total_commissions,
        })

    # ═══════════════════════════════════════════════════════
    # TAB 1 — POSITION SCORECARD
    # ═══════════════════════════════════════════════════════
    with tab1:
        st.markdown("##### All Positions — Scorecard")
        display_cols = [
            "Variant","Status","Entry","Regime","Contracts",
            "Long Strike","Long Exp","Long DTE","Long % Used",
            "Short Strike","Short Exp","Short DTE",
            "Coverage%","Total P&L $","Return%","Rolls",
        ]
        df = pd.DataFrame(rows)[display_cols].copy()
        df["Total P&L $"] = df["Total P&L $"].apply(lambda v: f"${v:+,.0f}")
        df["Return%"]     = df["Return%"].apply(lambda v: f"{v:+.1f}%")
        df["Coverage%"]   = df["Coverage%"].apply(lambda v: f"{v:.0f}%")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Per-variant summary
        st.markdown("##### By Variant")
        var_rows = []
        by_var = {}
        for r in rows:
            v = r["Variant"]
            by_var.setdefault(v, []).append(r)
        for v, vrows in sorted(by_var.items()):
            var_rows.append({
                "Variant":       v,
                "Positions":     len(vrows),
                "Open":          sum(1 for r in vrows if r["Status"]=="OPEN"),
                "Total P&L":     f"${sum(r['Total P&L $'] for r in vrows):+,.0f}",
                "Long P&L":      f"${sum(r['Long P&L $'] for r in vrows):+,.0f}",
                "Short Credits": f"${sum(r['Net Credits'] for r in vrows):+,.0f}",
                "Avg $/day":     f"${sum(r['$/day'] for r in vrows)/len(vrows):.2f}",
                "Avg Coverage%": f"{sum(r['Coverage%'] for r in vrows)/len(vrows):.0f}%",
                "Total Rolls":   sum(r["Rolls"] for r in vrows),
                "Commission":    f"${sum(r['Commission'] for r in vrows):.2f}",
            })
        st.dataframe(pd.DataFrame(var_rows),
                     use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════
    # TAB 2 — LONG DURATION ANALYSIS
    # ═══════════════════════════════════════════════════════
    with tab2:
        st.markdown("##### Long Leg Duration vs Credit Recovery")
        dur_cols = [
            "Variant","Status","Long Fill","Long Exp",
            "Long DTE","Long % Used","Days Open",
            "Long Cost","Long P&L $","Net Credits",
            "Coverage%","$/day","Proj Total","BE Date",
        ]
        df2 = pd.DataFrame(rows)[dur_cols].copy()
        df2["Long Cost"]    = df2["Long Cost"].apply(lambda v: f"${v:,.0f}")
        df2["Long P&L $"]   = df2["Long P&L $"].apply(lambda v: f"${v:+,.0f}")
        df2["Net Credits"]  = df2["Net Credits"].apply(lambda v: f"${v:,.0f}")
        df2["Coverage%"]    = df2["Coverage%"].apply(lambda v: f"{v:.0f}%")
        df2["$/day"]        = df2["$/day"].apply(lambda v: f"${v:.2f}")
        df2["Proj Total"]   = df2["Proj Total"].apply(lambda v: f"${v:,.0f}")
        st.dataframe(df2, use_container_width=True, hide_index=True)

        # Duration buckets
        st.markdown("##### P&L by Long Expiry Duration (weeks remaining)")
        buckets = {"0-4w": [], "4-8w": [], "8-13w": [], "13-26w": [], "26w+": []}
        for r in rows:
            d = r["Long DTE"]
            if   d <= 28:  buckets["0-4w"].append(r)
            elif d <= 56:  buckets["4-8w"].append(r)
            elif d <= 91:  buckets["8-13w"].append(r)
            elif d <= 182: buckets["13-26w"].append(r)
            else:          buckets["26w+"].append(r)
        brows = []
        for bucket, blist in buckets.items():
            if not blist: continue
            brows.append({
                "Long DTE Bucket": bucket,
                "Positions":       len(blist),
                "Avg Long P&L":    f"${sum(r['Long P&L $'] for r in blist)/len(blist):+,.0f}",
                "Avg Net Credits": f"${sum(r['Net Credits'] for r in blist)/len(blist):,.0f}",
                "Avg Total P&L":   f"${sum(r['Total P&L $'] for r in blist)/len(blist):+,.0f}",
                "Avg Coverage%":   f"{sum(r['Coverage%'] for r in blist)/len(blist):.0f}%",
                "Avg $/day":       f"${sum(r['$/day'] for r in blist)/len(blist):.2f}",
            })
        if brows:
            st.dataframe(pd.DataFrame(brows),
                         use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════
    # TAB 3 — P&L BREAKDOWN
    # ═══════════════════════════════════════════════════════
    with tab3:
        st.markdown("##### Full P&L Breakdown per Position")
        pnl_cols = [
            "Variant","Status","Contracts",
            "Long Cost","Long P&L $",
            "Gross Credits","Buy-backs","BB Drag%","Net Credits",
            "Commission","Total P&L $","Return%","Coverage%",
        ]
        df3 = pd.DataFrame(rows)[pnl_cols].copy()
        for col in ["Long Cost","Long P&L $","Gross Credits",
                    "Buy-backs","Net Credits","Commission","Total P&L $"]:
            df3[col] = df3[col].apply(
                lambda v: f"${v:+,.0f}" if isinstance(v, (int,float)) else v)
        df3["Return%"]  = df3["Return%"].apply(lambda v: f"{v:+.1f}%")
        df3["Coverage%"]= df3["Coverage%"].apply(lambda v: f"{v:.0f}%")
        st.dataframe(df3, use_container_width=True, hide_index=True)

        # Totals row
        st.markdown("##### Totals")
        t1,t2,t3,t4,t5,t6 = st.columns(6)
        t1.metric("Long Cost",     f"${sum(r['Long Cost'] for r in rows):,.0f}")
        t2.metric("Long P&L",      f"${sum(r['Long P&L $'] for r in rows):+,.0f}")
        t3.metric("Gross Credits", f"${sum(r['Gross Credits'] for r in rows):,.0f}")
        t4.metric("Net Credits",   f"${sum(r['Net Credits'] for r in rows):,.0f}")
        t5.metric("Commission",    f"${sum(r['Commission'] for r in rows):.2f}")
        t6.metric("Total P&L",     f"${sum(r['Total P&L $'] for r in rows):+,.0f}")

    # ═══════════════════════════════════════════════════════
    # TAB 4 — ROLL EFFICIENCY
    # ═══════════════════════════════════════════════════════
    with tab4:
        st.markdown("##### Roll Efficiency by Position")
        roll_cols = [
            "Variant","Status","Days Open","Rolls",
            "Gross Credits","Buy-backs","BB Drag%","Net Credits",
            "$/day","Proj Total","BE Date","Commission",
        ]
        df4 = pd.DataFrame(rows)[roll_cols].copy()
        for col in ["Gross Credits","Buy-backs","Net Credits","Proj Total","Commission"]:
            df4[col] = df4[col].apply(
                lambda v: f"${v:,.0f}" if isinstance(v,(int,float)) else v)
        df4["$/day"] = df4["$/day"].apply(lambda v: f"${v:.2f}")
        st.dataframe(df4, use_container_width=True, hide_index=True)

        # Roll detail — all rolls across all positions
        all_rolls = []
        for pos in all_pos:
            for r in pos.roll_history:
                all_rolls.append({
                    "Date":       r.roll_date,
                    "Variant":    pos.variant_name,
                    "Old Strike": f"${r.old_strike:.0f}",
                    "Old Exit":   f"${r.old_exit_price:.2f}",
                    "New Strike": f"${r.new_strike:.0f}",
                    "New Credit": f"${r.new_credit:.2f}",
                    "Net Credit": f"${r.roll_credit:.2f}",
                    "Underlying": f"${r.underlying_price:.2f}",
                    "Regime":     getattr(r, "regime", ""),
                })
        if all_rolls:
            st.markdown("##### All Roll History")
            st.dataframe(pd.DataFrame(all_rolls),
                         use_container_width=True, hide_index=True)'''

if old_func in src:
    src = src.replace(old_func, new_func)
    path.write_text(src)
    print("✓ Variant Analytics redesigned — 4 tabs, full spreadsheet")
else:
    print("✗ Function not found")
