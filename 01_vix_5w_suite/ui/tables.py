import streamlit as st
import pandas as pd
import numpy as np


def show_performance_metrics(perf: dict):
    st.subheader("Performance Metrics")
    c = st.columns(8)
    c[0].metric("Total Return (%)", f'{perf["total_return_pct"]:,.1f}%')
    c[1].metric("Total Return ($)", f'{perf["total_return_dollar"]:,.0f}')
    c[2].metric("Ending Equity ($)", f'{perf["ending_equity"]:,.0f}')
    c[3].metric("CAGR", f'{perf["cagr"]*100:,.1f}%')
    c[4].metric("Sharpe (ann.)", f'{perf["sharpe"]:,.2f}')
    c[5].metric("Max Drawdown", f'{perf["max_dd"]*100:,.1f}%')
    c[6].metric("Win Rate", f'{perf["win_rate"]*100:,.1f}%')
    c[7].metric("Trades", f'{perf["trades"]} (avg {perf["avg_trade_dur"]:.2f}w)')


def show_weekly_pnl_table(vix: pd.Series, bt_results: dict):
    st.subheader("Weekly P&L Breakdown")

    n = len(vix)

    # Safely convert everything to 1D numpy arrays of length n
    def as_1d(x, name):
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = arr.ravel()
        # pad / trim to match n
        if arr.size == 0:
            arr = np.zeros(n)
        if arr.size < n:
            # pad with last value (or 0 if scalar)
            last = arr[-1] if arr.size > 0 else 0.0
            pad = np.full(n - arr.size, last)
            arr = np.concatenate([arr, pad])
        elif arr.size > n:
            arr = arr[:n]
        return arr

    equity = as_1d(bt_results.get("equity", np.zeros(n)), "equity")
    weekly_returns = as_1d(bt_results.get("weekly_returns", np.zeros(n)), "weekly_returns")
    realized = as_1d(bt_results.get("realized_weekly", np.zeros(n)), "realized_weekly")
    unrealized = as_1d(bt_results.get("unrealized_weekly", np.zeros(n)), "unrealized_weekly")

    df = pd.DataFrame(index=vix.index)  # index first, then add columns

    df["VIX Close"] = np.asarray(vix.values).ravel()
    df["Strategy Weekly P&L (%)"] = weekly_returns * 100.0
    df["Realized PnL this week ($)"] = realized
    df["Unrealized PnL this week ($)"] = unrealized
    df["Equity ($)"] = equity

    st.dataframe(df.round(4), use_container_width=True)


def show_grid_scan_table(grid_df):
    """
    Safely display the grid scan results DataFrame.
    Called from app.py.
    """
    if grid_df is None:
        st.info("No grid scan results yet. Click 'Run / Update grid scan' to generate.")
        return

    if isinstance(grid_df, pd.DataFrame) and grid_df.empty:
        st.info("Grid scan ran but returned an empty table.")
        return

    st.markdown(
        "Sorted by **higher total return** (and other criteria if you add them later)."
    )
    st.dataframe(grid_df.round(4), use_container_width=True)