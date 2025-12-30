import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_and_vix(vix: pd.Series, bt_results: dict):
    equity = bt_results["equity"]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(vix.index, equity, label="Equity ($)")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(vix.index, vix.values, color="tab:orange", alpha=0.7, label="VIX")
    ax2.set_ylabel("VIX")
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)

def plot_vix_percentile(vix: pd.Series, lookback_weeks: int = 52):
    rolling = vix.rolling(window=lookback_weeks, min_periods=4)
    pct = rolling.rank(pct=True) * 100.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(vix.index, pct.values)
    ax.set_ylabel(f"VIX {lookback_weeks}w Percentile (%)")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)

def plot_trade_explorer_chart(vix: pd.Series, explorer_results: dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(vix.index, vix.values, color="lightcyan", label="VIX")
    ax.set_ylabel("VIX")
    ax.grid(alpha=0.2)
    longs = explorer_results.get("long_markers")
    shorts = explorer_results.get("short_markers")
    if longs is not None:
        ax.scatter(longs["date"], longs["price"], marker="^", color="lime", label="Long Entry")
        ax.scatter(longs["exit_date"], longs["exit_price"], marker="v", color="yellow", label="Long Exit")
    if shorts is not None:
        ax.scatter(shorts["date"], shorts["price"], marker="o", color="violet", label="Short Entry", alpha=0.7)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)
