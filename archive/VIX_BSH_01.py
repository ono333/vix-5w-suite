import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Black-Scholes for call pricing (simplified)
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Fetch data (VIX and ES weekly)
@st.cache_data
def load_data(start, end):
    # Convert date objects to datetime
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.min.time())
    # Adjust end date to yesterday to avoid future data
    end_dt = min(end_dt, datetime.now() - timedelta(days=1))
    # Fetch data and ensure Series output
    vix_data = yf.download('^VIX', start=start_dt, end=end_dt)['Close']
    es_data = yf.download('ES=F', start=start_dt, end=end_dt)['Close']
    # Resample to weekly (Monday) and get last value
    vix = vix_data.resample('W-MON').last()
    es = es_data.resample('W-MON').last()
    # Convert to DataFrame with named columns
    data = pd.DataFrame({'VIX': vix, 'ES': es}).fillna({'ES': 0}).dropna(subset=['VIX'])
    if len(data) < 10:
        st.warning(f"Insufficient data ({len(data)} weeks). Extending start date...")
        while len(data) < 10 and start_dt.year > 2000:
            start_dt = start_dt - timedelta(days=365)
            vix_data = yf.download('^VIX', start=start_dt, end=end_dt)['Close']
            es_data = yf.download('ES=F', start=start_dt, end=end_dt)['Close']
            vix = vix_data.resample('W-MON').last()
            es = es_data.resample('W-MON').last()
            data = pd.DataFrame({'VIX': vix, 'ES': es}).fillna({'ES': 0}).dropna(subset=['VIX'])
    return data['VIX']  # Return VIX series for VIX Hedge, ES ignored for now

# VIX Hedge Simulation (Video 1: Costless Ladder + Doomsday)
def simulate_vix_hedge(series, allocation=1000, ladder_dte=120/365, doomsday_dte=365/365, target_spike=1.5, r=0.04, sigma=0.20):
    values = series.values
    n = len(values)
    portfolio = allocation
    cum_ret = np.zeros(n)
    for i in range(n):
        vix = values[i]
        T_ladder = ladder_dte
        K_short = vix * 1.1  # 10% OTM short
        K_long = vix * 1.3   # 30% OTM long
        ladder_premium = black_scholes_call(vix, K_long, T_ladder, r, sigma) - black_scholes_call(vix, K_short, T_ladder, r, sigma)
        if ladder_premium <= 0:
            ladder_premium = vix * 0.01  # Fallback
        # Doomsday: Deep OTM call
        K_doom = vix * 2.0
        doomsday_cost = black_scholes_call(vix, K_doom, doomsday_dte, r, sigma) * 0.25  # 25% allocation
        net_cost = max(0, doomsday_cost - ladder_premium)  # Costless aim
        pnl = -net_cost * (allocation / vix)  # Simplified hedge P&L
        if i > 4 and vix / np.mean(values[max(0, i-4):i]) >= target_spike:  # 4-week avg spike
            pnl *= target_spike  # Protection payout
        portfolio += pnl
        cum_ret[i] = (portfolio - allocation) / allocation
    total_ret = cum_ret[-1]
    cagr = (1 + total_ret) ** (52 / n) - 1 if n > 0 else 0
    sharpe = np.mean(np.diff(cum_ret, prepend=0)) * 52 / (np.std(np.diff(cum_ret, prepend=0)) * np.sqrt(52)) if np.std(cum_ret) > 0 else 0
    max_dd = np.min(np.cumsum(cum_ret) - np.maximum.accumulate(np.cumsum(cum_ret)))
    win_rate = np.sum(np.diff(cum_ret, prepend=0) > 0) / n
    return cum_ret, {'Total Return': f"{total_ret:.1%}", 'CAGR': f"{cagr:.1%}", 'Sharpe': f"{sharpe:.2f}", 'Max DD': f"{max_dd:.1%}", 'Win Rate': f"{win_rate:.1%}"}

# Tracker BSH Simulation (Video 3: ES Puts on Dips + Hedge)
def simulate_tracker_bsh(series, allocation=1000, short_dte=90/365, hedge_ratio=5/3, dip_threshold=0.005, r=0.04, sigma=0.20):
    values = series.values
    n = len(values)
    portfolio = allocation
    cum_ret = np.zeros(n)
    net_hedge = 0
    for i in range(1, n):
        es_change = (values[i] - values[i-1]) / values[i-1]
        es = values[i]
        if es_change <= -dip_threshold:  # Enter on 0.5% dip
            K_short = es * 0.98  # 2% OTM
            short_credit = 3 * black_scholes_call(es, K_short, short_dte, r, sigma) * 0.03  # ~3% credit
            K_long = es * 0.95  # 5% OTM
            hedge_cost = hedge_ratio * black_scholes_call(es, K_long, short_dte, r, sigma)
            net_cost = hedge_cost - short_credit
            pnl = -net_cost * (allocation / es)
            if es / values[i-1] <= 0.92:  # 8% drop trigger
                pnl *= 2  # Hedge payout
            net_hedge += pnl
        portfolio += net_hedge * 0.01  # Scaled exposure
        cum_ret[i] = (portfolio - allocation) / allocation
        net_hedge *= 0.95  # Decay
    cum_ret[0] = 0
    total_ret = cum_ret[-1]
    cagr = (1 + total_ret) ** (52 / n) - 1 if n > 0 else 0
    sharpe = np.mean(np.diff(cum_ret, prepend=0)) * 52 / (np.std(np.diff(cum_ret, prepend=0)) * np.sqrt(52)) if np.std(cum_ret) > 0 else 0
    max_dd = np.min(np.cumsum(cum_ret) - np.maximum.accumulate(np.cumsum(cum_ret)))
    win_rate = np.sum(np.diff(cum_ret, prepend=0) > 0) / n
    return cum_ret, {'Total Return': f"{total_ret:.1%}", 'CAGR': f"{cagr:.1%}", 'Sharpe': f"{sharpe:.2f}", 'Max DD': f"{max_dd:.1%}", 'Win Rate': f"{win_rate:.1%}"}

# Grid Scan Function
def grid_scan(sim_func, data, params_grid):
    results = []
    for param_comb in params_grid:
        cum_ret, metrics = sim_func(data, **param_comb)
        results.append({**param_comb, **metrics})
    df_results = pd.DataFrame(results)
    return df_results

# Streamlit App
st.title("BSH Strategies Backtest & Grid Scan App")
st.markdown("Backtest VIX Hedge (Video 1) or Tracker BSH (Video 3). Use grid scan to optimize params.")

# Sidebar
st.sidebar.header("Strategy Selection")
strategy = st.sidebar.selectbox("Choose Strategy", ["VIX Hedge (Video 1)", "Tracker BSH (Video 3)"])
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime(2015, 1, 1))
end_date = col2.date_input("End Date", datetime(2025, 11, 13))  # Adjusted to yesterday
run_grid = st.sidebar.checkbox("Run Grid Scan", value=False)

if run_grid:
    st.sidebar.markdown("**Grid Params**")
    alloc_values = st.sidebar.multiselect("Allocations ($)", [500, 1000, 2000], default=[1000])
    target_values = st.sidebar.multiselect("Target Gain/Dip %", [1.2, 1.5, 2.0] if strategy == "VIX Hedge (Video 1)" else [0.003, 0.005, 0.01], default=[1.5] if strategy == "VIX Hedge (Video 1)" else [0.005])
    params_grid = [{'allocation': a, 'target_spike' if strategy == "VIX Hedge (Video 1)" else 'dip_threshold': t} for a in alloc_values for t in target_values]

# Load data
if start_date >= end_date:
    st.error("Start date must be before end date.")
else:
    with st.spinner("Fetching data..."):
        data = load_data(start_date, end_date)
    if len(data) < 10:
        st.warning("Insufficient data.")
    else:
        st.success(f"Loaded {len(data)} weeks ({data.index[0].date()} to {data.index[-1].date()}).")

        sim_func = simulate_vix_hedge if strategy == "VIX Hedge (Video 1)" else simulate_tracker_bsh
        default_params = {'allocation': 1000, 'target_spike': 1.5} if strategy == "VIX Hedge (Video 1)" else {'allocation': 1000, 'dip_threshold': 0.005}

        if run_grid:
            st.subheader("Grid Scan Results")
            df_scan = grid_scan(sim_func, data, params_grid)
            st.dataframe(df_scan)
            pivot = df_scan.pivot(index='allocation', columns='target_spike' if strategy == "VIX Hedge (Video 1)" else 'dip_threshold', values='CAGR')
            fig, ax = plt.subplots()
            sns.heatmap(pivot.apply(lambda x: float(x.strip('%')) / 100), annot=True, cmap='RdYlGn', ax=ax)
            st.pyplot(fig)
        else:
            cum_ret, metrics = sim_func(data, **default_params)
            st.subheader("Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Return", metrics['Total Return'])
            col2.metric("CAGR", metrics['CAGR'])
            col3.metric("Sharpe", metrics['Sharpe'])
            col4.metric("Max DD", f"{abs(float(metrics['Max DD'].strip('%'))):.1f}%")
            col5.metric("Win Rate", metrics['Win Rate'])

            st.subheader("Cumulative Returns Chart")
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(data.index, cum_ret * 100, label=strategy, color='blue')
            ax1.set_ylabel("Return (%)", color='blue')
            ax1.grid(True)
            ax2 = ax1.twinx()
            ax2.plot(data.index, data.values, label="VIX/ES", color='orange', alpha=0.5)
            ax2.set_ylabel("Index Level", color='orange')
            plt.title(f"{strategy} Backtest")
            st.pyplot(fig)

            st.subheader("Weekly Data & P&L")
            weekly_pnl = np.diff(cum_ret, prepend=0) * 100
            df = pd.DataFrame({'Date': data.index, 'Index': data.values, 'P&L (%)': weekly_pnl})
            st.dataframe(df, use_container_width=True)

st.markdown("---")
st.caption("Simplified simulations based on video descriptions. For education; test live with caution.")