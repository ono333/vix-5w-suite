import os
import zipfile
from pathlib import Path

ROOT = Path("01_vix_5w_suite")   # main project folder name

# --- Files to create: {relative_path: file_contents} ---
FILES = {
    "app.py": """\
import streamlit as st
from ui.styles import setup_page_config, inject_global_css
from ui.sidebar import build_sidebar
from ui.charts import plot_equity_and_vix, plot_vix_percentile, plot_trade_explorer_chart
from ui.tables import show_performance_metrics, show_weekly_pnl_table, show_grid_scan_table
from core.data_loader import load_vix_weekly
from core.metrics import summarize_performance
from core.backtester import run_backtest
from experiments.grid_scan import run_grid_scan
from core.trade_explorer_engine import run_trade_explorer

def main():
    setup_page_config()
    inject_global_css()

    st.title("VIX 5% Weekly Strategy Suite")

    sidebar_state = build_sidebar()
    page = sidebar_state["page"]
    params = sidebar_state["params"]

    vix = load_vix_weekly(params["start_date"], params["end_date"])
    if vix.empty:
        st.error("No VIX data for selected period.")
        return

    if page == "Backtester":
        st.subheader("Long-Term Backtest")
        bt_results = run_backtest(vix, params)
        perf = summarize_performance(bt_results, params)
        show_performance_metrics(perf)
        plot_equity_and_vix(vix, bt_results)
        plot_vix_percentile(vix)
        show_weekly_pnl_table(vix, bt_results)

        st.markdown("---")
        st.subheader("Grid Scan")
        if st.button("Run / Update grid scan"):
            grid_df = run_grid_scan(vix, params)
            st.session_state["grid_df"] = grid_df
        grid_df = st.session_state.get("grid_df")
        show_grid_scan_table(grid_df)

    elif page == "Trade Explorer":
        st.subheader("Trade Explorer")
        explorer = run_trade_explorer(vix, params)
        plot_trade_explorer_chart(vix, explorer)
        st.dataframe(explorer["long_trades"], use_container_width=True)
        st.dataframe(explorer["short_legs"], use_container_width=True)

if __name__ == "__main__":
    main()
""",

    "README.md": "# 01_vix_5w_suite\n\nFolder-structured version of your VIX 5% Weekly diagonal app.\n",
    "requirements.txt": "streamlit\nyfinance\npandas\nnumpy\nmatplotlib\nxlsxwriter\n",

    # --- config ---
    "config/__init__.py": "from .defaults import DEFAULTS\n",
    "config/defaults.py": """\
import datetime as dt

DEFAULTS = {
    "initial_capital": 250_000.0,
    "alloc_pct": 0.01,
    "long_dte_weeks": 26,
    "risk_free": 0.03,
    "fee_per_contract": 0.65,
    "entry_percentile": 0.10,
    "otm_points": 10.0,
    "target_multiple": 1.2,
    "sigma_mult": 0.5,
    "start_date": dt.date(2015, 1, 1),
    "end_date": dt.date.today(),
    "entry_mode": "percentile",   # or 'osc_roc'
    "position_structure": "diagonal",  # or 'long_only'
}
""",

    # --- core modules (stubs; you’ll paste real logic later) ---
    "core/__init__.py": "",
    "core/data_loader.py": """\
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

@st.cache_data
def load_vix_weekly(start, end):
    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d")
    data = yf.download("^VIX", start=start, end=end, interval="1wk", auto_adjust=False)
    if data.empty:
        return pd.Series(dtype=float)
    vix = data["Close"].copy()
    vix.index = pd.to_datetime(vix.index)
    vix.name = "VIX"
    return vix
""",

    "core/math_engine.py": """\
import numpy as np
from math import erf, sqrt

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
""",

    "core/indicators.py": """\
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd_3_10(series: pd.Series) -> pd.Series:
    fast = ema(series, 3)
    slow = ema(series, 10)
    return fast - slow

def roc(series: pd.Series, period: int = 2) -> pd.Series:
    return (series / series.shift(period) - 1.0) * 100.0

def rolling_percentile(series: pd.Series, window: int = 52) -> pd.Series:
    r = series.rolling(window=window, min_periods=4)
    return r.rank(pct=True)
""",

    "core/metrics.py": """\
import numpy as np

def compute_cagr(equity: np.ndarray, years: float) -> float:
    if len(equity) < 2 or equity[0] <= 0 or years <= 0:
        return 0.0
    return (equity[-1] / equity[0]) ** (1.0 / years) - 1.0

def compute_sharpe(weekly_returns: np.ndarray) -> float:
    if len(weekly_returns) < 2:
        return 0.0
    mean = np.mean(weekly_returns)
    std = np.std(weekly_returns, ddof=1)
    if std <= 0:
        return 0.0
    return (mean * 52.0) / (std * np.sqrt(52.0))

def compute_max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    cum_max = np.maximum.accumulate(equity)
    dd = (equity - cum_max) / cum_max
    return float(np.min(dd))

def summarize_performance(bt_results: dict, params: dict) -> dict:
    equity = bt_results["equity"]
    weekly_returns = bt_results["weekly_returns"]
    weeks = len(equity)
    years = weeks / 52.0 if weeks > 0 else 0.0
    total_return = (equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0
    from .metrics import compute_cagr, compute_sharpe, compute_max_drawdown  # self-import guard
    cagr = compute_cagr(equity, years)
    sharpe = compute_sharpe(weekly_returns[:-1])
    max_dd = compute_max_drawdown(equity)
    return {
        "total_return_pct": total_return * 100.0,
        "total_return_dollar": equity[-1] - equity[0],
        "ending_equity": float(equity[-1]),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": bt_results.get("win_rate", 0.0),
        "trades": bt_results.get("trades", 0),
        "avg_trade_dur": bt_results.get("avg_trade_dur", 0.0),
    }
""",

    "core/backtester.py": """\
import numpy as np
import pandas as pd
from core.indicators import macd_3_10, roc, rolling_percentile
from experiments.entry_rules import entry_signal_percentile, entry_signal_osc_roc
from experiments.exit_rules import apply_exit_rules

def run_backtest(vix: pd.Series, params: dict) -> dict:
    prices = vix.values.astype(float)
    n = len(prices)
    if n == 0:
        return {
            "equity": np.array([params["initial_capital"]]),
            "weekly_returns": np.zeros(1),
            "realized_weekly": np.zeros(1),
            "unrealized_weekly": np.zeros(1),
            "win_rate": 0.0,
            "trades": 0,
            "avg_trade_dur": 0.0,
        }

    macd = macd_3_10(vix)
    r2 = roc(vix, period=2)
    pct = rolling_percentile(vix, window=52)

    def entry_signal(i: int) -> bool:
        if params["entry_mode"] == "percentile":
            return entry_signal_percentile(i, prices, pct.values, params["entry_percentile"])
        else:
            return entry_signal_osc_roc(i, macd, r2)

    equity = np.zeros(n)
    weekly_returns = np.zeros(n)
    realized = np.zeros(n)
    unrealized = np.zeros(n)

    cash = params["initial_capital"]
    position = 0.0
    entry_price = 0.0
    trade_durations = []
    open_index = None
    wins = 0
    trades = 0

    equity[0] = cash

    for i in range(n - 1):
        price = prices[i]
        next_price = prices[i + 1]

        if position == 0.0 and entry_signal(i):
            position = 1.0
            entry_price = price
            open_index = i
            trades += 1

        if position != 0.0:
            exit_now, win_trade = apply_exit_rules(i, price, next_price, entry_price, params)
            if exit_now:
                pnl = (next_price - entry_price) * position
                cash += pnl
                realized[i + 1] += pnl
                position = 0.0
                if win_trade:
                    wins += 1
                if open_index is not None:
                    trade_durations.append((i + 1) - open_index)
                    open_index = None

        pos_value = position * prices[i]
        equity[i] = cash + pos_value
        if i > 0 and equity[i - 1] > 0:
            weekly_returns[i - 1] = (equity[i] - equity[i - 1]) / equity[i - 1]

    equity[-1] = cash + position * prices[-1]
    if n > 1 and equity[-2] > 0:
        weekly_returns[-1] = (equity[-1] - equity[-2]) / equity[-2]

    if trades > 0:
        win_rate = wins / trades
        avg_dur = float(np.mean(trade_durations)) if trade_durations else 0.0
    else:
        win_rate = 0.0
        avg_dur = 0.0

    return {
        "equity": equity,
        "weekly_returns": weekly_returns,
        "realized_weekly": realized,
        "unrealized_weekly": unrealized,
        "win_rate": win_rate,
        "trades": trades,
        "avg_trade_dur": avg_dur,
    }
""",

    "core/trade_explorer_engine.py": """\
import pandas as pd

def run_trade_explorer(vix: pd.Series, params: dict) -> dict:
    df = pd.DataFrame({"date": vix.index, "vix": vix.values})
    long_trades = []
    for i in range(0, len(df), 10):
        if i + 4 < len(df):
            long_trades.append(
                {
                    "entry_date": df["date"].iloc[i],
                    "exit_date": df["date"].iloc[i + 4],
                    "pnl_dollars": 0.0,
                    "pnl_pct_equity": 0.0,
                    "duration_weeks": 4,
                    "reason_exit": "placeholder",
                }
            )
    long_trades_df = pd.DataFrame(long_trades)
    short_legs_df = pd.DataFrame(columns=["date", "underlying", "strike", "contracts", "pnl_leg", "reason"])

    markers = None
    if not long_trades_df.empty:
        markers = {
            "date": long_trades_df["entry_date"],
            "price": vix.loc[long_trades_df["entry_date"]].values,
            "exit_date": long_trades_df["exit_date"],
            "exit_price": vix.loc[long_trades_df["exit_date"]].values,
        }

    return {
        "long_trades": long_trades_df,
        "short_legs": short_legs_df,
        "long_markers": markers,
        "short_markers": None,
    }
""",

    # --- experiments ---
    "experiments/__init__.py": "",
    "experiments/entry_rules.py": """\
import numpy as np
import pandas as pd

def entry_signal_percentile(i: int, prices: np.ndarray, pct_array: np.ndarray, threshold: float) -> bool:
    if i < 5:
        return False
    pct_i = pct_array[i]
    if np.isnan(pct_i):
        return False
    return pct_i <= threshold

def entry_signal_osc_roc(i: int, macd: pd.Series, roc2: pd.Series) -> bool:
    if i < 2:
        return False
    m = macd.iloc[i]
    m1 = macd.iloc[i - 1]
    m2 = macd.iloc[i - 2]
    r2 = roc2.iloc[i]
    if any(pd.isna(x) for x in (m, m1, m2, r2)):
        return False
    return (m < 0.0) and (m > m1) and (m1 <= m2) and (r2 <= 0.0)
""",

    "experiments/exit_rules.py": """\
def apply_exit_rules(i: int, price: float, next_price: float, entry_price: float, params: dict):
    target_mult = params.get("target_multiple", 1.2)
    if next_price >= entry_price * target_mult:
        return True, True
    if next_price <= entry_price * 0.5:
        return True, False
    return False, False
""",

    "experiments/grid_scan.py": """\
import pandas as pd
from core.backtester import run_backtest

def run_grid_scan(vix, base_params: dict) -> pd.DataFrame:
    entry_pcts = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    sigma_mults = [0.3, 0.5, 1.0]
    rows = []
    for ep in entry_pcts:
        for sig in sigma_mults:
            params = base_params.copy()
            params["entry_mode"] = "percentile"
            params["entry_percentile"] = ep
            params["sigma_mult"] = sig
            res = run_backtest(vix, params)
            eq = res["equity"]
            total_return = (eq[-1] / eq[0] - 1.0) if eq[0] > 0 else 0.0
            rows.append(
                {
                    "entry_pct": ep,
                    "sigma_mult": sig,
                    "total_return": total_return,
                    "trades": res["trades"],
                }
            )
    df = pd.DataFrame(rows)
    df.sort_values(by=["total_return"], ascending=False, inplace=True)
    return df
""",

    "experiments/roll_logic.py": '"""Placeholder for roll vs no-roll logic."""\n',

    # --- ui ---
    "ui/__init__.py": "",
    "ui/styles.py": """\
import streamlit as st

def setup_page_config():
    st.set_page_config(
        page_title="VIX 5% Weekly Suite",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def inject_global_css():
    st.markdown(
        '''
        <style>
        .main { background-color: #111111; }
        .stMetric-label, .stMarkdown, .stTextInput label, .stSelectbox label {
            color: #dddddd !important;
        }
        </style>
        ''',
        unsafe_allow_html=True,
    )
""",

    "ui/sidebar.py": """\
import streamlit as st
import datetime as dt
from config import DEFAULTS

def build_sidebar():
    with st.sidebar:
        st.header("Backtest Settings")

        page = st.radio("Page", ["Backtester", "Trade Explorer"], index=0)

        start_date = st.date_input(
            "Start date", value=DEFAULTS["start_date"],
            min_value=dt.date(2004, 1, 1), max_value=dt.date.today()
        )
        end_date = st.date_input(
            "End date", value=DEFAULTS["end_date"],
            min_value=dt.date(2004, 1, 1), max_value=dt.date.today()
        )

        cap_str = st.text_input(
            "Initial capital ($)",
            value=f'{DEFAULTS["initial_capital"]:,.0f}',
            key="initial_capital_str",
        )
        try:
            initial_capital = float(cap_str.replace(",", ""))
        except ValueError:
            initial_capital = DEFAULTS["initial_capital"]

        alloc_pct_percent = st.number_input(
            "Fraction of equity allocated to long call (%)",
            min_value=0.1, max_value=100.0,
            value=DEFAULTS["alloc_pct"] * 100.0,
            step=0.1, format="%.1f",
        )
        alloc_pct = alloc_pct_percent / 100.0

        long_dte_weeks = st.slider(
            "Long call maturity (weeks)",
            min_value=4, max_value=104,
            value=DEFAULTS["long_dte_weeks"], step=2,
        )

        risk_free = st.slider(
            "Risk-free rate (annual)",
            min_value=0.0, max_value=0.10,
            value=DEFAULTS["risk_free"], step=0.005, format="%.3f",
        )

        fee_per_contract = st.number_input(
            "Fee per options contract ($)",
            min_value=0.0, max_value=10.0,
            value=DEFAULTS["fee_per_contract"], step=0.05, format="%.2f",
        )

        entry_mode_label = st.selectbox(
            "Entry rule",
            ["Percentile only", "3-10 Oscillator + 2-week ROC"],
            index=0,
        )
        entry_mode = "percentile" if "Percentile" in entry_mode_label else "osc_roc"

        entry_percentile = st.slider(
            "Entry percentile (static mode only)",
            min_value=0.0, max_value=0.30,
            value=DEFAULTS["entry_percentile"], step=0.01,
        )

        otm_points = st.slider(
            "Short call OTM distance (pts)",
            min_value=1.0, max_value=20.0,
            value=DEFAULTS["otm_points"], step=0.5,
        )

        target_multiple = st.slider(
            "Exit when long call multiplies by",
            min_value=1.05, max_value=3.0,
            value=DEFAULTS["target_multiple"], step=0.05,
        )

        sigma_mult = st.slider(
            "Volatility multiplier (sigma_mult)",
            min_value=0.10, max_value=3.0,
            value=DEFAULTS["sigma_mult"], step=0.10,
        )

        position_structure = st.radio(
            "Position structure",
            ["Diagonal: LEAP + weekly OTM calls", "Long-only: VIX call (no weekly shorts)"],
            index=0,
        )
        if "Long-only" in position_structure:
            position_structure = "long_only"
        else:
            position_structure = "diagonal"

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "alloc_pct": alloc_pct,
        "long_dte_weeks": long_dte_weeks,
        "risk_free": risk_free,
        "fee_per_contract": fee_per_contract,
        "entry_mode": entry_mode,
        "entry_percentile": entry_percentile,
        "otm_points": otm_points,
        "target_multiple": target_multiple,
        "sigma_mult": sigma_mult,
        "position_structure": position_structure,
    }
    return {"page": page, "params": params}
""",

    "ui/charts.py": """\
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
""",

    "ui/tables.py": """\
import streamlit as st
import pandas as pd

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
    df = pd.DataFrame(
        {
            "Date": vix.index,
            "VIX Close": vix.values,
            "Strategy Weekly P&L (%)": bt_results["weekly_returns"] * 100.0,
            "Realized PnL this week ($)": bt_results.get("realized_weekly", 0.0),
            "Unrealized PnL this week ($)": bt_results.get("unrealized_weekly", 0.0),
            "Equity ($)": bt_results["equity"],
        }
    ).set_index("Date")
    st.dataframe(df.round(4), use_container_width=True)

def show_grid_scan_table(grid_df: pd.DataFrame | None):
    if grid_df is None or grid_df.empty:
        st.info("No grid scan results yet.")
        return
    st.markdown("Sorted by higher total return.")
    st.dataframe(grid_df.round(4), use_container_width=True)
""",
}

# --- Create folders + files ---
for rel_path, content in FILES.items():
    file_path = ROOT / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

# create empty dirs for backtests & charts
for extra_dir in ["backtests", "charts"]:
    (ROOT / extra_dir).mkdir(parents=True, exist_ok=True)

# --- Zip it ---
zip_name = "01_vix_5w_suite.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in ROOT.rglob("*"):
        zf.write(path, path.relative_to("."))

print(f"Created {zip_name} in", Path(".").resolve())