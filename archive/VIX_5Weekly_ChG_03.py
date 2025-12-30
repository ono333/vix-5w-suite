import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from math import erf, sqrt
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
import io

# ---------- Core math: normal CDF & Black-Scholes call ----------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Simple Black-Scholes call pricer for VIX options."""
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


# ---------- Helper: CAGR ----------

def _calc_cagr(total_return: float, n_weeks: int) -> float:
    if n_weeks <= 0:
        return np.nan
    if total_return <= -0.9999:
        return np.nan
    return (1.0 + total_return) ** (52.0 / n_weeks) - 1.0


# ---------- Helper: parse comma-separated floats ----------

def parse_float_list(text: str, default_vals):
    """Parse a comma-separated list of floats, fallback to default_vals on error/empty."""
    try:
        parts = [p.strip() for p in text.split(",")]
        vals = [float(p) for p in parts if p != ""]
        return vals if vals else default_vals
    except Exception:
        return default_vals


# ---------- Core backtest ----------

def backtest_vix_5pct(
    vix_weekly: pd.Series,
    initial_capital: float = 250_000.0,
    alloc_pct: float = 0.01,
    long_dte_weeks: int = 26,
    entry_lookback_weeks: int = 52,
    entry_percentile: float = 0.15,
    otm_points: float = 5.0,
    target_multiple: float = 3.0,
    sigma_mult: float = 0.8,
    r: float = 0.03,
    fee_per_contract: float = 0.65,
) -> dict:
    """
    Cash + position version:
    - Equity = cash + marked-to-market long option value
    - Realized PnL tracked separately from unrealized.
    """
    prices = np.asarray(vix_weekly, dtype=float).reshape(-1)
    dates = vix_weekly.index
    n = len(prices)
    if n < 5:
        return {
            "equity": np.array([initial_capital]),
            "weekly_returns": np.array([0.0]),
            "total_return": 0.0,
            "cagr": np.nan,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "dates": dates,
            "weekly_pnl_pct": np.array([0.0]),
            "realized_pnl_cum": np.array([0.0]),
            "realized_pnl_weekly": np.array([0.0]),
            "unrealized_pnl_weekly": np.array([0.0]),
        }

    equity = np.zeros(n, dtype=float)
    weekly_ret = np.zeros(n, dtype=float)
    weekly_pnl_pct = np.zeros(n, dtype=float)

    realized_pnl_cum = np.zeros(n, dtype=float)
    realized_pnl_weekly = np.zeros(n, dtype=float)
    unrealized_pnl_weekly = np.zeros(n, dtype=float)

    cash = initial_capital
    equity[0] = initial_capital
    long_value = 0.0
    has_long = False
    long_ttm_weeks = 0
    long_entry_price = 0.0
    long_contracts = 0
    long_cost_basis = 0.0  # premium*100*contracts + opening fee

    for i in range(n - 1):
        S = prices[i]
        S_next = prices[i + 1]

        # Mark-to-market equity at start of week i
        equity[i] = cash + long_value

        realized_this_week = 0.0  # dollars

        # ----- ENTRY: open long if regime condition -----
        if not has_long:
            look_start = max(0, i - entry_lookback_weeks + 1)
            look_prices = prices[look_start : i + 1]

            if len(look_prices) >= 4:
                threshold = np.quantile(look_prices, entry_percentile)
                if S <= threshold and S > 0:
                    T_long = long_dte_weeks / 52.0
                    sigma_long = max((S / 100.0) * sigma_mult, 0.01)
                    price_long = black_scholes_call(S, S, T_long, r, sigma_long)

                    if price_long > 0:
                        equity_now = cash + long_value
                        max_risk = equity_now * alloc_pct

                        per_contract_cost = price_long * 100.0 + 2.0 * fee_per_contract
                        contracts = int(max_risk / per_contract_cost)

                        if contracts > 0:
                            has_long = True
                            long_ttm_weeks = long_dte_weeks
                            long_entry_price = price_long
                            long_contracts = contracts

                            open_fee_total = fee_per_contract * contracts
                            cash -= price_long * 100.0 * contracts
                            cash -= open_fee_total
                            long_value = price_long * 100.0 * contracts
                            long_cost_basis = long_value + open_fee_total

        # ----- If long is open, weekly short call overlay + MTM -----
        if has_long and long_contracts > 0:
            # Short 1-week OTM call
            T_short = 1.0 / 52.0
            K_short = S + otm_points
            sigma_short = max((S / 100.0) * sigma_mult, 0.01)

            price_short = black_scholes_call(S, K_short, T_short, r, sigma_short)
            if price_short > 0:
                short_premium = price_short * 100.0 * long_contracts
                open_short_fee = fee_per_contract * long_contracts

                # Open short: cash up, realized later after payoff
                cash += short_premium
                cash -= open_short_fee

                # Settlement next week at S_next
                payoff_short = max(S_next - K_short, 0.0) * 100.0 * long_contracts
                close_short_fee = fee_per_contract * long_contracts
                cash -= payoff_short
                cash -= close_short_fee

                realized_short = short_premium - payoff_short - open_short_fee - close_short_fee
                realized_this_week += realized_short

            # Re-price long at next week
            long_ttm_weeks = max(0, long_ttm_weeks - 1)
            T_new = long_ttm_weeks / 52.0
            sigma_long_new = max((S_next / 100.0) * sigma_mult, 0.01)
            price_long_new = black_scholes_call(S_next, S, T_new, r, sigma_long_new)
            long_value_new = price_long_new * 100.0 * long_contracts if T_new > 0 else 0.0

            # Check exit condition
            exit_now = False
            if T_new <= 0:
                exit_now = True
            elif price_long_new >= target_multiple * long_entry_price:
                exit_now = True

            if exit_now:
                close_long_fee = fee_per_contract * long_contracts
                cash += long_value_new
                cash -= close_long_fee

                realized_long = long_value_new - long_cost_basis - close_long_fee
                realized_this_week += realized_long

                long_value_new = 0.0
                long_cost_basis = 0.0
                has_long = False
                long_contracts = 0

            long_value = long_value_new

        # Equity & weekly return at end of week i+1
        equity[i + 1] = cash + long_value
        eq_change = equity[i + 1] - equity[i]

        if equity[i] > 0:
            weekly_ret[i + 1] = eq_change / equity[i]
            weekly_pnl_pct[i + 1] = weekly_ret[i + 1] * 100.0
        else:
            weekly_ret[i + 1] = 0.0
            weekly_pnl_pct[i + 1] = 0.0

        # Record realized / unrealized decomposition
        realized_pnl_weekly[i + 1] = realized_this_week
        realized_pnl_cum[i + 1] = realized_pnl_cum[i] + realized_this_week
        unrealized_pnl_weekly[i + 1] = eq_change - realized_this_week

        # Hard stop if blown up
        if equity[i + 1] <= 0:
            equity[i + 1] = 0.0
            weekly_ret[i + 1 :] = -1.0
            weekly_pnl_pct[i + 1 :] = -100.0
            break

    total_return = equity[-1] / initial_capital - 1.0
    n_weeks_valid = max(1, np.count_nonzero(equity)) - 1
    cagr = _calc_cagr(total_return, n_weeks_valid)

    valid_returns = weekly_ret[1 : n_weeks_valid + 1]
    if len(valid_returns) > 1 and np.std(valid_returns) > 1e-8:
        sharpe = (np.mean(valid_returns) * 52.0 - r) / (np.std(valid_returns) * np.sqrt(52.0))
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    win_rate = float(np.mean(valid_returns > 0.0)) if len(valid_returns) else 0.0

    return {
        "equity": equity,
        "weekly_returns": weekly_ret,
        "total_return": float(total_return),
        "cagr": float(cagr) if not np.isnan(cagr) else np.nan,
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "dates": dates,
        "weekly_pnl_pct": weekly_pnl_pct,
        "realized_pnl_cum": realized_pnl_cum,
        "realized_pnl_weekly": realized_pnl_weekly,
        "unrealized_pnl_weekly": unrealized_pnl_weekly,
    }


# ---------- Grid scan (rank by high CAGR + low Max DD) ----------

def run_grid_scan(
    vix_weekly: pd.Series,
    initial_capital: float,
    alloc_pct: float,
    long_dte_weeks: int,
    r: float,
    fee_per_contract: float,
    entry_grid,
    otm_grid,
    target_grid,
    sigma_grid,
):
    rows = []
    for ep in entry_grid:
        for otm in otm_grid:
            for tgt in target_grid:
                for sig in sigma_grid:
                    res = backtest_vix_5pct(
                        vix_weekly=vix_weekly,
                        initial_capital=initial_capital,
                        alloc_pct=alloc_pct,
                        long_dte_weeks=long_dte_weeks,
                        entry_lookback_weeks=52,
                        entry_percentile=ep,
                        otm_points=otm,
                        target_multiple=tgt,
                        sigma_mult=sig,
                        r=r,
                        fee_per_contract=fee_per_contract,
                    )
                    rows.append(
                        {
                            "entry_pct": ep,
                            "otm_pts": otm,
                            "target_mult": tgt,
                            "sigma_mult": sig,
                            "total_return": res["total_return"],
                            "cagr": res["cagr"],
                            "sharpe": res["sharpe"],
                            "max_dd": res["max_dd"],
                            "win_rate": res["win_rate"],
                        }
                    )

    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        return grid_df, None

    # Sort: higher CAGR first, and among those, smaller (less negative) max drawdown
    sorted_df = grid_df.sort_values(["cagr", "max_dd"], ascending=[False, True])
    best_row = sorted_df.iloc[0]
    return sorted_df, best_row


# ---------- Streamlit App ----------

st.set_page_config(page_title="VIX 5% Weekly Strategy Backtester", layout="wide")

st.title("VIX 5% Weekly Strategy: Long-Term Backtester")
st.write(
    "Simulate the VIX 5% Weekly strategy on historical VIX data. "
    "Long-dated ATM calls with weekly short OTM calls, including fixed per-contract fees."
)

# ----- Parameter/session-state wiring -----

param_keys = [
    "initial_capital",
    "alloc_pct",          # stored as fraction (0.20 = 20%)
    "long_dte_weeks",
    "risk_free",
    "fee_per_contract",
    "entry_percentile",
    "otm_points",
    "target_multiple",
    "sigma_mult",
]

param_defaults = {
    "initial_capital": 250_000.0,
    "alloc_pct": 0.01,
    "long_dte_weeks": 26,
    "risk_free": 0.03,
    "fee_per_contract": 0.65,
    "entry_percentile": 0.15,
    "otm_points": 5.0,
    "target_multiple": 3.0,
    "sigma_mult": 0.8,
}

for k in param_keys:
    if k not in st.session_state:
        st.session_state[k] = param_defaults[k]
    default_key = f"default_{k}"
    if default_key not in st.session_state:
        st.session_state[default_key] = st.session_state[k]

if "apply_from_grid" not in st.session_state:
    st.session_state.apply_from_grid = False
if "apply_defaults" not in st.session_state:
    st.session_state.apply_defaults = False
if "last_grid_idx" not in st.session_state:
    st.session_state.last_grid_idx = 0
if "grid_results_df" not in st.session_state:
    st.session_state.grid_results_df = None

# Apply parameters chosen from grid
if st.session_state.get("apply_from_grid", False) and "grid_choice_params" in st.session_state:
    for k, v in st.session_state["grid_choice_params"].items():
        if k in param_keys:
            st.session_state[k] = float(v)
    st.session_state.apply_from_grid = False

# Apply saved defaults
if st.session_state.get("apply_defaults", False):
    for k in param_keys:
        st.session_state[k] = st.session_state[f"default_{k}"]
    st.session_state.apply_defaults = False


# ----- Sidebar: date range -----

with st.sidebar:
    st.header("Backtest Settings")

    min_date = dt.date(2004, 1, 1)
    max_date = dt.date.today()

    start_date = st.date_input(
        "Start date",
        value=dt.date(2015, 1, 1),
        min_value=min_date,
        max_value=max_date,
    )

    end_date = st.date_input(
        "End date",
        value=max_date,
        min_value=start_date,
        max_value=max_date,
    )

# ----- Sidebar: capital, risk, parameters, grid ranges -----

with st.sidebar:
    st.caption("Strategy & parameters")

    # --- Initial Capital (remember last value) ---
    # Initialise backing numeric state once
    if "initial_capital" not in st.session_state:
        st.session_state["initial_capital"] = param_defaults["initial_capital"]

    # Text field with commas, backed by numeric session_state["initial_capital"]
    cap_str_default = f'{st.session_state["initial_capital"]:,.0f}'
    cap_str = st.text_input(
        "Initial capital ($)",
        value=cap_str_default,
        key="initial_capital_str",
    )
    try:
        initial_capital = float(cap_str.replace(",", "").strip())
    except ValueError:
        # If user types junk, fall back to last valid numeric
        initial_capital = st.session_state["initial_capital"]

    # Store numeric so it’s remembered on next rerun & used by backtest
    st.session_state["initial_capital"] = initial_capital

    # --- Fraction of equity (remember last value) ---
    # We store alloc_pct internally as a fraction (0.20 = 20%)
    if "alloc_pct" not in st.session_state:
        st.session_state["alloc_pct"] = param_defaults["alloc_pct"]

    alloc_pct_percent = st.number_input(
        "Fraction of equity allocated to long call (%)",
        min_value=0.1,
        max_value=50.0,
        value=float(st.session_state["alloc_pct"] * 100.0),
        step=0.1,
        format="%.1f",
        key="alloc_pct_percent",
    )
    alloc_pct = alloc_pct_percent / 100.0
    st.session_state["alloc_pct"] = alloc_pct

    # --- Other parameters ---

    if "long_dte_weeks" not in st.session_state:
        st.session_state["long_dte_weeks"] = param_defaults["long_dte_weeks"]
    long_dte_weeks = st.slider(
        "Long call maturity (weeks)",
        4,
        104,
        value=int(st.session_state["long_dte_weeks"]),
        step=2,
        key="long_dte_weeks",
    )

    if "risk_free" not in st.session_state:
        st.session_state["risk_free"] = param_defaults["risk_free"]
    risk_free = st.slider(
        "Risk-free rate (annual)",
        0.0,
        0.10,
        value=float(st.session_state["risk_free"]),
        step=0.005,
        key="risk_free",
    )

    if "fee_per_contract" not in st.session_state:
        st.session_state["fee_per_contract"] = param_defaults["fee_per_contract"]
    fee_per_contract = st.number_input(
        "Fee per options contract ($)",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state["fee_per_contract"]),
        step=0.05,
        format="%.2f",
        key="fee_per_contract",
    )

    if "entry_percentile" not in st.session_state:
        st.session_state["entry_percentile"] = param_defaults["entry_percentile"]
    entry_percentile = st.slider(
        "Entry percentile of VIX (low regime)",
        0.0,
        0.30,
        value=float(st.session_state["entry_percentile"]),
        step=0.01,
        key="entry_percentile",
    )

    if "otm_points" not in st.session_state:
        st.session_state["otm_points"] = param_defaults["otm_points"]
    otm_points = st.slider(
        "Short call OTM distance (pts)",
        1.0,
        10.0,
        value=float(st.session_state["otm_points"]),
        step=0.5,
        key="otm_points",
    )

    if "target_multiple" not in st.session_state:
        st.session_state["target_multiple"] = param_defaults["target_multiple"]
    target_multiple = st.slider(
        "Exit when long call multiplies by",
        1.2,
        5.0,
        value=float(st.session_state["target_multiple"]),
        step=0.1,
        key="target_multiple",
    )

    if "sigma_mult" not in st.session_state:
        st.session_state["sigma_mult"] = param_defaults["sigma_mult"]
    sigma_mult = st.slider(
        "Volatility multiplier on VIX/100",
        0.5,
        3.0,
        value=float(st.session_state["sigma_mult"]),
        step=0.1,
        key="sigma_mult",
    )

    # Save / reset defaults now also capture initial_capital & alloc_pct
    if st.button("Save current as default"):
        for k in param_keys:
            st.session_state[f"default_{k}"] = st.session_state[k]

    if st.button("Reset to saved default"):
        for k in param_keys:
            st.session_state[k] = st.session_state[f"default_{k}"]
        st.session_state.apply_defaults = False
        st.rerun()

    # ----- Grid scan entry/exit setup (ranges) -----
    grid_exp = st.expander("Grid scan: entry/exit parameter ranges", expanded=False)
    with grid_exp:
        entry_grid_str = st.text_input(
            "Entry percentiles (decimals, e.g. 0.05, 0.10, 0.15)",
            value=st.session_state.get("grid_entry_list", "0.05, 0.10, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9"),
            key="grid_entry_list",
        )
        otm_grid_str = st.text_input(
            "OTM distances (points, e.g. 2, 3, 5)",
            value=st.session_state.get("grid_otm_list", "5"),
            key="grid_otm_list",
        )
        target_grid_str = st.text_input(
            "Target multiples (e.g. 1.5, 2, 3)",
            value=st.session_state.get("grid_target_list", "1.1, 1.2, 1.25, 1.3, 1.4, 1.5"),
            key="grid_target_list",
        )
        sigma_grid_str = st.text_input(
            "Sigma multipliers (e.g. 0.8, 1.0, 1.2)",
            value=st.session_state.get("grid_sigma_list", "0.8"),
            key="grid_sigma_list",
        )


# Parse grid ranges
entry_grid_vals = parse_float_list(entry_grid_str, [0.05, 0.10, 0.15])
otm_grid_vals = parse_float_list(otm_grid_str, [2.0, 3.0, 5.0])
target_grid_vals = parse_float_list(target_grid_str, [1.5, 2.0, 3.0])
sigma_grid_vals = parse_float_list(sigma_grid_str, [0.8, 1.0, 1.2])


# ----- Load VIX data -----

@st.cache_data(show_spinner=False)
def load_vix_weekly(start: datetime, end: datetime) -> pd.Series:
    data = yf.download("^VIX", start=start, end=end)
    if data.empty:
        return pd.Series(dtype=float)
    weekly = data["Close"].resample("W-FRI").last().dropna()
    return weekly

vix_weekly = load_vix_weekly(start_date, end_date)

if vix_weekly.empty:
    st.error("No VIX data was downloaded. Check your date range or internet connection.")
    st.stop()

n_weeks = len(vix_weekly)
st.success(
    f"Loaded {n_weeks} weeks of VIX data "
    f"({vix_weekly.index[0].date()} to {vix_weekly.index[-1].date()})."
)

# ----- Run main backtest -----

results = backtest_vix_5pct(
    vix_weekly=vix_weekly,
    initial_capital=initial_capital,
    alloc_pct=alloc_pct,
    long_dte_weeks=long_dte_weeks,
    entry_lookback_weeks=52,
    entry_percentile=entry_percentile,
    otm_points=otm_points,
    target_multiple=target_multiple,
    sigma_mult=sigma_mult,
    r=risk_free,
    fee_per_contract=fee_per_contract,
)

equity = np.asarray(results["equity"], dtype=float).reshape(-1)
weekly_returns = np.asarray(results["weekly_returns"], dtype=float).reshape(-1)
total_return = results["total_return"]
cagr = results["cagr"]
sharpe = results["sharpe"]
max_dd = results["max_dd"]
win_rate = results["win_rate"]
dates = results["dates"]
weekly_pnl_pct = results["weekly_pnl_pct"]
realized_pnl_cum = results["realized_pnl_cum"]
realized_pnl_weekly = results["realized_pnl_weekly"]
unrealized_pnl_weekly = results["unrealized_pnl_weekly"]

dates_1d = np.asarray(dates)
vix_values_weekly = np.asarray(vix_weekly.reindex(dates_1d).squeeze(), dtype=float).ravel()
weekly_pnl_1d = np.asarray(weekly_pnl_pct, dtype=float).ravel()

ending_equity = float(equity[-1])
total_return_dollar = ending_equity - initial_capital


# ----- Collapsible Strategy Implementation -----

strategy_expander = st.expander("Strategy Implementation (with your current settings)", expanded=True)
with strategy_expander:
    st.markdown(
        f"""
1. **Data & timeframe**  
   - Weekly VIX data (Friday closes of `^VIX`) from **{start_date}** to **{end_date}**.  

2. **Regime filter**  
   - Each week, look back **52 weeks** of VIX closes.  
   - If current VIX is at or below the **{entry_percentile*100:.1f}% percentile** of that window and no long is open, you can enter.

3. **Long VIX call entry**  
   - Buy an ATM VIX call (strike = current VIX).  
   - Maturity ≈ **{long_dte_weeks} weeks**.  
   - Risk per trade ≤ **{alloc_pct*100:.2f}%** of current equity.

4. **Weekly short call overlay**  
   - While long is open, each week sell a 1-week call at `VIX + {otm_points:.1f}`.  
   - Pay **${fee_per_contract:.2f} per contract** to open and close.

5. **Pricing model**  
   - Black–Scholes with `sigma = (VIX/100) × sigma_mult`.  
   - Current `sigma_mult = {sigma_mult:.2f}`.

6. **Realized vs unrealized PnL**  
   - **Realized**: closed shorts + closed long (after all fees).  
   - **Unrealized**: equity change minus realized for that week.

7. **Equity curve & metrics**  
   - Equity each week = cash + MTM long option.  
   - Metrics (Total Return, CAGR, Sharpe, Max DD, Win Rate) are based on this equity curve starting from **Initial Capital**.
"""
    )


# ----- Collapsible portfolio sizing & suggestion -----

portfolio_expander = st.expander("Portfolio sizing & option suggestion", expanded=False)
with portfolio_expander:
    try:
        spot = float(vix_weekly.iloc[-1])
        T_long = long_dte_weeks / 52.0
        sigma_long = max((spot / 100.0) * sigma_mult, 0.01)

        if spot > 0 and T_long > 0:
            d1 = (np.log(spot / spot) + (risk_free + 0.5 * sigma_long**2) * T_long) / (sigma_long * np.sqrt(T_long))
            delta_long = norm_cdf(d1)
            price_long = black_scholes_call(spot, spot, T_long, risk_free, sigma_long)
            per_contract_cost = price_long * 100.0 + 2.0 * fee_per_contract
            max_risk_dollars = initial_capital * alloc_pct
            contracts_suggested = int(max_risk_dollars // per_contract_cost) if per_contract_cost > 0 else 0

            st.markdown(
                f"""
- **Portfolio size:** ${initial_capital:,.0f}  
- **Risk allocation:** {alloc_pct*100:.2f}% → ≈ **${max_risk_dollars:,.0f}**  
- **Total realized PnL over backtest:** **${realized_pnl_cum[-1]:,.0f}**

- **Current VIX spot:** {spot:.2f}  
- **Suggested long call:** ATM, strike ≈ **{spot:.2f}**, maturity **{long_dte_weeks} weeks**  
  - Premium/contract ≈ **${price_long*100.0:,.2f}**  
  - Delta ≈ **{delta_long:.2f}**  
  - Per-contract cost (incl. fees) ≈ **${per_contract_cost:,.2f}**  

- **Suggested contracts (risk-based):** **{contracts_suggested}**
"""
            )

            K_short = spot + otm_points
            T_short = 1.0 / 52.0
            sigma_short = max((spot / 100.0) * sigma_mult, 0.01)
            price_short = black_scholes_call(spot, K_short, T_short, risk_free, sigma_short)

            st.markdown(
                f"""
- **Suggested short call:**  
  - Strike ≈ **{K_short:.2f}** (spot + {otm_points:.1f})  
  - Maturity: 1 week  
  - Premium/contract received ≈ **${price_short*100.0:,.2f}** (before fees)
"""
            )
        else:
            st.info("Not enough information to compute suggestion (spot or T_long invalid).")
    except Exception as e:
        st.warning(f"Could not compute suggestion: {e}")


# ----- Performance metrics -----

st.subheader("Performance Metrics")
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Total Return (%)", f"{total_return*100:.1f}%")
col2.metric("Total Return ($)", f"{total_return_dollar:,.0f}")
col3.metric("Ending Equity ($)", f"{ending_equity:,.0f}")
col4.metric("CAGR", f"{cagr*100:.1f}%" if not np.isnan(cagr) else "N/A")
col5.metric("Sharpe (ann.)", f"{sharpe:.2f}")
col6.metric("Max Drawdown", f"{max_dd*100:.1f}%")
col7.metric("Win Rate", f"{win_rate*100:.1f}%")


# ----- Charts: Equity+VIX (dark) & Weekly PnL (dark), side by side -----

st.subheader("Equity Curve with VIX & Weekly P&L")

plt.style.use("dark_background")

# Equity + VIX
fig1, ax1 = plt.subplots()
ax1.plot(dates, equity, label="Equity ($)", linewidth=1.0)
ax1.set_ylabel("Equity ($)")
ax2 = ax1.twinx()
ax2.plot(dates, vix_values_weekly, linestyle="--", linewidth=1.0, label="VIX")
ax2.set_ylabel("VIX")
ax1.set_title("Equity Curve and VIX")
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Weekly PnL (%)
fig2, ax2p = plt.subplots()
ax2p.bar(dates, weekly_pnl_pct)
ax2p.set_title("Weekly P&L (%)")
ax2p.set_ylabel("%")
ax2p.grid(True, alpha=0.3)

plt.style.use("default")

col_a, col_b = st.columns(2)
with col_a:
    st.pyplot(fig1, clear_figure=True)
with col_b:
    st.pyplot(fig2, clear_figure=True)


# ----- Weekly data table -----

df = pd.DataFrame(
    {
        "Date": dates_1d,
        "VIX Close": vix_values_weekly,
        "Strategy Weekly P&L (%)": weekly_pnl_1d,
        "Realized PnL this week ($)": realized_pnl_weekly,
        "Unrealized PnL this week ($)": unrealized_pnl_weekly,
        "Cum Realized PnL ($)": realized_pnl_cum,
        "Equity ($)": equity,
    }
)
st.dataframe(df, use_container_width=True)


# ----- Grid scan section (results persist; XLSX export) -----

show_grid = st.checkbox("Show parameter grid scan", value=False)

if show_grid:
    st.subheader("Parameter Grid Scan")

    if st.button("Run / Update grid scan"):
        sorted_df, best_row = run_grid_scan(
            vix_weekly=vix_weekly,
            initial_capital=initial_capital,
            alloc_pct=alloc_pct,
            long_dte_weeks=long_dte_weeks,
            r=risk_free,
            fee_per_contract=fee_per_contract,
            entry_grid=entry_grid_vals,
            otm_grid=otm_grid_vals,
            target_grid=target_grid_vals,
            sigma_grid=sigma_grid_vals,
        )
        st.session_state.grid_results_df = sorted_df
        st.session_state.last_grid_idx = 0

    grid_df = st.session_state.grid_results_df

    if grid_df is None or grid_df.empty:
        st.info("No grid scan results yet. Click **Run / Update grid scan** to compute.")
    else:
        st.markdown(
            "Sorted by **higher CAGR first**, and for the same CAGR, **smaller (less negative) Max Drawdown**."
        )
        st.dataframe(grid_df.round(4), use_container_width=True)

        # XLSX export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            grid_df.to_excel(writer, index=False, sheet_name="scan")
        st.download_button(
            "Download scan results as XLSX",
            data=buffer.getvalue(),
            file_name="vix_5pct_grid_scan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("Select a row to apply its parameters to the sidebar:")

        idx_options = list(grid_df.index)
        sel_idx = st.radio(
            "Grid rows (index)",
            options=idx_options,
            index=min(st.session_state.get("last_grid_idx", 0), len(idx_options) - 1),
            horizontal=True,
        )

        if sel_idx != st.session_state.get("last_grid_idx", 0):
            st.session_state["grid_choice_params"] = {
                "entry_percentile": float(grid_df.loc[sel_idx, "entry_pct"]),
                "otm_points": float(grid_df.loc[sel_idx, "otm_pts"]),
                "target_multiple": float(grid_df.loc[sel_idx, "target_mult"]),
                "sigma_mult": float(grid_df.loc[sel_idx, "sigma_mult"]),
            }
            st.session_state.last_grid_idx = sel_idx
            st.session_state.apply_from_grid = True
            st.rerun()

        st.write("Selected row:")
        st.dataframe(grid_df.loc[[sel_idx]].round(4), use_container_width=True)

st.markdown("---")
st.caption(
    "Educational use only. Simplified approximation of the VIX 5% Weekly idea, "
    "with Black-Scholes pricing, fixed per-contract fees, grid scanning, "
    "user-selectable parameter presets, and separated realized vs unrealized PnL."
)
