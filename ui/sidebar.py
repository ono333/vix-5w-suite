# ui/sidebar.py

import datetime as dt
import streamlit as st


# Registry of strategies (easy to extend later)
STRATEGIES = {
    "diagonal": "Diagonal: LEAP + weekly short calls",
    "long_only": "Long-only: VIX calls only",
    # Later:
    # "osc_310": "3-10 Oscillator + ROC (experimental)",
}


def build_sidebar() -> dict:
    """
    Build the sidebar and return a flat params dict.

    This dict is what app.py expects, plus a "page" field:
        params["page"], params["start_date"], params["end_date"],
        params["initial_capital"], ...
        plus strategy-specific knobs (entry_percentile, long_dte_weeks, etc.)
    """

    st.sidebar.title("VIX 5% Weekly Suite")

    # NOTE: Page selector is in app_improved.py main() - not here
    # This keeps all pages in one place for easier maintenance

    # -------------------------------------------------------
    # Pricing source selector (Synthetic vs Massive)
    # -------------------------------------------------------
    pricing_source = st.sidebar.selectbox(
        "Pricing source",
        options=["Synthetic (BS)", "Massive historical"],
        index=0,
        key="pricing_source",
        help=(
            "Synthetic (BS) = Blackâ€“Scholes-style pricing.\n"
            "Massive historical = real option chains via Massive API."
        ),
    )

    # --- Underlying (for Massive / future use) ---
    underlying = st.sidebar.selectbox(
        "Underlying symbol",
        options=["UVXY", "^VIX", "VXX"],
        index=0,
        key="underlying_symbol",
        help="UVXY recommended for live trading. ^VIX for index analysis.",
    )
    
    # -------------------------------------------------------
    # Global backtest settings (shared by all strategies)
    # -------------------------------------------------------
    start_date = st.sidebar.date_input(
        "Start date",
        value=dt.date(2004, 1, 1),
        key="start_date",
    )

    end_date = st.sidebar.date_input(
        "End date",
        value=dt.date.today(),
        key="end_date",
    )

    st.sidebar.markdown("### Capital & Risk")

    initial_capital = st.sidebar.number_input(
        "Initial capital ($)",
        min_value=0.0,
        value=250_000.0,  # your preferred default
        step=10_000.0,
        format="%.0f",
        key="initial_capital",
    )

    alloc_pct_percent = st.sidebar.number_input(
        "Allocation (% of equity put into strategy)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,     # 1% default as you requested
        step=0.1,
        key="alloc_pct_percent",
        help="Example: 1.0 = allocate 1% of current equity to each new trade.",
    )
    alloc_pct = alloc_pct_percent / 100.0

    risk_free = st.sidebar.number_input(
        "Risk-free rate (annual, r)",
        min_value=0.0,
        max_value=0.20,
        value=0.03,
        step=0.005,
        format="%.3f",
        key="risk_free",
    )

    fee_per_contract = st.sidebar.number_input(
        "Fee per contract ($)",
        min_value=0.0,
        max_value=5.0,
        value=0.65,
        step=0.05,
        format="%.2f",
        key="fee_per_contract",
    )

    realism = st.sidebar.slider(
        "Realism haircut (1.0 = no haircut)",
        min_value=0.5,
        max_value=1.0,
        value=1.0,
        step=0.05,
        key="realism",
        help="Multiply PnL by this factor to simulate fills / micro-issues.",
    )

    st.sidebar.markdown("### Execution / Slippage")

    slippage_bps = st.sidebar.number_input(
        "Slippage (bps vs mid)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        key="slippage_bps",
        help="5 bps â‰ˆ 0.05% disadvantage vs mid for each trade.",
    )

    st.sidebar.markdown("### Parameter history")

    use_best_from_history = st.sidebar.checkbox(
        "Use best scan params for this strategy (if available)",
        value=False,
        key="use_best_from_history",
        help="When ON, app.py will override sidebar knobs with the last "
             "stored 'best' grid-scan row for the selected strategy.",
    )

    st.sidebar.markdown("---")

    # -------------------------------------------------------
    # Strategy selector
    # -------------------------------------------------------
    strategy_key = st.sidebar.selectbox(
        "Strategy",
        options=list(STRATEGIES.keys()),
        format_func=lambda k: STRATEGIES[k],
        key="strategy",
    )

    # Strategy-specific controls
    if strategy_key == "diagonal":
        strategy_params = _build_diagonal_controls()
    elif strategy_key == "long_only":
        strategy_params = _build_long_only_controls()
    else:
        strategy_params = {"mode": strategy_key}

    # -------------------------------------------------------
    # Aggregate into a single flat params dict
    # -------------------------------------------------------
    params = {
        "pricing_source": pricing_source,    # <--- used to pick engine
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "alloc_pct": alloc_pct,
        "risk_free": risk_free,
        "fee_per_contract": fee_per_contract,
        "realism": realism,
        "slippage_bps": slippage_bps,
        "use_best_from_history": use_best_from_history,
        "strategy_key": strategy_key,
        "underlying_symbol": underlying,
    }

    # This includes mode, entry_percentile, etc.
    params.update(strategy_params)

    return params


# ============================================================
# Strategy-specific builders
# ============================================================

def _build_diagonal_controls() -> dict:
    """
    Sidebar controls for the diagonal strategy:
    LEAP long call + weekly short OTM calls.
    """
    st.sidebar.markdown("### Diagonal settings")

    entry_percentile = st.sidebar.slider(
        "Entry percentile (VIX 52-week)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,   # keep 10% default; you can change later
        step=0.01,
        key="diag_entry_percentile",
        help="Enter when current VIX is at or below this percentile of the last N weeks.",
    )

    entry_lookback_weeks = st.sidebar.number_input(
        "Percentile lookback (weeks)",
        min_value=4,
        max_value=260,
        value=52,
        step=4,
        key="diag_entry_lookback_weeks",
    )

    long_dte_weeks = st.sidebar.selectbox(
        "Long call DTE (weeks)",
        options=[13, 26, 52],
        index=1,  # default to 26 weeks
        key="diag_long_dte_weeks",
    )

    otm_pts = st.sidebar.number_input(
        "OTM distance (VIX points)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        key="diag_otm_pts",
    )

    target_mult = st.sidebar.number_input(
        "Profit target multiple on long call value",
        min_value=1.05,
        max_value=3.0,
        value=1.20,
        step=0.05,
        key="diag_target_mult",
        help="Example: 1.20 = take profit when long call â‰ˆ 20% above entry value.",
    )

    exit_mult = st.sidebar.number_input(
        "Stop multiple on long call value",
        min_value=0.1,
        max_value=1.0,
        value=0.50,
        step=0.05,
        key="diag_exit_mult",
        help="Example: 0.50 = stop if long call loses about 50% from entry value.",
    )

    sigma_mult = st.sidebar.number_input(
        "Sigma multiplier for long option pricing",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="diag_sigma_mult",
        help="Scales assumed volatility when pricing LEAPs (synthetic mode).",
    )

    return {
        "mode": "diagonal",
        "entry_percentile": float(entry_percentile),
        "entry_lookback_weeks": int(entry_lookback_weeks),
        "long_dte_weeks": int(long_dte_weeks),
        "otm_pts": float(otm_pts),
        "target_mult": float(target_mult),
        "exit_mult": float(exit_mult),
        "sigma_mult": float(sigma_mult),
    }


def _build_long_only_controls() -> dict:
    """
    Sidebar controls for long-only VIX call strategy.
    Very similar knobs, but no short weekly call.
    """
    st.sidebar.markdown("### Long-only settings")

    entry_percentile = st.sidebar.slider(
        "Entry percentile (VIX 52-week)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
        key="long_entry_percentile",
        help="Enter when current VIX is at or below this percentile of the last N weeks.",
    )

    entry_lookback_weeks = st.sidebar.number_input(
        "Percentile lookback (weeks)",
        min_value=4,
        max_value=260,
        value=52,
        step=4,
        key="long_entry_lookback_weeks",
    )

    long_dte_weeks = st.sidebar.selectbox(
        "Long call DTE (weeks)",
        options=[13, 26, 52],
        index=1,
        key="long_long_dte_weeks",
    )

    otm_pts = st.sidebar.number_input(
        "Strike offset (VIX points)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        key="long_otm_pts",
        help="Distance above spot VIX for the long strike.",
    )

    target_mult = st.sidebar.number_input(
        "Profit target multiple on long call value",
        min_value=1.05,
        max_value=3.0,
        value=1.20,
        step=0.05,
        key="long_target_mult",
    )

    exit_mult = st.sidebar.number_input(
        "Stop multiple on long call value",
        min_value=0.1,
        max_value=1.0,
        value=0.50,
        step=0.05,
        key="long_exit_mult",
    )

    sigma_mult = st.sidebar.number_input(
        "Sigma multiplier for long option pricing",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="long_sigma_mult",
    )

    return {
        "mode": "long_only",
        "entry_percentile": float(entry_percentile),
        "entry_lookback_weeks": int(entry_lookback_weeks),
        "long_dte_weeks": int(long_dte_weeks),
        "otm_pts": float(otm_pts),
        "target_mult": float(target_mult),
        "exit_mult": float(exit_mult),
        "sigma_mult": float(sigma_mult),
    }