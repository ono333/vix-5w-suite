#!/usr/bin/env python3
"""
VIX 5% Weekly Suite – MAIN APP (with Massive integration + UVXY support)

Pages:
- Dashboard
- Backtester (with grid scan)
- Trade Explorer (placeholder)

Relies on:
- ui.sidebar.build_sidebar()
- core.data_loader.load_vix_weekly()
- core.backtester.run_backtest()               # synthetic engine
- core.backtester_massive.run_backtest_massive # Massive API engine
- experiments.grid_scan.run_grid_scan()
- core.param_history.apply_best_if_requested(), get_best_for_strategy()
"""

import io
import datetime as dt
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from ui.sidebar import build_sidebar
from core.data_loader import load_vix_weekly

from core.backtester import run_backtest           # synthetic (BS)
from core.backtester_massive import run_backtest_massive  # Massive

from experiments.grid_scan import run_grid_scan
from core.param_history import apply_best_if_requested, get_best_for_strategy


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _fmt_dollar(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def _fmt_pct(x: float) -> str:
    try:
        return f"{x * 100:,.2f}%"
    except Exception:
        return "n/a"


def _compute_cagr(equity: np.ndarray, weeks_per_year: float = 52.0) -> float:
    if equity is None or len(equity) < 2 or equity[0] <= 0:
        return 0.0
    years = (len(equity) - 1) / weeks_per_year
    if years <= 0:
        return 0.0
    return (equity[-1] / equity[0]) ** (1.0 / years) - 1.0


def _compute_max_dd(equity: np.ndarray) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    e = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(e)
    dd = (e - peak) / peak
    return float(dd.min())


def _compute_vix_percentile_local(vix_weekly: pd.Series,
                                  lookback_weeks: int) -> pd.Series:
    """Rolling percentile of underlying level (VIX / UVXY / etc)."""
    prices = vix_weekly.values.astype(float)
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    lb = max(1, int(lookback_weeks))

    for i in range(lb, n):
        window = prices[i - lb: i]
        out[i] = (window < prices[i]).mean()

    return pd.Series(out, index=vix_weekly.index, name="vix_pct")


def _parse_float_list(s: str) -> list[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            continue
    return vals


def _parse_int_list(s: str) -> list[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            continue
    return vals


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="VIX 5% Weekly Suite",
        layout="wide",
    )

    # Simple confirmation that the app is running
    st.write("✅ debug: main() started")

    # -------------------------------------------------
    # 1. Read sidebar params
    # -------------------------------------------------
    params: Dict[str, Any] = build_sidebar()
    page = params.get("page", "Dashboard")

    start_date: dt.date = params["start_date"]
    end_date: dt.date = params["end_date"]

    # Basic UI selections (before history override)
    ui_pricing_source = params.get("pricing_source", "Synthetic (BS)")
    ui_underlying_symbol = params.get("underlying_symbol", "^VIX")

    # -------------------------------------------------
    # 2. Load data (weekly underlying series)
    # -------------------------------------------------
    # NOTE: load_vix_weekly is already wired on your side to switch
    # between VIX / UVXY depending on sidebar; we keep the call as-is.
    vix_weekly = load_vix_weekly(start_date, end_date)
    if vix_weekly is None or vix_weekly.empty:
        st.error("No underlying data available for the selected date range.")
        return

    # -------------------------------------------------
    # 3. Best-param override & effective params
    # -------------------------------------------------
    effective_params = apply_best_if_requested(params)

    # Make sure pricing_source / underlying_symbol keys exist
    if "pricing_source" not in effective_params:
        effective_params["pricing_source"] = ui_pricing_source
    if "underlying_symbol" not in effective_params:
        effective_params["underlying_symbol"] = ui_underlying_symbol

    pricing_source = effective_params.get("pricing_source", "Synthetic (BS)")
    underlying_symbol = effective_params.get("underlying_symbol", "^VIX")

    # Which engine label?
    if pricing_source == "Massive historical":
        engine_label = f"Massive historical ({underlying_symbol})"
    else:
        engine_label = "Synthetic (Black–Scholes)"

    st.write(f"Engine used: {engine_label}")

    # -------------------------------------------------
    # 4. Run backtest (with optional Massive progress bar)
    # -------------------------------------------------
    bt = None

    if pricing_source == "Massive historical":
        # Streamlit progress bar while Massive chains are being fetched /
        # repriced. We update once per week of the backtest.
        progress_text = st.empty()
        progress_bar = st.progress(0.0)

        def _progress_cb(step: int, total: int):
            if total <= 0:
                return
            frac = min(max(step / float(total), 0.0), 1.0)
            progress_bar.progress(frac)
            progress_text.text(
                f"Massive backtest / chain fetch: {step}/{total} weeks"
            )

        bt = run_backtest_massive(
            vix_weekly,
            effective_params,
            symbol=underlying_symbol,
            progress_cb=_progress_cb,
        )

        # clear progress UI
        progress_bar.empty()
        progress_text.empty()
    else:
        # Pure synthetic (Black–Scholes) engine
        bt = run_backtest(vix_weekly, effective_params)

    # -------------------------------------------------
    # 5. Normalize results & core metrics
    # -------------------------------------------------
    equity = np.asarray(bt["equity"], dtype=float).ravel()
    weekly_returns = np.asarray(bt.get("weekly_returns", []),
                                dtype=float).ravel()
    realized_weekly = np.asarray(bt.get("realized_weekly", []),
                                 dtype=float).ravel()
    unrealized_weekly = np.asarray(bt.get("unrealized_weekly", []),
                                   dtype=float).ravel()

    if len(equity) > 0:
        final_eq = float(equity[-1])
    else:
        final_eq = float(effective_params.get("initial_capital", 0.0))

    cagr = _compute_cagr(equity)
    max_dd = _compute_max_dd(equity)

    initial_cap = float(effective_params.get("initial_capital", 0.0))
    if initial_cap > 0:
        total_ret = final_eq / initial_cap - 1.0
    else:
        total_ret = 0.0

    # =================================================================
    # PAGE: Dashboard
    # =================================================================
    if page == "Dashboard":
        title_underlying = underlying_symbol
        st.title(f"VIX 5% Weekly – Dashboard")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Initial capital", _fmt_dollar(initial_cap))
        col2.metric("Final equity", _fmt_dollar(final_eq))
        col3.metric("Total return", _fmt_pct(total_ret))
        col4.metric("CAGR", _fmt_pct(cagr))
        col5.metric("Max drawdown", _fmt_pct(max_dd))

        # ---- Equity vs underlying chart ----
        st.markdown(f"### Equity Curve vs {title_underlying}")
        n_eq = len(equity)

        # ensure underlying is 1-D
        under_vals = np.asarray(vix_weekly.iloc[:n_eq]).astype(float).ravel()

        df_chart = pd.DataFrame(
            {
                "Equity": np.asarray(equity[:n_eq], dtype=float).ravel(),
                title_underlying: under_vals,
            },
            index=vix_weekly.index[:n_eq],
        )
        st.line_chart(df_chart)

        # ---- Percentile strip (heatmap-ish) ----
        st.markdown(f"### 52-week {title_underlying} Percentile (entry lens)")
        pct_lb = int(effective_params.get("entry_lookback_weeks", 52))
        vix_pct = _compute_vix_percentile_local(vix_weekly, pct_lb)
        df_pct = pd.DataFrame({"VIX_percentile": vix_pct})
        st.area_chart(df_pct)

        st.info(
            "Use this percentile strip as a research lens: "
            "it’s the same entry percentile concept used in the strategy."
        )
        return

    # =================================================================
    # PAGE: Backtester (with Grid Scan)
    # =================================================================
    if page == "Backtester":
        title_underlying = underlying_symbol
        st.title("VIX 5% Weekly – Backtester")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Initial capital", _fmt_dollar(initial_cap))
        col2.metric("Final equity", _fmt_dollar(final_eq))
        col3.metric("Total return", _fmt_pct(total_ret))
        col4.metric("CAGR", _fmt_pct(cagr))
        col5.metric("Max drawdown", _fmt_pct(max_dd))

        # ---------- Equity & underlying ----------
        st.markdown(f"### Equity & {title_underlying}")
        n_eq = len(equity)

        vix_vals_eq = np.asarray(vix_weekly.iloc[:n_eq]).astype(float).ravel()

        df_eq = pd.DataFrame(
            {
                "Equity": np.asarray(equity[:n_eq], dtype=float).ravel(),
                title_underlying: vix_vals_eq,
            },
            index=vix_weekly.index[:n_eq],
        )
        st.line_chart(df_eq)

        # ---------- Weekly PnL ----------
        st.markdown("### Weekly PnL (realized + unrealized)")
        n_pnl = min(len(realized_weekly), len(unrealized_weekly),
                    len(vix_weekly))
        df_pnl = pd.DataFrame(
            {
                "realized": realized_weekly[:n_pnl],
                "unrealized": unrealized_weekly[:n_pnl],
            },
            index=vix_weekly.index[:n_pnl],
        )
        st.bar_chart(df_pnl)

        st.markdown("---")

        # ---------- Grid Scan parameter range UI ----------
        st.subheader("Grid Scan")

        with st.expander("Grid scan parameter ranges", expanded=True):
            ep_str = st.text_input(
                "Entry percentiles (0–1, comma-separated)",
                value="0.10,0.30,0.50,0.70,0.90",
                key="grid_entry_percentiles",
            )
            sigma_str = st.text_input(
                "Sigma multipliers for long option",
                value="0.5,0.8,1.0",
                key="grid_sigma_mults",
            )
            otm_str = st.text_input(
                "OTM distances (underlying points)",
                value="1,2,3,4,5,10,15",
                key="grid_otm_pts",
            )
            dte_str = st.text_input(
                "Long call DTE choices (weeks)",
                value="3,5,15,26",
                key="grid_long_dte_weeks",
            )

        entry_percentiles = _parse_float_list(ep_str)
        sigma_mults = _parse_float_list(sigma_str)
        otm_pts_list = _parse_float_list(otm_str)
        long_dte_weeks_list = _parse_int_list(dte_str)
        _ = (entry_percentiles, sigma_mults,
             otm_pts_list, long_dte_weeks_list)  # quiet unused

        # ---------- Grid Scan proper ----------
        st.subheader("Grid Scan")

        opt_mode = st.radio(
            "Optimization focus",
            [
                "Balanced: high CAGR & low Max DD",
                "Max CAGR only",
                "Min Max Drawdown only",
            ],
            index=0,
            horizontal=False,
            key="grid_opt_mode",
        )

        if "Balanced" in opt_mode:
            criteria = "balanced"
        elif "Max CAGR" in opt_mode:
            criteria = "cagr"
        else:
            criteria = "maxdd"

        if st.button("Run / Update grid scan"):
            grid_df = run_grid_scan(
                vix_weekly,
                effective_params,
                criteria=criteria,
                entry_grid=entry_percentiles,
                sigma_grid=sigma_mults,
                otm_grid=otm_pts_list,
                dte_grid=long_dte_weeks_list,
            )
            st.session_state["grid_df"] = grid_df

        grid_df = st.session_state.get("grid_df")
        if grid_df is not None and not grid_df.empty:
            st.dataframe(grid_df, use_container_width=True)

            # Download XLSX
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                grid_df.to_excel(writer,
                                 index=False,
                                 sheet_name="grid_scan")
            buf.seek(0)
            st.download_button(
                "Download grid scan as XLSX",
                data=buf,
                file_name=f"{effective_params['mode']}_grid_scan.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
            )

            st.markdown("#### Best parameters for this mode (last scan)")
            best = get_best_for_strategy(effective_params["mode"])
            if best:
                st.json(best["row"])
            else:
                st.info("No best-parameter history yet for this mode.")
        else:
            st.info("Run the grid scan to see ranked parameter combos.")

        return

    # =================================================================
    # PAGE: Trade Explorer  (placeholder for now)
    # =================================================================
    if page == "Trade Explorer":
        st.title("VIX 5% Weekly – Trade Explorer")
        st.info(
            "Trade Explorer wiring will be restored after the unified "
            "backtester + Massive integration is fully stabilised. "
            "Dashboard and Backtester (with Grid Scan) should be usable now."
        )
        return


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()