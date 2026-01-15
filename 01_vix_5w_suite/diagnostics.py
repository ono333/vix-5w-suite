#!/usr/bin/env python3
"""
Backtest Diagnostics Tool

Helps identify why trades aren't executing by analyzing:
- Entry signal detection
- Capital availability
- Pricing validity
- Percentile calculations
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import streamlit as st


def analyze_no_trades(
    bt_results: Dict[str, Any],
    vix_weekly: pd.Series,
) -> Dict[str, Any]:
    """
    Analyze why trades didn't execute and provide diagnostic info.
    
    Returns
    -------
    dict with:
        - has_trades : bool
        - entry_signals_count : int
        - no_trade_reasons_count : int
        - percentile_valid : bool
        - diagnosis : str (human-readable)
        - recommendations : list[str]
    """
    
    trades = bt_results.get("trades", 0)
    entry_signals = bt_results.get("entry_signals", [])
    no_trade_reasons = bt_results.get("no_trade_reasons", [])
    
    diagnosis = []
    recommendations = []
    
    # Check 1: Any trades executed?
    has_trades = trades > 0
    
    if has_trades:
        diagnosis.append(f"✓ {trades} trades executed successfully")
        return {
            "has_trades": True,
            "entry_signals_count": len(entry_signals),
            "diagnosis": "\n".join(diagnosis),
            "recommendations": ["System working correctly"],
        }
    
    # Check 2: Entry signals detected?
    entry_count = len(entry_signals)
    
    if entry_count == 0:
        diagnosis.append("✗ No entry signals detected")
        
        # Sub-check: Is percentile data valid?
        regime_hist = bt_results.get("regime_history", [])
        if regime_hist:
            pcts = [r.get("vix_percentile") for r in regime_hist if "vix_percentile" in r]
            valid_pcts = [p for p in pcts if np.isfinite(p)]
            
            if len(valid_pcts) == 0:
                diagnosis.append("  → VIX percentile calculation failed (all NaN)")
                recommendations.append("Check that VIX data has enough history (need 52+ weeks)")
            else:
                min_pct = min(valid_pcts)
                max_pct = max(valid_pcts)
                diagnosis.append(f"  → VIX percentile range: {min_pct:.1%} - {max_pct:.1%}")
                
                if min_pct > 0.30:
                    diagnosis.append("  → VIX never low enough for entry")
                    recommendations.append(
                        "Consider raising entry_percentile threshold "
                        f"(current range {min_pct:.1%}-{max_pct:.1%})"
                    )
    else:
        diagnosis.append(f"✓ {entry_count} entry signals detected")
        
        # Check 3: Why didn't entries convert to trades?
        if no_trade_reasons:
            reason_counts = {}
            for r in no_trade_reasons:
                reason = r.get("reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            diagnosis.append("  Entry signals failed due to:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                diagnosis.append(f"    - {reason}: {count} times")
                
                # Specific recommendations
                if reason == "insufficient_capital":
                    recommendations.append(
                        "Increase initial_capital or reduce alloc_pct"
                    )
                elif reason == "invalid_long_price":
                    recommendations.append(
                        "Option pricing may be failing. Check sigma_mult parameter."
                    )
                elif reason == "quantity_too_small":
                    recommendations.append(
                        "Option prices too high relative to allocation. "
                        "Try: increase alloc_pct or reduce otm_pts."
                    )
                elif reason == "insufficient_cash":
                    recommendations.append(
                        "Not enough cash for computed position size. "
                        "Reduce alloc_pct or increase capital."
                    )
    
    # Check 4: Data quality
    if len(vix_weekly) < 52:
        diagnosis.append(f"⚠ VIX data only has {len(vix_weekly)} weeks (need 52+ for percentile)")
        recommendations.append("Extend date range to at least 1 year")
    
    return {
        "has_trades": False,
        "entry_signals_count": entry_count,
        "no_trade_reasons_count": len(no_trade_reasons),
        "diagnosis": "\n".join(diagnosis),
        "recommendations": recommendations,
    }


def show_diagnostic_panel(
    bt_results: Dict[str, Any],
    vix_weekly: pd.Series,
):
    """Streamlit panel showing backtest diagnostics"""
    
    st.subheader("🔍 Backtest Diagnostics")
    
    diag = analyze_no_trades(bt_results, vix_weekly)
    
    # Diagnosis
    with st.expander("Diagnosis", expanded=not diag["has_trades"]):
        st.text(diag["diagnosis"])
    
    # Recommendations
    if diag["recommendations"]:
        with st.expander("Recommendations", expanded=not diag["has_trades"]):
            for rec in diag["recommendations"]:
                st.info(rec)
    
    # Detailed entry signal analysis
    entry_signals = bt_results.get("entry_signals", [])
    if entry_signals:
        with st.expander(f"Entry Signals ({len(entry_signals)})"):
            df = pd.DataFrame(entry_signals)
            st.dataframe(df, use_container_width=True)
    
    # No-trade reasons
    no_trade_reasons = bt_results.get("no_trade_reasons", [])
    if no_trade_reasons:
        with st.expander(f"Failed Entry Attempts ({len(no_trade_reasons)})"):
            df = pd.DataFrame(no_trade_reasons)
            st.dataframe(df, use_container_width=True)
    
    # Regime history
    regime_hist = bt_results.get("regime_history", [])
    if regime_hist:
        with st.expander(f"Regime History ({len(regime_hist)} weeks)"):
            df = pd.DataFrame(regime_hist)
            
            # Show regime distribution
            if "regime" in df.columns:
                regime_counts = df["regime"].value_counts()
                st.write("Regime Distribution:")
                for regime, count in regime_counts.items():
                    pct = count / len(df) * 100
                    st.write(f"  - {regime}: {count} weeks ({pct:.1f}%)")
            
            st.dataframe(df.tail(20), use_container_width=True)


def create_trade_summary_table(trade_log: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert trade log to readable DataFrame"""
    
    if not trade_log:
        return pd.DataFrame()
    
    df = pd.DataFrame(trade_log)
    
    # Format columns
    if "pnl" in df.columns:
        df["pnl"] = df["pnl"].round(2)
    if "entry_vix" in df.columns:
        df["entry_vix"] = df["entry_vix"].round(2)
    if "exit_vix" in df.columns:
        df["exit_vix"] = df["exit_vix"].round(2)
    
    return df


def show_regime_performance_analysis(
    regime_adapter: Any,
    equity: np.ndarray,
    trade_log: List[Dict[str, Any]],
):
    """Show how each regime performed"""
    
    st.subheader("📊 Regime Performance Analysis")
    
    perf = regime_adapter.analyze_regime_performance(equity, trade_log)
    
    if not perf:
        st.info("Not enough data for regime analysis")
        return
    
    # Create summary table
    rows = []
    for regime_name, metrics in perf.items():
        config = regime_adapter.configs[regime_name]
        rows.append({
            "Regime": config.name,
            "Pct Range": f"{config.percentile_range[0]:.0%}-{config.percentile_range[1]:.0%}",
            "Weeks": metrics["weeks"],
            "Trades": metrics["trades"],
            "Return": f"{metrics['return']:.2%}",
            "Avg VIX": f"{metrics['avg_vix']:.2f}",
            "Mode": config.mode,
            "Alloc": f"{config.alloc_pct:.1%}",
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    
    # Chart regime transitions
    regime_df = regime_adapter.get_regime_summary()
    if not regime_df.empty and "date" in regime_df.columns:
        st.write("Regime over time:")
        
        # Create regime encoding for chart
        regime_map = {config.name: i for i, config in enumerate(regime_adapter.configs.values())}
        regime_df["regime_num"] = regime_df["regime"].map(regime_map)
        
        chart_df = regime_df.set_index("date")[["regime_num", "vix_level"]]
        st.line_chart(chart_df)


def calculate_key_metrics(bt_results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate key performance metrics"""
    
    equity = bt_results.get("equity", np.array([]))
    
    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "sharpe": 0.0,
        }
    
    # Total return
    total_return = (equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0
    
    # CAGR
    weeks = len(equity) - 1
    years = weeks / 52.0
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0 if years > 0 and equity[0] > 0 else 0.0
    
    # Max drawdown
    cummax = np.maximum.accumulate(equity)
    dd = (equity - cummax) / cummax
    max_dd = float(dd.min())
    
    # Sharpe
    weekly_returns = bt_results.get("weekly_returns", np.array([]))
    if len(weekly_returns) > 1:
        mean_ret = np.mean(weekly_returns)
        std_ret = np.std(weekly_returns, ddof=1)
        sharpe = (mean_ret * 52) / (std_ret * np.sqrt(52)) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0
    
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
    }
