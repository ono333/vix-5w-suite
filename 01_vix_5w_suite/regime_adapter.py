#!/usr/bin/env python3
"""
Adaptive Regime-Based Strategy Engine for VIX/UVXY

This module dynamically adjusts strategy parameters based on current VIX percentile
to optimize for different market regimes:

- LOW VIX (0-30%): Aggressive diagonal spreads, higher allocation
- MID VIX (30-70%): Balanced approach  
- HIGH VIX (70-100%): Conservative long-only or defensive

Key Innovation: Strategy changes DURING backtest based on rolling VIX percentile,
not just static parameters.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class RegimeConfig:
    """Strategy configuration for a specific VIX regime"""
    name: str
    percentile_range: tuple[float, float]  # (min, max)
    
    # Strategy selection
    mode: str  # "diagonal", "long_only", "defensive"
    
    # Position sizing
    alloc_pct: float
    
    # Entry/exit rules
    entry_percentile: float  # Enter when VIX <= this percentile
    target_mult: float
    exit_mult: float
    
    # Option structure
    long_dte_weeks: int
    otm_pts: float
    sigma_mult: float
    
    # Risk controls
    max_positions: int = 1
    use_stops: bool = True


# Pre-defined regime configurations
REGIME_CONFIGS = {
    "low_vix": RegimeConfig(
        name="Low VIX Regime",
        percentile_range=(0.0, 0.30),
        mode="diagonal",
        alloc_pct=0.02,  # 2% allocation (more aggressive)
        entry_percentile=0.20,  # Enter when VIX <= 20th percentile
        target_mult=1.50,  # Higher profit targets
        exit_mult=0.40,
        long_dte_weeks=26,
        otm_pts=15.0,  # Further OTM
        sigma_mult=0.8,
    ),
    
    "mid_vix": RegimeConfig(
        name="Mid VIX Regime",
        percentile_range=(0.30, 0.70),
        mode="diagonal",
        alloc_pct=0.01,  # 1% allocation (balanced)
        entry_percentile=0.50,
        target_mult=1.30,
        exit_mult=0.45,
        long_dte_weeks=15,
        otm_pts=10.0,
        sigma_mult=1.0,
    ),
    
    "high_vix": RegimeConfig(
        name="High VIX Regime",
        percentile_range=(0.70, 1.00),
        mode="long_only",  # No short leg when VIX is elevated
        alloc_pct=0.005,  # 0.5% allocation (conservative)
        entry_percentile=0.80,  # Only enter on extreme spikes
        target_mult=1.20,
        exit_mult=0.50,
        long_dte_weeks=8,  # Shorter duration
        otm_pts=5.0,  # Closer to money for quick gamma
        sigma_mult=1.2,
    ),
}


class RegimeAdapter:
    """
    Dynamically adapts strategy based on current VIX regime.
    
    This is the core innovation - the strategy changes DURING the backtest
    based on rolling VIX percentile, not just once at the start.
    """
    
    def __init__(
        self,
        vix_weekly: pd.Series,
        lookback_weeks: int = 52,
        regime_configs: Optional[Dict[str, RegimeConfig]] = None,
    ):
        # Ensure vix_weekly is a proper 1D Series
        if isinstance(vix_weekly, pd.DataFrame):
            vix_weekly = vix_weekly.iloc[:, 0]
        self.vix_weekly = pd.Series(vix_weekly).astype(float)
        
        self.lookback_weeks = lookback_weeks
        self.configs = regime_configs or REGIME_CONFIGS
        
        # Compute rolling percentile
        self.vix_percentile = self._compute_rolling_percentile()
        
        # Track regime history
        self.regime_history: List[Dict[str, Any]] = []
    
    def _compute_rolling_percentile(self) -> pd.Series:
        """Compute rolling percentile of VIX with lookback window"""
        prices = self.vix_weekly.values.astype(float)
        n = len(prices)
        pct = np.full(n, np.nan, dtype=float)
        
        lb = max(1, self.lookback_weeks)
        for i in range(lb, n):
            window = prices[i - lb: i]
            pct[i] = (window < prices[i]).mean()
        
        return pd.Series(pct, index=self.vix_weekly.index, name="vix_percentile")
    
    def get_regime(self, week_idx: int) -> RegimeConfig:
        """
        Determine current regime based on VIX percentile at given week.
        
        This is called every week during backtest to adapt strategy.
        """
        if week_idx < 0 or week_idx >= len(self.vix_percentile):
            return self.configs["mid_vix"]  # Default to mid regime
        
        pct = float(self.vix_percentile.iloc[week_idx])
        
        if not np.isfinite(pct):
            return self.configs["mid_vix"]
        
        # Find matching regime
        for regime_name, config in self.configs.items():
            min_pct, max_pct = config.percentile_range
            if min_pct <= pct < max_pct:
                return config
        
        # Fallback
        return self.configs["mid_vix"]
    
    def build_adaptive_params(
        self,
        base_params: Dict[str, Any],
        week_idx: int,
    ) -> Dict[str, Any]:
        """
        Build strategy parameters adapted to current regime.
        
        Called every week to get regime-appropriate parameters.
        """
        regime = self.get_regime(week_idx)
        
        # Start with base params
        params = dict(base_params)
        
        # Override with regime-specific settings
        params.update({
            "mode": regime.mode,
            "alloc_pct": regime.alloc_pct,
            "entry_percentile": regime.entry_percentile,
            "target_mult": regime.target_mult,
            "exit_mult": regime.exit_mult,
            "long_dte_weeks": regime.long_dte_weeks,
            "otm_pts": regime.otm_pts,
            "sigma_mult": regime.sigma_mult,
        })
        
        # Record regime transition
        current_pct = float(self.vix_percentile.iloc[week_idx])
        self.regime_history.append({
            "week_idx": week_idx,
            "date": self.vix_weekly.index[week_idx],
            "vix_level": float(self.vix_weekly.iloc[week_idx]),
            "vix_percentile": current_pct,
            "regime": regime.name,
            "mode": regime.mode,
            "alloc_pct": regime.alloc_pct,
        })
        
        return params
    
    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of regime changes throughout backtest"""
        if not self.regime_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.regime_history)
        
        # Add regime duration
        df["regime_duration"] = df.groupby(
            (df["regime"] != df["regime"].shift()).cumsum()
        ).cumcount() + 1
        
        return df
    
    def analyze_regime_performance(
        self,
        equity_series: np.ndarray,
        trade_log: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by regime to see which works best.
        
        This helps understand if the adaptive approach is actually beneficial.
        """
        if not self.regime_history or len(equity_series) == 0:
            return {}
        
        regime_df = self.get_regime_summary()
        
        results = {}
        for regime_name in self.configs.keys():
            regime_mask = regime_df["regime"] == self.configs[regime_name].name
            regime_weeks = regime_df.loc[regime_mask]
            
            if regime_weeks.empty:
                continue
            
            # Get equity values during this regime
            indices = regime_weeks["week_idx"].values
            valid_indices = indices[indices < len(equity_series)]
            
            if len(valid_indices) < 2:
                continue
            
            regime_equity = equity_series[valid_indices]
            
            # Calculate regime-specific metrics
            regime_return = (regime_equity[-1] / regime_equity[0] - 1.0) if regime_equity[0] > 0 else 0.0
            
            # Count trades in this regime
            regime_dates = set(regime_weeks["date"].values)
            regime_trades = [
                t for t in trade_log 
                if t.get("entry_date") in regime_dates
            ]
            
            results[regime_name] = {
                "weeks": len(valid_indices),
                "return": regime_return,
                "trades": len(regime_trades),
                "avg_vix": float(regime_weeks["vix_level"].mean()),
                "avg_percentile": float(regime_weeks["vix_percentile"].mean()),
            }
        
        return results


def create_regime_summary_table(
    regime_adapter: RegimeAdapter,
    equity: np.ndarray,
    trade_log: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Create a summary table showing how each regime performed.
    
    Useful for UI display.
    """
    perf = regime_adapter.analyze_regime_performance(equity, trade_log)
    
    if not perf:
        return pd.DataFrame()
    
    rows = []
    for regime_name, metrics in perf.items():
        config = regime_adapter.configs[regime_name]
        rows.append({
            "Regime": config.name,
            "Percentile Range": f"{config.percentile_range[0]:.0%}-{config.percentile_range[1]:.0%}",
            "Weeks": metrics["weeks"],
            "Trades": metrics["trades"],
            "Return": f"{metrics['return']:.2%}",
            "Avg VIX": f"{metrics['avg_vix']:.2f}",
            "Mode": config.mode,
            "Allocation": f"{config.alloc_pct:.1%}",
        })
    
    return pd.DataFrame(rows)
