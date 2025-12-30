#!/usr/bin/env python3
"""
Regime Adapter for VIX 5% Weekly Suite

Detects market volatility regimes based on VIX/UVXY percentile levels and 
provides regime-specific parameter configurations.

Regime Definitions (based on 52-week VIX percentile):
- ULTRA_LOW  : 0-10%   (extremely calm markets - rare opportunities)
- LOW        : 10-25%  (low volatility - ideal entry zone)  
- MEDIUM     : 25-50%  (normal volatility - moderate positioning)
- HIGH       : 50-75%  (elevated volatility - defensive mode)
- EXTREME    : 75-100% (crisis/spike - avoid new positions)

Each regime has optimized parameters for:
- Entry percentile thresholds
- Position sizing (alloc_pct)
- OTM distance
- DTE selection
- Profit/stop targets
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Regime Configuration Dataclass
# ============================================================

@dataclass
class RegimeConfig:
    """Configuration parameters for a specific market regime."""
    
    name: str
    percentile_range: Tuple[float, float]  # (low, high) e.g. (0.0, 0.10)
    
    # Entry parameters
    entry_percentile: float = 0.10
    entry_lookback_weeks: int = 52
    
    # Position structure
    mode: str = "diagonal"  # "diagonal" or "long_only"
    alloc_pct: float = 0.01  # fraction of equity per trade
    max_positions: int = 1
    
    # Option selection
    otm_pts: float = 10.0
    sigma_mult: float = 1.0
    long_dte_weeks: int = 26
    
    # Exit parameters
    target_mult: float = 1.20
    exit_mult: float = 0.50
    
    # Risk controls
    use_stops: bool = True
    max_loss_pct: float = 0.50  # max loss before forced exit
    
    # Regime-specific notes
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for use in backtester."""
        return asdict(self)
    
    def apply_to_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this regime's settings to base parameters."""
        params = dict(base_params)
        params.update({
            "entry_percentile": self.entry_percentile,
            "entry_lookback_weeks": self.entry_lookback_weeks,
            "mode": self.mode,
            "alloc_pct": self.alloc_pct,
            "otm_pts": self.otm_pts,
            "sigma_mult": self.sigma_mult,
            "long_dte_weeks": self.long_dte_weeks,
            "target_mult": self.target_mult,
            "exit_mult": self.exit_mult,
        })
        return params


# ============================================================
# Default Regime Configurations
# ============================================================

REGIME_CONFIGS: Dict[str, RegimeConfig] = {
    "ULTRA_LOW": RegimeConfig(
        name="Ultra Low",
        percentile_range=(0.0, 0.10),
        entry_percentile=0.08,      # Very tight entry - rare opportunity
        alloc_pct=0.015,            # Slightly higher allocation
        otm_pts=8.0,                # Closer strikes (lower premium)
        sigma_mult=0.8,             # Lower vol assumption
        long_dte_weeks=26,          # Standard duration
        target_mult=1.30,           # Higher target (vol expansion expected)
        exit_mult=0.40,             # Tighter stop
        description="Extremely calm markets - best entry conditions, rare",
    ),
    
    "LOW": RegimeConfig(
        name="Low",
        percentile_range=(0.10, 0.25),
        entry_percentile=0.15,      # Standard low-vol entry
        alloc_pct=0.01,             # Normal allocation
        otm_pts=10.0,               # Standard OTM
        sigma_mult=1.0,             # Normal vol
        long_dte_weeks=26,
        target_mult=1.20,
        exit_mult=0.50,
        description="Low volatility - primary entry zone",
    ),
    
    "MEDIUM": RegimeConfig(
        name="Medium",
        percentile_range=(0.25, 0.50),
        entry_percentile=0.30,      # More selective entry
        alloc_pct=0.008,            # Reduced sizing
        otm_pts=12.0,               # Further OTM
        sigma_mult=1.2,             # Higher vol assumption
        long_dte_weeks=20,          # Shorter duration
        target_mult=1.15,           # Lower target
        exit_mult=0.55,             # Wider stop
        description="Normal volatility - moderate positioning",
    ),
    
    "HIGH": RegimeConfig(
        name="High",
        percentile_range=(0.50, 0.75),
        entry_percentile=0.60,      # Very selective
        alloc_pct=0.005,            # Small sizing
        otm_pts=15.0,               # Much further OTM
        sigma_mult=1.5,             # Elevated vol
        long_dte_weeks=13,          # Shorter leaps
        target_mult=1.10,           # Quick profits
        exit_mult=0.60,             # Wider stop
        mode="long_only",           # No short leg in high vol
        description="Elevated volatility - defensive, reduce exposure",
    ),
    
    "EXTREME": RegimeConfig(
        name="Extreme",
        percentile_range=(0.75, 1.0),
        entry_percentile=0.90,      # Almost never enter
        alloc_pct=0.003,            # Minimal sizing
        otm_pts=20.0,               # Very far OTM
        sigma_mult=2.0,             # Extreme vol
        long_dte_weeks=8,           # Short duration only
        target_mult=1.05,           # Quick scalps
        exit_mult=0.70,             # Wide stop
        mode="long_only",           # No shorts
        use_stops=True,
        description="Crisis/spike - avoid new positions, manage existing",
    ),
}


# ============================================================
# Regime Adapter Class
# ============================================================

class RegimeAdapter:
    """
    Manages regime detection and parameter adaptation.
    
    Usage:
        adapter = RegimeAdapter(vix_weekly, lookback_weeks=52)
        adapter.compute_regime_history()
        
        # Get current regime
        regime = adapter.get_regime_at_index(i)
        config = adapter.get_config_at_index(i)
        
        # Apply optimized params
        params = config.apply_to_params(base_params)
    """
    
    def __init__(
        self,
        vix_weekly: pd.Series,
        lookback_weeks: int = 52,
        configs: Optional[Dict[str, RegimeConfig]] = None,
    ):
        self.vix_weekly = vix_weekly
        self.lookback_weeks = lookback_weeks
        self.configs = configs or REGIME_CONFIGS
        
        self.prices = vix_weekly.values.astype(float)
        self.dates = vix_weekly.index.tolist()
        self.n = len(self.prices)
        
        # Computed data
        self.percentiles: Optional[np.ndarray] = None
        self.regime_history: List[Dict[str, Any]] = []
        
    def compute_percentiles(self) -> np.ndarray:
        """Compute rolling percentile for each week."""
        pct = np.full(self.n, np.nan, dtype=float)
        lb = max(1, self.lookback_weeks)
        
        for i in range(lb, self.n):
            window = self.prices[i - lb:i]
            pct[i] = (window < self.prices[i]).mean()
        
        self.percentiles = pct
        return pct
    
    def _percentile_to_regime(self, pct: float) -> str:
        """Map a percentile value to a regime name."""
        if not np.isfinite(pct):
            return "MEDIUM"  # Default fallback
        
        for regime_name, config in self.configs.items():
            low, high = config.percentile_range
            if low <= pct < high:
                return regime_name
        
        # Edge case: pct == 1.0
        return "EXTREME"
    
    def compute_regime_history(self) -> List[Dict[str, Any]]:
        """
        Compute the regime for each week in the series.
        
        Returns list of dicts with:
            - date: datetime
            - price: underlying price
            - percentile: rolling percentile
            - regime: regime name
            - config: RegimeConfig object
        """
        if self.percentiles is None:
            self.compute_percentiles()
        
        history = []
        for i in range(self.n):
            pct = self.percentiles[i]
            regime_name = self._percentile_to_regime(pct)
            
            history.append({
                "date": self.dates[i],
                "index": i,
                "price": float(self.prices[i]),
                "percentile": float(pct) if np.isfinite(pct) else None,
                "regime": regime_name,
            })
        
        self.regime_history = history
        return history
    
    def get_regime_at_index(self, index: int) -> str:
        """Get regime name at a specific index."""
        if not self.regime_history:
            self.compute_regime_history()
        
        if 0 <= index < len(self.regime_history):
            return self.regime_history[index]["regime"]
        return "MEDIUM"
    
    def get_config_at_index(self, index: int) -> RegimeConfig:
        """Get RegimeConfig at a specific index."""
        regime_name = self.get_regime_at_index(index)
        return self.configs.get(regime_name, self.configs["MEDIUM"])
    
    def get_percentile_at_index(self, index: int) -> float:
        """Get percentile at a specific index."""
        if self.percentiles is None:
            self.compute_percentiles()
        
        if 0 <= index < len(self.percentiles):
            return float(self.percentiles[index])
        return 0.5
    
    def get_regime_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each regime.
        
        Returns DataFrame with:
            - regime: name
            - count: number of weeks
            - pct_of_total: percentage of total weeks
            - avg_price: average underlying price
            - min_price: minimum price
            - max_price: maximum price
        """
        if not self.regime_history:
            self.compute_regime_history()
        
        df = pd.DataFrame(self.regime_history)
        
        summary = df.groupby("regime").agg({
            "price": ["count", "mean", "min", "max"],
            "percentile": "mean",
        }).round(4)
        
        summary.columns = ["count", "avg_price", "min_price", "max_price", "avg_pct"]
        summary["pct_of_total"] = (summary["count"] / len(df) * 100).round(2)
        
        # Reorder columns
        summary = summary[["count", "pct_of_total", "avg_price", "min_price", "max_price", "avg_pct"]]
        
        return summary.reset_index()
    
    def get_regime_transitions(self) -> List[Dict[str, Any]]:
        """
        Get list of regime transitions (when regime changes).
        
        Returns list of dicts with:
            - date: transition date
            - from_regime: previous regime
            - to_regime: new regime
            - price: price at transition
        """
        if not self.regime_history:
            self.compute_regime_history()
        
        transitions = []
        for i in range(1, len(self.regime_history)):
            prev = self.regime_history[i - 1]
            curr = self.regime_history[i]
            
            if prev["regime"] != curr["regime"]:
                transitions.append({
                    "date": curr["date"],
                    "from_regime": prev["regime"],
                    "to_regime": curr["regime"],
                    "price": curr["price"],
                    "percentile": curr["percentile"],
                })
        
        return transitions
    
    def get_weeks_in_regime(self, regime_name: str) -> pd.Series:
        """
        Get a subset of vix_weekly for weeks that were in a specific regime.
        
        Useful for per-regime grid scans.
        """
        if not self.regime_history:
            self.compute_regime_history()
        
        mask = [h["regime"] == regime_name for h in self.regime_history]
        return self.vix_weekly[mask]
    
    def get_regime_indices(self, regime_name: str) -> List[int]:
        """Get list of indices that belong to a specific regime."""
        if not self.regime_history:
            self.compute_regime_history()
        
        return [h["index"] for h in self.regime_history if h["regime"] == regime_name]


# ============================================================
# Helper Functions
# ============================================================

def create_regime_timeline_df(adapter: RegimeAdapter) -> pd.DataFrame:
    """
    Create a DataFrame suitable for plotting regime timeline.
    
    Returns DataFrame with columns:
        - date: datetime index
        - price: underlying price
        - percentile: rolling percentile
        - regime: regime name
        - regime_num: numeric regime (for color coding)
    """
    if not adapter.regime_history:
        adapter.compute_regime_history()
    
    df = pd.DataFrame(adapter.regime_history)
    
    # Add numeric regime for plotting
    regime_order = ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"]
    df["regime_num"] = df["regime"].map({r: i for i, r in enumerate(regime_order)})
    
    return df


def get_optimized_config_for_regime(
    mode: str,
    regime_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Load optimized parameters for a specific regime from param_history.
    
    Falls back to static REGIME_CONFIGS if no optimization found.
    """
    try:
        from core.param_history import get_best_for_regime
        best = get_best_for_regime(mode, regime_name)
        if best and "row" in best:
            return best["row"]
    except ImportError:
        pass
    
    # Fallback to static config
    if regime_name in REGIME_CONFIGS:
        return REGIME_CONFIGS[regime_name].to_dict()
    
    return None


# ============================================================
# For backwards compatibility
# ============================================================

def detect_regime(vix_weekly: pd.Series, index: int, lookback_weeks: int = 52) -> str:
    """
    Simple function to detect regime at a specific index.
    
    For quick lookups without creating a full RegimeAdapter.
    """
    prices = vix_weekly.values.astype(float)
    n = len(prices)
    
    if index < lookback_weeks or index >= n:
        return "MEDIUM"
    
    window = prices[index - lookback_weeks:index]
    pct = (window < prices[index]).mean()
    
    for regime_name, config in REGIME_CONFIGS.items():
        low, high = config.percentile_range
        if low <= pct < high:
            return regime_name
    
    return "EXTREME"
