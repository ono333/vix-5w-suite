#!/usr/bin/env python3
"""
Regime Adapter for VIX 5% Weekly Suite

Implements regime-based adaptive parameter selection based on VIX percentile levels.
Different volatility environments require different strategy parameters for optimal performance.

Regimes:
    - Ultra Low  (0-10th percentile): VIX unusually low, high complacency
    - Low        (10-25th percentile): Normal calm markets
    - Medium     (25-50th percentile): Average volatility
    - High       (50-75th percentile): Elevated fear
    - Extreme    (75-100th percentile): Panic/crisis mode

Each regime has optimized parameters for:
    - entry_percentile: When to enter (more aggressive in low vol)
    - long_dte_weeks: Option duration (shorter in high vol for faster exits)
    - otm_pts: Strike distance (closer OTM in high vol for higher delta)
    - alloc_pct: Position sizing (smaller in extreme vol)
    - target_mult: Profit target (higher in low vol, lower in high vol)
    - exit_mult: Stop loss (tighter in high vol)
    - sigma_mult: Volatility adjustment for pricing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# Regime Definitions
# =============================================================================

@dataclass
class RegimeConfig:
    """Configuration for a single volatility regime."""
    name: str
    percentile_low: float   # Lower bound (inclusive)
    percentile_high: float  # Upper bound (exclusive)
    
    # Strategy parameters for this regime
    entry_percentile: float = 0.25
    long_dte_weeks: int = 26
    otm_pts: float = 5.0
    alloc_pct: float = 0.01
    target_mult: float = 1.20
    exit_mult: float = 0.50
    sigma_mult: float = 1.0
    max_contracts: int = 100
    
    # Risk controls
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.15
    
    def __post_init__(self):
        """Validate regime boundaries."""
        # Allow percentile_high up to 1.01 to capture 100th percentile edge case
        assert 0 <= self.percentile_low < self.percentile_high <= 1.01


# Default regime configurations (can be overridden by grid scan results)
DEFAULT_REGIMES: Dict[str, RegimeConfig] = {
    "ultra_low": RegimeConfig(
        name="Ultra Low",
        percentile_low=0.0,
        percentile_high=0.10,
        entry_percentile=0.35,    # More aggressive entry when VIX is very low
        long_dte_weeks=26,        # Longer duration, expect vol to stay low
        otm_pts=8.0,              # Further OTM for cheap premium
        alloc_pct=0.015,          # Slightly larger position
        target_mult=1.50,         # Higher profit target
        exit_mult=0.40,           # Wider stop
        sigma_mult=0.8,
        max_contracts=150,
    ),
    "low": RegimeConfig(
        name="Low",
        percentile_low=0.10,
        percentile_high=0.25,
        entry_percentile=0.30,
        long_dte_weeks=26,
        otm_pts=6.0,
        alloc_pct=0.012,
        target_mult=1.35,
        exit_mult=0.45,
        sigma_mult=0.9,
        max_contracts=120,
    ),
    "medium": RegimeConfig(
        name="Medium",
        percentile_low=0.25,
        percentile_high=0.50,
        entry_percentile=0.25,    # Standard entry
        long_dte_weeks=15,        # Medium duration
        otm_pts=5.0,
        alloc_pct=0.01,
        target_mult=1.25,
        exit_mult=0.50,
        sigma_mult=1.0,
        max_contracts=100,
    ),
    "high": RegimeConfig(
        name="High",
        percentile_low=0.50,
        percentile_high=0.75,
        entry_percentile=0.20,    # More selective
        long_dte_weeks=8,         # Shorter duration for faster exits
        otm_pts=3.0,              # Closer to ATM for higher delta
        alloc_pct=0.008,          # Smaller position
        target_mult=1.15,         # Lower profit target (take gains faster)
        exit_mult=0.55,           # Tighter stop
        sigma_mult=1.2,
        max_contracts=80,
    ),
    "extreme": RegimeConfig(
        name="Extreme",
        percentile_low=0.75,
        percentile_high=1.01,     # 1.01 to include 100th percentile
        entry_percentile=0.15,    # Very selective
        long_dte_weeks=5,         # Very short duration
        otm_pts=2.0,              # Near ATM
        alloc_pct=0.005,          # Small position (risk control)
        target_mult=1.10,         # Quick profit taking
        exit_mult=0.60,           # Tight stop
        sigma_mult=1.5,
        max_contracts=50,
        use_trailing_stop=True,
        trailing_stop_pct=0.12,
    ),
}


# =============================================================================
# Regime Adapter Class
# =============================================================================

@dataclass
class RegimeAdapter:
    """
    Adapts strategy parameters based on current VIX percentile regime.
    
    Usage:
        adapter = RegimeAdapter()
        current_pct = 0.35  # 35th percentile
        regime, params = adapter.get_regime_params(current_pct, base_params)
    """
    
    regimes: Dict[str, RegimeConfig] = field(default_factory=lambda: DEFAULT_REGIMES.copy())
    lookback_weeks: int = 52
    
    def get_current_regime(self, percentile: float) -> RegimeConfig:
        """
        Determine which regime applies for the given percentile.
        
        Args:
            percentile: Current VIX percentile (0-1)
            
        Returns:
            RegimeConfig for the matching regime
        """
        for regime in self.regimes.values():
            if regime.percentile_low <= percentile < regime.percentile_high:
                return regime
        
        # Fallback to medium if somehow not matched
        return self.regimes.get("medium", list(self.regimes.values())[2])
    
    def get_regime_params(
        self, 
        percentile: float, 
        base_params: Dict[str, Any]
    ) -> Tuple[RegimeConfig, Dict[str, Any]]:
        """
        Get adapted parameters for the current regime.
        
        Args:
            percentile: Current VIX percentile (0-1)
            base_params: Base parameters from sidebar
            
        Returns:
            Tuple of (regime_config, adapted_params)
        """
        regime = self.get_current_regime(percentile)
        
        # Start with base params
        adapted = dict(base_params)
        
        # Override with regime-specific values
        adapted["entry_percentile"] = regime.entry_percentile
        adapted["long_dte_weeks"] = regime.long_dte_weeks
        adapted["otm_pts"] = regime.otm_pts
        adapted["alloc_pct"] = regime.alloc_pct
        adapted["target_mult"] = regime.target_mult
        adapted["exit_mult"] = regime.exit_mult
        adapted["sigma_mult"] = regime.sigma_mult
        
        # Store regime info for logging
        adapted["_regime_name"] = regime.name
        adapted["_regime_max_contracts"] = regime.max_contracts
        adapted["_use_trailing_stop"] = regime.use_trailing_stop
        adapted["_trailing_stop_pct"] = regime.trailing_stop_pct
        
        return regime, adapted
    
    def compute_percentile_series(
        self, 
        prices: pd.Series, 
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling percentile series for the price data.
        
        Args:
            prices: Weekly price series (VIX/UVXY/etc)
            lookback: Lookback window in weeks (default: self.lookback_weeks)
            
        Returns:
            Series of percentile values (0-1)
        """
        lb = lookback or self.lookback_weeks
        values = prices.values.astype(float)
        n = len(values)
        pct = np.full(n, np.nan, dtype=float)
        
        for i in range(lb, n):
            window = values[i - lb:i]
            pct[i] = (window < values[i]).mean()
        
        return pd.Series(pct, index=prices.index, name="percentile")
    
    def get_regime_history(
        self, 
        prices: pd.Series
    ) -> pd.DataFrame:
        """
        Generate a history of regimes for the entire price series.
        
        Returns DataFrame with columns:
            - date, price, percentile, regime_name
        """
        pct_series = self.compute_percentile_series(prices)
        
        records = []
        for date, (price, pct) in zip(prices.index, zip(prices.values, pct_series.values)):
            if np.isnan(pct):
                regime_name = "N/A"
            else:
                regime = self.get_current_regime(pct)
                regime_name = regime.name
            
            records.append({
                "date": date,
                "price": float(price),
                "percentile": float(pct) if not np.isnan(pct) else None,
                "regime": regime_name,
            })
        
        return pd.DataFrame(records)
    
    def update_regime_params(
        self, 
        regime_key: str, 
        **kwargs
    ) -> None:
        """
        Update parameters for a specific regime.
        
        Args:
            regime_key: Key like "ultra_low", "low", etc.
            **kwargs: Parameter values to update
        """
        if regime_key not in self.regimes:
            raise ValueError(f"Unknown regime: {regime_key}")
        
        regime = self.regimes[regime_key]
        for key, value in kwargs.items():
            if hasattr(regime, key):
                setattr(regime, key, value)


# =============================================================================
# Regime-Based Backtest Runner
# =============================================================================

def run_regime_adaptive_backtest(
    prices: pd.Series,
    base_params: Dict[str, Any],
    adapter: Optional[RegimeAdapter] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Run a backtest that adapts parameters based on the current regime.
    
    Unlike the static backtest which uses fixed parameters, this version
    checks the current VIX percentile each week and adjusts the strategy
    parameters accordingly.
    
    Args:
        prices: Weekly price series
        base_params: Base parameters (risk controls, capital, etc.)
        adapter: RegimeAdapter instance (uses default if None)
        progress_cb: Optional callback for progress updates
        
    Returns:
        Dict with equity, trades, regime_log, etc.
    """
    from core.backtester import bs_call_price, OptionPosition
    
    if adapter is None:
        adapter = RegimeAdapter()
    
    # Core parameters (not regime-dependent)
    initial_cap = float(base_params.get("initial_capital", 250_000))
    fee = float(base_params.get("fee_per_contract", 0.65))
    r = float(base_params.get("risk_free", 0.03))
    realism = float(base_params.get("realism", 1.0))
    mode = base_params.get("mode", "diagonal")
    entry_lookback = int(base_params.get("entry_lookback_weeks", 52))
    
    MAX_QTY = 10_000
    
    price_vals = prices.values.astype(float)
    dates = prices.index.to_list()
    n = len(price_vals)
    
    if n < entry_lookback + 2:
        return {
            "equity": np.array([initial_cap]),
            "weekly_returns": np.array([]),
            "realized_weekly": np.array([]),
            "unrealized_weekly": np.array([]),
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_dur": 0.0,
            "trade_log": [],
            "regime_log": [],
        }
    
    # Compute percentile series
    pct_series = adapter.compute_percentile_series(prices, entry_lookback)
    pct = pct_series.values
    
    # State
    cash = initial_cap
    equity: List[float] = [initial_cap]
    realized_weekly: List[float] = [0.0]
    unrealized_weekly: List[float] = [0.0]
    weekly_returns: List[float] = [0.0]
    
    have_pos = False
    long_pos: Optional[OptionPosition] = None
    short_pos: Optional[OptionPosition] = None
    entry_equity: Optional[float] = None
    entry_idx: Optional[int] = None
    entry_price_long: float = 0.0
    entry_regime: str = ""
    
    # Current regime parameters (updated each week)
    current_params = dict(base_params)
    
    win_flags: List[bool] = []
    trade_durations: List[int] = []
    trade_log: List[Dict[str, Any]] = []
    regime_log: List[Dict[str, Any]] = []
    
    pos_value_prev = 0.0
    
    total_steps = max(n - 1, 1)
    
    for i in range(1, n):
        if progress_cb:
            try:
                progress_cb(i, total_steps)
            except:
                pass
        
        S = float(price_vals[i])
        current_pct = float(pct[i]) if not np.isnan(pct[i]) else 0.5
        
        # Get regime-adapted parameters for this week
        regime, current_params = adapter.get_regime_params(current_pct, base_params)
        
        # Extract regime-specific values
        entry_percentile = current_params["entry_percentile"]
        long_dte_weeks = current_params["long_dte_weeks"]
        otm_pts = current_params["otm_pts"]
        alloc_pct = current_params["alloc_pct"]
        target_mult = current_params["target_mult"]
        exit_mult = current_params["exit_mult"]
        sigma_mult = current_params["sigma_mult"]
        
        # Log regime for this week
        regime_log.append({
            "week_idx": i,
            "date": dates[i],
            "price": S,
            "percentile": current_pct,
            "regime": regime.name,
        })
        
        # Advance DTE on existing positions
        if long_pos is not None:
            long_pos.dte_weeks = max(long_pos.dte_weeks - 1, 0)
        if short_pos is not None:
            short_pos.dte_weeks = max(short_pos.dte_weeks - 1, 0)
        
        # Weekly short expiry & roll
        if short_pos is not None and short_pos.dte_weeks == 0:
            intrinsic = max(S - short_pos.strike, 0.0) * 100.0 * abs(short_pos.quantity)
            cash -= intrinsic * realism
            cash -= fee * abs(short_pos.quantity)
            
            if long_pos is not None and mode == "diagonal":
                strike_short = S + otm_pts
                short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
                sp = bs_call_price(S, strike_short, r, short_sigma, 1.0 / 52.0)
                if not np.isfinite(sp) or sp < 0.0:
                    sp = 0.0
                cash += sp * 100.0 * abs(short_pos.quantity) * realism
                cash -= fee * abs(short_pos.quantity)
                short_pos = OptionPosition(-abs(short_pos.quantity), strike_short, 1, False)
            else:
                short_pos = None
        
        # Mark positions to market
        pos_value = 0.0
        long_val = 0.0
        short_val = 0.0
        
        if long_pos is not None:
            T_long = long_pos.dte_weeks / 52.0
            base_sigma = 0.20 * sigma_mult
            sigma_eff = min(max(base_sigma, 0.10), 2.0)
            lv = bs_call_price(S, long_pos.strike, r, sigma_eff, T_long)
            long_val = max(lv, 0.0) * 100.0 * abs(long_pos.quantity)
            pos_value += long_val
        
        if short_pos is not None:
            T_short = short_pos.dte_weeks / 52.0
            short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
            sv = bs_call_price(S, short_pos.strike, r, short_sigma, T_short)
            short_val = max(sv, 0.0) * 100.0 * short_pos.quantity
            pos_value += short_val
        
        eq_prev = equity[-1]
        unreal_pnl = pos_value - pos_value_prev
        pos_value_prev = pos_value
        
        # Exit logic
        exit_trigger = False
        if long_pos is not None:
            if long_pos.dte_weeks <= 0:
                exit_trigger = True
            
            entry_notional = entry_price_long * abs(long_pos.quantity) * 100.0
            if entry_notional > 0:
                if long_val >= entry_notional * target_mult:
                    exit_trigger = True
                if long_val <= entry_notional * exit_mult:
                    exit_trigger = True
        
        if exit_trigger and long_pos is not None:
            cash += long_val * realism
            
            if short_pos is not None:
                cash += short_val * realism
                short_pos = None
            
            dur = i - (entry_idx if entry_idx is not None else i)
            eq_after_close = cash
            
            base_eq = entry_equity if entry_equity is not None else initial_cap
            pnl = eq_after_close - base_eq
            win_flags.append(pnl > 0)
            trade_durations.append(dur)
            
            trade_log.append({
                "entry_idx": entry_idx,
                "exit_idx": i,
                "entry_date": dates[entry_idx] if entry_idx else None,
                "exit_date": dates[i],
                "duration_weeks": dur,
                "entry_equity": entry_equity,
                "exit_equity": eq_after_close,
                "pnl": pnl,
                "pnl_pct": (pnl / base_eq * 100) if base_eq > 0 else 0,
                "entry_price_long": entry_price_long,
                "strike_long": long_pos.strike,
                "entry_regime": entry_regime,
                "exit_regime": regime.name,
            })
            
            long_pos = None
            have_pos = False
            pos_value_prev = 0.0
            pos_value = 0.0
        
        # Entry logic (using current regime's entry_percentile)
        if (not have_pos) and np.isfinite(pct[i]) and pct[i] <= entry_percentile:
            capital = cash * alloc_pct
            if capital > 0.0:
                strike_long = S + otm_pts
                base_sigma = 0.20 * sigma_mult
                sigma_eff = min(max(base_sigma, 0.10), 2.0)
                
                lp = bs_call_price(S, strike_long, r, sigma_eff, long_dte_weeks / 52.0)
                if lp is not None and np.isfinite(lp) and lp > 0.0:
                    denom = lp * 100.0
                    qty_float = capital / denom
                    
                    if np.isfinite(qty_float) and qty_float > 0.0:
                        qty = int(min(max(qty_float, 0.0), MAX_QTY))
                        if qty > 0:
                            cost_long = qty * lp * 100.0 + fee * qty
                            cash -= cost_long
                            
                            short_pos_local = None
                            if mode == "diagonal":
                                strike_short = S + otm_pts
                                short_sigma = max(0.10, min(2.0, sigma_mult * 0.80))
                                sp = bs_call_price(S, strike_short, r, short_sigma, 1.0 / 52.0)
                                if not np.isfinite(sp) or sp < 0.0:
                                    sp = 0.0
                                credit_short = sp * 100.0 * qty
                                cash += credit_short
                                cash -= fee * qty
                                short_pos_local = OptionPosition(-qty, strike_short, 1, False)
                            
                            long_pos = OptionPosition(qty, strike_long, long_dte_weeks, True)
                            short_pos = short_pos_local
                            have_pos = True
                            entry_equity = cash
                            entry_idx = i
                            entry_price_long = lp
                            entry_regime = regime.name
                            pos_value_prev = 0.0
                            pos_value = 0.0
        
        # End-of-week
        eq_end = cash + pos_value
        equity.append(eq_end)
        realized_weekly.append(eq_end - eq_prev - unreal_pnl)
        unrealized_weekly.append(unreal_pnl)
        weekly_returns.append((eq_end - eq_prev) / eq_prev if eq_prev > 0 else 0.0)
    
    equity_arr = np.asarray(equity, dtype=float)
    weekly_arr = np.asarray(weekly_returns, dtype=float)
    realized_arr = np.asarray(realized_weekly, dtype=float)
    unreal_arr = np.asarray(unrealized_weekly, dtype=float)
    
    trades = len(win_flags)
    win_rate = float(np.mean(win_flags)) if trades > 0 else 0.0
    avg_dur = float(np.mean(trade_durations)) if trade_durations else 0.0
    
    return {
        "equity": equity_arr,
        "weekly_returns": weekly_arr,
        "realized_weekly": realized_arr,
        "unrealized_weekly": unreal_arr,
        "trades": trades,
        "win_rate": win_rate,
        "avg_trade_dur": avg_dur,
        "trade_log": trade_log,
        "regime_log": regime_log,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def get_regime_summary(regime_log: List[Dict]) -> pd.DataFrame:
    """
    Summarize time spent in each regime.
    
    Returns DataFrame with regime name, week count, percentage of time.
    """
    if not regime_log:
        return pd.DataFrame()
    
    df = pd.DataFrame(regime_log)
    summary = df.groupby("regime").size().reset_index(name="weeks")
    summary["pct"] = (summary["weeks"] / len(df) * 100).round(1)
    summary = summary.sort_values("weeks", ascending=False)
    return summary


def get_regime_trade_stats(trade_log: List[Dict]) -> pd.DataFrame:
    """
    Compute trade statistics broken down by entry regime.
    
    Returns DataFrame with stats per regime.
    """
    if not trade_log:
        return pd.DataFrame()
    
    df = pd.DataFrame(trade_log)
    
    stats = []
    for regime in df["entry_regime"].unique():
        subset = df[df["entry_regime"] == regime]
        wins = (subset["pnl"] > 0).sum()
        total = len(subset)
        
        stats.append({
            "regime": regime,
            "trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "avg_pnl": subset["pnl"].mean(),
            "total_pnl": subset["pnl"].sum(),
            "avg_duration": subset["duration_weeks"].mean(),
        })
    
    return pd.DataFrame(stats)
