#!/usr/bin/env python3
"""
Trade Logger for VIX 5% Weekly Suite - Paper Trading Variants

Supports tracking trades from multiple strategy variants for comparison:
- Variant 1: Static parameters (baseline)
- Variant 2: Regime-adaptive
- Variant 3: Custom variation 1
- Variant 4: Custom variation 2  
- Variant 5: Custom variation 3

Each trade log entry captures:
- Variant ID and name
- Entry/exit timestamps
- Entry/exit prices and VIX levels
- Position details (strikes, DTE, contracts)
- P&L metrics
- Regime at entry/exit (if applicable)
- Strategy parameters used

The log can be exported to CSV/Excel for analysis across variants.
"""

from __future__ import annotations

import json
import datetime as dt
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


# =============================================================================
# Trade Entry Dataclass
# =============================================================================

@dataclass
class TradeEntry:
    """Single trade record with all relevant metadata."""
    
    # Identification
    trade_id: str                    # Unique ID: "{variant}_{timestamp}"
    variant_id: int                  # 1-5 for the 5 variants
    variant_name: str                # Descriptive name
    
    # Timing
    entry_date: str                  # ISO format: "2024-01-15"
    exit_date: Optional[str] = None  # None if still open
    entry_week_idx: int = 0
    exit_week_idx: Optional[int] = None
    duration_weeks: Optional[int] = None
    
    # Prices
    entry_underlying: float = 0.0   # VIX/UVXY at entry
    exit_underlying: float = 0.0    # VIX/UVXY at exit
    
    # Position details
    position_type: str = "diagonal"  # "diagonal" or "long_only"
    contracts: int = 0
    long_strike: float = 0.0
    long_dte_weeks: int = 0
    short_strike: Optional[float] = None  # None for long_only
    
    # Entry parameters (what triggered this trade)
    entry_percentile: float = 0.0   # VIX percentile at entry
    entry_threshold: float = 0.0    # Threshold that was used
    
    # Costs
    entry_cost: float = 0.0         # Total cost to enter
    entry_premium_long: float = 0.0 # Premium paid for long
    entry_credit_short: float = 0.0 # Credit from short (if diagonal)
    fees_paid: float = 0.0
    
    # Exit details
    exit_value: float = 0.0         # Total value at exit
    exit_reason: str = ""           # "target", "stop", "expiry", "manual"
    
    # P&L
    pnl_dollars: float = 0.0
    pnl_percent: float = 0.0        # As decimal (0.15 = 15%)
    is_winner: bool = False
    
    # Regime (if using adaptive mode)
    entry_regime: str = "N/A"
    exit_regime: str = "N/A"
    
    # Strategy parameters snapshot
    params_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: str = "open"            # "open", "closed", "cancelled"
    notes: str = ""


# =============================================================================
# Variant Configuration
# =============================================================================

@dataclass
class VariantConfig:
    """Configuration for a paper trading variant."""
    variant_id: int
    name: str
    description: str
    is_regime_adaptive: bool = False
    
    # Base parameters (can be overridden by regime if adaptive)
    entry_percentile: float = 0.25
    long_dte_weeks: int = 26
    otm_pts: float = 5.0
    alloc_pct: float = 0.01
    target_mult: float = 1.25
    exit_mult: float = 0.50
    sigma_mult: float = 1.0
    
    # Additional settings
    use_trailing_stop: bool = False
    max_contracts: int = 100


# Default 5 variants for paper trading
DEFAULT_VARIANTS: Dict[int, VariantConfig] = {
    1: VariantConfig(
        variant_id=1,
        name="Static Baseline",
        description="Fixed parameters, no adaptation",
        is_regime_adaptive=False,
        entry_percentile=0.25,
        long_dte_weeks=26,
        otm_pts=5.0,
        alloc_pct=0.01,
        target_mult=1.25,
        exit_mult=0.50,
    ),
    2: VariantConfig(
        variant_id=2,
        name="Regime Adaptive",
        description="Adjusts params based on VIX percentile regime",
        is_regime_adaptive=True,
        entry_percentile=0.25,  # Will be overridden by regime
        long_dte_weeks=26,
        otm_pts=5.0,
        alloc_pct=0.01,
    ),
    3: VariantConfig(
        variant_id=3,
        name="Aggressive Entry",
        description="Higher entry threshold, larger positions",
        is_regime_adaptive=False,
        entry_percentile=0.40,  # Enter more often
        long_dte_weeks=13,      # Shorter duration
        otm_pts=3.0,            # Closer to ATM
        alloc_pct=0.015,        # Larger position
        target_mult=1.15,       # Take profits faster
        exit_mult=0.55,
    ),
    4: VariantConfig(
        variant_id=4,
        name="Conservative",
        description="Lower entry threshold, smaller positions",
        is_regime_adaptive=False,
        entry_percentile=0.15,  # More selective
        long_dte_weeks=52,      # Longer duration
        otm_pts=8.0,            # Further OTM
        alloc_pct=0.008,        # Smaller position
        target_mult=1.50,       # Higher profit target
        exit_mult=0.40,         # Wider stop
    ),
    5: VariantConfig(
        variant_id=5,
        name="High VIX Contrarian",
        description="Enter when VIX is elevated (contrarian)",
        is_regime_adaptive=False,
        entry_percentile=0.75,  # Enter when VIX is HIGH
        long_dte_weeks=8,       # Short duration for quick trades
        otm_pts=2.0,            # Near ATM for high delta
        alloc_pct=0.01,
        target_mult=1.10,       # Quick profit taking
        exit_mult=0.60,         # Tight stop
    ),
}


# =============================================================================
# Trade Logger Class
# =============================================================================

class TradeLogger:
    """
    Manages trade logging for multiple paper trading variants.
    
    Usage:
        logger = TradeLogger()
        
        # Log a new trade entry
        trade = logger.log_entry(
            variant_id=1,
            entry_date="2024-01-15",
            entry_underlying=15.5,
            contracts=10,
            long_strike=20.0,
            ...
        )
        
        # Update when trade exits
        logger.log_exit(
            trade_id=trade.trade_id,
            exit_date="2024-02-01",
            exit_underlying=18.2,
            exit_value=5500.0,
            exit_reason="target",
        )
        
        # Export for analysis
        df = logger.to_dataframe()
        logger.export_csv("paper_trades.csv")
    """
    
    def __init__(
        self, 
        log_path: Optional[Path] = None,
        variants: Optional[Dict[int, VariantConfig]] = None,
    ):
        """
        Initialize the trade logger.
        
        Args:
            log_path: Path to JSON file for persistence (optional)
            variants: Custom variant configurations (uses defaults if None)
        """
        self.log_path = Path(log_path) if log_path else None
        self.variants = variants or DEFAULT_VARIANTS.copy()
        self.trades: List[TradeEntry] = []
        
        # Load existing trades if path exists
        if self.log_path and self.log_path.exists():
            self._load()
    
    def _generate_trade_id(self, variant_id: int) -> str:
        """Generate unique trade ID."""
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"V{variant_id}_{timestamp}"
    
    def log_entry(
        self,
        variant_id: int,
        entry_date: str,
        entry_underlying: float,
        contracts: int,
        long_strike: float,
        long_dte_weeks: int,
        entry_cost: float,
        entry_percentile: float,
        entry_threshold: float,
        entry_week_idx: int = 0,
        position_type: str = "diagonal",
        short_strike: Optional[float] = None,
        entry_premium_long: float = 0.0,
        entry_credit_short: float = 0.0,
        fees_paid: float = 0.0,
        entry_regime: str = "N/A",
        params_snapshot: Optional[Dict] = None,
        notes: str = "",
    ) -> TradeEntry:
        """
        Log a new trade entry.
        
        Returns:
            TradeEntry object with generated trade_id
        """
        variant = self.variants.get(variant_id)
        if not variant:
            raise ValueError(f"Unknown variant_id: {variant_id}")
        
        trade = TradeEntry(
            trade_id=self._generate_trade_id(variant_id),
            variant_id=variant_id,
            variant_name=variant.name,
            entry_date=entry_date,
            entry_week_idx=entry_week_idx,
            entry_underlying=entry_underlying,
            position_type=position_type,
            contracts=contracts,
            long_strike=long_strike,
            long_dte_weeks=long_dte_weeks,
            short_strike=short_strike,
            entry_percentile=entry_percentile,
            entry_threshold=entry_threshold,
            entry_cost=entry_cost,
            entry_premium_long=entry_premium_long,
            entry_credit_short=entry_credit_short,
            fees_paid=fees_paid,
            entry_regime=entry_regime,
            params_snapshot=params_snapshot or {},
            status="open",
            notes=notes,
        )
        
        self.trades.append(trade)
        self._save()
        
        return trade
    
    def log_exit(
        self,
        trade_id: str,
        exit_date: str,
        exit_underlying: float,
        exit_value: float,
        exit_reason: str,
        exit_week_idx: Optional[int] = None,
        exit_regime: str = "N/A",
        notes: str = "",
    ) -> Optional[TradeEntry]:
        """
        Update a trade with exit information.
        
        Returns:
            Updated TradeEntry or None if not found
        """
        for trade in self.trades:
            if trade.trade_id == trade_id:
                trade.exit_date = exit_date
                trade.exit_week_idx = exit_week_idx
                trade.exit_underlying = exit_underlying
                trade.exit_value = exit_value
                trade.exit_reason = exit_reason
                trade.exit_regime = exit_regime
                trade.status = "closed"
                
                # Calculate P&L
                trade.pnl_dollars = exit_value - trade.entry_cost
                if trade.entry_cost > 0:
                    trade.pnl_percent = trade.pnl_dollars / trade.entry_cost
                trade.is_winner = trade.pnl_dollars > 0
                
                # Calculate duration
                if trade.entry_week_idx is not None and exit_week_idx is not None:
                    trade.duration_weeks = exit_week_idx - trade.entry_week_idx
                
                if notes:
                    trade.notes = f"{trade.notes} | Exit: {notes}".strip(" |")
                
                self._save()
                return trade
        
        return None
    
    def get_open_trades(self, variant_id: Optional[int] = None) -> List[TradeEntry]:
        """Get all open trades, optionally filtered by variant."""
        trades = [t for t in self.trades if t.status == "open"]
        if variant_id is not None:
            trades = [t for t in trades if t.variant_id == variant_id]
        return trades
    
    def get_closed_trades(self, variant_id: Optional[int] = None) -> List[TradeEntry]:
        """Get all closed trades, optionally filtered by variant."""
        trades = [t for t in self.trades if t.status == "closed"]
        if variant_id is not None:
            trades = [t for t in trades if t.variant_id == variant_id]
        return trades
    
    def get_variant_stats(self, variant_id: int) -> Dict[str, Any]:
        """Get performance statistics for a specific variant."""
        closed = self.get_closed_trades(variant_id)
        
        if not closed:
            return {
                "variant_id": variant_id,
                "variant_name": self.variants[variant_id].name,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "avg_duration": 0.0,
                "profit_factor": 0.0,
            }
        
        wins = [t for t in closed if t.is_winner]
        losses = [t for t in closed if not t.is_winner]
        
        total_pnl = sum(t.pnl_dollars for t in closed)
        gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0
        
        durations = [t.duration_weeks for t in closed if t.duration_weeks is not None]
        
        return {
            "variant_id": variant_id,
            "variant_name": self.variants[variant_id].name,
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed) if closed else 0,
            "avg_winner": gross_profit / len(wins) if wins else 0,
            "avg_loser": -gross_loss / len(losses) if losses else 0,
            "avg_duration": np.mean(durations) if durations else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        }
    
    def get_all_variants_stats(self) -> pd.DataFrame:
        """Get comparison statistics for all variants."""
        stats = []
        for variant_id in self.variants:
            stats.append(self.get_variant_stats(variant_id))
        return pd.DataFrame(stats)
    
    def to_dataframe(self, variant_id: Optional[int] = None) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        trades = self.trades
        if variant_id is not None:
            trades = [t for t in trades if t.variant_id == variant_id]
        
        if not trades:
            return pd.DataFrame()
        
        records = [asdict(t) for t in trades]
        df = pd.DataFrame(records)
        
        # Format columns
        if "entry_date" in df.columns:
            df["entry_date"] = pd.to_datetime(df["entry_date"])
        if "exit_date" in df.columns:
            df["exit_date"] = pd.to_datetime(df["exit_date"])
        
        return df
    
    def export_csv(self, path: str, variant_id: Optional[int] = None):
        """Export trades to CSV."""
        df = self.to_dataframe(variant_id)
        df.to_csv(path, index=False)
    
    def export_excel(self, path: str):
        """Export all variants to Excel with one sheet per variant."""
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            # Summary sheet
            summary = self.get_all_variants_stats()
            summary.to_excel(writer, sheet_name="Summary", index=False)
            
            # All trades
            all_trades = self.to_dataframe()
            if not all_trades.empty:
                all_trades.to_excel(writer, sheet_name="All Trades", index=False)
            
            # Per-variant sheets
            for variant_id, variant in self.variants.items():
                df = self.to_dataframe(variant_id)
                if not df.empty:
                    sheet_name = f"V{variant_id}_{variant.name[:20]}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _save(self):
        """Persist trades to JSON file."""
        if not self.log_path:
            return
        
        data = {
            "trades": [asdict(t) for t in self.trades],
            "variants": {k: asdict(v) for k, v in self.variants.items()},
        }
        
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load(self):
        """Load trades from JSON file."""
        if not self.log_path or not self.log_path.exists():
            return
        
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
            
            self.trades = [TradeEntry(**t) for t in data.get("trades", [])]
            
            # Load custom variants if present
            if "variants" in data:
                for k, v in data["variants"].items():
                    self.variants[int(k)] = VariantConfig(**v)
        except Exception as e:
            print(f"Warning: Could not load trade log: {e}")
            self.trades = []


# =============================================================================
# Helper Functions
# =============================================================================

def create_trade_log_from_backtest(
    bt_results: Dict[str, Any],
    variant_id: int,
    vix_weekly: pd.Series,
    params: Dict[str, Any],
    logger: Optional[TradeLogger] = None,
) -> TradeLogger:
    """
    Convert backtest results into structured trade log entries.
    
    Args:
        bt_results: Results from run_backtest() or run_regime_adaptive_backtest()
        variant_id: Which variant this backtest represents (1-5)
        vix_weekly: Price series used in backtest
        params: Parameters used for the backtest
        logger: Existing logger to append to (creates new if None)
    
    Returns:
        TradeLogger with trades from this backtest
    """
    if logger is None:
        logger = TradeLogger()
    
    trade_log = bt_results.get("trade_log", [])
    
    for tr in trade_log:
        entry_idx = tr.get("entry_idx")
        exit_idx = tr.get("exit_idx")
        
        # Get dates
        entry_date = str(vix_weekly.index[entry_idx].date()) if entry_idx and entry_idx < len(vix_weekly) else None
        exit_date = str(vix_weekly.index[exit_idx].date()) if exit_idx and exit_idx < len(vix_weekly) else None
        
        # Get underlying prices
        entry_underlying = float(vix_weekly.iloc[entry_idx]) if entry_idx and entry_idx < len(vix_weekly) else 0
        exit_underlying = float(vix_weekly.iloc[exit_idx]) if exit_idx and exit_idx < len(vix_weekly) else 0
        
        if not entry_date:
            continue
        
        # Log entry
        trade = logger.log_entry(
            variant_id=variant_id,
            entry_date=entry_date,
            entry_underlying=entry_underlying,
            contracts=1,  # Not tracked in original log
            long_strike=tr.get("strike_long", 0),
            long_dte_weeks=params.get("long_dte_weeks", 26),
            entry_cost=tr.get("entry_equity", 0),
            entry_percentile=0,  # Would need to calculate
            entry_threshold=params.get("entry_percentile", 0.25),
            entry_week_idx=entry_idx,
            position_type=params.get("mode", "diagonal"),
            short_strike=tr.get("strike_short"),
            entry_regime=tr.get("entry_regime", "N/A"),
            params_snapshot=params,
        )
        
        # Log exit if closed
        if exit_date:
            logger.log_exit(
                trade_id=trade.trade_id,
                exit_date=exit_date,
                exit_underlying=exit_underlying,
                exit_value=tr.get("exit_equity", 0),
                exit_reason="backtest",
                exit_week_idx=exit_idx,
                exit_regime=tr.get("exit_regime", "N/A"),
            )
    
    return logger
