"""
Regime Detector for VIX 5% Weekly Suite

Detects current volatility regime based on VIX percentile and
other market indicators.

Regimes:
    CALM:      0-25%   VIX percentile - Low volatility
    RISING:   25-50%   VIX percentile - Volatility increasing
    STRESSED: 50-75%   VIX percentile - Elevated volatility
    DECLINING: 75-90%  VIX percentile - Post-spike decay
    EXTREME:  90-100%  VIX percentile - Crisis/spike
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Local import - NOT from numbered folder
from enums import VolatilityRegime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Percentile boundaries for regime classification
REGIME_BOUNDARIES = {
    VolatilityRegime.CALM: (0.00, 0.25),
    VolatilityRegime.RISING: (0.25, 0.50),
    VolatilityRegime.STRESSED: (0.50, 0.75),
    VolatilityRegime.DECLINING: (0.75, 0.90),
    VolatilityRegime.EXTREME: (0.90, 1.00),
}

# Default lookback for percentile calculation
DEFAULT_LOOKBACK_WEEKS = 52


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RegimeSnapshot:
    """A point-in-time regime detection result. Also aliased as RegimeState."""
    """A point-in-time regime detection result."""
    timestamp: dt.datetime
    regime: VolatilityRegime
    vix_level: float
    vix_percentile: float
    lookback_weeks: int
    
    # Regime transition info
    previous_regime: Optional[VolatilityRegime] = None
    regime_duration_days: int = 0
    
    # Additional context
    vix_change_1w: float = 0.0
    vix_change_1m: float = 0.0
    is_transition: bool = False
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.value,
            "vix_level": self.vix_level,
            "vix_percentile": self.vix_percentile,
            "lookback_weeks": self.lookback_weeks,
            "previous_regime": self.previous_regime.value if self.previous_regime else None,
            "regime_duration_days": self.regime_duration_days,
            "vix_change_1w": self.vix_change_1w,
            "vix_change_1m": self.vix_change_1m,
            "is_transition": self.is_transition,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RegimeSnapshot":
        return cls(
            timestamp=dt.datetime.fromisoformat(data["timestamp"]),
            regime=VolatilityRegime(data["regime"]),
            vix_level=data["vix_level"],
            vix_percentile=data["vix_percentile"],
            lookback_weeks=data["lookback_weeks"],
            previous_regime=VolatilityRegime(data["previous_regime"]) if data.get("previous_regime") else None,
            regime_duration_days=data.get("regime_duration_days", 0),
            vix_change_1w=data.get("vix_change_1w", 0.0),
            vix_change_1m=data.get("vix_change_1m", 0.0),
            is_transition=data.get("is_transition", False),
        )


# =============================================================================
# CORE DETECTION FUNCTIONS
# =============================================================================

def calculate_vix_percentile(
    vix_series: pd.Series,
    current_vix: float,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
) -> float:
    """
    Calculate the percentile rank of current VIX within the lookback window.
    
    Args:
        vix_series: Historical VIX data (weekly closes)
        current_vix: Current VIX level
        lookback_weeks: Number of weeks to look back
    
    Returns:
        Percentile (0.0 to 1.0)
    """
    if vix_series is None or len(vix_series) < lookback_weeks:
        # Not enough data - use rough heuristics
        if current_vix < 15:
            return 0.15
        elif current_vix < 20:
            return 0.35
        elif current_vix < 25:
            return 0.55
        elif current_vix < 35:
            return 0.75
        else:
            return 0.95
    
    # Get the lookback window
    window = vix_series.iloc[-lookback_weeks:].values
    
    # Calculate percentile
    percentile = (window < current_vix).mean()
    
    return float(percentile)


def classify_regime(percentile: float) -> VolatilityRegime:
    """
    Classify the volatility regime based on percentile.
    
    Args:
        percentile: VIX percentile (0.0 to 1.0)
    
    Returns:
        VolatilityRegime enum value
    """
    for regime, (low, high) in REGIME_BOUNDARIES.items():
        if low <= percentile < high:
            return regime
    
    # Edge case: percentile == 1.0
    return VolatilityRegime.EXTREME


def detect_regime(
    vix_series: pd.Series,
    current_vix: float,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    previous_snapshot: Optional[RegimeSnapshot] = None,
) -> RegimeSnapshot:
    """
    Detect the current volatility regime.
    
    Args:
        vix_series: Historical VIX data (weekly closes)
        current_vix: Current VIX level
        lookback_weeks: Number of weeks for percentile calculation
        previous_snapshot: Previous regime snapshot for transition detection
    
    Returns:
        RegimeSnapshot with current regime info
    """
    now = dt.datetime.now()
    
    # Calculate percentile
    percentile = calculate_vix_percentile(vix_series, current_vix, lookback_weeks)
    
    # Classify regime
    regime = classify_regime(percentile)
    
    # Calculate VIX changes
    vix_change_1w = 0.0
    vix_change_1m = 0.0
    
    if vix_series is not None and len(vix_series) >= 1:
        vix_1w_ago = vix_series.iloc[-1] if len(vix_series) >= 1 else current_vix
        vix_change_1w = (current_vix - vix_1w_ago) / vix_1w_ago if vix_1w_ago > 0 else 0.0
    
    if vix_series is not None and len(vix_series) >= 4:
        vix_1m_ago = vix_series.iloc[-4] if len(vix_series) >= 4 else current_vix
        vix_change_1m = (current_vix - vix_1m_ago) / vix_1m_ago if vix_1m_ago > 0 else 0.0
    
    # Check for regime transition
    is_transition = False
    previous_regime = None
    regime_duration_days = 0
    
    if previous_snapshot is not None:
        previous_regime = previous_snapshot.regime
        is_transition = (regime != previous_regime)
        
        if not is_transition:
            # Same regime - add to duration
            regime_duration_days = previous_snapshot.regime_duration_days + 7  # Weekly
        else:
            regime_duration_days = 0
    
    return RegimeSnapshot(
        timestamp=now,
        regime=regime,
        vix_level=current_vix,
        vix_percentile=percentile,
        lookback_weeks=lookback_weeks,
        previous_regime=previous_regime,
        regime_duration_days=regime_duration_days,
        vix_change_1w=vix_change_1w,
        vix_change_1m=vix_change_1m,
        is_transition=is_transition,
    )


# =============================================================================
# REGIME HISTORY MANAGEMENT
# =============================================================================

class RegimeHistory:
    """Manages historical regime data for analysis and persistence."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".vix_suite" / "regime_history.json"
        self.snapshots: List[RegimeSnapshot] = []
        self._load()
    
    def _load(self) -> None:
        """Load history from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                self.snapshots = [
                    RegimeSnapshot.from_dict(s) for s in data.get("snapshots", [])
                ]
            except Exception as e:
                print(f"Warning: Could not load regime history: {e}")
                self.snapshots = []
    
    def _save(self) -> None:
        """Save history to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, "w") as f:
                json.dump(
                    {"snapshots": [s.to_dict() for s in self.snapshots]},
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: Could not save regime history: {e}")
    
    def add(self, snapshot: RegimeSnapshot) -> None:
        """Add a new snapshot to history."""
        self.snapshots.append(snapshot)
        self._save()
    
    def get_latest(self) -> Optional[RegimeSnapshot]:
        """Get the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None
    
    def get_history(self, days: int = 365) -> List[RegimeSnapshot]:
        """Get snapshots from the last N days."""
        cutoff = dt.datetime.now() - dt.timedelta(days=days)
        return [s for s in self.snapshots if s.timestamp >= cutoff]
    
    def get_regime_distribution(self, days: int = 365) -> dict:
        """Get distribution of regimes over the last N days."""
        history = self.get_history(days)
        if not history:
            return {}
        
        counts = {}
        for s in history:
            regime = s.regime.value
            counts[regime] = counts.get(regime, 0) + 1
        
        total = len(history)
        return {k: v / total for k, v in counts.items()}
    
    def get_transitions(self, days: int = 365) -> List[Tuple[VolatilityRegime, VolatilityRegime]]:
        """Get all regime transitions in the last N days."""
        history = self.get_history(days)
        return [
            (s.previous_regime, s.regime)
            for s in history
            if s.is_transition and s.previous_regime is not None
        ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_regime_color(regime: VolatilityRegime) -> str:
    """Get display color for a regime."""
    colors = {
        VolatilityRegime.CALM: "#2ECC71",      # Green
        VolatilityRegime.RISING: "#F1C40F",    # Yellow
        VolatilityRegime.STRESSED: "#E67E22",  # Orange
        VolatilityRegime.DECLINING: "#3498DB", # Blue
        VolatilityRegime.EXTREME: "#E74C3C",   # Red
    }
    return colors.get(regime, "#95A5A6")


def get_regime_emoji(regime: VolatilityRegime) -> str:
    """Get emoji indicator for a regime."""
    emojis = {
        VolatilityRegime.CALM: "🟢",
        VolatilityRegime.RISING: "🟡",
        VolatilityRegime.STRESSED: "🟠",
        VolatilityRegime.DECLINING: "🔵",
        VolatilityRegime.EXTREME: "🔴",
    }
    return emojis.get(regime, "⚪")


def format_regime_display(snapshot: RegimeSnapshot) -> str:
    """Format regime snapshot for display."""
    emoji = get_regime_emoji(snapshot.regime)
    
    lines = [
        f"{emoji} **{snapshot.regime.value}** Regime",
        f"VIX: {snapshot.vix_level:.2f} ({snapshot.vix_percentile:.0%} percentile)",
        f"1W Change: {snapshot.vix_change_1w:+.1%} | 1M Change: {snapshot.vix_change_1m:+.1%}",
    ]
    
    if snapshot.is_transition:
        lines.append(f"⚡ **Transition** from {snapshot.previous_regime.value}")
    else:
        lines.append(f"Duration: {snapshot.regime_duration_days} days")
    
    return "\n".join(lines)


# =============================================================================
# REAL-TIME DATA FETCHING
# =============================================================================

def fetch_current_vix() -> Tuple[float, Optional[pd.Series]]:
    """
    Fetch current VIX level and recent history.
    
    Returns:
        (current_vix, vix_weekly_series)
    """
    try:
        import yfinance as yf
        
        # Fetch VIX data
        vix = yf.Ticker("^VIX")
        
        # Current price
        current = vix.info.get("regularMarketPrice") or vix.info.get("previousClose", 20.0)
        
        # Historical data for percentile calculation
        hist = vix.history(period="2y", interval="1wk")
        if hist is not None and not hist.empty:
            vix_series = hist["Close"]
        else:
            vix_series = None
        
        return float(current), vix_series
        
    except Exception as e:
        print(f"Warning: Could not fetch VIX data: {e}")
        return 20.0, None


def get_current_regime(
    vix_level: Optional[float] = None,
    vix_series: Optional[pd.Series] = None,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
) -> RegimeSnapshot:
    """
    Get the current regime, optionally fetching live data.
    
    Args:
        vix_level: Current VIX (fetched if None)
        vix_series: VIX history (fetched if None)
        lookback_weeks: Lookback period for percentile
    
    Returns:
        Current RegimeSnapshot
    """
    # Fetch data if not provided
    if vix_level is None or vix_series is None:
        fetched_vix, fetched_series = fetch_current_vix()
        vix_level = vix_level or fetched_vix
        vix_series = vix_series if vix_series is not None else fetched_series
    
    # Load previous snapshot for transition detection
    history = RegimeHistory()
    previous = history.get_latest()
    
    # Detect current regime
    snapshot = detect_regime(
        vix_series=vix_series,
        current_vix=vix_level,
        lookback_weeks=lookback_weeks,
        previous_snapshot=previous,
    )
    
    # Save to history
    history.add(snapshot)
    
    return snapshot


# =============================================================================
# TEST
# =============================================================================

# Alias for backwards compatibility
RegimeState = RegimeSnapshot


def get_regime_description(regime: VolatilityRegime) -> str:
    """Get description for a regime."""
    descriptions = {
        VolatilityRegime.CALM: "Low volatility environment - ideal for income harvesting strategies",
        VolatilityRegime.RISING: "Volatility increasing - caution advised, reduce new positions",
        VolatilityRegime.STRESSED: "Elevated volatility - hedge activation recommended",
        VolatilityRegime.DECLINING: "Post-spike decay phase - mean reversion opportunities",
        VolatilityRegime.EXTREME: "Crisis/spike conditions - tail strategies and protection",
    }
    return descriptions.get(regime, "Unknown regime")


def get_regime_recommendations(regime: VolatilityRegime) -> list:
    """Get trading recommendations for a regime."""
    recommendations = {
        VolatilityRegime.CALM: [
            "V1 Income Harvester is primary strategy",
            "Diagonal spreads with longer-dated longs",
            "Harvest theta from weekly short calls",
            "Keep position sizes moderate",
        ],
        VolatilityRegime.RISING: [
            "Reduce new income positions",
            "Tighten stops on existing trades",
            "Consider V3 Shock Absorber hedges",
            "Monitor for regime transition to STRESSED",
        ],
        VolatilityRegime.STRESSED: [
            "V3 Shock Absorber is active",
            "Long calls for protection",
            "Reduce or exit income positions",
            "Prepare for potential EXTREME conditions",
        ],
        VolatilityRegime.DECLINING: [
            "V2 Mean Reversion Accelerator is primary",
            "Capture post-spike decay",
            "Diagonal spreads with tighter targets",
            "Monitor for stabilization to CALM",
        ],
        VolatilityRegime.EXTREME: [
            "V4 Convex Tail Hunter for explosive moves",
            "V3 Shock Absorber for protection",
            "Minimal new income positions",
            "Prepare for mean reversion opportunities",
        ],
    }
    return recommendations.get(regime, [])


def get_active_variants_for_regime(regime: VolatilityRegime) -> list:
    """Get list of variant names active in a regime."""
    active = {
        VolatilityRegime.CALM: ["V1 Income Harvester", "V5 Regime-Aware Allocator"],
        VolatilityRegime.RISING: ["V5 Regime-Aware Allocator"],
        VolatilityRegime.STRESSED: ["V3 Shock Absorber", "V5 Regime-Aware Allocator"],
        VolatilityRegime.DECLINING: ["V1 Income Harvester", "V2 Mean Reversion Accelerator", "V5 Regime-Aware Allocator"],
        VolatilityRegime.EXTREME: ["V3 Shock Absorber", "V4 Convex Tail Hunter", "V5 Regime-Aware Allocator"],
    }
    return active.get(regime, [])


if __name__ == "__main__":
    print("Testing regime detector...")
    
    # Test with synthetic data
    np.random.seed(42)
    vix_history = pd.Series(np.random.uniform(12, 35, 52))
    
    # Test CALM regime
    snapshot = detect_regime(vix_history, current_vix=13.0)
    print(f"\nVIX 13.0: {snapshot.regime.value} ({snapshot.vix_percentile:.0%})")
    
    # Test STRESSED regime
    snapshot = detect_regime(vix_history, current_vix=28.0)
    print(f"VIX 28.0: {snapshot.regime.value} ({snapshot.vix_percentile:.0%})")
    
    # Test EXTREME regime
    snapshot = detect_regime(vix_history, current_vix=45.0)
    print(f"VIX 45.0: {snapshot.regime.value} ({snapshot.vix_percentile:.0%})")
    
    print("\n" + format_regime_display(snapshot))
