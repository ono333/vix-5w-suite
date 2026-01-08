"""
VIX Regime Suite - Regime-Adaptive Volatility Trading System

A comprehensive paper trading system with:
- 5 distinct strategy variants (V1-V5)
- Regime-based signal generation
- Robustness scoring for execution survivability
- Post-mortem analytics for variant promotion
- Leg-level trade logging
- Exit detection with priority taxonomy

Modules:
- regime_detector: 5-regime volatility classification
- variant_generator: Role-based variant generation
- trade_logger: Leg-level trade tracking
- robustness_scoring: Execution survivability metrics
- exit_detector: Regime-aware exit suggestions
- post_mortem: Diagnostic backbone for trade analysis
- notification_engine: Email alerts with throttling
- app_paper_trading: Streamlit application
"""

__version__ = "1.1.0"

from .regime_detector import (
    VolatilityRegime,
    RegimeState,
    RegimeTracker,
    detect_regime,
    get_regime_description,
    get_regime_color,
)

from .variant_generator import (
    VariantRole,
    VariantSignal,
    SignalBatch,
    generate_all_variants,
    get_variant_description,
)

from .trade_logger import (
    Trade,
    TradeLeg,
    LegSide,
    LegStatus,
    ExitReason,
    TradeLogManager,
    compute_suggested_tp_sl,
)

from .robustness_scoring import (
    RobustnessComponents,
    compute_robustness_score,
    get_robustness_rating,
    get_robustness_description,
    get_robustness_color,
)

from .exit_detector import (
    ExitSuggestion,
    ExitType,
    ExitUrgency,
    ExitEventManager,
    run_exit_detection,
)

from .post_mortem import (
    TradePostMortem,
    PostMortemStatus,
    ExitType as PMExitType,
    ExitCategory,
    ExecutionComparison,
    OperationalMetrics,
    PostMortemManager,
    calculate_robustness_adjustment,
    infer_exit_type,
    create_post_mortem_from_closed_trade,
)

__all__ = [
    # Regime
    "VolatilityRegime",
    "RegimeState",
    "RegimeTracker",
    "detect_regime",
    "get_regime_description",
    "get_regime_color",
    # Variants
    "VariantRole",
    "VariantSignal",
    "SignalBatch",
    "generate_all_variants",
    "get_variant_description",
    # Trade Logging
    "Trade",
    "TradeLeg",
    "LegSide",
    "LegStatus",
    "ExitReason",
    "TradeLogManager",
    "compute_suggested_tp_sl",
    # Robustness
    "RobustnessComponents",
    "compute_robustness_score",
    "get_robustness_rating",
    "get_robustness_description",
    "get_robustness_color",
    # Exit Detection
    "ExitSuggestion",
    "ExitType",
    "ExitUrgency",
    "ExitEventManager",
    "run_exit_detection",
    # Post-Mortem
    "TradePostMortem",
    "PostMortemStatus",
    "PMExitType",
    "ExitCategory",
    "ExecutionComparison",
    "OperationalMetrics",
    "PostMortemManager",
    "calculate_robustness_adjustment",
    "infer_exit_type",
    "create_post_mortem_from_closed_trade",
]
