"""
Regime utility functions for VIX 5% Weekly Suite
"""
from typing import Any


def extract_current_regime(regime_state: Any) -> Any:
    """
    Extract the current regime from a RegimeState object.
    
    Handles various input formats and ensures we get a single regime value.
    """
    if hasattr(regime_state, 'iloc'):
        # It's a Series or DataFrame row
        return regime_state.iloc[-1] if len(regime_state) > 0 else regime_state
    
    if hasattr(regime_state, 'regime'):
        # It's already a RegimeState
        return regime_state.regime
    
    # Return as-is
    return regime_state
