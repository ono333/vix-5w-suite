"""
Regime utilities for VIX 5% Weekly Suite
"""
from typing import Union, Optional
from enums import VolatilityRegime

def extract_current_regime(regime_input) -> VolatilityRegime:
    if isinstance(regime_input, VolatilityRegime):
        return regime_input
    if hasattr(regime_input, 'regime'):
        return regime_input.regime
    if isinstance(regime_input, dict):
        val = regime_input.get('regime') or regime_input.get('current_regime', 'CALM')
        if isinstance(val, VolatilityRegime):
            return val
        return VolatilityRegime(str(val).upper())
    if isinstance(regime_input, str):
        return VolatilityRegime(regime_input.upper())
    return VolatilityRegime.CALM

def get_regime_from_percentile(pct: float) -> VolatilityRegime:
    if pct <= 0.25:
        return VolatilityRegime.CALM
    elif pct <= 0.50:
        return VolatilityRegime.RISING
    elif pct <= 0.75:
        return VolatilityRegime.STRESSED
    elif pct <= 0.90:
        return VolatilityRegime.DECLINING
    else:
        return VolatilityRegime.EXTREME

def is_favorable_regime(regime: VolatilityRegime) -> bool:
    return regime in [VolatilityRegime.CALM, VolatilityRegime.DECLINING]
