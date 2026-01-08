# utils/regime_utils.py

import pandas as pd
from enums import VolatilityRegime


def extract_current_regime(regime):
    """
    Normalize regime input to a single VolatilityRegime.
    Accepts:
      - VolatilityRegime
      - pandas Series of VolatilityRegime
    Returns:
      - VolatilityRegime
    """
    if isinstance(regime, pd.Series):
        if regime.empty:
            raise ValueError("Regime Series is empty")
        regime = regime.iloc[-1]

    if not isinstance(regime, VolatilityRegime):
        raise TypeError(
            f"Expected VolatilityRegime, got {type(regime)}"
        )

    return regime
