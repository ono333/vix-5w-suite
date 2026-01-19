# core/options_selector.py
"""
Helpers for selecting specific option contracts from Massive option chains.

We:
- pick OTM calls near a target strike (spot + otm_pts)
- enforce basic liquidity & spread filters
- return (strike, mid_price, iv_annual)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from config.massive_config import TARGET_ANNUAL_IV
from core.massive_client import get_option_chain


def _mid(bid: float, ask: float) -> float:
    """Compute mid price, handling edge cases gracefully."""
    if bid is None or ask is None:
        return math.nan
    if ask <= 0:
        return math.nan
    if bid < 0:
        bid = 0.0
    return 0.5 * (bid + ask)


def _passes_liquidity_filter(opt: Dict[str, Any]) -> bool:
    """Basic liquidity filter: bid/ask sanity, spread <= 20%, vol & OI."""
    bid = float(opt.get("bid", 0.0))
    ask = float(opt.get("ask", 0.0))
    vol = float(opt.get("volume", 0.0))
    oi = float(opt.get("open_interest", 0.0))

    if bid <= 0 or ask <= bid:
        return False

    spread = ask - bid
    spread_pct = spread / ask if ask > 0 else 1.0
    if spread_pct > 0.20:  # max 20% spread
        return False

    if vol < 10 or oi < 10:
        return False

    return True


def choose_call(
    expiration: str,
    spot: float,
    otm_pts: float,
    call_type: str = "long",
) -> Tuple[float, float, float]:
    """
    Choose a call option for a given expiration & OTM distance.

    Parameters
    ----------
    expiration : str
        Expiration date, 'YYYY-MM-DD'.
    spot : float
        Spot price (e.g. VIX level).
    otm_pts : float
        Desired OTM distance in points (target strike = spot + otm_pts).
    call_type : str
        "long" or "short" (not used differently yet, but kept for future logic).

    Returns
    -------
    (strike, mid_price, iv_annual)
    """
    chain = get_option_chain(expiration)

    target_strike = spot + otm_pts
    candidates: List[Tuple[float, Dict[str, Any]]] = []

    for opt in chain:
        if str(opt.get("type", "")).upper() != "CALL":
            continue

        try:
            strike = float(opt.get("strike", 0.0))
        except Exception:
            continue

        # We only want OTM calls
        if strike < spot:
            continue

        dist = abs(strike - target_strike)
        candidates.append((dist, opt))

    if not candidates:
        raise RuntimeError("No call candidates found in option chain.")

    # Nearest to target_strike first
    candidates.sort(key=lambda x: x[0])

    # Pass 1: require liquidity filter
    for _, opt in candidates:
        if _passes_liquidity_filter(opt):
            strike = float(opt.get("strike"))
            bid = float(opt.get("bid", 0.0))
            ask = float(opt.get("ask", 0.0))
            mid = _mid(bid, ask)
            if not math.isfinite(mid) or mid <= 0:
                continue
            iv = float(opt.get("iv", TARGET_ANNUAL_IV))
            return strike, mid, iv

    # Pass 2: just nearest with a usable mid, even if illiquid
    for _, opt in candidates:
        strike = float(opt.get("strike", 0.0))
        bid = float(opt.get("bid", 0.0))
        ask = float(opt.get("ask", 0.0))
        mid = _mid(bid, ask)
        if not math.isfinite(mid) or mid <= 0:
            continue
        iv = float(opt.get("iv", TARGET_ANNUAL_IV))
        return strike, mid, iv

    raise RuntimeError("Could not find a usable call option for this expiry.")