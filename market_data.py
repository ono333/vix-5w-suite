"""
market_data.py — Live market data via yfinance.

Key fixes vs old codebase:
  • VIX percentile uses ^VIX 252-day RANK (not UVXY min/max normalisation)
  • IV estimated from UVXY 30-day historical vol × scaling factor
    (no Massive REST dependency)
"""
from __future__ import annotations
import math
import json
import os
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import yfinance as yf
import numpy as np
import pandas as pd

CACHE_DIR = os.path.expanduser("~/.vix_suite")
os.makedirs(CACHE_DIR, exist_ok=True)

# Regime percentile thresholds
REGIME_MAP = [
    (75, "EXTREME"),
    (60, "STRESSED"),
    (45, "RISING"),
    (30, "DECLINING"),
    (0,  "CALM"),
]


# ---------------------------------------------------------------------------
# VIX  (rank-based percentile over 252 trading days)
# ---------------------------------------------------------------------------
def get_vix_data(lookback_days: int = 252) -> dict:
    """
    Returns:
      vix       : current VIX level
      percentile: rank percentile (0-100) over trailing lookback_days closes
      regime    : CALM / DECLINING / RISING / STRESSED / EXTREME
      history   : pd.Series of closes used for the rank
    """
    try:
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="3y")["Close"].dropna()
        if len(hist) < 30:
            return _vix_fallback()
        current = float(hist.iloc[-1])
        window = hist.tail(lookback_days)
        pct = float((window < current).sum() / len(window) * 100)
        regime = _pct_to_regime(pct)
        return {
            "vix": round(current, 2),
            "percentile": round(pct, 1),
            "regime": regime,
            "history": window,
            "error": None,
        }
    except Exception as e:
        return _vix_fallback(str(e))


def _pct_to_regime(pct: float) -> str:
    for threshold, name in REGIME_MAP:
        if pct >= threshold:
            return name
    return "CALM"


def _vix_fallback(error: str = "fetch failed") -> dict:
    return {
        "vix": None,
        "percentile": None,
        "regime": "UNKNOWN",
        "history": pd.Series(dtype=float),
        "error": error,
    }


# ---------------------------------------------------------------------------
# UVXY price + HV-based IV estimate
# ---------------------------------------------------------------------------
def get_uvxy_data() -> dict:
    """
    Returns current price, 30-day HV, and IV estimate.
    IV ≈ HV × 1.35 (UVXY options consistently trade at premium to realised vol).
    Capped at 300% to avoid absurd prices after a split-adjusted spike.
    """
    try:
        ticker = yf.Ticker("UVXY")
        hist = ticker.history(period="90d")["Close"].dropna()
        if hist.empty:
            return _uvxy_fallback()
        price = float(hist.iloc[-1])
        returns = hist.pct_change().dropna()
        hv30 = float(returns.tail(30).std() * math.sqrt(252)) if len(returns) >= 20 else 1.20
        iv_est = min(hv30 * 1.35, 3.00)
        return {
            "price": round(price, 2),
            "hv30": round(hv30, 4),
            "iv_est": round(iv_est, 4),
            "history": hist,
            "error": None,
        }
    except Exception as e:
        return _uvxy_fallback(str(e))


def _uvxy_fallback(error: str = "fetch failed") -> dict:
    return {
        "price": None,
        "hv30": None,
        "iv_est": 1.20,
        "history": pd.Series(dtype=float),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Vol Triangle  (VIX / VVIX / VIX3M / VIX9D)
# ---------------------------------------------------------------------------
_VOL_TICKERS = {
    "VIX": "^VIX",
    "VVIX": "^VVIX",
    "VIX3M": "^VIX3M",
    "VIX9D": "^VIX9D",
}


def get_vol_triangle() -> dict:
    out = {}
    for label, sym in _VOL_TICKERS.items():
        try:
            h = yf.Ticker(sym).history(period="5d")["Close"].dropna()
            out[label] = round(float(h.iloc[-1]), 2) if not h.empty else None
        except Exception:
            out[label] = None
    # Spike Exhaustion Score (0-100): high when front-month VIX > long-dated VIX
    vix = out.get("VIX"); vix3m = out.get("VIX3M"); vvix = out.get("VVIX")
    ses = None
    if vix and vix3m and vvix:
        contango = (vix3m - vix) / vix * 100   # negative = backwardation
        vvix_z = max(0.0, (vvix - 80) / 40)    # normalise VVIX around 80–120 range
        raw = 50 - contango * 3 + vvix_z * 20
        ses = round(min(max(raw, 0), 100), 1)
    out["spike_exhaustion_score"] = ses
    return out


# ---------------------------------------------------------------------------
# Option expiry date helpers
# ---------------------------------------------------------------------------
def next_friday_from_target(today: date, target_dte: int) -> date:
    """
    Return the nearest Friday to (today + target_dte).
    Ensures the returned date is strictly in the future.
    """
    target = today + timedelta(days=target_dte)
    weekday = target.weekday()           # 0=Mon … 4=Fri … 6=Sun
    days_ahead = (4 - weekday) % 7       # how many days to next Friday
    candidate = target + timedelta(days=days_ahead)
    # If candidate is today or past, push one week
    if candidate <= today:
        candidate += timedelta(weeks=1)
    return candidate


def monthly_expiry_from_target(today: date, target_dte: int) -> date:
    """
    Third Friday of the month containing (today + target_dte).
    Ensures the returned date is > today.
    """
    target = today + timedelta(days=target_dte)
    for _ in range(3):  # at most step 3 months forward
        year, month = target.year, target.month
        first = date(year, month, 1)
        first_fri_delta = (4 - first.weekday()) % 7
        first_fri = first + timedelta(days=first_fri_delta)
        third_fri = first_fri + timedelta(weeks=2)
        if third_fri > today:
            return third_fri
        # advance one month
        if month == 12:
            target = date(year + 1, 1, 1)
        else:
            target = date(year, month + 1, 1)
    return third_fri   # fallback


def format_expiry(d: date) -> str:
    """'Jun 20' display format."""
    return d.strftime("%b %d")


def dte_from_expiry(expiry: date, today: Optional[date] = None) -> int:
    """Calendar days from today to expiry (never < 0)."""
    if today is None:
        today = date.today()
    return max((expiry - today).days, 0)


# ---------------------------------------------------------------------------
# Live option chain — executable bid/ask + liquidity
# ---------------------------------------------------------------------------
def fetch_live_option_price(
    symbol: str,
    expiry: date,
    strike: float,
    option_type: str = "call",
) -> dict:
    """
    Fetch live bid, ask, volume, OI for a single contract via yfinance.
    Falls back to all-None on any error.

    Returns dict with keys:
        bid, ask, mid, volume, open_interest, found,
        actual_strike, actual_expiry
    """
    _empty = {"bid": None, "ask": None, "mid": None,
              "volume": None, "open_interest": None, "found": False,
              "actual_strike": None, "actual_expiry": None}
    try:
        ticker = yf.Ticker(symbol)
        available = ticker.options
        if not available:
            return _empty
        # Find nearest available expiry within 5 calendar days
        exp_dates = [date.fromisoformat(e) for e in available]
        closest = min(exp_dates, key=lambda d: abs((d - expiry).days))
        if abs((closest - expiry).days) > 5:
            return _empty
        chain = ticker.option_chain(closest.isoformat())
        df = chain.calls if option_type == "call" else chain.puts
        if df.empty:
            return _empty
        df = df.copy()
        df["_diff"] = (df["strike"] - strike).abs()
        row = df.loc[df["_diff"].idxmin()]
        if float(row["_diff"]) > 5.0:
            return _empty
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        vol = int(row.get("volume", 0) or 0)
        oi  = int(row.get("openInterest", 0) or 0)
        mid = round((bid + ask) / 2, 2) if (bid + ask) > 0 else None
        return {
            "bid": round(bid, 2), "ask": round(ask, 2), "mid": mid,
            "volume": vol, "open_interest": oi, "found": True,
            "actual_strike": float(row["strike"]),
            "actual_expiry": closest.isoformat(),
        }
    except Exception:
        return _empty


def executable_price(
    bid: Optional[float],
    ask: Optional[float],
    side: str,
    aggressiveness: float = 0.25,
) -> Optional[float]:
    """
    Realistic fill estimate — conservative, not mid-based.

    sell (short leg):  bid + aggressiveness*(ask-bid)
    buy  (long leg):   ask - aggressiveness*(ask-bid)

    aggressiveness=0.25 = 25% toward mid.
    aggressiveness=0.0  = pure bid/ask (most pessimistic).
    aggressiveness=0.5  = mid (what BS gives you — overestimates credit).
    """
    if bid is None or ask is None or ask < bid or (bid == 0 and ask == 0):
        return None
    spread = ask - bid
    if side == "sell":
        return round(bid + aggressiveness * spread, 2)
    else:
        return round(ask - aggressiveness * spread, 2)


def fill_quality_score(
    bid: Optional[float],
    ask: Optional[float],
    volume: Optional[int],
    open_interest: Optional[int],
) -> int:
    """
    0–100 fill quality score.  Higher = tighter spread + more liquid.
      50 pts — bid/ask spread as % of mid (tight = good)
      30 pts — volume
      20 pts — open interest
    """
    if bid is None or ask is None:
        return 0
    score = 0.0
    mid = (bid + ask) / 2
    if mid > 0:
        spread_pct = (ask - bid) / mid
        score += max(0.0, 50.0 * (1 - spread_pct * 2))  # 0% spread=50, 50%=0
    vol = volume or 0
    score += min(30.0, vol / 500 * 30)
    oi = open_interest or 0
    score += min(20.0, oi / 1000 * 20)
    return int(round(score))


def liquidity_label(score: int) -> tuple[str, str]:
    """Returns (label, emoji)."""
    if score >= 70: return "Good",  "🟢"
    if score >= 45: return "Fair",  "🟡"
    if score >= 20: return "Thin",  "🟠"
    return              "Poor",  "🔴"
