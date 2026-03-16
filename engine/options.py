"""
Option Pricing Engine
---------------------
• Black-Scholes pricer with UVXY-specific IV surface
• Generates synthetic option chains at realistic strikes / expirations
• Estimates bid/ask spreads proportional to vega

UVXY IV Reality Check
~~~~~~~~~~~~~~~~~~~~~
UVXY is a 1.5× daily-leveraged VIX ETP. Its realized vol is typically
80-140 % annualised in calm markets, spiking to 200-300 %+ in crises.
IV surface: near-dated ATM IV ≈ min(VIX × 5.5, 280) % in calm regimes.
OTM calls on UVXY carry a *positive* skew at very high strikes because
traders pay for upside vol protection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    if T <= 0 or sigma <= 0:
        return None, None
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(S - K, 0.0)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(K - S, 0.0)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, cp: str = "C") -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return (1.0 if S >= K else 0.0) if cp == "C" else (-1.0 if K >= S else 0.0)
    if cp == "C":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, cp: str = "C") -> float:
    """Daily theta (divide annualised by 365)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None or T <= 0:
        return 0.0
    nd1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)  # φ(d1)
    term1 = -(S * nd1 * sigma) / (2 * math.sqrt(T))
    if cp == "C":
        theta = (term1 - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
    else:
        theta = (term1 + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365
    return theta


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega per 1 % move in IV."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if d1 is None or T <= 0:
        return 0.0
    nd1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return S * nd1 * math.sqrt(T) / 100  # per 1% IV change


# ── UVXY IV Surface ────────────────────────────────────────────────────────────

def uvxy_atm_iv(vix: float) -> float:
    """
    Estimate UVXY at-the-money implied vol from spot VIX.
    Empirical relationship: UVXY IV ≈ VIX × 4.5, clamped to [70%, 260%].
    """
    return min(max(vix * 4.5 / 100, 0.70), 2.60)


def uvxy_iv_surface(vix: float, moneyness: float, dte: int) -> float:
    """
    Approximate UVXY IV for a given moneyness (K/S) and DTE.

    Skew features:
    - Slight positive skew on calls > 1.10 moneyness (spike protection demand)
    - Steep put skew < 0.90 (rare for UVXY but exists)
    - Term structure: shorter DTE → higher IV (vol-of-vol effect)
    """
    base_iv = uvxy_atm_iv(vix)

    # Moneyness skew
    if moneyness > 1.0:
        # OTM call: gentle upward skew (+8% per 10% OTM)
        skew = 1.0 + (moneyness - 1.0) * 0.8
    elif moneyness < 1.0:
        # OTM put: downward skew (less common for UVXY)
        skew = 1.0 + (1.0 - moneyness) * 0.4
    else:
        skew = 1.0

    # Term structure: short-dated contracts have higher IV
    if dte <= 7:
        ts_factor = 1.20
    elif dte <= 14:
        ts_factor = 1.12
    elif dte <= 30:
        ts_factor = 1.05
    elif dte <= 60:
        ts_factor = 1.00
    else:
        ts_factor = 0.94  # LEAPS: slightly lower IV due to mean-reversion

    return min(base_iv * skew * ts_factor, 3.00)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class OptionLeg:
    symbol:    str
    cp:        str        # "C" or "P"
    strike:    float
    expiry:    date
    dte:       int
    theo:      float      # theoretical mid price (BS)
    bid:       float
    ask:       float
    delta:     float
    theta:     float      # daily theta per contract ($)
    vega:      float      # per 1% IV per contract ($)
    iv:        float      # implied vol used (decimal)
    moneyness: float      # K / S

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else 0.0


@dataclass
class DiagonalSpread:
    """A diagonal spread: long LEAP call + short near-term call."""
    long_leg:   OptionLeg
    short_leg:  OptionLeg
    net_debit:  float          # positive = debit paid (max loss)
    max_profit: float          # short strike - long strike - net_debit
    breakeven:  float          # long strike + net_debit

    @property
    def reward_risk(self) -> float:
        return self.max_profit / self.net_debit if self.net_debit > 0 else 0.0

    @property
    def net_delta(self) -> float:
        return self.long_leg.delta - self.short_leg.delta

    @property
    def net_theta(self) -> float:
        # We are LONG the LEAP (costs theta) and SHORT the near-term call (earns theta).
        # Both leg.theta values are stored as negative (BS convention for long calls).
        # Net = -short_leg.theta (we earn it) + long_leg.theta (we pay it)
        # = long_leg.theta - short_leg.theta
        # Positive result means net theta collection.
        return self.long_leg.theta - self.short_leg.theta

    @property
    def net_vega(self) -> float:
        return (self.long_leg.vega - self.short_leg.vega) * 100  # net per 1% IV


# ── Chain builder ──────────────────────────────────────────────────────────────

def _next_friday(ref: date = None) -> date:
    ref = ref or date.today()
    days = (4 - ref.weekday()) % 7
    if days == 0:
        days = 7
    return ref + timedelta(days=days)


def get_weekly_expiries(n: int = 8, ref: date = None) -> List[date]:
    """Return next `n` Friday expirations."""
    expiries = []
    d = _next_friday(ref)
    for _ in range(n):
        expiries.append(d)
        d += timedelta(weeks=1)
    return expiries


def get_monthly_expiries(n: int = 6, ref: date = None) -> List[date]:
    """Return next `n` third-Friday-of-month expirations."""
    ref = ref or date.today()
    expiries = []
    yr, mo = ref.year, ref.month
    count = 0
    while count < n:
        # third Friday
        first = date(yr, mo, 1)
        first_fri = first + timedelta(days=(4 - first.weekday()) % 7)
        third_fri = first_fri + timedelta(weeks=2)
        if third_fri > ref:
            expiries.append(third_fri)
            count += 1
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1
    return expiries


def build_option_leg(
    S: float,
    K: float,
    expiry: date,
    vix: float,
    cp: str = "C",
    r: float = 0.053,
    ref: date = None,
) -> OptionLeg:
    ref   = ref or date.today()
    dte   = max((expiry - ref).days, 0)
    T     = dte / 365.0
    mono  = K / S
    iv    = uvxy_iv_surface(vix, mono, dte)

    if cp == "C":
        theo = bs_call(S, K, T, r, iv)
    else:
        theo = bs_put(S, K, T, r, iv)

    delta = bs_delta(S, K, T, r, iv, cp)
    theta = bs_theta(S, K, T, r, iv, cp) * 100  # per contract
    vega  = bs_vega(S, K, T, r, iv) * 100       # per contract per 1% IV

    # Realistic bid/ask based on vega (wider for illiquid / low-price options)
    half_spread = max(theo * 0.08, 0.02, abs(vega) * 0.10)
    bid   = max(theo - half_spread, 0.01)
    ask   = theo + half_spread

    return OptionLeg(
        symbol    = f"UVXY {cp}{K:.1f} {expiry.strftime('%b%d')}",
        cp        = cp,
        strike    = K,
        expiry    = expiry,
        dte       = dte,
        theo      = round(theo, 2),
        bid       = round(bid,  2),
        ask       = round(ask,  2),
        delta     = round(delta, 3),
        theta     = round(theta, 3),
        vega      = round(vega,  3),
        iv        = round(iv,    4),
        moneyness = round(mono,  4),
    )


def price_diagonal(
    S: float,
    long_strike:   float,
    long_expiry:   date,
    short_strike:  float,
    short_expiry:  date,
    vix:           float,
    ref:           date = None,
) -> DiagonalSpread:
    long_leg  = build_option_leg(S, long_strike,  long_expiry,  vix, "C", ref=ref)
    short_leg = build_option_leg(S, short_strike, short_expiry, vix, "C", ref=ref)

    net_debit  = round(long_leg.ask - short_leg.bid, 2)
    max_profit = round(short_strike - long_strike - net_debit, 2) if short_strike > long_strike else 0.0
    breakeven  = round(long_strike + net_debit, 2)

    return DiagonalSpread(
        long_leg   = long_leg,
        short_leg  = short_leg,
        net_debit  = net_debit,
        max_profit = max_profit,
        breakeven  = breakeven,
    )


def mark_to_market(
    long_strike:   float,
    long_expiry,
    short_strike:  float,
    short_expiry,
    entry_debit:   float,
    S_now:         float,
    vix_now:       float,
    ref:           date = None,
) -> dict:
    """
    Re-price a diagonal spread and return P&L metrics.
    Accepts date objects or ISO date strings for expiry fields.
    Returns dict with keys: long_value, short_value, current_net, pnl, pnl_pct
    """
    if isinstance(long_expiry,  str): long_expiry  = date.fromisoformat(long_expiry)
    if isinstance(short_expiry, str): short_expiry = date.fromisoformat(short_expiry)
    long_now  = build_option_leg(S_now, long_strike,  long_expiry,  vix_now, "C", ref=ref)
    short_now = build_option_leg(S_now, short_strike, short_expiry, vix_now, "C", ref=ref)

    # We are long the LEAP and short the near-term call
    long_value  = long_now.mid
    short_value = short_now.mid   # our liability

    current_net = long_value - short_value
    pnl         = current_net - entry_debit
    pnl_pct     = pnl / abs(entry_debit) * 100 if entry_debit != 0 else 0.0

    return {
        "long_mark":     round(long_value, 2),
        "short_mark":    round(short_value, 2),
        "current_net":   round(current_net, 2),
        "pnl":           round(pnl, 2),
        "pnl_pct":       round(pnl_pct, 1),
        "long_leg":      long_now,
        "short_leg":     short_now,
    }
