"""
Premium Harvest Engine
Suggests optimal call/put strikes to sell based on portfolio state,
volatility regime, and synthetic Black-Scholes chain.
No external option chain required — uses pure BS pricing.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import log, sqrt, exp
from datetime import date, timedelta
from typing import Optional
import math


# ── Pure Python BS (no scipy) ─────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
           + t * (-1.821255978 + t * 1.330274429))))
    pdf  = exp(-0.5 * x * x) / sqrt(2 * math.pi)
    cdf  = 1.0 - pdf * poly
    return cdf if x >= 0 else 1.0 - cdf


def bs_call(S, K, T, r=0.05, sigma=0.85) -> tuple[float, float]:
    """Returns (price, delta)"""
    if T <= 0: return max(0, S - K), 1.0 if S > K else 0.0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    price = S*_norm_cdf(d1) - K*exp(-r*T)*_norm_cdf(d2)
    delta = _norm_cdf(d1)
    return round(price, 2), round(delta, 3)


def bs_put(S, K, T, r=0.05, sigma=0.85) -> tuple[float, float]:
    """Returns (price, delta) — delta is negative for puts"""
    if T <= 0: return max(0, K - S), -1.0 if K > S else 0.0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    price = K*exp(-r*T)*_norm_cdf(-d2) - S*_norm_cdf(-d1)
    delta = _norm_cdf(d1) - 1.0
    return round(price, 2), round(delta, 3)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class StrikeSuggestion:
    option_type:     str      # CALL / PUT
    strike:          float
    expiration_date: str
    dte:             int
    delta:           float
    credit:          float    # per share
    income_total:    float    # contracts * 100 * credit
    contracts:       int
    risk_score:      int      # 0-100
    risk_label:      str      # Safe / Caution / Danger
    rationale:       str
    has_hedge:       bool     # protected by long option


@dataclass
class HarvestPlan:
    uvxy_price:      float
    vix_pct:         float
    position_type:   str      # LONG_SHARES / SHORT_SHARES / NO_POSITION
    shares:          int
    call_suggestion: Optional[StrikeSuggestion]
    put_suggestion:  Optional[StrikeSuggestion]
    strategy_note:   str


# ── Implied Vol from VIX percentile ──────────────────────────────────────────

def _implied_vol(vix_level: float, vix_pct: float) -> float:
    """
    UVXY IV is typically 2–3× VIX. Elevated regimes push it higher.
    """
    base = (vix_level / 100) * 2.5
    regime_mult = 1.3 if vix_pct >= 0.90 else 1.15 if vix_pct >= 0.75 else 1.0
    return round(min(2.5, max(0.5, base * regime_mult)), 3)


# ── Risk Score ────────────────────────────────────────────────────────────────

def _risk_score(delta: float, dte: int, vix_pct: float,
                option_type: str, uvxy: float, strike: float) -> tuple[int, str]:
    score = 0
    score += int(abs(delta) * 60)            # delta contributes most
    score += max(0, 20 - dte)                # low DTE = more risk
    score += int(vix_pct * 20)               # high regime = more risk
    dist_pct = abs(strike - uvxy) / uvxy
    if dist_pct < 0.05: score += 20          # close to price
    elif dist_pct < 0.10: score += 10
    score = min(100, score)
    label = "🟢 Safe" if score < 35 else "🟡 Caution" if score < 65 else "🔴 Danger"
    return score, label


# ── Chain Scanner ─────────────────────────────────────────────────────────────

def _scan_chain(
    S: float,
    sigma: float,
    dte: int,
    option_type: str,
    target_delta_lo: float,
    target_delta_hi: float,
    min_credit: float,
) -> Optional[tuple[float, float, float]]:
    """
    Scan synthetic strikes from 50% to 200% of spot.
    Returns (strike, price, delta) of best match.
    """
    T = dte / 365.0
    best = None
    step = 0.50
    K = S * 0.50
    while K <= S * 2.0:
        if option_type == "CALL":
            price, delta = bs_call(S, K, T, sigma=sigma)
        else:
            price, delta = bs_put(S, K, T, sigma=sigma)
            delta = abs(delta)

        if target_delta_lo <= delta <= target_delta_hi and price >= min_credit:
            best = (K, price, delta)
            break   # first match in ascending strike order is best for calls
        K = round(K + step, 2)
        if option_type == "CALL" and K > S * 1.5: break
        if option_type == "PUT"  and K > S:        break

    # For puts, scan downward from spot
    if option_type == "PUT":
        best = None
        K = S
        while K >= S * 0.40:
            price, delta = bs_put(S, K, T, sigma=sigma)
            delta = abs(delta)
            if target_delta_lo <= delta <= target_delta_hi and price >= min_credit:
                best = (K, price, delta)
                break
            K = round(K - step, 2)

    return best


# ── Main Engine ───────────────────────────────────────────────────────────────

def generate_harvest_plan(
    snap,
    shares: int = 0,
    long_call_strike: float = 0.0,
    long_call_contracts: int = 0,
    dte_target: int = 14,
) -> HarvestPlan:
    """
    Core harvest engine.
    snap: VolSnapshot
    shares: current assigned shares (negative = short)
    long_call_strike: protective long call strike if any
    """
    S      = snap.uvxy
    sigma  = _implied_vol(snap.vix, snap.vix_pct)
    T      = dte_target / 365.0
    exp_dt = (date.today() + timedelta(days=dte_target)).strftime("%Y-%m-%d")

    # Regime-based delta targets
    if snap.vix_pct >= 0.90:
        call_delta_range = (0.10, 0.20)   # wider OTM in panic
        put_delta_range  = (0.12, 0.22)
        min_credit       = 0.50
    elif snap.vix_pct >= 0.75:
        call_delta_range = (0.15, 0.25)
        put_delta_range  = (0.15, 0.25)
        min_credit       = 0.35
    else:
        call_delta_range = (0.20, 0.30)
        put_delta_range  = (0.20, 0.30)
        min_credit       = 0.20

    # Determine position type
    if shares < 0:
        pos_type = "SHORT_SHARES"
        contracts = abs(shares) // 100
    elif shares > 0:
        pos_type = "LONG_SHARES"
        contracts = shares // 100
    else:
        pos_type = "NO_POSITION"
        contracts = 1

    contracts = max(1, contracts)

    # ── Call suggestion ──
    call_sugg = None
    call_match = _scan_chain(S, sigma, dte_target, "CALL",
                             *call_delta_range, min_credit)
    if call_match:
        K, price, delta = call_match
        # If short shares — call is a hedge overlay (not covered), cap contracts
        if pos_type == "SHORT_SHARES":
            c = max(1, contracts // 2)
            has_hedge = long_call_contracts > 0
            # NOTE: selling calls against short shares = naked call at IB
            # Flag as hedge BUY suggestion instead
            rationale = (f"⚠️ NAKED at IB — buy protective calls instead | "
                        f"Strike ${round(K,0):.0f} as LONG hedge cap | "
                        f"σ={sigma:.0%}")
        else:
            c = contracts
            has_hedge = long_call_strike > 0 and K < long_call_strike
            rationale = f"Covered call | {call_delta_range[0]:.0%}–{call_delta_range[1]:.0%} delta target | σ={sigma:.0%}"

        rs, rl = _risk_score(delta, dte_target, snap.vix_pct, "CALL", S, K)
        call_sugg = StrikeSuggestion(
            option_type="CALL", strike=round(K, 0),
            expiration_date=exp_dt, dte=dte_target,
            delta=delta, credit=price,
            income_total=round(price * c * 100, 0),
            contracts=c, risk_score=rs, risk_label=rl,
            rationale=rationale, has_hedge=has_hedge,
        )

    # ── Put suggestion ──
    put_sugg = None
    put_match = _scan_chain(S, sigma, dte_target, "PUT",
                            *put_delta_range, min_credit)
    if put_match:
        K, price, delta = put_match
        if pos_type == "LONG_SHARES":
            c = contracts
            has_hedge = False
            rationale = f"Cash-secured put | income against long shares | σ={sigma:.0%}"
        else:
            c = max(1, contracts // 2)
            has_hedge = False
            rationale = (f"Short put | mean-reversion income | "
                        f"{'High vol = elevated premium' if snap.vix_pct>=0.85 else 'Standard regime'} | "
                        f"σ={sigma:.0%}")

        rs, rl = _risk_score(delta, dte_target, snap.vix_pct, "PUT", S, K)
        put_sugg = StrikeSuggestion(
            option_type="PUT", strike=round(K, 0),
            expiration_date=exp_dt, dte=dte_target,
            delta=delta, credit=price,
            income_total=round(price * c * 100, 0),
            contracts=c, risk_score=rs, risk_label=rl,
            rationale=rationale, has_hedge=has_hedge,
        )

    # Strategy note
    if pos_type == "SHORT_SHARES":
        note = ("Short UVXY position — primary income: sell OTM puts below. "
                "Secondary: sell calls above as overlay. "
                "High vol = elevated credits.")
    elif pos_type == "LONG_SHARES":
        note = "Long UVXY position — sell covered calls above. Income yield from spike IV."
    else:
        note = "No assigned position — diagonal spread income only."

    return HarvestPlan(
        uvxy_price=S, vix_pct=snap.vix_pct,
        position_type=pos_type, shares=shares,
        call_suggestion=call_sugg,
        put_suggestion=put_sugg,
        strategy_note=note,
    )
