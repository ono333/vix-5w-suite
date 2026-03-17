"""
signal_engine.py

Generates option diagonal spread signals with:
  • Actual calendar expiry dates (real Fridays, not "21d")
  • Strike prices derived from live UVXY price per strategy rules
  • BS-estimated premium for each leg
  • entry_percentile threshold enforcement
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict

from .black_scholes import call_price, strike_from_pct_otm, find_strike_for_delta
from .market_data import (
    next_friday_from_target,
    monthly_expiry_from_target,
    format_expiry,
    dte_from_expiry,
    REGIME_MAP,
    fetch_live_option_price,
    executable_price,
    fill_quality_score,
    liquidity_label,
)

# ---------------------------------------------------------------------------
# Strategy catalogue
# ---------------------------------------------------------------------------
@dataclass
class StrategySpec:
    id: str
    name: str
    description: str
    active_regimes: List[str]
    entry_pct_min: float          # enter only when pct >= this
    entry_pct_max: float          # enter only when pct <= this (100 = no cap)
    long_dte_target: int          # target DTE for LEAP / long leg
    short_dte_target: int         # target DTE for weekly / short leg
    long_otm_pct: float           # e.g. 0.0 = ATM, 0.05 = 5% OTM, -0.05 = 5% ITM
    short_otm_pct: float          # short leg strike OTM %
    long_monthly: bool = True     # True → use monthly expiry for long leg
    short_monthly: bool = False   # False → use nearest weekly Friday


STRATEGIES: Dict[str, StrategySpec] = {
    "V1": StrategySpec(
        id="V1",
        name="V1_INCOME_HARVESTER",
        description="Core theta collection — ATM LEAP, far-OTM weekly short",
        active_regimes=["CALM", "DECLINING"],
        entry_pct_min=0,
        entry_pct_max=40,
        long_dte_target=90,
        short_dte_target=7,
        long_otm_pct=0.0,
        short_otm_pct=0.30,
        long_monthly=True,
        short_monthly=False,
    ),
    "V2": StrategySpec(
        id="V2",
        name="V2_MEAN_REVERSION",
        description="Vol pullback after spike — moderate OTM diagonal",
        active_regimes=["DECLINING", "RISING"],
        entry_pct_min=30,
        entry_pct_max=65,
        long_dte_target=60,
        short_dte_target=14,
        long_otm_pct=0.05,
        short_otm_pct=0.25,
        long_monthly=True,
        short_monthly=False,
    ),
    "V3": StrategySpec(
        id="V3",
        name="V3_SHOCK_ABSORBER",
        description="Crisis-mode — slightly ITM long, wide short strike",
        active_regimes=["RISING", "STRESSED", "EXTREME"],
        entry_pct_min=45,
        entry_pct_max=100,
        long_dte_target=120,
        short_dte_target=21,
        long_otm_pct=-0.05,
        short_otm_pct=0.35,
        long_monthly=True,
        short_monthly=False,
    ),
    "V4": StrategySpec(
        id="V4",
        name="V4_TAIL_HUNTER",
        description="Directional extreme — ATM long, 30% OTM short",
        active_regimes=["EXTREME"],
        entry_pct_min=75,
        entry_pct_max=100,
        long_dte_target=90,
        short_dte_target=21,
        long_otm_pct=0.0,
        short_otm_pct=0.30,
        long_monthly=True,
        short_monthly=False,
    ),
    "V5": StrategySpec(
        id="V5",
        name="V5_REGIME_ALLOCATOR",
        description="Meta-strategy — adapts allocation dynamically across regimes",
        active_regimes=["CALM", "DECLINING", "RISING", "STRESSED", "EXTREME"],
        entry_pct_min=0,
        entry_pct_max=100,
        long_dte_target=90,
        short_dte_target=14,
        long_otm_pct=0.0,
        short_otm_pct=0.25,
        long_monthly=True,
        short_monthly=False,
    ),
}


# ---------------------------------------------------------------------------
# Signal output dataclass
# ---------------------------------------------------------------------------
@dataclass
class DiagonalSignal:
    strategy_id: str
    strategy_name: str
    regime: str
    percentile: float
    uvxy_price: float
    iv_est: float
    vix: float

    # Long leg
    long_strike: float
    long_expiry: date
    long_dte: int
    long_premium: float

    # Short leg
    short_strike: float
    short_expiry: date
    short_dte: int
    short_premium: float

    # Signal metadata
    net_debit: float
    is_entry_eligible: bool     # percentile within entry window
    wait_reason: Optional[str] = None

    # Executable pricing (from live chain; None = BS fallback only)
    # Short leg: you SELL → receive bid-side price
    short_bid: Optional[float] = None
    short_ask: Optional[float] = None
    short_executable: Optional[float] = None   # bid + 0.25*(ask-bid)
    # Long leg: you BUY → pay ask-side price
    long_bid: Optional[float] = None
    long_ask: Optional[float] = None
    long_executable: Optional[float] = None    # ask - 0.25*(ask-bid)

    # Liquidity metadata (from live chain)
    short_volume: Optional[int] = None
    short_oi: Optional[int] = None
    short_fill_quality: int = 0        # 0-100
    liquidity_warning: bool = False    # True when score < 45

    # Flags
    collapse_flag: bool = False
    spike_exhaustion_score: Optional[float] = None


    # ----------------------------------------------------------------
    # Display helpers  ← KEY FIX: always show actual dates
    # ----------------------------------------------------------------
    def long_leg_str(self) -> str:
        """e.g.  'Buy 44.5C  Jun 20  (~$8.94)'"""
        return (
            f"Buy {self.long_strike:.1f}C  "
            f"{format_expiry(self.long_expiry)}  "
            f"(~${self.long_premium:.2f})"
        )

    def short_leg_str(self) -> str:
        """e.g.  'Sell 58.0C  Apr 07  (~$0.85)'"""
        return (
            f"Sell {self.short_strike:.1f}C  "
            f"{format_expiry(self.short_expiry)}  "
            f"(~${self.short_premium:.2f})"
        )

    def summary_line(self) -> str:
        return (
            f"{self.long_leg_str()}  /  {self.short_leg_str()}  "
            f"| Net debit ~${self.net_debit:.2f}"
        )

    def long_dte_display(self) -> str:
        """'Jun 20 (90d)'  — actual date + DTE in parens."""
        return f"{format_expiry(self.long_expiry)} ({self.long_dte}d)"

    def short_dte_display(self) -> str:
        """'Apr 07 (21d)'"""
        return f"{format_expiry(self.short_expiry)} ({self.short_dte}d)"


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.05   # approx Fed funds


def generate_signal(
    spec: StrategySpec,
    uvxy_price: float,
    iv_est: float,
    vix: float,
    percentile: float,
    regime: str,
    collapse_flag: bool = False,
    spike_exhaustion_score: Optional[float] = None,
    today: Optional[date] = None,
) -> DiagonalSignal:
    """
    Build a DiagonalSignal with real strike prices and real expiry dates.
    """
    if today is None:
        today = date.today()

    r = RISK_FREE_RATE
    S = uvxy_price
    sigma = iv_est

    # ── Expiry dates ──────────────────────────────────────────────────────
    if spec.long_monthly:
        long_expiry = monthly_expiry_from_target(today, spec.long_dte_target)
    else:
        long_expiry = next_friday_from_target(today, spec.long_dte_target)

    if spec.short_monthly:
        short_expiry = monthly_expiry_from_target(today, spec.short_dte_target)
    else:
        short_expiry = next_friday_from_target(today, spec.short_dte_target)

    long_dte  = dte_from_expiry(long_expiry, today)
    short_dte = dte_from_expiry(short_expiry, today)

    T_long  = long_dte  / 365.0
    T_short = short_dte / 365.0

    # ── Strike prices ──────────────────────────────────────────────────────
    long_strike  = strike_from_pct_otm(S, spec.long_otm_pct)
    short_strike = strike_from_pct_otm(S, spec.short_otm_pct)
    # Safety: short must be ≥ long
    if short_strike < long_strike:
        short_strike = long_strike + 1.0

    # ── Option premiums: live chain first, BS fallback ────────────────────
    # Black-Scholes theoretical MID (used as fallback / sanity check)
    bs_long  = call_price(S, long_strike,  T_long,  r, sigma)
    bs_short = call_price(S, short_strike, T_short, r, sigma)

    # Attempt live bid/ask from yfinance option chain
    short_live = fetch_live_option_price("UVXY", short_expiry, short_strike, "call")
    long_live  = fetch_live_option_price("UVXY", long_expiry,  long_strike,  "call")

    # Executable prices:
    #   short leg (SELL) → bid + 25% of spread  (conservative, not mid)
    #   long leg  (BUY)  → ask - 25% of spread  (conservative, not mid)
    short_exec = executable_price(
        short_live["bid"], short_live["ask"], side="sell", aggressiveness=0.25
    ) if short_live["found"] else None

    long_exec = executable_price(
        long_live["bid"], long_live["ask"], side="buy", aggressiveness=0.25
    ) if long_live["found"] else None

    # Use live executable price if available; fall back to haircutted BS.
    # BS mid overestimates short credit by 2-5x on thin UVXY options.
    # 0.55x: bid-side approximation for 40-50% spread instruments.
    # 1.05x on long: you pay ask, which is slightly above mid.
    BS_SHORT_HAIRCUT = 0.55
    BS_LONG_MARKUP   = 1.05
    short_premium = short_exec if short_exec is not None else round(bs_short * BS_SHORT_HAIRCUT, 2)
    long_premium  = long_exec  if long_exec  is not None else round(bs_long  * BS_LONG_MARKUP,   2)
    net_debit = long_premium - short_premium

    # Fill quality (short leg is the one that matters for credit received)
    fq_score = fill_quality_score(
        short_live["bid"], short_live["ask"],
        short_live["volume"], short_live["open_interest"],
    ) if short_live["found"] else 0
    liq_warn = (fq_score < 45) or (not short_live["found"])

    # ── Entry eligibility ─────────────────────────────────────────────────
    is_eligible = (
        regime in spec.active_regimes
        and spec.entry_pct_min <= percentile <= spec.entry_pct_max
    )
    wait_reason = None
    if not is_eligible:
        if regime not in spec.active_regimes:
            wait_reason = f"Regime {regime} not in {spec.active_regimes}"
        elif percentile < spec.entry_pct_min:
            wait_reason = f"Percentile {percentile:.0f}% below entry min {spec.entry_pct_min:.0f}%"
        else:
            wait_reason = f"Percentile {percentile:.0f}% above entry max {spec.entry_pct_max:.0f}%"

    if collapse_flag and spec.id == "V4":
        wait_reason = (wait_reason or "") + " | Collapse flag — wait 1 session"

    return DiagonalSignal(
        strategy_id=spec.id,
        strategy_name=spec.name,
        regime=regime,
        percentile=percentile,
        uvxy_price=S,
        iv_est=sigma,
        vix=vix,
        long_strike=long_strike,
        long_expiry=long_expiry,
        long_dte=long_dte,
        long_premium=round(long_premium, 2),
        short_strike=short_strike,
        short_expiry=short_expiry,
        short_dte=short_dte,
        short_premium=round(short_premium, 2),
        net_debit=round(net_debit, 2),
        is_entry_eligible=is_eligible,
        wait_reason=wait_reason,
        # Executable pricing fields
        short_bid=short_live.get("bid") if short_live["found"] else None,
        short_ask=short_live.get("ask") if short_live["found"] else None,
        short_executable=short_exec,
        long_bid=long_live.get("bid")   if long_live["found"]  else None,
        long_ask=long_live.get("ask")   if long_live["found"]  else None,
        long_executable=long_exec,
        short_volume=short_live.get("volume")        if short_live["found"] else None,
        short_oi=short_live.get("open_interest")     if short_live["found"] else None,
        short_fill_quality=fq_score,
        liquidity_warning=liq_warn,
        collapse_flag=collapse_flag,
        spike_exhaustion_score=spike_exhaustion_score,
    )


def generate_all_signals(
    uvxy_price: float,
    iv_est: float,
    vix: float,
    percentile: float,
    regime: str,
    collapse_flag: bool = False,
    spike_exhaustion_score: Optional[float] = None,
    today: Optional[date] = None,
) -> Dict[str, DiagonalSignal]:
    return {
        sid: generate_signal(
            spec, uvxy_price, iv_est, vix, percentile, regime,
            collapse_flag, spike_exhaustion_score, today,
        )
        for sid, spec in STRATEGIES.items()
    }
