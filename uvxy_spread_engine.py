#!/usr/bin/env python3
"""
uvxy_spread_engine.py
─────────────────────
Core strategy brain for the UVXY Weekly Credit Spread System.

Design principles:
  - Trend-independent: no market direction prediction
  - Defined risk: always buy $2 cap above short call
  - Momentum-adaptive: DTE and size adjust to UVXY 5-day momentum
  - Spike-aware: sell long legs on confirmed reversal (not peak prediction)
  - Away-time safe: fully automatable, no human decisions required
  - Retirement-grade: consistent income, survivable worst case

Target: $100k/year gross on $300k account, 30 contracts
Max loss any week: $6,000 (2% of account)
Max loss any historical spike: $24,000 (8% drawdown, COVID-scale)

This file contains ONLY decision logic — no broker API calls,
no file I/O beyond state persistence, no Streamlit.
Interfaces (dashboard, executor) import from here.

Author: VIX 5W Suite
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

# ── Storage ───────────────────────────────────────────────────────────────────

STORAGE_DIR   = Path.home() / ".vix_suite"
ENGINE_STATE  = STORAGE_DIR / "spread_engine_state.json"
ENGINE_LOG    = STORAGE_DIR / "spread_engine.log"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Enums ─────────────────────────────────────────────────────────────────────

class MarketState(str, Enum):
    CALM      = "CALM"        # UVXY stable, 5d momentum flat
    RISING    = "RISING"      # UVXY rising moderately (+5–15%)
    SPIKING   = "SPIKING"     # UVXY spiking fast (>+15%)
    FADING    = "FADING"      # Spike confirmed fading (momentum turned negative)
    FALLING   = "FALLING"     # UVXY falling (post-spike or calm decline)

class PositionStatus(str, Enum):
    OPEN      = "OPEN"
    CLOSED    = "CLOSED"
    EXPIRED   = "EXPIRED"
    ROLLED    = "ROLLED"
    ASSIGNED  = "ASSIGNED"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    """
    All tunable parameters in one place.
    Change here — everything else adapts.
    """
    # Account
    account_size:           float = 300_000.0
    target_annual_income:   float = 100_000.0

    # Spread structure
    spread_width:           float = 2.00      # $2 between short and long cap
    min_net_credit:         float = 0.30      # minimum credit after cap cost
    cap_cost_pct:           float = 0.22      # long cap costs ~22% of short credit

    # Position sizing by state
    contracts_calm:         int   = 30
    contracts_rising:       int   = 20
    contracts_spiking:      int   = 10
    contracts_fading:       int   = 20
    contracts_falling:      int   = 30
    max_total_contracts:    int   = 30        # hard cap across all open positions

    # DTE by state
    dte_calm:               int   = 7
    dte_rising:             int   = 14
    dte_spiking:            int   = 21
    dte_fading:             int   = 14
    dte_falling:            int   = 7

    # Strike distance minimums (% above UVXY price)
    strike_pct_calm:        float = 0.08      # 8% OTM
    strike_pct_rising:      float = 0.10      # 10% OTM
    strike_pct_spiking:     float = 0.15      # 15% OTM
    strike_pct_fading:      float = 0.10      # 10% OTM
    strike_pct_falling:     float = 0.08      # 8% OTM

    # Momentum thresholds (5-day UVXY % change)
    momentum_rising_threshold:  float = 0.05   # >5% = RISING
    momentum_spiking_threshold: float = 0.15   # >15% = SPIKING
    momentum_fading_threshold:  float = -0.03  # <-3% after spike = FADING
    momentum_falling_threshold: float = -0.03  # <-3% in calm = FALLING

    # Exit rules
    profit_target_pct:      float = 0.80      # close at 80% profit
    stop_loss_multiplier:   float = 2.00      # close if spread reaches 2× credit
    expensive_btc_fixed:    float = 1.50      # BTC is "expensive" above $1.50
    expensive_btc_pct:      float = 0.50      # or >50% of original credit

    # Roll rules
    max_rolls:              int   = 2         # max rolls per position
    roll_dte_threshold:     int   = 2         # roll if DTE ≤ 2 and ITM
    roll_new_dte:           int   = 21        # roll to 21 DTE

    # Long leg rules
    long_leg_sell_value_multiple: float = 2.0  # sell if value > 2× entry cost
    long_leg_sell_momentum_threshold: float = -0.05  # AND momentum < -5%

    # Minimum credit (absolute, not per-contract)
    min_weekly_credit_total: float = 0.30     # skip week if < $0.30 net credit

    # Reserve cash (smooth income during calm periods)
    cash_reserve_months:    int   = 3         # keep 3 months income in reserve

    def max_loss_per_week(self) -> float:
        return self.spread_width * 100 * self.contracts_calm

    def annual_income_estimate(self, contracts: int = None) -> float:
        c = contracts or self.contracts_calm
        # Weighted average by regime weeks
        calm_wks     = 8
        normal_wks   = 26
        stressed_wks = 14
        extreme_wks  = 4
        avg_credit   = (0.40*calm_wks + 0.70*normal_wks +
                        1.20*stressed_wks + 0.0*extreme_wks) / 52
        return avg_credit * 100 * c * (52 - extreme_wks)


# ── Market state detection ────────────────────────────────────────────────────

@dataclass
class MomentumReading:
    uvxy_now:       float
    uvxy_5d_ago:    float
    momentum_5d:    float   # % change
    was_spiking:    bool    # was in SPIKING state last week
    spike_peak:     Optional[float] = None

    @classmethod
    def compute(cls, uvxy_now: float, uvxy_5d_ago: float,
                was_spiking: bool = False,
                spike_peak: Optional[float] = None) -> "MomentumReading":
        if uvxy_5d_ago and uvxy_5d_ago > 0:
            mom = (uvxy_now - uvxy_5d_ago) / uvxy_5d_ago
        else:
            mom = 0.0
        return cls(
            uvxy_now=uvxy_now,
            uvxy_5d_ago=uvxy_5d_ago,
            momentum_5d=mom,
            was_spiking=was_spiking,
            spike_peak=spike_peak,
        )


def detect_market_state(reading: MomentumReading,
                        cfg: EngineConfig) -> MarketState:
    """
    Pure function — no side effects.
    Determines current market state from momentum reading.

    State machine:
      CALM    → RISING   when momentum > 5%
      RISING  → SPIKING  when momentum > 15%
      SPIKING → FADING   when momentum turns negative AND was_spiking
      FADING  → FALLING  when momentum < -5% (reversal confirmed)
      FALLING → CALM     when momentum flattens (< 5% either way)

    Key design: FADING only triggers from SPIKING state.
    A -3% day during a calm market is just FALLING, not FADING.
    """
    m = reading.momentum_5d

    # Spike fading — the harvest trigger
    # Only fires if we were previously in SPIKING state
    if reading.was_spiking and m < cfg.momentum_fading_threshold:
        return MarketState.FADING

    # Active spike
    if m > cfg.momentum_spiking_threshold:
        return MarketState.SPIKING

    # Rising moderately
    if m > cfg.momentum_rising_threshold:
        return MarketState.RISING

    # Falling (normal or post-spike without FADING trigger)
    if m < cfg.momentum_falling_threshold:
        return MarketState.FALLING

    # Default: calm
    return MarketState.CALM


# ── Weekly decision ───────────────────────────────────────────────────────────

@dataclass
class WeeklyDecision:
    """
    The complete answer to "what do I do this week?"
    Used by both dashboard (display) and executor (automation).
    """
    date:               str
    uvxy_price:         float
    market_state:       MarketState
    momentum_5d:        float

    # Short call to sell
    action:             str           # "SELL_SPREAD" / "SKIP" / "SELL_LONG_LEG"
    short_strike:       Optional[float] = None
    long_cap_strike:    Optional[float] = None
    expiry_date:        Optional[str]   = None
    dte:                int            = 0
    contracts:          int            = 0
    net_credit_target:  float          = 0.0
    gross_credit_est:   float          = 0.0
    cap_cost_est:       float          = 0.0

    # Long leg action (if applicable)
    sell_long_leg:      bool           = False
    long_leg_reason:    str            = ""

    # Roll action (if applicable)
    roll_needed:        bool           = False
    roll_position_id:   Optional[str]  = None

    # Human-readable summary
    reason:             str            = ""
    alerts:             list           = field(default_factory=list)
    income_this_week:   float          = 0.0


def make_weekly_decision(
    uvxy_now:       float,
    uvxy_5d_ago:    float,
    expirations:    list[date],
    option_chain:   list[dict],
    state:          "EngineState",
    cfg:            EngineConfig,
    today:          Optional[date] = None,
) -> WeeklyDecision:
    """
    Core decision function. Pure logic — no API calls.

    Args:
        uvxy_now:     Current UVXY price
        uvxy_5d_ago:  UVXY price 5 trading days ago
        expirations:  Available expiry dates from broker
        option_chain: List of option dicts {strike, bid, ask, delta, expiry}
        state:        Current engine state (positions, history)
        cfg:          Configuration
        today:        Date override for testing

    Returns:
        WeeklyDecision with complete instructions
    """
    today     = today or date.today()
    today_str = str(today)

    # 1. Compute momentum
    reading = MomentumReading.compute(
        uvxy_now    = uvxy_now,
        uvxy_5d_ago = uvxy_5d_ago,
        was_spiking = state.was_spiking,
        spike_peak  = state.spike_peak,
    )
    mkt_state = detect_market_state(reading, cfg)

    decision = WeeklyDecision(
        date         = today_str,
        uvxy_price   = uvxy_now,
        market_state = mkt_state,
        momentum_5d  = reading.momentum_5d,
        action       = "SKIP",
        alerts       = [],
    )

    # 2. Check long leg sell trigger
    long_sell = should_sell_long_leg(reading, state, cfg)
    if long_sell:
        decision.sell_long_leg  = True
        decision.long_leg_reason = long_sell
        decision.alerts.append(f"🎯 SELL LONG LEG: {long_sell}")

    # 3. Determine target DTE and contracts by state
    dte_map       = {
        MarketState.CALM:    cfg.dte_calm,
        MarketState.RISING:  cfg.dte_rising,
        MarketState.SPIKING: cfg.dte_spiking,
        MarketState.FADING:  cfg.dte_fading,
        MarketState.FALLING: cfg.dte_falling,
    }
    contracts_map = {
        MarketState.CALM:    cfg.contracts_calm,
        MarketState.RISING:  cfg.contracts_rising,
        MarketState.SPIKING: cfg.contracts_spiking,
        MarketState.FADING:  cfg.contracts_fading,
        MarketState.FALLING: cfg.contracts_falling,
    }

    target_dte       = dte_map[mkt_state]
    target_contracts = contracts_map[mkt_state]

    # 4. Check total open contract limit
    open_contracts = state.total_open_contracts()
    available      = max(0, cfg.max_total_contracts - open_contracts)
    target_contracts = min(target_contracts, available)

    if target_contracts == 0:
        decision.action = "SKIP"
        decision.reason = (f"At max open contracts ({open_contracts}/{cfg.max_total_contracts})"
                           f" — wait for positions to close")
        return decision

    # 5. Find target expiry
    min_date = today + timedelta(days=target_dte - 2)
    max_date = today + timedelta(days=target_dte + 5)
    valid_expiries = [e for e in expirations
                      if min_date <= e <= max_date and e.weekday() == 4]
    if not valid_expiries:
        # Relax to ±7 days
        valid_expiries = [e for e in expirations
                          if today + timedelta(days=target_dte-7) <= e
                          and e <= today + timedelta(days=target_dte+7)]
    if not valid_expiries:
        decision.action = "SKIP"
        decision.reason = f"No valid expiry found near {target_dte} DTE"
        return decision

    target_expiry = valid_expiries[0]
    actual_dte    = (target_expiry - today).days

    # 6. Find short strike
    min_strike_pct = {
        MarketState.CALM:    cfg.strike_pct_calm,
        MarketState.RISING:  cfg.strike_pct_rising,
        MarketState.SPIKING: cfg.strike_pct_spiking,
        MarketState.FADING:  cfg.strike_pct_fading,
        MarketState.FALLING: cfg.strike_pct_falling,
    }[mkt_state]

    min_strike = uvxy_now * (1 + min_strike_pct)
    best_short = find_short_strike(
        chain      = option_chain,
        expiry     = target_expiry,
        min_strike = min_strike,
        min_bid    = cfg.min_net_credit / (1 - cfg.cap_cost_pct),
    )

    if not best_short:
        decision.action = "SKIP"
        decision.reason = (f"No valid short strike found ≥ ${min_strike:.1f} "
                           f"({min_strike_pct:.0%} OTM minimum) for {target_expiry}")
        decision.alerts.append(
            f"⚠️ No strike found at {mkt_state.value} regime — "
            f"UVXY ${uvxy_now:.2f}, need strike ≥ ${min_strike:.1f}")
        return decision

    # 7. Calculate net credit
    short_bid  = best_short["bid"]
    cap_cost   = short_bid * cfg.cap_cost_pct
    net_credit = short_bid - cap_cost

    if net_credit < cfg.min_net_credit:
        decision.action = "SKIP"
        decision.reason = (f"Net credit ${net_credit:.2f} below minimum "
                           f"${cfg.min_net_credit:.2f} after cap cost")
        return decision

    # 8. Long cap strike
    cap_strike = best_short["strike"] + cfg.spread_width

    # 9. Build decision
    decision.action           = "SELL_SPREAD"
    decision.short_strike     = best_short["strike"]
    decision.long_cap_strike  = cap_strike
    decision.expiry_date      = str(target_expiry)
    decision.dte              = actual_dte
    decision.contracts        = target_contracts
    decision.gross_credit_est = round(short_bid, 2)
    decision.cap_cost_est     = round(cap_cost, 2)
    decision.net_credit_target = round(net_credit, 2)
    decision.income_this_week  = round(net_credit * 100 * target_contracts, 2)

    decision.reason = (
        f"{mkt_state.value} regime | momentum {reading.momentum_5d:+.1%} | "
        f"${best_short['strike']:.1f}C/{cap_strike:.1f}C {target_expiry} | "
        f"{target_contracts} contracts × ${net_credit:.2f} = "
        f"${decision.income_this_week:,.0f}"
    )

    return decision


# ── Long leg sell trigger ─────────────────────────────────────────────────────

def should_sell_long_leg(reading: MomentumReading,
                         state: "EngineState",
                         cfg: EngineConfig) -> Optional[str]:
    """
    Returns reason string if long leg should be sold, None otherwise.

    Two conditions BOTH must be true:
    1. Long leg value > 2× entry cost (profitable enough to sell)
    2. Momentum has turned negative (confirmed reversal, not peak prediction)

    This avoids selling at the peak (unknowable) and instead
    sells on confirmed reversal with a 1-2 day lag.
    """
    lg = state.long_leg
    if not lg:
        return None

    entry_cost  = lg.get("entry_cost", 0)
    current_val = lg.get("current_value", 0)

    if entry_cost <= 0:
        return None

    value_multiple = current_val / entry_cost if entry_cost > 0 else 0

    condition_1 = value_multiple >= cfg.long_leg_sell_value_multiple
    condition_2 = reading.momentum_5d <= cfg.long_leg_sell_momentum_threshold

    if condition_1 and condition_2:
        return (f"Value ${current_val:.2f} = {value_multiple:.1f}× entry cost "
                f"${entry_cost:.2f} | momentum {reading.momentum_5d:+.1%} "
                f"(confirmed reversal)")

    if condition_1 and not condition_2:
        # Value is there but spike hasn't confirmed reversal yet
        return None  # Wait for confirmation

    return None


# ── Roll decision ─────────────────────────────────────────────────────────────

@dataclass
class RollDecision:
    should_roll:    bool
    reason:         str
    action:         str      # "BTC_AND_ROLL" / "BTC_ONLY" / "HOLD"
    new_dte:        int      = 21
    roll_count:     int      = 0


def evaluate_roll(position: "SpreadPosition",
                  uvxy_now: float,
                  cfg:      EngineConfig,
                  today:    Optional[date] = None) -> RollDecision:
    """
    Determines if an open spread position needs to be rolled.

    Roll triggers (in priority order):
    1. DTE ≤ 2 AND short is ITM → assignment risk → BTC and roll
    2. Spread value > expensive threshold → close, deploy elsewhere
    3. Max rolls reached → close regardless

    Hold triggers:
    - Spread losing but still has DTE → trust the time, HOLD
    - Structural decay working in our favor → HOLD
    """
    today    = today or date.today()
    exp_date = date.fromisoformat(str(position.expiry_date))
    dte      = (exp_date - today).days

    # Rule: max rolls exceeded — close regardless
    if position.roll_count >= cfg.max_rolls:
        return RollDecision(
            should_roll = True,
            reason      = f"Max rolls ({cfg.max_rolls}) reached — closing position",
            action      = "BTC_ONLY",
            roll_count  = position.roll_count,
        )

    # Rule: DTE ≤ 2 AND ITM → assignment risk
    short_is_itm = uvxy_now >= position.short_strike
    if dte <= cfg.roll_dte_threshold and short_is_itm:
        return RollDecision(
            should_roll = True,
            reason      = (f"DTE={dte} ≤ {cfg.roll_dte_threshold} and ITM "
                           f"(UVXY ${uvxy_now:.2f} ≥ ${position.short_strike:.2f}) "
                           f"— assignment risk"),
            action      = "BTC_AND_ROLL",
            new_dte     = cfg.roll_new_dte,
            roll_count  = position.roll_count + 1,
        )

    # Rule: spread cost > expensive threshold → evaluate close
    if position.current_spread_value > 0:
        btc_expensive_fixed = position.current_spread_value > cfg.expensive_btc_fixed
        if position.entry_credit > 0:
            btc_pct = position.current_spread_value / position.entry_credit
            btc_expensive_pct = btc_pct > cfg.stop_loss_multiplier
        else:
            btc_expensive_pct = False

        if btc_expensive_fixed and btc_expensive_pct:
            return RollDecision(
                should_roll = True,
                reason      = (f"Spread at ${position.current_spread_value:.2f} "
                               f"= {btc_pct:.0%} of credit — "
                               f"exceeds both thresholds → roll to {cfg.roll_new_dte} DTE"),
                action      = "BTC_AND_ROLL",
                new_dte     = cfg.roll_new_dte,
                roll_count  = position.roll_count + 1,
            )

    # Hold — structural decay is working
    if dte > cfg.roll_dte_threshold:
        return RollDecision(
            should_roll = False,
            reason      = (f"DTE={dte} — holding. Structural decay works in our favor. "
                           f"UVXY ${uvxy_now:.2f} vs ${position.short_strike:.2f} strike."),
            action      = "HOLD",
            roll_count  = position.roll_count,
        )

    return RollDecision(
        should_roll = False,
        reason      = "No roll trigger — hold",
        action      = "HOLD",
        roll_count  = position.roll_count,
    )


# ── Exit decision ─────────────────────────────────────────────────────────────

def should_close_for_profit(position: "SpreadPosition",
                             cfg: EngineConfig) -> tuple[bool, str]:
    """
    Returns (True, reason) if position should be closed for profit target.
    80% profit = spread now worth 20% of original credit received.
    """
    if position.entry_credit <= 0:
        return False, "No entry credit recorded"

    current_cost_to_close = position.current_spread_value
    profit_pct = 1 - (current_cost_to_close / position.entry_credit)

    if profit_pct >= cfg.profit_target_pct:
        profit_dollars = (position.entry_credit - current_cost_to_close) * 100 * position.contracts
        return True, (f"Profit target reached: {profit_pct:.0%} ≥ {cfg.profit_target_pct:.0%} | "
                      f"close for ${profit_dollars:,.0f} profit")

    return False, f"Profit {profit_pct:.0%} — below {cfg.profit_target_pct:.0%} target"


# ── Strike finder ─────────────────────────────────────────────────────────────

def find_short_strike(chain:      list[dict],
                      expiry:     date,
                      min_strike: float,
                      min_bid:    float = 0.30) -> Optional[dict]:
    """
    Find the best short call strike from the option chain.

    Selection criteria:
    1. Call options only
    2. Strike ≥ min_strike (minimum OTM distance)
    3. Bid ≥ min_bid (minimum premium)
    4. Matching expiry
    5. Best = highest strike still meeting delta target (sell highest premium OTM)

    Returns dict with {strike, bid, ask, delta, symbol} or None.
    """
    expiry_str = str(expiry)
    candidates = []

    for opt in chain:
        if opt.get("option_type", "").lower() != "call":
            continue
        opt_exp = str(opt.get("expiry", opt.get("expiration_date", "")))
        if expiry_str not in opt_exp and opt_exp not in expiry_str:
            continue
        strike = float(opt.get("strike", 0))
        bid    = float(opt.get("bid", 0) or 0)
        ask    = float(opt.get("ask", 0) or 0)

        if strike < min_strike:
            continue
        if bid < min_bid:
            continue

        # Check liquidity — spread must not be absurdly wide
        if ask > 0 and bid > 0:
            mid    = (bid + ask) / 2
            spread = ask - bid
            if spread / mid > 0.80 and spread > 2.00:
                continue  # Too wide

        candidates.append({
            "strike": strike,
            "bid":    bid,
            "ask":    ask,
            "mid":    round((bid + ask) / 2, 2),
            "delta":  abs(float((opt.get("greeks") or {}).get("delta", 0) or 0)),
            "symbol": opt.get("symbol", ""),
        })

    if not candidates:
        return None

    # Sort by strike ascending — pick lowest valid strike
    # (closer to ATM = more premium = better income)
    candidates.sort(key=lambda x: x["strike"])
    return candidates[0]


# ── Income projection ─────────────────────────────────────────────────────────

@dataclass
class IncomeProjection:
    weekly_estimate:  float
    annual_estimate:  float
    monthly_estimate: float
    contracts:        int
    net_credit:       float
    market_state:     MarketState
    account_yield:    float   # % of account


def project_income(net_credit: float,
                   contracts:  int,
                   market_state: MarketState,
                   cfg: EngineConfig) -> IncomeProjection:
    weekly = net_credit * 100 * contracts
    # Annualize based on regime frequency
    regime_weeks = {
        MarketState.CALM:    8,
        MarketState.RISING:  10,
        MarketState.SPIKING: 4,
        MarketState.FADING:  4,
        MarketState.FALLING: 22,
    }
    active_weeks = sum(v for k, v in regime_weeks.items()
                       if k != MarketState.SPIKING)
    # Rough annual based on current rate × active weeks
    annual   = weekly * active_weeks
    monthly  = annual / 12
    yield_pct = annual / cfg.account_size

    return IncomeProjection(
        weekly_estimate  = round(weekly, 2),
        annual_estimate  = round(annual, 2),
        monthly_estimate = round(monthly, 2),
        contracts        = contracts,
        net_credit       = net_credit,
        market_state     = market_state,
        account_yield    = round(yield_pct, 4),
    )


# ── Position model ────────────────────────────────────────────────────────────

@dataclass
class SpreadPosition:
    """A single open credit spread position."""
    position_id:          str
    open_date:            str
    expiry_date:          str
    short_strike:         float
    long_cap_strike:      float
    contracts:            int
    entry_credit:         float    # net credit per contract received
    gross_credit:         float    # short call credit before cap cost
    cap_cost:             float    # cost of long cap per contract
    market_state_at_open: str
    dte_at_open:          int
    status:               str      = "OPEN"
    close_date:           Optional[str]   = None
    close_credit:         float          = 0.0   # cost to close per contract
    realized_pnl:         float          = 0.0
    roll_count:           int            = 0
    current_spread_value: float          = 0.0   # mark-to-market
    notes:                str            = ""

    def max_loss(self) -> float:
        """Maximum possible loss on this position."""
        return (self.long_cap_strike - self.short_strike - self.entry_credit) * 100 * self.contracts

    def max_profit(self) -> float:
        """Maximum profit if expires worthless."""
        return self.entry_credit * 100 * self.contracts

    def current_pnl(self) -> float:
        """Current mark-to-market P&L."""
        return (self.entry_credit - self.current_spread_value) * 100 * self.contracts

    def profit_pct(self) -> float:
        """Current profit as % of max profit."""
        if self.entry_credit <= 0:
            return 0.0
        return 1 - (self.current_spread_value / self.entry_credit)


@dataclass
class LongLegPosition:
    """Optional long call position used for spike harvest."""
    symbol:         str
    strike:         float
    expiry_date:    str
    contracts:      int
    entry_cost:     float    # per contract
    entry_date:     str
    current_value:  float    = 0.0
    sold:           bool     = False
    sell_date:      Optional[str]   = None
    sell_price:     float           = 0.0
    realized_pnl:   float           = 0.0


# ── Engine state ──────────────────────────────────────────────────────────────

@dataclass
class EngineState:
    """
    Persistent state — saved to disk between runs.
    Source of truth for the engine.
    """
    updated_at:     str             = ""
    positions:      list            = field(default_factory=list)
    long_leg:       Optional[dict]  = None
    was_spiking:    bool            = False
    spike_peak:     Optional[float] = None
    last_uvxy:      float           = 0.0
    last_5d_uvxy:   list            = field(default_factory=list)  # rolling 5-day window
    total_credits:  float           = 0.0
    total_realized: float           = 0.0
    week_count:     int             = 0

    def total_open_contracts(self) -> int:
        return sum(
            p.get("contracts", 0)
            for p in self.positions
            if p.get("status") == "OPEN"
        )

    def open_positions(self) -> list[dict]:
        return [p for p in self.positions if p.get("status") == "OPEN"]

    def update_uvxy_history(self, uvxy: float):
        """Maintain rolling 5-day window."""
        self.last_uvxy = uvxy
        self.last_5d_uvxy.append(uvxy)
        if len(self.last_5d_uvxy) > 5:
            self.last_5d_uvxy.pop(0)

    def get_5d_ago_uvxy(self) -> float:
        """Get UVXY price from 5 trading days ago."""
        if len(self.last_5d_uvxy) >= 5:
            return self.last_5d_uvxy[0]
        elif self.last_5d_uvxy:
            return self.last_5d_uvxy[0]
        return self.last_uvxy

    def update_market_state(self, state: MarketState):
        """Track whether we were in SPIKING state."""
        if state == MarketState.SPIKING:
            self.was_spiking = True
        elif state in (MarketState.CALM, MarketState.FALLING):
            self.was_spiking = False
            self.spike_peak  = None

    def record_spike_peak(self, uvxy: float):
        if self.was_spiking:
            if self.spike_peak is None or uvxy > self.spike_peak:
                self.spike_peak = uvxy


# ── State persistence ─────────────────────────────────────────────────────────

def load_state() -> EngineState:
    if not ENGINE_STATE.exists():
        return EngineState(updated_at=str(date.today()))
    try:
        data = json.loads(ENGINE_STATE.read_text())
        s = EngineState()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s
    except Exception as e:
        _log(f"State load error: {e} — starting fresh")
        return EngineState(updated_at=str(date.today()))


def save_state(state: EngineState):
    state.updated_at = str(date.today())
    ENGINE_STATE.write_text(
        json.dumps(asdict(state), indent=2, default=str)
    )


# ── Logger ────────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(ENGINE_LOG, "a") as f:
        f.write(line + "\n")


# ── Weekly summary ────────────────────────────────────────────────────────────

def weekly_summary(state: EngineState,
                   decision: WeeklyDecision,
                   cfg: EngineConfig) -> dict:
    """
    Human-readable weekly summary for email digest and dashboard.
    """
    open_pos  = state.open_positions()
    total_exp = sum(p.get("max_profit_est", 0) for p in open_pos)

    return {
        "date":             decision.date,
        "uvxy":             decision.uvxy_price,
        "market_state":     decision.market_state.value,
        "momentum":         f"{decision.momentum_5d:+.1%}",
        "action":           decision.action,
        "this_week_income": decision.income_this_week,
        "total_credits":    state.total_credits,
        "open_positions":   len(open_pos),
        "open_contracts":   state.total_open_contracts(),
        "max_contracts":    cfg.max_total_contracts,
        "sell_long_leg":    decision.sell_long_leg,
        "alerts":           decision.alerts,
        "reason":           decision.reason,
        "annual_pace":      round(state.total_credits / max(state.week_count, 1) * 48, 0),
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("UVXY Spread Engine — Self Test")
    print("=" * 60)

    cfg   = EngineConfig()
    state = EngineState()

    # Test market state detection
    # momentum = (now - 5d_ago) / 5d_ago
    # 28/26 = +7.7% → RISING (>5% threshold) ✓
    # 35/29 = +20.7% → SPIKING (>15% threshold) ✓
    # 55/35 = +57.1% → SPIKING ✓
    # 48/55 = -12.7% + was_spiking → FADING ✓
    # 25/28 = -10.7% → FALLING ✓
    # 28/27 = +3.7% → CALM (below 5% threshold) ✓
    tests = [
        (28.0, 27.0, False, None, MarketState.CALM),      # +3.7% → CALM
        (28.0, 26.0, False, None, MarketState.RISING),    # +7.7% → RISING
        (35.0, 29.0, False, None, MarketState.SPIKING),   # +20.7% → SPIKING
        (55.0, 35.0, False, None, MarketState.SPIKING),   # +57.1% → SPIKING
        (48.0, 55.0, True,  60.0, MarketState.FADING),   # -12.7% + was_spiking → FADING
        (25.0, 28.0, False, None, MarketState.FALLING),   # -10.7% → FALLING
    ]

    print("\nMarket state detection:")
    all_pass = True
    for uvxy_now, uvxy_5d, was_spiking, peak, expected in tests:
        reading = MomentumReading.compute(uvxy_now, uvxy_5d, was_spiking, peak)
        result  = detect_market_state(reading, cfg)
        status  = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"  {status} UVXY ${uvxy_now:.0f} / 5d ${uvxy_5d:.0f} "
              f"spiking={was_spiking} → {result.value} (expected {expected.value})")

    # Test income projection
    print(f"\nIncome projections:")
    for state_val, credit in [(MarketState.CALM, 0.40),
                               (MarketState.RISING, 0.70),
                               (MarketState.SPIKING, 1.20)]:
        proj = project_income(credit, cfg.contracts_calm, state_val, cfg)
        print(f"  {state_val.value:10} ${credit:.2f}/contract × {cfg.contracts_calm} = "
              f"${proj.weekly_estimate:,.0f}/week | ${proj.annual_estimate:,.0f}/yr | "
              f"{proj.account_yield:.1%} yield")

    # Test config
    print(f"\nConfig validation:")
    print(f"  Max loss/week:       ${cfg.max_loss_per_week():,.0f}")
    print(f"  Annual est (30c):    ${cfg.annual_income_estimate():,.0f}")
    print(f"  Max loss % account:  {cfg.max_loss_per_week()/cfg.account_size:.1%}")

    # Test state persistence
    print(f"\nState persistence:")
    test_state = EngineState(updated_at=str(date.today()))
    test_state.update_uvxy_history(28.73)
    save_state(test_state)
    loaded = load_state()
    match = loaded.last_uvxy == 28.73
    print(f"  {'✅' if match else '❌'} Save/load: last_uvxy={loaded.last_uvxy}")

    print(f"\n{'✅ All tests passed' if all_pass else '❌ Some tests failed'}")
    print(f"\nEngine ready. State at: {ENGINE_STATE}")
    print(f"Log at:                 {ENGINE_LOG}")
