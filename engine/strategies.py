"""
Strategy Configuration Engine
------------------------------
Defines the five VIX-regime–adaptive diagonal spread strategies and
their parameter sets.  Each strategy adapts its parameters based on
the current regime so you always have the *best* setup for the
current volatility environment.

Strategy roster
~~~~~~~~~~~~~~~
V1  Income Harvester   – Core carry engine; active all regimes
V2  Mean Reversion     – Activated when VIX is HIGH/EXTREME and FALLING
V3  Shock Absorber     – Wide strikes for HIGH/EXTREME; sells elevated premium
V4  Tail Hunter        – Directional: leans long vol in EXTREME regime
V5  Regime Allocator   – Meta-strategy: allocates across V1-V4 dynamically
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

from engine.regime import RegimeState


# ── Parameter dataclass ───────────────────────────────────────────────────────

@dataclass
class StrategyParams:
    # Identity
    code:        str
    name:        str
    emoji:       str

    # Active regimes
    active_regimes: List[str]

    # Long leg (LEAP)
    long_dte_min:    int    # minimum DTE at entry
    long_dte_target: int    # preferred DTE at entry
    long_moneyness:  float  # strike / spot  (1.0 = ATM, 0.95 = ITM)

    # Short leg (weekly / bi-weekly call)
    short_dte_min:   int
    short_dte_target:int
    short_moneyness: float  # 1.05 = 5% OTM

    # P&L management
    profit_target_pct: float   # close / roll short when premium decays this much
    stop_loss_pct:     float   # close position if loss exceeds this % of debit
    roll_at_dte:       int     # roll short leg when DTE ≤ this value

    # Sizing
    max_concurrent:    int     # max open positions for this strategy
    account_pct_per:   float   # % of account to risk per position (max loss basis)

    # Meta
    description: str = ""

    def is_active_in(self, regime: str, trend: str) -> bool:
        """Override this per-strategy for trend-sensitive logic."""
        return regime in self.active_regimes


# ── Strategy definitions ──────────────────────────────────────────────────────

_V1 = StrategyParams(
    code   = "V1",
    name   = "Income Harvester",
    emoji  = "💰",
    active_regimes   = ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"],
    long_dte_min     = 150,
    long_dte_target  = 210,
    long_moneyness   = 0.97,   # slightly ITM LEAP for lower extrinsic cost
    short_dte_min    = 5,
    short_dte_target = 10,
    short_moneyness  = 1.07,   # 7% OTM short call
    profit_target_pct= 0.50,   # take 50% of max profit on short
    stop_loss_pct    = 0.30,   # stop if position P&L = -30% of debit
    roll_at_dte      = 4,
    max_concurrent   = 4,
    account_pct_per  = 0.05,   # 5% of account per position
    description = (
        "Core theta-harvesting engine. Sells weekly calls on UVXY "
        "diagonal to collect time decay across all regimes. "
        "Roll the short leg when DTE ≤ 4 or 50% profit achieved."
    ),
)

_V2 = StrategyParams(
    code   = "V2",
    name   = "Mean Reversion",
    emoji  = "📉",
    active_regimes   = ["HIGH", "EXTREME"],
    long_dte_min     = 120,
    long_dte_target  = 180,
    long_moneyness   = 1.00,   # ATM LEAP to capture reversion
    short_dte_min    = 7,
    short_dte_target = 14,
    short_moneyness  = 1.12,   # sell further OTM during spikes for wider cushion
    profit_target_pct= 0.60,
    stop_loss_pct    = 0.40,
    roll_at_dte      = 5,
    max_concurrent   = 2,
    account_pct_per  = 0.04,
    description = (
        "Activates when VIX is elevated and trending down. "
        "Wider OTM short call benefits from vol crush post-spike. "
        "Only enter on HIGH or EXTREME with FALLING trend."
    ),
)

_V3 = StrategyParams(
    code   = "V3",
    name   = "Shock Absorber",
    emoji  = "🛡️",
    active_regimes   = ["HIGH", "EXTREME"],
    long_dte_min     = 90,
    long_dte_target  = 150,
    long_moneyness   = 0.95,   # deep ITM LEAP = high delta, acts like stock surrogate
    short_dte_min    = 7,
    short_dte_target = 21,
    short_moneyness  = 1.20,   # very far OTM short during crisis
    profit_target_pct= 0.40,
    stop_loss_pct    = 0.50,   # wider stop; spikes can overshoot
    roll_at_dte      = 7,
    max_concurrent   = 2,
    account_pct_per  = 0.06,
    description = (
        "Crisis mode. Deep ITM LEAP + far OTM short to harvest "
        "premium elevated to 150-200% IV without getting short squeezed. "
        "Use wider stop because spikes can extend before reversing."
    ),
)

_V4 = StrategyParams(
    code   = "V4",
    name   = "Tail Hunter",
    emoji  = "🎯",
    active_regimes   = ["EXTREME"],
    long_dte_min     = 60,
    long_dte_target  = 90,
    long_moneyness   = 1.00,   # ATM
    short_dte_min    = 14,
    short_dte_target = 21,
    short_moneyness  = 1.30,   # sell very far OTM → position benefits from continued rally
    profit_target_pct= 0.75,
    stop_loss_pct    = 0.60,
    roll_at_dte      = 7,
    max_concurrent   = 1,
    account_pct_per  = 0.03,   # small size; this is a speculative overlay
    description = (
        "EXTREME regime only. Rare setup that runs directional long "
        "delta exposure into sustained vol spikes. "
        "Sell very-far OTM calls to cap max loss; "
        "profits if UVXY continues to rally hard."
    ),
)

_V5 = StrategyParams(
    code   = "V5",
    name   = "Regime Allocator",
    emoji  = "🔄",
    active_regimes   = ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "EXTREME"],
    long_dte_min     = 120,
    long_dte_target  = 210,
    long_moneyness   = 0.98,
    short_dte_min    = 7,
    short_dte_target = 14,
    short_moneyness  = 1.10,
    profit_target_pct= 0.55,
    stop_loss_pct    = 0.35,
    roll_at_dte      = 5,
    max_concurrent   = 3,
    account_pct_per  = 0.05,
    description = (
        "Meta-strategy. Dynamically adjusts OTM offset, DTE, and "
        "position size based on the current regime. "
        "Essentially a smarter V1 that changes its character as "
        "the market transitions through regimes."
    ),
)


# ── Registry ──────────────────────────────────────────────────────────────────

STRATEGIES: Dict[str, StrategyParams] = {
    "V1": _V1,
    "V2": _V2,
    "V3": _V3,
    "V4": _V4,
    "V5": _V5,
}


def get_active_strategies(regime_state: RegimeState) -> List[StrategyParams]:
    """Return strategies that are active given current regime + trend."""
    active = []
    for s in STRATEGIES.values():
        if s.is_active_in(regime_state.regime, regime_state.vix_trend):
            # V2 also requires FALLING trend
            if s.code == "V2" and regime_state.vix_trend not in ("FALLING", "STABLE"):
                continue
            active.append(s)
    return active


# ── Regime-adaptive parameter adjustment ──────────────────────────────────────

def adapt_params(base: StrategyParams, regime: str) -> StrategyParams:
    """
    Return an adjusted copy of `base` params for the current regime.
    Parameters shift to be more conservative in EXTREME and more
    aggressive in ULTRA_LOW / LOW.
    """
    import copy
    p = copy.copy(base)

    if regime == "ULTRA_LOW":
        p.short_moneyness   = max(base.short_moneyness - 0.02, 1.03)  # tighter OTM ok
        p.profit_target_pct = min(base.profit_target_pct + 0.10, 0.70)
        p.stop_loss_pct     = max(base.stop_loss_pct - 0.05, 0.15)
        p.account_pct_per   = min(base.account_pct_per * 1.20, 0.10)  # can size up

    elif regime == "LOW":
        pass  # base params are calibrated for LOW

    elif regime == "MEDIUM":
        p.short_moneyness   = base.short_moneyness + 0.01  # slightly further OTM

    elif regime == "HIGH":
        p.short_moneyness   = base.short_moneyness + 0.04  # wider OTM for cushion
        p.stop_loss_pct     = min(base.stop_loss_pct + 0.10, 0.60)
        p.account_pct_per   = base.account_pct_per * 0.80  # size down

    elif regime == "EXTREME":
        p.short_moneyness   = base.short_moneyness + 0.08  # very wide OTM
        p.stop_loss_pct     = min(base.stop_loss_pct + 0.20, 0.70)
        p.account_pct_per   = base.account_pct_per * 0.60  # size down more
        p.long_dte_target   = max(base.long_dte_target - 30, 90)   # shorter LEAP ok (IV spike)
        p.short_dte_target  = min(base.short_dte_target + 7, 28)   # sell slightly longer dated

    return p


# ── Position size calculator ──────────────────────────────────────────────────

def calc_contracts(
    account_size:   float,
    net_debit:      float,
    account_pct:    float,
    round_lot:      int = 1,
) -> int:
    """
    Calculate number of contracts based on risk budget.
    Risk per contract = net_debit × 100 (max loss on 1 contract).
    """
    risk_budget = account_size * account_pct
    if net_debit <= 0:
        return 0
    contracts = int(risk_budget / (net_debit * 100))
    return max(contracts - (contracts % round_lot), round_lot)


# ── Signal text generators ─────────────────────────────────────────────────────

def entry_signal_text(
    strategy:   StrategyParams,
    regime_st:  RegimeState,
    contracts:  int,
    net_debit:  float,
    long_strike: float,
    long_expiry: date,
    short_strike: float,
    short_expiry: date,
) -> str:
    regime = regime_st.regime
    p = adapt_params(strategy, regime)
    target_credit = round(net_debit * (1 + p.profit_target_pct), 2)
    stop_debit    = round(net_debit * (1 - p.stop_loss_pct), 2)

    lines = [
        f"{strategy.emoji} **{strategy.code} – {strategy.name}** │ {regime} regime",
        f"   Entry:  Buy {long_strike:.2f}C {long_expiry.strftime('%b %d')}  /  Sell {short_strike:.2f}C {short_expiry.strftime('%b %d')}",
        f"   Debit:  ${net_debit:.2f} × {contracts} contracts = ${net_debit*100*contracts:,.0f} at risk",
        f"   Target: Net ≥ ${target_credit:.2f} (take profit)",
        f"   Stop:   Net ≤ ${stop_debit:.2f} (close / cut)",
        f"   Roll short at DTE ≤ {p.roll_at_dte}",
    ]
    return "\n".join(lines)
