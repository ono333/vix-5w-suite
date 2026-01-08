"""
Exit Detector for VIX 5% Weekly Suite

Monitors open trades and detects exit conditions based on:
- Price targets (profit/loss)
- Regime changes
- Time stops
- Expiration proximity
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from enums import VolatilityRegime, VariantRole, ExitReason


# =============================================================================
# EXIT SIGNAL DATA STRUCTURE
# =============================================================================

@dataclass
class ExitSignal:
    """Signal indicating a trade should be closed."""
    trade_id: str
    signal_time: dt.datetime
    reason: ExitReason
    urgency: str  # "immediate", "soon", "optional"
    
    # Current state
    current_value: float
    entry_value: float
    pnl_pct: float
    
    # Context
    current_regime: VolatilityRegime
    entry_regime: VolatilityRegime
    regime_changed: bool
    
    # Timing
    days_held: int
    days_to_expiration: int
    
    # Recommendation
    suggested_action: str
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "signal_time": self.signal_time.isoformat(),
            "reason": self.reason.value,
            "urgency": self.urgency,
            "current_value": self.current_value,
            "entry_value": self.entry_value,
            "pnl_pct": self.pnl_pct,
            "current_regime": self.current_regime.value,
            "entry_regime": self.entry_regime.value,
            "regime_changed": self.regime_changed,
            "days_held": self.days_held,
            "days_to_expiration": self.days_to_expiration,
            "suggested_action": self.suggested_action,
            "notes": self.notes,
        }


# =============================================================================
# EXIT DETECTION RULES
# =============================================================================

def check_profit_target(
    current_value: float,
    entry_value: float,
    target_mult: float,
) -> tuple[bool, str]:
    """Check if profit target is hit."""
    if entry_value <= 0:
        return False, ""
    
    target_value = entry_value * target_mult
    
    if current_value >= target_value:
        pnl_pct = (current_value / entry_value - 1) * 100
        return True, f"Profit target hit: {pnl_pct:.1f}% gain (target: {(target_mult-1)*100:.0f}%)"
    
    return False, ""


def check_stop_loss(
    current_value: float,
    entry_value: float,
    stop_mult: float,
) -> tuple[bool, str]:
    """Check if stop loss is hit."""
    if entry_value <= 0:
        return False, ""
    
    stop_value = entry_value * stop_mult
    
    if current_value <= stop_value:
        pnl_pct = (current_value / entry_value - 1) * 100
        return True, f"Stop loss hit: {pnl_pct:.1f}% loss (stop: {(stop_mult-1)*100:.0f}%)"
    
    return False, ""


def check_regime_exit(
    current_regime: VolatilityRegime,
    entry_regime: VolatilityRegime,
    variant_role: VariantRole,
) -> tuple[bool, str]:
    """Check if regime change warrants exit."""
    
    # Define which regime changes trigger exits per variant
    exit_triggers = {
        VariantRole.INCOME: {
            # Exit if market becomes stressed
            VolatilityRegime.CALM: [VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
            VolatilityRegime.DECLINING: [VolatilityRegime.STRESSED, VolatilityRegime.EXTREME],
        },
        VariantRole.DECAY: {
            # Exit if VIX spikes again
            VolatilityRegime.DECLINING: [VolatilityRegime.EXTREME, VolatilityRegime.RISING],
        },
        VariantRole.HEDGE: {
            # Exit when crisis abates
            VolatilityRegime.STRESSED: [VolatilityRegime.CALM],
            VolatilityRegime.EXTREME: [VolatilityRegime.CALM, VolatilityRegime.DECLINING],
        },
        VariantRole.CONVEX: {
            # Take profits when spike subsides
            VolatilityRegime.EXTREME: [VolatilityRegime.DECLINING, VolatilityRegime.CALM],
        },
        VariantRole.ADAPTIVE: {
            # No automatic regime exits for adaptive variant
        },
    }
    
    triggers = exit_triggers.get(variant_role, {})
    exit_regimes = triggers.get(entry_regime, [])
    
    if current_regime in exit_regimes:
        return True, f"Regime changed from {entry_regime.value} to {current_regime.value}"
    
    return False, ""


def check_time_stop(
    entry_date: dt.datetime,
    max_hold_weeks: int,
) -> tuple[bool, str]:
    """Check if max holding period exceeded."""
    days_held = (dt.datetime.now() - entry_date).days
    max_days = max_hold_weeks * 7
    
    if days_held >= max_days:
        return True, f"Max holding period exceeded: {days_held} days (max: {max_days})"
    
    return False, ""


def check_expiration_proximity(
    long_expiration: dt.date,
    warning_days: int = 14,
    critical_days: int = 7,
) -> tuple[bool, str, str]:
    """
    Check if long leg is close to expiration.
    
    Returns:
        (is_triggered, message, urgency)
    """
    days_to_exp = (long_expiration - dt.date.today()).days
    
    if days_to_exp <= critical_days:
        return True, f"CRITICAL: Long leg expires in {days_to_exp} days", "immediate"
    
    if days_to_exp <= warning_days:
        return True, f"WARNING: Long leg expires in {days_to_exp} days", "soon"
    
    return False, "", ""


# =============================================================================
# MAIN EXIT DETECTOR
# =============================================================================

def detect_exit_signals(
    trade,  # Trade object from trade_log
    current_regime: VolatilityRegime,
    current_value: float,
    long_expiration: Optional[dt.date] = None,
) -> List[ExitSignal]:
    """
    Check all exit conditions for a trade.
    
    Args:
        trade: Trade object with position details
        current_regime: Current volatility regime
        current_value: Current position value
        long_expiration: Expiration of long leg
    
    Returns:
        List of ExitSignal objects (empty if no exit triggered)
    """
    exit_signals = []
    now = dt.datetime.now()
    
    # Calculate P&L
    pnl_pct = (current_value / trade.entry_debit - 1) if trade.entry_debit > 0 else 0
    
    # Get long expiration from trade if not provided
    if long_expiration is None:
        for leg in trade.legs:
            if leg.leg_type == "long_call":
                long_expiration = leg.expiration
                break
    
    days_to_exp = (long_expiration - dt.date.today()).days if long_expiration else 999
    days_held = (now - trade.entry_date).days
    regime_changed = current_regime != trade.entry_regime
    
    # 1. Check profit target
    target_hit, target_msg = check_profit_target(
        current_value, trade.entry_debit, trade.target_mult
    )
    if target_hit:
        exit_signals.append(ExitSignal(
            trade_id=trade.trade_id,
            signal_time=now,
            reason=ExitReason.TARGET_HIT,
            urgency="soon",
            current_value=current_value,
            entry_value=trade.entry_debit,
            pnl_pct=pnl_pct,
            current_regime=current_regime,
            entry_regime=trade.entry_regime,
            regime_changed=regime_changed,
            days_held=days_held,
            days_to_expiration=days_to_exp,
            suggested_action="Close position to lock in profits",
            notes=target_msg,
        ))
    
    # 2. Check stop loss
    stop_hit, stop_msg = check_stop_loss(
        current_value, trade.entry_debit, trade.stop_mult
    )
    if stop_hit:
        exit_signals.append(ExitSignal(
            trade_id=trade.trade_id,
            signal_time=now,
            reason=ExitReason.STOP_HIT,
            urgency="immediate",
            current_value=current_value,
            entry_value=trade.entry_debit,
            pnl_pct=pnl_pct,
            current_regime=current_regime,
            entry_regime=trade.entry_regime,
            regime_changed=regime_changed,
            days_held=days_held,
            days_to_expiration=days_to_exp,
            suggested_action="Close position to limit losses",
            notes=stop_msg,
        ))
    
    # 3. Check regime exit
    regime_exit, regime_msg = check_regime_exit(
        current_regime, trade.entry_regime, trade.variant_role
    )
    if regime_exit:
        urgency = "immediate" if current_regime == VolatilityRegime.EXTREME else "soon"
        exit_signals.append(ExitSignal(
            trade_id=trade.trade_id,
            signal_time=now,
            reason=ExitReason.REGIME_EXIT,
            urgency=urgency,
            current_value=current_value,
            entry_value=trade.entry_debit,
            pnl_pct=pnl_pct,
            current_regime=current_regime,
            entry_regime=trade.entry_regime,
            regime_changed=True,
            days_held=days_held,
            days_to_expiration=days_to_exp,
            suggested_action="Close position due to regime change",
            notes=regime_msg,
        ))
    
    # 4. Check time stop
    time_stop, time_msg = check_time_stop(
        trade.entry_date, trade.max_hold_weeks if hasattr(trade, 'max_hold_weeks') else 12
    )
    if time_stop:
        exit_signals.append(ExitSignal(
            trade_id=trade.trade_id,
            signal_time=now,
            reason=ExitReason.TIME_STOP,
            urgency="soon",
            current_value=current_value,
            entry_value=trade.entry_debit,
            pnl_pct=pnl_pct,
            current_regime=current_regime,
            entry_regime=trade.entry_regime,
            regime_changed=regime_changed,
            days_held=days_held,
            days_to_expiration=days_to_exp,
            suggested_action="Close position - max hold time reached",
            notes=time_msg,
        ))
    
    # 5. Check expiration proximity
    if long_expiration:
        exp_trigger, exp_msg, exp_urgency = check_expiration_proximity(long_expiration)
        if exp_trigger:
            exit_signals.append(ExitSignal(
                trade_id=trade.trade_id,
                signal_time=now,
                reason=ExitReason.EXPIRATION,
                urgency=exp_urgency,
                current_value=current_value,
                entry_value=trade.entry_debit,
                pnl_pct=pnl_pct,
                current_regime=current_regime,
                entry_regime=trade.entry_regime,
                regime_changed=regime_changed,
                days_held=days_held,
                days_to_expiration=days_to_exp,
                suggested_action="Close or roll position before expiration",
                notes=exp_msg,
            ))
    
    return exit_signals


def get_highest_priority_exit(signals: List[ExitSignal]) -> Optional[ExitSignal]:
    """Get the most urgent exit signal."""
    if not signals:
        return None
    
    # Priority order: immediate > soon > optional
    urgency_order = {"immediate": 0, "soon": 1, "optional": 2}
    
    # Also prioritize by reason: STOP_HIT > others
    reason_order = {
        ExitReason.STOP_HIT: 0,
        ExitReason.EXPIRATION: 1,
        ExitReason.REGIME_EXIT: 2,
        ExitReason.TARGET_HIT: 3,
        ExitReason.TIME_STOP: 4,
        ExitReason.MANUAL: 5,
        ExitReason.ROLL: 6,
    }
    
    sorted_signals = sorted(
        signals,
        key=lambda s: (urgency_order.get(s.urgency, 99), reason_order.get(s.reason, 99))
    )
    
    return sorted_signals[0]


def format_exit_signal(signal: ExitSignal) -> str:
    """Format exit signal for display."""
    urgency_emoji = {
        "immediate": "🔴",
        "soon": "🟠",
        "optional": "🟡",
    }
    
    emoji = urgency_emoji.get(signal.urgency, "⚪")
    
    lines = [
        f"{emoji} **EXIT SIGNAL: {signal.reason.value.upper()}** ({signal.urgency})",
        f"Trade: {signal.trade_id}",
        f"P&L: ${signal.current_value - signal.entry_value:,.2f} ({signal.pnl_pct:.1%})",
        f"",
        f"**Context:**",
        f"- Days held: {signal.days_held}",
        f"- Days to expiration: {signal.days_to_expiration}",
        f"- Regime: {signal.entry_regime.value} → {signal.current_regime.value}",
        f"",
        f"**Action:** {signal.suggested_action}",
        f"**Notes:** {signal.notes}",
    ]
    
    return "\n".join(lines)


# =============================================================================
# PORTFOLIO-LEVEL MONITORING
# =============================================================================

def scan_all_trades_for_exits(
    trades: list,  # List of Trade objects
    current_regime: VolatilityRegime,
    value_lookup: dict,  # {trade_id: current_value}
) -> Dict[str, List[ExitSignal]]:
    """
    Scan all open trades for exit signals.
    
    Args:
        trades: List of open Trade objects
        current_regime: Current volatility regime
        value_lookup: Dictionary mapping trade_id to current position value
    
    Returns:
        Dictionary mapping trade_id to list of exit signals
    """
    all_signals = {}
    
    for trade in trades:
        current_value = value_lookup.get(trade.trade_id, trade.entry_debit)
        
        signals = detect_exit_signals(
            trade=trade,
            current_regime=current_regime,
            current_value=current_value,
        )
        
        if signals:
            all_signals[trade.trade_id] = signals
    
    return all_signals


def get_exit_summary(exit_signals: Dict[str, List[ExitSignal]]) -> str:
    """Generate summary of all exit signals."""
    if not exit_signals:
        return "✅ No exit signals detected"
    
    immediate = []
    soon = []
    optional = []
    
    for trade_id, signals in exit_signals.items():
        top_signal = get_highest_priority_exit(signals)
        if top_signal:
            if top_signal.urgency == "immediate":
                immediate.append(f"- {trade_id}: {top_signal.reason.value}")
            elif top_signal.urgency == "soon":
                soon.append(f"- {trade_id}: {top_signal.reason.value}")
            else:
                optional.append(f"- {trade_id}: {top_signal.reason.value}")
    
    lines = []
    
    if immediate:
        lines.append(f"🔴 **IMMEDIATE ACTION REQUIRED ({len(immediate)}):**")
        lines.extend(immediate)
        lines.append("")
    
    if soon:
        lines.append(f"🟠 **Action Needed Soon ({len(soon)}):**")
        lines.extend(soon)
        lines.append("")
    
    if optional:
        lines.append(f"🟡 **Optional Exits ({len(optional)}):**")
        lines.extend(optional)
    
    return "\n".join(lines) if lines else "✅ No exit signals detected"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing exit detector...")
    
    # Mock trade for testing
    from dataclasses import dataclass as dc
    
    @dc
    class MockLeg:
        leg_type: str
        expiration: dt.date
    
    @dc  
    class MockTrade:
        trade_id: str
        entry_date: dt.datetime
        entry_debit: float
        entry_regime: VolatilityRegime
        variant_role: VariantRole
        target_mult: float
        stop_mult: float
        legs: list
    
    trade = MockTrade(
        trade_id="TEST001",
        entry_date=dt.datetime.now() - dt.timedelta(days=30),
        entry_debit=1500.0,
        entry_regime=VolatilityRegime.CALM,
        variant_role=VariantRole.INCOME,
        target_mult=1.20,
        stop_mult=0.50,
        legs=[MockLeg(leg_type="long_call", expiration=dt.date.today() + dt.timedelta(days=10))]
    )
    
    # Test with profit target hit
    signals = detect_exit_signals(
        trade=trade,
        current_regime=VolatilityRegime.CALM,
        current_value=1900.0,  # 26% gain
    )
    
    print(f"\nTest 1 - Profit target hit:")
    for sig in signals:
        print(format_exit_signal(sig))
    
    # Test with stop loss hit
    signals = detect_exit_signals(
        trade=trade,
        current_regime=VolatilityRegime.CALM,
        current_value=700.0,  # 53% loss
    )
    
    print(f"\nTest 2 - Stop loss hit:")
    for sig in signals:
        print(format_exit_signal(sig))
    
    # Test with regime change
    signals = detect_exit_signals(
        trade=trade,
        current_regime=VolatilityRegime.EXTREME,
        current_value=1500.0,
    )
    
    print(f"\nTest 3 - Regime change:")
    for sig in signals:
        print(format_exit_signal(sig))
