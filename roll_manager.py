from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

SPIKE_PCT_THRESHOLD   = 0.07
DEFAULT_DELTA_TRIGGER = 0.45
DEFAULT_SPIKE_GUARD   = 2

@dataclass
class RollDecision:
    action:              str
    reason:              str
    urgency:             str
    expected_bb:         float
    expected_bb_0dte:    float
    theta_remaining_pct: float
    spike_active:        bool
    details: dict = field(default_factory=dict)

def estimate_theta_remaining(dte: int, weekly_dte: int = 5) -> float:
    if dte <= 0:
        return 0.12
    return round(math.sqrt(min(dte / weekly_dte, 1.0)), 4)

def estimate_buyback_cost(original_premium: float, dte: int, weekly_dte: int = 5) -> float:
    return round(original_premium * estimate_theta_remaining(dte, weekly_dte), 2)

def is_spike_active(last_spike_date: Optional[str], today: Optional[date] = None,
                    guard_days: int = DEFAULT_SPIKE_GUARD) -> bool:
    if last_spike_date is None:
        return False
    try:
        spike_dt = datetime.strptime(last_spike_date, "%Y-%m-%d").date()
    except ValueError:
        return False
    today = today or date.today()
    return (today - spike_dt).days <= int(guard_days * 1.4) + 1

def evaluate_roll(
    dte_remaining:    int,
    short_delta:      Optional[float],
    uvxy_price:       float,
    short_strike:     float,
    variant_params:   dict,
    last_spike_date:  Optional[str] = None,
    original_premium: Optional[float] = None,
    today:            Optional[date] = None,
) -> RollDecision:
    today = today or date.today()
    roll_dte  = variant_params.get("roll_dte_days",    0)
    delta_trig = variant_params.get("delta_trigger",   DEFAULT_DELTA_TRIGGER)
    guard_days = variant_params.get("spike_guard_days", DEFAULT_SPIKE_GUARD)
    pos_type   = variant_params.get("position_type",   "diagonal")

    if pos_type == "long_call" and short_strike == 0.0:
        return RollDecision("hold","Long call — no short leg.",
                            "none",0.0,0.0,0.0,False)

    theta_pct   = estimate_theta_remaining(dte_remaining)
    est_bb      = estimate_buyback_cost(original_premium or 1.0, dte_remaining)
    est_bb_0dte = estimate_buyback_cost(original_premium or 1.0, 0)
    moneyness   = (uvxy_price - short_strike) / short_strike
    is_deep_itm = moneyness > 0.05

    if short_delta is None:
        if moneyness > 0.05:  effective_delta = 0.65
        elif moneyness > 0.0: effective_delta = 0.52
        elif moneyness > -0.03: effective_delta = 0.40
        else: effective_delta = 0.25
    else:
        effective_delta = short_delta

    spike_active = is_spike_active(last_spike_date, today, guard_days)

    if effective_delta >= 0.70 or moneyness > 0.10:
        return RollDecision(
            "roll_early_itm",
            f"Short deeply ITM (UVXY ${uvxy_price:.2f} vs ${short_strike:.2f}, "
            f"{moneyness*100:+.1f}%). Roll now — delta {effective_delta:.2f}.",
            "critical", est_bb, est_bb_0dte, theta_pct, spike_active,
            dict(moneyness=moneyness, delta=effective_delta))

    if spike_active and not is_deep_itm:
        days_since = (today - datetime.strptime(last_spike_date, "%Y-%m-%d").date()).days
        return RollDecision(
            "spike_guard_hold",
            f"Spike guard active ({days_since}d since {last_spike_date}). "
            f"Wait {guard_days} trading days. P3 Jan-20 lesson: patience = +$11.60.",
            "low", est_bb, est_bb_0dte, theta_pct, True,
            dict(last_spike=last_spike_date, days_since=days_since))

    if effective_delta >= delta_trig and dte_remaining > roll_dte:
        return RollDecision(
            "roll_early_delta",
            f"Delta {effective_delta:.2f} >= trigger {delta_trig:.2f}. "
            f"Roll now. Est BB ${est_bb:.2f} (vs ${est_bb_0dte:.2f} at 0DTE).",
            "high", est_bb, est_bb_0dte, theta_pct, False,
            dict(delta=effective_delta, trigger=delta_trig))

    if dte_remaining == 1:
        return RollDecision(
            "order_roll",
            f"DTE=1 — Thursday signal. Place roll order NOW for Friday morning fill. "
            f"Est BB ~${est_bb:.2f} at open (theta ~{theta_pct*100:.0f}% remaining).",
            "medium", est_bb, est_bb_0dte, theta_pct, False,
            dict(dte=dte_remaining))

    if dte_remaining <= roll_dte:
        return RollDecision(
            "roll_now",
            f"DTE={dte_remaining} <= roll_dte_days={roll_dte}. "
            f"Expiry-day roll. Theta ~{theta_pct*100:.0f}%, BB ~${est_bb:.2f}.",
            "medium", est_bb, est_bb_0dte, theta_pct, False,
            dict(dte=dte_remaining))

    days_to_roll = dte_remaining - roll_dte
    return RollDecision(
        "hold",
        f"Hold {days_to_roll} more day(s). "
        f"Theta ~{theta_pct*100:.0f}% remaining, "
        f"BB drops from ${est_bb:.2f} to ${est_bb_0dte:.2f} by expiry. "
        f"Delta {effective_delta:.2f} < trigger {delta_trig:.2f}.",
        "none", est_bb, est_bb_0dte, theta_pct, False,
        dict(dte=dte_remaining, days_to_roll=days_to_roll))

def format_roll_signal(decision: RollDecision, variant_name: str,
                       short_strike: float, uvxy_price: float) -> str:
    icons = {"roll_now":"🔄","roll_early_delta":"⚡",
             "roll_early_itm":"🚨","spike_guard_hold":"🛡","hold":"✋"}
    return (f"{icons.get(decision.action,'•')} {variant_name}\n"
            f"   {decision.reason}")
