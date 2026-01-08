#!/usr/bin/env python3
"""
Post-Mortem Analytics Module for VIX 5% Weekly Suite

This module implements the diagnostic backbone for trade analysis:

1. Exit Classification Taxonomy (Priority-Ordered):
   - REGIME exits (Priority 1): Closed due to regime change
   - STRUCTURE exits (Priority 2): Closed due to structure invalidation
   - LOSS exits (Priority 3, soft): Price-based loss threshold
   - PLANNED exits: Time stop, expiration, profit target

2. Execution Reality Tracking:
   - Signal mid vs intended fill vs actual fill
   - Slippage attribution
   - Fill-within-envelope tracking

3. Post-Mortem Lifecycle:
   - CLOSED -> POST_MORTEM_READY -> REVIEWED -> ACKNOWLEDGED
   - Mandatory review before variant promotion decisions

4. Robustness Score Updates:
   - Track robustness at entry
   - Calculate outcome-adjusted robustness
   - Feed back into variant selection

This prevents self-deception by forcing structured reflection on every trade.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

from regime_detector import VolatilityRegime


# ============================================================================
# Exit Classification Taxonomy
# ============================================================================

class ExitCategory(Enum):
    """
    High-level exit categories in priority order.
    
    Priority determines which exit type "wins" when multiple triggers fire:
        REGIME > STRUCTURE > PLANNED > LOSS
    """
    REGIME = "REGIME"          # Priority 1: Regime-driven
    STRUCTURE = "STRUCTURE"    # Priority 2: Structure invalidation
    PLANNED = "PLANNED"        # Priority 3: Expected/designed exits
    LOSS = "LOSS"              # Priority 4 (soft): Price-based loss


class ExitType(Enum):
    """
    Detailed exit type classification.
    
    Each exit type belongs to a category and has specific semantics.
    """
    # REGIME category (Priority 1)
    REGIME_CALM_TO_RISING = "REGIME_CALM_TO_RISING"
    REGIME_CALM_TO_STRESSED = "REGIME_CALM_TO_STRESSED"
    REGIME_RISING_TO_EXTREME = "REGIME_RISING_TO_EXTREME"
    REGIME_STRESSED_TO_DECLINING = "REGIME_STRESSED_TO_DECLINING"
    REGIME_OTHER = "REGIME_OTHER"
    
    # STRUCTURE category (Priority 2)
    STRUCTURE_TERM_INVERSION = "STRUCTURE_TERM_INVERSION"
    STRUCTURE_SPREAD_EXCEEDED = "STRUCTURE_SPREAD_EXCEEDED"
    STRUCTURE_LIQUIDITY_FAILED = "STRUCTURE_LIQUIDITY_FAILED"
    STRUCTURE_OTHER = "STRUCTURE_OTHER"
    
    # PLANNED category (Priority 3)
    PLANNED_TP_TIME_DECAY = "PLANNED_TP_TIME_DECAY"        # Expire worthless by design
    PLANNED_TP_PROFIT_TARGET = "PLANNED_TP_PROFIT_TARGET"  # Hit profit condition
    PLANNED_TIME_STOP = "PLANNED_TIME_STOP"                # Max hold weeks reached
    PLANNED_EXPIRATION = "PLANNED_EXPIRATION"              # DTE warning/hit
    PLANNED_MANUAL_DISCRETION = "PLANNED_MANUAL_DISCRETION"
    
    # LOSS category (Priority 4 - soft, tertiary)
    LOSS_THRESHOLD_BREACH = "LOSS_THRESHOLD_BREACH"        # Standard loss stop
    LOSS_CATASTROPHIC = "LOSS_CATASTROPHIC"                # Emergency stop
    
    # Special
    UNKNOWN = "UNKNOWN"

    @property
    def category(self) -> ExitCategory:
        """Get the category for this exit type."""
        if self.value.startswith("REGIME_"):
            return ExitCategory.REGIME
        elif self.value.startswith("STRUCTURE_"):
            return ExitCategory.STRUCTURE
        elif self.value.startswith("PLANNED_"):
            return ExitCategory.PLANNED
        elif self.value.startswith("LOSS_"):
            return ExitCategory.LOSS
        return ExitCategory.PLANNED

    @property
    def priority(self) -> int:
        """Get priority (lower = higher priority)."""
        return {
            ExitCategory.REGIME: 1,
            ExitCategory.STRUCTURE: 2,
            ExitCategory.PLANNED: 3,
            ExitCategory.LOSS: 4,
        }.get(self.category, 99)

    @property
    def is_soft_stop(self) -> bool:
        """Is this a soft (tertiary) stop that should not dominate?"""
        return self.category == ExitCategory.LOSS


class PostMortemStatus(Enum):
    """
    Lifecycle status of a post-mortem review.
    
    Trades must progress through this lifecycle before contributing
    to variant promotion/demotion decisions.
    """
    POST_MORTEM_READY = "POST_MORTEM_READY"    # Trade closed, awaiting review
    UNDER_REVIEW = "UNDER_REVIEW"              # Review in progress
    REVIEWED = "REVIEWED"                       # Review complete, awaiting acknowledgment
    ACKNOWLEDGED = "ACKNOWLEDGED"              # Fully processed, can inform decisions


# ============================================================================
# Execution Reality Tracking
# ============================================================================

@dataclass
class ExecutionComparison:
    """
    Track the gap between signal and reality.
    
    This prevents self-deception about execution quality.
    """
    # Signal phase (Thursday 16:30)
    signal_mid_price: Optional[float] = None
    signal_bid: Optional[float] = None
    signal_ask: Optional[float] = None
    signal_spread_pct: Optional[float] = None
    signal_robustness: Optional[float] = None
    
    # Intended execution (operator's plan)
    intended_fill_price: Optional[float] = None
    intended_fill_time: Optional[datetime] = None
    
    # Actual execution
    actual_fill_price: Optional[float] = None
    actual_fill_time: Optional[datetime] = None
    actual_bid: Optional[float] = None
    actual_ask: Optional[float] = None
    
    # Derived metrics
    @property
    def signal_to_fill_slippage_bps(self) -> Optional[float]:
        """Slippage from signal mid to actual fill (basis points)."""
        if self.signal_mid_price and self.actual_fill_price and self.signal_mid_price > 0:
            return ((self.actual_fill_price - self.signal_mid_price) 
                    / self.signal_mid_price * 10000)
        return None
    
    @property
    def intended_to_fill_slippage_bps(self) -> Optional[float]:
        """Slippage from intended to actual fill (basis points)."""
        if self.intended_fill_price and self.actual_fill_price and self.intended_fill_price > 0:
            return ((self.actual_fill_price - self.intended_fill_price) 
                    / self.intended_fill_price * 10000)
        return None
    
    @property
    def fill_within_signal_envelope(self) -> Optional[bool]:
        """Was fill within signal bid-ask envelope?"""
        if (self.signal_bid is not None and 
            self.signal_ask is not None and 
            self.actual_fill_price is not None):
            return self.signal_bid <= self.actual_fill_price <= self.signal_ask
        return None
    
    @property
    def execution_delay_hours(self) -> Optional[float]:
        """Hours between signal and execution."""
        if self.signal_mid_price and self.actual_fill_time and self.intended_fill_time:
            # Approximate: assume signal was at intended_fill_time - delay
            delta = self.actual_fill_time - self.intended_fill_time
            return delta.total_seconds() / 3600
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_mid_price": self.signal_mid_price,
            "signal_bid": self.signal_bid,
            "signal_ask": self.signal_ask,
            "signal_spread_pct": self.signal_spread_pct,
            "signal_robustness": self.signal_robustness,
            "intended_fill_price": self.intended_fill_price,
            "intended_fill_time": self.intended_fill_time.isoformat() if self.intended_fill_time else None,
            "actual_fill_price": self.actual_fill_price,
            "actual_fill_time": self.actual_fill_time.isoformat() if self.actual_fill_time else None,
            "actual_bid": self.actual_bid,
            "actual_ask": self.actual_ask,
            # Computed
            "signal_to_fill_slippage_bps": self.signal_to_fill_slippage_bps,
            "intended_to_fill_slippage_bps": self.intended_to_fill_slippage_bps,
            "fill_within_signal_envelope": self.fill_within_signal_envelope,
            "execution_delay_hours": self.execution_delay_hours,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExecutionComparison':
        return cls(
            signal_mid_price=d.get("signal_mid_price"),
            signal_bid=d.get("signal_bid"),
            signal_ask=d.get("signal_ask"),
            signal_spread_pct=d.get("signal_spread_pct"),
            signal_robustness=d.get("signal_robustness"),
            intended_fill_price=d.get("intended_fill_price"),
            intended_fill_time=datetime.fromisoformat(d["intended_fill_time"]) if d.get("intended_fill_time") else None,
            actual_fill_price=d.get("actual_fill_price"),
            actual_fill_time=datetime.fromisoformat(d["actual_fill_time"]) if d.get("actual_fill_time") else None,
            actual_bid=d.get("actual_bid"),
            actual_ask=d.get("actual_ask"),
        )


# ============================================================================
# Operational Metrics
# ============================================================================

@dataclass
class OperationalMetrics:
    """
    Track how much attention and intervention a trade required.
    
    These metrics are crucial for variant promotion decisions:
    - High-performing but high-intervention variants may not be suitable for live
    - Operationally simple variants get preference even with slightly lower returns
    """
    # Intervention tracking
    intervention_required: bool = False
    intervention_count: int = 0
    manual_overrides: List[str] = field(default_factory=list)
    
    # Attention tracking (1-5 scale)
    attention_score: int = 1  # 1=minimal, 5=constant monitoring
    attention_events: List[str] = field(default_factory=list)
    
    # Time-based
    total_monitoring_minutes: int = 0
    alert_count: int = 0
    
    # Psychological impact
    stress_level: int = 1  # 1=none, 5=high anxiety
    would_repeat: bool = True  # Would you take this trade again?
    
    # Notes
    management_notes: str = ""
    
    @property
    def complexity_score(self) -> float:
        """
        Composite complexity score (0-100).
        
        Higher = more complex/demanding to manage.
        """
        base = 0.0
        
        # Intervention weight
        if self.intervention_required:
            base += 20
        base += min(self.intervention_count * 5, 30)  # Max 30 points
        
        # Attention weight
        base += (self.attention_score - 1) * 10  # 0-40 points
        
        # Alert weight
        base += min(self.alert_count * 2, 10)  # Max 10 points
        
        return min(base, 100.0)
    
    def log_intervention(self, description: str) -> None:
        """Log a manual intervention event."""
        self.intervention_required = True
        self.intervention_count += 1
        self.manual_overrides.append(
            f"{datetime.utcnow().isoformat()}: {description}"
        )
    
    def log_attention_event(self, description: str, attention_increase: int = 1) -> None:
        """Log an event that required attention."""
        self.attention_events.append(
            f"{datetime.utcnow().isoformat()}: {description}"
        )
        self.attention_score = min(5, self.attention_score + attention_increase)
        self.alert_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_required": self.intervention_required,
            "intervention_count": self.intervention_count,
            "manual_overrides": self.manual_overrides,
            "attention_score": self.attention_score,
            "attention_events": self.attention_events,
            "total_monitoring_minutes": self.total_monitoring_minutes,
            "alert_count": self.alert_count,
            "stress_level": self.stress_level,
            "would_repeat": self.would_repeat,
            "management_notes": self.management_notes,
            "complexity_score": self.complexity_score,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OperationalMetrics':
        return cls(
            intervention_required=d.get("intervention_required", False),
            intervention_count=d.get("intervention_count", 0),
            manual_overrides=d.get("manual_overrides", []),
            attention_score=d.get("attention_score", 1),
            attention_events=d.get("attention_events", []),
            total_monitoring_minutes=d.get("total_monitoring_minutes", 0),
            alert_count=d.get("alert_count", 0),
            stress_level=d.get("stress_level", 1),
            would_repeat=d.get("would_repeat", True),
            management_notes=d.get("management_notes", ""),
        )


# ============================================================================
# Trade Post-Mortem
# ============================================================================

@dataclass
class TradePostMortem:
    """
    Complete post-mortem record for a closed trade.
    
    This is the diagnostic artifact that prevents self-deception.
    Every closed trade MUST have a post-mortem before informing decisions.
    """
    # Identity
    post_mortem_id: str
    trade_id: str
    variant_id: str
    
    # Status
    status: PostMortemStatus = PostMortemStatus.POST_MORTEM_READY
    
    # Exit classification (the core diagnostic)
    exit_type: ExitType = ExitType.UNKNOWN
    exit_category: ExitCategory = ExitCategory.PLANNED
    exit_trigger_description: str = ""
    exit_was_as_designed: bool = True  # Did exit match variant's expected behavior?
    
    # Regime context
    regime_at_entry: Optional[VolatilityRegime] = None
    regime_at_exit: Optional[VolatilityRegime] = None
    regime_changed: bool = False
    
    # Execution comparison (per leg)
    long_leg_execution: Optional[ExecutionComparison] = None
    short_leg_execution: Optional[ExecutionComparison] = None
    
    # Performance
    gross_pnl: float = 0.0
    net_pnl: float = 0.0  # After fees/slippage
    pnl_vs_expectation: str = ""  # BETTER / AS_EXPECTED / WORSE / MUCH_WORSE
    
    # Robustness tracking
    robustness_at_entry: float = 0.0
    robustness_adjustment: float = 0.0  # Positive = increase, negative = decrease
    robustness_after_exit: float = 0.0
    
    # Operational metrics
    operational_metrics: OperationalMetrics = field(default_factory=OperationalMetrics)
    
    # Review lifecycle
    review_started_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    reviewer_notes: str = ""
    
    # Lessons learned
    lessons: List[str] = field(default_factory=list)
    would_take_again: bool = True
    recommended_adjustments: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def start_review(self, notes: str = "") -> None:
        """Begin the review process."""
        self.status = PostMortemStatus.UNDER_REVIEW
        self.review_started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if notes:
            self.reviewer_notes = notes
    
    def complete_review(self, notes: str = "") -> None:
        """Complete the review (awaiting acknowledgment)."""
        self.status = PostMortemStatus.REVIEWED
        self.reviewed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if notes:
            self.reviewer_notes = f"{self.reviewer_notes}\n{notes}".strip()
    
    def acknowledge(self, notes: str = "") -> None:
        """Acknowledge the review (fully processed)."""
        self.status = PostMortemStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if notes:
            self.reviewer_notes = f"{self.reviewer_notes}\n{notes}".strip()
    
    def classify_exit(
        self,
        exit_type: ExitType,
        description: str = "",
        was_as_designed: bool = True,
    ) -> None:
        """Classify the exit type (required during review)."""
        self.exit_type = exit_type
        self.exit_category = exit_type.category
        self.exit_trigger_description = description
        self.exit_was_as_designed = was_as_designed
        self.updated_at = datetime.utcnow()
    
    def update_robustness(
        self,
        new_robustness: float,
        reason: str = "",
    ) -> None:
        """
        Update robustness score based on trade outcome.
        
        Positive adjustments for:
        - Fill within envelope
        - Exit as designed
        - Low slippage
        
        Negative adjustments for:
        - Fill outside envelope
        - Unexpected exit type
        - High slippage
        - Intervention required
        """
        old = self.robustness_at_entry
        self.robustness_after_exit = new_robustness
        self.robustness_adjustment = new_robustness - old
        self.updated_at = datetime.utcnow()
        
        if reason:
            self.reviewer_notes = f"{self.reviewer_notes}\nRobustness: {reason}".strip()
    
    def add_lesson(self, lesson: str) -> None:
        """Add a lesson learned."""
        self.lessons.append(f"{datetime.utcnow().date()}: {lesson}")
        self.updated_at = datetime.utcnow()
    
    @property
    def is_complete(self) -> bool:
        """Is this post-mortem fully complete?"""
        return self.status == PostMortemStatus.ACKNOWLEDGED
    
    @property
    def can_inform_decisions(self) -> bool:
        """Can this post-mortem be used for variant promotion decisions?"""
        return self.status in [PostMortemStatus.REVIEWED, PostMortemStatus.ACKNOWLEDGED]
    
    @property
    def exit_priority(self) -> int:
        """Get the priority of the exit (lower = higher priority)."""
        return self.exit_type.priority
    
    @property
    def was_soft_stop(self) -> bool:
        """Was this a soft (tertiary) stop?"""
        return self.exit_type.is_soft_stop
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_mortem_id": self.post_mortem_id,
            "trade_id": self.trade_id,
            "variant_id": self.variant_id,
            "status": self.status.value,
            "exit_type": self.exit_type.value,
            "exit_category": self.exit_category.value,
            "exit_trigger_description": self.exit_trigger_description,
            "exit_was_as_designed": self.exit_was_as_designed,
            "regime_at_entry": self.regime_at_entry.value if self.regime_at_entry else None,
            "regime_at_exit": self.regime_at_exit.value if self.regime_at_exit else None,
            "regime_changed": self.regime_changed,
            "long_leg_execution": self.long_leg_execution.to_dict() if self.long_leg_execution else None,
            "short_leg_execution": self.short_leg_execution.to_dict() if self.short_leg_execution else None,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "pnl_vs_expectation": self.pnl_vs_expectation,
            "robustness_at_entry": self.robustness_at_entry,
            "robustness_adjustment": self.robustness_adjustment,
            "robustness_after_exit": self.robustness_after_exit,
            "operational_metrics": self.operational_metrics.to_dict(),
            "review_started_at": self.review_started_at.isoformat() if self.review_started_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "reviewer_notes": self.reviewer_notes,
            "lessons": self.lessons,
            "would_take_again": self.would_take_again,
            "recommended_adjustments": self.recommended_adjustments,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TradePostMortem':
        return cls(
            post_mortem_id=d["post_mortem_id"],
            trade_id=d["trade_id"],
            variant_id=d["variant_id"],
            status=PostMortemStatus(d.get("status", "POST_MORTEM_READY")),
            exit_type=ExitType(d.get("exit_type", "UNKNOWN")),
            exit_category=ExitCategory(d.get("exit_category", "PLANNED")),
            exit_trigger_description=d.get("exit_trigger_description", ""),
            exit_was_as_designed=d.get("exit_was_as_designed", True),
            regime_at_entry=VolatilityRegime(d["regime_at_entry"]) if d.get("regime_at_entry") else None,
            regime_at_exit=VolatilityRegime(d["regime_at_exit"]) if d.get("regime_at_exit") else None,
            regime_changed=d.get("regime_changed", False),
            long_leg_execution=ExecutionComparison.from_dict(d["long_leg_execution"]) if d.get("long_leg_execution") else None,
            short_leg_execution=ExecutionComparison.from_dict(d["short_leg_execution"]) if d.get("short_leg_execution") else None,
            gross_pnl=d.get("gross_pnl", 0.0),
            net_pnl=d.get("net_pnl", 0.0),
            pnl_vs_expectation=d.get("pnl_vs_expectation", ""),
            robustness_at_entry=d.get("robustness_at_entry", 0.0),
            robustness_adjustment=d.get("robustness_adjustment", 0.0),
            robustness_after_exit=d.get("robustness_after_exit", 0.0),
            operational_metrics=OperationalMetrics.from_dict(d["operational_metrics"]) if d.get("operational_metrics") else OperationalMetrics(),
            review_started_at=datetime.fromisoformat(d["review_started_at"]) if d.get("review_started_at") else None,
            reviewed_at=datetime.fromisoformat(d["reviewed_at"]) if d.get("reviewed_at") else None,
            acknowledged_at=datetime.fromisoformat(d["acknowledged_at"]) if d.get("acknowledged_at") else None,
            reviewer_notes=d.get("reviewer_notes", ""),
            lessons=d.get("lessons", []),
            would_take_again=d.get("would_take_again", True),
            recommended_adjustments=d.get("recommended_adjustments", []),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else datetime.utcnow(),
        )


# ============================================================================
# Post-Mortem Manager
# ============================================================================

class PostMortemManager:
    """
    Manages post-mortem records persistence and analytics.
    """
    
    def __init__(self, pm_path: Path = Path("post_mortems.json")):
        self.pm_path = pm_path
        self.post_mortems: Dict[str, TradePostMortem] = {}
        self._load()
    
    def _load(self) -> None:
        """Load post-mortems from disk."""
        if self.pm_path.exists():
            try:
                with open(self.pm_path, "r") as f:
                    data = json.load(f)
                for pm_dict in data.get("post_mortems", []):
                    pm = TradePostMortem.from_dict(pm_dict)
                    self.post_mortems[pm.post_mortem_id] = pm
            except Exception as e:
                print(f"Warning: Could not load post-mortems: {e}")
    
    def _save(self) -> None:
        """Save post-mortems to disk."""
        data = {
            "post_mortems": [pm.to_dict() for pm in self.post_mortems.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self.pm_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def create_post_mortem(
        self,
        trade_id: str,
        variant_id: str,
        regime_at_entry: Optional[VolatilityRegime] = None,
        regime_at_exit: Optional[VolatilityRegime] = None,
        robustness_at_entry: float = 0.0,
        gross_pnl: float = 0.0,
    ) -> TradePostMortem:
        """
        Create a new post-mortem for a closed trade.
        
        This should be called automatically when a trade is fully closed.
        """
        pm_id = f"PM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        
        pm = TradePostMortem(
            post_mortem_id=pm_id,
            trade_id=trade_id,
            variant_id=variant_id,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            regime_changed=(regime_at_entry != regime_at_exit if regime_at_entry and regime_at_exit else False),
            robustness_at_entry=robustness_at_entry,
            robustness_after_exit=robustness_at_entry,  # Will be adjusted later
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl,  # Will be adjusted for slippage later
        )
        
        self.post_mortems[pm_id] = pm
        self._save()
        return pm
    
    def get_post_mortem(self, pm_id: str) -> Optional[TradePostMortem]:
        """Get a specific post-mortem."""
        return self.post_mortems.get(pm_id)
    
    def get_for_trade(self, trade_id: str) -> Optional[TradePostMortem]:
        """Get post-mortem for a specific trade."""
        for pm in self.post_mortems.values():
            if pm.trade_id == trade_id:
                return pm
        return None
    
    def get_pending_reviews(self) -> List[TradePostMortem]:
        """Get post-mortems awaiting review."""
        return [
            pm for pm in self.post_mortems.values()
            if pm.status == PostMortemStatus.POST_MORTEM_READY
        ]
    
    def get_by_status(self, status: PostMortemStatus) -> List[TradePostMortem]:
        """Get post-mortems by status."""
        return [pm for pm in self.post_mortems.values() if pm.status == status]
    
    def get_by_variant(self, variant_id: str) -> List[TradePostMortem]:
        """Get all post-mortems for a variant."""
        return [pm for pm in self.post_mortems.values() if pm.variant_id == variant_id]
    
    def update_post_mortem(self, pm: TradePostMortem) -> None:
        """Update a post-mortem record."""
        pm.updated_at = datetime.utcnow()
        self.post_mortems[pm.post_mortem_id] = pm
        self._save()
    
    # =========================================================================
    # Analytics
    # =========================================================================
    
    def get_variant_analytics(self, variant_id: str) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a variant based on acknowledged post-mortems.
        
        Only uses post-mortems that can inform decisions (REVIEWED or ACKNOWLEDGED).
        """
        pms = [
            pm for pm in self.post_mortems.values()
            if pm.variant_id == variant_id and pm.can_inform_decisions
        ]
        
        if not pms:
            return {
                "variant_id": variant_id,
                "total_trades": 0,
                "has_sufficient_data": False,
            }
        
        # Basic counts
        total = len(pms)
        
        # Exit type distribution
        exit_type_counts = {}
        for pm in pms:
            et = pm.exit_type.value
            exit_type_counts[et] = exit_type_counts.get(et, 0) + 1
        
        # Exit category distribution
        exit_category_counts = {}
        for pm in pms:
            ec = pm.exit_category.value
            exit_category_counts[ec] = exit_category_counts.get(ec, 0) + 1
        
        # PnL stats
        gross_pnls = [pm.gross_pnl for pm in pms]
        net_pnls = [pm.net_pnl for pm in pms]
        
        # Robustness stats
        robustness_adjustments = [pm.robustness_adjustment for pm in pms]
        
        # Operational stats
        complexity_scores = [pm.operational_metrics.complexity_score for pm in pms]
        intervention_count = sum(1 for pm in pms if pm.operational_metrics.intervention_required)
        would_repeat_count = sum(1 for pm in pms if pm.would_take_again)
        
        # Exit as designed rate
        as_designed_count = sum(1 for pm in pms if pm.exit_was_as_designed)
        
        # Regime change impact
        regime_changed_count = sum(1 for pm in pms if pm.regime_changed)
        
        return {
            "variant_id": variant_id,
            "total_trades": total,
            "has_sufficient_data": total >= 3,
            
            # Exit analysis
            "exit_type_distribution": exit_type_counts,
            "exit_category_distribution": exit_category_counts,
            "exit_as_designed_rate": as_designed_count / total if total > 0 else 0.0,
            
            # Performance
            "total_gross_pnl": sum(gross_pnls),
            "total_net_pnl": sum(net_pnls),
            "avg_gross_pnl": sum(gross_pnls) / total if total > 0 else 0.0,
            "avg_net_pnl": sum(net_pnls) / total if total > 0 else 0.0,
            "win_rate": sum(1 for p in net_pnls if p > 0) / total if total > 0 else 0.0,
            
            # Robustness evolution
            "avg_robustness_adjustment": sum(robustness_adjustments) / total if total > 0 else 0.0,
            "robustness_trend": "IMPROVING" if sum(robustness_adjustments) > 0 else "DECLINING",
            
            # Operational
            "avg_complexity_score": sum(complexity_scores) / total if total > 0 else 0.0,
            "intervention_rate": intervention_count / total if total > 0 else 0.0,
            "would_repeat_rate": would_repeat_count / total if total > 0 else 0.0,
            
            # Regime impact
            "regime_change_rate": regime_changed_count / total if total > 0 else 0.0,
        }
    
    def get_promotion_recommendation(self, variant_id: str) -> Dict[str, Any]:
        """
        Get a recommendation on whether this variant should be promoted to live.
        
        Criteria (all must pass):
        1. Sufficient data (≥ 5 reviewed post-mortems)
        2. Would-repeat rate ≥ 80%
        3. Avg complexity score ≤ 40
        4. Win rate ≥ 30% OR positive total PnL
        5. Exit-as-designed rate ≥ 70%
        6. Intervention rate ≤ 30%
        """
        analytics = self.get_variant_analytics(variant_id)
        
        if not analytics.get("has_sufficient_data"):
            return {
                "variant_id": variant_id,
                "recommendation": "INSUFFICIENT_DATA",
                "reason": f"Only {analytics.get('total_trades', 0)} reviewed trades (need ≥3)",
                "passed_criteria": [],
                "failed_criteria": ["INSUFFICIENT_DATA"],
            }
        
        passed = []
        failed = []
        
        # Criterion 1: Would repeat
        if analytics.get("would_repeat_rate", 0) >= 0.80:
            passed.append("WOULD_REPEAT")
        else:
            failed.append(f"WOULD_REPEAT: {analytics.get('would_repeat_rate', 0):.0%} < 80%")
        
        # Criterion 2: Complexity
        if analytics.get("avg_complexity_score", 100) <= 40:
            passed.append("COMPLEXITY")
        else:
            failed.append(f"COMPLEXITY: {analytics.get('avg_complexity_score', 0):.1f} > 40")
        
        # Criterion 3: Profitability
        if analytics.get("win_rate", 0) >= 0.30 or analytics.get("total_net_pnl", 0) > 0:
            passed.append("PROFITABILITY")
        else:
            failed.append(f"PROFITABILITY: Win {analytics.get('win_rate', 0):.0%}, PnL ${analytics.get('total_net_pnl', 0):.0f}")
        
        # Criterion 4: Exit as designed
        if analytics.get("exit_as_designed_rate", 0) >= 0.70:
            passed.append("EXIT_AS_DESIGNED")
        else:
            failed.append(f"EXIT_AS_DESIGNED: {analytics.get('exit_as_designed_rate', 0):.0%} < 70%")
        
        # Criterion 5: Intervention rate
        if analytics.get("intervention_rate", 1) <= 0.30:
            passed.append("INTERVENTION")
        else:
            failed.append(f"INTERVENTION: {analytics.get('intervention_rate', 0):.0%} > 30%")
        
        # Overall recommendation
        if len(failed) == 0:
            recommendation = "PROMOTE"
            reason = "All criteria passed"
        elif len(failed) <= 2:
            recommendation = "CONDITIONAL"
            reason = f"Review needed: {', '.join(failed)}"
        else:
            recommendation = "DO_NOT_PROMOTE"
            reason = f"Multiple failures: {', '.join(failed)}"
        
        return {
            "variant_id": variant_id,
            "recommendation": recommendation,
            "reason": reason,
            "passed_criteria": passed,
            "failed_criteria": failed,
            "analytics_summary": {
                "total_trades": analytics.get("total_trades"),
                "total_net_pnl": analytics.get("total_net_pnl"),
                "win_rate": analytics.get("win_rate"),
                "avg_complexity": analytics.get("avg_complexity_score"),
                "would_repeat_rate": analytics.get("would_repeat_rate"),
            },
        }
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get summary across all variants."""
        all_pms = list(self.post_mortems.values())
        
        status_counts = {}
        for pm in all_pms:
            s = pm.status.value
            status_counts[s] = status_counts.get(s, 0) + 1
        
        variant_ids = list(set(pm.variant_id for pm in all_pms))
        
        return {
            "total_post_mortems": len(all_pms),
            "status_distribution": status_counts,
            "pending_review_count": status_counts.get("POST_MORTEM_READY", 0),
            "variants_with_data": variant_ids,
            "total_net_pnl": sum(pm.net_pnl for pm in all_pms if pm.can_inform_decisions),
        }


# ============================================================================
# Robustness Update Calculator
# ============================================================================

def calculate_robustness_adjustment(
    pm: TradePostMortem,
    base_adjustment: float = 0.0,
) -> float:
    """
    Calculate robustness adjustment based on trade outcome.
    
    Positive factors:
    - Fill within signal envelope: +5
    - Exit as designed: +5
    - No intervention required: +3
    - Low complexity (<20): +2
    
    Negative factors:
    - Fill outside envelope: -5
    - Unexpected exit type: -5
    - Intervention required: -5
    - High complexity (>60): -5
    - Soft stop triggered: -3
    """
    adjustment = base_adjustment
    
    # Fill within envelope
    if pm.long_leg_execution:
        if pm.long_leg_execution.fill_within_signal_envelope is True:
            adjustment += 5
        elif pm.long_leg_execution.fill_within_signal_envelope is False:
            adjustment -= 5
    
    # Exit as designed
    if pm.exit_was_as_designed:
        adjustment += 5
    else:
        adjustment -= 5
    
    # Intervention
    if pm.operational_metrics.intervention_required:
        adjustment -= 5
    else:
        adjustment += 3
    
    # Complexity
    complexity = pm.operational_metrics.complexity_score
    if complexity < 20:
        adjustment += 2
    elif complexity > 60:
        adjustment -= 5
    
    # Soft stop penalty
    if pm.was_soft_stop:
        adjustment -= 3
    
    return adjustment


# ============================================================================
# Exit Type Inference
# ============================================================================

def infer_exit_type(
    regime_at_entry: Optional[VolatilityRegime],
    regime_at_exit: Optional[VolatilityRegime],
    exit_reason_from_trade: str,
    pnl: float,
    weeks_held: int,
    max_hold_weeks: int,
    dte_at_exit: Optional[int] = None,
) -> ExitType:
    """
    Infer the most likely exit type based on available data.
    
    This helps classify exits when the operator doesn't specify explicitly.
    """
    # Check for regime change first (Priority 1)
    if regime_at_entry and regime_at_exit and regime_at_entry != regime_at_exit:
        transition = f"{regime_at_entry.value}_TO_{regime_at_exit.value}"
        if transition == "CALM_TO_RISING":
            return ExitType.REGIME_CALM_TO_RISING
        elif transition == "CALM_TO_STRESSED":
            return ExitType.REGIME_CALM_TO_STRESSED
        elif transition == "RISING_TO_EXTREME":
            return ExitType.REGIME_RISING_TO_EXTREME
        elif transition == "STRESSED_TO_DECLINING":
            return ExitType.REGIME_STRESSED_TO_DECLINING
        else:
            return ExitType.REGIME_OTHER
    
    # Check for time-based exits
    if weeks_held >= max_hold_weeks:
        return ExitType.PLANNED_TIME_STOP
    
    if dte_at_exit is not None and dte_at_exit <= 0:
        return ExitType.PLANNED_EXPIRATION
    
    # Check reason string
    reason_upper = exit_reason_from_trade.upper()
    if "TP" in reason_upper or "TARGET" in reason_upper or "PROFIT" in reason_upper:
        return ExitType.PLANNED_TP_PROFIT_TARGET
    if "DECAY" in reason_upper or "WORTHLESS" in reason_upper:
        return ExitType.PLANNED_TP_TIME_DECAY
    if "STRUCTURE" in reason_upper or "TERM" in reason_upper or "INVERSION" in reason_upper:
        return ExitType.STRUCTURE_TERM_INVERSION
    if "LOSS" in reason_upper or "STOP" in reason_upper:
        if "CATASTROPHIC" in reason_upper or "EMERGENCY" in reason_upper:
            return ExitType.LOSS_CATASTROPHIC
        return ExitType.LOSS_THRESHOLD_BREACH
    if "MANUAL" in reason_upper or "DISCRETION" in reason_upper:
        return ExitType.PLANNED_MANUAL_DISCRETION
    
    # Default based on PnL
    if pnl > 0:
        return ExitType.PLANNED_TP_PROFIT_TARGET
    else:
        return ExitType.LOSS_THRESHOLD_BREACH


# ============================================================================
# Integration Helper
# ============================================================================

def create_post_mortem_from_closed_trade(
    trade_dict: Dict[str, Any],
    regime_at_exit: Optional[VolatilityRegime],
    robustness_at_entry: float,
    pm_manager: PostMortemManager,
    signal_context: Optional[Dict[str, Any]] = None,
) -> TradePostMortem:
    """
    Create a post-mortem from a closed trade record.
    
    This is the integration point between TradeLogManager and PostMortemManager.
    Call this when a trade is fully closed.
    """
    trade_id = trade_dict.get("trade_id", "")
    variant_id = trade_dict.get("variant_id", "")
    
    # Parse regime at entry
    regime_at_entry = None
    if trade_dict.get("regime_at_open"):
        try:
            regime_at_entry = VolatilityRegime(trade_dict["regime_at_open"])
        except:
            pass
    
    # Calculate gross PnL from legs
    gross_pnl = 0.0
    for leg in trade_dict.get("legs", []):
        gross_pnl += leg.get("realized_pnl", 0.0)
    
    # Create base post-mortem
    pm = pm_manager.create_post_mortem(
        trade_id=trade_id,
        variant_id=variant_id,
        regime_at_entry=regime_at_entry,
        regime_at_exit=regime_at_exit,
        robustness_at_entry=robustness_at_entry,
        gross_pnl=gross_pnl,
    )
    
    # Add signal context if available
    if signal_context:
        long_exec = ExecutionComparison(
            signal_mid_price=signal_context.get("long_signal_mid"),
            signal_bid=signal_context.get("long_signal_bid"),
            signal_ask=signal_context.get("long_signal_ask"),
            signal_spread_pct=signal_context.get("long_signal_spread_pct"),
            signal_robustness=signal_context.get("robustness_score"),
        )
        pm.long_leg_execution = long_exec
        
        if signal_context.get("short_signal_mid"):
            short_exec = ExecutionComparison(
                signal_mid_price=signal_context.get("short_signal_mid"),
                signal_bid=signal_context.get("short_signal_bid"),
                signal_ask=signal_context.get("short_signal_ask"),
                signal_spread_pct=signal_context.get("short_signal_spread_pct"),
            )
            pm.short_leg_execution = short_exec
    
    # Infer exit type
    exit_type = infer_exit_type(
        regime_at_entry=regime_at_entry,
        regime_at_exit=regime_at_exit,
        exit_reason_from_trade=trade_dict.get("notes", ""),
        pnl=gross_pnl,
        weeks_held=trade_dict.get("weeks_held", 0),
        max_hold_weeks=trade_dict.get("max_hold_weeks", 26),
    )
    pm.classify_exit(exit_type)
    
    pm_manager.update_post_mortem(pm)
    return pm
