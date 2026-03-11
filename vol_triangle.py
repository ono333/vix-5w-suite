"""
Volatility Triangle — VIX/VVIX/Term Structure analytics.
Captures daily snapshots for proprietary historical dataset.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

STORAGE_DIR   = Path.home() / ".vix_suite"
SNAPSHOT_FILE = STORAGE_DIR / "vol_snapshots.json"
STORAGE_DIR.mkdir(exist_ok=True)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VolSnapshot:
    date:             str
    captured_at:      str

    # Raw levels
    vix:              float
    vvix:             float
    vix3m:            float
    vix9d:            float
    uvxy:             float

    # Percentiles (252-day rank)
    vix_pct:          float   # 0.0–1.0
    vvix_pct:         float
    uvxy_pct:         float

    # Term structure
    iv_ratio:         float   # VIX / VIX3M
    term_structure:   str     # Contango / Backwardation / Flat

    # Triangle composite
    spike_score:      float   # 0–100
    spike_label:      str     # Calm / Expansion / Possible Peak / Exhaustion

    # Regime flags
    regime:           str
    crisis_harvest:   bool    # VIX > 35
    collapse_flag:    bool    # VIX falling 3 consecutive days
    vvix_leads:       bool    # VVIX percentile > VIX percentile + 15%

    # 5-day momentum
    vix_slope_5d:     float   # % change
    vvix_slope_5d:    float

    # 1-day changes (for confirmation logic)
    vix_1d_change:    float = 0.0   # VIX % change today vs yesterday
    vvix_1d_change:   float = 0.0   # VVIX % change today vs yesterday

    # Aftershock risk
    aftershock_risk:  str   = "LOW"   # LOW / MEDIUM / HIGH
    aftershock_pct:   float = 0.0     # 0-100 probability

    # UVXY decay pressure
    uvxy_decay_pressure: str  = "MEDIUM"  # LOW / MEDIUM / HIGH
    uvxy_decay_score:    float = 0.0      # 0-100


# ── Snapshot store ────────────────────────────────────────────────────────────

class VolSnapshotStore:
    def __init__(self):
        self.snapshots: list[VolSnapshot] = []
        self._load()

    def _load(self):
        if SNAPSHOT_FILE.exists():
            try:
                raw = json.loads(SNAPSHOT_FILE.read_text())
                self.snapshots = [VolSnapshot(**r) for r in raw]
            except Exception as e:
                print(f"VolSnapshotStore load error: {e}")
                self.snapshots = []

    def _save(self):
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, bool):
                return bool(obj)
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            return obj
        SNAPSHOT_FILE.write_text(
            json.dumps([_clean(asdict(s)) for s in self.snapshots], indent=2)
        )

    def add(self, snap: VolSnapshot):
        # Replace if same date already exists
        self.snapshots = [s for s in self.snapshots if s.date != snap.date]
        self.snapshots.append(snap)
        self.snapshots.sort(key=lambda s: s.date)
        self._save()

    def latest(self) -> Optional[VolSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    def recent(self, n: int = 30) -> list[VolSnapshot]:
        return self.snapshots[-n:]

    def today_captured(self) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        return any(s.date == today for s in self.snapshots)

    def as_series(self, field: str) -> list:
        return [(s.date, getattr(s, field)) for s in self.snapshots]


# ── Data fetcher ──────────────────────────────────────────────────────────────

def _fetch_all() -> dict:
    """Fetch all triangle components from Yahoo Finance."""
    import yfinance as yf
    from datetime import datetime, timedelta

    end   = datetime.now()
    start = end - timedelta(days=400)   # enough for 252-day percentile

    results = {}
    for ticker in ["^VIX", "^VVIX", "^VIX3M", "^VIX9D", "UVXY"]:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, 'get_level_values'):
                try:
                    df.columns = df.columns.get_level_values(0)
                except Exception:
                    pass
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            series = df[col].dropna()
            results[ticker] = series
        except Exception as e:
            print(f"  ⚠️ {ticker}: {e}")
            results[ticker] = None

    return results


def _rank_percentile(series, today_val: float) -> float:
    """252-day rank-based percentile."""
    if series is None or len(series) < 5:
        return 0.5
    window = series.iloc[-252:] if len(series) >= 252 else series
    return round(float((window < today_val).sum()) / len(window), 4)


def _slope_5d(series) -> float:
    """5-day % change."""
    if series is None or len(series) < 5:
        return 0.0
    vals = series.iloc[-5:].values
    return round((float(vals[-1]) - float(vals[0])) / float(vals[0]), 4)


def _spike_score(vix_pct: float, vvix_pct: float, uvxy_mom: float,
                 iv_ratio: float, vix_slope: float) -> tuple[float, str]:
    """
    SpikeScore = 0.30*VIX_pct + 0.25*VVIX_pct + 0.20*UVXY_mom
               + 0.15*term_structure + 0.10*RSI_VIX_proxy
    """
    # Normalize UVXY momentum: +20% → 1.0, -20% → 0.0
    mom_norm = max(0.0, min(1.0, (uvxy_mom + 0.20) / 0.40))

    # Term structure: backwardation = high stress (ratio > 1 → high score)
    ts_score = max(0.0, min(1.0, (iv_ratio - 0.85) / 0.30))

    # VIX slope proxy for RSI (positive slope = rising = stress)
    slope_norm = max(0.0, min(1.0, (vix_slope + 0.10) / 0.20))

    raw = (0.30 * vix_pct
         + 0.25 * vvix_pct
         + 0.20 * mom_norm
         + 0.15 * ts_score
         + 0.10 * slope_norm)

    score = round(raw * 100, 1)

    if score < 40:   label = "Calm"
    elif score < 60: label = "Expansion"
    elif score < 80: label = "Possible Peak"
    else:            label = "Spike Exhaustion"

    return score, label


def _regime_from_pct(pct: float) -> str:
    if pct <= 0.35: return "CALM"
    if pct <= 0.55: return "RISING"
    if pct <= 0.75: return "STRESSED"
    if pct <= 0.90: return "EXPANSION"
    return "EXTREME"


def _collapse_flag(vix_series) -> bool:
    """True if VIX has fallen 3 consecutive days."""
    if vix_series is None or len(vix_series) < 4:
        return False
    recent = vix_series.iloc[-4:].values
    return (recent[-1] < recent[-2] < recent[-3] < recent[-4])


# ── Main capture ──────────────────────────────────────────────────────────────

def _aftershock_risk(vix_pct: float, vvix_pct: float,
                     spike_score: float, vix_slope: float) -> tuple[str, float]:
    """
    Aftershock probability — secondary spike risk after initial peak.
    Triggered when volatility peaked but conditions still unstable.
    """
    score = 0.0
    if vix_pct > 0.90:   score += 30
    elif vix_pct > 0.80: score += 15
    if vvix_pct > 0.90:  score += 25
    elif vvix_pct > 0.80:score += 12
    if spike_score > 80: score += 25
    elif spike_score > 70: score += 12
    if vix_slope > 0.15: score += 20   # still rising fast
    elif vix_slope > 0.05: score += 10

    score = min(100, score)
    if score >= 65:   risk = "HIGH"
    elif score >= 35: risk = "MEDIUM"
    else:             risk = "LOW"
    return risk, round(score, 1)


def _uvxy_decay_pressure(iv_ratio: float, vix_slope: float,
                          vix_pct: float) -> tuple[str, float]:
    """
    UVXY structural decay pressure from contango bleed + roll yield.
    HIGH decay = UVXY likely to fall even if VIX stays flat.
    """
    score = 0.0
    # Contango: iv_ratio < 1.0 means VIX < VIX3M → futures in contango → decay pressure
    if iv_ratio < 0.90:   score += 40   # strong contango → high decay
    elif iv_ratio < 0.95: score += 25
    elif iv_ratio < 1.00: score += 10
    # VIX falling momentum adds to decay
    if vix_slope < -0.05: score += 30
    elif vix_slope < 0:   score += 15
    # Very high percentile = extended, mean reversion likely
    if vix_pct > 0.90:    score += 20
    elif vix_pct > 0.80:  score += 10

    score = min(100, score)
    if score >= 60:   pressure = "HIGH"
    elif score >= 30: pressure = "MEDIUM"
    else:             pressure = "LOW"
    return pressure, round(score, 1)


def capture_snapshot(force: bool = False) -> VolSnapshot:
    """
    Fetch live data and store today's snapshot.
    Builds our proprietary historical dataset day by day.
    """
    store = VolSnapshotStore()

    if store.today_captured() and not force:
        print(f"📊 Vol snapshot already captured today — use force=True to refresh")
        return store.latest()

    print("📊 Capturing Volatility Triangle snapshot...")
    data = _fetch_all()

    vix_s   = data.get("^VIX")
    vvix_s  = data.get("^VVIX")
    vix3m_s = data.get("^VIX3M")
    vix9d_s = data.get("^VIX9D")
    uvxy_s  = data.get("UVXY")

    vix   = float(vix_s.iloc[-1])   if vix_s   is not None else 20.0
    vvix  = float(vvix_s.iloc[-1])  if vvix_s  is not None else 90.0
    vix3m = float(vix3m_s.iloc[-1]) if vix3m_s is not None else 20.0
    vix9d = float(vix9d_s.iloc[-1]) if vix9d_s is not None else 20.0
    uvxy  = float(uvxy_s.iloc[-1])  if uvxy_s  is not None else 30.0

    vix_pct  = _rank_percentile(vix_s,  vix)
    vvix_pct = _rank_percentile(vvix_s, vvix)
    uvxy_pct = _rank_percentile(uvxy_s, uvxy)

    iv_ratio = round(vix / vix3m, 3) if vix3m > 0 else 1.0
    if   iv_ratio < 0.90: term = "Strong Contango"
    elif iv_ratio < 1.00: term = "Mild Contango"
    elif iv_ratio < 1.05: term = "Flat"
    elif iv_ratio < 1.15: term = "Mild Backwardation"
    else:                 term = "Strong Backwardation"

    vix_slope  = _slope_5d(vix_s)
    vvix_slope = _slope_5d(vvix_s)
    uvxy_mom   = _slope_5d(uvxy_s)

    score, label = _spike_score(vix_pct, vvix_pct, uvxy_mom, iv_ratio, vix_slope)

    regime        = _regime_from_pct(vix_pct)
    crisis        = vix > 35.0
    collapse      = _collapse_flag(vix_s)
    vvix_leads    = (vvix_pct - vix_pct) > 0.15

    # 1-day changes
    vix_1d  = round((float(vix_s.iloc[-1]) - float(vix_s.iloc[-2])) / float(vix_s.iloc[-2]), 4) if vix_s is not None and len(vix_s) >= 2 else 0.0
    vvix_1d = round((float(vvix_s.iloc[-1]) - float(vvix_s.iloc[-2])) / float(vvix_s.iloc[-2]), 4) if vvix_s is not None and len(vvix_s) >= 2 else 0.0

    # Aftershock + decay
    aftershock_risk, aftershock_pct = _aftershock_risk(vix_pct, vvix_pct, score, vix_slope)
    decay_pressure, decay_score     = _uvxy_decay_pressure(iv_ratio, vix_slope, vix_pct)

    snap = VolSnapshot(
        date           = datetime.now().strftime("%Y-%m-%d"),
        captured_at    = datetime.now().isoformat(),
        vix=vix, vvix=vvix, vix3m=vix3m, vix9d=vix9d, uvxy=uvxy,
        vix_pct=vix_pct, vvix_pct=vvix_pct, uvxy_pct=uvxy_pct,
        iv_ratio=iv_ratio, term_structure=term,
        spike_score=score, spike_label=label,
        regime=regime,
        crisis_harvest=crisis,
        collapse_flag=collapse,
        vvix_leads=vvix_leads,
        vix_slope_5d=vix_slope,
        vvix_slope_5d=vvix_slope,
        vix_1d_change=vix_1d,
        vvix_1d_change=vvix_1d,
        aftershock_risk=aftershock_risk,
        aftershock_pct=aftershock_pct,
        uvxy_decay_pressure=decay_pressure,
        uvxy_decay_score=decay_score,
    )

    store.add(snap)
    print(f"   ✅ Saved: VIX={vix:.1f} ({vix_pct:.0%}) VVIX={vvix:.0f} ({vvix_pct:.0%})")
    print(f"   Spike Score: {score} ({label})  Crisis: {crisis}  Collapse: {collapse}")
    return snap


def get_latest_snapshot() -> Optional[VolSnapshot]:
    return VolSnapshotStore().latest()


def get_snapshot_store() -> VolSnapshotStore:
    return VolSnapshotStore()

