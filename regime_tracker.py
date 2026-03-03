"""Regime Snapshot Engine — VIX 5%% Weekly Suite"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json

SNAPSHOTS_PATH = Path.home() / ".vix_suite" / "regime_snapshots.json"

@dataclass
class RegimeSnapshot:
    date: str
    uvxy: float
    vix: float
    vix_percentile_1y: float
    vix_percentile_3y: float = 0.0
    iv_ratio: float = 0.0
    term_structure: str = ""
    regime_label: str = ""
    percentile_bucket: str = ""
    vix3m: float = 0.0
    vix9d: float = 0.0
    notes: str = ""
    created_at: str = ""

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

class RegimeTracker:
    def __init__(self):
        self.snapshots: List[RegimeSnapshot] = []
        self._load()

    def _load(self):
        if SNAPSHOTS_PATH.exists():
            try:
                data = json.loads(SNAPSHOTS_PATH.read_text())
                self.snapshots = [RegimeSnapshot.from_dict(s) for s in data.get("snapshots",[])]
            except Exception: self.snapshots = []

    def _save(self):
        SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_PATH.write_text(json.dumps({"snapshots":[s.to_dict() for s in self.snapshots]},indent=2))

    def capture_today(self, uvxy, vix, vix_percentile_1y, iv_ratio=0.0,
                      term_structure="", regime_label="", vix3m=0.0, vix9d=0.0):
        from datetime import date, datetime
        today = date.today().isoformat()
        p = vix_percentile_1y*100 if vix_percentile_1y<=1.0 else vix_percentile_1y
        if p<=5: bucket="0-5%% (Extreme Calm)"
        elif p<=20: bucket="5-20%% (Calm)"
        elif p<=50: bucket="20-50%% (Neutral)"
        elif p<=80: bucket="50-80%% (Elevated)"
        else: bucket="80-100%% (Panic)"
        snap = RegimeSnapshot(date=today,uvxy=round(uvxy,2),vix=round(vix,2),
            vix_percentile_1y=round(vix_percentile_1y,4),iv_ratio=round(iv_ratio,4),
            term_structure=term_structure,regime_label=regime_label,percentile_bucket=bucket,
            vix3m=round(vix3m,2),vix9d=round(vix9d,2),created_at=datetime.now().isoformat())
        self.snapshots = [s for s in self.snapshots if s.date != today]
        self.snapshots.append(snap)
        self._save()
        return snap

    def recent(self, n=30): return sorted(self.snapshots,key=lambda s:s.date,reverse=True)[:n]
    def by_regime(self, r): return [s for s in self.snapshots if s.regime_label.upper()==r.upper()]

_inst = None
def get_regime_tracker():
    global _inst
    if _inst is None: _inst = RegimeTracker()
    return _inst

def capture_regime_snapshot(uvxy, vix, pct_1y, regime=""):
    try:
        import yfinance as yf
        vix3m = float(yf.Ticker("^VIX3M").history(period="2d")["Close"].iloc[-1])
        vix9d = float(yf.Ticker("^VIX9D").history(period="2d")["Close"].iloc[-1])
        iv_ratio = round(vix/vix3m,4) if vix3m>0 else 1.0
        term = "Backwardation" if iv_ratio>1.05 else "Contango" if iv_ratio<0.95 else "Flat"
    except Exception: vix3m=vix9d=iv_ratio=0.0; term=""
    return get_regime_tracker().capture_today(uvxy=uvxy,vix=vix,vix_percentile_1y=pct_1y,
        iv_ratio=iv_ratio,term_structure=term,regime_label=regime,vix3m=vix3m,vix9d=vix9d)
