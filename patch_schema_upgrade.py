"""
Schema Upgrade: VIX 5% Weekly Suite — Trade Log → Regime Analytics Engine
Per ChatGPT specification.
"""
from pathlib import Path
import json

def patch(path, old, new, label):
    src = path.read_text()
    if old in src:
        path.write_text(src.replace(old, new))
        print(f"  ✓ {label}")
        return True
    print(f"  ✗ NOT FOUND: {label}")
    return False

# ══ 1. trade_log.py — Paper DiagonalPosition upgrade ══
print("\n── trade_log.py ──")
tl = Path("trade_log.py")
src = tl.read_text()

old = '''    entry_regime: str = ""
    entry_vix_level: float = 0.0
    entry_percentile: float = 0.0'''
new = '''    entry_regime:           str   = ""
    entry_vix_level:        float = 0.0
    entry_uvxy:             float = 0.0
    entry_percentile:       float = 0.0
    entry_iv_ratio:         float = 0.0
    entry_term_structure:   str   = ""
    long_delta_entry:       float = 0.0
    initial_short_delta:    float = 0.0
    long_fill_price:        float = 0.0
    broker:                 str   = "Paper"
    account_id:             str   = "PAPER"'''
if old in src: src = src.replace(old, new); print("  ✓ DiagonalPosition: entry fields")
else: print("  ✗ DiagonalPosition entry fields not found")

old2 = '''    @property
    def long_pnl(self) -> float:'''
new2 = '''    @staticmethod
    def percentile_to_bucket(pct: float) -> str:
        p = pct * 100 if pct <= 1.0 else pct
        if p <= 5:    return "0-5% (Extreme Calm)"
        elif p <= 20: return "5-20% (Calm)"
        elif p <= 50: return "20-50% (Neutral)"
        elif p <= 80: return "50-80% (Elevated)"
        else:         return "80-100% (Panic)"

    @property
    def percentile_bucket(self) -> str:
        return self.percentile_to_bucket(self.entry_percentile)

    @property
    def lifecycle_summary(self) -> dict:
        from datetime import date
        try:
            entry = date.fromisoformat(str(self.entry_date)[:10])
            exit_d = date.fromisoformat(str(self.exit_date)[:10]) if self.exit_date else date.today()
            days = (exit_d - entry).days
        except Exception:
            days = 0
        creds = sum(float(r.new_credit) for r in self.roll_history)
        if self.short_legs: creds += float(self.short_legs[0].entry_credit)
        cost = float(self.long_entry_price) * float(self.contracts) * 100
        rc = len(self.roll_history)
        arc = (sum(float(r.roll_credit) for r in self.roll_history)/rc if rc else 0.0)
        cvr = (creds/cost*100) if cost else 0.0
        cnvx = (float(self.long_delta_entry)/float(self.initial_short_delta)
                if self.initial_short_delta > 0 else 0.0)
        teff = (creds/days) if days else 0.0
        return {
            "position_id": self.position_id, "strategy": self.variant_id,
            "account_type": getattr(self,"broker","Paper"), "days_in_trade": days,
            "total_short_credit": round(creds,2), "total_long_cost": round(cost,2),
            "net_realized_pnl": round(float(getattr(self,"total_pnl",0)),2),
            "roll_count": rc, "avg_roll_credit": round(arc,4),
            "coverage_ratio_pct": round(cvr,1), "convexity_ratio": round(cnvx,3),
            "time_efficiency": round(teff,4),
            "entry_percentile": round(float(self.entry_percentile)*100,1),
            "entry_percentile_bucket": self.percentile_bucket,
            "entry_regime": self.entry_regime,
            "entry_iv_ratio": round(float(self.entry_iv_ratio),3),
            "entry_term_structure": self.entry_term_structure,
            "exit_reason": getattr(self,"exit_reason",""), "status": self.status,
        }

    @property
    def long_pnl(self) -> float:'''
if old2 in src: src = src.replace(old2, new2); print("  ✓ DiagonalPosition: percentile_bucket + lifecycle_summary")
else: print("  ✗ DiagonalPosition long_pnl not found")

old3 = '''    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None'''
new3 = '''    delta_at_entry:       float = 0.0
    iv_at_entry:          float = 0.0
    vix_percentile_entry: float = 0.0
    exit_date:    Optional[str]   = None
    exit_price:   Optional[float] = None
    exit_reason:  Optional[str]   = None'''
if old3 in src: src = src.replace(old3, new3); print("  ✓ ShortLeg: delta/iv entry fields")
else: print("  ✗ ShortLeg exit fields not found")

tl.write_text(src)

# ══ 2. real_trade_log.py ══
print("\n── real_trade_log.py ──")
rl = Path("real_trade_log.py")
src = rl.read_text()

old = '''    entry_regime:      str
    entry_vix_level:   float
    entry_percentile:  float'''
new = '''    entry_regime:           str
    entry_vix_level:        float
    entry_uvxy:             float = 0.0
    entry_percentile:       float
    entry_iv_ratio:         float = 0.0
    entry_term_structure:   str   = ""
    long_delta_entry:       float = 0.0
    initial_short_delta:    float = 0.0'''
if old in src: src = src.replace(old, new); print("  ✓ RealDiagonalPosition: entry fields")
else: print("  ✗ RealDiagonalPosition fields not found")

old2 = '''    @property
    def long_pnl(self) -> float:'''
new2 = '''    @staticmethod
    def percentile_to_bucket(pct: float) -> str:
        p = pct * 100 if pct <= 1.0 else pct
        if p <= 5:    return "0-5% (Extreme Calm)"
        elif p <= 20: return "5-20% (Calm)"
        elif p <= 50: return "20-50% (Neutral)"
        elif p <= 80: return "50-80% (Elevated)"
        else:         return "80-100% (Panic)"

    @property
    def percentile_bucket(self) -> str:
        return self.percentile_to_bucket(self.entry_percentile)

    @property
    def lifecycle_summary(self) -> dict:
        from datetime import date
        try:
            entry = date.fromisoformat(str(self.entry_date)[:10])
            exit_d = date.fromisoformat(str(self.close_date)[:10]) if self.close_date else date.today()
            days = (exit_d - entry).days
        except Exception:
            days = 0
        creds  = float(self.net_short_credits)
        cost   = float(self.long_fill_price) * float(self.contracts) * 100
        net    = float(self.total_pnl)
        rc     = len(self.roll_history)
        arc    = (sum(float(r.roll_credit) for r in self.roll_history)/rc if rc else 0.0)
        cvr    = (creds/cost*100) if cost else 0.0
        cnvx   = (float(self.long_delta_entry)/float(self.initial_short_delta)
                  if self.initial_short_delta > 0 else 0.0)
        teff   = (creds/days) if days else 0.0
        tc     = float(self.total_commissions)
        ts     = float(self.total_slippage)
        return {
            "position_id": self.position_id, "strategy": self.variant_id,
            "account_type": f"Real ({self.broker})", "account_id": self.account_id,
            "days_in_trade": days, "total_short_credit": round(creds,2),
            "total_long_cost": round(cost,2), "net_realized_pnl": round(net,2),
            "total_commissions": round(tc,2), "total_slippage": round(ts,2),
            "net_after_costs": round(net-tc-ts,2),
            "roll_count": rc, "avg_roll_credit": round(arc,4),
            "coverage_ratio_pct": round(cvr,1), "convexity_ratio": round(cnvx,3),
            "time_efficiency": round(teff,4),
            "entry_percentile": round(float(self.entry_percentile)*100,1),
            "entry_percentile_bucket": self.percentile_bucket,
            "entry_regime": self.entry_regime,
            "entry_iv_ratio": round(float(self.entry_iv_ratio),3),
            "entry_term_structure": self.entry_term_structure,
            "exit_reason": getattr(self,"close_reason",""), "status": self.status,
        }

    @property
    def long_pnl(self) -> float:'''
if old2 in src: src = src.replace(old2, new2); print("  ✓ RealDiagonalPosition: percentile_bucket + lifecycle_summary")
else: print("  ✗ RealDiagonalPosition long_pnl not found")

old3 = '''    notes:            str   = ""
    created_at:       str   = ""'''
new3 = '''    delta_at_entry:       float = 0.0
    iv_at_entry:          float = 0.0
    vix_percentile_entry: float = 0.0
    notes:                str   = ""
    created_at:           str   = ""'''
if old3 in src: src = src.replace(old3, new3); print("  ✓ RealShortLeg: delta/iv entry fields")
else: print("  ✗ RealShortLeg notes not found")

rl.write_text(src)

# ══ 3. regime_tracker.py — new file ══
print("\n── regime_tracker.py (new) ──")
Path("regime_tracker.py").write_text('''"""Regime Snapshot Engine — VIX 5%% Weekly Suite"""
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
''')
print("  ✓ regime_tracker.py created")

# ══ 4. Hook into daily_signal.py ══
print("\n── daily_signal.py ──")
ds = Path("daily_signal.py")
src = ds.read_text()
old = "        # 5b. Check and send automated roll alerts"
new = '''        # 5a. Capture daily regime snapshot
        try:
            from regime_tracker import capture_regime_snapshot
            snap = capture_regime_snapshot(uvxy=uvxy_price,vix=vix_level,
                pct_1y=percentile,regime=regime_state.regime.value.upper())
            print(f"   📸 Regime: {snap.regime_label} ({snap.vix_percentile_1y:.0%}) {snap.percentile_bucket}")
        except Exception as _se: print(f"   ⚠️ Snapshot: {_se}")

        # 5b. Check and send automated roll alerts'''
if old in src: src=src.replace(old,new); ds.write_text(src); print("  ✓ Regime snapshot hooked into main()")
else: print("  ✗ main() anchor not found")

# ══ 5. Add Lifecycle tab to real trade log in app.py ══
print("\n── app.py ──")
app = Path("app.py")
src = app.read_text()

old = '''    tab_open, tab_new, tab_history, tab_analytics = st.tabs([
        "📋 Open Positions", "➕ New Entry",
        "📊 History", "📈 Analytics"
    ])'''
new = '''    tab_open, tab_new, tab_history, tab_analytics, tab_lifecycle = st.tabs([
        "📋 Open Positions", "➕ New Entry",
        "📊 History", "📈 Analytics", "🔬 Lifecycle"
    ])'''
if old in src: src=src.replace(old,new); print("  ✓ Added Lifecycle tab")
else: print("  ✗ Real trade tabs not found")

old2 = "    # ══ HISTORY ══════════════════════════════════════════════\n    with tab_history:"
new2 = '''    with tab_lifecycle:
        st.markdown("### 🔬 Lifecycle Analytics")
        st.caption("Auto-computed: convexity ratio, coverage ratio, time efficiency per ChatGPT spec.")
        import pandas as pd
        summaries = []
        for pos in rtl.diagonal_positions.values():
            try: summaries.append(pos.lifecycle_summary)
            except Exception: pass
        if summaries:
            df_lc = pd.DataFrame(summaries)
            st.dataframe(df_lc, use_container_width=True, hide_index=True)
            if "entry_percentile_bucket" in df_lc.columns and len(df_lc) > 1:
                st.markdown("#### 📊 By Percentile Bucket")
                grp = df_lc.groupby("entry_percentile_bucket").agg(
                    count=("position_id","count"),
                    avg_pnl=("net_realized_pnl","mean"),
                    avg_coverage=("coverage_ratio_pct","mean"),
                ).reset_index()
                st.dataframe(grp, use_container_width=True, hide_index=True)
        else:
            st.info("Lifecycle data populates as positions close.")
        st.markdown("---")
        st.markdown("#### 📸 Regime Snapshot History")
        try:
            from regime_tracker import get_regime_tracker
            snaps = get_regime_tracker().recent(30)
            if snaps:
                sdf = pd.DataFrame([s.to_dict() for s in snaps])
                sdf["pct"] = (sdf["vix_percentile_1y"]*100).round(1).astype(str)+"%"
                st.dataframe(sdf[["date","uvxy","vix","pct","iv_ratio","term_structure",
                                   "regime_label","percentile_bucket"]],
                             use_container_width=True, hide_index=True)
            else: st.info("Regime snapshots captured on each daily signal run.")
        except Exception as _e: st.warning(f"Regime tracker: {_e}")

    # ══ HISTORY ══════════════════════════════════════════════
    with tab_history:'''
if old2 in src: src=src.replace(old2,new2); print("  ✓ Lifecycle tab content added")
else: print("  ✗ History tab anchor not found")

app.write_text(src)

# ══ 6. Migrate existing JSON ══
print("\n── JSON migration ──")
for json_path, pos_fields, leg_fields in [
    (Path.home()/".vix_suite/real_trade_log.json",
     {"entry_uvxy":0.0,"entry_iv_ratio":0.0,"entry_term_structure":"",
      "long_delta_entry":0.0,"initial_short_delta":0.0},
     {"delta_at_entry":0.0,"iv_at_entry":0.0,"vix_percentile_entry":0.0}),
    (Path.home()/".vix_suite/trade_log.json",
     {"entry_uvxy":0.0,"entry_iv_ratio":0.0,"entry_term_structure":"",
      "long_delta_entry":0.0,"initial_short_delta":0.0,
      "long_fill_price":0.0,"broker":"Paper","account_id":"PAPER"},
     {"delta_at_entry":0.0,"iv_at_entry":0.0,"vix_percentile_entry":0.0}),
]:
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            positions = data.get("diagonal_positions",{})
            n = 0
            for pos in positions.values():
                for k,v in pos_fields.items():
                    if k not in pos: pos[k]=v; n+=1
                for leg in pos.get("short_legs",[]):
                    for k,v in leg_fields.items():
                        if k not in leg: leg[k]=v; n+=1
            json_path.write_text(json.dumps(data,indent=2))
            print(f"  ✓ {json_path.name}: {n} fields migrated")
        except Exception as e: print(f"  ✗ {json_path.name}: {e}")

print("\n✅ Schema upgrade complete")
