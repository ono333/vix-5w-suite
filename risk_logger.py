#!/usr/bin/env python3
"""risk_logger.py — daily volatility risk-measure logger for the vix_suite DB.

Logs, per trading day, the measures that gate short-premium edge:
  vix               CBOE VIX (30d implied)
  vix3m             CBOE VIX3M     -> slope_ratio = vix/vix3m  (>1 = backwardation)
  vix9d             CBOE VIX9D     -> ratio_9d   = vix9d/vix   (>1 = near-term stress)
  vvix              CBOE VVIX      -> vvix_ratio = vvix/vix    (vol-of-vol richness)
  spx / rv21        S&P 500 close and 21d realized vol (annualized, %)
  vrp               vix - rv21     (volatility risk premium; thin = underpaid)
  spread_flag_rate  share of own reprice/placement quote rows flagged wide-spread
                    that day (market-maker stress from your own database)

Data: yfinance (^VIX, ^VVIX, ^GSPC) + CBOE daily CSVs (VIX3M, VIX9D — Yahoo
lags these by ~a week) + own quotes table.
Writes only the risk_measures table in ~/.vix_suite/market_data.db. Additive;
never touches existing tables. Idempotent — safe to run any day, any number
of times (rows are upserted by date; spread_flag_rate is preserved).

Usage:
  python3 risk_logger.py            # upsert last ~3 months (the daily run)
  python3 risk_logger.py --seed     # backfill ~15 years (enables percentiles)
  python3 risk_logger.py --report   # latest values + historical percentiles

Suggested cron (after the 16:45 vol_triangle snapshot):
  50 16 * * 1-5 cd /home/shin/vix_suite && /home/shin/vix_suite/venv/bin/python3 risk_logger.py >> /home/shin/vix_suite/risk_logger.log 2>&1
"""

import os
import sqlite3
import sys
from datetime import datetime

DB_PATH = os.path.expanduser("~/.vix_suite/market_data.db")

TICKERS = {"vix": "^VIX", "vix3m": "^VIX3M", "vix9d": "^VIX9D",
           "vvix": "^VVIX", "spx": "^GSPC"}

MEASURES = ("vix", "slope_ratio", "ratio_9d", "vvix_ratio", "vrp",
            "spread_flag_rate")

SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_measures(
  date TEXT PRIMARY KEY,
  vix REAL, vix3m REAL, vix9d REAL, vvix REAL,
  spx REAL, rv21 REAL, vrp REAL,
  slope_ratio REAL, ratio_9d REAL, vvix_ratio REAL,
  spread_flag_rate REAL);
"""

# Interpretation hints shown in --report (measure, high-percentile meaning)
HINTS = {
    "vix":              "raw fear level",
    "slope_ratio":      "high = term structure flat/inverted -> carry edge thin",
    "ratio_9d":         "high = near-term stress arriving (9d > 30d)",
    "vvix_ratio":       "high = hedging demand rich under calm surface",
    "vrp":              "LOW percentile = market underpaying for spike risk",
    "spread_flag_rate": "high = market makers nervous (own-quote spreads wide)",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.executescript(SCHEMA)
    return conn


# ------------------------------------------------------------- fetching -----
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"


def fetch_cboe_close(name):
    """CLOSE series from a CBOE daily-prices CSV (DATE,OPEN,HIGH,LOW,CLOSE)."""
    import io
    import pandas as pd
    import requests
    r = requests.get(CBOE_URL.format(name), timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index("DATE")["CLOSE"]


def fetch_closes(period):
    """Daily closes as a DataFrame with friendly column names.
    VIX/VVIX/SPX via yfinance; VIX3M/VIX9D via CBOE CSVs (Yahoo lags them)."""
    import pandas as pd
    import yfinance as yf
    yf_keys = ("vix", "vvix", "spx")
    df = yf.download([TICKERS[k] for k in yf_keys], period=period,
                     interval="1d", progress=False, auto_adjust=False)["Close"]
    df = df.rename(columns={TICKERS[k]: k for k in yf_keys})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    for key, sym in (("vix3m", "VIX3M"), ("vix9d", "VIX9D")):
        try:
            s = fetch_cboe_close(sym)
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            df[key] = s.reindex(df.index)
        except Exception as e:
            log(f"risk: CBOE {sym} fetch failed ({e}); column left empty")
            df[key] = float("nan")
    return df.dropna(how="all")


# ------------------------------------------------------------ computing -----
def compute(df):
    """Add derived measures. Pure function of the closes DataFrame."""
    import numpy as np
    out = df.copy()
    r = np.log(out["spx"] / out["spx"].shift(1))
    out["rv21"] = r.rolling(21).std() * np.sqrt(252) * 100
    out["vrp"] = out["vix"] - out["rv21"]
    out["slope_ratio"] = out["vix"] / out["vix3m"]
    out["ratio_9d"] = out["vix9d"] / out["vix"]
    out["vvix_ratio"] = out["vvix"] / out["vix"]
    return out


def spread_flag_rate(conn, day_iso):
    """Share of that day's reprice/placement quote rows flagged wide-spread.
    Uses the existing quotes table read-only; returns None if the table has
    no recognizable timestamp column or no rows that day."""
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(quotes)")]
    except sqlite3.Error:
        return None
    tcol = next((c for c in cols if c.lower() in
                 ("ts", "timestamp", "created_at", "time", "date", "dt")), None)
    if not tcol:
        return None
    rows = conn.execute(
        f"SELECT decision, COUNT(*) FROM quotes "
        f"WHERE substr({tcol},1,10)=? AND context IN ('reprice','placement') "
        f"GROUP BY decision", (day_iso,)).fetchall()
    total = sum(n for _, n in rows)
    if not total:
        return None
    flagged = sum(n for d, n in rows if "wide_spread" in (d or ""))
    return round(flagged / total, 3)


def pct_rank(vals, x):
    vals = [v for v in vals if v is not None]
    if len(vals) < 60 or x is None:
        return None
    below = sum(1 for v in vals if v < x)
    return round(100.0 * below / len(vals), 1)


# ------------------------------------------------------------- running ------
def run(period="3mo"):
    conn = connect()
    df = compute(fetch_closes(period))
    n = 0
    for idx, row in df.iterrows():
        day = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
        vals = {k: (None if _isnan(row.get(k)) else round(float(row[k]), 4))
                for k in ("vix", "vix3m", "vix9d", "vvix", "spx", "rv21",
                          "vrp", "slope_ratio", "ratio_9d", "vvix_ratio")}
        if vals["vix"] is None:
            continue
        conn.execute(
            "INSERT INTO risk_measures(date,vix,vix3m,vix9d,vvix,spx,rv21,vrp,"
            "slope_ratio,ratio_9d,vvix_ratio) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(date) DO UPDATE SET vix=excluded.vix,"
            "vix3m=excluded.vix3m,vix9d=excluded.vix9d,vvix=excluded.vvix,"
            "spx=excluded.spx,rv21=excluded.rv21,vrp=excluded.vrp,"
            "slope_ratio=excluded.slope_ratio,ratio_9d=excluded.ratio_9d,"
            "vvix_ratio=excluded.vvix_ratio",
            (day, vals["vix"], vals["vix3m"], vals["vix9d"], vals["vvix"],
             vals["spx"], vals["rv21"], vals["vrp"], vals["slope_ratio"],
             vals["ratio_9d"], vals["vvix_ratio"]))
        n += 1
    # own-DB spread stress for the most recent ~10 stored days
    recent = [r[0] for r in conn.execute(
        "SELECT date FROM risk_measures ORDER BY date DESC LIMIT 10")]
    for day in recent:
        rate = spread_flag_rate(conn, day)
        if rate is not None:
            conn.execute(
                "UPDATE risk_measures SET spread_flag_rate=? WHERE date=?",
                (rate, day))
    conn.commit()
    last = conn.execute(
        "SELECT date,vix,slope_ratio,ratio_9d,vvix_ratio,vrp,spread_flag_rate "
        "FROM risk_measures ORDER BY date DESC LIMIT 1").fetchone()
    conn.close()
    if last:
        d, vix, sl, r9, vv, vrp, sr = last
        log(f"risk: upserted {n} rows; latest {d}: VIX={vix} "
            f"slope={_f(sl)} 9d/30d={_f(r9)} vvix/vix={_f(vv)} "
            f"vrp={_f(vrp)} spread_rate={sr if sr is not None else 'n/a'}")


def _isnan(x):
    try:
        return x is None or x != x
    except Exception:
        return True


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


# -------------------------------------------------------------- report ------
def report():
    conn = connect()
    last = conn.execute(
        "SELECT * FROM risk_measures ORDER BY date DESC LIMIT 1").fetchone()
    if not last:
        print("risk_measures is empty — run with --seed first")
        return
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM risk_measures LIMIT 1").description]
    latest = dict(zip(cols, last))
    nrows = conn.execute("SELECT COUNT(*) FROM risk_measures").fetchone()[0]
    print(f"risk_measures — {nrows} days of history; latest {latest['date']}\n")
    print(f"{'measure':<18}{'latest':>10}{'pctile':>9}   interpretation")
    for m in MEASURES:
        hist = [r[0] for r in conn.execute(
            f"SELECT {m} FROM risk_measures WHERE {m} IS NOT NULL")]
        x = latest.get(m)
        p = pct_rank(hist, x)
        print(f"{m:<18}{_f(x) if x is not None else 'n/a':>10}"
              f"{(str(p) if p is not None else 'n/a'):>9}   {HINTS[m]}")
    conn.close()


# ---------------------------------------------------------------- main ------
if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--seed" in sys.argv:
        run(period="15y")
    else:
        run(period="3mo")
