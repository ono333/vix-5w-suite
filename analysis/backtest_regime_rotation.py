#!/usr/bin/env python3
"""
backtest_regime_rotation.py — Strategy #5: SVXY / mid-term-vol / GLD rotation
on the VIX/VIX3M term-structure slope. Free daily closes only.

Rules (evaluated at each close, position taken at NEXT close — one-day lag,
conservative, no lookahead):
    ratio = VIX / VIX3M
    ratio <  T_CONTANGO      -> SVXY   (short-vol harvest)
    ratio >= T_BACKWARD      -> MIDVOL (long mid-term vol)
    else                     -> GLD    (neutral zone)
Grid over (T_CONTANGO, T_BACKWARD). Cost = COST_BPS per switch.

Mid-term leg defaults to VIXM (ProShares, 2011 inception) because today's VXZ
is the short-history Series B ETN. Override with --midvol VXZ if desired.

Reported per parameter pair:
    CAGR, MaxDD, Sharpe (daily, rf=0), switches/yr, time in each sleeve,
    split into pre / post Feb 28 2018 (SVXY deleveraged -1x -> -0.5x;
    pre-2018 numbers are NOT achievable with today's fund — shown for
    signal validation only).
Benchmarks: buy-hold SVXY, buy-hold GLD.

Run:  python3 backtest_regime_rotation.py [--midvol VIXM] [--cost_bps 5]
"""

import argparse
import sys
import numpy as np
import pandas as pd

DELEVER_DATE = "2018-02-28"

CONTANGO_GRID = [0.90, 0.925, 0.95]
BACKWARD_GRID = [1.00, 1.05]
COST_BPS_DEFAULT = 5.0


def fetch(midvol):
    import yfinance as yf
    tickers = ["SVXY", midvol, "GLD", "^VIX", "^VIX3M"]
    px = yf.download(tickers, period="max", auto_adjust=True, progress=False)
    close = px["Close"]
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        sys.exit(f"yfinance missing: {missing}")
    df = close.dropna(how="any")
    if len(df) < 500:
        sys.exit(f"Only {len(df)} joint rows — check tickers "
                 f"(earliest per ticker: "
                 f"{ {t: str(close[t].first_valid_index().date()) for t in tickers} })")
    print(f"Joint data: {df.index[0].date()} -> {df.index[-1].date()} "
          f"({len(df)} days)")
    return df


def stats(rets, freq=252):
    rets = rets.dropna()
    if rets.empty:
        return dict(cagr=np.nan, maxdd=np.nan, sharpe=np.nan)
    eq = (1 + rets).cumprod()
    yrs = len(rets) / freq
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    maxdd = (eq / eq.cummax() - 1).min()
    sd = rets.std()
    sharpe = rets.mean() / sd * np.sqrt(freq) if sd > 0 else np.nan
    return dict(cagr=cagr, maxdd=maxdd, sharpe=sharpe)


def backtest(df, midvol, t_c, t_b, cost_bps):
    ratio = df["^VIX"] / df["^VIX3M"]
    sleeve = pd.Series("GLD", index=df.index)
    sleeve[ratio < t_c] = "SVXY"
    sleeve[ratio >= t_b] = midvol
    sleeve = sleeve.shift(1)          # signal at close t -> hold t+1 (no lookahead)

    rets = pd.DataFrame({t: df[t].pct_change() for t in ["SVXY", midvol, "GLD"]})
    strat = pd.Series(np.nan, index=df.index)
    for t in ["SVXY", midvol, "GLD"]:
        m = sleeve == t
        strat[m] = rets[t][m]
    switches = (sleeve != sleeve.shift(1)) & sleeve.notna() & sleeve.shift(1).notna()
    strat = strat - switches.astype(float) * (cost_bps / 1e4)
    return strat, sleeve, switches


def report_row(label, strat, sleeve, switches, midvol):
    s = stats(strat)
    yrs = max(len(strat.dropna()) / 252, 1e-9)
    mix = sleeve.value_counts(normalize=True)
    return {
        "params": label,
        "CAGR%": s["cagr"] * 100, "MaxDD%": s["maxdd"] * 100,
        "Sharpe": s["sharpe"], "sw/yr": switches.sum() / yrs,
        "%SVXY": mix.get("SVXY", 0) * 100,
        "%MID": mix.get(midvol, 0) * 100,
        "%GLD": mix.get("GLD", 0) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midvol", default="VIXM")
    ap.add_argument("--cost_bps", type=float, default=COST_BPS_DEFAULT)
    args = ap.parse_args()

    df = fetch(args.midvol)
    cut = pd.Timestamp(DELEVER_DATE)

    for era, dfe in [("FULL", df),
                     ("PRE-DELEVER (signal validation only)", df[df.index < cut]),
                     ("POST-DELEVER (achievable today)", df[df.index >= cut])]:
        if len(dfe) < 300:
            continue
        print(f"\n=== {era}: {dfe.index[0].date()} -> {dfe.index[-1].date()} ===")
        rows = []
        for t_c in CONTANGO_GRID:
            for t_b in BACKWARD_GRID:
                strat, sleeve, sw = backtest(dfe, args.midvol, t_c, t_b,
                                             args.cost_bps)
                rows.append(report_row(f"c<{t_c:g} b>={t_b:g}", strat, sleeve,
                                       sw, args.midvol))
        for t in ["SVXY", "GLD"]:
            r = dfe[t].pct_change()
            s = stats(r)
            rows.append({"params": f"buy-hold {t}", "CAGR%": s["cagr"] * 100,
                         "MaxDD%": s["maxdd"] * 100, "Sharpe": s["sharpe"],
                         "sw/yr": 0, "%SVXY": np.nan, "%MID": np.nan,
                         "%GLD": np.nan})
        tbl = pd.DataFrame(rows)
        pd.set_option("display.float_format", lambda v: f"{v:0.2f}")
        print(tbl.to_string(index=False))

    print("\nReminders:")
    print("  - Pre-2018 SVXY was -1x; today's fund is -0.5x. Pre-era rows")
    print("    validate the SIGNAL, not achievable returns.")
    print("  - VTS marketing claim (~30%) remains unverified; compare against")
    print("    the post-delever table, which is the honest benchmark.")
    print("  - Next-close execution + per-switch cost included; slippage and")
    print("    taxes are not.")


if __name__ == "__main__":
    main()
