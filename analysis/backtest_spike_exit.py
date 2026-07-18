#!/usr/bin/env python3
"""
backtest_spike_exit.py — Strategy #3 exit-signal ranking on free UVXY daily data.

Scope (honest): ranks WHICH exit signal harvests spike episodes best.
It does NOT model option carry; the calm-carry >= 0 viability gate is
answered by the shadow book, not here.

Method:
  1. Download UVXY + ^VIX daily closes via yfinance (full history).
  2. Define spike episodes on VIX (cleaner than UVXY, which decays):
     episode starts when VIX close >= (1 + SPIKE_TRIG) * min(VIX close, prior LOOKBACK days).
     Episode tracks UVXY from the start date; peak = max UVXY close in episode.
     Episode ends when UVXY falls back below (1 + EPISODE_FLOOR) * entry UVXY,
     or after MAX_EP_DAYS, whichever first.
  3. Inside each episode, run three exit-signal families on UVXY:
       ma_cross:   exit first day close < SMA(n),          n in {10, 20, 50}
       trail_stop: exit when drawdown from episode high >= p, p in {10,15,20,25}%
       atr_stop:   exit when close < episode_high - k*ATR(14), k in {2, 3, 4}
  4. Metric per (signal, param): capture ratio =
       (exit_px - entry_px) / (peak_px - entry_px), per episode.
     Report median + mean capture, mean days held, and how often the signal
     never fired (rode the spike all the way back down = capture at episode end).

Run:  python3 backtest_spike_exit.py
Deps: pip install yfinance pandas numpy  (already on server via risk_logger)
"""

import sys
import numpy as np
import pandas as pd

SPIKE_TRIG = 0.25      # VIX +25% off its trailing low starts an episode
LOOKBACK = 10          # trailing-low window (days)
EPISODE_FLOOR = 0.05   # episode ends when UVXY back within 5% of entry
MAX_EP_DAYS = 60
MIN_PEAK_GAIN = 0.25   # keep episode only if UVXY peak >= entry * 1.25
MAJOR_PEAK_GAIN = 0.60 # second table: majors only

MA_WINDOWS = [10, 20, 50]
TRAIL_PCTS = [0.10, 0.15, 0.20, 0.25]
ATR_KS = [2.0, 3.0, 4.0]
ATR_N = 14


def fetch():
    import yfinance as yf
    px = yf.download(["UVXY", "^VIX"], period="max", auto_adjust=True,
                     progress=False)
    close = px["Close"].dropna(how="any")
    if close.empty or "UVXY" not in close or "^VIX" not in close:
        sys.exit("yfinance returned no joint UVXY/^VIX data")
    # High/Low for ATR (UVXY only)
    high = px["High"]["UVXY"].reindex(close.index)
    low = px["Low"]["UVXY"].reindex(close.index)
    df = pd.DataFrame({
        "uvxy": close["UVXY"], "vix": close["^VIX"],
        "uvxy_h": high, "uvxy_l": low,
    }).dropna()
    print(f"Data: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} days)")
    return df


def find_episodes(df):
    vix = df["vix"]
    trail_min = vix.shift(1).rolling(LOOKBACK).min()
    trigger = vix >= trail_min * (1 + SPIKE_TRIG)
    episodes = []
    next_free = 0                       # no new episode while one is open
    dates = df.index
    for i in range(len(dates)):
        if i < next_free or not trigger.iloc[i] or i + 2 >= len(dates):
            continue
        entry_px = df["uvxy"].iloc[i]
        end_i = min(i + MAX_EP_DAYS, len(dates) - 1)
        # episode end: first day AFTER a peak where uvxy <= entry*(1+floor)
        window = df["uvxy"].iloc[i:end_i + 1]
        peak_rel = window.cummax()
        fell_back = (window <= entry_px * (1 + EPISODE_FLOOR)) & (peak_rel > entry_px * 1.10)
        if fell_back.any():
            end_i = i + int(np.argmax(fell_back.values))
        peak_px = df["uvxy"].iloc[i:end_i + 1].max()
        next_free = end_i + 1           # block overlapping triggers either way
        if peak_px < entry_px * (1 + MIN_PEAK_GAIN):
            continue                    # dud: nothing material to harvest
        episodes.append({"start": i, "end": end_i, "entry": entry_px,
                         "peak": peak_px})
    print(f"Episodes kept (peak >= +{MIN_PEAK_GAIN:.0%}): {len(episodes)}")
    return episodes


def atr_series(df, n=ATR_N):
    h, l, c = df["uvxy_h"], df["uvxy_l"], df["uvxy"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    return tr.rolling(n).mean()


def run_signal(df, episodes, kind, param, atr):
    """Return list of (capture_ratio, days_held, fired_bool) per episode."""
    out = []
    uvxy = df["uvxy"]
    for ep in episodes:
        i0, i1, entry, peak = ep["start"], ep["end"], ep["entry"], ep["peak"]
        exit_i, fired = i1, False
        run_high = entry
        for i in range(i0 + 1, i1 + 1):
            px = uvxy.iloc[i]
            run_high = max(run_high, px)
            if kind == "ma_cross":
                sma = uvxy.iloc[max(0, i - param + 1):i + 1].mean()
                hit = px < sma and run_high > entry * 1.05
            elif kind == "trail_stop":
                hit = (run_high - px) / run_high >= param
            elif kind == "atr_stop":
                a = atr.iloc[i]
                hit = not np.isnan(a) and px < run_high - param * a
            else:
                raise ValueError(kind)
            if hit:
                exit_i, fired = i, True
                break
        exit_px = uvxy.iloc[exit_i]
        denom = peak - entry
        cap = (exit_px - entry) / denom if denom > 0 else np.nan
        out.append((cap, exit_i - i0, fired))
    return out


def rank_table(df, episodes, atr, label):
    rows = []
    grid = ([("ma_cross", n) for n in MA_WINDOWS]
            + [("trail_stop", p) for p in TRAIL_PCTS]
            + [("atr_stop", k) for k in ATR_KS])
    for kind, param in grid:
        res = run_signal(df, episodes, kind, param, atr)
        caps = np.array([r[0] for r in res], dtype=float)
        days = np.array([r[1] for r in res], dtype=float)
        fired = np.array([r[2] for r in res])
        rows.append({
            "signal": kind, "param": param,
            "median_capture": np.nanmedian(caps),
            "mean_capture": np.nanmean(caps),
            "mean_days_held": days.mean(),
            "pct_never_fired": 100.0 * (~fired).mean(),
            "n_episodes": len(res),
        })
    tbl = pd.DataFrame(rows).sort_values("median_capture", ascending=False)
    print(f"\n--- {label} ---")
    print(tbl.to_string(index=False))


def main():
    df = fetch()
    episodes = find_episodes(df)
    if not episodes:
        sys.exit("No episodes — loosen SPIKE_TRIG or MIN_PEAK_GAIN")
    atr = atr_series(df)
    majors = [ep for ep in episodes
              if ep["peak"] >= ep["entry"] * (1 + MAJOR_PEAK_GAIN)]

    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
    print("\nCapture ratio = fraction of entry->peak UVXY move kept at exit.")
    print("1.0 = sold the top; 0.0 = round-tripped to entry; <0 = below entry.")
    rank_table(df, episodes, atr,
               f"ALL material episodes (peak >= +{MIN_PEAK_GAIN:.0%}), "
               f"n={len(episodes)}")
    if majors:
        rank_table(df, majors, atr,
                   f"MAJORS only (peak >= +{MAJOR_PEAK_GAIN:.0%}), "
                   f"n={len(majors)}")

    print("\nEpisode dates (* = major):")
    for ep in episodes:
        star = "*" if ep in majors else " "
        print(f" {star}{df.index[ep['start']].date()} -> "
              f"{df.index[ep['end']].date()}"
              f"  entry {ep['entry']:g}  peak {ep['peak']:g}"
              f"  ({(ep['peak']/ep['entry']-1)*100:0.0f}%)")


if __name__ == "__main__":
    main()
