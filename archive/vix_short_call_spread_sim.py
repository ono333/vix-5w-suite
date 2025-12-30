# vix_short_call_spread_sim.py
# Sean Seah–style short VIX/UVXY call-credit spread simulator (options-free proxy).

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


# ---------- Helpers ----------

def ema(s, span):
    s = pd.Series(s).astype(float)
    return s.ewm(span=span, adjust=False).mean()


def anti_trade_sell(spy):
    macd_fast = spy.rolling(3).mean()
    macd_slow = spy.rolling(10).mean()
    macd = macd_fast - macd_slow
    signal = macd.rolling(16).mean()
    hh = macd.rolling(60).max()
    cross_dn = (macd.shift(1) >= signal.shift(1)) & (macd < signal)
    recent_hh = (macd == hh).rolling(20).max().astype(bool)
    return (cross_dn & recent_hh).fillna(False)


def week_ends(idx):
    df = pd.DataFrame(index=idx)
    df["w"] = df.index.to_period("W-FRI")
    return df.groupby("w").tail(1).index


def month_ends(idx):
    ser = pd.Series(idx, index=idx)
    return ser.to_period("M").groupby(level=0).apply(lambda s: s.iloc[-1]).values


def contango_flag(vix, vix3m):
    vix3m = vix3m.reindex(vix.index).fillna(method="ffill")
    return (vix < vix3m).astype(bool)


def premium_ratio(vix_level, base=0.22, beta=0.6, cap=0.6, floor=0.12):
    raw = base * (max(vix_level, 5) / 20.0) ** beta
    return float(min(max(raw, floor), cap))


# ---------- UI ----------

st.set_page_config(page_title="Short VIX/UVXY Call-Credit Spread Simulator", layout="wide")
st.title("📉 Short VIX/UVXY Call-Credit Spread Backtester")

with st.sidebar:
    st.header("Data & Schedule")
    start = st.date_input("Start", dt.date(2012, 1, 1))
    end = st.date_input("End", dt.date.today())
    sched = st.radio("Expiry cadence", ["Weekly", "Monthly"], index=0)

    st.header("Instrument")
    instr = st.radio("Trade on", ["VIX", "UVXY"], index=0)

    st.header("Spread Design")
    width = st.slider("Spread width", 2.0, 30.0, 10.0, step=0.5)
    dist = st.slider("Short-strike distance above ref", 2.0, 40.0, 8.0, step=0.5)
    base_pr = st.slider("Base premium ratio at VIX = 20", 0.10, 0.50, 0.22, step=0.01)
    cap_pr = st.slider("Premium ratio cap", 0.20, 0.80, 0.60, step=0.01)

    st.header("Entry Filters")
    vix_min = st.slider("Enter when VIX ≥", 10, 60, 20, step=1)
    vix_max = st.slider("… and VIX ≤", 15, 80, 35, step=1)
    use_contango = st.checkbox("Require contango (VIX < VIX3M)", value=True)
    use_AT = st.checkbox("Use Anti-Trade SELL filter (SPY)", value=False)

    st.header("Sizing & Risk")
    acct0 = st.number_input("Starting account ($)", 10000, 10_000_000, 100_000, step=1000)
    alloc = st.slider("Per-trade allocation (% of account)", 1, 40, 15, step=1) / 100.0
    max_risk = st.slider("Max simultaneous risk cap (% of account)", 10, 100, 60, step=5) / 100.0
    stop_mult = st.slider("Early stop at loss = N× credit (0 = off)", 0.0, 5.0, 2.0, step=0.25)

    run_btn = st.button("Run Backtest")


if yf is None:
    st.error("yfinance is required.")
    st.stop()


# ---------- Main Backtest ----------

if run_btn:
    with st.spinner("Downloading ^VIX, ^VIX3M, SPY, UVXY …"):
        tickers = ["^VIX", "^VIX3M", "SPY", "UVXY"]
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if data is None or data.empty:
        st.warning("No data downloaded.")
        st.stop()

    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna()
    vix, vix3m, spy, uvxy = [close[c].dropna() for c in ["^VIX", "^VIX3M", "SPY", "UVXY"]]

    f_range = (vix >= vix_min) & (vix <= vix_max)
    f_contango = contango_flag(vix, vix3m) if use_contango else pd.Series(True, index=vix.index)
    f_AT = anti_trade_sell(spy) if use_AT else pd.Series(True, index=vix.index)
    entry_mask = (f_range & f_contango & f_AT).fillna(False)

    expiries = week_ends(vix.index) if sched == "Weekly" else month_ends(vix.index)
    expiries = pd.DatetimeIndex(expiries)
    entries = vix.index.intersection(expiries)

    def next_expiry(d):
        nxt = expiries[expiries > d]
        return nxt[0] if len(nxt) else None

    acct = acct0
    peak = acct
    open_trades = []
    hist = []

    for d in vix.index:
        day_vix = float(vix.loc[d])
        day_uvxy = float(uvxy.loc[d]) if d in uvxy.index else np.nan

        # mark-to-market and stops
        new_open = []
        for tr in open_trades:
            cur = day_vix if tr["mode"] == "VIX" else day_uvxy
            if np.isnan(cur):
                new_open.append(tr)
                continue
            intrinsic = max(0.0, min(cur - tr["short_k"], tr["width"]))
            u_pnl = tr["credit_usd"] - intrinsic * tr["dollar_per_point"]
            if stop_mult > 0 and (tr["credit_usd"] - u_pnl) >= stop_mult * tr["credit_usd"]:
                acct += u_pnl
            else:
                new_open.append(tr)
        open_trades = new_open

        # expiry settlement
        new_open = []
        for tr in open_trades:
            if d >= tr["exp"]:
                settle = float(vix.loc[tr["exp"]]) if tr["mode"] == "VIX" else float(uvxy.loc[tr["exp"]])
                intrinsic = max(0.0, min(settle - tr["short_k"], tr["width"]))
                pnl = tr["credit_usd"] - intrinsic * tr["dollar_per_point"]
                acct += pnl
            else:
                new_open.append(tr)
        open_trades = new_open

        # new entry
        if d in entries:
            exp = next_expiry(d)
            if exp is not None and entry_mask.loc[d]:
                at_risk = sum(t["max_loss_usd"] for t in open_trades)
                room = max(0.0, max_risk * acct - at_risk)
                if room > 0:
                    ref = day_vix if instr == "VIX" else day_uvxy
                    cur_pr = premium_ratio(float(vix.loc[d]), base=base_pr, cap=cap_pr)
                    dollar_per_point = 100.0
                    credit_per_spread = width * cur_pr * dollar_per_point
                    max_loss_per_spread = width * (1 - cur_pr) * dollar_per_point
                    target_risk = alloc * acct
                    n = int(max(0, min(target_risk, room) // max_loss_per_spread))
                    if n >= 1:
                        short_k = ref + dist
                        open_trades.append(dict(
                            mode=instr,
                            open=d, exp=exp, ref=ref,
                            credit_usd=credit_per_spread * n,
                            max_loss_usd=max_loss_per_spread * n,
                            short_k=short_k, width=width,
                            dollar_per_point=dollar_per_point
                        ))

        peak = max(peak, acct)
        hist.append({"Date": d, "Equity": acct, "DD%": acct / peak - 1})

    # final settlement
    for tr in open_trades:
        settle = float(vix.loc[tr["exp"]]) if tr["mode"] == "VIX" else float(uvxy.loc[tr["exp"]])
        intrinsic = max(0.0, min(settle - tr["short_k"], tr["width"]))
        pnl = tr["credit_usd"] - intrinsic * tr["dollar_per_point"]
        acct += pnl
        peak = max(peak, acct)
        hist.append({"Date": tr["exp"], "Equity": acct, "DD%": acct / peak - 1})

    ts = pd.DataFrame(hist).drop_duplicates("Date").set_index("Date").sort_index()
    eq, dd = ts["Equity"], ts["DD%"]

    st.subheader("Equity & Drawdown")
    col1, col2 = st.columns(2)
    col1.line_chart(eq)
    col2.line_chart(dd)

    ret = eq.iloc[-1] / eq.iloc[0] - 1
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd_min = dd.min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final equity", f"${eq.iloc[-1]:,.0f}")
    c2.metric("CAGR", f"{cagr*100:.2f}%")
    c3.metric("Max drawdown", f"{dd_min*100:.1f}%")
    c4.metric("Total return", f"{ret*100:.1f}%")

    st.caption("Options-free proxy: credit = width×premium_ratio(VIX); loss = intrinsic at expiry.")
else:
    st.info("Set parameters and click **Run Backtest**.")
