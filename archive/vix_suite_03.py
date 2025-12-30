# vix_suite_03.py
# Volatility Portfolio Suite (AMPI + Short-Vol + Two-Layer Optimizer + OA VIX Hedge)
# - Fixes: pandas rename bug, float(Series) deprecation, 'M' resample alias
# 2025-11 — You + ChatGPT

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

# =================== App shell & global settings ===================

st.set_page_config(page_title="Volatility Portfolio Suite", layout="wide")

# Initialize session state AFTER Streamlit is ready
for key in ["AMPI_EQ", "SV_SERIES", "HEDGE_SERIES", "OA_SERIES", "HEDGE_X1", "HEDGE_X2"]:
    if key not in st.session_state:
        st.session_state[key] = (pd.Series(dtype=float) if "SERIES" in key else 0.0)

# Global display control
st.sidebar.markdown("### Display Settings")
use_log = st.sidebar.checkbox("Use logarithmic scale for equity charts", value=True)
st.session_state["LOG_SCALE"] = use_log

st.title("📊 Volatility Portfolio Suite")

# =================== Shared helpers ===================

def week_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # last business day of each ISO week ending Friday
    df = pd.DataFrame(index=idx)
    df["w"] = df.index.to_period("W-FRI")
    return df.groupby("w").tail(1).index

def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # future-proof month-end picker (no deprecated alias)
    by_month = pd.Series(index=idx, dtype=float)
    ends = by_month.groupby(by_month.index.to_period("M")).apply(lambda s: s.index[-1])
    return pd.DatetimeIndex(ends.values)

def add_months(ts: pd.Timestamp, m: int) -> pd.Timestamp:
    y = ts.year + (ts.month + m - 1) // 12
    mo = (ts.month + m - 1) % 12 + 1
    dd = min(ts.day, 28)
    return pd.Timestamp(year=y, month=mo, day=dd)

def anti_trade_sell(spy: pd.Series) -> pd.Series:
    # LBR 3-10 Osc (SMA) Anti-Trade SELL proxy
    macd = spy.rolling(3).mean() - spy.rolling(10).mean()
    signal = macd.rolling(16).mean()
    hh = macd.rolling(60).max()
    cross_dn = (macd.shift(1) >= signal.shift(1)) & (macd < signal)
    recent_hh = (macd == hh).rolling(20).max().astype(bool)
    return (cross_dn & recent_hh).fillna(False)

def premium_ratio(vix_level, base=0.22, beta=0.6, cap=0.6, floor=0.12):
    raw = base * (max(vix_level, 5) / 20.0) ** beta
    return float(min(max(raw, floor), cap))

def plot_equity(df: pd.DataFrame, rename_map=None):
    if rename_map:
        df = df.rename(columns=rename_map)
    if st.session_state.get("LOG_SCALE", False):
        df_plot = np.log10(df.clip(lower=1))
        df_plot = df_plot.rename(columns=lambda c: f"log₁₀({c})")
        st.line_chart(df_plot, use_container_width=True)
        st.caption("Log scale: equal Y-axis steps represent ×10 changes in equity.")
    else:
        st.line_chart(df, use_container_width=True)

def fill_or_flat(eq: pd.Series | None, idx: pd.DatetimeIndex, base_val=100000.0) -> pd.Series:
    if eq is None or (isinstance(eq, pd.Series) and eq.empty):
        return pd.Series(base_val, index=idx)
    return eq.reindex(idx).ffill().bfill()

# Rolling percentile helper
def rolling_pct_rank(series: pd.Series, win: int) -> pd.Series:
    def _pct(a):
        s = pd.Series(a)
        return (s.rank(pct=True).iloc[-1]) if len(s)==win else np.nan
    return series.rolling(win).apply(_pct, raw=False).fillna(0.5)

# =================== Tabs ===================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "AMPI (PoS model)",
    "Short-Vol Income",
    "Two-Layer Hedge Optimizer",
    "Combined Portfolio",
    "Option Alpha VIX Hedge",
])

# -------------------- 1) AMPI PoS model (with VIX-driven crash months) --------------------
with tab1:
    st.subheader("AMPI – Probability-of-Success Model (monthly cadence → daily curve)")

    c1, c2, c3, c4 = st.columns(4)
    start = c1.date_input("Start", dt.date(2012, 1, 1), key="ampi_start")
    end   = c2.date_input("End", dt.date.today(), key="ampi_end")
    acct0_ampi = c3.number_input("Starting account ($)", 10000, 10_000_000, 100_000, step=1000, key="ampi_acct0")
    seed_ampi  = c4.number_input("Random seed", 0, 10_000, 42, step=1, key="ampi_seed")

    c5, c6, c7, c8 = st.columns(4)
    p_win   = c5.slider("Non-crash win rate p", 0.0, 1.0, 0.80, step=0.01, key="ampi_pwin")
    win_m   = c6.slider("Avg win per non-crash month (%)", 0.0, 20.0, 6.0, step=0.1, key="ampi_win") / 100.0
    loss_m  = c7.slider("Avg loss per non-crash month (%)", 0.0, 60.0, 6.0, step=0.5, key="ampi_loss") / 100.0
    vol_smooth = c8.slider("Daily smoothing factor (days)", 1, 22, 5, key="ampi_smooth")

    st.markdown("### Crash modeling (optional)")
    d1, d2, d3, d4 = st.columns(4)
    use_vix_crash   = d1.checkbox("Use VIX-driven crash months", True, key="ampi_use_vix")
    crash_vix_thr   = d2.slider("Crash threshold (VIX monthly max ≥)", 15, 80, 35, step=1, key="ampi_thr")
    crash_loss_min  = d3.slider("Crash loss min (%)", 5, 60, 20, step=1, key="ampi_cmin")/100.0
    crash_loss_max  = d4.slider("Crash loss max (%)", 10, 80, 40, step=1, key="ampi_cmax")/100.0
    drag_after_crash = st.slider("Post-crash hangover months (mild -X%/mo)", 0, 6, 2, step=1, key="ampi_tailM")
    hang_drag = st.slider("Hangover drag per month (%)", 0.0, 10.0, 2.0, step=0.25, key="ampi_hdrag")/100.0

    # Build daily index + VIX
    if yf is None:
        idx = pd.date_range(start, end, freq="B")
        vix_daily = None
    else:
        base = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(base, pd.DataFrame) and "Close" in base.columns:
            vix_daily = base["Close"].rename("VIX")
        else:
            st.warning("VIX download returned no 'Close' column.")
            vix_daily = pd.Series(dtype=float, name="VIX")
        idx = vix_daily.index if not vix_daily.empty else pd.date_range(start, end, freq="B")

    months = pd.PeriodIndex(idx, freq="M").unique().to_timestamp()
    rng = np.random.default_rng(int(seed_ampi))

    # Determine crash months from VIX (monthly max ≥ threshold), month-end alias 'ME'
    crash_mask = pd.Series(False, index=months)
    if use_vix_crash and vix_daily is not None and not vix_daily.empty:
        vmx = vix_daily.resample("ME").max()
        crash_months = vmx[vmx >= crash_vix_thr].index
        crash_mask.loc[crash_months.intersection(months)] = True

    # Build monthly return series with clustered crashes + hangover
    rets = pd.Series(0.0, index=months, dtype=float)
    last_crash_i = None
    for i, m in enumerate(months):
        if crash_mask.loc[m]:
            crash_loss = -rng.uniform(crash_loss_min, crash_loss_max)  # e.g., -0.2 .. -0.4
            rets.iloc[i] = crash_loss
            last_crash_i = i
        else:
            if (last_crash_i is not None) and (i - last_crash_i <= drag_after_crash):
                rets.iloc[i] = -hang_drag
            else:
                is_win = rng.binomial(1, p_win) == 1
                rets.iloc[i] = (win_m if is_win else -loss_m)

    # expand to daily
    daily = pd.Series(0.0, index=idx, dtype=float)
    for m in months:
        seg = daily.loc[daily.index.to_period("M").to_timestamp()==m]
        if len(seg)==0: 
            continue
        r = rets.loc[m]
        per_day = (1 + r) ** (1/len(seg)) - 1
        daily.loc[seg.index] = per_day
    daily = daily.rolling(vol_smooth, min_periods=1).mean()

    eq_ampi = (1 + daily).cumprod() * float(acct0_ampi)
    dd_ampi = eq_ampi / eq_ampi.cummax() - 1
    plot_equity(pd.DataFrame({"AMPI Equity": eq_ampi}))
    st.caption(
        f"CAGR ~ {((eq_ampi.iloc[-1]/eq_ampi.iloc[0])**(365.25/((idx[-1]-idx[0]).days)) - 1):.2%} ; "
        f"Max DD {dd_ampi.min():.2%}"
    )
    st.session_state["AMPI_EQ"] = eq_ampi

# -------------------- 2) Short-Vol Income (with grid search) --------------------
with tab2:
    st.subheader("Short-Vol Income – VIX/UVXY Call-Credit Spread Proxy")

    c1, c2, c3, c4 = st.columns(4)
    start_sv = c1.date_input("Start", dt.date(2012,1,1), key="sv_start")
    end_sv   = c2.date_input("End", dt.date.today(), key="sv_end")
    instr    = c3.radio("Trade on", ["VIX","UVXY"], index=0, key="sv_instr")
    sched    = c4.radio("Cadence", ["Weekly","Monthly"], index=0, key="sv_cad")

    c5, c6, c7, c8 = st.columns(4)
    width   = c5.slider("Spread width (pts or $)", 2.0, 30.0, 10.0, step=0.5, key="sv_width")
    dist    = c6.slider("Short strike distance", 2.0, 40.0, 8.0, step=0.5, key="sv_dist")
    base_pr = c7.slider("Base prem ratio @ VIX=20", 0.10, 0.50, 0.22, step=0.01, key="sv_basepr")
    cap_pr  = c8.slider("Prem ratio cap", 0.20, 0.80, 0.60, step=0.01, key="sv_cappr")

    c9, c10, c11, c12 = st.columns(4)
    beta_pr = c9.slider("Premium slope β", 0.2, 1.2, 0.6, step=0.05, key="sv_beta")
    vix_min = c10.slider("Enter VIX ≥", 10, 60, 20, step=1, key="sv_vixmin")
    vix_max = c11.slider("Enter VIX ≤", 15, 80, 35, step=1, key="sv_vixmax")
    use_contango = c12.checkbox("Require contango (VIX < VIX3M)", value=True, key="sv_contango")

    c13, c14, c15, c16 = st.columns(4)
    use_AT   = c13.checkbox("Anti-Trade SELL filter (SPY)", value=False, key="sv_at")
    acct0_sv = c14.number_input("Starting account ($)", 10000, 10_000_000, 100_000, step=1000, key="sv_acct0")
    alloc    = c15.slider("Per-trade allocation (% acct)", 1, 40, 15, step=1, key="sv_alloc") / 100.0
    max_risk = c16.slider("Max simultaneous risk cap (% acct)", 10, 100, 60, step=5, key="sv_risk") / 100.0
    stop_mult = st.slider("Early stop at N× credit (0 = off)", 0.0, 5.0, 2.0, step=0.25, key="sv_stop")

    run_sv = st.button("Run Short-Vol Backtest", key="sv_run")

    # ===== Run button engine =====
    if run_sv:
        if yf is None:
            st.error("yfinance required for this tab.")
        else:
            with st.spinner("Downloading ^VIX, ^VIX3M, SPY, UVXY …"):
                data = yf.download(["^VIX","^VIX3M","SPY","UVXY"],
                                   start=start_sv, end=end_sv,
                                   auto_adjust=True, progress=False)
            if data is not None and not data.empty:
                close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
                close = close.dropna(how="all")
            else:
                close = pd.DataFrame()

            if close.empty or not all(c in close.columns for c in ["^VIX","^VIX3M","SPY","UVXY"]):
                st.warning("Missing data — keeping previous Short-Vol series.")
            else:
                vix  = close["^VIX"].dropna()
                vix3 = close["^VIX3M"].reindex(vix.index).ffill()
                spy  = close["SPY"].reindex(vix.index).ffill()
                uvxy = close["UVXY"].reindex(vix.index).ffill()

                f_range = (vix >= vix_min) & (vix <= vix_max)
                f_cont  = (vix < vix3) if use_contango else pd.Series(True, index=vix.index)
                f_at    = anti_trade_sell(spy) if use_AT else pd.Series(True, index=vix.index)
                entry_mask = (f_range & f_cont & f_at).fillna(False)

                expiries = week_ends(vix.index) if sched=="Weekly" else month_ends(vix.index)
                expiries = pd.DatetimeIndex(expiries)
                entries  = vix.index.intersection(expiries)

                def next_expiry(d):
                    nxt = expiries[expiries > d]
                    return nxt[0] if len(nxt) else None

                acct = float(acct0_sv); peak = acct
                open_trades = []; hist = []

                for d in vix.index:
                    V = float(vix.at[d]) if d in vix.index else float("nan")
                    U = float(uvxy.at[d]) if d in uvxy.index else float("nan")

                    # mark-to-market + stops
                    new_open = []
                    for tr in open_trades:
                        cur = V if tr["mode"]=="VIX" else U
                        if pd.isna(cur):
                            new_open.append(tr); continue
                        intrinsic = max(0.0, min(cur - tr["short_k"], tr["width"]))
                        u_pnl = tr["credit_usd"] - intrinsic * tr["dollar_per_point"]
                        if stop_mult>0 and (tr["credit_usd"] - u_pnl) >= stop_mult*tr["credit_usd"]:
                            acct += u_pnl
                        else:
                            new_open.append(tr)
                    open_trades = new_open

                    # expiry
                    new_open = []
                    for tr in open_trades:
                        if d >= tr["exp"]:
                            settle = float(vix.at[tr["exp"]]) if tr["mode"]=="VIX" else float(uvxy.at[tr["exp"]])
                            intrinsic = max(0.0, min(settle - tr["short_k"], tr["width"]))
                            pnl = tr["credit_usd"] - intrinsic * tr["dollar_per_point"]
                            acct += pnl
                        else:
                            new_open.append(tr)
                    open_trades = new_open

                    # new entry
                    if d in entries and entry_mask.loc[d]:
                        at_risk = sum(t["max_loss_usd"] for t in open_trades)
                        room = max(0.0, max_risk*acct - at_risk)
                        if room > 0:
                            ref = V if instr=="VIX" else U
                            if pd.isna(ref): 
                                continue
                            cur_pr = premium_ratio(V, base=base_pr, beta=beta_pr, cap=cap_pr)
                            dollar_per_point = 100.0
                            credit_per_spread = width * cur_pr * dollar_per_point
                            max_loss_per_spread = width * (1-cur_pr) * dollar_per_point
                            target_risk = alloc * acct
                            n = int(max(0, min(target_risk, room) // max_loss_per_spread))
                            if n >= 1:
                                short_k = ref + dist
                                open_trades.append(dict(
                                    mode=instr, open=d, exp=next_expiry(d), ref=ref,
                                    credit_usd=credit_per_spread*n,
                                    max_loss_usd=max_loss_per_spread*n,
                                    short_k=short_k, width=width,
                                    dollar_per_point=dollar_per_point
                                ))

                    peak = max(peak, acct)
                    hist.append({"Date": d, "Equity": acct, "DD%": acct/peak - 1})

                # final settle
                for tr in open_trades:
                    settle = float(vix.at[tr["exp"]]) if tr["mode"]=="VIX" else float(uvxy.at[tr["exp"]])
                    intrinsic = max(0.0, min(settle - tr["short_k"], tr["width"]))
                    pnl = tr["credit_usd"] - intrinsic * tr["dollar_per_point"]
                    acct += pnl
                    peak = max(peak, acct)
                    hist.append({"Date": tr["exp"], "Equity": acct, "DD%": acct/peak - 1})

                ts = pd.DataFrame(hist).drop_duplicates("Date").set_index("Date").sort_index()
                if not ts.empty and "Equity" in ts.columns:
                    eq_sv = ts["Equity"]
                    plot_equity(pd.DataFrame({"Short-Vol Equity": eq_sv}))
                    st.session_state["SV_SERIES"] = eq_sv
                else:
                    st.warning("Short-Vol produced no timeline — keeping previous series.")

    # ===== Grid Search (fast proxy) =====
    st.markdown("### 🔎 Grid Search (Short-Vol Income)")
    gs1, gs2, gs3 = st.columns(3)
    w_list   = gs1.text_input("Width list", "6,8,10,12,14", key="sv_g_w")
    d_list   = gs2.text_input("Short-strike distance list", "6,8,10,12", key="sv_g_d")
    beta_list= gs3.text_input("β list", "0.5,0.6,0.8,1.0", key="sv_g_beta")
    gs4, gs5, gs6 = st.columns(3)
    base_list = gs4.text_input("Base prem @VIX=20 list", "0.18,0.22,0.26", key="sv_g_base")
    cap_list  = gs5.text_input("Cap list", "0.5,0.6,0.7", key="sv_g_cap")
    vix_lohi  = gs6.text_input("VIX band pairs (lo-hi)", "18-30,20-35", key="sv_g_band")
    gs7, gs8, gs9 = st.columns(3)
    alloc_list= gs7.text_input("Per-trade alloc % list", "10,15,20", key="sv_g_alloc")
    risk_list = gs8.text_input("Risk cap % list", "40,60,80", key="sv_g_risk")
    sched_list= gs9.text_input("Cadence list (W,M)", "W,M", key="sv_g_sched")
    run_grid_sv = st.button("Run Short-Vol Grid Search", key="sv_run_grid")

    def _parse_floats(s): return [float(x) for x in s.replace("，",",").split(",") if x.strip()]
    def _parse_ints(s):   return [int(float(x)) for x in s.replace("，",",").split(",") if x.strip()]
    def _parse_pairs(s):
        out=[]
        for t in s.replace("，",",").split(","):
            if "-" in t:
                a,b=t.split("-")
                out.append((int(float(a)), int(float(b))))
        return out

    if run_grid_sv:
        if yf is None:
            st.error("yfinance required.")
        else:
            with st.spinner("Downloading ^VIX,^VIX3M,SPY,UVXY …"):
                data = yf.download(["^VIX","^VIX3M","SPY","UVXY"],
                                   start=start_sv, end=end_sv,
                                   auto_adjust=True, progress=False)
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        close = close.dropna(how="all")
        if close.empty or not all(c in close.columns for c in ["^VIX","^VIX3M","SPY","UVXY"]):
            st.warning("Missing tickers for grid.")
        else:
            vix  = close["^VIX"].dropna()
            vix3 = close["^VIX3M"].reindex(vix.index).ffill()
            spy  = close["SPY"].reindex(vix.index).ffill()
            uvxy = close["UVXY"].reindex(vix.index).ffill()

            widths   = _parse_floats(w_list)
            dists    = _parse_floats(d_list)
            betas    = _parse_floats(beta_list)
            bases    = _parse_floats(base_list)
            caps     = _parse_floats(cap_list)
            bands    = _parse_pairs(vix_lohi)
            allocs   = [x/100.0 for x in _parse_floats(alloc_list)]
            risks    = [x/100.0 for x in _parse_floats(risk_list)]
            scads    = [s.strip().upper() for s in sched_list.split(",") if s.strip()]

            from itertools import product
            combos = list(product(widths,dists,betas,bases,caps,bands,allocs,risks,scads))
            rows=[]; best=None
            prog = st.progress(0.0)

            def _run_once(wid,dist_,beta_,base_,cap_,band_,alloc_,risk_,sched_):
                f_range = (vix >= band_[0]) & (vix <= band_[1])
                f_cont  = (vix < vix3) if use_contango else pd.Series(True, index=vix.index)
                f_at    = anti_trade_sell(spy) if use_AT else pd.Series(True, index=vix.index)
                entry_mask = (f_range & f_cont & f_at).fillna(False)

                expiries = week_ends(vix.index) if sched_=="W" else month_ends(vix.index)
                expiries = pd.DatetimeIndex(expiries)
                def next_expiry(d):
                    nxt = expiries[expiries > d]
                    return nxt[0] if len(nxt) else None

                acct = float(acct0_sv); peak = acct
                open_trades=[]

                for d in vix.index:
                    V = float(vix.at[d]) if d in vix.index else float("nan")
                    U = float(uvxy.at[d]) if d in uvxy.index else float("nan")

                    # Stops
                    new=[]
                    for tr in open_trades:
                        cur = V if tr["mode"]=="VIX" else U
                        if pd.isna(cur): new.append(tr); continue
                        intrinsic = max(0.0, min(cur - tr["short_k"], tr["width"]))
                        u_pnl = tr["credit_usd"] - intrinsic*tr["dollar_per_point"]
                        if stop_mult>0 and (tr["credit_usd"] - u_pnl) >= stop_mult*tr["credit_usd"]:
                            acct += u_pnl
                        else:
                            new.append(tr)
                    open_trades = new

                    # Expiry
                    new=[]
                    for tr in open_trades:
                        if d >= tr["exp"]:
                            settle = float(vix.at[tr["exp"]]) if tr["mode"]=="VIX" else float(uvxy.at[tr["exp"]])
                            intrinsic = max(0.0, min(settle - tr["short_k"], tr["width"]))
                            pnl = tr["credit_usd"] - intrinsic*tr["dollar_per_point"]
                            acct += pnl
                        else:
                            new.append(tr)
                    open_trades = new

                    # Entry
                    if d in vix.index.intersection(expiries) and entry_mask.loc[d]:
                        at_risk = sum(t["max_loss_usd"] for t in open_trades)
                        room = max(0.0, risk_*acct - at_risk)
                        if room>0:
                            ref = V if instr=="VIX" else U
                            if pd.isna(ref): continue
                            pr = premium_ratio(V, base=base_, beta=beta_, cap=cap_)
                            credit = wid * pr * 100.0
                            maxloss = wid * (1-pr) * 100.0
                            n = int(max(0, min(alloc_*acct, room)//maxloss))
                            if n>=1:
                                open_trades.append(dict(
                                    mode=instr, open=d, exp=next_expiry(d), ref=ref,
                                    credit_usd=credit*n, max_loss_usd=maxloss*n,
                                    short_k=ref + dist_, width=wid, dollar_per_point=100.0
                                ))
                    peak=max(peak,acct)

                # finalize:
                for tr in open_trades:
                    settle = float(vix.at[tr["exp"]]) if tr["mode"]=="VIX" else float(uvxy.at[tr["exp"]])
                    intrinsic = max(0.0, min(settle - tr["short_k"], tr["width"]))
                    pnl = tr["credit_usd"] - intrinsic*tr["dollar_per_point"]
                    acct += pnl

                yrs = max(1e-9,(vix.index[-1]-vix.index[0]).days/365.25)
                cagr = (acct/float(acct0_sv))**(1/yrs) - 1 if yrs>0 else 0.0
                mdd_proxy = -risk_
                return cagr, mdd_proxy, acct

            total = len(combos) if combos else 1
            for i,(wid,dist_,beta_,base_,cap_,band_,alloc_,risk_,scad_) in enumerate(combos,1):
                cagr,mdd,final = _run_once(wid,dist_,beta_,base_,cap_,band_,alloc_,risk_,scad_)
                rec = dict(width=wid,dist=dist_,beta=beta_,base=base_,cap=cap_,
                           vix_lo=band_[0],vix_hi=band_[1],alloc=alloc_,risk=risk_,
                           sched=scad_,CAGR=cagr,MaxDD=mdd,Final=final)
                rows.append(rec)
                if (best is None) or (cagr>best["CAGR"] or (abs(cagr-best["CAGR"])<1e-12 and mdd>best["MaxDD"])):
                    best = rec
                prog.progress(min(1.0, i/total))

            res = pd.DataFrame(rows).sort_values(["CAGR","MaxDD","Final"], ascending=[False, False, False])
            st.dataframe(res.head(30), use_container_width=True)
            st.success("Pick a top row and re-run with the Run button to save its full curve.")

# -------------------- 3) Two-Layer Hedge Optimizer (with grid search) --------------------
with tab3:
    st.subheader("Two-Layer Hedge Optimizer (min-drag to meet crash target)")

    target_cov = st.slider("Target crash payout (of account)", 0.00, 1.00, 0.35, step=0.01, key="hedge_target")
    slack      = st.slider("Allow shortfall (abs % of acct)", 0.00, 0.20, 0.00, step=0.01, key="hedge_slack")

    c1, c2, c3 = st.columns(3)
    p1 = c1.slider("Core p₁", 0.00, 0.40, 0.08, step=0.01, key="hedge_p1")
    R1 = c2.slider("Core R₁", 1.0, 30.0, 14.0, step=0.5, key="hedge_R1")
    u1 = c3.slider("Core max spend / mo (acct %)", 0.00, 0.05, 0.01, step=0.001, key="hedge_u1")

    c4, c5, c6 = st.columns(3)
    p2 = c4.slider("Tail p₂", 0.00, 0.40, 0.05, step=0.01, key="hedge_p2")
    R2 = c5.slider("Tail R₂", 1.0, 60.0, 35.0, step=0.5, key="hedge_R2")
    u2 = c6.slider("Tail max spend / mo (acct %)", 0.00, 0.05, 0.01, step=0.001, key="hedge_u2")

    spend_cap  = st.slider("Total spend cap / mo (acct %)", 0.00, 0.10, 0.02, step=0.001, key="hedge_cap")
    prefer_pos_ev = st.checkbox("Prefer ≥ 0 EV if ties", value=True, key="hedge_prefpos")

    def layer_coeffs(p, R):
        a = max(R - 1.0, 0.0)         # coverage per $1 spend
        ev_per_spend = p * a - 1.0
        c = max(-ev_per_spend, 0.0)   # drag per $1 spend (0 if EV >= 0)
        return a, c, ev_per_spend

    a1, c1c, _ = layer_coeffs(p1, R1)
    a2, c2c, _ = layer_coeffs(p2, R2)

    def solve_two_layer(target, a1, a2, c1, c2, u1, u2, cap, slack, prefer_pos):
        cand = set()
        def add(x1, x2):
            if np.isfinite(x1) and np.isfinite(x2) and x1>=0 and x2>=0:
                cand.add((round(float(x1),10), round(float(x2),10)))
        add(0,0); add(min(u1,cap),0); add(0,min(u2,cap))
        add(min(u1,cap), max(0.0, cap-min(u1,cap)))
        add(max(0.0, cap-min(u2,cap)), min(u2,cap))
        if a1>0: add(min(target/a1, u1, cap), 0)
        if a2>0: add(0, min(target/a2, u2, cap))
        for x1 in [0, min(u1,cap), max(0.0, cap-u2)]:
            need = max(0.0, target - a1*x1)
            x2 = need/a2 if a2>0 else float("inf")
            add(x1, x2)
        for x2 in [0, min(u2,cap), max(0.0, cap-u1)]:
            need = max(0.0, target - a2*x2)
            x1 = need/a1 if a1>0 else float("inf")
            add(x1, x2)
        rows=[]
        for x1,x2 in cand:
            if x1>u1+1e-12 or x2>u2+1e-12 or x1+x2>cap+1e-12: 
                continue
            cov = a1*x1 + a2*x2
            short = max(0.0, target - cov)
            ev = -(c1*x1 + c2*x2)
            rows.append(dict(x1=x1,x2=x2,coverage=cov,shortfall=short,exp_monthly=ev))
        df = pd.DataFrame(rows)
        if df.empty: 
            return None
        feas = df[df["shortfall"]<=slack+1e-12].copy()
        if feas.empty:
            best = df.sort_values(["shortfall","exp_monthly"], ascending=[True,False]).iloc[0]
            return best.to_dict()
        feas["abs_ev"]=feas["exp_monthly"].abs()
        feas["tie"]=np.where(feas["exp_monthly"]>=0,0,1) if prefer_pos else 0
        best = feas.sort_values(["abs_ev","tie","x1","x2"], ascending=[True,True,True,True]).iloc[0]
        return best.to_dict()

    best = solve_two_layer(target_cov, a1, a2, c1c, c2c, u1, u2, spend_cap, slack, prefer_pos_ev)
    if best is None:
        st.error("No candidates inside bounds/cap. Widen limits.")
    else:
        hdr = st.columns(5)
        hdr[0].metric("Core spend x₁", f"{100*best['x1']:.2f}%/mo")
        hdr[1].metric("Tail spend x₂", f"{100*best['x2']:.2f}%/mo")
        hdr[2].metric("Crash payout", f"{100*best['coverage']:.1f}% of acct")
        hdr[3].metric("Exp. monthly (hedge layer)", f"{100*best['exp_monthly']:.2f}%")
        hdr[4].metric("Shortfall", f"{100*best['shortfall']:.2f}%")
        st.session_state["HEDGE_X1"] = float(best["x1"])
        st.session_state["HEDGE_X2"] = float(best["x2"])

    # Grid search for parameters
    st.markdown("### 🔎 Grid Search (Two-Layer parameters)")
    g1, g2, g3 = st.columns(3)
    p1_list = g1.text_input("Core p₁ list", "0.06,0.08,0.10", key="hg_p1")
    R1_list = g2.text_input("Core R₁ list", "10,14,18", key="hg_R1")
    u1_list = g3.text_input("Core spend% list", "0.005,0.01,0.015", key="hg_u1")
    g4, g5, g6 = st.columns(3)
    p2_list = g4.text_input("Tail p₂ list", "0.03,0.05,0.07", key="hg_p2")
    R2_list = g5.text_input("Tail R₂ list", "25,35,45", key="hg_R2")
    u2_list = g6.text_input("Tail spend% list", "0.005,0.01,0.015", key="hg_u2")
    cap_list = st.text_input("Total spend cap list", "0.015,0.02,0.03", key="hg_caplist")
    run_hgrid = st.button("Run Two-Layer Grid Search", key="hg_run")

    def _pfloats(s): return [float(x) for x in s.replace("，",",").split(",") if x.strip()]
    if run_hgrid:
        P1=_pfloats(p1_list); R1s=[float(x) for x in R1_list.split(",")]
        U1=_pfloats(u1_list); P2=_pfloats(p2_list); R2s=[float(x) for x in R2_list.split(",")]
        U2=_pfloats(u2_list); CAP=_pfloats(cap_list)

        from itertools import product
        rows=[]
        for p1g,R1g,u1g,p2g,R2g,u2g,capg in product(P1,R1s,U1,P2,R2s,U2,CAP):
            a1g,c1g,_=layer_coeffs(p1g,R1g)
            a2g,c2g,_=layer_coeffs(p2g,R2g)
            cand=solve_two_layer(target_cov,a1g,a2g,c1g,c2g,u1g,u2g,capg,slack,prefer_pos_ev)
            if cand is None: 
                continue
            rows.append(dict(p1=p1g,R1=R1g,u1=u1g,p2=p2g,R2=R2g,u2=u2g,cap=capg,
                             x1=cand["x1"],x2=cand["x2"],coverage=cand["coverage"],
                             shortfall=cand["shortfall"],exp_monthly=cand["exp_monthly"]))
        if not rows:
            st.warning("No feasible combos. Widen lists.")
        else:
            df=pd.DataFrame(rows)
            feas=df[df["shortfall"]<=slack+1e-12].copy()
            if feas.empty:
                show=df.sort_values(["shortfall","exp_monthly"],ascending=[True,False])
                st.info("No combo met the shortfall cap. Showing nearest shortfall.")
            else:
                show=feas.sort_values(["exp_monthly","x1","x2"],ascending=[False,True,True])
            show_disp=show.copy()
            for col in ["u1","u2","cap","x1","x2","coverage","shortfall","exp_monthly"]:
                show_disp[col]=(100*show_disp[col]).round(2)
            show_disp=show_disp.rename(columns={
                "u1":"u₁ %/mo","u2":"u₂ %/mo","cap":"cap %/mo",
                "x1":"x₁ spend %/mo","x2":"x₂ spend %/mo",
                "coverage":"coverage %","shortfall":"shortfall %","exp_monthly":"exp /mo %"
            })
            st.dataframe(show_disp.head(30), use_container_width=True)
            top=show.iloc[0]
            st.success(f"Suggested spends → x₁={100*top['x1']:.2f}%/mo, x₂={100*top['x2']:.2f}%/mo  |  coverage={100*top['coverage']:.1f}%, shortfall={100*top['shortfall']:.2f}%")
            st.session_state["HEDGE_X1"]=float(top["x1"]); st.session_state["HEDGE_X2"]=float(top["x2"])

# -------------------- 4) Combined Portfolio --------------------
with tab4:
    st.subheader("Combined Portfolio – AMPI + Short-Vol + Two-Layer Hedge + OA VIX")

    if yf is None:
        idx = pd.date_range(dt.date(2012,1,1), dt.date.today(), freq="B")
    else:
        base = yf.download(["SPY"], start=dt.date(2012,1,1), end=dt.date.today(), auto_adjust=True, progress=False)
        base = (base["Close"] if isinstance(base.columns, pd.MultiIndex) else base).dropna()
        idx = base.index

    eq_ampi  = fill_or_flat(st.session_state.get("AMPI_EQ"), idx)
    eq_sv    = fill_or_flat(st.session_state.get("SV_SERIES"), idx)
    eq_oa    = fill_or_flat(st.session_state.get("OA_SERIES"), idx)
    hedge_eq = st.session_state.get("HEDGE_SERIES")

    # synth hedge path if not provided yet
    if hedge_eq is None or (isinstance(hedge_eq, pd.Series) and hedge_eq.empty):
        x1 = float(st.session_state.get("HEDGE_X1", 0.005))
        x2 = float(st.session_state.get("HEDGE_X2", 0.005))
        months = pd.PeriodIndex(idx, freq="M").unique().to_timestamp()
        rng = np.random.default_rng(7)
        u = rng.random(len(months))
        r_m = []
        for s in u:
            if s < 0.04:
                r_m.append(x2*34 + x1*14)   # tail hit
            elif s < 0.10:
                r_m.append(x1*14)           # core hit
            else:
                r_m.append(-(x1+x2))        # drag
        r_m = pd.Series(r_m, index=months)
        h_daily = pd.Series(0.0, index=idx)
        for m in r_m.index:
            seg = h_daily.loc[h_daily.index.to_period("M").to_timestamp()==m]
            if len(seg)==0: continue
            per = (1 + r_m.loc[m]) ** (1/len(seg)) - 1
            h_daily.loc[seg.index] = per
        hedge_eq = (1 + h_daily).cumprod() * 100000.0
        st.session_state["HEDGE_SERIES"] = hedge_eq
    else:
        hedge_eq = fill_or_flat(hedge_eq, idx)

    # Normalize to 100k
    A = (eq_ampi  / eq_ampi.iloc[0])  * 100000.0
    S = (eq_sv    / eq_sv.iloc[0])    * 100000.0
    H = (hedge_eq / hedge_eq.iloc[0]) * 100000.0
    O = (eq_oa    / eq_oa.iloc[0])    * 100000.0

    st.markdown("#### Allocation Weights")
    w1, w2, w3, w4 = st.columns(4)
    w_ampi = w1.slider("AMPI", 0.0, 1.0, 0.50, step=0.05, key="w_ampi")
    w_sv   = w2.slider("Short-Vol", 0.0, 1.0, 0.15, step=0.05, key="w_sv")
    w_hd   = w3.slider("Hedge (2-layer)", 0.0, 1.0, 0.20, step=0.05, key="w_hd")
    w_oa   = w4.slider("OA VIX", 0.0, 1.0, 0.15, step=0.05, key="w_oa")
    s = max(1e-9, w_ampi + w_sv + w_hd + w_oa)
    w_ampi, w_sv, w_hd, w_oa = [w/s for w in (w_ampi, w_sv, w_hd, w_oa)]

    combo = w_ampi*A + w_sv*S + w_hd*H + w_oa*O
    out = pd.DataFrame({"Combined": combo, "AMPI": A, "Short-Vol": S, "Hedge": H, "OA VIX": O})
    plot_equity(out[["Combined","AMPI","Short-Vol","Hedge","OA VIX"]])

    yrs = (idx[-1]-idx[0]).days/365.25
    cagr = (combo.iloc[-1]/combo.iloc[0])**(1/yrs) - 1 if yrs>0 else 0.0
    dd = combo/combo.cummax() - 1
    c1, c2, c3 = st.columns(3)
    c1.metric("Final equity", f"${combo.iloc[-1]:,.0f}")
    c2.metric("CAGR", f"{cagr*100:.2f}%")
    c3.metric("Max DD", f"{dd.min()*100:.1f}%")

# -------------------- 5) Option Alpha VIX Hedge (convexity + grid search) --------------------
with tab5:
    st.subheader("Option Alpha–style VIX Hedge (Short-Term Spikes + Doomsday) — with Grid Search")

    # Core dates & sizing
    c1, c2, c3, c4 = st.columns(4)
    start_oa = c1.date_input("Start", dt.date(2012, 1, 1), key="oa_start")
    end_oa   = c2.date_input("End", dt.date.today(), key="oa_end")
    acct0_oa = c3.number_input("Starting account ($)", 10_000, 10_000_000, 100_000, step=1000, key="oa_acct0")
    risk_cap = c4.slider("Max hedge spend at any time (% of account)", 0.0, 0.10, 0.03, step=0.001, key="oa_cap")

    st.markdown("### Strategy 1 — Short-Term Spikes (monthly ladder)")
    s1a, s1b, s1c, s1d = st.columns(4)
    s1_spend = s1a.slider("Spend per month (acct %)", 0.0, 0.03, 0.01, step=0.001, key="oa_s1_spend")
    s1_mty   = s1b.slider("Expiry (months)", 1, 3, 1, step=1, key="oa_s1_mty")
    s1_k1    = s1c.slider("Long strike (K1)", 15, 60, 28, step=1, key="oa_s1_k1")
    s1_k2    = s1d.slider("Short strike (K2) [K2>K1]", 20, 80, 32, step=1, key="oa_s1_k2")
    s1_ratio = st.slider("Ratio N:1 (buy N long vs sell 1 short)", 1, 3, 2, step=1, key="oa_s1_ratio")

    st.markdown("### Strategy 2 — Doomsday / Tail (ladder every few months)")
    d1, d2, d3, d4 = st.columns(4)
    s2_spend = d1.slider("Spend per ladder (acct %)", 0.0, 0.03, 0.01, step=0.001, key="oa_s2_spend")
    s2_freqM = d2.slider("Ladder frequency (months)", 2, 12, 3, step=1, key="oa_s2_freq")
    s2_mty   = d3.slider("Expiry (months)", 3, 12, 6, step=1, key="oa_s2_mty")
    s2_k     = d4.slider("Far-OTM long call strike (K)", 25, 80, 45, step=1, key="oa_s2_k")

    st.markdown("### Entry/Exit Filters (S1)")
    f1, f2, f3, f4 = st.columns(4)
    use_contango_oa = f1.checkbox("Require contango (VIX < VIX3M)", True, key="oa_contango")
    vix_lo = f2.slider("S1 enter when VIX ≥", 10, 35, 16, step=1, key="oa_vix_lo")
    vix_hi = f3.slider("… and VIX ≤", 15, 60, 32, step=1, key="oa_vix_hi")
    tp_pct_single = f4.slider("Take-profit if VIX up ≥ (%)", 5, 100, 25, step=5, key="oa_tp_pct")
    exit_on_strike = st.checkbox("Exit when VIX crosses strike (S1: K2, S2: K)", True, key="oa_exit_strike")

    st.markdown("### Pricing Proxy & Convexity")
    e1, e2, e3, e4 = st.columns(4)
    pr_base = e1.slider("Premium factor at VIX=20", 0.10, 0.60, 0.25, step=0.01, key="oa_base")
    pr_cap  = e2.slider("Premium cap", 0.20, 0.90, 0.65, step=0.01, key="oa_capr")
    pr_beta = e3.slider("Premium VIX-slope β", 0.2, 1.5, 0.8, step=0.05, key="oa_beta")
    vro_settle = e4.checkbox("Use VIX close at expiry as VRO proxy", value=True, key="oa_vro")

    g1, g2, g3, g4 = st.columns(4)
    gamma_s1_single = g1.slider("Convexity boost (S1)", 0.0, 3.0, 1.2, step=0.1, key="oa_gamma_s1")
    gamma_s2_single = g2.slider("Convexity boost (S2)", 0.0, 5.0, 2.0, step=0.1, key="oa_gamma_s2")
    pctl_thr_single  = g3.slider("Exit if VIX ≥ rolling pctile", 50, 100, 90, step=1, key="oa_pctile")
    roll_days        = g4.slider("Percentile lookback (days)", 120, 504, 252, step=6, key="oa_roll")

    # Buttons
    cbtn1, cbtn2 = st.columns(2)
    run_oa = cbtn1.button("Run OA VIX Hedge (single run)", key="oa_run")
    run_grid = cbtn2.button("Run Grid Search", key="oa_grid")

    # Data
    if yf is None:
        st.error("yfinance required.")
        st.stop()
    with st.spinner("Downloading ^VIX and ^VIX3M …"):
        raw = yf.download(["^VIX","^VIX3M"], start=start_oa, end=end_oa, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        st.warning("No data downloaded.")
        st.stop()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    vix  = close["^VIX"].dropna().rename("VIX")
    vix3 = close["^VIX3M"].reindex(vix.index).ffill().rename("VIX3M")
    idx = vix.index

    vix_rank = rolling_pct_rank(vix, int(roll_days))

    def premium_ratio_local(V, base, beta, cap):
        raw = base * (max(V, 5) / 20.0) ** beta
        return float(min(max(raw, 0.05), cap))

    # core simulator (single parameter set)
    def run_oa_once(params: dict):
        s1_sp, s1M, K1, K2, ratio = params["s1_sp"], params["s1_mty"], params["K1"], params["K2"], params["ratio"]
        s2_sp, s2M, s2freq, Kd     = params["s2_sp"], params["s2_mty"], params["s2_freqM"], params["K"]
        gamma1, gamma2 = params["gamma_s1"], params["gamma_s2"]
        tp_pct, pctl_thr = params["tp_pct"], params["pctl_thr"]

        # entry mask for S1
        if use_contango_oa:
            cont = (vix < vix3)
        else:
            cont = pd.Series(True, index=vix.index)
        band = (vix >= vix_lo) & (vix <= vix_hi)
        s1_ok = (cont & band).astype(bool)

        expiries = pd.DatetimeIndex(month_ends(idx))
        acct = float(acct0_oa); peak = acct
        positions = []; history = []; last_s2_added = None

        def room_spend():
            outstanding = sum(p["cost_usd"] for p in positions)
            return max(0.0, risk_cap * acct - outstanding)

        for d in idx:
            V = float(vix.at[d]) if d in vix.index else float("nan")

            # 1) exits
            keep=[]
            for p in positions:
                if p["kind"] == "S1":
                    long_intr  = max(0.0, V - p["K1"])
                    short_intr = max(0.0, V - p["K2"])
                    spread_intr = max(0.0, p["ratio"] * long_intr - short_intr)
                    rel_jump = max(0.0, V / p["V_at_open"] - 1.0)
                    boost_pts = gamma1 * rel_jump * (p["K2"] - p["K1"])
                    payoff_pts = max(spread_intr, boost_pts)
                    strike_cross = V >= p["K2"]
                else:
                    intr = max(0.0, V - p["K"])
                    rel_jump = max(0.0, V / p["V_at_open"] - 1.0)
                    boost_pts = gamma2 * rel_jump
                    payoff_pts = max(intr, boost_pts)
                    strike_cross = V >= p["K"]

                mtm = payoff_pts * 100.0 * p["qty"] - p["cost_usd"]
                hit_tp = (V / p["V_at_open"] - 1.0) >= (tp_pct/100.0)
                hit_strike = strike_cross if exit_on_strike else False
                hit_pctile = (vix_rank.get(d, 0.0) >= pctl_thr/100.0)

                if d >= p["exp"] or hit_tp or hit_strike or hit_pctile:
                    acct += mtm
                else:
                    keep.append(p)
            positions = keep

            # 2) ladders
            if d in expiries:
                # S1 monthly if allowed
                if s1_sp > 0 and s1_ok.get(d, False):
                    room = room_spend()
                    if room > 1:
                        spend_usd = s1_sp * acct
                        pr = premium_ratio_local(V, pr_base, pr_beta, pr_cap)
                        unit_cost = (K2 - K1) * pr * 100.0 * max(1, ratio) if K2 > K1 else pr*100.0
                        qty = int(max(0, min(spend_usd, room)//unit_cost))
                        if qty >= 1:
                            positions.append(dict(kind="S1", open=d, exp=add_months(d, int(s1M)),
                                                  cost_usd=unit_cost*qty, qty=qty, K1=float(K1), K2=float(K2),
                                                  ratio=int(ratio), V_at_open=float(V)))
                            acct -= unit_cost*qty

                # S2 every s2freq months
                s2_due = (last_s2_added is None) or ((d.year-last_s2_added.year)*12 + (d.month-last_s2_added.month) >= int(s2freq))
                if s2_due and s2_sp > 0:
                    room = room_spend()
                    if room > 1:
                        spend_usd = s2_sp * acct
                        pr2 = premium_ratio_local(V, pr_base*0.8, pr_beta, pr_cap)
                        unit_cost = pr2 * 100.0
                        qty = int(max(0, min(spend_usd, room)//unit_cost))
                        if qty >= 1:
                            positions.append(dict(kind="S2", open=d, exp=add_months(d, int(s2M)),
                                                  cost_usd=unit_cost*qty, qty=qty, K=float(Kd),
                                                  V_at_open=float(V)))
                            acct -= unit_cost*qty
                            last_s2_added = d

            # 3) track
            peak = max(peak, acct)
            history.append({"Date": d, "Equity": acct, "DD%": acct/peak - 1})

        # final settle
        Vlast = float(vix.iloc[-1])
        for p in positions:
            if p["kind"] == "S1":
                long_intr  = max(0.0, Vlast - p["K1"])
                short_intr = max(0.0, Vlast - p["K2"])
                payoff_pts = max(0.0, p["ratio"]*long_intr - short_intr)
                pnl = payoff_pts*100.0*p["qty"] - p["cost_usd"]
            else:
                intr = max(0.0, Vlast - p["K"])
                pnl = intr*100.0*p["qty"] - p["cost_usd"]
            acct += pnl
            peak = max(peak, acct)
            history.append({"Date": p["exp"], "Equity": acct, "DD%": acct/peak - 1})

        ts = pd.DataFrame(history).drop_duplicates("Date").set_index("Date").sort_index()
        eq = ts["Equity"]
        yrs = max(1e-9, (eq.index[-1]-eq.index[0]).days/365.25)
        cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1 if yrs>0 else 0.0
        mdd = (eq/eq.cummax() - 1).min()
        return eq, dict(CAGR=cagr, MaxDD=mdd, Final=eq.iloc[-1])

    # Single run
    if run_oa:
        params_single = dict(
            s1_sp=s1_spend, s1_mty=s1_mty, K1=s1_k1, K2=s1_k2, ratio=s1_ratio,
            s2_sp=s2_spend, s2_mty=s2_mty, s2_freqM=s2_freqM, K=s2_k,
            gamma_s1=gamma_s1_single, gamma_s2=gamma_s2_single,
            tp_pct=tp_pct_single, pctl_thr=pctl_thr_single
        )
        eq_oa, met = run_oa_once(params_single)
        plot_equity(pd.DataFrame({"OA VIX Hedge Equity": eq_oa}))
        c1, c2, c3 = st.columns(3)
        c1.metric("Final equity", f"${met['Final']:,.0f}")
        c2.metric("CAGR", f"{met['CAGR']*100:.2f}%")
        c3.metric("Max DD", f"{met['MaxDD']*100:.1f}%")
        st.session_state["OA_SERIES"] = eq_oa

    # Grid Search
    if run_grid:
        st.markdown("#### Grid Search Space")

        gg1, gg2, gg3 = st.columns(3)
        K1_list = gg1.text_input("K1 list", "24,26,28", key="oa_g_k1").strip()
        K2_list = gg2.text_input("K2 list (K2>K1)", "30,32,34", key="oa_g_k2").strip()
        ratio_list = gg3.text_input("Ratio list (N:1)", "1,2", key="oa_g_ratio").strip()

        gg4, gg5, gg6 = st.columns(3)
        tp_list = gg4.text_input("TP% list", "20,25,30", key="oa_g_tp").strip()
        g1_list = gg5.text_input("Gamma S1 list", "1.0,1.4,1.8", key="oa_g_g1").strip()
        g2_list = gg6.text_input("Gamma S2 list", "2.0,3.0,4.0", key="oa_g_g2").strip()

        gg7, gg8, gg9 = st.columns(3)
        pctl_list = gg7.text_input("Percentile exit list", "85,90,95", key="oa_g_pct").strip()
        s1sp_list = gg8.text_input("S1 spend %/mo list", "0.005,0.01", key="oa_g_s1sp").strip()
        s2sp_list = gg9.text_input("S2 spend %/ladder list", "0.005,0.01", key="oa_g_s2sp").strip()

        capN = st.slider("Max combinations to evaluate", 50, 800, 300, step=50, key="oa_capN")
        dd_cap = st.slider("Max DD cap (reject if < -X%)", 10, 95, 60, step=5, key="oa_ddcap")

        def parse_floats(s): 
            return [float(x) for x in s.replace("，",",").split(",") if x.strip()!=""]
        def parse_ints(s):
            return [int(float(x)) for x in s.replace("，",",").split(",") if x.strip()!=""]

        K1_candidates    = parse_ints(K1_list)
        K2_candidates    = parse_ints(K2_list)
        ratio_candidates = parse_ints(ratio_list)
        tp_candidates    = parse_ints(tp_list)
        g1_candidates    = parse_floats(g1_list)
        g2_candidates    = parse_floats(g2_list)
        pctl_candidates  = parse_ints(pctl_list)
        s1sp_candidates  = parse_floats(s1sp_list)
        s2sp_candidates  = parse_floats(s2sp_list)

        from itertools import product, islice
        grid_iter = product(K1_candidates, K2_candidates, ratio_candidates,
                            tp_candidates, g1_candidates, g2_candidates,
                            pctl_candidates, s1sp_candidates, s2sp_candidates)
        grid = list(islice(grid_iter, int(capN)))

        rows = []
        best_eq = None
        prog = st.progress(0.0)
        total = len(grid) if len(grid)>0 else 1

        for i, (K1g,K2g,rg,tp,g1g,g2g,pctl,s1spg,s2spg) in enumerate(grid, start=1):
            if K2g <= K1g: 
                prog.progress(min(1.0, i/total)); 
                continue
            params = dict(
                s1_sp=s1spg, s1_mty=s1_mty, K1=K1g, K2=K2g, ratio=rg,
                s2_sp=s2spg, s2_mty=s2_mty, s2_freqM=s2_freqM, K=s2_k,
                gamma_s1=g1g, gamma_s2=g2g, tp_pct=tp, pctl_thr=pctl
            )
            eq, met = run_oa_once(params)
            if met["MaxDD"] <= -dd_cap/100.0:
                prog.progress(min(1.0, i/total))
                continue
            rows.append(dict(
                K1=K1g, K2=K2g, ratio=rg, TP=tp, gamma_s1=g1g, gamma_s2=g2g,
                pctl=pctl, s1_sp=s1spg, s2_sp=s2spg,
                CAGR=met["CAGR"], MaxDD=met["MaxDD"], Final=met["Final"]
            ))
            if best_eq is None or (met["CAGR"] > best_eq[1]["CAGR"] or
               (abs(met["CAGR"]-best_eq[1]["CAGR"])<1e-12 and met["MaxDD"]>best_eq[1]["MaxDD"])):
                best_eq = (eq, met, params)
            prog.progress(min(1.0, i/total))

        if len(rows)==0:
            st.warning("No parameter set passed the drawdown cap. Loosen the cap or widen the grid.")
        else:
            res = pd.DataFrame(rows)
            res_disp = res.copy()
            res_disp["CAGR %"]  = (100*res_disp["CAGR"]).round(2)
            res_disp["MaxDD %"] = (100*res_disp["MaxDD"]).round(1)
            res_disp["s1_sp %"] = (100*res_disp["s1_sp"]).round(2)
            res_disp["s2_sp %"] = (100*res_disp["s2_sp"]).round(2)
            res_disp = res_disp.drop(columns=["CAGR","MaxDD"])
            res_disp = res_disp.sort_values(["CAGR %","MaxDD %","Final"], ascending=[False, False, False])
            st.markdown("#### Top Grid Results")
            st.dataframe(res_disp.head(25), use_container_width=True)

            # show best equity
            eq_best, met_best, params_best = best_eq
            st.markdown("#### Best Curve from Grid")
            plot_equity(pd.DataFrame({"OA VIX Hedge (best)": eq_best}))
            c1, c2, c3 = st.columns(3)
            c1.metric("Final equity", f"${met_best['Final']:,.0f}")
            c2.metric("CAGR", f"{met_best['CAGR']*100:.2f}%")
            c3.metric("Max DD", f"{met_best['MaxDD']*100:.1f}%")

            if st.button("Use Best Parameters & Save to Combined Portfolio", key="oa_apply_best"):
                st.session_state["OA_SERIES"] = eq_best
                st.success("Applied. Check the Combined Portfolio tab.")