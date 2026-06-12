#!/usr/bin/env python3
"""
tradier_monitor.py
──────────────────
Standalone Streamlit app for monitoring Tradier auto-trade performance.
Reads from:
  - ~/.vix_suite/execution_quality.json  (all fills)
  - ~/.vix_suite/tradier_long_state.json (current positions)
  - ~/.vix_suite/orchestrator.log        (run history)
  - Tradier sandbox API                  (live quotes)

Run: streamlit run tradier_monitor.py --server.port 8502
"""

import json
import os
import re
import requests
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

STORAGE_DIR   = Path.home() / ".vix_suite"
EXEC_QUALITY  = STORAGE_DIR / "execution_quality.json"
LONG_STATE    = STORAGE_DIR / "tradier_long_state.json"
ORCH_LOG      = STORAGE_DIR / "orchestrator.log"

TRADIER_BASE    = "https://sandbox.tradier.com/v1"
TRADIER_TOKEN   = os.environ.get("TRADIER_SANDBOX_TOKEN", "")
TRADIER_ACCOUNT = "VA88395338"

VARIANT_COLORS  = {
    "V1": "#4CAF50", "V2": "#2196F3", "V3": "#FF9800",
    "V4": "#E91E63", "V5": "#9C27B0",
}
DELTA_CEILINGS  = {"V1": 0.28, "V2": 0.22, "V3": 0.18, "V4": 0.15, "V5": 0.22}


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_fills() -> pd.DataFrame:
    if not EXEC_QUALITY.exists():
        return pd.DataFrame()
    data = json.loads(EXEC_QUALITY.read_text())
    df = pd.DataFrame(data)
    df["ts"]     = pd.to_datetime(df["ts"])
    df["date"]   = df["ts"].dt.date
    df["week"]   = df["ts"].dt.to_period("W").apply(lambda x: x.start_time.date())
    df["credit"] = df["fill_price"] * 100 * df["qty"]
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df


@st.cache_data(ttl=30)
def load_state() -> dict:
    if not LONG_STATE.exists():
        return {}
    try:
        return json.loads(LONG_STATE.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=30)
def load_log_summary() -> list:
    if not ORCH_LOG.exists():
        return []
    runs = []
    current = None
    for line in ORCH_LOG.read_text().splitlines():
        m = re.search(r"Tradier Orchestrator — (\d{4}-\d{2}-\d{2})", line)
        if m:
            current = {"date": m.group(1), "actions": [], "skips": [], "errors": []}
            runs.append(current)
            continue
        if current is None:
            continue
        if "• V" in line:
            a = re.search(r"• (V\d): STO \$(\S+) (\S+) @ \$(\S+)", line)
            if a:
                current["actions"].append({
                    "variant": a.group(1), "strike": a.group(2),
                    "expiry":  a.group(3), "credit": float(a.group(4))
                })
        if "No short strike found" in line:
            v = re.search(r"\] (V\d) —", line)
            current["skips"].append(v.group(1) if v else "?")
        if "❌" in line:
            current["errors"].append(line.strip())
    return runs


@st.cache_data(ttl=20)
def fetch_live_quotes(symbols: tuple) -> dict:
    if not symbols or not TRADIER_TOKEN:
        return {}
    try:
        hdrs = {"Authorization": f"Bearer {TRADIER_TOKEN}",
                "Accept": "application/json"}
        r = requests.get(
            f"{TRADIER_BASE}/markets/quotes",
            headers=hdrs,
            params={"symbols": ",".join(symbols), "greeks": "true"},
            timeout=10)
        r.raise_for_status()
        quotes = r.json().get("quotes", {}).get("quote", [])
        if isinstance(quotes, dict):
            quotes = [quotes]
        return {q["symbol"]: q for q in quotes}
    except Exception:
        return {}


@st.cache_data(ttl=20)
def fetch_positions() -> list:
    if not TRADIER_TOKEN:
        return []
    try:
        hdrs = {"Authorization": f"Bearer {TRADIER_TOKEN}",
                "Accept": "application/json"}
        r = requests.get(
            f"{TRADIER_BASE}/accounts/{TRADIER_ACCOUNT}/positions",
            headers=hdrs, timeout=10)
        r.raise_for_status()
        pos = r.json().get("positions", {})
        if not pos or pos == "null":
            return []
        p = pos.get("position", [])
        return p if isinstance(p, list) else [p]
    except Exception:
        return []


# ── Helpers ────────────────────────────────────────────────────────────────────

def color_dte(dte: int) -> str:
    if dte <= 7:  return "#cc0000"
    if dte <= 14: return "#cc6600"
    return "#16a34a"


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_overview():
    st.header("📊 Overview")
    fills = load_fills()
    state = load_state()
    if fills.empty:
        st.warning("No fill data in execution_quality.json")
        return

    sto = fills[fills["side"] == "STO"]
    total_credits = sto["credit"].sum()
    long_cost = sum(
        v.get("long", {}).get("fill_price", 0) * 100
        for v in state.get("variants", {}).values()
        if v.get("long")
    )
    net_pnl = total_credits - long_cost

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Credits",   f"${total_credits:,.0f}")
    c2.metric("Long Leg Cost",   f"${long_cost:,.0f}")
    c3.metric("Net P&L",         f"${net_pnl:,.0f}",
              delta=f"{(net_pnl/long_cost*100):.1f}%" if long_cost else None)
    c4.metric("Total STO Fills", len(sto))
    c5.metric("Avg Credit/Fill", f"${sto['credit'].mean():.2f}")

    st.divider()

    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("Weekly Credits by Variant")
    weekly = sto.groupby(["week","variant"])["credit"].sum().reset_index()
    weekly["week_str"] = weekly["week"].astype(str)
    fig = px.bar(weekly, x="week_str", y="credit", color="variant",
                 color_discrete_map=VARIANT_COLORS, barmode="group", height=300,
                 labels={"week_str":"Week","credit":"Credit ($)"})
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font_color="#fafafa", margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Credits")
    sto_s = sto.sort_values("ts").copy()
    sto_s["cumulative"] = sto_s["credit"].cumsum()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sto_s["ts"], y=sto_s["cumulative"],
        mode="lines+markers", line=dict(color="#4CAF50", width=2),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.1)"))
    fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                       font_color="#fafafa", height=260,
                       yaxis_title="Cumulative ($)",
                       margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig2, use_container_width=True)


def page_positions():
    st.header("📋 Live Positions")
    state    = load_state()
    today    = date.today()
    positions = fetch_positions()
    symbols  = tuple(p.get("symbol","") for p in positions if p.get("symbol",""))
    quotes   = fetch_live_quotes(symbols)

    for key in ["V1","V2","V3","V4","V5"]:
        vs      = state.get("variants",{}).get(key,{})
        lg      = vs.get("long")
        sh      = vs.get("short")
        color   = VARIANT_COLORS[key]
        ceiling = DELTA_CEILINGS[key]

        with st.expander(f"{key}  —  δ ceiling {ceiling}", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Long Leg**")
                if lg:
                    exp  = date.fromisoformat(str(lg["expiry"]))
                    dte  = (exp - today).days
                    q    = quotes.get(lg.get("symbol",""), {})
                    bid  = float(q.get("bid",0) or 0)
                    ask  = float(q.get("ask",0) or 0)
                    mid  = round((bid+ask)/2, 2)
                    cost = lg["fill_price"] * 100
                    curr = mid * 100
                    pnl  = curr - cost
                    pct  = (pnl/cost*100) if cost else 0
                    st.markdown(
                        f"<span style='color:{color};font-weight:700'>${lg['strike']:.0f}C</span> "
                        f"exp <span style='color:{color_dte(dte)}'>{exp} DTE={dte}</span>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"Cost **${cost:.0f}** · Now **${curr:.0f}** · "
                        f"<span style='color:{'#16a34a' if pnl>=0 else '#cc0000'}'>"
                        f"P&L ${pnl:+.0f} ({pct:+.1f}%)</span>",
                        unsafe_allow_html=True)
                    if dte <= 14:
                        st.error(f"⚠️ DTE={dte} — Roll needed soon")
                else:
                    st.markdown("❌ No long leg")

            with c2:
                st.markdown("**Short Leg**")
                if sh:
                    exp    = date.fromisoformat(str(sh["expiry"]))
                    dte    = (exp - today).days
                    q      = quotes.get(sh.get("symbol",""), {})
                    bid    = float(q.get("bid",0) or 0)
                    greeks = q.get("greeks") or {}
                    delta  = abs(float(greeks.get("delta",0) or 0))
                    cost   = sh.get("fill_price",0) * 100
                    curr   = bid * 100
                    pnl    = cost - curr
                    dc     = "#cc0000" if delta >= ceiling else "#16a34a"
                    st.markdown(
                        f"<span style='color:{color};font-weight:700'>${sh['strike']:.0f}C</span> "
                        f"exp <span style='color:{color_dte(dte)}'>{exp} DTE={dte}</span>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"Credit **${cost:.0f}** · BTC **${curr:.0f}** · "
                        f"<span style='color:{'#16a34a' if pnl>=0 else '#cc0000'}'>"
                        f"P&L ${pnl:+.0f}</span> · "
                        f"δ=<span style='color:{dc}'>{delta:.3f}</span>",
                        unsafe_allow_html=True)
                else:
                    st.markdown("📋 No short — enters next Mon/Fri 10:15am ET")


def page_history():
    st.header("📜 Trade History")
    fills = load_fills()
    if fills.empty:
        st.warning("No fill data found.")
        return

    c1, c2, c3 = st.columns(3)
    vf = c1.multiselect("Variant", ["V1","V2","V3","V4","V5"],
                        default=["V1","V2","V3","V4","V5"])
    sf = c2.multiselect("Side", fills["side"].unique().tolist(),
                        default=fills["side"].unique().tolist())
    n  = c3.selectbox("Show last", [20,50,100,999], index=1)

    df = fills[fills["variant"].isin(vf) & fills["side"].isin(sf)]
    df = df.sort_values("ts", ascending=False).head(n).copy()
    df["ts"]         = df["ts"].dt.strftime("%Y-%m-%d %H:%M")
    df["fill_price"] = df["fill_price"].map("${:.2f}".format)
    df["target_mid"] = df["target_mid"].map("${:.2f}".format)
    df["slippage"]   = df["slippage"].map("${:+.2f}".format)
    df["credit"]     = df["credit"].map("${:.0f}".format)

    st.dataframe(df[["ts","variant","side","strike","expiry",
                      "fill_price","target_mid","slippage","credit","order_id"]],
                 use_container_width=True, height=480)

    st.subheader("Per-Variant Summary")
    sto = fills[fills["side"]=="STO"]
    summary = sto.groupby("variant").agg(
        fills=("credit","count"),
        total=("credit","sum"),
        avg=("credit","mean"),
        avg_slip=("slippage","mean"),
    ).reset_index()

    cols = st.columns(5)
    for i, row in summary.iterrows():
        c = VARIANT_COLORS.get(row["variant"],"#888")
        cols[i%5].markdown(
            f"<div style='background:#1a1a2e;border-left:3px solid {c};"
            f"padding:12px;border-radius:4px'>"
            f"<div style='color:{c};font-weight:700'>{row['variant']}</div>"
            f"<div style='color:#fff;font-size:20px;font-weight:700'>${row['total']:.0f}</div>"
            f"<div style='color:#aaa;font-size:11px'>{row['fills']:.0f} fills · "
            f"avg ${row['avg']:.2f}</div>"
            f"<div style='color:#888;font-size:11px'>slip ${row['avg_slip']*100:+.2f}</div>"
            f"</div>", unsafe_allow_html=True)


def page_execution():
    st.header("⚡ Execution Quality")
    fills = load_fills()
    if fills.empty:
        st.warning("No fill data.")
        return

    import plotly.express as px
    sto = fills[fills["side"]=="STO"].copy()
    sto["slip_pct"] = sto["slippage"] / sto["target_mid"] * 100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Slippage",    f"${sto['slippage'].mean()*100:+.2f}")
    c2.metric("Worst Slippage",  f"${sto['slippage'].min()*100:+.2f}")
    c3.metric("Best Slippage",   f"${sto['slippage'].max()*100:+.2f}")
    c4.metric("Avg vs Mid",      f"{sto['slip_pct'].mean():+.1f}%")

    st.divider()
    fig = px.scatter(sto, x="ts", y="slippage", color="variant",
                     color_discrete_map=VARIANT_COLORS, height=280,
                     hover_data=["strike","fill_price","target_mid"],
                     labels={"ts":"Date","slippage":"Slippage ($)"})
    fig.add_hline(y=0, line_dash="dash", line_color="#666")
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font_color="#fafafa", margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)


def page_runlog():
    st.header("🗓️ Orchestrator Run Log")
    runs = load_log_summary()
    if not runs:
        st.warning("No run history in orchestrator.log")
        return

    rows = []
    for r in reversed(runs):
        credits = sum(a["credit"] for a in r["actions"]) * 100
        rows.append({
            "Date":     r["date"],
            "Actions":  len(r["actions"]),
            "Skips":    len(r["skips"]),
            "Errors":   len(r["errors"]),
            "Credits":  f"${credits:.0f}" if credits else "$0",
            "Variants": ", ".join(a["variant"] for a in r["actions"]) or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=380)

    st.subheader("Last 10 Runs — Detail")
    for r in reversed(runs[-10:]):
        credits = sum(a["credit"] for a in r["actions"]) * 100
        with st.expander(f"{r['date']} — {len(r['actions'])} actions · ${credits:.0f}"):
            for a in r["actions"]:
                c = VARIANT_COLORS.get(a["variant"],"#888")
                st.markdown(
                    f"<span style='color:{c}'>{a['variant']}</span> "
                    f"STO ${a['strike']}C {a['expiry']} @ **${a['credit']:.2f}**",
                    unsafe_allow_html=True)
            if r["skips"]:
                st.warning(f"No strike found: {', '.join(set(r['skips']))}")
            for e in r["errors"][:3]:
                st.error(e)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Tradier Monitor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("📈 Tradier Monitor")
    st.sidebar.caption(f"Account: {TRADIER_ACCOUNT}")
    st.sidebar.caption(f"As of: {datetime.now().strftime('%H:%M:%S ET')}")
    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    page = st.sidebar.radio("Page", [
        "📊 Overview", "📋 Positions", "📜 Trade History",
        "⚡ Execution Quality", "🗓️ Run Log",
    ])

    fills = load_fills()
    total = fills[fills["side"]=="STO"]["credit"].sum() if not fills.empty else 0
    state = load_state()
    urgent = [
        k for k,v in state.get("variants",{}).items()
        if v.get("long") and
        (date.fromisoformat(str(v["long"]["expiry"])) - date.today()).days <= 14
    ]

    ca, cb, cc = st.columns([2,2,1])
    ca.markdown(
        f"<div style='background:#1a1a2e;padding:8px 16px;border-radius:4px;"
        f"border-left:3px solid #4CAF50'>"
        f"<span style='color:#aaa;font-size:11px'>TOTAL CREDITS</span><br>"
        f"<span style='color:#4CAF50;font-size:22px;font-weight:700'>${total:,.0f}</span>"
        f"</div>", unsafe_allow_html=True)
    cb.markdown(
        f"<div style='background:#1a1a2e;padding:8px 16px;border-radius:4px;"
        f"border-left:3px solid #2196F3'>"
        f"<span style='color:#aaa;font-size:11px'>FILLS RECORDED</span><br>"
        f"<span style='color:#2196F3;font-size:22px;font-weight:700'>"
        f"{len(fills) if not fills.empty else 0}</span>"
        f"</div>", unsafe_allow_html=True)
    uc = "#cc0000" if urgent else "#4CAF50"
    cc.markdown(
        f"<div style='background:#1a1a2e;padding:8px 16px;border-radius:4px;"
        f"border-left:3px solid {uc}'>"
        f"<span style='color:#aaa;font-size:11px'>URGENT</span><br>"
        f"<span style='color:{uc};font-size:22px;font-weight:700'>"
        f"{', '.join(urgent) if urgent else '✓'}</span>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if   page == "📊 Overview":          page_overview()
    elif page == "📋 Positions":         page_positions()
    elif page == "📜 Trade History":     page_history()
    elif page == "⚡ Execution Quality": page_execution()
    elif page == "🗓️ Run Log":           page_runlog()


if __name__ == "__main__":
    main()
