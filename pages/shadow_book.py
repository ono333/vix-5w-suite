#!/usr/bin/env python3
"""pages/shadow_book.py — read-only Streamlit view of the shadow strategist book.

Auto-discovered by the existing 8501 app (app.py) via the pages/ directory.
Reads ~/.vix_suite/market_data.db only; never writes.

Shows: per-strategy P&L summary, cumulative P&L curves built from marks,
open positions with latest mark, recent trades, live take-bid refs.
"""

import os
import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

DB_PATH = os.path.expanduser("~/.vix_suite/market_data.db")
STRATS = ["vertical", "vertical_gated", "vertical_defended",
          "ratio_wing", "diagonal", "long_spike"]

st.set_page_config(page_title="Shadow Book", layout="wide")
st.title("Shadow Book")
st.caption("Forward-only hypothetical strategies, mid fills. Never places orders. "
           "Read-only view; source of record is shadow_strategist.py.")


@st.cache_data(ttl=300)
def load_tables():
    conn = sqlite3.connect(DB_PATH)
    try:
        pos = pd.read_sql_query("SELECT * FROM shadow_positions", conn)
        trades = pd.read_sql_query(
            "SELECT * FROM shadow_trades ORDER BY id DESC LIMIT 400", conn)
        marks = pd.read_sql_query("SELECT * FROM shadow_marks", conn)
    finally:
        conn.close()
    return pos, trades, marks


try:
    pos, trades, marks = load_tables()
except Exception as e:
    st.error(f"Cannot read {DB_PATH}: {e}")
    st.stop()

if pos.empty and trades.empty:
    st.info("Shadow book is empty — first entries land on the next "
            "market-hours run (Mon/Fri after the orchestrator).")
    st.stop()

# ---------------------------------------------------------------- summary ---
closed = pos[(pos["status"] != "open") & pos["close_price"].notna()].copy()
closed["pnl"] = (closed["close_price"] - closed["open_price"]) * closed["qty"] * 100

open_pos = pos[pos["status"] == "open"].copy()
last_mark = (marks.sort_values("id").groupby("position_id")["mid"].last()
             if not marks.empty else pd.Series(dtype=float))
open_pos["last_mid"] = open_pos["id"].map(last_mark)
open_pos["unreal"] = ((open_pos["last_mid"] - open_pos["open_price"])
                      * open_pos["qty"] * 100)

rows = []
for s in STRATS:
    r = closed.loc[closed["strategy"] == s, "pnl"].sum()
    u = open_pos.loc[open_pos["strategy"] == s, "unreal"].sum(min_count=1)
    u = 0.0 if pd.isna(u) else u
    rows.append({"strategy": s,
                 "open": int((open_pos["strategy"] == s).sum()),
                 "closed": int((closed["strategy"] == s).sum()),
                 "realized": round(r, 2), "unrealized": round(u, 2),
                 "total": round(r + u, 2)})
summary = pd.DataFrame(rows)

c1, c2, c3 = st.columns(3)
c1.metric("Total P&L (all arms)", f"${summary['total'].sum():,.2f}")
c2.metric("Open legs", int(summary["open"].sum()))
c3.metric("Closed legs", int(summary["closed"].sum()))
st.dataframe(summary, hide_index=True, use_container_width=True)

# ------------------------------------------------------------- P&L curves ---
if not marks.empty:
    m = marks.merge(pos[["id", "open_price", "qty"]].rename(
        columns={"id": "position_id"}), on="position_id", how="left")
    m["unreal"] = (m["mid"] - m["open_price"]) * m["qty"] * 100
    m["day"] = m["ts"].str[:10]
    unreal_daily = (m.groupby(["day", "strategy"])["unreal"].sum()
                    .reset_index())

    if not closed.empty:
        closed["day"] = closed["close_ts"].str[:10]
        real_daily = (closed.groupby(["day", "strategy"])["pnl"].sum()
                      .groupby(level=1).cumsum().reset_index()
                      .rename(columns={"pnl": "realized_cum"}))
    else:
        real_daily = pd.DataFrame(columns=["day", "strategy", "realized_cum"])

    days = sorted(unreal_daily["day"].unique())
    curve = []
    for s in STRATS:
        u = unreal_daily[unreal_daily["strategy"] == s].set_index("day")["unreal"]
        rc = (real_daily[real_daily["strategy"] == s]
              .set_index("day")["realized_cum"]) if not real_daily.empty \
            else pd.Series(dtype=float)
        rc = rc.reindex(days).ffill().fillna(0.0)
        total = rc.add(u.reindex(days).fillna(0.0))
        for d, v in total.items():
            curve.append({"day": d, "strategy": s, "total_pnl": v})
    curve = pd.DataFrame(curve)
    if not curve.empty and len(days) > 1:
        st.subheader("Cumulative P&L by strategy")
        st.line_chart(curve.pivot(index="day", columns="strategy",
                                  values="total_pnl"))
    elif not curve.empty:
        st.caption("P&L curve appears once there are marks on 2+ days.")

# --------------------------------------------------------- open positions ---
st.subheader("Open positions")
if open_pos.empty:
    st.caption("None.")
else:
    view = open_pos[["strategy", "occ", "expiry", "strike", "qty",
                     "open_price", "last_mid", "unreal", "open_ts", "note"]].copy()
    view["dte"] = view["expiry"].map(
        lambda e: (date.fromisoformat(e) - date.today()).days)
    st.dataframe(view.sort_values(["strategy", "expiry"]),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- trades ----
st.subheader("Recent trades")
tview = trades[trades["strategy"] != "live_takebid_ref"]
if tview.empty:
    st.caption("None.")
else:
    st.dataframe(tview[["ts", "strategy", "action", "occ", "strike", "qty",
                        "price", "bid", "ask", "spot", "vix", "vix_pctile",
                        "note"]].head(120),
                 hide_index=True, use_container_width=True)

# ------------------------------------------------------ live take-bid ref ---
tb = trades[trades["strategy"] == "live_takebid_ref"]
if not tb.empty:
    st.subheader("Live take-bid references")
    st.caption("Fill vs bid at entry (from the live orchestrator hook).")
    st.dataframe(tb[["ts", "occ", "price", "bid", "ask", "note"]],
                 hide_index=True, use_container_width=True)

st.caption(f"DB: {DB_PATH} — cached 5 min; rerun page to refresh sooner.")
