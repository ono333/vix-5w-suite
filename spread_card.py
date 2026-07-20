#!/usr/bin/env python3
"""
spread_card.py — weekly defined-risk spread decision card.

Prints, for each width (2/5/10), the STO/BTO strikes, net credit, contracts
(constant-risk sized), max loss, and a GO / STAND-DOWN verdict against the
same gate the shadow arms use (3%-of-width credit floor + backwardation
stand-down via risk_measures slope_ratio).

Decision support for the live Fidelity spread — NOT an executor. Mirrors the
shadow arms' logic exactly by importing their primitives, so card and shadow
book always agree.

Usage:
    python3 spread_card.py                # next weekly expiry
    python3 spread_card.py --expiry 2026-07-31
"""
import sys
from datetime import date

import shadow_strategist as ss   # reuse chain fetch, strike pick, gate helpers


def build_card(target_expiry=None):
    c = ss.get_client()
    q = {x.get("symbol"): x for x in ss.as_list(
        ss.api(c, "/markets/quotes", symbols="UVXY", greeks="false")
        .get("quotes", {}).get("quote"))}
    spot = float((q.get("UVXY") or {}).get("last") or 0)

    exps = ss.as_list(ss.api(c, "/markets/options/expirations", symbol="UVXY",
                             includeAllRoots="true", strikes="false")
                      .get("expirations", {}).get("date"))
    wexp = target_expiry or ss.pick_exp(exps, *ss.SHORT_DTE)
    if not wexp:
        return None, spot, None, []

    calls = [o for o in ss.as_list(
        ss.api(c, "/markets/options/chains", symbol="UVXY", expiration=wexp,
               greeks="true").get("options", {}).get("option"))
        if o.get("option_type") == "call"]

    conn = ss.connect()
    slope = ss.latest_slope(conn)
    conn.close()

    s = ss.pick_by_delta(calls, ss.SPREAD_SHORT_DELTA)
    rows = []
    for strat, width in ss.SPREAD_WIDTHS.items():
        if not s:
            rows.append({"width": width, "verdict": "NO SHORT", "detail":
                         f"no call near delta {ss.SPREAD_SHORT_DELTA}"})
            continue
        lng = ss.pick_at_or_above(calls, s["strike"] + width)
        if not lng:
            rows.append({"width": width, "verdict": "NO LONG",
                         "detail": f"no strike at +{width:g}"})
            continue
        aw = lng["strike"] - s["strike"]
        sm, lm = ss.mid(s), ss.mid(lng)
        net = round(sm - lm, 2)
        ct = max(1, int(ss.SPREAD_RISK_BUDGET / (aw * 100)))
        maxloss = round((aw - net) * 100 * ct)
        floor = ss.GATE_CREDIT_PCT * aw
        if slope is not None and slope >= ss.GATE_SLOPE_MAX:
            verdict = "STAND-DOWN"
            detail = f"backwardation slope {slope:.3f} >= {ss.GATE_SLOPE_MAX}"
        elif net < floor:
            verdict = "STAND-DOWN"
            detail = f"net {net:.2f} < floor {floor:.2f} (3% x {aw:g})"
        else:
            verdict = "GO"
            detail = f"credit ${net*100*ct:,.0f} for ${maxloss:,.0f} risk"
        rows.append({"width": width, "short": s["strike"], "long": lng["strike"],
                     "net": net, "contracts": ct, "maxloss": maxloss,
                     "verdict": verdict, "detail": detail,
                     "short_delta": round(s["greeks"]["delta"], 3)})
    return wexp, spot, slope, rows


def main():
    target = None
    if "--expiry" in sys.argv:
        target = sys.argv[sys.argv.index("--expiry") + 1]
    wexp, spot, slope, rows = build_card(target)
    print(f"\nUVXY ${spot:.2f}   expiry {wexp}   "
          f"slope {slope if slope is None else round(slope,3)}   "
          f"{date.today()}")
    if not rows:
        print("No expiry / chain available.")
        return
    print("-" * 64)
    for r in rows:
        if r["verdict"] in ("NO SHORT", "NO LONG"):
            print(f"  ${r['width']:>2g}-wide   {r['verdict']:<11} {r['detail']}")
            continue
        print(f"  ${r['width']:>2g}-wide   {r['verdict']:<11} "
              f"STO {r['short']:g}C (d{r['short_delta']}) / BTO {r['long']:g}C  "
              f"net {r['net']:.2f} x{r['contracts']}")
        print(f"{'':<14}{r['detail']}")
    print("-" * 64)
    gos = [r for r in rows if r.get("verdict") == "GO"]
    if gos:
        best = max(gos, key=lambda r: r["net"] * 100 * r["contracts"])
        print(f"  Best credit/risk this week: ${best['width']:g}-wide "
              f"(${best['net']*100*best['contracts']:,.0f} credit)")
    else:
        print("  All widths STAND-DOWN — no qualifying spread this week.")


if __name__ == "__main__":
    main()
