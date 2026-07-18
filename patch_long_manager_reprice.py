#!/usr/bin/env python3
from pathlib import Path
p = Path("/home/shin/vix_suite/tradier_long_manager.py")
src = p.read_text()

old = '''def execute_with_reprice(client: TradierClient, underlying: str,
                         option_symbol: str, side: str,
                         quantity: int, initial_mid: float,
                         ask: float, preview: bool = False) -> dict:
    price = round(initial_mid, 2)

    if preview:
        action = "BTO" if "buy" in side else "STO"
        print(f"   [PREVIEW] {action} {option_symbol} ×{quantity} @ ${price:.2f}")
        return {"status": "preview", "price": price}

    print(f"   Placing: {side} {option_symbol} ×{quantity} @ ${price:.2f}")
    result = client.place_order(underlying, option_symbol,
                                side, quantity, price)
    order  = result.get("order", {})
    oid    = order.get("id")

    if not oid:
        print(f"   ❌ Order failed: {result}")
        return {"status": "failed"}

    print(f"   Order ID: {oid} — polling...")

    for attempt in range(MAX_REPRICE):
        time.sleep(REPRICE_SEC)
        status = client.get_order(oid)
        state  = status.get("status", "")
        print(f"   [{attempt+1}] status={state}")

        if state == "filled":
            fill_px = float(status.get("avg_fill_price", price))
            print(f"   ✅ Filled @ ${fill_px:.2f}")
            return {"status": "filled", "order_id": oid,
                    "fill_price": fill_px, "quantity": quantity,
                    "option_symbol": option_symbol}

        if state in ("canceled", "expired", "rejected"):
            return {"status": state, "order_id": oid}

        new_price = min(round(price + NUDGE * (attempt + 1), 2), ask)
        if new_price != price:
            try:
                print(f"   Repricing ${price:.2f} → ${new_price:.2f}")
                client.modify_order(oid, new_price)
                price = new_price
            except Exception as _me:
                print(f"   ⚠️ Reprice failed (market closed?): {_me}")
                print(f"   Order {oid} remains working at ${price:.2f}")
                break

    return {"status": "working", "order_id": oid, "last_price": price}'''

new = '''def execute_with_reprice(client: TradierClient, underlying: str,
                         option_symbol: str, side: str,
                         quantity: int, bid: float, ask: float,
                         sandbox: bool = True,
                         preview: bool = False) -> dict:
    """
    Most profitable execution for long/short legs.

    BTO (buy):  start at mid, nudge toward ask each interval
    STO (sell): start at mid, nudge toward bid each interval

    Sandbox: cancel + re-place (modify not supported)
    Live:    true order modification
    """
    is_buy  = "buy" in side.lower()
    mid     = round((bid + ask) / 2, 2)
    price   = mid
    floor   = bid  if not is_buy else mid
    ceil    = ask  if is_buy else mid
    nudge   = 0.05 if is_buy else 0.01
    action  = "BTO" if is_buy else "STO"

    if preview:
        print(f"   [PREVIEW] {action} {option_symbol} ×{quantity} "
              f"@ mid ${price:.2f} (bid=${bid:.2f} ask=${ask:.2f})")
        return {"status": "preview", "price": price}

    print(f"   {action} {option_symbol} ×{quantity} "
          f"@ ${price:.2f} (bid=${bid:.2f} ask=${ask:.2f})")

    try:
        result = client.place_order(underlying, option_symbol,
                                    side, quantity, price)
        oid = result.get("order", {}).get("id")
        if not oid:
            print(f"   ❌ Order failed: {result}")
            return {"status": "failed"}
        print(f"   Order {oid} placed")
    except Exception as e:
        print(f"   ❌ Place failed: {e}")
        return {"status": "failed"}

    for attempt in range(MAX_REPRICE):
        time.sleep(REPRICE_SEC)
        try:
            status = client.get_order(oid)
            state  = status.get("status", "")
        except Exception:
            state = "unknown"

        print(f"   [{attempt+1}] status={state} @ ${price:.2f}")

        if state == "filled":
            fill_px = float(status.get("avg_fill_price", price))
            print(f"   ✅ Filled @ ${fill_px:.2f}")
            return {"status": "filled", "order_id": oid,
                    "fill_price": fill_px, "quantity": quantity,
                    "option_symbol": option_symbol}

        if state in ("canceled", "expired", "rejected"):
            return {"status": state, "order_id": oid}

        # Nudge toward fill
        if is_buy:
            new_price = min(round(price + nudge, 2), ceil)
        else:
            new_price = max(round(price - nudge, 2), floor)

        if new_price == price:
            print(f"   At limit ${price:.2f} — waiting")
            continue

        print(f"   Repricing ${price:.2f} → ${new_price:.2f}")

        if sandbox:
            try:
                client.session.delete(
                    f"{client.base}/accounts/{client.account}/orders/{oid}",
                )
            except Exception:
                pass
            try:
                result  = client.place_order(underlying, option_symbol,
                                             side, quantity, new_price)
                new_oid = result.get("order", {}).get("id")
                if new_oid:
                    oid   = new_oid
                    price = new_price
            except Exception as e:
                print(f"   ⚠️ Re-place failed: {e}")
        else:
            try:
                client.modify_order(oid, new_price)
                price = new_price
            except Exception as e:
                print(f"   ⚠️ Modify failed: {e}")

    return {"status": "working", "order_id": oid, "last_price": price}'''

if old not in src:
    print("❌ not found")
else:
    p.write_text(src.replace(old, new, 1))
    print("✅ execute_with_reprice rebuilt in long manager")
