def apply_exit_rules(i: int, price: float, next_price: float, entry_price: float, params: dict):
    target_mult = params.get("target_multiple", 1.2)
    if next_price >= entry_price * target_mult:
        return True, True
    if next_price <= entry_price * 0.5:
        return True, False
    return False, False
