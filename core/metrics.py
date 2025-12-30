import numpy as np

def compute_cagr(equity: np.ndarray, years: float) -> float:
    if len(equity) < 2 or equity[0] <= 0 or years <= 0:
        return 0.0
    return (equity[-1] / equity[0]) ** (1.0 / years) - 1.0

def compute_sharpe(weekly_returns: np.ndarray) -> float:
    if len(weekly_returns) < 2:
        return 0.0
    mean = np.mean(weekly_returns)
    std = np.std(weekly_returns, ddof=1)
    if std <= 0:
        return 0.0
    return (mean * 52.0) / (std * np.sqrt(52.0))

def compute_max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    cum_max = np.maximum.accumulate(equity)
    dd = (equity - cum_max) / cum_max
    return float(np.min(dd))

def summarize_performance(bt_results: dict, params: dict) -> dict:
    equity = bt_results["equity"]
    weekly_returns = bt_results["weekly_returns"]
    weeks = len(equity)
    years = weeks / 52.0 if weeks > 0 else 0.0
    total_return = (equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0
    from .metrics import compute_cagr, compute_sharpe, compute_max_drawdown  # self-import guard
    cagr = compute_cagr(equity, years)
    sharpe = compute_sharpe(weekly_returns[:-1])
    max_dd = compute_max_drawdown(equity)
    return {
        "total_return_pct": total_return * 100.0,
        "total_return_dollar": equity[-1] - equity[0],
        "ending_equity": float(equity[-1]),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": bt_results.get("win_rate", 0.0),
        "trades": bt_results.get("trades", 0),
        "avg_trade_dur": bt_results.get("avg_trade_dur", 0.0),
    }
