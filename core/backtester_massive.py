#!/usr/bin/env python3
"""
Realistic VIX / UVXY 5-Weekly backtester using Massive historical option aggregates.

UPDATED: Uses Massive's contract ticker format and list_aggs endpoint for
real historical option prices instead of chain snapshots.

Massive Contract Ticker Format:
    O:{UNDERLYING}{YYMMDD}{C/P}{STRIKE_8_DIGITS}
    Example: O:UVXY241220C00040000 = UVXY Dec 20 2024 $40 Call

Pricing Strategy:
    1. Generate contract tickers for selected long/short strikes
    2. Pull historical aggregates (day bars) for that date
    3. Use close price, or (high+low)/2 as fallback
    4. Fallback to Black-Scholes if no data (common for early years/thin strikes)

Position structures:
    - mode == "diagonal": long LEAP call + short weekly call
    - mode == "long_only": long call only
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import log, sqrt, exp
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------
# Black-Scholes fallback pricing
# ---------------------------------------------------------------------

def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Vanilla Black-Scholes call price.
    Used as fallback when Massive data is unavailable.
    """
    try:
        if S <= 0.0 or K <= 0.0:
            return 0.0
        if T <= 0.0:
            return max(S - K, 0.0)
        if sigma <= 0.0:
            return max(S - K * exp(-r * T), 0.0)

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# Massive API Client with Keychain Support
# ---------------------------------------------------------------------

def get_api_key_from_keychain(service: str = "MASSIVE_API_KEY", account: str = "MASSIVE_API_KEY") -> Optional[str]:
    """
    Retrieve API key from macOS Keychain.
    
    Falls back to environment variable if keychain access fails.
    """
    import subprocess
    import platform
    
    # Try macOS Keychain first
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                [
                    "security", "find-generic-password",
                    "-s", service,
                    "-a", account,
                    "-w"  # Output password only
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            if os.getenv("MASSIVE_DEBUG") == "1":
                print(f"[Massive] Keychain access failed: {e}")
    
    # Fallback to environment variable
    return os.getenv("MASSIVE_API_KEY")


_massive_client = None
_client_type = None  # Track which client library we're using

def get_massive_client():
    """Get or create Massive/Polygon REST client."""
    global _massive_client, _client_type
    
    if _massive_client is None:
        # Try keychain first, then environment variable
        api_key = get_api_key_from_keychain()
        
        if not api_key:
            raise RuntimeError(
                "MASSIVE_API_KEY not found. Either:\n"
                "1. Add to macOS Keychain: security add-generic-password -s MASSIVE_API_KEY -a MASSIVE_API_KEY -w 'your-key'\n"
                "2. Set environment variable: export MASSIVE_API_KEY='your-key'"
            )
        
        # Try different client libraries
        client = None
        
        # Try 1: massive-client
        try:
            from massive import RESTClient
            client = RESTClient(api_key)
            _client_type = "massive"
            if os.getenv("MASSIVE_DEBUG") == "1":
                print("[Massive] Using massive-client library")
        except ImportError:
            pass
        
        # Try 2: polygon-api-client (similar API)
        if client is None:
            try:
                from polygon import RESTClient as PolygonRESTClient
                client = PolygonRESTClient(api_key)
                _client_type = "polygon"
                if os.getenv("MASSIVE_DEBUG") == "1":
                    print("[Massive] Using polygon-api-client library")
            except ImportError:
                pass
        
        if client is None:
            raise RuntimeError(
                "No compatible API client installed. Install one of:\n"
                "  pip install massive-client\n"
                "  pip install polygon-api-client"
            )
        
        _massive_client = client
    
    return _massive_client


def get_contract_ticker(
    underlying: str,
    exp_date_str: str,
    strike: float,
    call_put: str = 'C'
) -> str:
    """
    Generate Massive option contract ticker.
    
    Format: O:{UNDERLYING}{YYMMDD}{C/P}{STRIKE_8_DIGITS}
    
    Parameters
    ----------
    underlying : str
        Underlying symbol (e.g., "UVXY", "VIX")
    exp_date_str : str
        Expiration date as "YYYY-MM-DD"
    strike : float
        Strike price (e.g., 40.0)
    call_put : str
        "C" for call, "P" for put
        
    Returns
    -------
    str
        Contract ticker like "O:UVXY241220C00040000"
    """
    # Parse date and format as YYMMDD
    if '-' in exp_date_str:
        # YYYY-MM-DD format
        exp_dt = datetime.strptime(exp_date_str, '%Y-%m-%d')
    else:
        exp_dt = datetime.strptime(exp_date_str, '%Y%m%d')
    
    exp_yymmdd = exp_dt.strftime('%y%m%d')
    
    # Strike padded to 8 digits with 3 decimal places implied
    # e.g., 40.0 -> 00040000, 12.50 -> 00012500
    strike_int = int(strike * 1000)
    strike_padded = f"{strike_int:08d}"
    
    # Call/Put indicator
    cp = 'C' if call_put.upper() in ('C', 'CALL') else 'P'
    
    return f"O:{underlying.upper()}{exp_yymmdd}{cp}{strike_padded}"


def get_historical_price(
    contract_ticker: str,
    trade_date_str: str,
    use_cache: bool = True,
) -> Optional[float]:
    """
    Get historical option price from Polygon.io aggregates.
    
    Parameters
    ----------
    contract_ticker : str
        Options contract ticker (e.g., "O:UVXY241220C00040000")
    trade_date_str : str
        Trade date as "YYYY-MM-DD"
    use_cache : bool
        Whether to use local caching (not implemented yet)
        
    Returns
    -------
    float or None
        Option price, or None if not available
    """
    debug = os.getenv("MASSIVE_DEBUG") == "1"
    
    try:
        client = get_massive_client()
        
        # Calculate next day for range
        trade_dt = datetime.strptime(trade_date_str, '%Y-%m-%d')
        next_day = (trade_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        
        if debug:
            print(f"[Polygon] Requesting: {contract_ticker} from {trade_date_str} to {next_day}")
        
        # Try get_aggs first (simpler API, returns list directly)
        try:
            aggs = client.get_aggs(
                contract_ticker,
                1,  # multiplier
                "day",  # timespan
                trade_date_str,  # from
                next_day  # to
            )
            
            if aggs and len(aggs) > 0:
                bar = aggs[0]
                if debug:
                    print(f"[Polygon] Got bar: {bar}")
                
                # get_aggs returns Agg objects with attributes
                if hasattr(bar, 'close') and bar.close and bar.close > 0:
                    return float(bar.close)
                if hasattr(bar, 'open') and bar.open and bar.open > 0:
                    return float(bar.open)
                if hasattr(bar, 'high') and hasattr(bar, 'low') and bar.high and bar.low:
                    return (float(bar.high) + float(bar.low)) / 2
                if hasattr(bar, 'vwap') and bar.vwap and bar.vwap > 0:
                    return float(bar.vwap)
                    
        except AttributeError:
            # Client might not have get_aggs, try list_aggs
            pass
        
        # Fallback to list_aggs with millisecond timestamps
        from_ts = int(trade_dt.timestamp() * 1000)
        to_ts = int((trade_dt + timedelta(days=1)).timestamp() * 1000)
        
        if debug:
            print(f"[Polygon] Trying list_aggs with timestamps: {from_ts} to {to_ts}")
        
        aggs = client.list_aggs(
            contract_ticker,
            multiplier=1,
            timespan="day",
            from_=from_ts,
            to=to_ts
        )
        
        # list_aggs might return a generator
        aggs_list = list(aggs) if hasattr(aggs, '__iter__') and not isinstance(aggs, (list, tuple)) else aggs
        
        if debug:
            print(f"[Polygon] list_aggs returned: {type(aggs_list)}, length: {len(aggs_list) if aggs_list else 0}")
        
        if aggs_list and len(aggs_list) > 0:
            bar = aggs_list[0]
            
            if debug:
                print(f"[Polygon] Bar type: {type(bar)}, content: {bar}")
            
            # Handle both dict-style and attribute-style access
            def get_val(obj, *keys):
                """Get value from either dict or object, trying multiple key names."""
                for key in keys:
                    if isinstance(obj, dict):
                        if key in obj:
                            return obj[key]
                    else:
                        if hasattr(obj, key):
                            return getattr(obj, key)
                return None
            
            # Try close price first
            close = get_val(bar, 'c', 'close')
            if close is not None and close > 0:
                if debug:
                    print(f"[Polygon] Got close={close} for {contract_ticker}")
                return float(close)
            
            # Try open price
            open_px = get_val(bar, 'o', 'open')
            if open_px is not None and open_px > 0:
                return float(open_px)
            
            # Fallback to (high + low) / 2
            high = get_val(bar, 'h', 'high')
            low = get_val(bar, 'l', 'low')
            if high is not None and low is not None and high > 0 and low > 0:
                return (float(high) + float(low)) / 2
            
            # Try volume-weighted average
            vwap = get_val(bar, 'vw', 'vwap')
            if vwap is not None and vwap > 0:
                return float(vwap)
        else:
            if debug:
                print(f"[Polygon] No results for {contract_ticker} on {trade_date_str}")
                
    except Exception as e:
        if debug:
            print(f"[Polygon] Error {contract_ticker} {trade_date_str}: {e}")
            import traceback
            traceback.print_exc()
    
    return None


def get_available_expirations(
    underlying: str,
    trade_date: datetime,
    target_dte_days: int,
    dte_tolerance_days: int = 28,
) -> List[str]:
    """
    Generate potential expiration dates to try.
    
    Since we can't query available expirations directly, we generate
    likely candidates (Fridays for weekly, 3rd Friday for monthly).
    
    Returns list of date strings in YYYY-MM-DD format.
    """
    expirations = []
    
    # Calculate target expiration date
    target_exp = trade_date + timedelta(days=target_dte_days)
    
    # Generate Friday dates around the target
    for delta in range(-dte_tolerance_days, dte_tolerance_days + 1, 7):
        check_date = target_exp + timedelta(days=delta)
        # Find nearest Friday
        days_to_friday = (4 - check_date.weekday()) % 7
        friday = check_date + timedelta(days=days_to_friday)
        
        if friday > trade_date:
            exp_str = friday.strftime('%Y-%m-%d')
            if exp_str not in expirations:
                expirations.append(exp_str)
    
    # Sort by distance from target
    expirations.sort(key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_exp).days))
    
    return expirations[:10]  # Limit to 10 candidates


def find_best_contract(
    underlying: str,
    trade_date_str: str,
    spot: float,
    target_strike: float,
    target_dte_days: int,
    call_put: str = 'C',
    sigma: float = 0.80,
    r: float = 0.03,
    sigma_mult: float = 1.0,
) -> Tuple[Optional[str], float, Optional[str], int]:
    """
    Find the best available contract near target parameters.
    
    Returns
    -------
    Tuple of:
        - contract_ticker (or None if using BS fallback)
        - price
        - expiration_str (or None)
        - actual_dte
    """
    trade_dt = datetime.strptime(trade_date_str, '%Y-%m-%d')
    
    # Try different strike levels around target
    strike_candidates = [
        round(target_strike),
        round(target_strike * 2) / 2,  # Round to nearest 0.5
        round(target_strike + 0.5),
        round(target_strike - 0.5),
        round(target_strike + 1),
        round(target_strike - 1),
        round(target_strike + 2),
        round(target_strike - 2),
    ]
    # Remove duplicates and invalid strikes
    strike_candidates = [s for s in dict.fromkeys(strike_candidates) if s > 0]
    
    # Get potential expirations
    expirations = get_available_expirations(underlying, trade_dt, target_dte_days)
    
    debug = os.getenv("MASSIVE_DEBUG") == "1"
    
    # Try to find a contract with real data
    for exp_str in expirations:
        for strike in strike_candidates:
            ticker = get_contract_ticker(underlying, exp_str, strike, call_put)
            price = get_historical_price(ticker, trade_date_str)
            
            if price is not None and price > 0:
                exp_dt = datetime.strptime(exp_str, '%Y-%m-%d')
                actual_dte = (exp_dt - trade_dt).days
                if debug:
                    print(f"[Massive] FOUND: {ticker} @ ${price:.2f} (target strike={target_strike:.1f}, actual={strike})")
                return ticker, price, exp_str, actual_dte
    
    # Fallback to Black-Scholes
    T = target_dte_days / 365.0
    bs_price = bs_call_price(spot, target_strike, r, sigma * sigma_mult, T)
    
    if debug:
        print(f"[Massive] FALLBACK BS: {underlying} strike={target_strike:.1f} dte={target_dte_days}d @ ${bs_price:.2f}")
    
    return None, bs_price, None, target_dte_days


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class OptionPosition:
    symbol: str
    quantity: int
    strike: float
    expiration_str: Optional[str]
    dte_weeks: int
    option_type: str  # "C" or "P"
    is_long: bool
    contract_ticker: Optional[str] = None
    using_bs_fallback: bool = False


# ---------------------------------------------------------------------
# Main Massive-based backtest
# ---------------------------------------------------------------------

def run_backtest_massive(
    vix_weekly: pd.Series,
    params: Dict[str, Any],
    *,
    symbol: str = "UVXY",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Realistic backtest using Massive historical option aggregates.
    
    Parameters
    ----------
    vix_weekly : pd.Series
        Weekly underlying closes (index = weekly dates)
    params : dict
        Same schema as synthetic run_backtest
    symbol : str
        Underlying symbol for Massive (e.g., "UVXY", "VXX")
    progress_cb : callable
        Optional callback (cur_step, total_steps) for UI progress
        
    Returns
    -------
    dict
        equity, weekly_returns, realized_weekly, unrealized_weekly,
        trades, win_rate, avg_trade_dur, trade_log
    """
    
    # ----------------- Parameters -----------------
    initial_cap = float(params.get("initial_capital", 250_000))
    
    alloc_raw = float(params.get("alloc_pct", 0.01))
    if alloc_raw > 1.0:
        alloc_pct = alloc_raw / 100.0
    else:
        alloc_pct = alloc_raw
    
    mode = params.get("mode", "diagonal")
    
    entry_pct = float(params.get("entry_percentile", 0.10))
    entry_lb = int(params.get("entry_lookback_weeks", 52))
    
    otm_pts = float(params.get("otm_pts", 10.0))
    target_mult = float(params.get("target_mult", 1.20))
    exit_mult = float(params.get("exit_mult", 0.50))
    sigma_mult = float(params.get("sigma_mult", 1.0))
    
    raw_long_dte_weeks = int(params.get("long_dte_weeks", 26))
    
    # Cap DTE for decaying ETPs
    underlying_upper = symbol.upper()
    if underlying_upper in ("UVXY", "VXX"):
        long_dte_weeks = min(raw_long_dte_weeks, 13)  # ~3 months max
    else:
        long_dte_weeks = raw_long_dte_weeks
    
    fee = float(params.get("fee_per_contract", 0.65))
    r = float(params.get("risk_free", params.get("risk_free_rate", 0.03)))
    realism = float(params.get("realism", 1.0))
    slippage_bps = float(params.get("slippage_bps", 5.0))
    
    # Base sigma for BS fallback
    base_sigma = 0.80 if underlying_upper in ("UVXY", "VXX") else 0.50
    
    MAX_QTY = 10_000
    LIQUIDATION_FLOOR = 0.0
    
    prices = np.asarray(vix_weekly.values).ravel().astype(float)
    dates = vix_weekly.index.tolist()
    n = len(prices)
    
    if n < entry_lb + 2:
        return {
            "equity": np.asarray([initial_cap], float),
            "weekly_returns": np.asarray([], float),
            "realized_weekly": np.asarray([], float),
            "unrealized_weekly": np.asarray([], float),
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_dur": 0.0,
            "trade_log": [],
        }
    
    # ----------------- Percentile series -----------------
    pct = np.full(n, np.nan, float)
    lb = max(1, entry_lb)
    for i in range(lb, n):
        w = prices[i - lb:i]
        pct[i] = (w < prices[i]).mean()
    
    # ----------------- State -----------------
    equity: List[float] = [initial_cap]
    realized_weekly: List[float] = [0.0]
    unrealized_weekly: List[float] = [0.0]
    weekly_returns: List[float] = [0.0]
    trade_log: List[Dict[str, Any]] = []
    win_flags: List[bool] = []
    durations: List[int] = []
    
    have_pos = False
    long_pos: Optional[OptionPosition] = None
    short_pos: Optional[OptionPosition] = None
    entry_notional = 0.0
    entry_week_idx: Optional[int] = None
    entry_long_price = 0.0
    
    total_steps = max(n - 1, 1)
    bs_fallback_count = 0
    massive_price_count = 0
    
    # ----------------- Main Loop -----------------
    for i in range(1, n):
        if progress_cb is not None:
            try:
                progress_cb(i, total_steps)
            except Exception:
                pass
        
        S = float(prices[i])
        trade_date = pd.to_datetime(dates[i])
        trade_date_str = trade_date.strftime('%Y-%m-%d')
        prev_eq = float(equity[-1])
        
        # If wiped out and flat, stay flat
        if prev_eq <= LIQUIDATION_FLOOR and not have_pos:
            equity.append(LIQUIDATION_FLOOR)
            realized_weekly.append(0.0)
            unrealized_weekly.append(0.0)
            weekly_returns.append(0.0)
            continue
        
        # ----------------------------------------------------------
        # ENTRY
        # ----------------------------------------------------------
        if (not have_pos) and np.isfinite(pct[i]) and pct[i] <= entry_pct:
            capital = prev_eq * alloc_pct
            if capital <= 0.0:
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue
            
            # Target strikes
            long_strike = S + otm_pts
            short_strike = S + otm_pts  # Same OTM for diagonal
            
            # Find long call contract
            long_ticker, long_price, long_exp_str, long_dte = find_best_contract(
                underlying=symbol,
                trade_date_str=trade_date_str,
                spot=S,
                target_strike=long_strike,
                target_dte_days=long_dte_weeks * 7,
                call_put='C',
                sigma=base_sigma,
                r=r,
                sigma_mult=sigma_mult,
            )
            
            if long_price <= 0 or not np.isfinite(long_price):
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue
            
            # Track if using BS fallback
            using_bs = (long_ticker is None)
            if using_bs:
                bs_fallback_count += 1
            else:
                massive_price_count += 1
            
            # Apply slippage (we're buying)
            long_px_adj = long_price * (1.0 + slippage_bps / 10_000.0)
            denom = long_px_adj * 100.0
            
            if denom <= 0.0:
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue
            
            qty_float = capital / denom
            if not np.isfinite(qty_float) or qty_float <= 0:
                equity.append(prev_eq)
                realized_weekly.append(0.0)
                unrealized_weekly.append(0.0)
                weekly_returns.append(0.0)
                continue
            
            qty = int(min(max(qty_float, 1), MAX_QTY))
            
            # Create long position
            long_pos = OptionPosition(
                symbol=symbol,
                quantity=qty,
                strike=long_strike,
                expiration_str=long_exp_str,
                dte_weeks=long_dte // 7,
                option_type="C",
                is_long=True,
                contract_ticker=long_ticker,
                using_bs_fallback=using_bs,
            )
            
            cost_long = qty * long_px_adj * 100.0 + fee * qty
            
            # Short leg for diagonal
            credit_short = 0.0
            if mode == "diagonal":
                # Find weekly short call
                short_ticker, short_price, short_exp_str, short_dte = find_best_contract(
                    underlying=symbol,
                    trade_date_str=trade_date_str,
                    spot=S,
                    target_strike=short_strike,
                    target_dte_days=7,  # Weekly
                    call_put='C',
                    sigma=base_sigma,
                    r=r,
                    sigma_mult=sigma_mult * 0.8,
                )
                
                if short_price > 0 and np.isfinite(short_price):
                    # Apply slippage (we're selling)
                    short_px_adj = short_price * (1.0 - slippage_bps / 10_000.0)
                    
                    short_pos = OptionPosition(
                        symbol=symbol,
                        quantity=-qty,
                        strike=short_strike,
                        expiration_str=short_exp_str,
                        dte_weeks=max(1, short_dte // 7),
                        option_type="C",
                        is_long=False,
                        contract_ticker=short_ticker,
                        using_bs_fallback=(short_ticker is None),
                    )
                    
                    credit_short = qty * short_px_adj * 100.0 - fee * qty
            
            net_cash = (-cost_long + credit_short) * realism
            eq_now = prev_eq + net_cash
            
            have_pos = True
            entry_week_idx = i
            entry_notional = long_px_adj * qty * 100.0
            entry_long_price = long_price
            
            equity.append(max(eq_now, LIQUIDATION_FLOOR))
            realized_weekly.append(net_cash)
            unrealized_weekly.append(0.0)
            weekly_returns.append((eq_now - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
            continue
        
        # ----------------------------------------------------------
        # POSITION MANAGEMENT
        # ----------------------------------------------------------
        if have_pos and long_pos is not None:
            # Decrement DTE
            long_pos.dte_weeks = max(0, long_pos.dte_weeks - 1)
            if short_pos is not None:
                short_pos.dte_weeks = max(0, short_pos.dte_weeks - 1)
            
            # Reprice long position
            if long_pos.contract_ticker and not long_pos.using_bs_fallback:
                # Try to get real price
                new_long_price = get_historical_price(
                    long_pos.contract_ticker,
                    trade_date_str
                )
                if new_long_price is None or new_long_price <= 0:
                    # Fallback to BS
                    T = max(long_pos.dte_weeks * 7, 1) / 365.0
                    new_long_price = bs_call_price(S, long_pos.strike, r, base_sigma * sigma_mult, T)
            else:
                # Use BS
                T = max(long_pos.dte_weeks * 7, 1) / 365.0
                new_long_price = bs_call_price(S, long_pos.strike, r, base_sigma * sigma_mult, T)
            
            long_val = max(new_long_price, 0.0) * 100.0 * long_pos.quantity
            
            # Reprice short position
            short_val = 0.0
            if short_pos is not None:
                if short_pos.contract_ticker and not short_pos.using_bs_fallback:
                    new_short_price = get_historical_price(
                        short_pos.contract_ticker,
                        trade_date_str
                    )
                    if new_short_price is None or new_short_price <= 0:
                        T = max(short_pos.dte_weeks * 7, 1) / 365.0
                        new_short_price = bs_call_price(S, short_pos.strike, r, base_sigma * sigma_mult * 0.8, T)
                else:
                    T = max(short_pos.dte_weeks * 7, 1) / 365.0
                    new_short_price = bs_call_price(S, short_pos.strike, r, base_sigma * sigma_mult * 0.8, T)
                
                short_val = max(new_short_price, 0.0) * 100.0 * short_pos.quantity  # qty is negative
            
            pos_val = long_val + short_val
            
            # Exit rules
            exit_trigger = False
            
            # Expiration
            if long_pos.dte_weeks <= 0:
                exit_trigger = True
            
            # Profit target / stop loss
            if entry_notional > 0:
                if long_val >= entry_notional * target_mult:
                    exit_trigger = True
                if long_val <= entry_notional * exit_mult:
                    exit_trigger = True
            
            # Short expiration - roll or close
            if short_pos is not None and short_pos.dte_weeks <= 0:
                # For simplicity, close everything when short expires
                # (In production, you'd roll the short)
                pass  # Could add rolling logic here
            
            if exit_trigger:
                # Apply slippage on exit
                exit_long_val = long_val * (1.0 - slippage_bps / 10_000.0)
                exit_short_val = short_val * (1.0 + slippage_bps / 10_000.0) if short_pos else 0.0
                
                payoff = (exit_long_val + exit_short_val) * realism
                eq2 = prev_eq + payoff
                
                dur = i - (entry_week_idx if entry_week_idx is not None else i)
                durations.append(dur)
                win_flags.append(payoff > 0)
                
                trade_log.append({
                    "entry_idx": entry_week_idx,
                    "exit_idx": i,
                    "entry_date": dates[entry_week_idx] if entry_week_idx is not None else None,
                    "exit_date": dates[i],
                    "entry_notional": entry_notional,
                    "exit_value": payoff,
                    "duration_weeks": dur,
                    "strike_long": long_pos.strike,
                    "strike_short": short_pos.strike if short_pos else None,
                    "long_ticker": long_pos.contract_ticker,
                    "short_ticker": short_pos.contract_ticker if short_pos else None,
                    "used_bs_fallback": long_pos.using_bs_fallback,
                })
                
                have_pos = False
                long_pos = None
                short_pos = None
                
                eq2_clamped = max(eq2, LIQUIDATION_FLOOR)
                equity.append(eq2_clamped)
                realized_weekly.append(payoff)
                unrealized_weekly.append(0.0)
                weekly_returns.append((eq2_clamped - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
                continue
            
            # Hold
            unreal_pnl = pos_val - entry_notional
            eq_hold = prev_eq + unreal_pnl
            eq_hold_clamped = max(eq_hold, LIQUIDATION_FLOOR)
            equity.append(eq_hold_clamped)
            realized_weekly.append(0.0)
            unrealized_weekly.append(unreal_pnl)
            weekly_returns.append((eq_hold_clamped - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
        else:
            # Flat
            equity.append(prev_eq)
            realized_weekly.append(0.0)
            unrealized_weekly.append(0.0)
            weekly_returns.append(0.0)
    
    # ----------------- Results -----------------
    equity_arr = np.asarray(equity, float)
    weekly_arr = np.asarray(weekly_returns, float)
    realized_arr = np.asarray(realized_weekly, float)
    unreal_arr = np.asarray(unrealized_weekly, float)
    
    trades = len(win_flags)
    win_rate = float(np.mean(win_flags)) if trades > 0 else 0.0
    avg_dur = float(np.mean(durations)) if durations else 0.0
    
    # Log data source usage
    if os.getenv("MASSIVE_DEBUG") == "1":
        print(f"[Massive Backtest] Massive prices: {massive_price_count}, BS fallback: {bs_fallback_count}")
    
    return {
        "equity": equity_arr,
        "weekly_returns": weekly_arr,
        "realized_weekly": realized_arr,
        "unrealized_weekly": unreal_arr,
        "trades": trades,
        "win_rate": win_rate,
        "avg_trade_dur": avg_dur,
        "trade_log": trade_log,
        "massive_price_count": massive_price_count,
        "bs_fallback_count": bs_fallback_count,
    }
