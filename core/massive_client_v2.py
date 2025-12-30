#!/usr/bin/env python3
"""
Massive API Client - Historical Option Chain Builder

Based on Massive support clarification (Dec 2024):
- Use OPTIONS CONTRACTS endpoint to get available contracts for a date
- Use AGGREGATES endpoint to get OHLCV pricing for each contract
- Build option chain manually from these two sources
- Data: Quotes back to 2022, Aggregates/Trades back to 2014
- No IV/Greeks for historical expired options (we'll compute IV ourselves)

Rate limits: Stay under 100 requests/second for optimal performance.
"""

from __future__ import annotations

import os
import time
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException


# ============================================================
# Configuration
# ============================================================

@dataclass
class MassiveConfig:
    """Configuration for Massive API"""
    api_key: str
    base_url: str = "https://api.massive.com"
    cache_dir: Path = Path.home() / ".cache" / "massive_options"
    requests_per_second: float = 50.0  # Stay well under 100/sec limit
    timeout: int = 30
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def get_api_key_from_keychain(service_name: str = "MASSIVE_API_KEY") -> Optional[str]:
    """
    Retrieve API key from macOS Keychain.
    
    The key should be stored as a "generic password" in Keychain.
    You can add it via:
        security add-generic-password -a "$USER" -s "MASSIVE_API_KEY" -w "your_api_key_here"
    
    Or via Keychain Access app:
        1. Open Keychain Access
        2. File > New Password Item
        3. Keychain Item Name: MASSIVE_API_KEY
        4. Account Name: your username
        5. Password: your API key
    """
    import subprocess
    import platform
    
    if platform.system() != "Darwin":
        return None
    
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s", service_name,
                "-w",  # Output only the password
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception:
        return None


def load_config() -> MassiveConfig:
    """
    Load config from macOS Keychain or environment variable.
    
    Priority:
    1. macOS Keychain (service name: MASSIVE_API_KEY)
    2. Environment variable MASSIVE_API_KEY
    """
    api_key = None
    
    # Try Keychain first (macOS)
    api_key = get_api_key_from_keychain("MASSIVE_API_KEY")
    
    if api_key:
        print("[Massive] API key loaded from macOS Keychain")
    else:
        # Fall back to environment variable
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if api_key:
            print("[Massive] API key loaded from environment variable")
    
    if not api_key:
        raise RuntimeError(
            "MASSIVE_API_KEY not found.\n\n"
            "Option 1 - macOS Keychain (recommended):\n"
            "  security add-generic-password -a \"$USER\" -s \"MASSIVE_API_KEY\" -w \"your_key\"\n\n"
            "Option 2 - Environment variable:\n"
            "  export MASSIVE_API_KEY='your_key_here'"
        )
    
    cache_dir = os.environ.get("MASSIVE_CACHE_DIR", "")
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        cache_path = Path.home() / ".cache" / "massive_options"
    
    return MassiveConfig(
        api_key=api_key,
        cache_dir=cache_path,
    )


# ============================================================
# Rate Limiter
# ============================================================

class RateLimiter:
    """Simple rate limiter to stay under Massive's recommended 100 req/sec"""
    
    def __init__(self, requests_per_second: float = 50.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    
    def wait(self):
        """Wait if necessary to maintain rate limit"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


# ============================================================
# API Client
# ============================================================

class MassiveClient:
    """
    Client for Massive API with caching and rate limiting.
    
    Usage:
        client = MassiveClient()
        chain = client.get_option_chain("UVXY", date(2023, 6, 15))
    """
    
    def __init__(self, config: Optional[MassiveConfig] = None):
        self.config = config or load_config()
        self.rate_limiter = RateLimiter(self.config.requests_per_second)
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with rate limiting and error handling"""
        self.rate_limiter.wait()
        
        # Add API key to params
        params["apiKey"] = self.config.api_key
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"[Massive] Request failed: {e}")
            return {"results": [], "error": str(e)}
    
    def _cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        key_str = f"{prefix}_{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cache_path(self, cache_key: str) -> Path:
        """Get path for cached data"""
        return self.config.cache_dir / f"{cache_key}.json"
    
    def _load_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if available"""
        path = self._cache_path(cache_key)
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _save_cache(self, cache_key: str, data: Dict[str, Any]):
        """Save data to cache"""
        path = self._cache_path(cache_key)
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Massive] Cache save failed: {e}")
    
    # ----------------------------------------------------------
    # Options Contracts Endpoint
    # ----------------------------------------------------------
    
    def get_contracts(
        self,
        underlying: str,
        expiration_date: Optional[date] = None,
        contract_type: Optional[str] = None,  # "call" or "put"
        expired: bool = True,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get available option contracts for an underlying.
        
        Parameters
        ----------
        underlying : str
            Underlying ticker (e.g., "UVXY", "VIX")
        expiration_date : date, optional
            Filter by expiration date
        contract_type : str, optional
            "call" or "put"
        expired : bool
            Include expired contracts (needed for historical)
        limit : int
            Max contracts to return per request
        
        Returns
        -------
        List of contract dicts with keys like:
            - ticker: option contract ticker (e.g., "O:UVXY230120C00015000")
            - underlying_ticker: "UVXY"
            - expiration_date: "2023-01-20"
            - strike_price: 15.0
            - contract_type: "call"
        """
        cache_key = self._cache_key(
            "contracts",
            underlying=underlying,
            expiration_date=str(expiration_date) if expiration_date else None,
            contract_type=contract_type,
        )
        
        cached = self._load_cache(cache_key)
        if cached:
            return cached.get("results", [])
        
        params = {
            "underlying_ticker": underlying,
            "expired": str(expired).lower(),
            "limit": limit,
            "order": "asc",
            "sort": "strike_price",
        }
        
        if expiration_date:
            params["expiration_date"] = expiration_date.isoformat()
        
        if contract_type:
            params["contract_type"] = contract_type.lower()
        
        all_results = []
        
        # Paginate to get all contracts
        while True:
            response = self._make_request(
                "/v3/reference/options/contracts",
                params,
            )
            
            results = response.get("results", [])
            all_results.extend(results)
            
            # Check for next page
            next_url = response.get("next_url")
            if not next_url or len(results) < limit:
                break
            
            # Extract cursor for next request
            # Massive uses cursor-based pagination
            if "cursor" in response:
                params["cursor"] = response["cursor"]
            else:
                break
        
        # Cache results
        self._save_cache(cache_key, {"results": all_results})
        
        return all_results
    
    # ----------------------------------------------------------
    # Aggregates Endpoint (OHLCV)
    # ----------------------------------------------------------
    
    def get_aggregates(
        self,
        ticker: str,
        trade_date: date,
        timespan: str = "day",
        multiplier: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get OHLCV aggregate data for an option contract.
        
        Parameters
        ----------
        ticker : str
            Option contract ticker (e.g., "O:UVXY230120C00015000")
        trade_date : date
            Date to get data for
        timespan : str
            "day", "hour", "minute"
        multiplier : int
            Multiplier for timespan
        
        Returns
        -------
        Dict with keys: open, high, low, close, volume, vwap
        Or None if no data available
        """
        cache_key = self._cache_key(
            "agg",
            ticker=ticker,
            date=str(trade_date),
        )
        
        cached = self._load_cache(cache_key)
        if cached:
            return cached.get("result")
        
        # Aggregates endpoint format: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
        from_date = trade_date.isoformat()
        to_date = trade_date.isoformat()
        
        response = self._make_request(
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            {},
        )
        
        results = response.get("results", [])
        
        if results:
            result = results[0]  # First (and likely only) bar for the day
            self._save_cache(cache_key, {"result": result})
            return result
        
        # Cache empty result too (to avoid re-querying)
        self._save_cache(cache_key, {"result": None})
        return None
    
    # ----------------------------------------------------------
    # Build Full Option Chain
    # ----------------------------------------------------------
    
    def get_option_chain(
        self,
        underlying: str,
        trade_date: date,
        expiration_range_days: int = 60,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Build a full option chain for a given date.
        
        This is the main method you'll use for backtesting.
        
        Parameters
        ----------
        underlying : str
            Underlying ticker (e.g., "UVXY", "VIX")
        trade_date : date
            Date to get chain for
        expiration_range_days : int
            Get expirations up to this many days out
        progress_callback : callable, optional
            Called with (current, total) for progress updates
        
        Returns
        -------
        DataFrame with columns:
            - trade_date
            - underlying
            - expiration
            - strike
            - option_type ("C" or "P")
            - open, high, low, close, volume
            - mid (computed as (open + close) / 2)
            - dte (days to expiration)
        """
        print(f"[Massive] Building chain for {underlying} on {trade_date}")
        
        # Step 1: Get all contracts expiring in our range
        min_exp = trade_date
        max_exp = trade_date + timedelta(days=expiration_range_days)
        
        all_contracts = []
        
        # Get contracts for each expiration Friday (VIX options expire on Wed, others on Fri)
        current_exp = min_exp
        while current_exp <= max_exp:
            # Try this date as expiration
            contracts = self.get_contracts(
                underlying=underlying,
                expiration_date=current_exp,
                expired=True,
            )
            all_contracts.extend(contracts)
            current_exp += timedelta(days=1)
        
        if not all_contracts:
            print(f"[Massive] No contracts found for {underlying} around {trade_date}")
            return pd.DataFrame()
        
        # Remove duplicates (by ticker)
        seen_tickers = set()
        unique_contracts = []
        for c in all_contracts:
            ticker = c.get("ticker")
            if ticker and ticker not in seen_tickers:
                seen_tickers.add(ticker)
                unique_contracts.append(c)
        
        print(f"[Massive] Found {len(unique_contracts)} unique contracts")
        
        # Step 2: Get pricing data for each contract
        rows = []
        total = len(unique_contracts)
        
        for idx, contract in enumerate(unique_contracts):
            if progress_callback:
                progress_callback(idx + 1, total)
            
            ticker = contract.get("ticker")
            if not ticker:
                continue
            
            # Get aggregate data for trade_date
            agg = self.get_aggregates(ticker, trade_date)
            
            if agg is None:
                # No trading data for this contract on this date
                continue
            
            # Parse contract info
            strike = contract.get("strike_price", 0.0)
            exp_str = contract.get("expiration_date", "")
            contract_type = contract.get("contract_type", "").upper()
            
            if contract_type == "CALL":
                opt_type = "C"
            elif contract_type == "PUT":
                opt_type = "P"
            else:
                opt_type = contract_type[0] if contract_type else "?"
            
            # Parse expiration
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except Exception:
                continue
            
            # Calculate DTE
            dte = (exp_date - trade_date).days
            
            # Extract OHLCV
            o = float(agg.get("o", agg.get("open", 0.0)))
            h = float(agg.get("h", agg.get("high", 0.0)))
            l = float(agg.get("l", agg.get("low", 0.0)))
            c = float(agg.get("c", agg.get("close", 0.0)))
            v = float(agg.get("v", agg.get("volume", 0.0)))
            vwap = float(agg.get("vw", agg.get("vwap", 0.0)))
            
            # Compute mid (use VWAP if available, else avg of open/close)
            if vwap > 0:
                mid = vwap
            elif o > 0 and c > 0:
                mid = (o + c) / 2.0
            else:
                mid = c if c > 0 else o
            
            rows.append({
                "trade_date": trade_date,
                "underlying": underlying,
                "ticker": ticker,
                "expiration": exp_date,
                "strike": strike,
                "option_type": opt_type,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "vwap": vwap,
                "mid": mid,
                "dte": dte,
            })
        
        if not rows:
            print(f"[Massive] No pricing data found for {trade_date}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df = df.sort_values(["expiration", "strike", "option_type"])
        df = df.reset_index(drop=True)
        
        print(f"[Massive] Built chain with {len(df)} rows")
        return df
    
    # ----------------------------------------------------------
    # Convenience: Get chain for backtesting
    # ----------------------------------------------------------
    
    def get_chain_for_backtest(
        self,
        underlying: str,
        trade_date: date,
        min_dte: int = 1,
        max_dte: int = 60,
    ) -> pd.DataFrame:
        """
        Get option chain filtered for backtesting needs.
        
        Filters out:
        - Contracts with no volume
        - Contracts outside DTE range
        - Invalid prices
        """
        chain = self.get_option_chain(
            underlying=underlying,
            trade_date=trade_date,
            expiration_range_days=max_dte + 7,
        )
        
        if chain.empty:
            return chain
        
        # Filter
        chain = chain[
            (chain["dte"] >= min_dte) &
            (chain["dte"] <= max_dte) &
            (chain["mid"] > 0) &
            (chain["volume"] > 0)
        ].copy()
        
        return chain


# ============================================================
# IV Calculation (since Massive doesn't provide it)
# ============================================================

def compute_iv_newton(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.03,
    option_type: str = "C",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Compute implied volatility using Newton-Raphson method.
    
    Since Massive doesn't provide IV for historical options,
    we compute it ourselves from the option price.
    """
    from math import log, sqrt, exp
    from scipy.stats import norm
    
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return np.nan
    
    # Initial guess based on ATM approximation
    sigma = sqrt(2 * abs(log(S / K) + r * T) / T) if T > 0 else 0.3
    sigma = max(0.01, min(sigma, 5.0))
    
    for _ in range(max_iter):
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if option_type.upper() == "C":
            price_est = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:
            price_est = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        diff = price_est - price
        
        if abs(diff) < tol:
            return sigma
        
        # Vega
        vega = S * sqrt(T) * norm.pdf(d1)
        
        if vega < 1e-10:
            break
        
        sigma = sigma - diff / vega
        sigma = max(0.01, min(sigma, 5.0))
    
    return sigma if 0.01 < sigma < 5.0 else np.nan


def add_iv_to_chain(
    chain: pd.DataFrame,
    underlying_price: float,
    r: float = 0.03,
) -> pd.DataFrame:
    """Add implied volatility column to option chain"""
    
    if chain.empty:
        return chain
    
    chain = chain.copy()
    
    ivs = []
    for _, row in chain.iterrows():
        iv = compute_iv_newton(
            price=row["mid"],
            S=underlying_price,
            K=row["strike"],
            T=row["dte"] / 365.0,
            r=r,
            option_type=row["option_type"],
        )
        ivs.append(iv)
    
    chain["iv"] = ivs
    return chain


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Example: Get UVXY option chain for a specific date
    
    print("Massive API Client - Example")
    print("=" * 50)
    
    try:
        client = MassiveClient()
        
        # Get chain for a historical date
        test_date = date(2023, 6, 15)
        
        print(f"\nFetching option chain for UVXY on {test_date}...")
        
        chain = client.get_chain_for_backtest(
            underlying="UVXY",
            trade_date=test_date,
            min_dte=1,
            max_dte=30,
        )
        
        if not chain.empty:
            print(f"\nSuccess! Got {len(chain)} contracts")
            print("\nSample data:")
            print(chain.head(10).to_string())
            
            # Add IV
            print("\nComputing implied volatilities...")
            # You'd need to get the underlying price for this date
            # chain = add_iv_to_chain(chain, underlying_price=15.0)
        else:
            print("No data returned")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure MASSIVE_API_KEY is set in your environment")