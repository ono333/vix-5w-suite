#!/usr/bin/env python3
"""
Data loading helpers for VIX 5% Weekly Suite.

We expose a generic `load_weekly(symbol, start, end)` and a small wrapper
`load_vix_weekly` for backwards compatibility.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _to_date(d) -> dt.date:
    if isinstance(d, dt.date):
        return d
    return pd.to_datetime(d).date()


def load_weekly(
    symbol: str,
    start_date,
    end_date,
    *,
    column: str = "Adj Close",
) -> pd.Series:
    """
    Load weekly data for `symbol` using yfinance and return a weekly
    close series between start_date and end_date (inclusive).

    - Resamples to weekly (Friday) using last available close.
    - Returns an empty Series on any hard failure.
    """
    try:
        s = _to_date(start_date)
        e = _to_date(end_date)
    except Exception:
        s = _to_date("2004-01-01")
        e = _to_date(dt.date.today())

    try:
        df = yf.download(
            symbol,
            start=s,
            end=e + dt.timedelta(days=3),
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    col = column if column in df.columns else "Close"
    if col not in df.columns:
        return pd.Series(dtype=float)

    ser = df[col].copy()

    # Resample to weekly with last observed value
    weekly = ser.resample("W-FRI").last().dropna()
    weekly.name = symbol
    return weekly


def load_vix_weekly(start_date, end_date) -> pd.Series:
    """Convenience wrapper that loads ^VIX weekly data."""
    return load_weekly("^VIX", start_date, end_date)