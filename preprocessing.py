"""
Preprocessing steps.

Compute daily log returns and estimate annualized volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series) -> pd.Series:
    """
    Daily log returns from close prices.

    r_t = log(S_t / S_{t-1})
    """
    if close is None or close.empty:
        raise ValueError("Close series is empty.")

    rets = np.log(close / close.shift(1)).dropna()
    rets.name = "log_return"
    return rets


def estimate_annualized_volatility(log_returns: pd.Series, trading_days: int = 252) -> float:
    """
    Annualized volatility estimate.
    sigma = std(daily_returns) * sqrt(trading_days)
    """
    if log_returns is None or log_returns.empty:
        raise ValueError("Log returns series is empty.")

    daily_std = float(log_returns.std(ddof=1))
    return daily_std * float(np.sqrt(trading_days))


