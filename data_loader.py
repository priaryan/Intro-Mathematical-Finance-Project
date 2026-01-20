"""
Data loading utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class MarketSeries:
    ticker: str
    close: pd.Series


def fetch_daily_close(
    ticker: str,
    start_date,
    end_date,
) -> MarketSeries:
    """
    Fetch daily close prices using yfinance.

    Parameters
    ticker: strn, start_date, end_date

    Returns
    MarketSeries with a pandas Series of close prices indexed by date.
    """
    import yfinance as yf

    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError("No data returned. Check ticker and date range.")

    if "Close" not in df.columns:
        raise ValueError("Expected Close column not found in yfinance output.")

    close = df["Close"].dropna().copy()
    close.name = "Close"
    return MarketSeries(ticker=ticker, close=close)
