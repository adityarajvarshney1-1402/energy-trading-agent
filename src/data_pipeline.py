import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# EIA v2 open-data API. The hourly RTO series exposes strong daily/weekly
# seasonality, which is what the Prophet/LSTM pipeline expects. Free API key:
# https://www.eia.gov/opendata/register.php
EIA_API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

def generate_mock_energy_data(start_date: str = "2023-01-01", days: int = 365) -> pd.DataFrame:
    """
    Generates mock hourly energy price data for testing the pipeline without an API key.
    Includes daily seasonality, weekly seasonality, and random noise.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start + timedelta(hours=i) for i in range(days * 24)]
    
    # Base price
    base_price = 50.0
    
    # Trend component (slight upward trend)
    trend = np.linspace(0, 20, len(dates))
    
    # Daily seasonality (higher during day, lower at night)
    daily_seasonality = np.array([10 * np.sin(2 * np.pi * (d.hour - 6) / 24) for d in dates])
    
    # Weekly seasonality (lower on weekends)
    weekly_seasonality = np.array([-15 if d.weekday() >= 5 else 5 for d in dates])
    
    # Random noise
    noise = np.random.normal(0, 5, len(dates))
    
    # Calculate final prices
    prices = base_price + trend + daily_seasonality + weekly_seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices
    })
    
    return df

def fetch_eia_data(respondent: str = "PJM", hours: int = 24 * 90,
                   api_key: str | None = None) -> pd.DataFrame:
    """
    Fetches recent hourly electricity demand (MWh) from the EIA v2 API for a
    given grid operator/region and returns it as a ``timestamp``/``price`` frame
    (the value column is named ``price`` for pipeline compatibility).

    Requires an API key, read from the ``EIA_API_KEY`` env var if not passed.
    Raises on missing key, network error, or empty response so callers can fall
    back to mock data.
    """
    import requests

    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY is not set. Register free at "
                         "https://www.eia.gov/opendata/register.php")

    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": "D",            # D = demand
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": int(hours),
    }
    resp = requests.get(EIA_API_URL, params=params, timeout=30)
    resp.raise_for_status()

    rows = resp.json().get("response", {}).get("data", [])
    if not rows:
        raise ValueError(f"EIA returned no data for respondent={respondent!r}.")

    df = pd.DataFrame(rows)
    df = df[["period", "value"]].rename(columns={"period": "timestamp", "value": "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_yfinance_data(ticker: str = "NG=F", period: str = "90d",
                        interval: str = "60m") -> pd.DataFrame:
    """
    Fetches real market prices from Yahoo Finance via ``yfinance`` and returns a
    ``timestamp``/``price`` frame (the closing price is used as ``price``).

    No API key required. Useful tickers for energy trading:
      NG=F  Natural Gas futures      CL=F  WTI Crude Oil futures
      BZ=F  Brent Crude futures      XLE   Energy Select Sector ETF
      UNG   Natural Gas ETF          XOM   ExxonMobil stock

    Note: commodity markets close overnight/weekends, so the hourly series has
    gaps (Prophet handles this fine). Raises on empty response so callers can
    fall back to mock data.
    """
    import yfinance as yf

    data = yf.download(ticker, period=period, interval=interval,
                       progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"yfinance returned no data for ticker={ticker!r}.")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):       # MultiIndex columns for some tickers
        close = close.iloc[:, 0]

    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = pd.DataFrame({
        "timestamp": idx,
        "price": pd.to_numeric(close.values, errors="coerce"),
    })
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"yfinance data for {ticker!r} had no usable price points.")
    return df


def fetch_and_save_data(filename: str = "historical_prices.csv", source: str = "mock",
                        respondent: str = "PJM", ticker: str = "NG=F"):
    """
    Fetches data and saves it to the data directory.

    source="mock"     -> synthetic generator (no key needed)
    source="eia"      -> live EIA hourly data          (requires EIA_API_KEY)
    source="yfinance" -> real market prices (gas/oil/ETFs) via Yahoo Finance (no key)

    Note: "live" is kept as an alias for "eia" for backward compatibility.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if source in ("eia", "live"):
        df = fetch_eia_data(respondent=respondent)
    elif source == "yfinance":
        df = fetch_yfinance_data(ticker=ticker)
    else:
        df = generate_mock_energy_data()
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    return filepath

def load_data(filename: str = "historical_prices.csv") -> pd.DataFrame:
    """
    Loads historical data from the data directory.
    """
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        filepath = fetch_and_save_data(filename)
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

if __name__ == "__main__":
    fetch_and_save_data()
    print("Mock data generated successfully.")
