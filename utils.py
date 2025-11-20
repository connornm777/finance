from __future__ import annotations
import os, sys, json, hashlib, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')
FRED_API_KEY = os.getenv('FRED_API_KEY')

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_yahoo_options(ticker: str, expiration: Optional[str] = None) -> pd.DataFrame:
    t = yf.Ticker(str(ticker).strip())
    exps = list(t.options or [])
    if not exps: return pd.DataFrame()
    if expiration is None:
        expiration = exps[0]
    elif expiration not in exps:
        raise ValueError(f"expiration {expiration!r} not in {exps}")
    chain = t.option_chain(expiration)
    calls = chain.calls.copy(); calls["type"] = "call"
    puts  = chain.puts.copy();  puts["type"]  = "put"
    df = pd.concat([calls, puts], axis=0, ignore_index=True)
    df["expiration"] = expiration
    if "contractSymbol" in df.columns:
        df.set_index("contractSymbol", inplace=True)
    return df

def fetch_yahoo_price(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    interval: Literal["1d", "1wk", "1mo"] = "1d",
    price_field: Literal["Adj Close", "Close", "Open"] = "Adj Close",
    prepost: bool = False,
) -> pd.DataFrame:
    tkr = str(ticker).strip()
    df = yf.download(
        tkr, start=start, end=end, interval=interval,
        auto_adjust=False, progress=False, prepost=prepost,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=[tkr])
    if price_field not in df.columns:
        price_field = "Adj Close" if "Adj Close" in df.columns else "Close"
    out = df[[price_field]].rename(columns={price_field: tkr})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.columns = [str(c) for c in out.columns]
    return out


def fetch_fred(series_id: str, start: str = "1900-01-01", end: str = "2100-01-01") -> pd.DataFrame:
    params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type":"json",
            "observation_start": start,
            "observation_end": end,
            }

    r = requests.get(FRED_OBS_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()


    p = dict(params)
    p.update({"api_key": FRED_API_KEY, "file_type": "json"})
    r = requests.get(FRED_OBS_URL, params=p, timeout=30)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=[series_id])
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return pd.to_numeric(df["value"], errors="coerce").to_frame(series_id).dropna()


def save_csv(
    df: pd.DataFrame,
    name: str,
    *,
    index: bool = True,
    **to_csv_kwargs: Any,
) -> Path:
    """
    Save a DataFrame as CSV under DATA_DIR.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    name : str
        Logical name or filename. '.csv' is appended if missing.
    index : bool
        Whether to write row index.
    **to_csv_kwargs
        Extra arguments forwarded to DataFrame.to_csv().

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    if not DATA_DIR:
        raise RuntimeError("DATA_DIR environment variable is not set.")

    data_dir = Path(DATA_DIR).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    fname = name if name.lower().endswith(".csv") else f"{name}.csv"
    path = data_dir / fname

    df.to_csv(path, index=index, **to_csv_kwargs)
    return path


def load_csv(
    name: str,
    *,
    index_col: Optional[int | str] = None,
    parse_dates: Optional[List[int | str]] = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV from DATA_DIR into a DataFrame.

    Parameters
    ----------
    name : str
        Logical name or filename. '.csv' is appended if missing.
    index_col : int or str, optional
        Column to use as index.
    parse_dates : list, optional
        Columns to parse as dates.
    **read_csv_kwargs
        Extra arguments forwarded to pd.read_csv().

    Returns
    -------
    pd.DataFrame
    """
    if not DATA_DIR:
        raise RuntimeError("DATA_DIR environment variable is not set.")

    data_dir = Path(DATA_DIR).expanduser().resolve()
    fname = name if name.lower().endswith(".csv") else f"{name}.csv"
    path = data_dir / fname

    if not path.exists():
        raise FileNotFoundError(f"No CSV found at {path}")

    return pd.read_csv(
        path,
        index_col=index_col,
        parse_dates=parse_dates,
        **read_csv_kwargs,
    )

