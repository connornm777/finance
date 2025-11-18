from __future__ import annotations
import os, sys, json, hashlib, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')


me/connor/Dropbox/Data/finance



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


