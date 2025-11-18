"""
utils.py — lean econ data utils (Yahoo prices + FRED; no options)

Fetch:
  fetch_yahoo_price(ticker, start=None, end=None, interval="1d", price_field="Adj Close", prepost=False)
  fetch_fred(series_id, start="1900-01-01", end="2100-01-01")

Store (flat, metadata-driven):
  save_data(df, meta, write_meta_files=True) -> Path
  load_data(dsid) -> DataFrame
  get_meta(dsid) -> dict
  list_datasets() -> list[str]
  find_datasets(**filters) -> list[dict]

Catalog & Sync:
  sync_catalog_from_disk(write_meta_files=True) -> int
  update_series(dsid) -> bool
  reload_all(progress: Optional[Callable[[int,int,str,bool],None]] = None) -> dict

Layout under DATA_DIR/econ:
  data/<id>.csv
  meta/<id>.json
  catalog.json

<id> = "{source}.{symbol}"  e.g., "yahoo.SPY", "fred.CPIAUCSL"
"""

from __future__ import annotations
import os, json, hashlib, math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Callable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# ---------------- Env & paths ----------------
load_dotenv(override=False)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

ROOT = DATA_DIR / "econ"
DATA_DIR_FLAT = ROOT / "data"
META_DIR = ROOT / "meta"
CATALOG = ROOT / "catalog.json"
for p in (DATA_DIR_FLAT, META_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- JSON sanitizers -------------
def _to_py(o):
    if o is None:
        return None
    if isinstance(o, (np.generic,)):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o

def _sanitize_meta(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _sanitize_meta(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_to_py(x) for x in v]
        else:
            out[k] = _to_py(v)
    return out

# ---------------- Metadata -------------------
@dataclass
class DatasetMeta:
    source: str               # 'yahoo' | 'fred' | 'derived'
    symbol: str               # ticker or FRED id
    # semantics
    kind: str = "series"      # time series
    frequency: Optional[str] = None
    units: Optional[str] = None
    adjusted: Optional[bool] = None
    # auto/populated
    id: Optional[str] = None
    rows: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None
    sha256: Optional[str] = None
    fetched_at: str = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    version: int = 1
    extra: Optional[Dict[str, Any]] = None

    def make_id(self) -> str:
        return f"{self.source}.{self.symbol}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["id"] = d["id"] or self.make_id()
        return d

# --------------- Catalog helpers -------------
def _read_catalog() -> Dict[str, Any]:
    return json.loads(CATALOG.read_text()) if CATALOG.exists() else {"datasets": {}}

def _write_catalog(cat: Dict[str, Any]) -> None:
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CATALOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(cat, indent=2))
    tmp.replace(CATALOG)

def list_datasets() -> List[str]:
    return sorted(_read_catalog()["datasets"].keys())

def find_datasets(**filters) -> List[Dict[str, Any]]:
    cat = _read_catalog()
    out: List[Dict[str, Any]] = []
    for dsid, rec in cat["datasets"].items():
        meta = rec.get("meta", {})
        if all(meta.get(k) == v for k, v in filters.items()):
            out.append(rec)
    return out

def get_meta(dsid: str) -> Dict[str, Any]:
    rec = _read_catalog()["datasets"].get(dsid)
    if not rec:
        raise KeyError(f"{dsid!r} not found")
    return rec["meta"]

# --------------- Storage core ----------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _normalize_series_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out

def save_data(df: pd.DataFrame, meta: Dict[str, Any], *, write_meta_files: bool = True) -> Path:
    """
    Save DataFrame; update catalog.
    meta: {'source','symbol', optional: 'kind','frequency','units','adjusted','extra'}
    For time series, index should be DatetimeIndex (we normalize).
    """
    if "source" not in meta or "symbol" not in meta:
        raise ValueError("meta must include 'source' and 'symbol'")

    keep = {k: v for k, v in meta.items() if k in DatasetMeta.__annotations__}
    m = DatasetMeta(**keep)
    dsid = m.make_id()

    df_out = _normalize_series_index(df) if isinstance(df.index, pd.DatetimeIndex) else df
    # Ensure string column names to avoid writer issues
    df_out.columns = [str(c) for c in df_out.columns]
    csv_path = DATA_DIR_FLAT / f"{dsid}.csv"
    if isinstance(df_out.index, pd.DatetimeIndex):
        df_out.to_csv(csv_path, index_label="date")
    else:
        df_out.to_csv(csv_path)

    m.id = dsid
    m.rows = len(df_out)
    if isinstance(df_out.index, pd.DatetimeIndex) and len(df_out):
        m.start = str(df_out.index.min().date())
        m.end = str(df_out.index.max().date())
    m.sha256 = _sha256_file(csv_path)
    md = _sanitize_meta(m.to_dict())

    if write_meta_files:
        (META_DIR / f"{dsid}.json").write_text(json.dumps(md, indent=2))

    cat = _read_catalog()
    cat["datasets"][dsid] = {"path": str(csv_path), "meta": md}
    _write_catalog(cat)
    return csv_path

def load_data(dsid: str) -> pd.DataFrame:
    rec = _read_catalog()["datasets"].get(dsid)
    if not rec:
        raise FileNotFoundError(f"{dsid} not in catalog")
    path = Path(rec["path"])
    try:
        return pd.read_csv(path, parse_dates=["date"], index_col="date")
    except Exception:
        return pd.read_csv(path, index_col=0)

# --------------- Fetchers --------------------
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

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

def _fred_get(params: Dict[str, Any]) -> Dict[str, Any]:
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not set")
    p = dict(params)
    p.update({"api_key": FRED_API_KEY, "file_type": "json"})
    r = requests.get(FRED_OBS_URL, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_fred(series_id: str, start: str = "1900-01-01", end: str = "2100-01-01") -> pd.DataFrame:
    data = _fred_get({"series_id": series_id, "observation_start": start, "observation_end": end})
    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=[series_id])
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return pd.to_numeric(df["value"], errors="coerce").to_frame(series_id).dropna()

# ---------- Catalog bootstrap from disk ----------
def sync_catalog_from_disk(write_meta_files: bool = True) -> int:
    """
    Scan DATA_DIR/econ/data/*.csv and ensure each file is present in catalog.json.
    Returns count of newly-added entries.
    """
    cat = _read_catalog()
    existing = set(cat["datasets"].keys())
    added = 0
    for csv_path in DATA_DIR_FLAT.glob("*.csv"):
        dsid = csv_path.stem
        if dsid in existing:
            continue
        try:
            try_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            kind = "series"
        except Exception:
            try_df = pd.read_csv(csv_path, index_col=0)
            kind = "table"
        rows = len(try_df)
        start = end = None
        if kind == "series" and rows:
            start = str(pd.to_datetime(try_df.index.min()).date())
            end = str(pd.to_datetime(try_df.index.max()).date())
        parts = dsid.split(".", 1)
        source = parts[0]
        symbol = parts[1] if len(parts) > 1 else dsid
        meta = DatasetMeta(
            source=source, symbol=symbol, kind=kind,
            id=dsid, rows=rows, start=start, end=end,
            sha256=_sha256_file(csv_path)
        ).to_dict()
        meta = _sanitize_meta(meta)
        if write_meta_files:
            (META_DIR / f"{dsid}.json").write_text(json.dumps(meta, indent=2))
        cat["datasets"][dsid] = {"path": str(csv_path), "meta": meta}
        added += 1
    _write_catalog(cat)
    return added

# ---------- Incremental update helpers ----------
def _combine_series(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        return new_df.sort_index()
    out = pd.concat([old_df, new_df])
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()

def update_series(dsid: str) -> bool:
    """
    Incrementally update one dataset to 'today' if it's a fetchable series (yahoo/fred).
    """
    cat = _read_catalog()
    rec = cat["datasets"].get(dsid)
    if not rec:
        return False
    meta = rec["meta"]
    source = meta.get("source")
    symbol = meta.get("symbol")
    kind = meta.get("kind", "series")
    if kind != "series" or source not in ("yahoo", "fred"):
        return False

    df_old = load_data(dsid)
    last_date = None
    if isinstance(df_old.index, pd.DatetimeIndex) and len(df_old):
        last_date = pd.to_datetime(df_old.index.max()).normalize()
    start = None if last_date is None else (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    if source == "yahoo":
        df_new = fetch_yahoo_price(symbol, start=start, end=end)
    else:
        df_new = fetch_fred(symbol, start=start or "1900-01-01", end=end)

    if df_new is None or df_new.empty:
        return False

    df_combined = _combine_series(df_old, df_new)
    meta_base = {
        "source": source,
        "symbol": symbol,
        "kind": "series",
        "frequency": meta.get("frequency"),
        "units": meta.get("units"),
        "adjusted": meta.get("adjusted"),
        "extra": meta.get("extra"),
    }
    save_data(df_combined, meta_base)
    return True

def reload_all(progress: Optional[Callable[[int, int, str, bool], None]] = None) -> dict:
    """
    Update all fetchable series (yahoo/fred) to today.
    """
    cat = _read_catalog()
    dsids = sorted(cat["datasets"].keys())
    n = len(dsids)
    updated = 0
    skipped = 0
    for i, dsid in enumerate(dsids, 1):
        rec = cat["datasets"][dsid]["meta"]
        if rec.get("kind", "series") != "series" or rec.get("source") not in ("yahoo", "fred"):
            skipped += 1
            if progress:
                progress(i, n, dsid, False)
            continue
        ok = update_series(dsid)
        if ok:
            updated += 1
        else:
            skipped += 1
        if progress:
            progress(i, n, dsid, ok)
    return {"total": n, "updated": updated, "skipped": skipped}
