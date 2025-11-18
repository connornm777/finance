#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
options_gui.py — Yahoo options toolkit with storage, plotting, portfolio, and simulation.

Storage (flat):
  DATA_DIR/options/
    data/<id>.csv         # id = yahoo.<TICKER>.options.<YYYY-MM-DD>
    meta/<id>.json
    catalog.json

Features
- Fetch ALL expirations by default (LEAPS included) or a single expiry. Dedup/upsert by CSV hash.
- View table(s); flexible plots: Scatter / Line / Heatmap with any X vs Y (+Value for heatmap).
- Portfolio dock: add/remove/clear legs, quick strategies (Straddle, Strangle, Call/Put Vertical).
- Simulator window on the portfolio:
    * Lognormal or Custom sampler (expr → np.ndarray), S0, μ, σ, T, N, multiplier.
    * Live update option: updates P&L histogram and stats as knobs move.
"""

from __future__ import annotations
import os, sys, json, hashlib, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# ---------- env & storage ----------
load_dotenv(override=False)
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
ROOT = DATA_DIR / "options"
DATA_DIR_FLAT = ROOT / "data"
META_DIR = ROOT / "meta"
CATALOG = ROOT / "catalog.json"
for p in (DATA_DIR_FLAT, META_DIR):
    p.mkdir(parents=True, exist_ok=True)

def _read_catalog() -> Dict[str, Any]:
    return json.loads(CATALOG.read_text()) if CATALOG.exists() else {"datasets": {}}

def _write_catalog(cat: Dict[str, Any]) -> None:
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CATALOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(cat, indent=2))
    tmp.replace(CATALOG)

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _df_hash_csv(df: pd.DataFrame) -> str:
    return _sha256_bytes(df.to_csv(index=True).encode("utf-8"))

def list_datasets() -> List[str]:
    return sorted(_read_catalog()["datasets"].keys())

def _dsid(source: str, symbol: str, tag: str) -> str:
    return f"{source}.{symbol}.{tag}"  # e.g., yahoo.SPY.options.2025-12-19

def save_table_upsert(df: pd.DataFrame, *, source: str, symbol: str, tag: str,
                      meta_overrides: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    dsid = _dsid(source, symbol, tag)
    csv_path = DATA_DIR_FLAT / f"{dsid}.csv"
    new_sha = _df_hash_csv(df)
    cat = _read_catalog()
    existing = cat["datasets"].get(dsid)

    if existing:
        old_meta = existing.get("meta", {})
        old_sha = old_meta.get("sha256")
        if old_sha == new_sha and Path(existing["path"]).exists():
            return {"saved": False, "path": existing["path"], "meta": old_meta, "reason": "identical"}
        # upsert
        df.to_csv(csv_path)
        rows = len(df)
        meta = {
            "id": dsid, "source": source, "symbol": symbol, "tag": tag,
            "kind": "options", "rows": rows, "sha256": new_sha,
            "fetched_at": pd.Timestamp.now(tz="UTC").isoformat()
        }
        if meta_overrides: meta.update(meta_overrides)
        (META_DIR / f"{dsid}.json").write_text(json.dumps(meta, indent=2))
        cat["datasets"][dsid] = {"path": str(csv_path), "meta": meta}
        _write_catalog(cat)
        return {"saved": True, "path": str(csv_path), "meta": meta, "reason": "updated"}

    # new
    df.to_csv(csv_path)
    rows = len(df)
    meta = {
        "id": dsid, "source": source, "symbol": symbol, "tag": tag,
        "kind": "options", "rows": rows, "sha256": new_sha,
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat()
    }
    if meta_overrides: meta.update(meta_overrides)
    (META_DIR / f"{dsid}.json").write_text(json.dumps(meta, indent=2))
    cat["datasets"][dsid] = {"path": str(csv_path), "meta": meta}
    _write_catalog(cat)
    return {"saved": True, "path": str(csv_path), "meta": meta, "reason": "new"}

def load_table_by_id(dsid: str) -> pd.DataFrame:
    rec = _read_catalog()["datasets"].get(dsid)
    if not rec:
        raise FileNotFoundError(f"{dsid} not in catalog")
    df = pd.read_csv(rec["path"], index_col=0)
    df["__dsid__"] = dsid
    if "expiration" not in df.columns:
        parts = dsid.split(".")
        if len(parts) >= 4 and parts[-2] == "options":
            df["expiration"] = parts[-1]
    return df

def load_many(dsids: List[str]) -> pd.DataFrame:
    frames = []
    for dsid in dsids:
        try:
            frames.append(load_table_by_id(dsid))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)

# ---------- fetch helpers ----------
def list_yahoo_option_expirations(ticker: str) -> List[str]:
    t = yf.Ticker(str(ticker).strip())
    return list(t.options or [])

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

def fetch_and_upsert_yahoo_chain(ticker: str, expiration: Optional[str]) -> Dict[str, Any]:
    df = fetch_yahoo_options(ticker, expiration)
    if df.empty: raise RuntimeError("No options returned")
    exp = str(df["expiration"].iloc[0])
    tag = f"options.{exp}"
    return save_table_upsert(df, source="yahoo", symbol=ticker, tag=tag,
                             meta_overrides={"expiration": exp, "ticker": ticker})

def fetch_all_and_upsert_yahoo_chains(ticker: str, *, limit: Optional[int] = None, sleep_sec: float = 0.0) -> Dict[str, Any]:
    """Fetch ALL currently listed expirations and upsert each. Returns a summary."""
    tkr = str(ticker).strip()
    t = yf.Ticker(tkr)
    exps = list(t.options or [])
    if not exps:
        return {"ticker": tkr, "total": 0, "saved": 0, "updated": 0, "identical": 0, "items": []}
    if limit is not None:
        exps = exps[:int(limit)]
    out_items, saved, updated, identical = [], 0, 0, 0
    for exp in exps:
        chain = t.option_chain(exp)
        calls = chain.calls.copy(); calls["type"] = "call"
        puts  = chain.puts.copy();  puts["type"]  = "put"
        df = pd.concat([calls, puts], axis=0, ignore_index=True)
        df["expiration"] = exp
        if "contractSymbol" in df.columns:
            df.set_index("contractSymbol", inplace=True)
        res = save_table_upsert(
            df, source="yahoo", symbol=tkr, tag=f"options.{exp}",
            meta_overrides={"expiration": exp, "ticker": tkr}
        )
        out_items.append(res)
        if res["reason"] == "identical": identical += 1
        elif res["reason"] == "updated": updated += 1
        else: saved += 1
        if sleep_sec > 0:
            import time; time.sleep(sleep_sec)
    return {"ticker": tkr, "total": len(exps), "saved": saved, "updated": updated,
            "identical": identical, "items": out_items}

# ---------- derived columns ----------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "expiration" in out.columns:
        exp_dt = pd.to_datetime(out["expiration"], errors="coerce", utc=True)
        out["_exp_dt"] = exp_dt
        now = pd.Timestamp.now(tz="UTC")
        out["T"] = (exp_dt - now).dt.total_seconds() / (365.0 * 24 * 3600)
        out["days_to_exp"] = (exp_dt - now).dt.total_seconds() / (24 * 3600)
    else:
        out["_exp_dt"] = pd.NaT; out["T"] = np.nan; out["days_to_exp"] = np.nan
    if "mid" not in out.columns:
        if "bid" in out.columns and "ask" in out.columns:
            out["mid"] = (out["bid"].fillna(0) + out["ask"].fillna(0)) / 2.0
        elif "lastPrice" in out.columns:
            out["mid"] = out["lastPrice"]
    return out

# ---------- GUI bits ----------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig); self.setParent(parent); self.fig.tight_layout()
    def clear(self):
        self.fig.clf(); self.ax = self.fig.add_subplot(111); self.draw_idle()

def enable_dark(app: QtWidgets.QApplication):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(pal)

class FetchOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("Fetch — Yahoo Options")
        self.ticker = QtWidgets.QLineEdit()
        self.exps = QtWidgets.QComboBox()
        self.btn = QtWidgets.QPushButton("Load Expirations"); self.btn.clicked.connect(self.load_exps)
        form = QtWidgets.QFormLayout()
        form.addRow("Ticker:", self.ticker)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.exps); row.addWidget(self.btn)
        form.addRow("Expiration:", row)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(form); lay.addWidget(box)

    def load_exps(self):
        t = self.ticker.text().strip()
        if not t:
            QtWidgets.QMessageBox.warning(self, "Ticker", "Enter a ticker."); return
        try:
            exps = list_yahoo_option_expirations(t)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e)); return
        self.exps.clear()
        self.exps.addItem("ALL (every listed)")  # default = ALL
        self.exps.addItems(exps or [])
        self.exps.setCurrentIndex(0)

    def params(self)->Dict[str,Any]:
        choice = self.exps.currentText().strip()
        expiration = None if (not choice or choice.startswith("ALL")) else choice
        return dict(ticker=self.ticker.text().strip(), expiration=expiration)

class PlotControls(QtWidgets.QGroupBox):
    request_plot = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__("Plot Controls", parent)
        grid = QtWidgets.QGridLayout(self)
        self.cmb_plot_kind = QtWidgets.QComboBox(); self.cmb_plot_kind.addItems(["Scatter","Line","Heatmap"])
        self.cmb_x = QtWidgets.QComboBox(); self.cmb_y = QtWidgets.QComboBox()
        self.cmb_val = QtWidgets.QComboBox(); self.cmb_val.setEnabled(False)
        self.cmb_plot_kind.currentTextChanged.connect(lambda t: self.cmb_val.setEnabled(t=="Heatmap"))
        self.cmb_type = QtWidgets.QComboBox(); self.cmb_type.addItems(["both","call","put"])
        self.btn_plot = QtWidgets.QPushButton("Plot"); self.btn_plot.clicked.connect(self.request_plot.emit)
        grid.addWidget(QtWidgets.QLabel("Kind:"),   0,0); grid.addWidget(self.cmb_plot_kind, 0,1)
        grid.addWidget(QtWidgets.QLabel("X:"),      1,0); grid.addWidget(self.cmb_x,         1,1)
        grid.addWidget(QtWidgets.QLabel("Y:"),      2,0); grid.addWidget(self.cmb_y,         2,1)
        grid.addWidget(QtWidgets.QLabel("Value:"),  3,0); grid.addWidget(self.cmb_val,       3,1)
        grid.addWidget(QtWidgets.QLabel("Type:"),   0,2); grid.addWidget(self.cmb_type,      0,3)
        grid.addWidget(self.btn_plot,               0,4,4,1)
    def set_columns(self, cols: List[str]):
        priority = ["strike","mid","lastPrice","impliedVolatility","volume","openInterest",
                    "T","days_to_exp","expiration","_exp_dt","bid","ask","type"]
        ordered = list(dict.fromkeys(priority + cols))
        self.cmb_x.clear(); self.cmb_y.clear(); self.cmb_val.clear()
        self.cmb_x.addItems(ordered); self.cmb_y.addItems(ordered); self.cmb_val.addItems(ordered)
    def get_params(self) -> Dict[str, Any]:
        return {"kind": self.cmb_plot_kind.currentText(),
                "x": self.cmb_x.currentText(), "y": self.cmb_y.currentText(),
                "val": self.cmb_val.currentText(), "otype": self.cmb_type.currentText()}

# ---------- Portfolio Dock ----------
class PortfolioDock(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Portfolio", parent)
        w = QtWidgets.QWidget(); self.setWidget(w)
        v = QtWidgets.QVBoxLayout(w)

        # Toolbar
        bar = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add Selected")
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_straddle = QtWidgets.QPushButton("Straddle")
        self.btn_strangle = QtWidgets.QPushButton("Strangle")
        self.btn_call_vert = QtWidgets.QPushButton("Call Vertical")
        self.btn_put_vert = QtWidgets.QPushButton("Put Vertical")
        for b in (self.btn_add, self.btn_remove, self.btn_clear, self.btn_straddle, self.btn_strangle, self.btn_call_vert, self.btn_put_vert):
            bar.addWidget(b)
        v.addLayout(bar)

        # Table: legs
        self.tbl = QtWidgets.QTableWidget()
        self.cols = ["type","strike","premium","expiration","T","qty"]
        self.tbl.setColumnCount(len(self.cols))
        self.tbl.setHorizontalHeaderLabels(self.cols)
        v.addWidget(self.tbl, 1)

    def _append_leg(self, leg: Dict[str, Any]):
        r = self.tbl.rowCount(); self.tbl.insertRow(r)
        for j, c in enumerate(self.cols):
            it = QtWidgets.QTableWidgetItem(str(leg.get(c,"")))
            if c == "qty": it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(r, j, it)

    def legs_df(self) -> pd.DataFrame:
        rows = []
        for i in range(self.tbl.rowCount()):
            row = {}
            for j, c in enumerate(self.cols):
                it = self.tbl.item(i, j)
                row[c] = it.text() if it else ""
            rows.append(row)
        df = pd.DataFrame(rows)
        # enforce numeric where applicable
        for c in ("strike","premium","T","qty"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def clear(self):
        self.tbl.setRowCount(0)

# ---------- Simulator ----------
class SimulatorWindow(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, legs: pd.DataFrame):
        super().__init__(parent); self.setWindowTitle("Portfolio Simulator"); self.resize(1200, 800)
        self.legs = legs.copy()

        # Distribution controls
        left = QtWidgets.QGroupBox("Terminal price model")
        form = QtWidgets.QFormLayout(left)
        self.cmb_model = QtWidgets.QComboBox(); self.cmb_model.addItems(["Lognormal","Custom sampler"])
        self.sp_S0   = QtWidgets.QDoubleSpinBox(); self.sp_S0.setRange(0.01, 1e9); self.sp_S0.setDecimals(4); self.sp_S0.setValue(self._guess_S0())
        self.sp_mu   = QtWidgets.QDoubleSpinBox(); self.sp_mu.setRange(-1.0, 1.0); self.sp_mu.setDecimals(4); self.sp_mu.setValue(0.0)
        self.sp_sig  = QtWidgets.QDoubleSpinBox(); self.sp_sig.setRange(0.0001, 5.0); self.sp_sig.setDecimals(4); self.sp_sig.setValue(0.20)
        self.sp_T    = QtWidgets.QDoubleSpinBox(); self.sp_T.setRange(0.0001, 10.0); self.sp_T.setDecimals(6); self.sp_T.setValue(self._guess_T())
        self.sp_N    = QtWidgets.QSpinBox(); self.sp_N.setRange(100, 2_000_000); self.sp_N.setValue(100000)
        self.ed_custom = QtWidgets.QLineEdit(); self.ed_custom.setPlaceholderText("e.g., S0*np.exp(np.random.normal(-0.02, 0.25, size=N))")
        self.cmb_model.currentTextChanged.connect(lambda t: self.ed_custom.setEnabled(t=="Custom sampler"))
        self.ed_custom.setEnabled(False)
        self.sp_mult = QtWidgets.QSpinBox(); self.sp_mult.setRange(1, 10000); self.sp_mult.setValue(100)
        self.chk_live = QtWidgets.QCheckBox("Live update"); self.chk_live.setChecked(True)
        for label, w in [("Model:", self.cmb_model), ("S0:", self.sp_S0), ("μ:", self.sp_mu), ("σ:", self.sp_sig),
                         ("T (yrs):", self.sp_T), ("N:", self.sp_N), ("Custom S_T expr:", self.ed_custom),
                         ("Contract multiplier:", self.sp_mult), ("", self.chk_live)]:
            form.addRow(label, w)

        # Results/plot
        right = QtWidgets.QGroupBox("Results")
        rLay = QtWidgets.QVBoxLayout(right)
        self.canvas = MplCanvas(); rLay.addWidget(self.canvas, 3)
        self.txt_stats = QtWidgets.QPlainTextEdit(); self.txt_stats.setReadOnly(True)
        rLay.addWidget(self.txt_stats, 2)
        self.btn_run = QtWidgets.QPushButton("Run Simulation"); rLay.addWidget(self.btn_run)

        # Layout
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(left, 0); lay.addWidget(right, 1)

        # Wire events
        self.btn_run.clicked.connect(self.run_sim)
        for w in (self.sp_S0, self.sp_mu, self.sp_sig, self.sp_T, self.sp_N, self.sp_mult, self.cmb_model):
            w.valueChanged.connect(self._maybe_live) if hasattr(w, "valueChanged") else w.currentTextChanged.connect(self._maybe_live)
        self.ed_custom.textChanged.connect(self._maybe_live)
        self.run_sim()

    def _guess_S0(self) -> float:
        try:
            k = self.legs["strike"].median()
            if np.isfinite(k): return float(k)
        except Exception: pass
        return 100.0

    def _guess_T(self) -> float:
        try:
            t = self.legs["T"].median()
            if np.isfinite(t): return float(t)
        except Exception: pass
        return 0.25

    def _maybe_live(self, *args):
        if self.chk_live.isChecked():
            self.run_sim()

    def _sample_ST(self, S0: float, N: int) -> np.ndarray:
        if self.cmb_model.currentText() == "Lognormal":
            mu = float(self.sp_mu.value()); sig = float(self.sp_sig.value()); T = float(self.sp_T.value())
            m = (mu - 0.5*sig*sig)*T; s = sig*np.sqrt(T)
            return S0*np.exp(np.random.normal(m, s, size=N))
        expr = self.ed_custom.text().strip()
        if not expr: raise ValueError("Provide a custom sampler expression that returns an array of prices.")
        local = {"np": np, "S0": float(self.sp_S0.value()), "N": int(self.sp_N.value())}
        arr = eval(expr, {"__builtins__": {}}, local)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1: raise ValueError("Custom sampler must produce a 1-D array of prices.")
        return arr

    def run_sim(self):
        try:
            S0 = float(self.sp_S0.value()); N = int(self.sp_N.value()); mult = int(self.sp_mult.value())
            legs = self.legs.dropna(subset=["type","strike","premium","T","qty"])
            if legs.empty:
                self.txt_stats.setPlainText("No legs to simulate."); return
            # all same expiration assumed (same T)
            Ts = sorted(set(round(t, 10) for t in legs["T"] if np.isfinite(t)))
            if len(Ts) != 1:
                self.txt_stats.setPlainText("All legs must share the same expiration (same T)."); return
            ST = self._sample_ST(S0, N)
            pnl = np.zeros_like(ST, dtype=float)
            for _, p in legs.iterrows():
                K = float(p["strike"]); prem = float(p["premium"]); qty = float(p["qty"])
                kind = str(p["type"]).lower().strip()
                intrinsic = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
                pnl += qty * mult * (intrinsic - prem)
            # Plot
            self.canvas.clear(); ax = self.canvas.ax
            ax.hist(pnl, bins=100); ax.set_title("Portfolio P&L at expiration")
            ax.set_xlabel("P&L"); ax.set_ylabel("count"); ax.grid(True); self.canvas.draw_idle()
            # Stats
            mean = float(np.mean(pnl)); std = float(np.std(pnl, ddof=1))
            p5, p50, p95 = np.percentile(pnl, [5,50,95])
            var5 = float(np.quantile(pnl, 0.05)); es5 = float(np.mean(pnl[pnl <= var5])) if np.any(pnl <= var5) else float("nan")
            self.txt_stats.setPlainText("\n".join([
                f"N = {N:,}", f"E[P&L] = {mean:,.2f}", f"Std = {std:,.2f}",
                f"P5 / P50 / P95 = {p5:,.2f} / {p50:,.2f} / {p95:,.2f}",
                f"VaR(5%) = {var5:,.2f}", f"ES(5%) = {es5:,.2f}",
            ]))
        except Exception as e:
            self.txt_stats.setPlainText("Error:\n" + "".join(traceback.format_exc()))

# ---------- Main Window ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Options Analyzer — Dark"); self.resize(1600, 950)
        self._portfolio = PortfolioDock(self); self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._portfolio)

        splitter = QtWidgets.QSplitter(); splitter.setOrientation(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(left)
        self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText("Filter… (press Enter)")
        self.search.returnPressed.connect(self.refresh_list)
        self.list = QtWidgets.QListWidget(); self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        row = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_fetch   = QtWidgets.QPushButton("Fetch Chain(s)")
        self.btn_view    = QtWidgets.QPushButton("View Table")
        self.btn_sim     = QtWidgets.QPushButton("Simulator…")
        self.btn_delete  = QtWidgets.QPushButton("Delete")
        for b in (self.btn_refresh, self.btn_fetch, self.btn_view, self.btn_sim, self.btn_delete):
            row.addWidget(b)
        v.addWidget(self.search); v.addWidget(self.list, 1); v.addLayout(row)

        right = QtWidgets.QWidget(); rlay = QtWidgets.QVBoxLayout(right)
        self.controls = PlotControls(); rlay.addWidget(self.controls)
        self.canvas = MplCanvas(); rlay.addWidget(self.canvas, 2)
        self.table = QtWidgets.QTableWidget(); rlay.addWidget(self.table, 3)

        splitter.addWidget(left); splitter.addWidget(right); splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Connect buttons
        self.btn_refresh.clicked.connect(self.refresh_list)
        self.btn_fetch.clicked.connect(self.fetch_chain)
        self.btn_view.clicked.connect(self.view_selected)
        self.btn_sim.clicked.connect(self.open_simulator)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.controls.request_plot.connect(self.plot_selected)
        self.list.itemSelectionChanged.connect(self.populate_columns_from_selection)

        # Portfolio actions
        self._portfolio.btn_add.clicked.connect(self.add_selected_to_portfolio)
        self._portfolio.btn_remove.clicked.connect(self.remove_selected_from_portfolio)
        self._portfolio.btn_clear.clicked.connect(self._portfolio.clear)
        self._portfolio.btn_straddle.clicked.connect(lambda: self.add_strategy("straddle"))
        self._portfolio.btn_strangle.clicked.connect(lambda: self.add_strategy("strangle"))
        self._portfolio.btn_call_vert.clicked.connect(lambda: self.add_strategy("call_vertical"))
        self._portfolio.btn_put_vert.clicked.connect(lambda: self.add_strategy("put_vertical"))

        # autosync from disk
        self.sync_from_disk(); self.refresh_list()

    # ---- disk sync & list
    def sync_from_disk(self):
        cat = _read_catalog(); existing = set(cat["datasets"].keys()); added = 0
        for csv_path in DATA_DIR_FLAT.glob("*.csv"):
            dsid = csv_path.stem
            if dsid in existing: continue
            parts = dsid.split(".", 2)
            source = parts[0]; rest = parts[1] if len(parts)>1 else dsid
            symbol = rest.split(".")[0]
            try:
                rows = sum(1 for _ in open(csv_path)) - 1
                with open(csv_path, "rb") as f: sha = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                rows, sha = None, None
            meta = {"id": dsid, "source": source, "symbol": symbol, "kind": "options", "rows": rows, "sha256": sha}
            (META_DIR / f"{dsid}.json").write_text(json.dumps(meta, indent=2))
            cat["datasets"][dsid] = {"path": str(csv_path), "meta": meta}; added += 1
        if added: _write_catalog(cat)

    def selected_ids(self) -> List[str]:
        return [i.text() for i in self.list.selectedItems()]

    def refresh_list(self):
        self.list.clear(); filt = self.search.text().strip().lower()
        for dsid in list_datasets():
            if filt and filt not in dsid.lower(): continue
            self.list.addItem(dsid)

    # ---- fetch
    def fetch_chain(self):
        d = FetchOptionsDialog(self)
        if d.exec_()!=QtWidgets.QDialog.Accepted: return
        p = d.params()
        try:
            if p["expiration"] is None:
                summary = fetch_all_and_upsert_yahoo_chains(p["ticker"], limit=None, sleep_sec=0.0)
                msg = (f"{summary['ticker']}: {summary['total']} expirations — "
                       f"new {summary['saved']}, updated {summary['updated']}, "
                       f"identical {summary['identical']}")
                QtWidgets.QMessageBox.information(self, "Fetch ALL", msg)
            else:
                result = fetch_and_upsert_yahoo_chain(p["ticker"], p["expiration"])
                reason = result["reason"]
                msg = f"{'Saved' if result['saved'] else 'Skipped'} ({reason}): {Path(result['path']).name}"
                QtWidgets.QMessageBox.information(self, "Fetch", msg)
            self.refresh_list()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fetch error", str(e))

    # ---- table / plotting
    def view_selected(self):
        dsids = self.selected_ids()
        if not dsids:
            QtWidgets.QMessageBox.information(self, "Select", "Pick one or more saved chains."); return
        df = load_many(dsids)
        if df.empty:
            QtWidgets.QMessageBox.warning(self, "Load", "No data loaded."); return
        self.show_table(add_derived_columns(df))

    def show_table(self, df: pd.DataFrame):
        self.table.clear()
        df_show = df.copy()
        self.table.setRowCount(len(df_show)); self.table.setColumnCount(len(df_show.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df_show.columns])
        max_rows = min(len(df_show), 5000)
        for i in range(max_rows):
            row = df_show.iloc[i]
            for j, col in enumerate(df_show.columns):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(str(row[col])))
        self.table.resizeColumnsToContents()

    def populate_columns_from_selection(self):
        dsids = self.selected_ids()
        if not dsids:
            self.controls.set_columns([]); return
        df = load_many(dsids)
        if df.empty:
            self.controls.set_columns([]); return
        df = add_derived_columns(df)
        cols = [c for c in df.columns if c != "__dsid__"]
        self.controls.set_columns([str(c) for c in cols])

    def _is_datetime_like(self, s: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(s): return True
        try:
            pd.to_datetime(s.dropna().astype(str).head(5), errors="raise"); return True
        except Exception: return False

    def plot_selected(self):
        dsids = self.selected_ids()
        if not dsids:
            QtWidgets.QMessageBox.information(self, "Select", "Pick one or more saved chains."); return
        df = load_many(dsids)
        if df.empty:
            QtWidgets.QMessageBox.warning(self, "Load", "No data loaded."); return
        params = self.controls.get_params()
        df = add_derived_columns(df)
        if "type" in df.columns and params.get("otype") in ("call","put"):
            df = df[df["type"] == params["otype"]]
        if df.empty:
            QtWidgets.QMessageBox.information(self, "Filter", "No rows after filtering."); return
        kind = params["kind"]; xcol = params["x"]; ycol = params["y"]; vcol = params["val"]
        self.canvas.clear(); ax = self.canvas.ax

        def maybe_to_datetime(series: pd.Series):
            if self._is_datetime_like(series): return pd.to_datetime(series, errors="coerce")
            return series

        if xcol not in df.columns or ycol not in df.columns:
            QtWidgets.QMessageBox.warning(self, "Plot", "X or Y column not in data."); return

        X = maybe_to_datetime(df[xcol]); Y = maybe_to_datetime(df[ycol])
        try:
            if kind in ("Scatter","Line"):
                mask = pd.notnull(X) & pd.notnull(Y)
                Xp = X[mask]; Yp = Y[mask]
                if kind == "Scatter":
                    ax.scatter(Xp, Yp, s=8)
                else:
                    if self._is_datetime_like(Xp):
                        order = np.argsort(pd.to_datetime(Xp).astype(np.int64).values)
                    else:
                        order = np.argsort(np.array(Xp))
                    ax.plot(np.array(Xp)[order], np.array(Yp)[order])
                ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.grid(True)
                if self._is_datetime_like(Xp):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    self.canvas.fig.autofmt_xdate()
            else:
                if vcol not in df.columns:
                    QtWidgets.QMessageBox.warning(self, "Heatmap", f"Value column '{vcol}' not found."); return
                Xh = X.dt.strftime("%Y-%m-%d %H:%M") if self._is_datetime_like(X) else X.astype(str)
                Yh = Y.dt.strftime("%Y-%m-%d %H:%M") if self._is_datetime_like(Y) else Y.astype(str)
                pv = pd.pivot_table(df.assign(_X_=Xh, _Y_=Yh), index="_Y_", columns="_X_", values=vcol, aggfunc="mean")
                pv = pv.sort_index().sort_index(axis=1)
                im = ax.imshow(pv.values, aspect="auto", origin="lower")
                ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
                ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns, rotation=90)
                ax.set_xlabel(xcol); ax.set_ylabel(ycol)
                self.canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=vcol)
            self.canvas.draw_idle()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Plot error", str(e))

    # ---- portfolio ops
    def _selected_table_rows_df(self) -> pd.DataFrame:
        # Build DF from visible table rows currently selected
        sel = self.table.selectedIndexes()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Portfolio", "Select one or more rows in the table on the right.")
            return pd.DataFrame()
        rows = sorted(set(ix.row() for ix in sel))
        cols = [self.table.horizontalHeaderItem(j).text() for j in range(self.table.columnCount())]
        data = []
        for i in rows:
            row_vals = []
            for j in range(self.table.columnCount()):
                it = self.table.item(i, j)
                row_vals.append(None if it is None else it.text())
            data.append(row_vals)
        df = pd.DataFrame(data, columns=cols)
        for c in ("strike","mid","T"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def add_selected_to_portfolio(self):
        df = self._selected_table_rows_df()
        if df.empty: return
        df = add_derived_columns(df)
        needed = ["type","strike","mid","expiration","T"]
        for c in needed:
            if c not in df.columns:
                QtWidgets.QMessageBox.warning(self, "Portfolio", f"Missing column: {c}"); return
        for _, r in df.iterrows():
            leg = {"type": str(r["type"]).lower(), "strike": float(r["strike"]),
                   "premium": float(r["mid"]), "expiration": str(r["expiration"]),
                   "T": float(r["T"]), "qty": 1}
            self._portfolio._append_leg(leg)

    def remove_selected_from_portfolio(self):
        tbl = self._portfolio.tbl
        sel_rows = sorted(set(ix.row() for ix in tbl.selectedIndexes()), reverse=True)
        for r in sel_rows:
            tbl.removeRow(r)

    def add_strategy(self, kind: str):
        df = self._selected_table_rows_df()
        if df.empty: return
        df = add_derived_columns(df)
        df = df.sort_values("strike")
        exp_set = set(df["expiration"].astype(str))
        if len(exp_set) != 1:
            QtWidgets.QMessageBox.warning(self, "Strategy", "Select rows from a single expiration."); return
        exp = list(exp_set)[0]; T = float(df["T"].iloc[0])

        def add_leg(otype, K, prem, qty):
            self._portfolio._append_leg({"type": otype, "strike": float(K), "premium": float(prem),
                                         "expiration": exp, "T": T, "qty": qty})

        # choose ATM/OTM from selected rows
        strikes = df["strike"].values
        mids = df["mid"].values if "mid" in df.columns else np.zeros_like(strikes)
        K_low, K_high = strikes.min(), strikes.max()
        K_mid = strikes[np.argsort(np.abs(strikes - np.median(strikes)))[0]]
        mid_atm = float(df.loc[df["strike"]==K_mid, "mid"].iloc[0]) if (df["strike"]==K_mid).any() else float(np.nan)

        if kind == "straddle":
            # +1 call @ K_mid, +1 put @ K_mid
            add_leg("call", K_mid, mid_atm if np.isfinite(mid_atm) else 0.0, +1)
            add_leg("put",  K_mid, mid_atm if np.isfinite(mid_atm) else 0.0, +1)
        elif kind == "strangle":
            # +1 call @ K_high, +1 put @ K_low
            prem_high = float(df.loc[df["strike"]==K_high, "mid"].iloc[0]) if (df["strike"]==K_high).any() else 0.0
            prem_low  = float(df.loc[df["strike"]==K_low, "mid"].iloc[0]) if (df["strike"]==K_low).any() else 0.0
            add_leg("call", K_high, prem_high, +1); add_leg("put", K_low, prem_low, +1)
        elif kind == "call_vertical":
            # +1 call lower K, -1 call higher K
            prem_low  = float(df.loc[df["strike"]==K_low, "mid"].iloc[0]) if (df["strike"]==K_low).any() else 0.0
            prem_high = float(df.loc[df["strike"]==K_high, "mid"].iloc[0]) if (df["strike"]==K_high).any() else 0.0
            add_leg("call", K_low,  prem_low,  +1); add_leg("call", K_high, prem_high, -1)
        elif kind == "put_vertical":
            # +1 put higher K, -1 put lower K (debit put spread)
            prem_high = float(df.loc[df["strike"]==K_high, "mid"].iloc[0]) if (df["strike"]==K_high).any() else 0.0
            prem_low  = float(df.loc[df["strike"]==K_low, "mid"].iloc[0]) if (df["strike"]==K_low).any() else 0.0
            add_leg("put", K_high, prem_high, +1); add_leg("put", K_low, prem_low, -1)

    # ---- simulator
    def open_simulator(self):
        legs = self._portfolio.legs_df()
        if legs.empty:
            QtWidgets.QMessageBox.information(self, "Simulator", "Portfolio is empty. Add legs first.")
            return
        sim = SimulatorWindow(self, legs)
        sim.exec_()

    # ---- delete datasets
    def delete_selected(self):
        dsids = self.selected_ids()
        if not dsids: return
        if QtWidgets.QMessageBox.question(self, "Delete", f"Delete {len(dsids)} dataset(s)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        cat = _read_catalog()
        for dsid in dsids:
            rec = cat["datasets"].get(dsid)
            if not rec: continue
            try:
                Path(rec["path"]).unlink(missing_ok=True)
                (META_DIR / f"{dsid}.json").unlink(missing_ok=True)
            except Exception: pass
            cat["datasets"].pop(dsid, None)
        _write_catalog(cat)
        self.refresh_list()
        QtWidgets.QMessageBox.information(self, "Deleted", f"Deleted {len(dsids)} dataset(s).")

def main():
    app = QtWidgets.QApplication(sys.argv)
    enable_dark(app)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
