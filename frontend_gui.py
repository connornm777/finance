#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frontend_gui.py — dark-mode GUI (Yahoo + FRED; no options)

Features:
  - Auto-sync catalog from DATA_DIR/econ/data/*.csv on launch
  - List catalog, fetch (Yahoo/FRED), plot multiple series, delete
  - "Reload All" to bring all fetchable series current to today

Run:
  pip install PyQt5 matplotlib pandas numpy yfinance requests python-dotenv
  python frontend_gui.py
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import utils
except Exception:
    sys.path.insert(0, os.getcwd())
    import utils

# ---------------- Matplotlib canvas ----------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

    def clear(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.draw_idle()

# ---------------- Dark theme -----------------------
def enable_dark(app: QtWidgets.QApplication):
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

# ---------------- Fetch dialogs --------------------
class FetchYahooDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fetch — Yahoo Price")
        self.ticker = QtWidgets.QLineEdit()
        self.start = QtWidgets.QLineEdit()
        self.end = QtWidgets.QLineEdit()
        self.interval = QtWidgets.QComboBox(); self.interval.addItems(["1d","1wk","1mo"])
        self.field = QtWidgets.QComboBox(); self.field.addItems(["Adj Close","Close","Open"])
        self.prepost = QtWidgets.QCheckBox("Pre/Post market")

        form = QtWidgets.QFormLayout()
        form.addRow("Ticker:", self.ticker)
        form.addRow("Start (YYYY-MM-DD):", self.start)
        form.addRow("End (YYYY-MM-DD):", self.end)
        form.addRow("Interval:", self.interval)
        form.addRow("Field:", self.field)
        form.addRow("", self.prepost)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(form); lay.addWidget(buttons)

    def params(self) -> Dict[str, Any]:
        return dict(
            ticker=self.ticker.text().strip(),
            start=(self.start.text().strip() or None),
            end=(self.end.text().strip() or None),
            interval=self.interval.currentText(),
            price_field=self.field.currentText(),
            prepost=self.prepost.isChecked(),
        )

class FetchFREDDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fetch — FRED")
        self.sid = QtWidgets.QLineEdit()
        self.start = QtWidgets.QLineEdit()
        self.end = QtWidgets.QLineEdit()

        form = QtWidgets.QFormLayout()
        form.addRow("Series ID:", self.sid)
        form.addRow("Start (YYYY-MM-DD):", self.start)
        form.addRow("End (YYYY-MM-DD):", self.end)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(form); lay.addWidget(buttons)

    def params(self) -> Dict[str, Any]:
        return dict(
            series_id=self.sid.text().strip(),
            start=(self.start.text().strip() or "1900-01-01"),
            end=(self.end.text().strip() or "2100-01-01"),
        )

# ---------------- Main window -----------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Econ Data — Dark")
        self.resize(1200, 800)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)

        # left controls
        left = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(left)
        self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText("Filter…  (press Enter)")
        self.search.returnPressed.connect(self.refresh_list)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        row = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_fetch_y = QtWidgets.QPushButton("Fetch Yahoo")
        self.btn_fetch_f = QtWidgets.QPushButton("Fetch FRED")
        self.btn_plot    = QtWidgets.QPushButton("Plot")
        self.btn_reload  = QtWidgets.QPushButton("Reload All")
        self.btn_delete  = QtWidgets.QPushButton("Delete")
        for b in (self.btn_refresh, self.btn_fetch_y, self.btn_fetch_f, self.btn_plot, self.btn_reload, self.btn_delete):
            row.addWidget(b)
        v.addWidget(self.search); v.addWidget(self.list, 1); v.addLayout(row)

        # right: plot
        self.canvas = MplCanvas()
        splitter.addWidget(left); splitter.addWidget(self.canvas); splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # connect
        self.btn_refresh.clicked.connect(self.refresh_list)
        self.btn_fetch_y.clicked.connect(self.fetch_yahoo)
        self.btn_fetch_f.clicked.connect(self.fetch_fred)
        self.btn_plot.clicked.connect(self.plot_selected)
        self.btn_reload.clicked.connect(self.reload_all_series)
        self.btn_delete.clicked.connect(self.delete_selected)

        # auto-sync existing CSVs
        try:
            added = utils.sync_catalog_from_disk()
            if added:
                QtWidgets.QMessageBox.information(self, "Catalog", f"Imported {added} dataset(s) from disk.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Catalog sync", f"Disk sync skipped: {e}")

        self.refresh_list()

    # ---- helpers ----
    def selected_dsids(self) -> List[str]:
        return [i.text() for i in self.list.selectedItems()]

    def refresh_list(self):
        self.list.clear()
        filt = self.search.text().strip().lower()
        try:
            dsids = utils.list_datasets()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Catalog error", str(e)); return
        for ds in dsids:
            if filt and filt not in ds.lower():
                continue
            self.list.addItem(ds)

    # ---- fetch ----
    def fetch_yahoo(self):
        d = FetchYahooDialog(self)
        if d.exec_() != QtWidgets.QDialog.Accepted: return
        p = d.params()
        try:
            df = utils.fetch_yahoo_price(**p)
            if df.empty: raise RuntimeError("Empty dataset")
            meta = dict(source="yahoo", symbol=p["ticker"], kind="series", frequency=p["interval"])
            utils.save_data(df, meta)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved yahoo.{p['ticker']}")
            self.refresh_list()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fetch error", str(e))

    def fetch_fred(self):
        d = FetchFREDDialog(self)
        if d.exec_() != QtWidgets.QDialog.Accepted: return
        p = d.params()
        try:
            df = utils.fetch_fred(**p)
            if df.empty: raise RuntimeError("Empty dataset")
            meta = dict(source="fred", symbol=p["series_id"], kind="series")
            utils.save_data(df, meta)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved fred.{p['series_id']}")
            self.refresh_list()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fetch error", str(e))

    # ---- plot ----
    def plot_selected(self):
        dsids = self.selected_dsids()
        if not dsids:
            QtWidgets.QMessageBox.information(self, "Select", "Pick one or more series."); return
        self.canvas.clear(); ax = self.canvas.ax; plotted = 0
        for dsid in dsids:
            try:
                df = utils.load_data(dsid)
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"]); df.set_index("date", inplace=True)
                ser = df.iloc[:, 0]
                ax.plot(ser.index, ser.values, label=dsid); plotted += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load error", f"{dsid}\n{e}")
        if plotted:
            ax.legend(loc="best"); ax.grid(True); ax.set_xlabel("Date"); ax.set_ylabel("Value"); self.canvas.draw_idle()

    # ---- reload all ----
    def reload_all_series(self):
        dsids = utils.list_datasets()
        if not dsids:
            QtWidgets.QMessageBox.information(self, "Reload", "No datasets in catalog."); return
        prog = QtWidgets.QProgressDialog("Reloading datasets…", "Cancel", 0, len(dsids), self)
        prog.setWindowModality(QtCore.Qt.ApplicationModal); prog.setMinimumDuration(0)

        def step(i, n, dsid, updated):
            prog.setLabelText(f"[{i}/{n}] {dsid} — {'updated' if updated else 'skipped'}")
            prog.setValue(i); QtWidgets.QApplication.processEvents()

        try:
            summary = utils.reload_all(progress=step)
            prog.setValue(len(dsids)); self.refresh_list()
            QtWidgets.QMessageBox.information(self, "Reload complete",
                f"Total: {summary['total']}\nUpdated: {summary['updated']}\nSkipped: {summary['skipped']}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Reload error", str(e))

    # ---- delete ----
    def delete_selected(self):
        dsids = self.selected_dsids()
        if not dsids: return
        if QtWidgets.QMessageBox.question(self, "Delete", f"Delete {len(dsids)} dataset(s)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        cat = utils._read_catalog()
        for dsid in dsids:
            rec = cat["datasets"].get(dsid)
            if not rec: continue
            try:
                Path(rec["path"]).unlink(missing_ok=True)
                (utils.META_DIR / f"{dsid}.json").unlink(missing_ok=True)
            except Exception:
                pass
            cat["datasets"].pop(dsid, None)
        utils._write_catalog(cat)
        self.refresh_list()
        QtWidgets.QMessageBox.information(self, "Deleted", f"Deleted {len(dsids)} dataset(s).")

# ---------------- entry ----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    enable_dark(app)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
