"""Matplotlib-based analysis panels used by the interactive viewer."""

from __future__ import annotations

from ._imports import require_viewer_dependencies

_, _, _, QtCore, _, QtWidgets = require_viewer_dependencies()

# Cross-binding Signal: PySide2/6 uses QtCore.Signal, PyQt5/6 uses QtCore.pyqtSignal.
try:
    _Signal = QtCore.Signal      # PySide2 / PySide6
except AttributeError:
    _Signal = QtCore.pyqtSignal  # PyQt5 / PyQt6

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:  # pragma: no cover - compatibility fallback
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class _BgWorker(QtCore.QThread):
    """Run a callable in a background thread; emit (key, result) on finish.

    *result* is either the return value of *fn* or an ``Exception`` instance.
    Connect to ``finished`` before calling ``start()``.  Disconnect before
    discarding this object to avoid stale callbacks.
    """

    finished = _Signal(object, object)  # (key, result_or_exception)

    def __init__(self, key, fn, parent=None):
        super().__init__(parent)
        self._key = key
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001
            result = exc
        self.finished.emit(self._key, result)


COMPONENT_COLORS = {
    "Z": "tab:blue",
    "E": "tab:orange",
    "N": "tab:green",
    "z": "tab:blue",
    "e": "tab:orange",
    "n": "tab:green",
}


class TracePanel(QtWidgets.QWidget):
    """Embedded matplotlib panel showing node traces."""

    def __init__(self, session, parent=None, *, demand: str | None = None, title: str | None = None):
        super().__init__(parent)
        self.session = session
        self.demand = None if demand is None else str(demand)
        self.fixed_title = title

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self._trace_lines = []
        self._time_cursors = []
        self._current_key = None

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)

        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time" and self._current_key is not None:
            self._update_time_cursor()
            self.canvas.draw_idle()
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_key = None
            self._trace_lines = []
            self._time_cursors = []
            for ax in self.axes:
                ax.clear()
            self.title_label.setText("No node selected")
            for ax, label in zip(self.axes, ("Z", "E", "N")):
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.25)
            self.axes[-1].set_xlabel("Time [s]")
            self.canvas.draw_idle()
            return

        demand = self.demand or self.session.state.demand
        trace = (
            self.session.adapter.trace(node_id, demand)
            if self.demand is not None
            else self.session.current_trace()
        )
        time = self.session.adapter.time
        labels = ("Z", "E", "N")
        node_key = (node_id, demand)

        if self._current_key != node_key:
            self._current_key = node_key
            self._trace_lines = []
            self._time_cursors = []
            for ax in self.axes:
                ax.clear()

            title = self.fixed_title or f"Node {node_id} | {demand} traces"
            self.title_label.setText(title)
            for ax, values, label in zip(self.axes, trace, labels):
                line, = ax.plot(
                    time,
                    values,
                    linewidth=1.2,
                    color=COMPONENT_COLORS[label],
                    label=label,
                )
                cursor = ax.axvline(self.session.current_time(), color="tab:red", alpha=0.35)
                self._trace_lines.append(line)
                self._time_cursors.append(cursor)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="upper right")
            self.axes[-1].set_xlabel("Time [s]")
        else:
            self.title_label.setText(self.fixed_title or f"Node {node_id} | {demand} traces")
            for line, values in zip(self._trace_lines, trace):
                line.set_ydata(values)

        self._update_time_cursor()
        self.canvas.draw_idle()

    def _update_time_cursor(self):
        current_time = self.session.current_time()
        for cursor in self._time_cursors:
            cursor.set_xdata([current_time, current_time])


class SpectrumPanel(QtWidgets.QWidget):
    """Panel that computes and shows Newmark PSa on demand."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None
        self._table_rows: list[tuple[float, float, float, float]] = []
        self._worker: _BgWorker | None = None  # background computation thread

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.title_label.setStyleSheet("font-size: 11px; color: #1e3558; font-weight: 600;")
        self.title_label.setContentsMargins(2, 2, 2, 0)
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(330)
        self.canvas.setMaximumHeight(390)
        self.axes = self.figure.subplots(3, 1, sharex=True)

        plot_box = QtWidgets.QGroupBox("Plots")
        plot_layout = QtWidgets.QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.addWidget(self.canvas)

        table_box = QtWidgets.QGroupBox("Data")
        table_layout = QtWidgets.QVBoxLayout(table_box)
        table_layout.setContentsMargins(6, 8, 6, 6)
        table_layout.setSpacing(6)

        table_header = QtWidgets.QWidget()
        table_header_layout = QtWidgets.QHBoxLayout(table_header)
        table_header_layout.setContentsMargins(0, 0, 0, 0)
        table_header_layout.setSpacing(6)
        table_label = QtWidgets.QLabel("T, Sa (Z), Sa (E), Sa (N)")
        table_label.setStyleSheet("color: #1e3558; font-weight: 600;")
        self.copy_table_button = QtWidgets.QPushButton("Copy")
        self.copy_table_button.setToolTip("Copy spectrum table to clipboard")
        self.copy_table_button.setEnabled(False)
        self.copy_table_button.clicked.connect(self._copy_table_to_clipboard)
        table_header_layout.addWidget(table_label, 1)
        table_header_layout.addWidget(self.copy_table_button, 0)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["T", "Sa (Z)", "Sa (E)", "Sa (N)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(180)
        self.table.setMaximumHeight(260)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        for col in range(4):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.Stretch)

        table_layout.addWidget(table_header)
        table_layout.addWidget(self.table)

        layout.addWidget(self.title_label)
        layout.addWidget(plot_box, 0)
        layout.addWidget(table_box, 1)

        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time":
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._cancel_worker()
            self._current_node = None
            for ax in self.axes:
                ax.clear()
                ax.grid(True, alpha=0.25)
            self.title_label.setText("No node selected")
            self._set_table_rows([])
            self.axes[-1].set_xlabel("Period [s]")
            self.canvas.draw_idle()
            return

        if self._current_node == node_id and reason not in {"selection", "full", "init"}:
            return

        self._current_node = node_id
        self._cancel_worker()

        # Show a placeholder immediately so the panel doesn't look frozen.
        for ax in self.axes:
            ax.clear()
            ax.grid(True, alpha=0.25)
        self.axes[-1].set_xlabel("Period [s]")
        self.title_label.setText(f"Node {node_id} | Computing spectrum…")
        self.copy_table_button.setEnabled(False)
        self._set_table_rows([])
        self.canvas.draw_idle()

        # Capture node_id so the lambda always calls adapter.spectrum for THIS node,
        # not whatever selected_node happens to be when the thread finally runs.
        _node_id = node_id
        fn = lambda: self.session.adapter.spectrum(_node_id)  # noqa: E731
        worker = _BgWorker(_node_id, fn, parent=self)
        worker.finished.connect(self._on_spectrum_ready)
        self._worker = worker
        worker.start()

    def _cancel_worker(self):
        """Disconnect and release any running background worker (result is discarded)."""
        if self._worker is not None:
            try:
                self._worker.finished.disconnect(self._on_spectrum_ready)
            except Exception:  # noqa: BLE001
                pass
            self._worker = None

    def _on_spectrum_ready(self, node_id, result):
        """Slot called on the main thread when the background worker finishes."""
        self._worker = None
        if node_id != self._current_node:
            return  # Node changed while computing — discard stale result.

        if isinstance(result, Exception):
            self.title_label.setText(f"Spectrum unavailable: {result}")
            for ax in self.axes:
                ax.grid(True, alpha=0.25)
            self._set_table_rows([])
            self.canvas.draw_idle()
            return

        spectrum = result
        for ax in self.axes:
            ax.clear()
        self.title_label.setText(f"Node {node_id} | Newmark PSa")
        for ax, label in zip(self.axes, ("z", "e", "n")):
            ax.plot(
                spectrum["T"],
                spectrum[f"PSa_{label}"],
                linewidth=1.2,
                color=COMPONENT_COLORS[label],
                label=label.upper(),
            )
            ax.set_ylabel(label.upper())
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
        self.axes[-1].set_xlabel("Period [s]")
        self._set_table_rows(
            zip(
                spectrum["T"],
                spectrum["PSa_z"],
                spectrum["PSa_e"],
                spectrum["PSa_n"],
            )
        )
        self.canvas.draw_idle()

    def _set_table_rows(self, rows):
        self._table_rows = [
            (float(t), float(sa_z), float(sa_e), float(sa_n))
            for t, sa_z, sa_e, sa_n in rows
        ]
        self.table.setRowCount(len(self._table_rows))
        for row_idx, row in enumerate(self._table_rows):
            for col_idx, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(f"{value:.8g}")
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(row_idx, col_idx, item)
        self.copy_table_button.setEnabled(bool(self._table_rows))

    def _copy_table_to_clipboard(self):
        lines = ["T\tSa (Z)\tSa (E)\tSa (N)"]
        lines.extend(
            f"{t:.10g}\t{sa_z:.10g}\t{sa_e:.10g}\t{sa_n:.10g}"
            for t, sa_z, sa_e, sa_n in self._table_rows
        )
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))


class AriasIntensityPanel(QtWidgets.QWidget):
    """Panel showing Arias intensity curves for the selected node."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None
        self._metric_rows: list[tuple[str, float, float, float, float, float]] = []
        self._worker: _BgWorker | None = None  # background computation thread

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.title_label.setStyleSheet("font-size: 11px; color: #1e3558; font-weight: 600;")
        self.title_label.setContentsMargins(2, 2, 2, 0)
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(330)
        self.canvas.setMaximumHeight(390)
        self.axes = self.figure.subplots(3, 1, sharex=True)

        plot_box = QtWidgets.QGroupBox("Plots")
        plot_layout = QtWidgets.QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.addWidget(self.canvas)

        metrics_box = QtWidgets.QGroupBox("Arias metrics")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(6, 8, 6, 6)
        metrics_layout.setSpacing(6)

        self.metrics_table = QtWidgets.QTableWidget(0, 6)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Comp.", "Ia total", "t5", "t95", "D5-95", "pot_dest"]
        )
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.metrics_table.setMinimumHeight(96)
        self.metrics_table.setMaximumHeight(138)
        metrics_header = self.metrics_table.horizontalHeader()
        metrics_header.setStretchLastSection(False)
        for col in range(6):
            metrics_header.setSectionResizeMode(col, QtWidgets.QHeaderView.Stretch)
        metrics_layout.addWidget(self.metrics_table)

        layout.addWidget(self.title_label)
        layout.addWidget(plot_box, 0)
        layout.addWidget(metrics_box, 0)
        layout.addStretch(1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time":
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._cancel_worker()
            self._current_node = None
            for ax in self.axes:
                ax.clear()
                ax.grid(True, alpha=0.25)
            self.title_label.setText("No node selected")
            self._set_metric_rows([])
            self.axes[-1].set_xlabel("Time [s]")
            self.canvas.draw_idle()
            return

        if self._current_node == node_id and reason not in {"selection", "full", "init"}:
            return

        self._current_node = node_id
        self._cancel_worker()

        # Show a placeholder immediately so the panel doesn't look frozen.
        for ax in self.axes:
            ax.clear()
            ax.grid(True, alpha=0.25)
        self.axes[-1].set_xlabel("Time [s]")
        self.title_label.setText(f"Node {node_id} | Computing Arias…")
        self._set_metric_rows([])
        self.canvas.draw_idle()

        # Capture node_id so the lambda always calls adapter.arias for THIS node.
        _node_id = node_id
        fn = lambda: self.session.adapter.arias(_node_id)  # noqa: E731
        worker = _BgWorker(_node_id, fn, parent=self)
        worker.finished.connect(self._on_arias_ready)
        self._worker = worker
        worker.start()

    def _cancel_worker(self):
        """Disconnect and release any running background worker (result is discarded)."""
        if self._worker is not None:
            try:
                self._worker.finished.disconnect(self._on_arias_ready)
            except Exception:  # noqa: BLE001
                pass
            self._worker = None

    def _on_arias_ready(self, node_id, result):
        """Slot called on the main thread when the background worker finishes."""
        self._worker = None
        if node_id != self._current_node:
            return  # Node changed while computing — discard stale result.

        if isinstance(result, Exception):
            self.title_label.setText(f"Arias unavailable: {result}")
            for ax in self.axes:
                ax.grid(True, alpha=0.25)
            self._set_metric_rows([])
            self.canvas.draw_idle()
            return

        arias = result
        time = arias["time"]
        for ax in self.axes:
            ax.clear()
        self.title_label.setText(f"Node {node_id} | Arias Intensity")
        metric_rows = []
        for ax, label in zip(self.axes, ("z", "e", "n")):
            item = arias["components"][label]
            ax.plot(
                time,
                item["IA_pct"],
                linewidth=1.2,
                color=COMPONENT_COLORS[label],
                label=label.upper(),
            )
            ax.axvline(item["t_start"], color=COMPONENT_COLORS[label], linestyle="--", linewidth=1, alpha=0.45)
            ax.axvline(item["t_end"], color=COMPONENT_COLORS[label], linestyle="--", linewidth=1, alpha=0.45)
            ax.axhline(5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.axhline(95, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_ylabel(label.upper())
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left")
            t5 = float(item["t_start"])
            t95 = float(item["t_end"])
            metric_rows.append(
                (
                    label.upper(),
                    float(item["ia_total"]),
                    t5,
                    t95,
                    t95 - t5,
                    float(item.get("extra", 0.0)),
                )
            )
        self.axes[-1].set_xlabel("Time [s]")
        self._set_metric_rows(metric_rows)
        self.canvas.draw_idle()

    def _set_metric_rows(self, rows):
        self._metric_rows = list(rows)
        self.metrics_table.setRowCount(len(self._metric_rows))
        for row_idx, row in enumerate(self._metric_rows):
            comp, ia_total, t5, t95, duration, pot_dest = row
            values = (
                comp,
                f"{ia_total:.8g}",
                f"{t5:.8g}",
                f"{t95:.8g}",
                f"{duration:.8g}",
                f"{pot_dest:.8g}",
            )
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                align = QtCore.Qt.AlignCenter if col_idx == 0 else QtCore.Qt.AlignRight
                item.setTextAlignment(align | QtCore.Qt.AlignVCenter)
                self.metrics_table.setItem(row_idx, col_idx, item)


class CombinedTracePanel(QtWidgets.QWidget):
    """9-axis combined trace panel: Acceleration / Velocity / Displacement × Z / E / N.

    Refresh contract
    ----------------
    ``"time"``
        Move time cursors only — no HDF5 I/O, no redraw of lines.
    ``"selection"`` / ``"full"`` / ``"init"``
        Reload all three demand traces for the current node and redraw.
    Any other reason (``"demand"``, ``"component"``, ``"warp"`` …)
        No-op — traces are node-specific, not demand/component-specific.
    """

    _DEMANDS = ("accel", "vel", "disp")
    _DEMAND_TITLES = {
        "accel": "Acceleration",
        "vel":   "Velocity",
        "disp":  "Displacement",
    }
    _COMP_LABELS = ("Z", "E", "N")

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None
        self._trace_lines: list = []    # 9 Line2D objects in axis order
        self._time_cursors: list = []   # 9 axvline objects (animated=True)
        self._bg = None                 # blitting background cache

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 13), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(680)

        # 9 axes sharing the x-axis (time).
        self.axes = self.figure.subplots(9, 1, sharex=True)

        self._init_axes_labels()

        # draw_event fires after every full canvas.draw() — use it to cache
        # the background (without animated cursors) for cheap blitting.
        self.canvas.mpl_connect("draw_event", self._on_draw)

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)

        self.refresh("init")

    # ── Axes labelling ────────────────────────────────────────────────────────

    def _init_axes_labels(self):
        for grp, demand in enumerate(self._DEMANDS):
            for ci, comp in enumerate(self._COMP_LABELS):
                ax = self.axes[grp * 3 + ci]
                ax.set_ylabel(comp, fontsize=8)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if ci == 0:
                    ax.set_title(
                        self._DEMAND_TITLES[demand],
                        fontsize=9, fontweight="bold", loc="left", pad=2,
                    )
        self.axes[-1].set_xlabel("Time [s]", fontsize=8)

    # ── Blitting ──────────────────────────────────────────────────────────────

    def _on_draw(self, event):
        """Cache the static background after every full canvas.draw().

        Because cursors are created with ``animated=True``, matplotlib skips
        them during a regular draw, so the cached bitmap is cursor-free.  We
        then immediately redraw the cursors at their current position so the
        display looks correct after a resize or zoom.
        """
        if not self._time_cursors:
            return
        try:
            self._bg = self.canvas.copy_from_bbox(self.figure.bbox)
            t = self.session.current_time()
            for ax, cur in zip(self.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
        except Exception:
            self._bg = None

    def _blit_cursors(self, t: float) -> bool:
        """Move all 9 cursors to *t* using blitting.  Returns True on success."""
        if self._bg is None or not self._time_cursors:
            return False
        try:
            self.canvas.restore_region(self._bg)
            for ax, cur in zip(self.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
            return True
        except Exception:
            self._bg = None
            return False

    # ── Refresh ───────────────────────────────────────────────────────────────

    def refresh(self, reason: str = "full"):
        # ── Time: blit-only cursor update (< 1 ms, zero I/O) ─────────────────
        if reason == "time":
            if self._current_node is not None and self._time_cursors:
                t = self.session.current_time()
                if not self._blit_cursors(t):
                    # No cached background yet — fall back to draw_idle once.
                    for cur in self._time_cursors:
                        cur.set_xdata([t, t])
                    self.canvas.draw_idle()
            return

        # ── Traces don't depend on demand / component / warp / visibility ────
        if reason not in {"selection", "full", "init"}:
            return

        node_id = self.session.state.selected_node

        if node_id is None:
            self._current_node = None
            self._trace_lines = []
            self._time_cursors = []
            self._bg = None
            for ax in self.axes:
                ax.clear()
                ax.grid(True, alpha=0.25)
            self._init_axes_labels()
            self.title_label.setText("No node selected")
            self.canvas.draw_idle()
            return

        # Same node and not a forced full redraw — just sync the cursor.
        if self._current_node == node_id and reason not in {"full", "init"}:
            if self._time_cursors:
                t = self.session.current_time()
                if not self._blit_cursors(t):
                    for cur in self._time_cursors:
                        cur.set_xdata([t, t])
                    self.canvas.draw_idle()
            return

        self._current_node = node_id
        self._draw_all_traces(node_id)

    def _draw_all_traces(self, node_id):
        self._trace_lines = []
        self._time_cursors = []

        t = self.session.adapter.time
        current_t = self.session.current_time()

        try:
            traces = {
                d: self.session.adapter.trace(node_id, d)
                for d in self._DEMANDS
            }
        except Exception as exc:
            self.title_label.setText(f"Error loading traces: {exc}")
            return

        for grp, demand in enumerate(self._DEMANDS):
            data = traces[demand]   # shape (3, N_time): rows = [Z, E, N]
            for ci, comp in enumerate(self._COMP_LABELS):
                ax = self.axes[grp * 3 + ci]
                ax.clear()

                color = COMPONENT_COLORS[comp]
                (line,) = ax.plot(t, data[ci], linewidth=1.0, color=color, label=comp)
                # animated=True → matplotlib skips this artist in regular
                # draw() calls, keeping the blitting background cursor-free.
                cursor = ax.axvline(
                    current_t, color="tab:red", alpha=0.35, linewidth=1.0,
                    animated=True,
                )

                self._trace_lines.append(line)
                self._time_cursors.append(cursor)

                ax.set_ylabel(comp, fontsize=8)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if ci == 0:
                    ax.set_title(
                        self._DEMAND_TITLES[demand],
                        fontsize=9, fontweight="bold", loc="left", pad=2,
                    )

        self.axes[-1].set_xlabel("Time [s]", fontsize=8)
        self.title_label.setText(f"Node {node_id}  —  Accel / Vel / Disp")
        # Full draw establishes the blitting background via _on_draw callback.
        self._bg = None
        self.canvas.draw()


import numpy as _np   # local alias — avoids polluting the module namespace

# ── Palette & defaults for ResponsesPanel ────────────────────────────────────

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_DEFAULT_CURVES = [
    ("accel", "z"), ("accel", "e"), ("accel", "n"),
    ("vel",   "z"), ("vel",   "e"), ("vel",   "n"),
    ("disp",  "z"), ("disp",  "e"), ("disp",  "n"),
]

_COMP_ROW = {"z": 0, "e": 1, "n": 2}


class CurveSpec:
    """One user-defined curve: demand × component + display colour."""

    def __init__(self, demand: str, component: str, color: str = ""):
        self.demand    = demand
        self.component = component
        self.color     = color

    @property
    def label(self) -> str:
        return f"{self.demand}/{self.component.upper()}"


# ── ResponsesPanel ────────────────────────────────────────────────────────────

class _LegacyResponsesPanel(QtWidgets.QWidget):
    """Dynamic multi-curve response viewer.

    The user builds an ordered list of :class:`CurveSpec` objects — each one
    a (demand, component) pair for the currently selected node.  Two layout
    modes are supported:

    ``stacked``
        One subplot per curve, all sharing the same time axis (x).  Best for
        comparing magnitudes across demands / components.

    ``overlay``
        All curves drawn on a single subplot with a shared legend.  Best for
        comparing shapes / phase relationships.

    Refresh contract (same as :class:`CombinedTracePanel`)
    -------------------------------------------------------
    ``"time"``
        Blit-only cursor update — zero HDF5 I/O.
    ``"selection"`` / ``"full"`` / ``"init"``
        Reload data for the selected node and redraw.
    Any other reason
        No-op — traces don't depend on demand / warp / colormap changes.
    """

    _DEMANDS    = ("accel", "vel", "disp")
    _COMPONENTS = ("z", "e", "n")

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        # ── State ─────────────────────────────────────────────────────────
        self._curves: list[CurveSpec] = [
            CurveSpec(d, c, color=_PALETTE[i % len(_PALETTE)])
            for i, (d, c) in enumerate(_DEFAULT_CURVES)
        ]
        self._layout_mode: str = "stacked"
        self._current_node = None
        self._bg           = None          # blitting background cache
        self._trace_lines: list = []
        self._time_cursors: list = []      # animated=True axvline objects

        # ── Root layout ───────────────────────────────────────────────────
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────
        tb = QtWidgets.QWidget()
        tbl = QtWidgets.QHBoxLayout(tb)
        tbl.setContentsMargins(0, 0, 0, 0)
        tbl.setSpacing(4)

        tbl.addWidget(QtWidgets.QLabel("Layout:"))

        self._stacked_btn = QtWidgets.QPushButton("Stacked")
        self._overlay_btn = QtWidgets.QPushButton("Overlay")
        for btn in (self._stacked_btn, self._overlay_btn):
            btn.setCheckable(True)
            btn.setFixedHeight(22)
        self._stacked_btn.setChecked(True)

        mode_grp = QtWidgets.QButtonGroup(self)
        mode_grp.setExclusive(True)
        mode_grp.addButton(self._stacked_btn)
        mode_grp.addButton(self._overlay_btn)

        self._stacked_btn.clicked.connect(lambda: self._set_layout("stacked"))
        self._overlay_btn.clicked.connect(lambda: self._set_layout("overlay"))
        tbl.addWidget(self._stacked_btn)
        tbl.addWidget(self._overlay_btn)
        tbl.addStretch(1)

        add_btn = QtWidgets.QPushButton("＋ Add curve")
        add_btn.setFixedHeight(22)
        add_btn.clicked.connect(self._add_curve)
        tbl.addWidget(add_btn)
        root.addWidget(tb)

        # ── Curve list (scrollable, compact) ──────────────────────────────
        self._curve_list_widget = QtWidgets.QWidget()
        self._curve_list_lay    = QtWidgets.QVBoxLayout(self._curve_list_widget)
        self._curve_list_lay.setContentsMargins(0, 0, 0, 0)
        self._curve_list_lay.setSpacing(2)

        curve_scroll = QtWidgets.QScrollArea()
        curve_scroll.setWidgetResizable(True)
        curve_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        curve_scroll.setMaximumHeight(62)
        curve_scroll.setWidget(self._curve_list_widget)
        root.addWidget(curve_scroll)

        # ── Title label ───────────────────────────────────────────────────
        self.title_label = QtWidgets.QLabel("No node selected")
        self.title_label.setStyleSheet("font-size: 11px; color: #404040;")
        root.addWidget(self.title_label)

        # ── Matplotlib canvas (scrollable for many stacked curves) ────────
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(220)
        self.canvas.mpl_connect("draw_event", self._on_draw)

        canvas_scroll = QtWidgets.QScrollArea()
        canvas_scroll.setWidgetResizable(True)
        canvas_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        canvas_scroll.setWidget(self.canvas)
        root.addWidget(canvas_scroll, 1)

        # ── Initial build ─────────────────────────────────────────────────
        self._rebuild_curve_list_ui()
        self.refresh("init")

    # ── Color helpers ─────────────────────────────────────────────────────

    def _next_color(self) -> str:
        used = {c.color for c in self._curves}
        for col in _PALETTE:
            if col not in used:
                return col
        return _PALETTE[len(self._curves) % len(_PALETTE)]

    # ── Curve list UI ─────────────────────────────────────────────────────

    def _rebuild_curve_list_ui(self):
        while self._curve_list_lay.count():
            item = self._curve_list_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, curve in enumerate(self._curves):
            self._curve_list_lay.addWidget(self._make_row(i, curve))

    def _make_row(self, idx: int, curve: CurveSpec) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(4)

        swatch = QtWidgets.QLabel("■")
        swatch.setStyleSheet(f"color: {curve.color}; font-size: 15px;")
        swatch.setFixedWidth(16)
        lay.addWidget(swatch)

        demand_cb = QtWidgets.QComboBox()
        for d in self._DEMANDS:
            demand_cb.addItem(d)
        demand_cb.setCurrentText(curve.demand)
        demand_cb.setFixedWidth(58)
        demand_cb.currentTextChanged.connect(
            lambda v, i=idx: self._update_curve(i, demand=v)
        )
        lay.addWidget(demand_cb)

        comp_cb = QtWidgets.QComboBox()
        for c in self._COMPONENTS:
            comp_cb.addItem(c.upper(), c)
        comp_cb.setCurrentText(curve.component.upper())
        comp_cb.currentIndexChanged.connect(
            lambda _, cb=comp_cb, i=idx: self._update_curve(i, component=cb.currentData())
        )
        comp_cb.setFixedWidth(46)
        lay.addWidget(comp_cb)

        rm = QtWidgets.QPushButton("✕")
        rm.setFixedSize(22, 22)
        rm.setStyleSheet(
            "QPushButton { color: #c0392b; border: none; font-weight: bold; }"
            "QPushButton:hover { color: #e74c3c; }"
        )
        rm.clicked.connect(lambda _, i=idx: self._remove_curve(i))
        lay.addWidget(rm)

        return row

    # ── Curve CRUD ────────────────────────────────────────────────────────

    def _add_curve(self):
        self._curves.append(CurveSpec("accel", "z", color=self._next_color()))
        self._rebuild_curve_list_ui()
        self._rebuild_figure()

    def _remove_curve(self, idx: int):
        if 0 <= idx < len(self._curves):
            self._curves.pop(idx)
            self._rebuild_curve_list_ui()
            self._rebuild_figure()

    def _update_curve(self, idx: int, *, demand: str | None = None, component: str | None = None):
        if 0 <= idx < len(self._curves):
            if demand is not None:
                self._curves[idx].demand = demand
            if component is not None:
                self._curves[idx].component = component
            self._rebuild_figure()

    def _set_layout(self, mode: str):
        self._layout_mode = mode
        self._rebuild_figure()

    # ── Figure helpers ────────────────────────────────────────────────────

    def _rebuild_figure(self):
        """Redraw figure from scratch using the current curves + node."""
        self.figure.clear()
        self._trace_lines = []
        self._time_cursors = []
        self._bg = None

        if self._current_node is None or not self._curves:
            self.canvas.draw_idle()
            return
        self._draw_curves(self._current_node)

    def _draw_curves(self, node_id):
        """Load traces and paint all curves into the figure."""
        self.figure.clear()
        self._trace_lines = []
        self._time_cursors = []
        self._bg = None

        t         = self.session.adapter.time
        current_t = self.session.current_time()

        # Load demand data — one HDF5 call per unique demand.
        traces: dict[str, _np.ndarray | None] = {}
        for curve in self._curves:
            if curve.demand not in traces:
                try:
                    traces[curve.demand] = self.session.adapter.trace(node_id, curve.demand)
                except Exception:
                    traces[curve.demand] = None

        n = len(self._curves)

        if self._layout_mode == "stacked" and n > 0:
            # ── Stacked: one subplot per curve ────────────────────────────
            self.canvas.setMinimumHeight(max(220, n * 110))
            raw = self.figure.subplots(n, 1, sharex=True, squeeze=False)
            axes = list(raw.flatten())

            for ax, curve in zip(axes, self._curves):
                data = traces.get(curve.demand)
                ax.set_ylabel(curve.label, fontsize=8)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if data is None:
                    continue
                row = _COMP_ROW[curve.component]
                (line,) = ax.plot(
                    t, data[row], linewidth=1.0, color=curve.color, label=curve.label
                )
                cursor = ax.axvline(
                    current_t, color="tab:red", alpha=0.35, linewidth=1.0, animated=True
                )
                self._trace_lines.append(line)
                self._time_cursors.append(cursor)

            if axes:
                axes[-1].set_xlabel("Time [s]", fontsize=8)

        else:
            # ── Overlay: single subplot, all curves ───────────────────────
            self.canvas.setMinimumHeight(220)
            ax = self.figure.subplots(1, 1)
            ax.set_xlabel("Time [s]", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

            for curve in self._curves:
                data = traces.get(curve.demand)
                if data is None:
                    continue
                row = _COMP_ROW[curve.component]
                (line,) = ax.plot(
                    t, data[row], linewidth=1.0, color=curve.color, label=curve.label
                )
                self._trace_lines.append(line)

            if self._trace_lines:
                ax.legend(loc="upper right", fontsize=8)

            # Single shared cursor for the overlay axes.
            cursor = ax.axvline(
                current_t, color="tab:red", alpha=0.35, linewidth=1.0, animated=True
            )
            self._time_cursors.append(cursor)

        self.title_label.setText(f"Node {node_id}")
        # Full draw → _on_draw caches the blitting background.
        self.canvas.draw()

    # ── Blitting ──────────────────────────────────────────────────────────

    def _on_draw(self, event):
        """Cache static background after every full canvas.draw() for blitting."""
        if not self._time_cursors:
            return
        try:
            self._bg = self.canvas.copy_from_bbox(self.figure.bbox)
            t = self.session.current_time()
            for ax, cur in zip(self.figure.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
        except Exception:
            self._bg = None

    def _blit_cursors(self, t: float) -> bool:
        """Move all cursors to *t* using blitting.  Returns True on success."""
        if self._bg is None or not self._time_cursors:
            return False
        try:
            self.canvas.restore_region(self._bg)
            for ax, cur in zip(self.figure.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
            return True
        except Exception:
            self._bg = None
            return False

    # ── Refresh ───────────────────────────────────────────────────────────

    def refresh(self, reason: str = "full"):
        # ── Time: blit-only cursor move (zero I/O) ────────────────────────
        if reason == "time":
            if self._current_node is not None and self._time_cursors:
                t = self.session.current_time()
                if not self._blit_cursors(t):
                    for cur in self._time_cursors:
                        cur.set_xdata([t, t])
                    self.canvas.draw_idle()
            return

        # ── Responses don't depend on demand/component/warp changes ───────
        if reason not in {"selection", "full", "init"}:
            return

        node_id = self.session.state.selected_node

        if node_id is None:
            self._current_node = None
            self._trace_lines  = []
            self._time_cursors = []
            self._bg           = None
            self.figure.clear()
            self.title_label.setText("No node selected")
            self.canvas.draw_idle()
            return

        self._current_node = node_id
        self._draw_curves(node_id)


_DEFAULT_RESPONSE_CHECKS = {
    ("accel", "z"): True,
    ("accel", "e"): True,
    ("accel", "n"): True,
    ("vel", "z"): True,
    ("vel", "e"): True,
    ("vel", "n"): True,
    ("disp", "z"): True,
    ("disp", "e"): True,
    ("disp", "n"): True,
}


class ResponsesPanel(QtWidgets.QWidget):
    """Fixed multi-response viewer driven by grouped checkboxes."""

    _DEMANDS = ("accel", "vel", "disp")
    _COMPONENTS = ("z", "e", "n")
    _GROUP_TITLES = {
        "accel": "Acceleration",
        "vel": "Velocity",
        "disp": "Displacement",
    }
    _GROUP_ORDER = ("accel", "vel", "disp")
    _CURVE_STYLES = {
        ("accel", "z"): (_PALETTE[0], "-", "Acceleration Z"),
        ("accel", "e"): (_PALETTE[1], "-", "Acceleration E"),
        ("accel", "n"): (_PALETTE[2], "-", "Acceleration N"),
        ("vel", "z"): (_PALETTE[3], "-", "Velocity Z"),
        ("vel", "e"): (_PALETTE[4], "-", "Velocity E"),
        ("vel", "n"): (_PALETTE[5], "-", "Velocity N"),
        ("disp", "z"): (_PALETTE[6], "-", "Displacement Z"),
        ("disp", "e"): (_PALETTE[7], "-", "Displacement E"),
        ("disp", "n"): (_PALETTE[8], "-", "Displacement N"),
    }

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._layout_mode: str = "stacked"
        self._current_node = None
        self._bg = None
        self._time_cursors: list = []
        self._active_checks = dict(_DEFAULT_RESPONSE_CHECKS)
        self._checkboxes: dict[tuple[str, str], QtWidgets.QCheckBox] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.title_label.setStyleSheet("font-size: 11px; color: #1e3558; font-weight: 600;")
        self.title_label.setContentsMargins(2, 2, 2, 0)
        root.addWidget(self.title_label)

        layout_box = QtWidgets.QGroupBox("Layout")
        layout_box_lay = QtWidgets.QVBoxLayout(layout_box)
        layout_box_lay.setContentsMargins(6, 8, 6, 6)
        layout_box_lay.setSpacing(5)

        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(4)

        self._stacked_btn = QtWidgets.QPushButton("Stacked")
        self._overlay_btn = QtWidgets.QPushButton("Overlay")
        for btn in (self._stacked_btn, self._overlay_btn):
            btn.setCheckable(True)
            btn.setFixedHeight(22)
        self._stacked_btn.setChecked(True)

        mode_group = QtWidgets.QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self._stacked_btn)
        mode_group.addButton(self._overlay_btn)
        self._stacked_btn.clicked.connect(lambda: self._set_layout("stacked"))
        self._overlay_btn.clicked.connect(lambda: self._set_layout("overlay"))

        toolbar_layout.addWidget(self._stacked_btn)
        toolbar_layout.addWidget(self._overlay_btn)
        toolbar_layout.addStretch(1)
        layout_box_lay.addWidget(toolbar)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_scroll.setMinimumHeight(88)
        controls_scroll.setMaximumHeight(96)

        controls_host = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)

        for demand in self._GROUP_ORDER:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(6, 2, 6, 2)
            row_layout.setSpacing(6)

            title = QtWidgets.QLabel(self._GROUP_TITLES[demand])
            title.setMinimumWidth(78)
            title.setStyleSheet("font-weight: 600;")
            row_layout.addWidget(title)

            for component in self._COMPONENTS:
                key = (demand, component)
                checkbox = QtWidgets.QCheckBox(component.upper())
                checkbox.setChecked(self._active_checks[key])
                checkbox.toggled.connect(
                    lambda checked, d=demand, c=component: self._set_curve_enabled(d, c, checked)
                )
                self._checkboxes[key] = checkbox
                row_layout.addWidget(checkbox)

            row_layout.addStretch(1)

            controls_layout.addWidget(row)

        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls_host)
        layout_box_lay.addWidget(controls_scroll)
        root.addWidget(layout_box, 0)

        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(220)
        self.canvas.mpl_connect("draw_event", self._on_draw)

        plot_box = QtWidgets.QGroupBox("Plots")
        plot_layout = QtWidgets.QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(6, 8, 6, 6)

        canvas_scroll = QtWidgets.QScrollArea()
        canvas_scroll.setWidgetResizable(True)
        canvas_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        canvas_scroll.setWidget(self.canvas)
        plot_layout.addWidget(canvas_scroll)
        root.addWidget(plot_box, 1)

        self.refresh("init")

    def _set_curve_enabled(self, demand: str, component: str, enabled: bool):
        self._active_checks[(demand, component)] = bool(enabled)
        self._rebuild_figure()

    def _set_layout(self, mode: str):
        self._layout_mode = mode
        self._rebuild_figure()

    def _active_curve_keys(self) -> list[tuple[str, str]]:
        return [
            (demand, component)
            for demand in self._GROUP_ORDER
            for component in self._COMPONENTS
            if self._active_checks.get((demand, component), False)
        ]

    def _rebuild_figure(self):
        self.figure.clear()
        self._time_cursors = []
        self._bg = None

        if self._current_node is None or not self._active_curve_keys():
            self.canvas.draw_idle()
            return
        self._draw_curves(self._current_node)

    def _draw_curves(self, node_id):
        self.figure.clear()
        self._time_cursors = []
        self._bg = None

        active_keys = self._active_curve_keys()
        if not active_keys:
            self.title_label.setText(f"Node {node_id} (no active curves)")
            self.canvas.draw_idle()
            return

        t = self.session.adapter.time
        current_t = self.session.current_time()
        traces: dict[str, _np.ndarray | None] = {}
        for demand, _component in active_keys:
            if demand not in traces:
                try:
                    traces[demand] = self.session.adapter.trace(node_id, demand)
                except Exception:
                    traces[demand] = None

        n = len(active_keys)
        if self._layout_mode == "stacked":
            self.canvas.setMinimumHeight(max(220, n * 110))
            raw = self.figure.subplots(n, 1, sharex=True, squeeze=False)
            axes = list(raw.flatten())

            for ax, (demand, component) in zip(axes, active_keys):
                color, linestyle, label = self._CURVE_STYLES[(demand, component)]
                data = traces.get(demand)
                ax.set_ylabel(label, fontsize=8)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if data is not None:
                    row = _COMP_ROW[component]
                    ax.plot(t, data[row], linewidth=1.0, color=color, linestyle=linestyle)
                cursor = ax.axvline(
                    current_t, color="tab:red", alpha=0.35, linewidth=1.0, animated=True
                )
                self._time_cursors.append(cursor)

            if axes:
                axes[-1].set_xlabel("Time [s]", fontsize=8)
        else:
            self.canvas.setMinimumHeight(220)
            ax = self.figure.subplots(1, 1)
            ax.set_xlabel("Time [s]", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

            any_line = False
            for demand, component in active_keys:
                color, linestyle, label = self._CURVE_STYLES[(demand, component)]
                data = traces.get(demand)
                if data is None:
                    continue
                row = _COMP_ROW[component]
                ax.plot(
                    t,
                    data[row],
                    linewidth=1.0,
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
                any_line = True

            if any_line:
                ax.legend(loc="upper right", fontsize=8)

            cursor = ax.axvline(
                current_t, color="tab:red", alpha=0.35, linewidth=1.0, animated=True
            )
            self._time_cursors.append(cursor)

        self.title_label.setText(f"Node {node_id}")
        self.canvas.draw()

    def _on_draw(self, event):
        if not self._time_cursors:
            return
        try:
            self._bg = self.canvas.copy_from_bbox(self.figure.bbox)
            t = self.session.current_time()
            for ax, cur in zip(self.figure.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
        except Exception:
            self._bg = None

    def _blit_cursors(self, t: float) -> bool:
        if self._bg is None or not self._time_cursors:
            return False
        try:
            self.canvas.restore_region(self._bg)
            for ax, cur in zip(self.figure.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
            return True
        except Exception:
            self._bg = None
            return False

    def refresh(self, reason: str = "full"):
        if reason == "time":
            if self._current_node is not None and self._time_cursors:
                t = self.session.current_time()
                if not self._blit_cursors(t):
                    for cur in self._time_cursors:
                        cur.set_xdata([t, t])
                    self.canvas.draw_idle()
            return

        if reason not in {"selection", "full", "init"}:
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_node = None
            self._time_cursors = []
            self._bg = None
            self.figure.clear()
            self.title_label.setText("No node selected")
            self.canvas.draw_idle()
            return

        self._current_node = node_id
        self._draw_curves(node_id)


# ── Green Function panel colours & display limits ─────────────────────────────
_GF_DIAG_COLOR       = "#1565C0"  # blue  – diagonal terms (G_ZZ, G_NN, G_EE)
_GF_OFFDIAG_COLOR    = "#9e9e9e"  # gray  – coupling terms
_GF_OFFDIAG_ALPHA    = 0.55       # slight transparency so off-diag reads softer
_GF_MAX_PLOT_SAMPLES = 2000       # downsample display when nt_gf exceeds this


class GFPanel(QtWidgets.QWidget):
    """9-row × 1-column Green Function tensor viewer for the selected node.

    Rows
    ----
    G_ZZ, G_ZN, G_ZE  (receiver Z, source Z / N / E)
    G_NZ, G_NN, G_NE  (receiver N, source Z / N / E)
    G_EZ, G_EN, G_EE  (receiver E, source Z / N / E)

    When the model returns only 1-D GF data (diagonal only), three rows are
    shown: G_ZZ, G_NN, G_EE.

    Visual style
    ------------
    Diagonal terms (G_ZZ, G_NN, G_EE)  →  blue  ``#1565C0``,   α = 1.0
    Off-diagonal coupling terms         →  gray  ``#9e9e9e``,   α = 0.55
    Time cursor                         →  red axvline, animated (blitting)
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None
        self._current_subfault: int = 0
        self._time_cursors: list = []
        self._bg = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Subfault selector ─────────────────────────────────────────────────
        ctrl_row = QtWidgets.QWidget()
        ctrl_lay = QtWidgets.QHBoxLayout(ctrl_row)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_lay.setSpacing(6)
        ctrl_lay.addWidget(QtWidgets.QLabel("Subfault:"))

        self._sf_spin = QtWidgets.QSpinBox()
        self._sf_spin.setMinimum(0)
        self._sf_spin.setMaximum(max(session.gf_subfault_count() - 1, 0))
        self._sf_spin.setWrapping(True)
        self._sf_spin.setValue(0)
        self._sf_spin.valueChanged.connect(self._on_subfault_changed)
        ctrl_lay.addWidget(self._sf_spin)
        ctrl_lay.addStretch(1)
        root.addWidget(ctrl_row)

        # ── Title label ───────────────────────────────────────────────────────
        self.title_label = QtWidgets.QLabel("No node selected")
        self.title_label.setStyleSheet("font-size: 11px; color: #404040;")
        root.addWidget(self.title_label)

        # ── Matplotlib canvas ─────────────────────────────────────────────────
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(900)  # 9 rows × ~100 px each
        self.canvas.mpl_connect("draw_event", self._on_draw)

        canvas_scroll = QtWidgets.QScrollArea()
        canvas_scroll.setWidgetResizable(True)
        canvas_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        canvas_scroll.setWidget(self.canvas)
        root.addWidget(canvas_scroll, 1)

        self.refresh("init")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _on_subfault_changed(self, val: int):
        self._current_subfault = int(val)
        if self._current_node is not None:
            self._draw_tensor(self._current_node, self._current_subfault)

    def _draw_tensor(self, node_id, subfault_id: int):
        """Rebuild all nine subplots and install blitted cursors."""
        self.figure.clear()
        self._time_cursors = []
        self._bg = None

        tensor = self.session.current_gf_tensor(subfault_id)
        if tensor is None:
            self.title_label.setText(f"Node {node_id} – GF not available")
            self.canvas.draw_idle()
            return

        time_arr = tensor["time"]
        gf_rows = tensor["rows"]
        n = len(gf_rows)
        if n == 0:
            self.title_label.setText(f"Node {node_id} – no GF data")
            self.canvas.draw_idle()
            return

        # Resize canvas height to accommodate all rows comfortably.
        self.canvas.setMinimumHeight(max(900, n * 100))
        raw_axes = self.figure.subplots(n, 1, sharex=True, squeeze=False)
        axes = list(raw_axes.flatten())

        current_t = self.session.current_time()

        # Compute display stride once — shared across all 9 rows.
        nt = len(time_arr)
        if nt > _GF_MAX_PLOT_SAMPLES:
            _stride = max(1, nt // _GF_MAX_PLOT_SAMPLES)
            _t_plot = time_arr[::_stride]
        else:
            _stride = 1
            _t_plot = time_arr

        for ax, (label, is_diag, data) in zip(axes, gf_rows):
            color = _GF_DIAG_COLOR if is_diag else _GF_OFFDIAG_COLOR
            alpha = 1.0 if is_diag else _GF_OFFDIAG_ALPHA
            # Downsampled line for rendering — cursor still spans the full
            # x-axis range (set by xlim from the shared time_arr).
            ax.plot(
                _t_plot,
                data[::_stride],
                linewidth=0.9, color=color, alpha=alpha,
            )
            # Rotate label 0° so it fits cleanly in the narrow panel; use
            # labelpad to push it left of the y-axis tick numbers.
            ax.set_ylabel(label, fontsize=7, rotation=0,
                          labelpad=36, va="center")
            ax.grid(True, alpha=0.20)
            ax.tick_params(labelsize=6)
            cursor = ax.axvline(
                current_t, color="tab:red",
                alpha=0.35, linewidth=1.0, animated=True,
            )
            self._time_cursors.append(cursor)

        if axes:
            axes[-1].set_xlabel("Time [s]", fontsize=8)

        self.title_label.setText(
            f"Node {node_id}  |  subfault {subfault_id}"
        )
        self.canvas.draw()   # triggers _on_draw → captures blitting background

    def _on_draw(self, event):
        """Capture the static background immediately after a full redraw."""
        if not self._time_cursors:
            return
        try:
            self._bg = self.canvas.copy_from_bbox(self.figure.bbox)
            t = self.session.current_time()
            for ax, cur in zip(self.figure.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
        except Exception:
            self._bg = None

    def _blit_cursors(self, t: float) -> bool:
        """Move all cursors to time *t* using the cached background (< 1 ms)."""
        if self._bg is None or not self._time_cursors:
            return False
        try:
            self.canvas.restore_region(self._bg)
            for ax, cur in zip(self.figure.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
            return True
        except Exception:
            self._bg = None
            return False

    # ── Public refresh API (matches _LazyPage contract) ───────────────────────

    def refresh(self, reason: str = "full"):
        if reason == "time":
            if self._current_node is not None and self._time_cursors:
                t = self.session.current_time()
                if not self._blit_cursors(t):
                    for cur in self._time_cursors:
                        cur.set_xdata([t, t])
                    self.canvas.draw_idle()
            return

        if reason not in {"selection", "full", "init"}:
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_node = None
            self._time_cursors = []
            self._bg = None
            self.figure.clear()
            self.title_label.setText("No node selected")
            self.canvas.draw_idle()
            return

        self._current_node = node_id
        self._draw_tensor(node_id, self._current_subfault)


__all__ = [
    "CombinedTracePanel",
    "TracePanel",
    "SpectrumPanel",
    "AriasIntensityPanel",
    "ResponsesPanel",
    "GFPanel",
    "CurveSpec",
]
