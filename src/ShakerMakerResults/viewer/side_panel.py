"""Right-side panel for the interactive viewer.

Layout
------
A narrow vertical nav-menu (84 px) selects pages in a ``QStackedWidget``.
Only the active page is ever visible.

Performance contract
--------------------
Lightweight pages (Node → Warp)
    Built at startup.  Refreshed unconditionally on every ``refresh()``
    call except ``"time"`` — they are pure Qt-label / combo updates.

Heavy pages (Responses, GF — Matplotlib canvases)
    Built lazily on first navigation.  Refreshed **only** when active.

    ``Responses`` contains three lazy tabs: Time History, Spectrum and Arias.
    *Dirty tracking*: when a ``"selection"`` or ``"full"`` update arrives
    while a heavy page is *not* active, that page is flagged dirty.  The
    next time the user navigates to it, it receives a ``"full"`` refresh
    before becoming visible, so it always shows current data.

    This means Spectrum and Arias never compute Newmark or Arias-intensity
    unless the user actually visits those pages.

``"time"`` updates
    Only the active heavy page (trace cursors) is touched, throttled to
    one redraw per 120 ms during playback.
"""

from __future__ import annotations

import time

from ._imports import require_viewer_dependencies
from .busy_dialog import BusyDialog
from .colors import COLORMAP_OPTIONS, colormap_for_component
from .icons import icon
from .session import VALID_STATIC_COLOR_BY
from .theme import LIGHT_PALETTE
from .trace_panel import AriasIntensityPanel, GFPanel, ResponsesPanel, SpectrumPanel

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


# ── Menu configuration ────────────────────────────────────────────────────────

_MENU_ENTRIES: list[tuple[str, str, str]] = [
    ("Node",       "nav_node",       "Node"),
    ("Display",    "nav_display",    "Display"),
    ("Visibility", "nav_visibility", "Vis."),
    ("Warp",       "nav_warp",       "Warp"),
    # ── separator ── analysis pages below ──
    ("Responses",  "nav_responses",  "Resp."),
    ("GF",         "nav_gf",         "GF"),
]

# Keys whose pages are heavy (lazy-created; refresh only when active).
_HEAVY_KEYS: frozenset[str] = frozenset({"Responses", "GF"})

# Reasons that make a heavy page's content stale if it wasn't active.
_STALE_REASONS: frozenset[str] = frozenset({"selection", "full"})

_TRACE_TIME_REFRESH_INTERVAL_S = 0.08


# ── Lazy page wrapper ─────────────────────────────────────────────────────────

class _LazyPage(QtWidgets.QWidget):
    """Delays construction of a heavy widget until first navigation.

    ``ensure_created()`` triggers the factory and embeds the result.
    ``refresh()`` is a no-op until the widget is created.
    """

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self._factory = factory
        self._widget: QtWidgets.QWidget | None = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lay = lay

    def ensure_created(self) -> QtWidgets.QWidget:
        if self._widget is None:
            self._widget = self._factory()
            self._lay.addWidget(self._widget, 1)
        return self._widget

    def refresh(self, reason: str):
        if self._widget is not None:
            self._widget.refresh(reason)

    @property
    def is_created(self) -> bool:
        return self._widget is not None


class _PageScrollArea(QtWidgets.QScrollArea):
    """Per-page vertical scroll container that respects the active panel width."""

    def __init__(self, page: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        page.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setWidget(page)


class _ResponsesAnalysisTabs(QtWidgets.QWidget):
    """Tabbed analysis page: time histories, response spectra and Arias."""

    _TAB_KEYS = ("Time History", "Spectrum", "Arias")

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._dirty_tabs: set[str] = set()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tabs)

        self._pages: dict[str, _LazyPage] = {
            "Time History": _LazyPage(lambda: ResponsesPanel(session)),
            "Spectrum": _LazyPage(lambda: SpectrumPanel(session)),
            "Arias": _LazyPage(lambda: AriasIntensityPanel(session)),
        }

        for key in self._TAB_KEYS:
            self._tabs.addTab(self._pages[key], key)

        self._ensure_current_tab()

    def refresh(self, reason: str):
        active_key = self._active_key()
        if reason == "time":
            self._pages[active_key].refresh("time")
            return

        if reason in _STALE_REASONS:
            for key in self._TAB_KEYS:
                if key != active_key:
                    self._dirty_tabs.add(key)

        page = self._pages[active_key]
        page.ensure_created()
        page.refresh(reason)
        self._dirty_tabs.discard(active_key)

    def _on_tab_changed(self, _index: int):
        key = self._active_key()
        page = self._pages[key]
        already_existed = page.is_created
        page.ensure_created()
        if already_existed and key in self._dirty_tabs:
            page.refresh("full")
        elif key in self._dirty_tabs:
            page.refresh("full")
        self._dirty_tabs.discard(key)

    def _ensure_current_tab(self):
        self._pages[self._active_key()].ensure_created()

    def _active_key(self) -> str:
        key = self._tabs.tabText(self._tabs.currentIndex())
        return key if key in self._pages else "Time History"


# ── Shared base for Apply-button sections ─────────────────────────────────────

class _SectionBase(QtWidgets.QWidget):

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._syncing = False
        self._dirty = False
        self.apply_button: QtWidgets.QPushButton | None = None

    def _run_heavy(self, fn, message: str = "Updating scene…"):
        """Show a 'Ladruno working' overlay while *fn()* blocks the event loop.

        Calls ``processEvents()`` once after showing the dialog so the OS has
        time to paint it before the heavy work starts.
        """
        dlg = BusyDialog(message, self.window())
        dlg.show()
        QtWidgets.QApplication.processEvents()
        try:
            fn()
        finally:
            dlg.close()

    def _make_apply_button(self, callback, message: str = "Updating scene…") -> QtWidgets.QPushButton:
        """Create a disabled Apply button that shows the busy overlay on click."""
        def _wrapped():
            self._run_heavy(callback, message)

        btn = QtWidgets.QPushButton("Apply")
        btn.setEnabled(False)
        btn.clicked.connect(_wrapped)
        self.apply_button = btn
        return btn

    def _set_dirty(self, dirty: bool = True):
        if self._syncing:
            return
        self._dirty = bool(dirty)
        if self.apply_button is not None:
            self.apply_button.setEnabled(self._dirty)

    def _clear_dirty(self):
        self._dirty = False
        if self.apply_button is not None:
            self.apply_button.setEnabled(False)

    @staticmethod
    def _set_combo(combo, value):
        b = combo.blockSignals(True)
        combo.setCurrentText(str(value))
        combo.blockSignals(b)

    @staticmethod
    def _set_combo_data(combo, value):
        idx = combo.findData(value)
        if idx < 0:
            idx = combo.findText(str(value))
        if idx < 0:
            return
        b = combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(b)

    @staticmethod
    def _coord_spin() -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(3)
        s.setRange(-1e12, 1e12)
        s.setSingleStep(1.0)
        return s

    @staticmethod
    def _value_spin() -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(8)
        s.setRange(-1e20, 1e20)
        s.setSingleStep(0.1)
        return s

    @staticmethod
    def _fmt_xyz(xyz) -> str:
        return "-" if xyz is None else f"{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}"


# ── Lightweight section widgets ───────────────────────────────────────────────

class _StationTable(QtWidgets.QTableWidget):
    """Editable station table with tab-separated clipboard paste support."""

    HEADERS = ("Station", "coord x", "coord y", "coord z")

    def __init__(self, parent=None):
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(list(self.HEADERS))
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(120)
        self.setMaximumHeight(140)
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)

    def keyPressEvent(self, event):  # noqa: N802
        if event.matches(QtGui.QKeySequence.Paste):
            self._paste_clipboard()
            return
        super().keyPressEvent(event)

    def _paste_clipboard(self):
        text = QtWidgets.QApplication.clipboard().text()
        if not text.strip():
            return
        rows = [line.split("\t") for line in text.splitlines() if line.strip()]
        if not rows:
            return
        start_row = max(self.currentRow(), 0)
        start_col = max(self.currentColumn(), 0)
        while self.rowCount() < start_row + len(rows):
            self.insertRow(self.rowCount())
        for r_offset, values in enumerate(rows):
            for c_offset, value in enumerate(values[: self.columnCount() - start_col]):
                self.setItem(
                    start_row + r_offset,
                    start_col + c_offset,
                    QtWidgets.QTableWidgetItem(value.strip()),
                )

    def rows_as_station_entries(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for row in range(self.rowCount()):
            cells = []
            for col in range(self.columnCount()):
                item = self.item(row, col)
                cells.append("" if item is None else item.text().strip())
            if not any(cells):
                continue
            try:
                xyz_model_m = (float(cells[1]), float(cells[2]), float(cells[3]))
            except ValueError:
                continue
            entries.append(
                {
                    "name": cells[0] or f"S{len(entries) + 1}",
                    "xyz_display_m": xyz_model_m,
                }
            )
        return entries


class NodeSearchSection(_SectionBase):
    """Selected-node info + coordinate search."""

    def __init__(self, session, parent=None):
        super().__init__(session, parent)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        form_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)

        self.node_id_spin = QtWidgets.QSpinBox()
        self.node_id_spin.setMinimum(0)
        self.node_id_spin.setMaximum(max(len(session.adapter.node_ids) - 1, 0))
        self.node_select_btn = QtWidgets.QPushButton("Select")
        self.node_select_btn.clicked.connect(self._select_by_id)

        node_row = QtWidgets.QWidget()
        nrl = QtWidgets.QHBoxLayout(node_row)
        nrl.setContentsMargins(0, 0, 0, 0)
        nrl.addWidget(self.node_id_spin, 1)
        nrl.addWidget(self.node_select_btn)

        self.x_spin = _SectionBase._coord_spin()
        self.y_spin = _SectionBase._coord_spin()
        self.z_spin = _SectionBase._coord_spin()
        self.find_btn = QtWidgets.QPushButton("Find nearest")
        self.find_btn.clicked.connect(self._find_nearest)
        self.nearest_lbl = QtWidgets.QLabel("-")

        self.node_lbl      = QtWidgets.QLabel("None")
        self.type_lbl      = QtWidgets.QLabel("-")
        self.xyz_model_lbl = QtWidgets.QLabel("-")
        self.xyz_view_lbl  = QtWidgets.QLabel("-")
        self.gf_slot_lbl   = QtWidgets.QLabel("-")

        form.addRow("Node ID",       node_row)
        form.addRow("X [m]",         self.x_spin)
        form.addRow("Y [m]",         self.y_spin)
        form.addRow("Z [m]",         self.z_spin)
        form.addRow("",              self.find_btn)
        form.addRow("Nearest",       self.nearest_lbl)
        form.addRow("Selected",      self.node_lbl)
        form.addRow("Type",          self.type_lbl)
        form.addRow("XYZ model [m]", self.xyz_model_lbl)
        form.addRow("XYZ view [m]",  self.xyz_view_lbl)
        form.addRow("GF slot s0",    self.gf_slot_lbl)
        outer.addWidget(form_widget)

        station_box = QtWidgets.QGroupBox("Stations")
        station_layout = QtWidgets.QVBoxLayout(station_box)
        station_layout.setContentsMargins(6, 6, 6, 6)
        station_layout.setSpacing(6)

        self.station_table = _StationTable()
        station_layout.addWidget(self.station_table)

        station_btn_row = QtWidgets.QWidget()
        sbr = QtWidgets.QHBoxLayout(station_btn_row)
        sbr.setContentsMargins(0, 0, 0, 0)
        sbr.setSpacing(4)
        add_row_btn = QtWidgets.QPushButton("Add row")
        remove_row_btn = QtWidgets.QPushButton("Remove selected")
        add_row_btn.clicked.connect(self._add_station_row)
        remove_row_btn.clicked.connect(self._remove_station_rows)
        sbr.addWidget(add_row_btn)
        sbr.addWidget(remove_row_btn)
        sbr.addStretch(1)
        station_layout.addWidget(station_btn_row)

        station_help = QtWidgets.QLabel("Paste viewer-space coordinates using Station, coord x, coord y, coord z.")
        station_help.setWordWrap(True)
        station_help.setStyleSheet("color: #666; font-size: 11px;")
        station_layout.addWidget(station_help)

        outer.addWidget(station_box)
        outer.addWidget(self._make_apply_button(self._apply_stations, "Plotting station tags…"))
        outer.addStretch(1)

        self.station_table.itemChanged.connect(lambda *_: self._set_dirty())
        for _ in range(3):
            self._add_station_row(mark_dirty=False)
        self.refresh("init")

    def refresh(self, _reason: str = "full"):
        self._syncing = True
        try:
            if _reason == "init":
                self._load_station_rows(self.session.current_station_tags())
                self._clear_dirty()
            info = self.session.current_node_info()
            if info is None:
                self.node_lbl.setText("None")
                for lbl in (self.type_lbl, self.xyz_model_lbl,
                            self.xyz_view_lbl, self.gf_slot_lbl):
                    lbl.setText("-")
                return
            nid = info["node_id"]
            if nid not in ("QA", "qa"):
                self.node_id_spin.setValue(int(nid))
            xyz_m = info.get("xyz_model_m")
            self.node_lbl.setText(str(nid))
            self.type_lbl.setText(str(info["type"]))
            self.xyz_model_lbl.setText(_SectionBase._fmt_xyz(xyz_m))
            self.xyz_view_lbl.setText(_SectionBase._fmt_xyz(info.get("xyz_m")))
            self.gf_slot_lbl.setText(
                "-" if info["gf_slot_s0"] is None else str(info["gf_slot_s0"])
            )
            if xyz_m is not None:
                self.x_spin.setValue(float(xyz_m[0]))
                self.y_spin.setValue(float(xyz_m[1]))
                self.z_spin.setValue(float(xyz_m[2]))
        finally:
            self._syncing = False

    def _select_by_id(self):
        if not self._syncing:
            self.session.select_node(int(self.node_id_spin.value()))

    def _find_nearest(self):
        if self._syncing:
            return
        nid, dist = self.session.select_nearest_display_coordinate_m(
            self.x_spin.value(), self.y_spin.value(), self.z_spin.value()
        )
        self.nearest_lbl.setText(f"{nid} | dist={dist:.3f} m")

    def _add_station_row(self, *, mark_dirty: bool = True):
        self.station_table.insertRow(self.station_table.rowCount())
        if mark_dirty:
            self._set_dirty(True)

    def _remove_station_rows(self):
        rows = sorted({idx.row() for idx in self.station_table.selectedIndexes()}, reverse=True)
        if not rows and self.station_table.rowCount():
            rows = [self.station_table.rowCount() - 1]
        for row in rows:
            self.station_table.removeRow(row)
        self._set_dirty(True)

    def _apply_stations(self):
        self.session.set_station_tags(self.station_table.rows_as_station_entries())
        self._clear_dirty()

    def _load_station_rows(self, stations: list[dict[str, object]]):
        old = self.station_table.blockSignals(True)
        try:
            self.station_table.setRowCount(0)
            for station in stations:
                row = self.station_table.rowCount()
                self.station_table.insertRow(row)
                xyz = station.get("xyz_display_m", station.get("xyz_model_m", ("", "", "")))
                values = [station.get("name", ""), *xyz]
                for col, value in enumerate(values):
                    self.station_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))
            while self.station_table.rowCount() < 3:
                self.station_table.insertRow(self.station_table.rowCount())
        finally:
            self.station_table.blockSignals(old)


class DisplaySection(_SectionBase):
    """Merged Field + Color Map panel — one Apply triggers one 3-D rebuild.

    Combines what was previously *DataSection* (demand / component) and
    *ColorMapSection* (colormap / color range) into a single panel that calls
    :meth:`~.session.ViewerSession.apply_display_settings` atomically.

    Layout
    ------
    Field
      Demand:      [ accel ▼ ]
      Component:   [ resultant ▼ ]

    Color Map
      Preset:      [ RdBu_r ▼ ]
      Auto range:  -1.23 / 1.23
      User min:    [ ... ]
      User max:    [ ... ]
      [ ] Clamp to range
      [ Reset to auto ]

    [ Apply ]
    """

    _REFRESH_REASONS = frozenset({
        "init", "full", "demand", "component",
        "appearance", "color_range", "panel_apply",
    })
    _STATIC_COLOR_REFRESH_REASONS = frozenset({
        "init", "full", "static_color", "playback",
    })
    _STATIC_COLOR_LABELS = {
        "elevation_z": "Elevation Z",
    }

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self._static_color_dirty = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # ── Field group ───────────────────────────────────────────────────
        field_box = QtWidgets.QGroupBox("Field")
        field_form = QtWidgets.QFormLayout(field_box)
        field_form.setContentsMargins(6, 4, 6, 6)

        self.demand_combo = QtWidgets.QComboBox()
        for demand_value, demand_label in session.adapter.display_demand_options():
            self.demand_combo.addItem(demand_label, demand_value)
        self.demand_combo.currentIndexChanged.connect(self._on_demand_changed)

        self.component_combo = QtWidgets.QComboBox()
        self.component_combo.currentIndexChanged.connect(lambda *_: self._set_dirty())
        self._populate_component_combo(self.demand_combo.currentData() or self.session.state.demand)
        self.gf_subfault_spin = QtWidgets.QSpinBox()
        self.gf_subfault_spin.setMinimum(0)
        self.gf_subfault_spin.setMaximum(max(session.gf_subfault_count() - 1, 0))
        self.gf_subfault_spin.setWrapping(True)
        self.gf_subfault_spin.valueChanged.connect(lambda *_: self._set_dirty())

        field_form.addRow("Demand",    self.demand_combo)
        field_form.addRow("Component", self.component_combo)
        field_form.addRow("Subfault",  self.gf_subfault_spin)
        outer.addWidget(field_box)

        # ── Color Map group ───────────────────────────────────────────────
        color_box = QtWidgets.QGroupBox("Color Map")
        color_form = QtWidgets.QFormLayout(color_box)
        color_form.setContentsMargins(6, 4, 6, 6)

        self.cmap_combo = QtWidgets.QComboBox()
        for cmap in COLORMAP_OPTIONS:
            self.cmap_combo.addItem(cmap, cmap)
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._set_dirty())

        self.auto_lbl  = QtWidgets.QLabel("-")
        self.vmin_spin = self._value_spin()
        self.vmax_spin = self._value_spin()
        self.vmin_spin.valueChanged.connect(lambda *_: self._set_dirty())
        self.vmax_spin.valueChanged.connect(lambda *_: self._set_dirty())

        self.clamp_cb = QtWidgets.QCheckBox("Clamp to range")
        self.clamp_cb.toggled.connect(lambda *_: self._set_dirty())

        self.reset_btn = QtWidgets.QPushButton("Reset to auto")
        self.reset_btn.clicked.connect(self._reset_range)

        color_form.addRow("Preset",     self.cmap_combo)
        color_form.addRow("Auto range", self.auto_lbl)
        color_form.addRow("User min",   self.vmin_spin)
        color_form.addRow("User max",   self.vmax_spin)
        color_form.addRow("",           self.clamp_cb)
        color_form.addRow("",           self.reset_btn)
        outer.addWidget(color_box)

        outer.addWidget(
            self._make_apply_button(self._apply, "Applying display settings…")
        )

        static_color_box = QtWidgets.QGroupBox("Color By")
        static_color_form = QtWidgets.QFormLayout(static_color_box)
        static_color_form.setContentsMargins(6, 4, 6, 6)

        self.static_color_combo = QtWidgets.QComboBox()
        for color_by_value in VALID_STATIC_COLOR_BY:
            self.static_color_combo.addItem(
                self._STATIC_COLOR_LABELS.get(color_by_value, color_by_value),
                color_by_value,
            )
        self.static_color_combo.currentIndexChanged.connect(lambda *_: self._set_static_color_dirty())

        self.static_cmap_combo = QtWidgets.QComboBox()
        for cmap in COLORMAP_OPTIONS:
            self.static_cmap_combo.addItem(cmap, cmap)
        self.static_cmap_combo.currentTextChanged.connect(lambda *_: self._set_static_color_dirty())

        self.static_auto_lbl = QtWidgets.QLabel("-")
        self.static_vmin_spin = self._value_spin()
        self.static_vmax_spin = self._value_spin()
        self.static_vmin_spin.valueChanged.connect(lambda *_: self._set_static_color_dirty())
        self.static_vmax_spin.valueChanged.connect(lambda *_: self._set_static_color_dirty())

        self.static_clamp_cb = QtWidgets.QCheckBox("Clamp to range")
        self.static_clamp_cb.toggled.connect(lambda *_: self._set_static_color_dirty())

        self.static_reset_btn = QtWidgets.QPushButton("Reset to auto")
        self.static_reset_btn.clicked.connect(self._reset_static_range)

        self.static_apply_btn = QtWidgets.QPushButton("Apply")
        self.static_apply_btn.setEnabled(False)
        self.static_apply_btn.clicked.connect(
            lambda: self._run_heavy(self._apply_static_color, "Applying color by…")
        )

        static_color_form.addRow("Source", self.static_color_combo)
        static_color_form.addRow("Color ramp", self.static_cmap_combo)
        static_color_form.addRow("Auto range", self.static_auto_lbl)
        static_color_form.addRow("User min", self.static_vmin_spin)
        static_color_form.addRow("User max", self.static_vmax_spin)
        static_color_form.addRow("", self.static_clamp_cb)
        static_color_form.addRow("", self.static_reset_btn)
        static_color_form.addRow("", self.static_apply_btn)
        outer.addWidget(static_color_box)
        outer.addStretch(1)
        self.refresh("init")

    # ── Sync from session state ───────────────────────────────────────────

    def refresh(self, reason: str = "full"):
        sync_display = reason in self._REFRESH_REASONS or not self._dirty
        sync_static = reason in self._STATIC_COLOR_REFRESH_REASONS or not self._static_color_dirty
        if not sync_display and not sync_static:
            return
        self._syncing = True
        try:
            if sync_display:
                self._set_combo_data(self.demand_combo, self.session.state.demand)
                self._populate_component_combo(
                    self.session.state.demand,
                    selected_component=self.session.state.component,
                )
                gf_count = self.session.gf_subfault_count()
                gf_max = max(gf_count - 1, 0)
                self.gf_subfault_spin.setMaximum(gf_max)
                # TODO: Keep the UI pinned to the current GF subfault domain so a
                # stale value never leaks into Display -> Green Functions apply.
                self.gf_subfault_spin.setValue(min(self.session.current_display_gf_subfault(), gf_max))
                self.gf_subfault_spin.setEnabled(self.session.state.demand == "gf")
                self._set_combo(
                    self.cmap_combo,
                    self.session.state.colormap or colormap_for_component(self.session.state.component),
                )
                lo, hi = self._field_auto_limits()
                self.auto_lbl.setText(f"{lo:.4g}  /  {hi:.4g}")
                if (self.session.state.user_vmin is None
                        or self.session.state.user_vmax is None):
                    self.session.state.set_user_color_range(lo, hi)
                self.vmin_spin.setValue(float(self.session.state.user_vmin))
                self.vmax_spin.setValue(float(self.session.state.user_vmax))
                self.clamp_cb.setChecked(self.session.state.clamp_enabled)
                self._clear_dirty()
            if sync_static:
                current_static = self.session.current_static_color_by() or VALID_STATIC_COLOR_BY[0]
                self._set_combo_data(self.static_color_combo, current_static)
                self._set_combo(self.static_cmap_combo, self.session.current_static_colormap())
                lo, hi = self.session.current_static_auto_limits()
                self.static_auto_lbl.setText(f"{lo:.4g}  /  {hi:.4g}")
                static_vmin, static_vmax = self.session.current_static_user_range()
                if static_vmin is None or static_vmax is None:
                    static_vmin, static_vmax = lo, hi
                self.static_vmin_spin.setValue(float(static_vmin))
                self.static_vmax_spin.setValue(float(static_vmax))
                self.static_clamp_cb.setChecked(self.session.current_static_clamp_enabled())
                self._clear_static_color_dirty()
            allow_static = not self.session.state.is_playing
            self.static_color_combo.setEnabled(allow_static)
            self.static_cmap_combo.setEnabled(allow_static)
            self.static_vmin_spin.setEnabled(allow_static)
            self.static_vmax_spin.setEnabled(allow_static)
            self.static_clamp_cb.setEnabled(allow_static)
            self.static_reset_btn.setEnabled(allow_static)
            self.static_apply_btn.setEnabled(allow_static and self._static_color_dirty)
        finally:
            self._syncing = False

    def _set_static_color_dirty(self, dirty: bool = True):
        if self._syncing:
            return
        self._static_color_dirty = bool(dirty)
        if not self.session.state.is_playing:
            self.static_apply_btn.setEnabled(self._static_color_dirty)

    def _clear_static_color_dirty(self):
        self._static_color_dirty = False
        self.static_apply_btn.setEnabled(False)

    def _reset_static_range(self):
        lo, hi = self.session.current_static_auto_limits()
        for spin, val in ((self.static_vmin_spin, lo), (self.static_vmax_spin, hi)):
            b = spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(b)
        b = self.static_clamp_cb.blockSignals(True)
        self.static_clamp_cb.setChecked(False)
        self.static_clamp_cb.blockSignals(b)
        self._set_static_color_dirty(True)

    def _field_auto_limits(self) -> tuple[float, float]:
        gf_subfault = self.session.current_display_gf_subfault() if self.session.state.demand == "gf" else 0
        return self.session.adapter.default_scalar_limits(
            self.session.state.demand,
            self.session.state.component,
            subfault_id=gf_subfault,
        )

    def _reset_range(self):
        lo, hi = self._field_auto_limits()
        for spin, val in ((self.vmin_spin, lo), (self.vmax_spin, hi)):
            b = spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(b)
        b = self.clamp_cb.blockSignals(True)
        self.clamp_cb.setChecked(False)
        self.clamp_cb.blockSignals(b)
        self._set_dirty(True)

    # ── Single atomic apply ───────────────────────────────────────────────

    def _apply(self):
        if self._syncing:
            return
        self.session.apply_display_settings(
            demand=str(self.demand_combo.currentData()),
            component=str(self.component_combo.currentData()),
            gf_subfault_id=self.gf_subfault_spin.value(),
            colormap=self.cmap_combo.currentText(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            clamp_enabled=self.clamp_cb.isChecked(),
        )
        self._clear_dirty()

    def _apply_static_color(self):
        if self._syncing:
            return
        self.session.apply_static_color_settings(
            color_by=str(self.static_color_combo.currentData()),
            colormap=self.static_cmap_combo.currentText(),
            vmin=self.static_vmin_spin.value(),
            vmax=self.static_vmax_spin.value(),
            clamp_enabled=self.static_clamp_cb.isChecked(),
        )
        self._clear_static_color_dirty()

    def _on_demand_changed(self):
        if self._syncing:
            return
        demand = str(self.demand_combo.currentData())
        self._populate_component_combo(demand)
        self.gf_subfault_spin.setEnabled(demand == "gf")
        self._set_dirty(True)

    def _populate_component_combo(self, demand: str, *, selected_component: str | None = None):
        selected_component = (
            selected_component
            if selected_component in self.session.adapter.available_components_for_demand(demand)
            else None
        )
        options = self.session.adapter.display_component_options(demand)
        previous = selected_component or self.component_combo.currentData()
        block = self.component_combo.blockSignals(True)
        self.component_combo.clear()
        for component_value, component_label in options:
            self.component_combo.addItem(component_label, component_value)
        restore = previous if previous in {value for value, _ in options} else options[0][0]
        idx = self.component_combo.findData(restore)
        if idx >= 0:
            self.component_combo.setCurrentIndex(idx)
        self.component_combo.blockSignals(block)


class VisualizationSection(_SectionBase):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        self.internal_cb = QtWidgets.QCheckBox("Show internal nodes")
        self.external_cb = QtWidgets.QCheckBox("Show external nodes")
        self.qa_cb       = QtWidgets.QCheckBox("Show QA")
        for cb in (self.internal_cb, self.external_cb, self.qa_cb):
            cb.toggled.connect(lambda *_: self._set_dirty())
            lay.addWidget(cb)

        lay.addWidget(self._make_apply_button(self._apply, "Updating node visibility…"))
        lay.addStretch(1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason not in {"init", "full", "visibility"} and self._dirty:
            return
        self._syncing = True
        try:
            self.internal_cb.setChecked(self.session.state.show_internal)
            self.external_cb.setChecked(self.session.state.show_external)
            self.qa_cb.setChecked(self.session.state.show_qa)
            self._clear_dirty()
        finally:
            self._syncing = False

    def _apply(self):
        if self._syncing:
            return
        self.session.apply_visibility_settings(
            show_internal=self.internal_cb.isChecked(),
            show_external=self.external_cb.isChecked(),
            show_qa=self.qa_cb.isChecked(),
        )
        self._clear_dirty()


class WarpSection(_SectionBase):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self.warp_cb = QtWidgets.QCheckBox("Enable displacement warp")
        self.warp_cb.toggled.connect(lambda *_: self._set_dirty())
        outer.addWidget(self.warp_cb)

        axes_row = QtWidgets.QWidget()
        arl = QtWidgets.QHBoxLayout(axes_row)
        arl.setContentsMargins(0, 0, 0, 0)
        arl.setSpacing(6)
        arl.addWidget(QtWidgets.QLabel("Axes:"))
        self.x_cb = QtWidgets.QCheckBox("X (E)")
        self.y_cb = QtWidgets.QCheckBox("Y (N)")
        self.z_cb = QtWidgets.QCheckBox("Z")
        for cb in (self.x_cb, self.y_cb, self.z_cb):
            cb.setChecked(True)
            cb.toggled.connect(lambda *_: self._set_dirty())
            arl.addWidget(cb)
        arl.addStretch()
        outer.addWidget(axes_row)

        scale_row = QtWidgets.QWidget()
        srl = QtWidgets.QHBoxLayout(scale_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(4)
        srl.addWidget(QtWidgets.QLabel("Scale:"))
        for lbl, val in (("×10", 10), ("×100", 100), ("×1k", 1_000), ("×10k", 10_000)):
            btn = QtWidgets.QPushButton(lbl)
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _c, v=val: self._preset(v))
            srl.addWidget(btn)
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setDecimals(1)
        self.scale_spin.setRange(1.0, 1_000_000.0)
        self.scale_spin.setSingleStep(10.0)
        self.scale_spin.setValue(100.0)
        self.scale_spin.setSuffix("×")
        self.scale_spin.valueChanged.connect(lambda *_: self._set_dirty())
        srl.addWidget(self.scale_spin, 1)
        auto_btn = QtWidgets.QPushButton("Auto")
        auto_btn.setToolTip("Suggest scale so peak displacement fills ~5 % of domain.")
        auto_btn.clicked.connect(self._auto)
        srl.addWidget(auto_btn)
        outer.addWidget(scale_row)

        outer.addWidget(self._make_apply_button(self._apply, "Applying warp settings…"))
        outer.addStretch(1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason not in {"init", "full", "warp"} and self._dirty:
            return
        self._syncing = True
        try:
            self.warp_cb.setChecked(self.session.state.disp_warp_enabled)
            ax = self.session.state.warp_axes
            self.x_cb.setChecked(ax[0])
            self.y_cb.setChecked(ax[1])
            self.z_cb.setChecked(ax[2])
            scale = (
                float(self.session.state.warp_scale)
                if self.session.state.warp_scale is not None
                else float(self.session.suggested_warp_scale())
            )
            self.scale_spin.setValue(scale)
            self._clear_dirty()
        finally:
            self._syncing = False

    def _preset(self, val: float):
        if self._syncing:
            return
        b = self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(float(val))
        self.scale_spin.blockSignals(b)
        self._set_dirty(True)

    def _auto(self):
        b = self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(float(self.session.suggested_warp_scale()))
        self.scale_spin.blockSignals(b)
        self._set_dirty(True)

    def _apply(self):
        if self._syncing:
            return
        self.session.apply_warp_settings(
            warp_enabled=self.warp_cb.isChecked(),
            warp_axes=(self.x_cb.isChecked(), self.y_cb.isChecked(), self.z_cb.isChecked()),
            warp_scale=self.scale_spin.value(),
        )
        self._clear_dirty()


# ── Main side panel ───────────────────────────────────────────────────────────

class ViewerSidePanel(QtWidgets.QWidget):
    """Vertical nav-menu + ``QStackedWidget`` right panel.

    Menu: Node | Display | Visibility | Warp || Responses | GF
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._active_key: str = "Node"
        self._last_trace_refresh_at: float = 0.0

        # Heavy pages that missed a stale-making update while not active.
        self._dirty_heavy: set[str] = set()

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Vertical nav menu ─────────────────────────────────────────────────
        nav = QtWidgets.QWidget()
        nav.setObjectName("SidePanelNav")
        nav.setFixedWidth(72)
        nav_lay = QtWidgets.QVBoxLayout(nav)
        nav_lay.setContentsMargins(4, 6, 4, 6)
        nav_lay.setSpacing(2)

        self._nav_buttons: dict[str, QtWidgets.QToolButton] = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)

        for key, icon_name, short_label in _MENU_ENTRIES:
            if key == "Responses":
                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.HLine)
                sep.setFixedHeight(1)
                sep.setStyleSheet("background: #d7dbe2;")
                nav_lay.addSpacing(4)
                nav_lay.addWidget(sep)
                nav_lay.addSpacing(4)

            btn = QtWidgets.QToolButton()
            btn.setObjectName("SidePanelNavButton")
            btn.setText(short_label)
            btn.setIcon(icon(icon_name, LIGHT_PALETTE.navy, 18))
            btn.setIconSize(QtCore.QSize(18, 18))
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c, k=key: self._navigate(k))
            self._nav_buttons[key] = btn
            group.addButton(btn)
            nav_lay.addWidget(btn, alignment=QtCore.Qt.AlignHCenter)

        nav_lay.addStretch(1)
        nav.setStyleSheet(
            "QWidget#SidePanelNav { background: #f3f4f6; border-right: 1px solid #d7dbe2; }"
            "QToolButton#SidePanelNavButton {"
            "  min-width: 60px; max-width: 60px; min-height: 50px;"
            "  border: 1px solid transparent; border-radius: 5px;"
            "  background: transparent; color: #1e3558;"
            "  font-size: 10px; padding: 3px 2px;"
            "}"
            "QToolButton#SidePanelNavButton:checked {"
            "  background: #e8f0fe; border-color: #2a6bc2; font-weight: 600;"
            "}"
            "QToolButton#SidePanelNavButton:hover:!checked { background: #e6ebef; }"
        )
        root.addWidget(nav)

        # ── Content area ──────────────────────────────────────────────────────
        self._stack = QtWidgets.QStackedWidget()
        root.addWidget(self._stack, 1)

        # ── Build pages ───────────────────────────────────────────────────────
        self._node_page       = NodeSearchSection(session)
        self._display_page    = DisplaySection(session)
        self._visibility_page = VisualizationSection(session)
        self._warp_page       = WarpSection(session)

        # Heavy pages: created lazily on first navigation.
        self._responses_page = _LazyPage(lambda: _ResponsesAnalysisTabs(session))
        self._gf_page        = _LazyPage(lambda: GFPanel(session))

        self._pages: dict[str, QtWidgets.QWidget] = {
            "Node":       self._node_page,
            "Display":    self._display_page,
            "Visibility": self._visibility_page,
            "Warp":       self._warp_page,
            "Responses":  self._responses_page,
            "GF":         self._gf_page,
        }

        self._page_views: dict[str, QtWidgets.QWidget] = {}
        for key, page in self._pages.items():
            view = _PageScrollArea(page)
            self._page_views[key] = view
            self._stack.addWidget(view)

        self._navigate("Node")
        self._sync_analysis_nav()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate(self, key: str):
        self._active_key = key

        btn = self._nav_buttons.get(key)
        if btn is not None:
            btn.setChecked(True)

        page = self._pages[key]

        if isinstance(page, _LazyPage):
            already_existed = page.is_created
            page.ensure_created()
            # If the page existed but missed a selection/full update, catch up now.
            if already_existed and key in self._dirty_heavy:
                page.refresh("full")
        elif key in self._dirty_heavy:
            page.refresh("full")

        self._dirty_heavy.discard(key)
        view = self._page_views[key]
        self._stack.setCurrentWidget(view)

    # ── Refresh routing ───────────────────────────────────────────────────────

    def refresh(self, reason: str):
        """Route a session-update reason to the correct sub-widgets.

        Only the active page is refreshed for heavy content.  Non-active heavy
        pages that depend on ``reason`` are flagged dirty and will refresh the
        next time the user navigates to them.
        """
        # ── Time: cursor-only update on the active heavy page ─────────────────
        # Responses/Time History uses matplotlib blitting (< 1 ms per update),
        # while Spectrum and Arias ignore "time" updates inside their tabs.
        if reason == "time":
            if self._active_key in _HEAVY_KEYS:
                if self._active_key == "Responses" and self.session.state.is_playing:
                    now = time.perf_counter()
                    if (now - self._last_trace_refresh_at) < _TRACE_TIME_REFRESH_INTERVAL_S:
                        return
                    self._last_trace_refresh_at = now
                page = self._pages.get(self._active_key)
                if page is not None:
                    page.refresh("time")
            return

        if reason == "playback":
            self._display_page.refresh("playback")
            if not self.session.state.is_playing and self._active_key in {"Responses", "GF"}:
                page = self._pages.get(self._active_key)
                if page is not None:
                    page.refresh("time")
            return

        if reason == "multi_selection":
            self._sync_analysis_nav()
            return

        # ── All other reasons: lightweight pages always, heavy only if active ──

        # Lightweight pages — cheap Qt-label updates, always safe to refresh.
        self._node_page.refresh(reason)
        self._display_page.refresh(reason)
        self._visibility_page.refresh(reason)
        self._warp_page.refresh(reason)

        # Mark non-active heavy pages dirty when their content would change.
        if reason in _STALE_REASONS:
            for key in _HEAVY_KEYS:
                if key != self._active_key:
                    self._dirty_heavy.add(key)

        # Refresh the active heavy page immediately.
        if self._active_key in _HEAVY_KEYS:
            page = self._pages.get(self._active_key)
            if page is not None:
                page.refresh(reason)
                self._dirty_heavy.discard(self._active_key)


    def _sync_analysis_nav(self):
        """Disable heavy analysis pages while a multi-node selection is active."""
        multi_on = self.session.has_multi_selection()
        for key in ("Responses", "GF"):
            btn = self._nav_buttons.get(key)
            if btn is not None:
                btn.setEnabled(not multi_on)
        if multi_on and self._active_key in _HEAVY_KEYS:
            self._navigate("Node")


__all__ = ["ViewerSidePanel"]
