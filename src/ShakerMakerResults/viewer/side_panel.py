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
from .cmap_preview import ColormapPreview
from .colors import COLORMAP_OPTIONS, colormap_for_component
from .icons import icon
from .session import VALID_STATIC_COLOR_BY
from .theme import LIGHT_PALETTE, active_palette
from .trace_panel import AriasIntensityPanel, GFPanel, ResponsesPanel, SpectrumPanel
from .visual_editors import LegendEditorDialog, TransferFunctionDialog

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


def _stack(*widgets) -> QtWidgets.QWidget:
    """Return a small QWidget that stacks *widgets* vertically with zero margin.

    Used by the side-panel sections to pair a combo box with its colormap
    preview inside one row of a form layout.
    """
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    for widget in widgets:
        layout.addWidget(widget)
    return container


# ── Menu configuration ────────────────────────────────────────────────────────

_MENU_ENTRIES: list[tuple[str, str, str]] = [
    ("Node",       "nav_node",       "Node"),
    ("Display",    "nav_display",    "Display"),
    ("Visibility", "nav_visibility", "Vis."),
    ("Warp",       "nav_warp",       "Warp"),
    # ── separator ── analysis pages below ──
    ("Responses",  "nav_responses",  "Resp."),
    ("GF",         "nav_gf",         "GF"),
    # ── pinned to bottom ──
    ("Geographic", "nav_geographic", "Geo."),
]

# Keys whose pages are heavy (lazy-created; refresh only when active).
_HEAVY_KEYS: frozenset[str] = frozenset({"Responses", "GF"})

# Reasons that make a heavy page's content stale if it wasn't active.
_STALE_REASONS: frozenset[str] = frozenset({"selection", "full"})

_TRACE_TIME_REFRESH_INTERVAL_S = 0.08


# ── Lazy page wrapper ─────────────────────────────────────────────────────────

class _LazyPage(QtWidgets.QWidget):
    """Wrapper that delays the construction of a heavy side-panel page.

    The widget is instantiated up-front (so the nav menu can dock it) but
    the real content is only built the first time :meth:`ensure_created`
    fires. Used for the Responses / GF / Information analysis pages.
    """

    """Delays construction of a heavy widget until first navigation.

    ``ensure_created()`` triggers the factory and embeds the result.
    ``refresh()`` is a no-op until the widget is created.
    """

    def __init__(self, factory, parent=None):
        """Store the factory callable; do not run it yet."""
        super().__init__(parent)
        self._factory = factory
        self._widget: QtWidgets.QWidget | None = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lay = lay

    def ensure_created(self) -> QtWidgets.QWidget:
        """Build the real widget on first call and embed it into the layout."""
        if self._widget is None:
            self._widget = self._factory()
            self._lay.addWidget(self._widget, 1)
        return self._widget

    def refresh(self, reason: str):
        """Forward ``refresh(reason)`` to the embedded widget when it exists."""
        if self._widget is not None:
            self._widget.refresh(reason)

    @property
    def is_created(self) -> bool:
        """Return ``True`` once the factory has run at least once."""
        return self._widget is not None


class _PageScrollArea(QtWidgets.QScrollArea):
    """Per-page vertical scroll container that respects the active panel width."""

    def __init__(self, page: QtWidgets.QWidget, parent=None):
        """Wrap ``page`` in a vertical scroll area without an outer frame."""
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
        """Build the three lazy analysis tabs (Time History / Spectrum / Arias)."""
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
        """Forward refresh to the active tab; queue the reason for the others."""
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
        page.ensure_created()
        if key in self._dirty_tabs:
            page.refresh("full")
        self._dirty_tabs.discard(key)

    def _ensure_current_tab(self):
        self._pages[self._active_key()].ensure_created()

    def _active_key(self) -> str:
        key = self._tabs.tabText(self._tabs.currentIndex())
        return key if key in self._pages else "Time History"


# ── Shared base for Apply-button sections ─────────────────────────────────────

class _SectionBase(QtWidgets.QWidget):
    """Shared plumbing for every side-panel section.

    Each subclass:

    * Owns a small form of widgets bound to a slice of the session state.
    * Tracks a ``_dirty`` flag so the Apply button can show the user when
      pending edits are not yet committed.
    * Routes its ``_apply`` call through :meth:`_route_apply`, which honours
      the toolbar's "Apply to" selector (global / active pane / all panes
      in the active tab).
    * Implements :meth:`refresh` so the section can re-sync from the
      current state whenever the window broadcasts a relevant reason.

    The ``_syncing`` flag is set while we push fresh values into the
    widgets so the ``valueChanged`` / ``currentTextChanged`` signals
    don't bounce back as "user edits".
    """

    def __init__(self, session, parent=None):
        """Wire the section to a session and reset the dirty / syncing flags."""
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
        """Create an always-clickable Apply button that shows the busy overlay.

        Earlier the button was gated by an internal "dirty" flag — it
        stayed disabled until the user touched a control.  In practice
        background refreshes (playback ticks, pane state syncs) cleared
        the flag faster than the user could click, so the button looked
        permanently locked.  The user explicitly asked for it to be
        always pressable; we keep ``_dirty`` to remember whether the
        section has unsaved edits (for refresh-skipping logic) but no
        longer use it to disable the button.
        """
        def _wrapped():
            self._run_heavy(callback, message)

        btn = QtWidgets.QPushButton("Apply")
        btn.setEnabled(True)
        btn.clicked.connect(_wrapped)
        self.apply_button = btn
        return btn

    def _set_dirty(self, dirty: bool = True):
        if self._syncing:
            return
        self._dirty = bool(dirty)
        # Button stays clickable; flag only drives refresh-skipping.

    def _clear_dirty(self):
        self._dirty = False
        # Button stays clickable.

    def resync_to_active_pane(self) -> None:
        """Resync the section to whichever pane is now active.

        Called when the user switches the active viewport.  Drops any
        in-progress edits (they targeted the previous pane, so carrying
        them over would silently apply the wrong settings) and any
        local "user typed something" locks, then refreshes from the
        new pane's effective state.  Sections that own their own locks
        (e.g. ``_range_user_locked`` in DisplaySection /
        StaticColorSection) get them reset here uniformly.
        """
        self._dirty = False
        if hasattr(self, "_range_user_locked"):
            try:
                self._range_user_locked = False
            except Exception:
                pass
        try:
            self.refresh("full")
        except Exception:
            pass

    @staticmethod
    def _set_combo(combo, value):
        b = combo.blockSignals(True)
        combo.setCurrentText(str(value))
        combo.blockSignals(b)

    @staticmethod
    def _set_combo_data(combo, value):
        idx = 0 if value is None else combo.findData(value)
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

    # ── Apply-to target plumbing (shared by every editor section) ───────────

    def _resolve_apply_target(self) -> str:
        """Return the active Apply-to value from the window toolbar."""
        w = self.window()
        if w is not None and hasattr(w, "apply_target"):
            try:
                return w.apply_target()
            except Exception:
                pass
        return "all"

    def _apply_target_panes(self) -> list:
        """Resolve the current Apply-to selection into a list of ViewPanes.

        Empty list ⇒ the section should fall back to the global apply.
        """
        target = self._resolve_apply_target()
        if target == "all":
            return []
        w = self.window()
        multi = getattr(w, "multi_view", None) if w is not None else None
        if multi is None:
            return []
        if target == "tab":
            current = getattr(multi, "current", None) or multi
            return list(getattr(current, "_panes", []) or [])
        pane = getattr(multi, "active_pane", None)
        return [pane] if pane is not None else []

    @staticmethod
    def _iter_all_panes(multi):
        """Yield every pane across every tab of *multi*."""
        tabs = getattr(multi, "_tabs", None)
        if tabs is None:
            for pane in getattr(multi, "_panes", []) or []:
                yield pane
            return
        for i in range(tabs.count()):
            area = tabs.widget(i)
            for pane in getattr(area, "_panes", []) or []:
                yield pane

    def _route_apply(
        self,
        *,
        pane_payload: dict,
        global_apply,
        clear_attrs=None,
    ) -> None:
        """Route an Apply through the active target.

        * ``pane_payload`` — ``{pane_state_attr: value}``; written to
          each target pane's ``pane_state`` when target ≠ all.
        * ``global_apply`` — zero-arg callable run when target == all.
        * ``clear_attrs`` — names to ``clear_overrides`` on every pane
          in the "all" path so per-pane overrides don't shadow the new
          global value.  Defaults to ``pane_payload.keys()``.
        """
        target = self._resolve_apply_target()
        w = self.window()
        multi = getattr(w, "multi_view", None) if w is not None else None
        attrs = list(clear_attrs) if clear_attrs is not None else list(pane_payload.keys())

        if target == "all":
            if multi is not None:
                for pane in self._iter_all_panes(multi):
                    ps = getattr(pane, "pane_state", None)
                    if ps is not None:
                        try:
                            ps.clear_overrides(attrs)
                        except Exception:
                            pass
            try:
                global_apply()
            except Exception:
                pass
            return

        for pane in self._apply_target_panes():
            ps = getattr(pane, "pane_state", None)
            if ps is None:
                continue
            for attr_name, value in pane_payload.items():
                try:
                    setattr(ps, attr_name, value)
                except Exception:
                    pass
            scene = getattr(pane, "scene", None)
            if scene is not None:
                try:
                    scene.rebuild_scalar_actor(render=False)
                    pane.plotter.render()
                except Exception:
                    pass


# ── Lightweight section widgets ───────────────────────────────────────────────

class _StationTable(QtWidgets.QTableWidget):
    """Editable station table with tab-separated clipboard paste support."""

    HEADERS = ("Station", "coord x", "coord y", "coord z")

    def __init__(self, parent=None):
        """Build the empty 4-column table (Station / x / y / z) with paste support."""
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
        """Build the selected-node form (id, coords, dist) and the station table."""
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

    def refresh(self, _reason: str = "full"):  # noqa: ARG002
        """Re-sync the selected-node form and the station table from the session."""
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
            self.session.add_node_to_multi_selection(int(self.node_id_spin.value()))

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


# ── Vector Field section ─────────────────────────────────────────────────────

class VectorFieldSection(_SectionBase):
    """Stand-alone editor for the arrow-glyph vector field overlay.

    Used to be a sub-group inside :class:`DisplaySection`; now lives in its
    own dock so the user does not confuse the vector overlay's color ramp /
    apply with the main field's color ramp / apply.  Backend is unchanged —
    every Apply goes through :meth:`ViewerSession.apply_vector_field_settings`.
    """

    _REFRESH_REASONS = frozenset({"init", "full", "vector_field"})
    _VECTOR_DEMANDS = [
        ("disp",  "Displacement"),
        ("vel",   "Velocity"),
        ("accel", "Acceleration"),
    ]

    def __init__(self, session, parent=None):
        """Build the vector-field form (enable / field / colormap / scale / Apply)."""
        super().__init__(session, parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        box = QtWidgets.QGroupBox("Vector Field")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(6, 4, 6, 6)

        self.enabled_cb = QtWidgets.QCheckBox("Show arrow glyphs")
        self.enabled_cb.toggled.connect(lambda *_: self._set_dirty())
        form.addRow("", self.enabled_cb)

        self.demand_combo = QtWidgets.QComboBox()
        for val, lbl in self._VECTOR_DEMANDS:
            self.demand_combo.addItem(lbl, val)
        self.demand_combo.currentIndexChanged.connect(lambda *_: self._set_dirty())
        form.addRow("Field", self.demand_combo)

        self.cmap_combo = QtWidgets.QComboBox()
        for cmap in COLORMAP_OPTIONS:
            self.cmap_combo.addItem(cmap, cmap)
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._set_dirty())
        self.cmap_preview = ColormapPreview(self.cmap_combo.currentText())
        self.cmap_combo.currentTextChanged.connect(self.cmap_preview.setColormap)
        # Click on the ramp opens the advanced color editor — same dialog
        # the explicit "Advanced color editor" button uses in the other
        # sections, so users get a single canonical place to redesign the
        # transfer function regardless of which preview they clicked.
        self.cmap_preview.clicked.connect(self._open_transfer_editor)
        form.addRow("Color ramp", _stack(self.cmap_combo, self.cmap_preview))

        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 100.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.25)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setSuffix("×")
        self.scale_spin.valueChanged.connect(lambda *_: self._set_dirty())
        form.addRow("Scale", self.scale_spin)

        outer.addWidget(box)
        outer.addWidget(
            self._make_apply_button(self._apply, "Computing vectors…")
        )
        outer.addStretch(1)
        self.refresh("init")

    # ── Refresh ────────────────────────────────────────────────────────────

    def refresh(self, reason: str = "full") -> None:
        """Re-sync the vector-field form from the session, unless edits are pending."""
        if reason not in self._REFRESH_REASONS and self._dirty:
            return
        self._syncing = True
        try:
            block = self.enabled_cb.blockSignals(True)
            self.enabled_cb.setChecked(bool(self.session.state.vector_field_enabled))
            self.enabled_cb.blockSignals(block)

            self._set_combo_data(self.demand_combo, self.session.state.vector_field_demand)
            self._set_combo(self.cmap_combo, self.session.state.vector_field_colormap)
            self.cmap_preview.setColormap(self.cmap_combo.currentText())

            block = self.scale_spin.blockSignals(True)
            self.scale_spin.setValue(float(self.session.state.vector_field_scale))
            self.scale_spin.blockSignals(block)
            self._clear_dirty()
        finally:
            self._syncing = False

    # ── Apply ──────────────────────────────────────────────────────────────

    def _apply(self):
        """Commit the vector-field settings (enable / demand / scale / colormap)."""
        if self._syncing:
            return
        enabled = self.enabled_cb.isChecked()
        demand = str(self.demand_combo.currentData())
        scale = self.scale_spin.value()
        colormap = str(self.cmap_combo.currentData() or "viridis")
        self._route_apply(
            pane_payload={
                "vector_field_enabled":  enabled,
                "vector_field_demand":   demand,
                "vector_field_scale":    scale,
                "vector_field_colormap": colormap,
            },
            global_apply=lambda: self.session.apply_vector_field_settings(
                enabled=enabled,
                demand=demand,
                scale=scale,
                colormap=colormap,
            ),
        )
        self._clear_dirty()

    # ── Ramp click → open advanced colour editor ───────────────────────────

    def _open_transfer_editor(self):
        """Open the same advanced colour editor the other sections use.

        The TransferFunctionDialog edits global colour preferences applied
        to the active scalar field; the vector overlay then re-renders
        with the new ramp on its next Apply.
        """
        dlg = TransferFunctionDialog(self.session, self.window(), initial_colormap=self.cmap_combo.currentText())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        values = dlg.values()
        self.session.apply_transfer_function_preferences(
            colormap=values.colormap,
            inverted=values.invert,
            discrete=values.discrete,
            bins=values.bins,
            nan_color=values.nan_color,
            below_color=values.below_color,
            above_color=values.above_color,
            use_below=values.use_below,
            use_above=values.use_above,
            symmetric_range=values.symmetric_range,
            percentile_clip=values.percentile_clip,
            custom_colormap_object=dlg.build_custom_colormap(),
        )


# ── Static Color (Color By) section ──────────────────────────────────────────

class StaticColorSection(_SectionBase):
    """Stand-alone editor for the static colour-by overlay.

    Pulled out of :class:`DisplaySection` so the dense Newmark / Arias
    parameter list does not crowd the Field + Color Map editor.  All
    state still lives in :class:`ViewerSession`; every Apply routes
    through :meth:`ViewerSession.apply_static_color_settings`.
    """

    _REFRESH_REASONS = frozenset({
        "init", "full", "static_color", "playback",
    })
    _STATIC_COLOR_LABELS = {
        "elevation_z": "Elevation Z",
        "newmark_sa": "Sa / Newmark",
        "arias": "Arias",
    }

    def __init__(self, session, parent=None):
        """Build the Color-By form (source / colormap / range / Newmark / Arias / Apply)."""
        super().__init__(session, parent)
        # Same lock-on-edit semantics as DisplaySection for the
        # static color User min / User max spinboxes.
        self._range_user_locked = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        box = QtWidgets.QGroupBox("Color By")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(6, 4, 6, 6)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("Default", None)
        for color_by_value in VALID_STATIC_COLOR_BY:
            self.source_combo.addItem(
                self._STATIC_COLOR_LABELS.get(color_by_value, color_by_value),
                color_by_value,
            )
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        self.cmap_combo = QtWidgets.QComboBox()
        for cmap in COLORMAP_OPTIONS:
            self.cmap_combo.addItem(cmap, cmap)
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._set_dirty())
        self.cmap_preview = ColormapPreview(self.cmap_combo.currentText())
        self.cmap_combo.currentTextChanged.connect(self.cmap_preview.setColormap)
        self.cmap_preview.clicked.connect(self._open_transfer_editor)

        self.auto_lbl = QtWidgets.QLabel("-")
        self.vmin_spin = self._value_spin()
        self.vmax_spin = self._value_spin()
        # Same lock-on-edit semantics as DisplaySection — the spinboxes
        # stay where the user left them until they press Reset to auto.
        self.vmin_spin.valueChanged.connect(lambda *_: self._on_user_range_edited())
        self.vmax_spin.valueChanged.connect(lambda *_: self._on_user_range_edited())

        self.clamp_cb = QtWidgets.QCheckBox("Clamp to range")
        self.clamp_cb.toggled.connect(lambda *_: self._set_dirty())

        self.reset_btn = QtWidgets.QPushButton("Reset to auto")
        self.reset_btn.clicked.connect(self._reset_range)
        self.transfer_btn = QtWidgets.QPushButton("Advanced color editor")
        self.transfer_btn.clicked.connect(self._open_transfer_editor)
        self.legend_btn = QtWidgets.QPushButton("Legend settings")
        self.legend_btn.clicked.connect(self._open_legend_editor)

        # Newmark parameters.
        self.newmark_T_spin = QtWidgets.QDoubleSpinBox()
        self.newmark_T_spin.setDecimals(3)
        self.newmark_T_spin.setRange(0.0, 20.0)
        self.newmark_T_spin.setSingleStep(0.05)
        self.newmark_T_spin.setValue(0.0)
        self.newmark_T_spin.valueChanged.connect(lambda *_: self._set_dirty())

        self.newmark_component_combo = QtWidgets.QComboBox()
        for component in ("resultant", "z", "e", "n"):
            self.newmark_component_combo.addItem(component, component)
        self.newmark_component_combo.currentIndexChanged.connect(lambda *_: self._set_dirty())

        self.newmark_data_combo = QtWidgets.QComboBox()
        for demand in ("accel", "vel", "disp"):
            self.newmark_data_combo.addItem(demand, demand)
        self.newmark_data_combo.currentIndexChanged.connect(lambda *_: self._set_dirty())

        self.newmark_spectral_combo = QtWidgets.QComboBox()
        for spectral_type in ("PSa", "Sa", "PSv", "Sv", "Sd"):
            self.newmark_spectral_combo.addItem(spectral_type, spectral_type)
        self.newmark_spectral_combo.currentIndexChanged.connect(lambda *_: self._set_dirty())

        self.newmark_factor_spin = self._value_spin()
        self.newmark_factor_spin.setValue(1.0 / 9.81)
        self.newmark_factor_spin.valueChanged.connect(lambda *_: self._set_dirty())

        self.newmark_jobs_spin = QtWidgets.QSpinBox()
        self.newmark_jobs_spin.setRange(-32, 128)
        self.newmark_jobs_spin.setValue(-2)
        self.newmark_jobs_spin.setToolTip("-1 uses all CPUs; -2 uses all minus one.")
        self.newmark_jobs_spin.valueChanged.connect(lambda *_: self._set_dirty())

        # Wave blend (only meaningful for Elevation Z).
        self.wave_blend_cb = QtWidgets.QCheckBox("Blend wave propagation")
        self.wave_blend_cb.setToolTip(
            "During playback the wave field shifts the elevation colour.\n"
            "Disable it for pure topography during the animation."
        )
        self.wave_blend_cb.toggled.connect(lambda *_: self._set_dirty())

        self.wave_blend_strength_spin = QtWidgets.QDoubleSpinBox()
        self.wave_blend_strength_spin.setRange(0.0, 1.0)
        self.wave_blend_strength_spin.setSingleStep(0.05)
        self.wave_blend_strength_spin.setDecimals(2)
        self.wave_blend_strength_spin.setValue(0.5)
        self.wave_blend_strength_spin.setToolTip(
            "How strongly the wave shifts the elevation colour.\n"
            "0.00 = pure topography  ·  1.00 = full elevation range."
        )
        self.wave_blend_strength_spin.valueChanged.connect(lambda *_: self._set_dirty())
        self.wave_blend_cb.toggled.connect(
            lambda checked: self.wave_blend_strength_spin.setEnabled(checked)
        )

        self.apply_button = QtWidgets.QPushButton("Apply")
        # Always clickable — matches the rest of the side panel, where the
        # dirty flag only drives refresh-skipping logic now.
        self.apply_button.setEnabled(True)
        self.apply_button.clicked.connect(self._apply_with_busy)

        form.addRow("Source", self.source_combo)
        form.addRow("Color ramp", _stack(self.cmap_combo, self.cmap_preview))
        form.addRow("Auto range", self.auto_lbl)
        form.addRow("User min", self.vmin_spin)
        form.addRow("User max", self.vmax_spin)
        form.addRow("", self.clamp_cb)
        form.addRow("", self.reset_btn)
        form.addRow("", self.transfer_btn)
        form.addRow("", self.legend_btn)
        form.addRow("T target", self.newmark_T_spin)
        form.addRow("Component", self.newmark_component_combo)
        form.addRow("Data", self.newmark_data_combo)
        form.addRow("Sa type", self.newmark_spectral_combo)
        form.addRow("Factor", self.newmark_factor_spin)
        form.addRow("Jobs", self.newmark_jobs_spin)
        form.addRow("", self.wave_blend_cb)
        form.addRow("Blend strength", self.wave_blend_strength_spin)
        form.addRow("", self.apply_button)

        outer.addWidget(box)
        outer.addStretch(1)
        self.refresh("init")

    # ── Refresh ────────────────────────────────────────────────────────────

    def refresh(self, reason: str = "full") -> None:
        """Pull Color-By state from the session and update every editor widget."""
        if reason not in self._REFRESH_REASONS and self._dirty:
            return
        self._syncing = True
        try:
            current_static = self.session.current_static_color_by()
            self._set_combo_data(self.source_combo, current_static)
            self._set_combo(self.cmap_combo, self.session.current_static_colormap())
            self.cmap_preview.setColormap(self.cmap_combo.currentText())
            lo, hi = self.session.current_static_auto_limits()
            self.auto_lbl.setText(f"{lo:.4g}  /  {hi:.4g}")
            static_vmin, static_vmax = self.session.current_static_user_range()
            if static_vmin is None or static_vmax is None:
                static_vmin, static_vmax = lo, hi
            # Respect the user-locked range — once the user types a
            # value in User min / max we leave the spinboxes alone
            # until they press Reset to auto.
            if not self._range_user_locked:
                self.vmin_spin.setValue(float(static_vmin))
                self.vmax_spin.setValue(float(static_vmax))
            self.clamp_cb.setChecked(self.session.current_static_clamp_enabled())
            newmark = self.session.current_newmark_static_settings()
            self.newmark_T_spin.setValue(float(newmark["T_target"]))
            self._set_combo_data(self.newmark_component_combo, newmark["component"])
            self._set_combo_data(self.newmark_data_combo, newmark["data_type"])
            self._set_combo_data(self.newmark_spectral_combo, newmark["spectral_type"])
            self.newmark_factor_spin.setValue(float(newmark["factor"]))
            self.newmark_jobs_spin.setValue(int(newmark["n_jobs"]))
            block = self.wave_blend_cb.blockSignals(True)
            self.wave_blend_cb.setChecked(self.session.current_wave_blend_enabled())
            self.wave_blend_cb.blockSignals(block)
            block = self.wave_blend_strength_spin.blockSignals(True)
            self.wave_blend_strength_spin.setValue(self.session.current_wave_blend_strength())
            self.wave_blend_strength_spin.blockSignals(block)
            self.wave_blend_strength_spin.setEnabled(self.session.current_wave_blend_enabled())
            self._clear_dirty()

            # Per-source enable / disable.
            allow = not self.session.state.is_playing
            self.source_combo.setEnabled(allow)
            self.cmap_combo.setEnabled(allow)
            self.vmin_spin.setEnabled(allow)
            self.vmax_spin.setEnabled(allow)
            self.clamp_cb.setEnabled(allow)
            self.reset_btn.setEnabled(allow)
            source = self.source_combo.currentData()
            analysis_active = source in ("newmark_sa", "arias")
            for widget in (
                self.newmark_component_combo,
                self.newmark_data_combo,
                self.newmark_factor_spin,
                self.newmark_jobs_spin,
            ):
                widget.setEnabled(allow and analysis_active)
            self.newmark_T_spin.setEnabled(allow and source == "newmark_sa")
            self.newmark_spectral_combo.setEnabled(allow and source == "newmark_sa")
            self.wave_blend_cb.setEnabled(allow)
            self.wave_blend_strength_spin.setEnabled(
                allow and self.session.current_wave_blend_enabled()
            )
            # Apply stays clickable regardless of dirty flag — user
            # explicitly asked for this.  We still respect the playback
            # lock-out (no static-color recompute while animating).
            self.apply_button.setEnabled(allow)
        finally:
            self._syncing = False

    # ── Slots ──────────────────────────────────────────────────────────────

    def _set_dirty(self, dirty: bool = True) -> None:
        if self._syncing:
            return
        self._dirty = bool(dirty)
        # Button stays clickable; flag only drives refresh-skipping.

    def _clear_dirty(self) -> None:
        self._dirty = False
        # Button stays clickable.

    def _on_source_changed(self, *_):
        if self._syncing:
            return
        source = self.source_combo.currentData()
        analysis_active = source in ("newmark_sa", "arias")
        playing = self.session.state.is_playing
        for widget in (
            self.newmark_component_combo,
            self.newmark_data_combo,
            self.newmark_factor_spin,
            self.newmark_jobs_spin,
        ):
            widget.setEnabled(analysis_active and not playing)
        self.newmark_T_spin.setEnabled(source == "newmark_sa" and not playing)
        self.newmark_spectral_combo.setEnabled(source == "newmark_sa" and not playing)
        self._set_dirty(True)

    def _reset_range(self):
        lo, hi = self.session.current_static_auto_limits()
        for spin, val in ((self.vmin_spin, lo), (self.vmax_spin, hi)):
            block = spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(block)
        block = self.clamp_cb.blockSignals(True)
        self.clamp_cb.setChecked(False)
        self.clamp_cb.blockSignals(block)
        # Release the lock so refresh resumes auto-syncing from session state.
        self._range_user_locked = False
        self._set_dirty(True)

    def _on_user_range_edited(self) -> None:
        """User typed a value in User min / max — lock the range and
        flag the section dirty so Apply re-reads our values."""
        if self._syncing:
            return
        self._range_user_locked = True
        self._set_dirty(True)

    def _open_transfer_editor(self):
        dlg = TransferFunctionDialog(self.session, self.window(), initial_colormap=self.cmap_combo.currentText())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        values = dlg.values()
        self.session.apply_transfer_function_preferences(
            colormap=values.colormap,
            inverted=values.invert,
            discrete=values.discrete,
            bins=values.bins,
            nan_color=values.nan_color,
            below_color=values.below_color,
            above_color=values.above_color,
            use_below=values.use_below,
            use_above=values.use_above,
            symmetric_range=values.symmetric_range,
            percentile_clip=values.percentile_clip,
            custom_colormap_object=dlg.build_custom_colormap(),
        )

    def _open_legend_editor(self):
        dlg = LegendEditorDialog(self.session, self.window())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        values = dlg.values()
        self.session.apply_legend_preferences(
            visible=values.visible,
            title=values.title,
            orientation=values.orientation,
            position=values.position,
            label_count=values.label_count,
            label_font_size=values.label_font_size,
            title_font_size=values.title_font_size,
            show_outline=values.show_outline,
            show_background=values.show_background,
        )

    # ── Apply ──────────────────────────────────────────────────────────────

    def _apply(self):
        """Commit the Color-By selection (defers to ``_apply_with_busy`` for heavy work)."""
        if self._syncing:
            return
        self.session.apply_static_color_settings(
            color_by=self.source_combo.currentData(),
            colormap=self.cmap_combo.currentText(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            clamp_enabled=self.clamp_cb.isChecked(),
            wave_blend_enabled=self.wave_blend_cb.isChecked(),
            wave_blend_strength=(
                self.wave_blend_strength_spin.value()
                if self.wave_blend_cb.isChecked()
                else 0.0
            ),
            newmark_T_target=self.newmark_T_spin.value(),
            newmark_component=str(self.newmark_component_combo.currentData()),
            newmark_data_type=str(self.newmark_data_combo.currentData()),
            newmark_spectral_type=str(self.newmark_spectral_combo.currentData()),
            newmark_factor=self.newmark_factor_spin.value(),
            newmark_n_jobs=self.newmark_jobs_spin.value(),
        )
        self._clear_dirty()

    def _apply_with_busy(self):
        source = self.source_combo.currentData()
        if source == "newmark_sa":
            msg = (
                "Computing Newmark surface color...\n"
                "This uses the same parallel cached spectrum path as plot_surface_newmark."
            )
        elif source == "arias":
            msg = (
                "Computing Arias surface color...\n"
                "This uses the same parallel cached intensity path as plot_surface_arias."
            )
        else:
            msg = "Applying color by..."
        self._run_heavy(self._apply, msg)


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

    def __init__(self, session, parent=None):
        """Build the Display form (Field + Color Map + range + Apply)."""
        super().__init__(session, parent)
        # When the user types a value into User min / User max we lock
        # the range — subsequent refreshes leave the spinboxes alone so
        # the typed values stay visible.  "Reset to auto" clears the
        # lock so the spinboxes resume auto-syncing with session state.
        self._range_user_locked = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)
        # Apply-to selector lives in the global toolbar
        # (:class:`~.toolbar.ApplyTargetToolBar`) — every editor dock
        # reads it through ``self._resolve_apply_target()`` so the user
        # has a single visible place to choose where Apply writes.

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
        self.cmap_preview = ColormapPreview(self.cmap_combo.currentText())
        self.cmap_combo.currentTextChanged.connect(self.cmap_preview.setColormap)
        self.cmap_preview.clicked.connect(self._open_transfer_editor)
        # Per-stop marker overrides need to reach the scene; without this
        # the QColorDialog repaints the preview locally but the 3-D
        # viewport and the per-pane state never see the picked colour.
        self.cmap_preview.markerColorChanged.connect(
            self._on_cmap_marker_changed
        )
        cmap_row = _stack(self.cmap_combo, self.cmap_preview)

        self.auto_lbl  = QtWidgets.QLabel("-")
        self.vmin_spin = self._value_spin()
        self.vmax_spin = self._value_spin()
        # Connecting to ``_on_user_range_edited`` (not just ``_set_dirty``)
        # also flips the lock flag so subsequent refreshes leave the
        # spinboxes alone.  ``valueChanged`` does not fire while signals
        # are blocked (i.e. while refresh is auto-populating), so the
        # lock only flips on a real user-driven edit.
        self.vmin_spin.valueChanged.connect(lambda *_: self._on_user_range_edited())
        self.vmax_spin.valueChanged.connect(lambda *_: self._on_user_range_edited())

        self.clamp_cb = QtWidgets.QCheckBox("Clamp to range")
        self.clamp_cb.toggled.connect(lambda *_: self._set_dirty())

        self.reset_btn = QtWidgets.QPushButton("Reset to auto")
        self.reset_btn.clicked.connect(self._reset_range)
        self.transfer_btn = QtWidgets.QPushButton("Advanced color editor")
        self.transfer_btn.clicked.connect(self._open_transfer_editor)
        self.legend_btn = QtWidgets.QPushButton("Legend settings")
        self.legend_btn.clicked.connect(self._open_legend_editor)

        color_form.addRow("Preset",     cmap_row)
        color_form.addRow("Auto range", self.auto_lbl)
        color_form.addRow("User min",   self.vmin_spin)
        color_form.addRow("User max",   self.vmax_spin)
        color_form.addRow("",           self.clamp_cb)
        color_form.addRow("",           self.reset_btn)
        color_form.addRow("",           self.transfer_btn)
        color_form.addRow("",           self.legend_btn)
        outer.addWidget(color_box)

        outer.addWidget(
            self._make_apply_button(self._apply, "Applying display settings…")
        )
        outer.addStretch(1)
        self.refresh("init")

    # ── Sync from session state ───────────────────────────────────────────

    def _effective_state(self):
        """Return the state we should display: the active pane's overrides
        falling through to ``session.state``, or ``session.state`` directly
        when no multi-view / active pane is available.

        Reading from the pane proxy keeps the dock in sync with the pane
        the user is actually editing — otherwise an Apply-to=Active pane
        workflow commits to ``pane_state`` but the dock keeps showing
        ``session.state`` (which the user never touched), so Play / any
        refresh resets the combos back to the global demand.
        """
        w = self.window()
        multi = getattr(w, "multi_view", None) if w is not None else None
        pane = getattr(multi, "active_pane", None) if multi is not None else None
        ps = getattr(pane, "pane_state", None)
        return ps if ps is not None else self.session.state

    def refresh(self, reason: str = "full"):
        """Re-sync demand / component / colormap / range from the active pane.

        Honours the user-locked range: once the user types into the
        vmin / vmax spin boxes the auto-sync stops until the "reset"
        button clears the lock.
        """
        # Vector field and static color have moved to their own docks
        # (VectorFieldSection / StaticColorSection); ignore those reasons here.
        if reason in ("vector_field", "static_color"):
            return
        if reason not in self._REFRESH_REASONS and self._dirty:
            return
        self._syncing = True
        try:
            eff = self._effective_state()
            # Read-only / bounds widgets always sync — they reflect dataset
            # facts, not user choices.
            gf_count = self.session.gf_subfault_count()
            gf_max = max(gf_count - 1, 0)
            self.gf_subfault_spin.setMaximum(gf_max)
            lo, hi = self._field_auto_limits()
            self.auto_lbl.setText(f"{lo:.4g}  /  {hi:.4g}")
            if (self.session.state.user_vmin is None
                    or self.session.state.user_vmax is None):
                self.session.state.set_user_color_range(lo, hi)

            # User-editable widgets: a session-driven refresh while the
            # user is mid-edit (e.g. picked vel, typed limits, then opened
            # the Advanced color editor) must NOT wipe their pending
            # selections back to ``session.state``.  Editable widgets only
            # sync when the section is clean; the user keeps full control
            # until they press Apply (which commits and clears dirty) or
            # Reset to auto.
            if not self._dirty:
                self._set_combo_data(self.demand_combo, eff.demand)
                self._populate_component_combo(
                    eff.demand,
                    selected_component=eff.component,
                )
                self.gf_subfault_spin.setValue(
                    min(self.session.current_display_gf_subfault(), gf_max)
                )
                self.gf_subfault_spin.setEnabled(eff.demand == "gf")
                self._set_combo(
                    self.cmap_combo,
                    eff.colormap or colormap_for_component(eff.component),
                )
                self.cmap_preview.setColormap(self.cmap_combo.currentText())
                # ``_range_user_locked`` is set to True once the user has
                # typed anything into User min / User max.  While locked we
                # keep the typed values in the spinboxes regardless of what
                # the session global says — the user retains full control
                # over the range until they press "Reset to auto".
                if not self._range_user_locked:
                    user_vmin = eff.user_vmin
                    user_vmax = eff.user_vmax
                    if user_vmin is None:
                        user_vmin = lo
                    if user_vmax is None:
                        user_vmax = hi
                    self.vmin_spin.setValue(float(user_vmin))
                    self.vmax_spin.setValue(float(user_vmax))
                self.clamp_cb.setChecked(bool(eff.clamp_enabled))
            else:
                # Keep the GF subfault spinbox enable state in sync with the
                # pending demand selection so the UI stays coherent mid-edit.
                current_demand = str(self.demand_combo.currentData() or "")
                self.gf_subfault_spin.setEnabled(current_demand == "gf")
        finally:
            self._syncing = False

    def _on_user_range_edited(self) -> None:
        """User typed a value in User min / max — lock the range and
        flag the section dirty so Apply re-reads our values."""
        if self._syncing:
            return
        self._range_user_locked = True
        self._set_dirty(True)

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
        # Clear the lock so future refreshes resume auto-syncing the
        # spinboxes with session state until the user edits them again.
        self._range_user_locked = False
        self._set_dirty(True)

    def _open_transfer_editor(self):
        dlg = TransferFunctionDialog(self.session, self.window(), initial_colormap=self.cmap_combo.currentText())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        values = dlg.values()
        custom_cmap = dlg.build_custom_colormap()

        # Step 1 — commit the session-wide transfer-function preferences
        # (invert / discrete / bins / NaN / below / above / symmetric /
        # percentile_clip).  These are NOT per-pane attributes so they
        # always land on the global state.  This also sets
        # session.state.colormap; the per-pane routing below may
        # override it again for the active pane.
        self.session.apply_transfer_function_preferences(
            colormap=values.colormap,
            inverted=values.invert,
            discrete=values.discrete,
            bins=values.bins,
            nan_color=values.nan_color,
            below_color=values.below_color,
            above_color=values.above_color,
            use_below=values.use_below,
            use_above=values.use_above,
            symmetric_range=values.symmetric_range,
            percentile_clip=values.percentile_clip,
            custom_colormap_object=custom_cmap,
        )

        # Step 2 — honour the Apply-to selector for the colormap itself.
        # If the active pane has a stale ``colormap`` override left over
        # from a previous Apply-to=pane, step 1 alone never reaches it,
        # so the Preset combo + viewport keep showing the old preset.
        # Routing through ``_route_apply`` clears the per-pane override
        # for target == "all", or rewrites it for target ∈ {pane, tab}.
        self._route_apply(
            pane_payload={
                "colormap":        values.colormap,
                "colormap_object": custom_cmap,
            },
            global_apply=lambda: None,
        )

        # Step 3 — reflect the new preset in the local combo + preview
        # immediately so the user sees the change even when the section
        # is mid-edit (dirty), which would otherwise suppress the
        # combo update in ``refresh``.
        self._syncing = True
        try:
            self._set_combo(self.cmap_combo, values.colormap)
            self.cmap_preview.setColormap(values.colormap)
        finally:
            self._syncing = False

    def _on_cmap_marker_changed(self, _idx: int, _hex: str) -> None:
        """Push per-stop marker overrides from the preview to the scene.

        ``ColormapPreview`` stores marker overrides locally and emits
        ``markerColorChanged`` after each pick/reset.  Without a listener
        the gradient repaints in the preview only — the 3-D viewport and
        ``pane_state`` never see the change.  Building the custom
        :class:`matplotlib.colors.Colormap` from the preview and routing
        it through ``_route_apply`` keeps everything in sync, honouring
        the Apply-to selector.
        """
        if self._syncing:
            return
        custom_cmap = self.cmap_preview.build_custom_colormap()
        cmap_name = str(self.cmap_combo.currentText())
        self._route_apply(
            pane_payload={
                "colormap":        cmap_name,
                "colormap_object": custom_cmap,
            },
            global_apply=lambda: self._global_apply_custom_cmap(
                cmap_name, custom_cmap
            ),
        )

    def _global_apply_custom_cmap(self, cmap_name: str, custom_cmap) -> None:
        """Commit a colormap (+ optional custom palette) to the session.

        Used by the marker-change path when Apply-to is "all":
        ``apply_transfer_function_preferences`` keeps the rest of the
        session-wide TF preferences as-is while replacing only the
        colormap name and the custom palette object.
        """
        st = self.session.state
        self.session.apply_transfer_function_preferences(
            colormap=cmap_name,
            inverted=bool(getattr(st, "colormap_inverted", False)),
            discrete=bool(getattr(st, "colormap_discrete", False)),
            bins=int(getattr(st, "colormap_bins", 12)),
            nan_color=getattr(st, "nan_color", "#8c8c8c"),
            below_color=getattr(st, "below_range_color", "#15304b"),
            above_color=getattr(st, "above_range_color", "#f3b23f"),
            use_below=bool(getattr(st, "use_below_range_color", False)),
            use_above=bool(getattr(st, "use_above_range_color", False)),
            symmetric_range=bool(getattr(st, "symmetric_color_range", False)),
            percentile_clip=float(getattr(st, "percentile_clip", 0.0)),
            custom_colormap_object=custom_cmap,
        )

    def _open_legend_editor(self):
        dlg = LegendEditorDialog(self.session, self.window())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        values = dlg.values()
        self.session.apply_legend_preferences(
            visible=values.visible,
            title=values.title,
            orientation=values.orientation,
            position=values.position,
            label_count=values.label_count,
            label_font_size=values.label_font_size,
            title_font_size=values.title_font_size,
            show_outline=values.show_outline,
            show_background=values.show_background,
        )

    # ── Single atomic apply ───────────────────────────────────────────────

    def _apply(self):
        """Commit demand / component / colormap / range / GF subfault in one batch."""
        if self._syncing:
            return
        cmap_value = self.cmap_combo.currentText()
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        clamp = self.clamp_cb.isChecked()
        demand = str(self.demand_combo.currentData())
        component = str(self.component_combo.currentData())
        gf_subfault = self.gf_subfault_spin.value()

        # GF subfault is session-global (the HDF5 cache holds one slot
        # at a time; there is no per-pane override).  Commit it on the
        # session BEFORE routing the apply so the scene rebuild that
        # follows reads the new slot — otherwise an Apply-to=Active pane
        # path would write the pane payload but the scene would still
        # query subfault 0 and show stale data.
        if demand == "gf":
            max_subfault = max(0, self.session.gf_subfault_count() - 1)
            self.session._display_gf_subfault = min(
                max(0, int(gf_subfault)), max_subfault
            )

        # ``demand`` MUST be in the pane payload — otherwise an Active
        # pane Apply would keep reading the session global demand and
        # the user ends up with the wrong field rendered (e.g. accel
        # data clamped to the velocity range they typed).
        self._route_apply(
            pane_payload={
                "demand":        demand,
                "component":     component,
                "colormap":      cmap_value,
                "user_vmin":     vmin,
                "user_vmax":     vmax,
                "clamp_enabled": clamp,
            },
            global_apply=lambda: self.session.apply_display_settings(
                demand=demand,
                component=component,
                gf_subfault_id=gf_subfault,
                colormap=cmap_value,
                vmin=vmin,
                vmax=vmax,
                clamp_enabled=clamp,
            ),
        )
        self._clear_dirty()

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
    """Side-panel section with the geometry-visibility toggles (internal / external / QA / labels)."""

    def __init__(self, session, parent=None):
        """Build the visibility checkboxes and the Apply button."""
        super().__init__(session, parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        self.internal_cb = QtWidgets.QCheckBox("Show internal nodes")
        self.external_cb = QtWidgets.QCheckBox("Show external nodes")
        self.qa_cb       = QtWidgets.QCheckBox("Show QA")
        self.selection_labels_cb = QtWidgets.QCheckBox("Label selected nodes")
        for cb in (self.internal_cb, self.external_cb, self.qa_cb):
            cb.toggled.connect(lambda *_: self._set_dirty())
            lay.addWidget(cb)
        self.selection_labels_cb.toggled.connect(self.session.set_selection_labels_enabled)
        lay.addWidget(self.selection_labels_cb)

        lay.addWidget(self._make_apply_button(self._apply, "Updating node visibility…"))
        lay.addStretch(1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        """Pull current visibility flags from the session. Skips when the user has unsaved edits."""
        if reason not in {"init", "full", "visibility"} and self._dirty:
            return
        self._syncing = True
        try:
            self.internal_cb.setChecked(self.session.state.show_internal)
            self.external_cb.setChecked(self.session.state.show_external)
            self.qa_cb.setChecked(self.session.state.show_qa)
            b = self.selection_labels_cb.blockSignals(True)
            self.selection_labels_cb.setChecked(self.session.state.selection_labels_enabled)
            self.selection_labels_cb.blockSignals(b)
            self._clear_dirty()
        finally:
            self._syncing = False

    def _apply(self):
        """Commit the three visibility flags and the Apply-to scope."""
        if self._syncing:
            return
        self.session.apply_visibility_settings(
            show_internal=self.internal_cb.isChecked(),
            show_external=self.external_cb.isChecked(),
            show_qa=self.qa_cb.isChecked(),
        )
        self._clear_dirty()


class WarpSection(_SectionBase):
    """Section that controls the displacement warp (toggle, axes, scale, ghost ref)."""

    def __init__(self, session, parent=None):
        """Build the warp form: enable / ghost / per-axis / scale spin + presets / Apply."""
        super().__init__(session, parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self.warp_cb = QtWidgets.QCheckBox("Enable displacement warp")
        self.warp_cb.toggled.connect(lambda *_: self._set_dirty())
        outer.addWidget(self.warp_cb)
        self.ghost_cb = QtWidgets.QCheckBox("Show undeformed reference cloud")
        self.ghost_cb.toggled.connect(lambda *_: self._set_dirty())
        outer.addWidget(self.ghost_cb)

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
        """Pull warp toggle / axes / scale from the session. Skips when edits are pending."""
        if reason not in {"init", "full", "warp"} and self._dirty:
            return
        self._syncing = True
        try:
            self.warp_cb.setChecked(self.session.state.disp_warp_enabled)
            self.ghost_cb.setChecked(self.session.state.ghost_warp_reference)
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
        """Snap the scale spin to a power-of-ten preset (x10 / x100 / x1k / x10k)."""
        if self._syncing:
            return
        b = self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(float(val))
        self.scale_spin.blockSignals(b)
        self._set_dirty(True)

    def _auto(self):
        """Fill the scale spin with the adapter's auto-suggested value."""
        b = self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(float(self.session.suggested_warp_scale()))
        self.scale_spin.blockSignals(b)
        self._set_dirty(True)

    def _apply(self):
        """Commit warp settings and re-prewarm the disp triplet if needed."""
        if self._syncing:
            return
        warp_enabled = self.warp_cb.isChecked()
        warp_axes = (
            self.x_cb.isChecked(),
            self.y_cb.isChecked(),
            self.z_cb.isChecked(),
        )
        warp_scale = self.scale_spin.value()
        # Ghost reference is a *global* visual hint (single state flag
        # consumed by every scene) — keep it on session.state regardless
        # of target.  The warp parameters themselves honour the
        # Apply-to selector via the shared _route_apply helper.
        try:
            self.session.set_ghost_warp_reference(self.ghost_cb.isChecked())
        except Exception:
            pass

        def _global_apply():
            self.session.apply_warp_settings(
                warp_enabled=warp_enabled,
                warp_axes=warp_axes,
                warp_scale=warp_scale,
            )

        self._route_apply(
            pane_payload={
                "disp_warp_enabled": warp_enabled,
                "warp_axes":         warp_axes,
                "warp_scale":        warp_scale,
            },
            global_apply=_global_apply,
        )
        self._clear_dirty()


class GeographicSection(_SectionBase):
    """UTM coordinate reference + display transform matrix panel."""

    def __init__(self, session, parent=None):
        """Build the UTM form (surface + translate coords) and the 3x3 transform editor."""
        super().__init__(session, parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # ── UTM Coordinates ───────────────────────────────────────────────
        utm_box = QtWidgets.QGroupBox("UTM Coordinates")
        utm_lay = QtWidgets.QVBoxLayout(utm_box)
        utm_lay.setContentsMargins(6, 6, 6, 6)
        utm_lay.setSpacing(4)

        surf_lbl = QtWidgets.QLabel("Surface Coordinate")
        surf_lbl.setStyleSheet("font-size: 11px; font-weight: 600;")
        utm_lay.addWidget(surf_lbl)
        self._surf_spins = self._xyz_spins(utm_lay)

        utm_lay.addSpacing(4)
        trans_lbl = QtWidgets.QLabel("Translate Coordinate")
        trans_lbl.setStyleSheet("font-size: 11px; font-weight: 600;")
        utm_lay.addWidget(trans_lbl)
        self._trans_spins = self._xyz_spins(utm_lay)

        outer.addWidget(utm_box)
        outer.addWidget(self._make_apply_button(self._apply_utm, "Saving UTM reference…"))

        # ── Display Transform ─────────────────────────────────────────────
        tf_box = QtWidgets.QGroupBox("Display Transform")
        tf_lay = QtWidgets.QVBoxLayout(tf_box)
        tf_lay.setContentsMargins(6, 6, 6, 6)
        tf_lay.setSpacing(3)

        self._transform_spins: dict[str, list[QtWidgets.QDoubleSpinBox]] = {}
        for axis in ("X", "Y", "Z"):
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(3)
            axis_lbl = QtWidgets.QLabel(axis)
            axis_lbl.setFixedWidth(12)
            rl.addWidget(axis_lbl)
            spins = []
            for _ in range(3):
                s = self._tf_spin()
                s.valueChanged.connect(self._mark_tf_dirty)
                rl.addWidget(s, 1)
                spins.append(s)
            self._transform_spins[axis] = spins
            tf_lay.addWidget(row)

        self._tf_apply_btn = QtWidgets.QPushButton("Apply")
        self._tf_apply_btn.setEnabled(False)
        self._tf_apply_btn.clicked.connect(self._apply_transform)
        tf_lay.addWidget(self._tf_apply_btn)
        outer.addWidget(tf_box)
        outer.addStretch(1)

        self._load_transform()

    # ── helpers ───────────────────────────────────────────────────────────

    def _xyz_spins(self, parent_layout):
        spins = []
        for lbl in ("X", "Y", "Z"):
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(12, 0, 0, 0)
            rl.setSpacing(4)
            l = QtWidgets.QLabel(lbl)
            l.setFixedWidth(12)
            rl.addWidget(l)
            s = self._coord_spin()
            s.valueChanged.connect(lambda *_: self._set_dirty())
            rl.addWidget(s, 1)
            spins.append(s)
            parent_layout.addWidget(row)
        return spins

    @staticmethod
    def _tf_spin() -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(0)
        s.setRange(-1_000_000.0, 1_000_000.0)
        s.setSingleStep(1.0)
        s.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        return s

    def _mark_tf_dirty(self, *_):
        self._tf_apply_btn.setEnabled(True)

    def _load_transform(self):
        import numpy as np
        try:
            matrix = np.asarray(self.session.current_display_transform(), dtype=float)
        except Exception:
            return
        for row_idx, axis in enumerate(("X", "Y", "Z")):
            for col_idx, spin in enumerate(self._transform_spins[axis]):
                b = spin.blockSignals(True)
                spin.setValue(float(matrix[row_idx, col_idx]))
                spin.blockSignals(b)
        self._tf_apply_btn.setEnabled(False)

    def _apply_utm(self):
        self._clear_dirty()

    def _apply_transform(self):
        import numpy as np
        matrix = np.array(
            [
                [int(s.value()) for s in self._transform_spins["X"]],
                [int(s.value()) for s in self._transform_spins["Y"]],
                [int(s.value()) for s in self._transform_spins["Z"]],
            ],
            dtype=float,
        )
        try:
            self.session.apply_display_transform(matrix)
            self._load_transform()
        except Exception:
            pass

    def refresh(self, reason: str = "full"):
        """Re-read the active display transform from the session into the 3x3 editor."""
        if reason in {"init", "full", "geometry_transform"}:
            self._load_transform()


# ── Main side panel ───────────────────────────────────────────────────────────

class ViewerSidePanel(QtWidgets.QWidget):
    """Vertical nav-menu + ``QStackedWidget`` right panel.

    Menu: Node | Display | Visibility | Warp || Responses | GF
    """

    def __init__(self, session, parent=None):
        """Build the nav menu + stacked pages (lightweight first, lazy heavy ones)."""
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
            elif key == "Geographic":
                nav_lay.addStretch(1)
                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.HLine)
                sep.setFixedHeight(1)
                sep.setStyleSheet("background: #d7dbe2;")
                nav_lay.addWidget(sep)

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
        self._geographic_page = GeographicSection(session)

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
            "Geographic": self._geographic_page,
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
            if self._active_key == "Responses":
                page = self._pages.get("Responses")
                if page is not None:
                    page.refresh("full")
            else:
                self._dirty_heavy.add("Responses")
            return

        # ── All other reasons: lightweight pages always, heavy only if active ──

        # Lightweight pages — cheap Qt-label updates, always safe to refresh.
        self._node_page.refresh(reason)
        self._display_page.refresh(reason)
        self._visibility_page.refresh(reason)
        self._warp_page.refresh(reason)
        self._geographic_page.refresh(reason)

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
        """Keep multi-node compatible analysis available."""
        multi_on = self.session.has_multi_selection()
        responses_btn = self._nav_buttons.get("Responses")
        if responses_btn is not None:
            responses_btn.setEnabled(True)
        gf_btn = self._nav_buttons.get("GF")
        if gf_btn is not None:
            gf_btn.setEnabled(not multi_on)
        if multi_on and self._active_key == "GF":
            self._navigate("Node")


__all__ = ["ViewerSidePanel"]
