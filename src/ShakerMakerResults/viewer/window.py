"""Qt main window for the interactive viewer — ParaView-style layout.

Layout shape (mirrors ParaView's Pipeline / Properties / Information / View):

    ┌──────── menu / themed top toolbars ────────────────────────────────┐
    │ Camera | View Presets | Overlays | Selection | Capture | Display   │
    ├────────────┬─────────────────────────────────┬─────────────────────┤
    │ Pipeline   │                                  │ Display             │
    │ Properties │         3-D Render Views          │ Appearance          │
    │ Information│         (tabbed multi-view)       │                     │
    │ Responses  │                                  │                     │
    │ Green Func.│                                  │                     │
    ├────────────┴─────────────────────────────────┴─────────────────────┤
    │ ◀◀ ◀ ▶ ▶▶  ━━●━━━━ time ── speed                                  │
    │ status chips                                                       │
    └────────────────────────────────────────────────────────────────────┘

All previously-loose side-panel sections live in dedicated
:class:`QDockWidget` instances.  They can be floated, tabbed against each
other, hidden through ``View → Panels`` or restored via
``View → Reset window layout``.  Dock geometry and the active theme are
persisted to :class:`QtCore.QSettings` between sessions.

Data flow is left untouched — every interaction goes through the same
:class:`ViewerSession` setters used by the legacy side panel.
"""

from __future__ import annotations

import time

from ._imports import require_viewer_dependencies
from .appearance_dock import AppearanceDock
from .busy_dialog import BusyDialog
from .controls import HeaderBar, StatusChipBar, TimeControls
from .information_panel import InformationDock
from .multi_view import TabbedMultiViewArea
from .pipeline_browser import (
    NODE_GEOGRAPHIC,
    NODE_GF,
    NODE_ROOT,
    NODE_SELECTION,
    NODE_STATIONS,
    PipelineBrowserDock,
)
from .properties_dock import PropertiesDock
from .side_panel import (
    DisplaySection,
    StaticColorSection,
    VectorFieldSection,
    WarpSection,
    _LazyPage,
    _PageScrollArea,
    _ResponsesAnalysisTabs,
)
from .theme import (
    build_stylesheet,
    palette_by_name,
    save_theme_name,
    saved_theme_name,
)
from .toolbar import build_viewer_toolbars
from .trace_panel import GFPanel

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


_SETTINGS_ORG = "ShakerMakerResults"
_SETTINGS_APP = "Viewer"
_SETTINGS_GEOMETRY = "window/geometry"
_SETTINGS_STATE = "window/state"


def _settings() -> "QtCore.QSettings":
    return QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)


# ── Dock configuration ───────────────────────────────────────────────────────
#: ``(key, title, area, default_visible)``
#: ``key`` matches the dock factory in ``_init_dock_factories``.
_DOCK_SPECS: list[tuple[str, str, str, bool]] = [
    # Left column — inspection / data context + global appearance.
    # Scene Browser starts hidden by default (it adds little over a click
    # on the Properties tab directly); discoverable through View → Panels.
    ("pipeline",    "Scene Browser",    "left",  False),
    ("properties",  "Properties",       "left",  True),
    ("information", "Information",      "left",  True),
    ("appearance",  "Appearance",       "left",  True),
    ("responses",   "Responses",        "left",  True),
    ("gf",          "Green Functions",  "left",  False),
    # Right column — visual editors that map scene data → pixels
    ("display",      "Display",      "right", True),
    ("vector_field", "Vector Field", "right", False),
    ("color_by",     "Color By",     "right", False),
    ("warp",         "Warp",         "right", True),
]

_DOCK_AREAS = {
    "right":  QtCore.Qt.RightDockWidgetArea,
    "left":   QtCore.Qt.LeftDockWidgetArea,
    "top":    QtCore.Qt.TopDockWidgetArea,
    "bottom": QtCore.Qt.BottomDockWidgetArea,
}


class ViewerMainWindow(QtWidgets.QMainWindow):
    """Main window with ParaView-style docks, themed toolbars and a Pipeline."""

    def __init__(self, session):
        super().__init__()
        self.session = session
        self._closing = False
        self._docks: dict[str, QtWidgets.QDockWidget] = {}
        self._lazy_dock_factories: dict[str, callable] = {}
        self._side_pages: dict[str, QtWidgets.QWidget] = {}
        self._dock_actions: dict[str, QtGui.QAction] = {}
        self._toolbar_actions: dict[str, QtGui.QAction] = {}

        summary = session.adapter.summary()
        self.setWindowTitle(f"ShakerMaker Results | {summary.name}")
        self.resize(1700, 950)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        # Allow nested + tabbed docks (so the user can pin Responses next to
        # the 3-D view, or tab Display behind Appearance on the right).
        self.setDockOptions(
            QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
            | QtWidgets.QMainWindow.AnimatedDocks
            | QtWidgets.QMainWindow.GroupedDragging
        )
        self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)

        # Apply the persisted theme to the entire window before any child
        # widget is constructed so the first paint already reflects it.
        self._current_theme = saved_theme_name()
        self.setStyleSheet(build_stylesheet(palette_by_name(self._current_theme)))

        # Sync the renderer background with the persisted theme on startup
        # so a dark-themed window doesn't open with a bright white viewport.
        # Only adjust when the current preset is theme-tied (White / Dark)
        # — keep "Gray" or anything else as the user left it.
        try:
            current_bg = getattr(session.state, "background", "White")
            if current_bg in ("White", "Dark"):
                session.state.set_background(
                    "Dark" if self._current_theme == "dark" else "White"
                )
        except Exception:
            pass

        # ── Central area: header + tabbed multi-view + transport ─────────────
        central = QtWidgets.QWidget()
        central.setObjectName("ViewerCentral")
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.header = HeaderBar(session)
        root.addWidget(self.header)

        self.multi_view = TabbedMultiViewArea(session)
        root.addWidget(self.multi_view, 1)

        self.time_controls = TimeControls(session)
        root.addWidget(self.time_controls)

        # ── Themed top toolbars ─────────────────────────────────────────────
        # Camera + View Presets together on the first row; the rest break
        # to a second row so a 1600 px window doesn't truncate the icons.
        self._toolbars = build_viewer_toolbars(self.multi_view, session, self)
        for index, tb in enumerate(self._toolbars):
            tb.setMovable(True)
            tb.setFloatable(True)
            tb.setObjectName(f"ViewerToolBar_{tb.windowTitle().replace(' ', '_')}")
            self.addToolBar(QtCore.Qt.TopToolBarArea, tb)
            # Break after the View Presets toolbar so it sits beside Camera
            # and the rest wrap to the next line below.
            if index == 1:
                self.addToolBarBreak(QtCore.Qt.TopToolBarArea)

        self.toolbar = self._toolbars[0] if self._toolbars else None
        # Recording state lives in CaptureToolBar; route playback frame writes.
        self._recording_toolbar = next(
            (tb for tb in self._toolbars if hasattr(tb, "write_frame_if_recording")),
            None,
        )

        # ── Docks ───────────────────────────────────────────────────────────
        self._init_dock_factories()
        for key, title, area, visible in _DOCK_SPECS:
            dock = self._build_dock(key, title)
            self.addDockWidget(_DOCK_AREAS[area], dock)
            dock.setVisible(visible)
        self._apply_default_dock_grouping()

        # Pipeline browser drives the active properties tab — bridge wires it.
        pipeline = self._docks.get("pipeline")
        if pipeline is not None:
            try:
                pipeline.browser.activeNodeChanged.connect(self._on_pipeline_node_changed)
            except Exception:
                pass

        # ── Status bar ──────────────────────────────────────────────────────
        self.status_chip_bar = StatusChipBar()
        self.statusBar().addPermanentWidget(self.status_chip_bar, 1)
        self._update_status()

        # ── Menu bar ────────────────────────────────────────────────────────
        self._build_menu_bar()

        # ── Playback timer ──────────────────────────────────────────────────
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(16)
        self._play_timer.timeout.connect(self._advance_playback)
        self._playback_last_tick: float | None = None
        self._playback_frame_accumulator: float = 0.0
        self._advancing_playback: bool = False
        self._last_render_ms: float = 16.0

        self._space_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        self._space_shortcut.activated.connect(self._toggle_play_shortcut)

        self._prewarming: bool = False

        # Restore persisted geometry + dock layout (silently no-op on first run).
        self._restore_window_state()

        # Fire the visual prewarm once the first paint is on screen.
        QtCore.QTimer.singleShot(0, self._prewarm_on_show)

    # ── Dock factories ──────────────────────────────────────────────────────

    def _init_dock_factories(self) -> None:
        """Register a factory per dock so heavy panels stay lazy until shown."""
        session = self.session

        # Native dock classes — fully self-contained widgets that ALSO act as
        # their own QDockWidget (Pipeline, Properties, Information,
        # Appearance).  These are returned as full docks and bypass the
        # generic ``_build_dock`` wrapping.
        self._lazy_dock_factories["pipeline"]    = lambda parent=self: PipelineBrowserDock(session, parent)
        self._lazy_dock_factories["properties"]  = lambda parent=self: PropertiesDock(session, parent)
        self._lazy_dock_factories["information"] = lambda parent=self: InformationDock(session, parent)
        self._lazy_dock_factories["appearance"]  = lambda parent=self: AppearanceDock(session, parent)

        # Sections that live inside a generic dock wrapper.
        self._lazy_dock_factories["display"]      = lambda: DisplaySection(session)
        self._lazy_dock_factories["vector_field"] = lambda: VectorFieldSection(session)
        self._lazy_dock_factories["color_by"]     = lambda: StaticColorSection(session)
        # Warp used to be a tab inside Properties; promote to its own dock
        # so it lives next to the other visual editors on the right.
        self._lazy_dock_factories["warp"]         = lambda: WarpSection(session)
        self._lazy_dock_factories["responses"]    = lambda: _LazyPage(
            lambda: _ResponsesAnalysisTabs(session)
        )
        self._lazy_dock_factories["gf"]           = lambda: _LazyPage(lambda: GFPanel(session))

    # ── Dock building ───────────────────────────────────────────────────────

    def _build_dock(self, key: str, title: str) -> QtWidgets.QDockWidget:
        """Return the dock for *key*.

        Some factories return a fully-formed ``QDockWidget`` (Pipeline,
        Properties, Information, Appearance) — we use them directly so each
        dock keeps its own internal refresh logic.  Generic content widgets
        (DisplaySection, lazy panels) are wrapped in a thin ``QDockWidget``
        with a scroll area.
        """
        factory = self._lazy_dock_factories[key]
        page_or_dock = factory()

        if isinstance(page_or_dock, QtWidgets.QDockWidget):
            dock = page_or_dock
            # Adopt the factory-built dock as a child of the main window so
            # geometry / save-state work correctly on first show.
            dock.setParent(self)
            dock.setWindowTitle(title)
            dock.setObjectName(f"Dock_{key}")
            inner = dock.widget()
            self._side_pages[key] = inner if inner is not None else dock
        else:
            dock = QtWidgets.QDockWidget(title, self)
            dock.setObjectName(f"Dock_{key}")
            dock.setAllowedAreas(
                QtCore.Qt.LeftDockWidgetArea
                | QtCore.Qt.RightDockWidgetArea
                | QtCore.Qt.BottomDockWidgetArea
            )
            dock.setMinimumWidth(280)
            page = page_or_dock
            self._side_pages[key] = page
            scroll = _PageScrollArea(page)
            dock.setWidget(scroll)

        self._docks[key] = dock
        # Lazy heavy pages materialise on first show.
        dock.visibilityChanged.connect(lambda v, k=key: self._on_dock_visibility(k, v))
        return dock

    def _on_dock_visibility(self, key: str, visible: bool) -> None:
        if not visible:
            return
        page = self._side_pages.get(key)
        if isinstance(page, _LazyPage):
            page.ensure_created()
            page.refresh("full")

    def _apply_default_dock_grouping(self) -> None:
        """Stack secondary panels behind their primary on first launch.

        Left column primary = Properties.  GF stacks behind Responses.
        Right column primary = Display.
        """
        # Left side: keep Pipeline + Properties + Information + Responses
        # individually visible by default.  GF tabs behind Responses.
        responses = self._docks.get("responses")
        gf = self._docks.get("gf")
        if responses is not None and gf is not None:
            self.tabifyDockWidget(responses, gf)
        # Right side: stack Vector Field, Color By and Warp behind
        # Display so they share the right-area real-estate as tabs.
        display = self._docks.get("display")
        for key in ("vector_field", "color_by", "warp"):
            sibling = self._docks.get(key)
            if display is not None and sibling is not None:
                self.tabifyDockWidget(display, sibling)
        if display is not None:
            display.raise_()

        # Left side: tab the secondary panels (Appearance, Responses) under
        # Information so the visible-by-default column doesn't grow taller
        # than the screen.  Pipeline + Properties stay un-tabbed at top.
        information = self._docks.get("information")
        for key in ("appearance", "responses"):
            sibling = self._docks.get(key)
            if information is not None and sibling is not None:
                self.tabifyDockWidget(information, sibling)
        if information is not None:
            information.raise_()

        # Set reasonable initial widths so the central view stays dominant.
        if "pipeline" in self._docks and "display" in self._docks:
            self.resizeDocks(
                [self._docks["pipeline"], self._docks["display"]],
                [320, 320],
                QtCore.Qt.Horizontal,
            )

    # ── Pipeline ↔ Properties bridge ────────────────────────────────────────

    def _on_pipeline_node_changed(self, node_key: str) -> None:
        """Focus the matching Properties tab when the pipeline selection changes."""
        props = self._docks.get("properties")
        if props is None or not hasattr(props, "focus_tab"):
            return
        # Map pipeline node keys → properties tab names.  Warp is no
        # longer a tab here (moved to its own dock on the right), so
        # GF → Visibility as the closest meaningful target.
        tab_for_node = {
            NODE_ROOT:       "Node",
            NODE_STATIONS:   "Node",        # stations table lives inside the Node tab
            NODE_SELECTION:  "Node",
            NODE_GF:         "Visibility",
            NODE_GEOGRAPHIC: "Geographic",
        }
        target = tab_for_node.get(node_key)
        if target is not None:
            try:
                props.focus_tab(target)
                if not props.isVisible():
                    props.setVisible(True)
                    props.raise_()
            except Exception:
                pass

    # ── Menu bar ────────────────────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        menubar = self.menuBar()

        view_menu = menubar.addMenu("&View")

        # Panels submenu — one entry per dock toggling visibility.
        panels_menu = view_menu.addMenu("&Panels")
        for key, title, *_ in _DOCK_SPECS:
            dock = self._docks.get(key)
            if dock is None:
                continue
            action = dock.toggleViewAction()
            action.setText(title)
            panels_menu.addAction(action)
            self._dock_actions[key] = action

        # Toolbars submenu.
        toolbars_menu = view_menu.addMenu("&Toolbars")
        for tb in self._toolbars:
            action = tb.toggleViewAction()
            action.setText(tb.windowTitle())
            toolbars_menu.addAction(action)
            self._toolbar_actions[tb.objectName()] = action

        view_menu.addSeparator()

        theme_menu = view_menu.addMenu("&Theme")
        self._theme_actions: dict[str, QtGui.QAction] = {}
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        for name, label in (("light", "Light"), ("dark", "Dark")):
            act = QtGui.QAction(label, self, checkable=True)
            act.setData(name)
            act.setChecked(self._current_theme == name)
            act.triggered.connect(lambda _checked, n=name: self.set_theme(n))
            group.addAction(act)
            theme_menu.addAction(act)
            self._theme_actions[name] = act

        view_menu.addSeparator()

        reset_act = QtGui.QAction("Reset window layout", self)
        reset_act.triggered.connect(self._reset_window_layout)
        view_menu.addAction(reset_act)

    # ── Theme switching ─────────────────────────────────────────────────────

    def set_theme(self, name: str) -> None:
        """Apply *name* (``"light"`` or ``"dark"``) to the entire UI.

        The renderer background is also flipped so the 3-D viewport never
        stays bright white inside a dark window (or pitch-charcoal inside
        a light window).  Side-panel sections that style themselves through
        the palette at construction time get a fresh ``"full"`` refresh so
        their hard-coded label colours pick up the new theme.
        """
        name = str(name).lower()
        save_theme_name(name)
        self._current_theme = name
        palette = palette_by_name(name)
        self.setStyleSheet(build_stylesheet(palette))

        # Widget-owned colours that bypass the stylesheet.
        try:
            self.multi_view.refresh_theme()
        except Exception:
            pass

        # Renderer background follows the theme.  Only override when the
        # current background is one of the "theme-tied" presets (White / Dark)
        # — leave a custom "Gray" alone so the user's explicit pick is
        # respected.
        try:
            current_bg = getattr(self.session.state, "background", "White")
            if current_bg in ("White", "Dark"):
                self.session.set_background("Dark" if name == "dark" else "White")
        except Exception:
            pass

        # Pipeline browser repaints its icons against the new palette.
        try:
            pipe = self._docks.get("pipeline")
            if pipe is not None and hasattr(pipe, "refresh"):
                pipe.refresh("full")
        except Exception:
            pass

        # Side panels that bake palette colours into setStyleSheet at
        # construction time (Information, Appearance) need a "full" refresh
        # so they read the new active_palette() values.
        for key, page in self._side_pages.items():
            if not hasattr(page, "refresh"):
                continue
            try:
                page.refresh("full")
            except Exception:
                pass

        for n, act in getattr(self, "_theme_actions", {}).items():
            block = act.blockSignals(True)
            act.setChecked(n == name)
            act.blockSignals(block)

        # 3-D viewports — rebroadcast so scalar-bar / branding pick up the
        # new luminance-aware foreground colour and the new background.
        try:
            self.multi_view.on_session_updated("appearance")
        except Exception:
            pass

    # ── Save / restore window state ─────────────────────────────────────────

    def _restore_window_state(self) -> None:
        s = _settings()
        geom = s.value(_SETTINGS_GEOMETRY)
        state = s.value(_SETTINGS_STATE)
        if geom is not None:
            try:
                self.restoreGeometry(geom)
            except Exception:
                pass
        if state is not None:
            try:
                self.restoreState(state)
            except Exception:
                pass

    def _save_window_state(self) -> None:
        s = _settings()
        try:
            s.setValue(_SETTINGS_GEOMETRY, self.saveGeometry())
            s.setValue(_SETTINGS_STATE, self.saveState())
        except Exception:
            pass

    def _reset_window_layout(self) -> None:
        s = _settings()
        s.remove(_SETTINGS_GEOMETRY)
        s.remove(_SETTINGS_STATE)
        for key, title, area, visible in _DOCK_SPECS:
            dock = self._docks.get(key)
            if dock is None:
                continue
            dock.setFloating(False)
            self.addDockWidget(_DOCK_AREAS[area], dock)
            dock.setVisible(visible)
        self._apply_default_dock_grouping()

    # ── Session update routing ──────────────────────────────────────────────

    def on_session_updated(self, reason: str):
        """Broadcast *reason* to every affected sub-widget."""
        self.header.sync_from_state()
        self.time_controls.sync_from_state()

        if reason == "playback":
            if self.session.state.is_playing:
                self.session.adapter.open_playback_handle()
            if (
                self.session.current_static_color_by() == "elevation_z"
                and self.session.current_wave_blend_enabled()
            ):
                self.multi_view.on_session_updated("static_color")
            self._sync_play_state()
            self._refresh_side_pages("playback")
            return

        if reason in ("panel_apply", "demand", "component"):
            from .adapter import GF_DEMAND
            demand = self.session.state.demand
            if demand != GF_DEMAND:
                self._prewarm_display_field(
                    demand, self.session.state.component, no_evict=False
                )
        elif reason == "warp" and self.session.state.disp_warp_enabled:
            disp_ready = all(
                ("disp", component) in self.session.adapter._series_cache
                for component in ("e", "n", "z")
            )
            if not disp_ready:
                self._prewarm_demands(["disp"], no_evict=True)
        elif reason == "vector_field" and self.session.state.vector_field_enabled:
            self._prewarm_demands(
                [self.session.state.vector_field_demand], no_evict=False
            )
        elif reason == "static_color" and self.session.current_wave_blend_enabled():
            from .adapter import GF_DEMAND
            demand = self.session.state.demand
            if demand != GF_DEMAND:
                self._prewarm_display_field(
                    demand, self.session.state.component, no_evict=False
                )

        self.multi_view.on_session_updated(reason)
        self._refresh_side_pages(reason)
        if reason == "time" and self.session.state.is_playing:
            self.status_chip_bar.update_time_chip(
                f"time {self.session.current_time():.3f}s"
            )
        else:
            self._update_status()

    def _refresh_side_pages(self, reason: str) -> None:
        """Refresh every visible side-panel page with *reason*.

        Hidden / collapsed docks are skipped, and lazy pages re-materialise
        themselves only when the user actually shows them.
        """
        for key, page in self._side_pages.items():
            dock = self._docks.get(key)
            if dock is None or not dock.isVisible():
                continue
            try:
                if isinstance(page, _LazyPage):
                    if page.is_created:
                        page.refresh(reason)
                elif hasattr(page, "refresh"):
                    page.refresh(reason)
            except Exception:
                pass

    # ── Playback ────────────────────────────────────────────────────────────

    def _toggle_play_shortcut(self):
        self.session.toggle_playing()

    def _on_play_toggled(self, _is_playing: bool):
        self._sync_play_state()

    def _sync_play_state(self):
        if self.session.state.is_playing:
            self._playback_last_tick = time.perf_counter()
            self._playback_frame_accumulator = 0.0
            self._play_timer.setInterval(self._play_interval_ms())
            self._play_timer.start()
        else:
            self._play_timer.stop()
            self._playback_last_tick = None
            self._playback_frame_accumulator = 0.0
        self.time_controls.sync_from_state()
        self._update_status()

    def _advance_playback(self):
        if self._advancing_playback:
            return
        self._advancing_playback = True
        try:
            return self._advance_playback_once()
        finally:
            self._advancing_playback = False

    def _advance_playback_once(self):
        max_index = max(len(self.session.adapter.time) - 1, 0)
        if self.session.state.time_index >= max_index:
            self.session.set_playing(False)
            self._write_recording_frame()
            return

        now = time.perf_counter()
        last_tick = self._playback_last_tick
        self._playback_last_tick = now
        if last_tick is None:
            return

        elapsed_s = max(0.0, now - last_tick)
        base_frame_s = max(self._base_frame_duration_s(), 1.0e-6)
        speed = max(float(self.session.state.playback_speed), 0.1)
        self._playback_frame_accumulator += (elapsed_s * speed) / base_frame_s

        frames_to_advance = int(self._playback_frame_accumulator)
        if frames_to_advance <= 0:
            return

        self._playback_frame_accumulator -= frames_to_advance
        remaining = max_index - self.session.state.time_index
        step = max(1, min(frames_to_advance, remaining))
        _t0 = time.perf_counter()
        self.session.step_time(step)
        _render_ms = (time.perf_counter() - _t0) * 1000.0
        self._last_render_ms = 0.75 * self._last_render_ms + 0.25 * _render_ms
        self._play_timer.setInterval(self._play_interval_ms())
        self._write_recording_frame()

    def _write_recording_frame(self) -> None:
        rec = self._recording_toolbar
        if rec is None:
            return
        try:
            rec.write_frame_if_recording()
        except Exception:
            pass

    # ── Status bar ──────────────────────────────────────────────────────────

    def _update_status(self):
        selected = self.session.state.selected_node
        selected_label = "Node -" if selected is None else f"Node {selected}"
        mode = "clamp" if self.session.state.clamp_enabled else "auto"
        vmin, vmax = self.session.current_color_limits()
        chips = [
            f"time {self.session.current_time():.3f}s",
            f"{self.session.state.demand} / {self.session.state.component}",
            f"{mode} [{vmin:.3g}, {vmax:.3g}]",
            selected_label,
            self._cache_summary(),
            self._cache_contents_summary(),
        ]
        self.status_chip_bar.update_values(chips)

    def _cache_summary(self) -> str:
        info = self.session.adapter.cache_info
        mb = info["bytes"] / (1024 * 1024)
        budget_mb = info["budget"] / (1024 * 1024)
        return f"cache {mb:.1f}/{budget_mb:.0f} MB"

    def _cache_contents_summary(self) -> str:
        adapter = self.session.adapter
        cache = getattr(adapter, "_series_cache", {})
        component_labels = (("e", "e"), ("n", "n"), ("z", "z"), ("resultant", "r"))
        parts = []
        for demand in ("accel", "vel", "disp"):
            loaded = [short for component, short in component_labels if (demand, component) in cache]
            if loaded:
                parts.append(f"{demand}/{','.join(loaded)}")

        disp_triplet = [short for component, short in component_labels[:3] if ("disp", component) in cache]
        if len(disp_triplet) == 3:
            warp = "warp ready"
        elif disp_triplet:
            warp = f"warp {','.join(disp_triplet)}"
        elif getattr(adapter, "_disp_window_cache", None) is not None:
            warp = "warp window"
        else:
            warp = "warp -"

        loaded = " ".join(parts) if parts else "-"
        return f"loaded {loaded} | {warp}"

    def _play_interval_ms(self) -> int:
        return max(16, int(self._last_render_ms * 1.1))

    @staticmethod
    def _base_frame_duration_s() -> float:
        return 0.08

    # ── Shared prewarm helpers ──────────────────────────────────────────────

    def _prewarm_display_field(self, demand: str, component: str, *, no_evict: bool = False) -> None:
        if self._closing or self._prewarming:
            return
        from .adapter import GF_DEMAND
        if demand == GF_DEMAND:
            return
        adapter = self.session.adapter
        try:
            if (demand, component) in adapter._series_cache:
                return
            adapter.scalar_series(demand, component, no_evict=no_evict)
        except Exception:
            adapter.open_playback_handle()

    def _prewarm_demands(self, demands: list, *, no_evict: bool = False) -> None:
        if self._closing or self._prewarming:
            return

        from .adapter import GF_DEMAND

        adapter = self.session.adapter
        one_series_bytes = adapter._estimated_series_bytes()
        triplet_bytes = 3 * one_series_bytes

        to_load = [
            d for d in demands
            if d != GF_DEMAND
            and triplet_bytes > 0
            and triplet_bytes <= adapter.max_cache_bytes
            and any((d, c) not in adapter._series_cache for c in ("e", "n", "z"))
        ]
        if not to_load:
            return

        total_bytes = len(to_load) * triplet_bytes
        busy = BusyDialog(
            "Preparing simulation data cache...",
            self,
            total_steps=1000,
        )
        busy.show()
        QtWidgets.QApplication.processEvents()

        self._prewarming = True
        bytes_before_demand = 0
        last_events = [time.monotonic()]

        def _make_cb(label: str, demand_start_bytes: int, idx: int, count: int, t_start: float):
            def _cb(done: int, total: int) -> None:
                global_done = demand_start_bytes + done
                elapsed     = time.monotonic() - t_start
                rate        = done / elapsed if elapsed > 0.1 else 0.0
                remaining   = (total - done) / rate if rate > 0 else 0.0

                gb_done  = done  / 1_073_741_824
                gb_total = total / 1_073_741_824
                pct      = done * 100 // total if total > 0 else 0
                cache = adapter.cache_info
                cache_gb = cache["bytes"] / 1_073_741_824
                budget_gb = cache["budget"] / 1_073_741_824

                speed_str = (
                    f"{rate / 1_048_576:.0f} MB/s | ETA {int(remaining)}s"
                    if rate > 0 else "Measuring read speed..."
                )
                busy.set_message(
                    f"Loading {label} vector components [{idx + 1}/{count}]\n"
                    f"Read: {gb_done:.2f} / {gb_total:.2f} GiB ({pct}%)\n"
                    f"{speed_str}\n"
                    f"Cache: {cache_gb:.2f} / {budget_gb:.2f} GiB | precision: float16"
                )
                busy.set_step(
                    int(global_done * 1000 // total_bytes) if total_bytes > 0 else 0
                )
                now = time.monotonic()
                if now - last_events[0] >= 0.10:
                    QtWidgets.QApplication.processEvents()
                    last_events[0] = now
            return _cb

        try:
            for i, d in enumerate(to_load):
                t0 = time.monotonic()
                cb = _make_cb(d, bytes_before_demand, i, len(to_load), t0)
                adapter.prewarm_component_triplet(d, progress_cb=cb, no_evict=no_evict)
                bytes_before_demand += triplet_bytes
                busy.set_step(
                    int(bytes_before_demand * 1000 // total_bytes)
                    if total_bytes > 0 else 1000
                )
                QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        finally:
            self._prewarming = False
            busy.close()
            self._update_status()

    def _prewarm_startup_visual_cache(self) -> None:
        if self._closing or self._prewarming:
            return

        from .adapter import GF_DEMAND

        adapter = self.session.adapter
        one_series_bytes = adapter._estimated_series_bytes()
        if one_series_bytes <= 0 or one_series_bytes > adapter.max_cache_bytes:
            return

        active = self.session.state.demand
        visual_demands: list[str] = []
        for demand in (active, "accel", "vel", "disp"):
            if demand != GF_DEMAND and demand not in visual_demands:
                visual_demands.append(demand)

        visual_jobs = [
            (demand, "resultant")
            for demand in visual_demands
            if (demand, "resultant") not in adapter._series_cache
        ]
        disp_triplet_missing = any(
            ("disp", comp) not in adapter._series_cache for comp in ("e", "n", "z")
        )
        planned_visual_bytes = len(visual_jobs) * one_series_bytes
        free_bytes = adapter.max_cache_bytes - adapter.cache_info["bytes"]
        disp_triplet_fits = (planned_visual_bytes + 3 * one_series_bytes) <= free_bytes
        needs_disp_triplet = disp_triplet_missing and disp_triplet_fits

        if not visual_jobs and not needs_disp_triplet:
            return

        total_units = len(visual_jobs) + (3 if needs_disp_triplet else 0)
        busy = BusyDialog("Preparing visual cache...", self, total_steps=1000)
        busy.show()
        QtWidgets.QApplication.processEvents()

        self._prewarming = True
        completed_units = 0

        def _cache_line() -> str:
            info = adapter.cache_info
            used = info["bytes"] / 1_073_741_824
            budget = info["budget"] / 1_073_741_824
            return f"Cache: {used:.2f} / {budget:.2f} GiB | precision: float16"

        def _set_progress(message: str, units_done: int = completed_units) -> None:
            busy.set_message(message)
            busy.set_step(int(units_done * 1000 // max(total_units, 1)))
            QtWidgets.QApplication.processEvents()

        try:
            for idx, (demand, component) in enumerate(visual_jobs, start=1):
                _set_progress(
                    f"Loading visual field {demand}/{component} [{idx}/{len(visual_jobs)}]\n"
                    f"Estimated field size: {one_series_bytes / 1_073_741_824:.2f} GiB\n"
                    f"{_cache_line()}"
                )
                adapter.scalar_series(demand, component, no_evict=True)
                completed_units += 1
                _set_progress(
                    f"Ready: {demand}/{component}\n"
                    f"{_cache_line()}",
                    completed_units,
                )

            if needs_disp_triplet:
                start_units = completed_units

                def _disp_cb(done: int, total: int) -> None:
                    frac = (done / total) if total > 0 else 0.0
                    units = start_units + int(frac * 3)
                    gb_done = done / 1_073_741_824
                    gb_total = total / 1_073_741_824
                    pct = int(frac * 100)
                    _set_progress(
                        "Preparing warp displacement vectors: disp/E,N,Z\n"
                        f"Read: {gb_done:.2f} / {gb_total:.2f} GiB ({pct}%)\n"
                        f"{_cache_line()}",
                        min(units, total_units),
                    )

                adapter.prewarm_component_triplet("disp", progress_cb=_disp_cb, no_evict=True)
                completed_units = total_units
                _set_progress(
                    "Visual cache ready\n"
                    f"{_cache_line()}",
                    completed_units,
                )
        except Exception:
            try:
                adapter.open_playback_handle()
            except Exception:
                pass
        finally:
            self._prewarming = False
            busy.close()
            self._update_status()

    # ── Startup pre-warm ────────────────────────────────────────────────────

    def _prewarm_on_show(self):
        if self._closing:
            return

        from .adapter import GF_DEMAND

        demand = self.session.state.demand
        if demand == GF_DEMAND:
            return

        self._prewarm_startup_visual_cache()

        try:
            self.session.adapter.open_playback_handle()
        except Exception:
            pass
        self.multi_view.on_session_updated("panel_apply")

    # ── Window close / cleanup ──────────────────────────────────────────────

    def closeEvent(self, event):  # noqa: N802
        if self._closing:
            event.accept()
            return

        self._closing = True
        try:
            self._save_window_state()
        except Exception:
            pass
        try:
            try:
                self._play_timer.stop()
            except Exception:
                pass
            for tb in self._toolbars:
                try:
                    if hasattr(tb, "dispose"):
                        tb.dispose()
                except Exception:
                    pass
            try:
                self.multi_view.dispose()
            except Exception:
                pass
            try:
                self._space_shortcut.setEnabled(False)
            except Exception:
                pass
            try:
                self.session._on_window_closed()
            except Exception:
                pass
        finally:
            super().closeEvent(event)
