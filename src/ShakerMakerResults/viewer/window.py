"""Qt main window for the interactive viewer."""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .controls import HeaderBar, StatusChipBar, TimeControls
from .scene import ViewerScene
from .toolbar import ViewerToolBar
from .trace_panel import AriasIntensityPanel, SpectrumPanel, TracePanel, ViewerPropertiesPanel

_, QtInteractor, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


class ViewerMainWindow(QtWidgets.QMainWindow):
    """Thin Qt shell around the plotter, properties, analysis tabs, and transport controls."""

    def __init__(self, session):
        super().__init__()
        self.session = session
        summary = session.adapter.summary()

        self.setWindowTitle(f"ShakerMaker Results | {summary.name}")
        self.resize(1600, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.header = HeaderBar(session)
        root.addWidget(self.header)

        splitter = QtWidgets.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        self.plotter = QtInteractor(self)
        splitter.addWidget(self.plotter.interactor)

        # Toolbar is created after plotter (needs the interactor reference)
        self.toolbar = ViewerToolBar(self.plotter, self)
        root.addWidget(self.toolbar)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_panel.setChildrenCollapsible(False)

        self.properties_panel = ViewerPropertiesPanel(session)
        right_panel.addWidget(self.properties_panel)

        self.analysis_tabs = QtWidgets.QTabWidget()
        self.trace_panel = TracePanel(session)
        self.spectrum_panel = SpectrumPanel(session)
        self.arias_panel = AriasIntensityPanel(session)
        self.analysis_tabs.addTab(self.trace_panel, "Traces")
        self.analysis_tabs.addTab(self.spectrum_panel, "Spectrum")
        self.analysis_tabs.addTab(self.arias_panel, "Arias Intensity")
        right_panel.addWidget(self.analysis_tabs)
        right_panel.setSizes([420, 460])

        splitter.addWidget(right_panel)
        splitter.setSizes([1240, 360])

        root.addWidget(splitter, 1)

        self.time_controls = TimeControls(session, on_play_toggled=self._on_play_toggled)
        root.addWidget(self.time_controls)

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(80)
        self._play_timer.timeout.connect(self._advance_playback)

        self.scene = ViewerScene(self.plotter, session)
        self.scene.build()

        self.status_chip_bar = StatusChipBar()
        self.statusBar().addPermanentWidget(self.status_chip_bar, 1)
        self._update_status()

        self._space_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        self._space_shortcut.activated.connect(self._toggle_play_shortcut)

    def on_session_updated(self, reason: str):
        self.header.sync_from_state()
        self.time_controls.sync_from_state()
        self.properties_panel.refresh(reason)

        if reason == "time":
            self.scene.refresh_scalars(render=False)
            self.trace_panel.refresh("time")
            self.spectrum_panel.refresh("time")
            self.arias_panel.refresh("time")
        elif reason == "selection":
            self.scene.refresh_selection(render=False)
            self.trace_panel.refresh("selection")
            self.spectrum_panel.refresh("selection")
            self.arias_panel.refresh("selection")
        elif reason in {"demand", "component"}:
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
            self.trace_panel.refresh("demand")
            self.spectrum_panel.refresh("demand")
            self.arias_panel.refresh("demand")
        elif reason == "visibility":
            self.scene.rebuild_for_visibility(render=False)
            self.trace_panel.refresh("full")
            self.spectrum_panel.refresh("full")
            self.arias_panel.refresh("full")
        elif reason == "color_range":
            self.scene.apply_color_range(render=False)
            self.trace_panel.refresh("full")
            self.spectrum_panel.refresh("full")
            self.arias_panel.refresh("full")
        elif reason == "appearance":
            self.scene.apply_appearance(render=False)
            self.scene.refresh_selection(render=False)
            self.trace_panel.refresh("full")
            self.spectrum_panel.refresh("full")
            self.arias_panel.refresh("full")
        elif reason == "playback":
            self._sync_play_state()
        elif reason == "warp":
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
            self.properties_panel.refresh("warp")
        else:
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
            self.trace_panel.refresh("full")
            self.spectrum_panel.refresh("full")
            self.arias_panel.refresh("full")

        self._update_status()
        self.plotter.render()

    def _toggle_play_shortcut(self):
        self.session.toggle_playing()
        self._sync_play_state()

    def _on_play_toggled(self, _is_playing: bool):
        self._sync_play_state()

    def _sync_play_state(self):
        if self.session.state.is_playing:
            self._play_timer.setInterval(self._play_interval_ms())
            self._play_timer.start()
        else:
            self._play_timer.stop()
        self.time_controls.sync_from_state()
        self._update_status()

    def _advance_playback(self):
        max_index = max(len(self.session.adapter.time) - 1, 0)
        if self.session.state.time_index >= max_index:
            self.session.set_playing(False)
            self._sync_play_state()
            self.toolbar.write_frame_if_recording()   # flush last frame then stop
            return
        self.session.step_time(1)
        # step_time → on_session_updated → plotter.render() fires synchronously,
        # so the frame is already rendered when we capture it here.
        self.toolbar.write_frame_if_recording()

    def _update_status(self):
        selected = self.session.state.selected_node
        selected_label = "Node -" if selected is None else f"Node {selected}"
        mode = "clamp" if self.session.state.clamp_enabled else "auto"
        vmin, vmax = self.session.current_color_limits()
        chips = [
            f"⏱ {self.session.current_time():.3f}s",
            f"📊 {self.session.state.demand} · {self.session.state.component}",
            f"🎨 {mode} [{vmin:.3g}, {vmax:.3g}]",
            f"📍 {selected_label}",
            f"💾 {self._cache_summary()}",
        ]
        self.status_chip_bar.update_values(chips)

    def _cache_summary(self) -> str:
        info = self.session.adapter.cache_info
        mb = info["bytes"] / (1024 * 1024)
        return f"cache {mb:.1f} MB"

    def _play_interval_ms(self) -> int:
        speed = max(float(self.session.state.playback_speed), 0.1)
        base_ms = 80.0
        return max(10, int(round(base_ms / speed)))
