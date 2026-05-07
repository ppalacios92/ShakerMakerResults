"""Qt main window for the interactive viewer."""

from __future__ import annotations

import time

from .busy_dialog import BusyDialog
from ._imports import require_viewer_dependencies
from .controls import HeaderBar, StatusChipBar, TimeControls
from .multi_view import MultiViewArea
from .side_panel import ViewerSidePanel
from .theme import LIGHT_PALETTE, build_stylesheet
from .toolbar import ViewerToolBar

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


class ViewerMainWindow(QtWidgets.QMainWindow):
    """Thin Qt shell that wires together the multi-view area, side panel and
    transport controls around a single shared :class:`~.session.ViewerSession`.

    All session state (time, demand, component, warp, selection …) is global.
    ``on_session_updated(reason)`` fans the update out to:

    * ``multi_view``  — refreshes every visible 3-D viewport.
    * ``side_panel``  — routes to the active nav page (lazy heavy pages are
                        skipped when inactive).
    """

    def __init__(self, session):
        super().__init__()
        self.session = session
        self._closing = False
        summary = session.adapter.summary()

        self.setWindowTitle(f"ShakerMaker Results | {summary.name}")
        self.resize(1600, 900)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setStyleSheet(build_stylesheet(LIGHT_PALETTE))

        central = QtWidgets.QWidget()
        central.setObjectName("ViewerCentral")
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────────────
        self.header = HeaderBar(session)
        root.addWidget(self.header)

        # ── Toolbar ───────────────────────────────────────────────────────────
        # Built before multi_view is added to the splitter so it sits above.
        # Deferred addWidget call happens after multi_view is constructed.

        # ── Main splitter: multi-view | side panel ────────────────────────────
        splitter = QtWidgets.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        self.multi_view = MultiViewArea(session)
        splitter.addWidget(self.multi_view)

        self.toolbar = ViewerToolBar(self.multi_view, session, self)
        root.addWidget(self.toolbar)

        self.side_panel = ViewerSidePanel(session)
        splitter.addWidget(self.side_panel)
        splitter.setSizes([1240, 360])

        root.addWidget(splitter, 1)

        # ── Transport controls ────────────────────────────────────────────────
        self.time_controls = TimeControls(session, on_play_toggled=self._on_play_toggled)
        root.addWidget(self.time_controls)

        # ── Playback timer ────────────────────────────────────────────────────
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(16)
        self._play_timer.timeout.connect(self._advance_playback)
        self._playback_last_tick: float | None = None
        self._playback_frame_accumulator: float = 0.0

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_chip_bar = StatusChipBar()
        self.statusBar().addPermanentWidget(self.status_chip_bar, 1)
        self._update_status()

        # ── Keyboard shortcut ─────────────────────────────────────────────────
        self._space_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        self._space_shortcut.activated.connect(self._toggle_play_shortcut)

        # ── Startup data pre-warm ─────────────────────────────────────────────
        # Fire once after the window is fully painted.  Pre-warms the active
        # scalar series into RAM so the first Play press is instant.
        # Uses QTimer.singleShot(0) so the event loop paints the window first.
        QtCore.QTimer.singleShot(0, self._prewarm_on_show)

    # ── Session update routing ────────────────────────────────────────────────

    def on_session_updated(self, reason: str):
        """Fan a state-change reason out to every affected sub-widget."""
        self.header.sync_from_state()
        self.time_controls.sync_from_state()

        if reason == "playback":
            self._sync_play_state()
            self.side_panel.refresh("playback")
        else:
            self.multi_view.on_session_updated(reason)
            self.side_panel.refresh(reason)

        # During frame-by-frame playback only the time chip changes — skip the
        # full status rebuild (color limits, cache MB, etc.) to reduce Qt
        # widget pressure on every animation tick.
        if reason == "time" and self.session.state.is_playing:
            self.status_chip_bar.update_time_chip(
                f"time {self.session.current_time():.3f}s"
            )
        else:
            self._update_status()

    # ── Playback ──────────────────────────────────────────────────────────────

    def _toggle_play_shortcut(self):
        self.session.toggle_playing()
        self._sync_play_state()

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
        max_index = max(len(self.session.adapter.time) - 1, 0)
        if self.session.state.time_index >= max_index:
            self.session.set_playing(False)
            self._sync_play_state()
            self.toolbar.write_frame_if_recording()
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
        # step_time fires on_session_updated("time") → multi_view renders each
        # pane, side_panel throttles the active trace cursor update.
        self.session.step_time(step)
        self.toolbar.write_frame_if_recording()

    # ── Status bar ────────────────────────────────────────────────────────────

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
        ]
        self.status_chip_bar.update_values(chips)

    def _cache_summary(self) -> str:
        info = self.session.adapter.cache_info
        mb = info["bytes"] / (1024 * 1024)
        return f"cache {mb:.1f} MB"

    @staticmethod
    def _play_interval_ms() -> int:
        return 16

    @staticmethod
    def _base_frame_duration_s() -> float:
        return 0.08

    # ── Startup pre-warm ──────────────────────────────────────────────────────

    def _prewarm_on_show(self):
        """Pre-warm the active scalar series once the window is on screen.

        Runs synchronously on the main thread (one HDF5 read).  The status bar
        shows "Ladruno warming data…" for the duration so the user knows why
        the UI is briefly unresponsive for large datasets.

        Skipped when:
        - The window is already closing.
        - The active demand is GF (GF warms lazily / on Play press).
        - The full series would exceed the cache budget (large models fall back
          to per-frame single-column HDF5 reads, which is already fast).
        - The series is already cached (viewer was re-shown after a close).

        Load strategy:
        1. Always try to warm the active demand/component first (primary).
        2. Optionally warm disp for instant Warp — but only when the combined
           total of primary + disp fits within the cache budget.  This prevents
           silent MemoryError on machines where one series fits but six do not.
        3. When the session was opened with field=, skip the disp bonus entirely
           so only the requested field is loaded.
        """
        if self._closing:
            return

        from .adapter import GF_DEMAND

        demand = self.session.state.demand
        if demand == GF_DEMAND:
            return  # GF warmed lazily on first Play press

        one_series_bytes = self.session.adapter._estimated_series_bytes()

        # ── Build primary load list (active demand + component) ───────────────
        comp = self.session.state.component
        if comp == "resultant":
            primary: list[tuple[str, str]] = [(demand, "e"), (demand, "n"), (demand, "z")]
        else:
            primary = [(demand, comp)]

        # Skip already-cached entries so restarts are instant.
        primary = [
            (d, c) for (d, c) in primary
            if (d, c) not in self.session.adapter._series_cache
        ]

        primary_bytes = len(primary) * one_series_bytes
        if primary_bytes > self.session.adapter.max_cache_bytes:
            return  # Even the primary series doesn't fit — per-frame reads are the correct path

        # ── Optionally add disp for instant Warp ─────────────────────────────
        # Only include disp when the COMBINED total (primary + disp) fits within
        # the cache budget.  A per-series check (old behaviour) passed even when
        # loading all six series would exceed available RAM.
        # Skipped entirely when the session was opened with field= so the user
        # gets only what they asked for and nothing extra is loaded.
        series_to_load: list[tuple[str, str]] = list(primary)
        if demand != "disp" and not self.session._field_only:
            disp_candidates = [
                (d, c) for (d, c) in [("disp", "e"), ("disp", "n"), ("disp", "z")]
                if (d, c) not in self.session.adapter._series_cache
            ]
            disp_bytes = len(disp_candidates) * one_series_bytes
            combined_bytes = primary_bytes + disp_bytes
            if combined_bytes <= self.session.adapter.max_cache_bytes:
                series_to_load += disp_candidates

        if not series_to_load:
            return  # Everything already cached

        # ── Show the unified busy dialog ──────────────────────────────────────
        busy = BusyDialog("Loading simulation data...", self, total_steps=len(series_to_load))
        busy.show()
        QtWidgets.QApplication.processEvents()

        try:
            for i, (d, c) in enumerate(series_to_load):
                busy.set_message(
                    f"Loading simulation data...\n"
                    f"Series {i + 1} / {len(series_to_load)}  -  {d} · {c}"
                )
                self.session.adapter.scalar_series(d, c)
                busy.set_step(i + 1)
                QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        finally:
            busy.close()
            self._update_status()

    # ── Window close / cleanup ────────────────────────────────────────────────

    def closeEvent(self, event):  # noqa: N802
        """Release every VTK / OpenGL / RAM resource when the viewer closes.

        Cleanup order
        -------------
        1. Stop the playback timer — no callbacks fire during teardown.
        2. ``toolbar.dispose()`` — stops any active recording cleanly.
        3. ``multi_view.dispose()`` — calls ``scene.dispose()`` on every pane,
           then ``plotter.close()``; releases all OpenGL contexts so subsequent
           PyVista / matplotlib calls in the same process are not blocked.
        4. Disable the Space shortcut so it cannot fire post-close.
        5. ``session._on_window_closed()`` — sets ``is_playing=False``,
           clears the series / spectrum / arias caches, and sets
           ``session.window = None`` so ``show()`` can build a fresh window.

        The ``_closing`` flag prevents re-entrancy when ``WA_DeleteOnClose``
        causes Qt to call ``closeEvent`` a second time.
        """
        if self._closing:
            event.accept()
            return

        self._closing = True
        try:
            try:
                self._play_timer.stop()
            except Exception:
                pass
            try:
                self.toolbar.dispose()
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
