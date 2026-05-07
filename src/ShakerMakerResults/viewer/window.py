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
        """Pre-warm demand triplets (E/N/Z) once the window is fully painted.

        Strategy
        --------
        1. Build a load order: active demand first, then ``vel``, then ``disp``
           (deduped).  ``disp`` is always included so Warp is instant.
        2. A full E/N/Z triplet (3 × series_bytes) must fit in the cache budget
           simultaneously.  If it does not, per-frame HDF5 reads handle the
           animation instead (fast on SSD with the persistent handle).
        3. Determine how many triplets fit (``n_fits``); only load the first
           ``n_fits`` demands so we never evict a just-loaded triplet while
           loading the next one.
        4. Show a BusyDialog with real-time GB/s and ETA so the user always
           knows what is happening.

        Skipped when closing or when the active demand is GF (GF warms lazily
        on the first Play press).
        """
        if self._closing:
            return

        from .adapter import GF_DEMAND

        demand = self.session.state.demand
        if demand == GF_DEMAND:
            return

        adapter = self.session.adapter
        one_series_bytes = adapter._estimated_series_bytes()
        triplet_bytes    = 3 * one_series_bytes

        # If even one triplet doesn't fit the budget, skip (per-frame reads win).
        if triplet_bytes == 0 or triplet_bytes > adapter.max_cache_bytes:
            return

        # Build candidate list: active demand first, always add vel + disp.
        candidates: list[str] = [demand]
        for _d in ("vel", "disp"):
            if _d not in candidates:
                candidates.append(_d)

        # How many full triplets fit simultaneously?
        n_fits = max(1, adapter.max_cache_bytes // triplet_bytes)
        candidates = candidates[:n_fits]

        # Skip already fully-cached demands.
        to_load = [
            d for d in candidates
            if any((d, c) not in adapter._series_cache for c in ("e", "n", "z"))
        ]
        if not to_load:
            return

        total_bytes = len(to_load) * triplet_bytes
        busy = BusyDialog(
            "Cargando datos de simulación...",
            self,
            total_steps=1000,   # 0-1000 scale → smooth progress bar
        )
        busy.show()
        QtWidgets.QApplication.processEvents()

        bytes_before_demand = 0
        _last_events = [time.monotonic()]

        def _make_cb(label: str, demand_start_bytes: int, idx: int, count: int, t_start: float):
            def _cb(done: int, total: int) -> None:
                global_done = demand_start_bytes + done
                elapsed     = time.monotonic() - t_start
                rate        = done / elapsed if elapsed > 0.1 else 0.0
                remaining   = (total - done) / rate if rate > 0 else 0.0

                gb_done  = done  / 1_073_741_824
                gb_total = total / 1_073_741_824
                pct      = done * 100 // total if total > 0 else 0

                speed_str = (
                    f"{rate / 1_048_576:.0f} MB/s  —  ETA {int(remaining)}s"
                    if rate > 0 else "midiendo velocidad..."
                )
                busy.set_message(
                    f"Cargando  {label}  [{idx + 1}/{count}]\n"
                    f"{gb_done:.2f} / {gb_total:.2f} GB  ({pct}%)\n"
                    f"{speed_str}"
                )
                global_pct = int(global_done * 1000 // total_bytes) if total_bytes > 0 else 0
                busy.set_step(global_pct)

                now = time.monotonic()
                if now - _last_events[0] >= 0.10:
                    QtWidgets.QApplication.processEvents()
                    _last_events[0] = now
            return _cb

        try:
            for i, d in enumerate(to_load):
                t0 = time.monotonic()
                cb = _make_cb(d, bytes_before_demand, i, len(to_load), t0)
                adapter.prewarm_component_triplet(d, progress_cb=cb)
                bytes_before_demand += triplet_bytes
                busy.set_step(
                    int(bytes_before_demand * 1000 // total_bytes)
                    if total_bytes > 0 else 1000
                )
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
