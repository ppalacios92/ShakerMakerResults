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
        self.time_controls = TimeControls(session)
        root.addWidget(self.time_controls)

        # ── Playback timer ────────────────────────────────────────────────────
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(16)
        self._play_timer.timeout.connect(self._advance_playback)
        self._playback_last_tick: float | None = None
        self._playback_frame_accumulator: float = 0.0
        self._advancing_playback: bool = False
        self._last_render_ms: float = 16.0

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_chip_bar = StatusChipBar()
        self.statusBar().addPermanentWidget(self.status_chip_bar, 1)
        self._update_status()

        # ── Keyboard shortcut ─────────────────────────────────────────────────
        self._space_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        self._space_shortcut.activated.connect(self._toggle_play_shortcut)

        # Guard flag: prevents re-entrant prewarm calls while one load is
        # already running.
        self._prewarming: bool = False

        # ── Startup data pre-warm ─────────────────────────────────────────────
        # Fire once after the window is fully painted.  Pre-warms the active
        # scalar series into RAM so the first Play press is instant.
        # Uses QTimer.singleShot(0) so the event loop paints the window first.
        QtCore.QTimer.singleShot(0, self._prewarm_on_show)

    # ── Session update routing ────────────────────────────────────────────────

    def on_session_updated(self, reason: str):
        """Fan a state-change reason out to every affected sub-widget.

        For reasons that change the active data field (``panel_apply``,
        ``demand``, ``component``, ``warp``, ``vector_field``, ``playback``),
        ``_prewarm_demands`` is called first so the 3-D render always hits the
        cache rather than blocking on an HDF5 read.
        """
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
            self.side_panel.refresh("playback")
            return
            # Prewarm the display demand BEFORE starting the play timer so the
            # first frame is served from cache and animation is smooth.
            #
            # NOTE: displacement (disp) for warp is intentionally NOT included
            # here.  For large models a single E/N/Z triplet already fills the
            # cache budget, so adding "disp" would evict the display demand via
            # LRU on the next load — causing infinite thrashing.  Instead,
            # apply_warp_settings opens a persistent HDF5 handle that makes
            # per-frame displacement reads just as fast as a cache hit.
            if self.session.state.is_playing:
                from .adapter import GF_DEMAND
                demand = self.session.state.demand
                if demand != GF_DEMAND:
                    needs = [demand]
                    # Only include disp alongside the display demand when the
                    # cache budget can hold both triplets simultaneously
                    # (budget ≥ 6 × one_series_bytes).
                    if self.session.state.disp_warp_enabled and demand != "disp":
                        one = self.session.adapter._estimated_series_bytes()
                        if 6 * one <= self.session.adapter.max_cache_bytes:
                            needs.append("disp")
                    self._prewarm_demands(needs)
            self._sync_play_state()
            self.side_panel.refresh("playback")
        else:
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
            self.side_panel.refresh(reason)
            if reason == "time" and self.session.state.is_playing:
                self.status_chip_bar.update_time_chip(
                    f"time {self.session.current_time():.3f}s"
                )
            else:
                self._update_status()
            return
            # Prewarm whenever the active field or warp state changes so the
            # following 3-D rebuild reads from RAM, not from HDF5.
            if reason in ("panel_apply", "demand", "component"):
                from .adapter import GF_DEMAND
                demand = self.session.state.demand
                if demand != GF_DEMAND:
                    needs = [demand]
                    # Same "2-triplet coexistence" guard as above.
                    if self.session.state.disp_warp_enabled and demand != "disp":
                        one = self.session.adapter._estimated_series_bytes()
                        if 6 * one <= self.session.adapter.max_cache_bytes:
                            needs.append("disp")
                    self._prewarm_demands(needs)
            elif reason == "warp" and self.session.state.disp_warp_enabled:
                # Only prewarm disp when it can coexist with the display
                # demand in the cache (budget ≥ 6 × one_series_bytes).
                # When it doesn't fit, the persistent HDF5 handle opened in
                # apply_warp_settings is used for fast per-frame reads instead.
                from .adapter import GF_DEMAND
                demand = self.session.state.demand
                if demand != GF_DEMAND:
                    one = self.session.adapter._estimated_series_bytes()
                    if 6 * one <= self.session.adapter.max_cache_bytes:
                        self._prewarm_demands([demand, "disp"])
            elif reason == "vector_field" and self.session.state.vector_field_enabled:
                self._prewarm_demands([self.session.state.vector_field_demand])
            elif reason == "static_color" and self.session.current_wave_blend_enabled():
                # Wave blend just activated — prewarm the wave demand.
                from .adapter import GF_DEMAND
                demand = self.session.state.demand
                if demand != GF_DEMAND:
                    needs = [demand]
                    if self.session.state.disp_warp_enabled and demand != "disp":
                        one = self.session.adapter._estimated_series_bytes()
                        if 6 * one <= self.session.adapter.max_cache_bytes:
                            needs.append("disp")
                    self._prewarm_demands(needs)

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
        return

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
        _t0 = time.perf_counter()
        self.session.step_time(step)
        _render_ms = (time.perf_counter() - _t0) * 1000.0
        self._last_render_ms = 0.75 * self._last_render_ms + 0.25 * _render_ms
        self._play_timer.setInterval(self._play_interval_ms())
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

    # ── Shared prewarm helper ─────────────────────────────────────────────────

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
        """Load *demands* E/N/Z triplets into RAM, showing a BusyDialog.

        Skips demands that are already fully cached, don't fit the budget, or
        are the special GF demand (which is warmed lazily on demand).
        Re-entrant calls while a prewarm is running are silently ignored.
        """
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
        """Warm startup visual fields without forcing every demand to E/N/Z.

        The visual cache should make Apply instant for the common scalar fields
        while still preparing displacement components for warp when memory
        allows it.  Resultant fields cost one cached matrix each; full triplets
        cost three, so only ``disp`` gets the triplet treatment for warp.
        """
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

    # ── Startup pre-warm ──────────────────────────────────────────────────────

    def _prewarm_on_show(self):
        """Pre-warm visual fields and warp displacement once the window is painted.

        Skipped when closing or when the active demand is GF (GF warms lazily
        on the first Play press).
        """
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
