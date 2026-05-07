"""Main orchestration object for an interactive viewer session."""

from __future__ import annotations

from .adapter import GF_DEMAND, REGULAR_DEMANDS, ViewerDataAdapter
from .colors import BACKGROUND_PRESETS, colormap_for_component, scalar_limits
from .state import ViewerState

VALID_STATIC_COLOR_BY = ("elevation_z",)


class ViewerSession:
    """Coordinate adapter, state, and optional GUI window."""

    def __init__(
        self,
        model_or_adapter,
        *,
        show: bool = False,
        field: str | None = None,
        demand: str = "accel",
        component: str = "resultant",
        time_index: int = 0,
        selected_node=None,
        title: str | None = None,
        cache_time_series: bool = True,
        max_cache_bytes: int | None = None,
        max_cache_entries: int = 8,
    ) -> None:
        # ── field= shorthand ──────────────────────────────────────────────────
        # ``field='vel'`` is sugar for ``demand='vel', component='resultant'``
        # and tells the pre-warm to load ONLY that demand (skipping the bonus
        # disp series that the normal path adds for instant Warp).
        if field is not None:
            demand = str(field).lower()
            component = "resultant"
        self._field_only: bool = field is not None

        if isinstance(model_or_adapter, ViewerDataAdapter):
            self.adapter = model_or_adapter
        else:
            self.adapter = ViewerDataAdapter(
                model_or_adapter,
                cache_time_series=cache_time_series,
                max_cache_bytes=max_cache_bytes,
                max_cache_entries=max_cache_entries,
            )

        self._static_color_by: str | None = None
        self._static_color_map: str = "terrain"
        self._static_clamp_enabled: bool = False
        self._static_user_vmin: float | None = None
        self._static_user_vmax: float | None = None
        max_index = max(len(self.adapter.time) - 1, 0)
        self.state = ViewerState(
            time_index=time_index,
            demand=demand,
            component=component,
            selected_node=selected_node,
        )
        self.state.set_time_index(self.state.time_index, max_index)
        self.state.set_user_color_range(*self.default_color_limits())

        # Lock in the auto point-size at session creation so actor rebuilds
        # (demand/visibility changes) never silently change the rendered size.
        # Users can still override it at any time via the point-size slider.
        if self.state.point_size is None:
            self.state.set_point_size(self.suggested_point_size())

        self.title = title or self.adapter.summary().name
        self.window = None
        self._qt_app = None
        self._owns_qt_app = False
        self._closing = False
        self._closed = False
        self._station_tags: list[dict[str, object]] = []
        self._show_station_tags = True
        self._display_gf_subfault: int = 0
        self._prev_multi_selection: frozenset = frozenset()

        if show:
            self.show()

    def show(self, *, start_event_loop: bool | None = None):
        """Build and show the GUI window."""
        import os
        import sys

        # PyVistaQt + PyQt5 is unstable on some Wayland sessions and may crash
        # with a fatal X BadWindow during startup. Prefer xcb on Linux unless
        # the user explicitly set another stable platform.
        if sys.platform.startswith("linux"):
            qpa = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
            if qpa == "" or qpa.startswith("wayland"):
                os.environ["QT_QPA_PLATFORM"] = "xcb"

        # Local viewer style override.
        # Must happen BEFORE QApplication is created.
        if os.environ.get("QT_STYLE_OVERRIDE", "").lower() == "kvantum":
            os.environ["QT_STYLE_OVERRIDE"] = "Fusion"

        from .window import ViewerMainWindow
        from ._imports import require_viewer_dependencies

        _, _, _, _, _, QtWidgets = require_viewer_dependencies()

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
            self._owns_qt_app = True

        self._qt_app = app

        try:
            if "Fusion" in QtWidgets.QStyleFactory.keys():
                app.setStyle("Fusion")
        except Exception:
            pass

        if self.window is None:
            self.window = ViewerMainWindow(self)
        else:
            # The Qt C++ object may have been deleted (e.g. the user closed the
            # window but the Python reference wasn't cleared).  Detect this via
            # a lightweight attribute probe and recreate if needed.
            try:
                _ = self.window.isVisible()
            except RuntimeError:
                # "Internal C++ object already deleted."
                self.window = ViewerMainWindow(self)
        self.window.show()

        # Some Linux/X11 environments can crash with a fatal X BadWindow error
        # when forcing raise_/activate immediately after show().
        if not sys.platform.startswith("linux"):
            self.window.raise_()
            self.window.activateWindow()

        if start_event_loop is None:
            start_event_loop = self._owns_qt_app
        if start_event_loop:
            exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
            if exec_fn is not None:
                exec_fn()
        return self

    def close(self) -> None:
        """Explicitly tear down the GUI session and release viewer resources."""
        if self._closed or self._closing:
            return

        self._closing = True
        try:
            if self.window is not None:
                try:
                    self.window.close()
                except Exception:
                    pass
                if self.window is not None:
                    self._on_window_closed()
            else:
                self._on_window_closed()
        finally:
            self._closing = False

    def _on_window_closed(self) -> None:
        """Finalize the session after the Qt window has been closed."""
        if self._closed:
            return

        self.state.set_playing(False)
        self.window = None
        self.adapter.clear_runtime_caches()
        self._qt_app = None
        self._owns_qt_app = False
        self._closed = True

    def set_time_index(self, time_index: int):
        self.state.set_time_index(time_index, max(len(self.adapter.time) - 1, 0))
        self._notify_window("time")
        return self.state.time_index

    def set_demand(self, demand: str):
        self.state.set_demand(demand)
        # Auto-correct the component when crossing between regular and GF demand.
        # e.g. "resultant" is invalid for GF;  "g11" is invalid for accel/vel/disp.
        available = self.adapter.available_components_for_demand(demand)
        if self.state.component not in available:
            self.state.set_component(available[0])
        self.state.set_user_color_range(*self.default_color_limits())
        self._notify_window("demand")
        return self.state.demand

    def set_component(self, component: str):
        self.state.set_component(component)
        self.state.set_user_color_range(*self.default_color_limits())
        self._notify_window("component")
        return self.state.component

    def select_node(self, node_id):
        self.state.set_selected_node(node_id)
        self.state.multi_selection = frozenset()
        self.state.set_selection_visibility("all")
        self._notify_window("selection")
        self._notify_window("multi_selection")
        return self.state.selected_node

    def select_nearest_coordinate_m(self, x_m: float, y_m: float, z_m: float):
        node_id, distance_m = self.adapter.nearest_node_from_model_xyz_m([x_m, y_m, z_m])
        self.select_node(node_id)
        return node_id, distance_m

    def select_nearest_display_coordinate_m(self, x_m: float, y_m: float, z_m: float):
        import numpy as np

        point = [x_m, y_m, z_m]
        node_id = self.adapter.nearest_node_id(point)
        node_point = self.adapter.point_for_node(node_id)
        distance_m = float(np.linalg.norm(np.asarray(node_point, dtype=float) - np.asarray(point, dtype=float)))
        self.select_node(node_id)
        return node_id, distance_m

    def clear_selection(self):
        self.state.set_selected_node(None)
        self.state.multi_selection = frozenset()
        self.state.set_selection_visibility("all")
        self._notify_window("selection")
        self._notify_window("multi_selection")

    # ── Multi-node selection (visualization only) ─────────────────────────────

    def set_selection_visibility(self, mode: str):
        self.state.set_selection_visibility(mode)
        self._notify_window("multi_selection")
        return self.state.selection_visibility

    def set_node_opacity(self, opacity: float):
        self.state.set_node_opacity(opacity)
        self._notify_window("node_opacity")
        return self.state.node_opacity

    def apply_selection_filter(self, mode: str):
        """Apply a visibility filter using the current selection set."""
        if self.state.multi_selection:
            self._prev_multi_selection = frozenset(self.state.multi_selection)
        self.state.set_selection_visibility(mode)
        self._notify_window("multi_selection")

    def restore_prev_selection(self):
        """Restore the selection set saved before the last filter was applied."""
        self.state.multi_selection = frozenset(self._prev_multi_selection)
        self.state.set_selection_visibility("all")
        self._notify_window("multi_selection")

    def add_nodes_to_multi_selection(self, node_ids):
        """Add a batch of nodes to the selection set, then notify once."""
        selected = set(self.state.multi_selection)
        for nid in node_ids:
            selected.add(nid)
        self.state.multi_selection = frozenset(selected)
        self.state.set_selected_node(None)
        self.state.set_selection_visibility("all")
        self._notify_window("selection")
        self._notify_window("multi_selection")
        return self.state.multi_selection

    def has_multi_selection(self) -> bool:
        return bool(self.state.multi_selection)

    def current_visible_node_ids(self) -> list:
        return self.adapter.visible_node_ids

    def set_background(self, background: str):
        self.state.set_background(background)
        self._notify_window("appearance")
        return self.state.background

    def set_colormap(self, colormap: str):
        self.state.set_colormap(colormap)
        self._notify_window("appearance")
        return self.state.colormap

    def set_point_size(self, point_size: float | None):
        self.state.set_point_size(point_size)
        self._notify_window("point_size")
        return self.state.point_size

    def set_show_scalar_bar(self, show_scalar_bar: bool):
        self.state.set_show_scalar_bar(show_scalar_bar)
        self._notify_window("appearance")
        return self.state.show_scalar_bar

    def set_color_range(self, vmin: float | None, vmax: float | None, *, clamp: bool | None = None):
        self.state.set_user_color_range(vmin, vmax)
        if clamp is not None:
            self.state.set_clamp_enabled(clamp)
        self._notify_window("color_range")
        return self.state.user_vmin, self.state.user_vmax

    def set_clamp_enabled(self, enabled: bool):
        self.state.set_clamp_enabled(enabled)
        self._notify_window("color_range")
        return self.state.clamp_enabled

    def set_node_visibility(
        self,
        *,
        show_internal: bool | None = None,
        show_external: bool | None = None,
        show_qa: bool | None = None,
    ):
        self.state.set_node_visibility(
            show_internal=show_internal,
            show_external=show_external,
            show_qa=show_qa,
        )
        self._notify_window("visibility")
        return self.state.show_internal, self.state.show_external, self.state.show_qa

    def set_warp_enabled(self, enabled: bool):
        """Enable or disable 3-D displacement warp."""
        self.state.set_warp_enabled(enabled)
        if enabled:
            try:
                self.adapter.prewarm_component_triplet("disp")
            except Exception:
                pass
        self._notify_window("warp")
        return self.state.disp_warp_enabled

    def set_warp_axes(self, *, x: bool | None = None, y: bool | None = None, z: bool | None = None):
        """Toggle which spatial axes participate in the warp displacement."""
        axes = list(self.state.warp_axes)
        if x is not None:
            axes[0] = bool(x)
        if y is not None:
            axes[1] = bool(y)
        if z is not None:
            axes[2] = bool(z)
        self.state.set_warp_axes(tuple(axes))
        self._notify_window("warp")
        return self.state.warp_axes

    def set_warp_scale(self, scale: float | None):
        """Set the displacement exaggeration factor (None = auto)."""
        self.state.set_warp_scale(scale)
        self._notify_window("warp")
        return self.state.warp_scale

    def apply_display_settings(
        self,
        *,
        demand: str,
        component: str,
        gf_subfault_id: int | None = None,
        colormap: str,
        vmin: float | None,
        vmax: float | None,
        clamp_enabled: bool,
    ):
        """Apply field + colour-map settings atomically in a single 3-D rebuild.

        Replaces the old two-step ``apply_data_settings`` + ``apply_color_settings``
        pattern used by the (now merged) ``DisplaySection`` panel.
        """
        self.state.set_demand(demand)
        self.state.set_component(component)
        if gf_subfault_id is not None:
            # TODO: Keep Display -> Green Functions aligned with the GF+MAP
            # subfault domain exposed by the adapter, not with raw slot ids.
            max_subfault = max(0, self.gf_subfault_count() - 1)
            self._display_gf_subfault = min(max(0, int(gf_subfault_id)), max_subfault)
        self.state.set_colormap(colormap)
        self.state.set_user_color_range(vmin, vmax)
        self.state.set_clamp_enabled(clamp_enabled)

        # Pre-warm the requested series so the scene rebuild is pure NumPy
        # (no HDF5 I/O during the VTK render call that follows).
        if demand != GF_DEMAND:
            if component == "resultant":
                try:
                    self.adapter.prewarm_component_triplet(demand)
                except Exception:
                    pass
            else:
                try:
                    self.adapter.scalar_series(demand, component)
                except Exception:
                    pass

        # "panel_apply" triggers a full scalar-actor rebuild that picks up every
        # state change above in one render pass.
        self._notify_window("panel_apply")
        return self.state.demand, self.state.component

    def apply_static_color_settings(
        self,
        *,
        color_by: str,
        colormap: str,
        vmin: float | None,
        vmax: float | None,
        clamp_enabled: bool,
    ):
        self._static_color_by = self._validate_static_color_by(color_by)
        self._static_color_map = str(colormap).strip() or self._static_color_map
        self._static_user_vmin = None if vmin is None else float(vmin)
        self._static_user_vmax = None if vmax is None else float(vmax)
        self._static_clamp_enabled = bool(clamp_enabled)
        self._notify_window("static_color")
        return self._static_color_by, self._static_color_map

    def current_display_gf_subfault(self) -> int:
        return int(self._display_gf_subfault)

    def current_static_auto_limits(self) -> tuple[float, float]:
        if self._static_color_by == "elevation_z":
            return self.adapter.elevation_limits()
        return self.adapter.elevation_limits()

    def current_static_color_limits(self, scalars=None) -> tuple[float, float]:
        if self._static_clamp_enabled and self._static_user_vmin is not None and self._static_user_vmax is not None:
            vmin = float(self._static_user_vmin)
            vmax = float(self._static_user_vmax)
            if vmax <= vmin:
                return vmin, vmin + 1.0
            return vmin, vmax
        if scalars is None:
            return self.current_static_auto_limits()
        return self.current_static_auto_limits()

    def apply_data_settings(self, *, demand: str, component: str):
        self.state.set_demand(demand)
        self.state.set_component(component)
        self.state.set_user_color_range(*self.default_color_limits())
        self._notify_window("demand")
        return self.state.demand, self.state.component

    def apply_color_settings(
        self,
        *,
        colormap: str,
        vmin: float | None,
        vmax: float | None,
        clamp_enabled: bool,
    ):
        self.state.set_colormap(colormap)
        self.state.set_user_color_range(vmin, vmax)
        self.state.set_clamp_enabled(clamp_enabled)
        self._notify_window("appearance")
        return self.state.colormap, self.state.user_vmin, self.state.user_vmax

    def apply_visibility_settings(
        self,
        *,
        show_internal: bool,
        show_external: bool,
        show_qa: bool,
    ):
        self.state.set_node_visibility(
            show_internal=show_internal,
            show_external=show_external,
            show_qa=show_qa,
        )
        self._notify_window("visibility")
        return self.state.show_internal, self.state.show_external, self.state.show_qa

    def apply_warp_settings(
        self,
        *,
        warp_enabled: bool,
        warp_axes: tuple[bool, bool, bool],
        warp_scale: float | None,
    ):
        was_warp_enabled = bool(self.state.disp_warp_enabled)
        self.state.set_warp_enabled(warp_enabled)
        self.state.set_warp_axes(tuple(bool(v) for v in warp_axes))
        self.state.set_warp_scale(warp_scale)

        if warp_enabled and not was_warp_enabled:
            try:
                self.adapter.prewarm_component_triplet("disp")
            except Exception:
                pass

        self._notify_window("warp")
        return self.state.disp_warp_enabled, self.state.warp_axes, self.state.warp_scale

    # ── Vector-field overlay ──────────────────────────────────────────────────

    def apply_vector_field_settings(
        self,
        *,
        enabled: bool,
        demand: str,
        scale: float,
        colormap: str = "viridis",
    ):
        """Pre-warm the 3 vector components then toggle the arrow overlay.

        The pre-warm reads E, N, Z series into the adapter cache so the first
        ``refresh_vector_field()`` in the scene is pure NumPy with no HDF5 I/O.
        """
        demand_lower = str(demand).lower()
        if demand_lower not in ("accel", "vel", "disp"):
            demand_lower = "disp"
        self.state.vector_field_enabled = bool(enabled)
        self.state.vector_field_demand = demand_lower
        self.state.vector_field_scale = max(0.01, float(scale))
        self.state.vector_field_colormap = str(colormap) or "viridis"
        if enabled:
            try:
                self.adapter.prewarm_component_triplet(demand_lower)
            except Exception:
                pass
        self._notify_window("vector_field")

    def current_vector_data(self):
        """Return ``(points, vectors)`` for the current time step.

        ``points`` is an N×3 float array of visible node positions.  When 3-D
        warp is active the warped positions are returned so that every arrow
        originates from the displaced node, not from its rest position.
        ``vectors`` is an N×3 float array with columns [E, N, Z] of the active
        vector-field demand.
        """
        import numpy as np

        demand = self.state.vector_field_demand
        t = self.state.time_index
        kwargs = dict(
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )
        e = self.adapter.visible_scalars(
            self.adapter.scalar_snapshot(t, demand, "e"), **kwargs
        )
        n = self.adapter.visible_scalars(
            self.adapter.scalar_snapshot(t, demand, "n"), **kwargs
        )
        z = self.adapter.visible_scalars(
            self.adapter.scalar_snapshot(t, demand, "z"), **kwargs
        )
        # Use warped positions when warp is active so the arrow bases track the
        # displaced geometry.  Falls back to base positions when warp is off.
        points = self.current_warped_points()
        vectors = np.stack([e, n, z], axis=1)
        return points, vectors

    def apply_panel_settings(
        self,
        *,
        demand: str,
        component: str,
        colormap: str,
        vmin: float | None,
        vmax: float | None,
        clamp_enabled: bool,
        show_internal: bool,
        show_external: bool,
        show_qa: bool,
        warp_enabled: bool,
        warp_axes: tuple[bool, bool, bool],
        warp_scale: float | None,
    ):
        """Apply right-panel settings in one batch refresh."""
        was_warp_enabled = bool(self.state.disp_warp_enabled)

        self.state.set_demand(demand)
        self.state.set_component(component)
        self.state.set_colormap(colormap)
        self.state.set_user_color_range(vmin, vmax)
        self.state.set_clamp_enabled(clamp_enabled)
        self.state.set_node_visibility(
            show_internal=show_internal,
            show_external=show_external,
            show_qa=show_qa,
        )
        self.state.set_warp_enabled(warp_enabled)
        self.state.set_warp_axes(tuple(bool(v) for v in warp_axes))
        self.state.set_warp_scale(warp_scale)

        if warp_enabled and not was_warp_enabled:
            try:
                self.adapter.prewarm_component_triplet("disp")
            except Exception:
                pass

        self._notify_window("panel_apply")
        return {
            "demand": self.state.demand,
            "component": self.state.component,
            "colormap": self.state.colormap,
            "clamp_enabled": self.state.clamp_enabled,
            "warp_enabled": self.state.disp_warp_enabled,
        }

    def suggested_warp_scale(self) -> float:
        """Return the adapter's auto-suggested warp scale for this dataset."""
        return self.adapter.suggested_warp_scale()

    def set_playing(self, is_playing: bool):
        had_static_color_override = bool(self._static_color_by)
        if is_playing:
            self._static_color_by = None
        if is_playing and not self.state.is_playing:
            if self.state.demand == GF_DEMAND:
                # ── GF demand: pre-warm the full (n_nodes, nt_gf) series in
                # one HDF5 read so every animation frame is pure NumPy.
                # GF series are compact (nt_gf << nt_sim) and almost always fit
                # in cache, so no size-guard is needed here.
                try:
                    comp_idx = self.adapter._gf_component_index(self.state.component)
                    self.adapter.warm_gf_series(self._display_gf_subfault, comp_idx)
                except Exception:
                    pass
            else:
                # ── Regular demand: only pre-warm when the full series fits.
                # When it doesn't fit, _try_direct_component_snapshot handles
                # each frame via a fast single-column HDF5 read.  Calling
                # scalar_series() on an oversized dataset just runs the slow
                # O(T) loop and then discards the result.
                demand = self.state.demand
                if self.adapter.cache_time_series:
                    if self.state.component == "resultant":
                        # Warm E, N, Z individually.  scalar_snapshot derives
                        # resultant from the cached trio on every frame (free
                        # NumPy sqrt, no I/O).  scalar_series("resultant") would
                        # also work after Fix A in adapter, but this is more
                        # direct and avoids an unnecessary cache entry.
                        try:
                            self.adapter.prewarm_component_triplet(demand)
                        except Exception:
                            pass
                    else:
                        try:
                            self.adapter.scalar_series(demand, self.state.component)
                        except Exception:
                            pass
                    # Pre-warm displacement series when warp is active.
                    if self.state.disp_warp_enabled:
                        try:
                            self.adapter.prewarm_component_triplet("disp")
                        except Exception:
                            pass
                if not any((demand, _c) in self.adapter._series_cache for _c in ("e", "n", "z")):
                    # Series too large to cache — keep an HDF5 handle open for
                    # the duration of playback so _try_direct_component_snapshot
                    # avoids the ~1-5 ms file-open cost on every animation frame.
                    self.adapter.open_playback_handle()

        elif not is_playing and self.state.is_playing:
            # Stopping: release the persistent handle (no-op if never opened).
            self.adapter.close_playback_handle()

        if had_static_color_override and is_playing:
            self._notify_window("static_color")
        self.state.set_playing(is_playing)
        self._notify_window("playback")
        return self.state.is_playing

    def set_playback_speed(self, playback_speed: float):
        self.state.set_playback_speed(playback_speed)
        self._notify_window("playback")
        return self.state.playback_speed

    def toggle_playing(self):
        return self.set_playing(not self.state.is_playing)

    def step_time(self, delta: int = 1):
        return self.set_time_index(self.state.time_index + int(delta))

    def jump_time(self, delta: int = 10):
        return self.step_time(delta)

    def current_scalars(self):
        if self._static_color_by == "elevation_z":
            return self.adapter.elevation_snapshot()
        gf_subfault = self._display_gf_subfault if self.state.demand == GF_DEMAND else 0
        return self.adapter.scalar_snapshot(
            self.state.time_index,
            self.state.demand,
            self.state.component,
            subfault_id=gf_subfault,
        )

    def current_visible_points(self):
        return self.adapter.visible_points(
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )

    def current_warped_points(self) -> "np.ndarray":
        """Return visible points displaced by the current displacement field.

        When ``disp_warp_enabled`` is False returns the base visible points so
        callers need not branch on warp state.

        This method is called on **every animation frame** when warp is active,
        so it deliberately avoids ``current_visible_points()`` which runs an
        O(N) Python list-comprehension to update ``_visible_node_ids``.  That
        update is needed for node-picking but not for rendering — we call
        ``current_visible_points()`` only in ``_rebuild_point_cloud()`` (scene
        rebuilds), not on every frame.
        """
        import numpy as _np

        # Compute the boolean mask without triggering the _visible_node_ids
        # list-comprehension that lives in visible_points().
        mask = self.adapter.visibility_mask(
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )
        base = self.adapter._display_points[mask]

        if not self.state.disp_warp_enabled:
            return base

        scale = self.state.warp_scale
        if scale is None:
            scale = self.adapter.suggested_warp_scale()
        scale = float(scale)

        t = self.state.time_index
        disp_all = self.adapter.displacement_snapshot(t)   # (N_display, 3)
        disp_visible = disp_all[mask]                      # (N_visible, 3) [E, N, Z]

        axes = self.state.warp_axes                        # (x_enable, y_enable, z_enable)
        axis_mask = _np.array([float(axes[0]), float(axes[1]), float(axes[2])])
        try:
            return base + scale * disp_visible * axis_mask[_np.newaxis, :]
        except Exception:
            return base

    def current_visible_scalars(self):
        return self.adapter.visible_scalars(
            self.current_scalars(),
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )

    def default_color_limits(self) -> tuple[float, float]:
        if self._static_color_by == "elevation_z":
            return self.adapter.elevation_limits()
        gf_subfault = self._display_gf_subfault if self.state.demand == GF_DEMAND else 0
        return self.adapter.default_scalar_limits(
            self.state.demand,
            self.state.component,
            subfault_id=gf_subfault,
        )

    def current_color_limits(self, scalars=None) -> tuple[float, float]:
        if self._static_color_by is not None:
            return self.current_static_color_limits(scalars)
        if self.state.clamp_enabled and self.state.user_vmin is not None and self.state.user_vmax is not None:
            vmin = float(self.state.user_vmin)
            vmax = float(self.state.user_vmax)
            if vmax <= vmin:
                return vmin, vmin + 1.0
            return vmin, vmax
        if scalars is None:
            try:
                return self.default_color_limits()
            except Exception:
                scalars = self.current_visible_scalars()
        return scalar_limits(scalars, self.state.component)

    def current_trace(self):
        node_id = self.state.selected_node
        if node_id is None:
            return None
        demand = self.state.demand if self.state.demand in REGULAR_DEMANDS else "accel"
        return self.adapter.trace(node_id, demand)

    def current_accel_trace(self):
        node_id = self.state.selected_node
        if node_id is None:
            return None
        return self.adapter.trace(node_id, "accel")

    def current_spectrum(self):
        node_id = self.state.selected_node
        if node_id is None:
            return None
        return self.adapter.spectrum(node_id)

    def current_arias(self):
        node_id = self.state.selected_node
        if node_id is None:
            return None
        return self.adapter.arias(node_id)

    def current_node_info(self):
        node_id = self.state.selected_node
        if node_id is None:
            return None
        return self.adapter.node_info(node_id)

    def set_station_tags(self, stations: list[dict[str, object]]):
        cleaned: list[dict[str, object]] = []
        for entry in stations:
            name = str(entry.get("name", "")).strip()
            xyz_model_m = entry.get("xyz_model_m")
            xyz_display_m = entry.get("xyz_display_m")
            if not name or (xyz_model_m is None and xyz_display_m is None):
                continue
            if xyz_display_m is not None:
                # TODO: Manual station input in the Node page is defined in
                # viewer/display coordinates and must not be transformed again.
                xyz_display_m = tuple(float(v) for v in xyz_display_m)
            if xyz_model_m is not None:
                xyz_model_m = tuple(float(v) for v in xyz_model_m)
                if xyz_display_m is None:
                    xyz_display_m = tuple(
                        float(v) for v in self.adapter.display_points_from_model_xyz_m([xyz_model_m])[0]
                    )
            cleaned.append(
                {
                    "name": name,
                    "xyz_model_m": xyz_model_m,
                    "xyz_display_m": xyz_display_m,
                }
            )
        self._station_tags = cleaned
        self._notify_window("stations")
        return list(self._station_tags)

    def current_display_transform(self):
        return self.adapter.display_transform

    def apply_display_transform(self, matrix):
        """Apply a global geometry transform to every viewer pane."""
        # TODO: Route all future viewer-space transform debugging through this global session entrypoint.
        self.adapter.set_display_transform(matrix)
        self._refresh_station_display_points()
        self._notify_window("geometry_transform")
        return self.adapter.display_transform

    def current_station_tags(self) -> list[dict[str, object]]:
        return list(self._station_tags)

    def set_show_station_tags(self, visible: bool):
        self._show_station_tags = bool(visible)
        self._notify_window("stations_visibility")
        return self._show_station_tags

    def show_station_tags(self) -> bool:
        return bool(self._show_station_tags)

    def _refresh_station_display_points(self) -> None:
        if not self._station_tags:
            return
        for entry in self._station_tags:
            xyz_model_m = entry.get("xyz_model_m")
            if xyz_model_m is None:
                continue
            entry["xyz_display_m"] = tuple(
                float(v)
                for v in self.adapter.display_points_from_model_xyz_m([xyz_model_m])[0]
            )

    def gf_subfault_count(self) -> int:
        """Number of subfaults available in the GF dataset (0 when absent)."""
        return self.adapter.gf_subfault_count()

    def current_gf_tensor(self, subfault_id: int = 0) -> dict | None:
        """Return the GF tensor dict for the selected node, or *None* if unavailable.

        The node_id is passed as-is to the adapter so ``"QA"`` is handled
        correctly (no int() cast that would raise on a string).
        """
        node_id = self.state.selected_node
        if node_id is None or not self.adapter.has_gf:
            return None
        try:
            return self.adapter.gf_tensor(node_id, subfault_id)
        except Exception:
            return None

    def current_time(self) -> float:
        if len(self.adapter.time) == 0:
            return 0.0
        return float(self.adapter.time[self.state.time_index])

    def current_background_color(self) -> str:
        return BACKGROUND_PRESETS[self.state.background]

    def current_colormap(self) -> str:
        if self._static_color_by is not None:
            return self._static_color_map
        return self.state.colormap or colormap_for_component(self.state.component)

    def current_static_color_by(self) -> str | None:
        return self._static_color_by

    def current_static_colormap(self) -> str:
        return self._static_color_map

    def current_static_clamp_enabled(self) -> bool:
        return self._static_clamp_enabled

    def current_static_user_range(self) -> tuple[float | None, float | None]:
        return self._static_user_vmin, self._static_user_vmax

    def current_scalar_bar_title(self) -> str:
        if self._static_color_by == "elevation_z":
            return "Elevation Z [m]"
        return self.state.demand

    def suggested_point_size(self) -> float:
        if self.state.point_size is not None:
            return float(self.state.point_size)
        npts = len(self.current_visible_points())
        if npts < 2_000:
            return 11.0
        if npts < 10_000:
            return 8.0
        return 5.0

    def _notify_window(self, reason: str) -> None:
        if self.window is not None:
            self.window.on_session_updated(reason)

    @staticmethod
    def _validate_static_color_by(color_by: str) -> str:
        color_by = str(color_by).strip().lower()
        if color_by not in VALID_STATIC_COLOR_BY:
            raise KeyError(
                f"Unknown static color source '{color_by}'. "
                f"Use one of {', '.join(VALID_STATIC_COLOR_BY)}."
            )
        return color_by
