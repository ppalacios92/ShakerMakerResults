"""Main orchestration object for an interactive viewer session."""

from __future__ import annotations

from .adapter import GF_DEMAND, REGULAR_DEMANDS, ViewerDataAdapter
from .colors import (
    BACKGROUND_PRESETS,
    colormap_for_component,
    effective_colormap_name,
    scalar_limits,
    scalars_to_rgb,
)
from .state import ViewerState

VALID_STATIC_COLOR_BY = ("elevation_z", "newmark_sa", "arias")


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
        self._newmark_static_T_target: float = 0.0
        self._newmark_static_component: str = "resultant"
        self._newmark_static_data_type: str = "accel"
        self._newmark_static_spectral_type: str = "PSa"
        self._newmark_static_factor: float = 1.0 / 9.81
        self._newmark_static_n_jobs: int = -2
        # ── Wave-blend: modulate elevation Z coloring with wave amplitude ─────
        self._wave_blend_enabled: bool = False
        self._wave_blend_strength: float = 0.5   # 0 = pure elevation, 1 = full shift
        self._wave_blend_gamma: float = 1.5
        self._blend_base_rgb_key = None
        self._blend_base_rgb = None
        self._wave_global_max_key = None
        self._wave_global_max = None
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
        self._warped_points_cache_key = None
        self._warped_points_cache = None
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
        self._invalidate_frame_cache()
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

    def add_node_to_multi_selection(self, node_id):
        """Add one node to the comparison set without clearing existing nodes."""
        selected = set(self.state.multi_selection)
        active = self.state.selected_node
        if active is not None:
            selected.add(active)
        selected.add(node_id)
        self.state.multi_selection = frozenset(selected)
        self.state.set_selected_node(node_id)
        self.state.set_selection_visibility("all")
        self._notify_window("selection")
        self._notify_window("multi_selection")
        return self.state.multi_selection

    def toggle_node_in_multi_selection(self, node_id):
        """Ctrl-click helper: add/remove one node from the comparison set."""
        selected = set(self.state.multi_selection)
        active = self.state.selected_node
        if active is not None:
            selected.add(active)

        if node_id in selected and node_id != active:
            selected.remove(node_id)
        else:
            selected.add(node_id)
            self.state.set_selected_node(node_id)

        if not selected:
            self.state.set_selected_node(None)
        elif self.state.selected_node not in selected:
            self.state.set_selected_node(node_id)

        self.state.multi_selection = frozenset(selected)
        self.state.set_selection_visibility("all")
        self._notify_window("selection")
        self._notify_window("multi_selection")
        return self.state.multi_selection

    def selected_response_node_ids(self) -> tuple:
        selected = set(self.state.multi_selection)
        active = self.state.selected_node
        if active is not None:
            selected.add(active)
        if not selected:
            return tuple()
        ordered = []
        if active in selected:
            ordered.append(active)
        ordered.extend(sorted((node for node in selected if node != active), key=lambda x: str(x)))
        return tuple(ordered)

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

    def apply_transfer_function_preferences(
        self,
        *,
        colormap: str,
        inverted: bool,
        discrete: bool,
        bins: int,
        nan_color: str,
        below_color: str,
        above_color: str,
        use_below: bool,
        use_above: bool,
        symmetric_range: bool,
        percentile_clip: float,
        custom_colormap_object=None,
    ):
        """Apply preferences from the Advanced Color Editor.

        ``custom_colormap_object`` is an optional
        :class:`matplotlib.colors.Colormap` instance built from the
        per-stop triangle colours the user picked in the preview.  When
        provided it overrides the named ``colormap``; passing ``None``
        clears any previous custom palette so the named preset is used.
        """
        self.state.set_colormap(colormap)
        # Store / clear the custom Colormap instance.  We assign
        # directly (no setter on state.py) — ``current_colormap`` reads
        # it and pyvista accepts the instance.
        self.state.colormap_object = custom_colormap_object
        self.state.set_transfer_function_preferences(
            inverted=inverted,
            discrete=discrete,
            bins=bins,
            nan_color=nan_color,
            below_color=below_color,
            above_color=above_color,
            use_below=use_below,
            use_above=use_above,
            symmetric_range=symmetric_range,
            percentile_clip=percentile_clip,
        )
        self._notify_window("appearance")
        return self.current_colormap()

    def apply_legend_preferences(
        self,
        *,
        visible: bool,
        title: str,
        orientation: str,
        position: str,
        label_count: int,
        label_font_size: int,
        title_font_size: int,
        show_outline: bool,
        show_background: bool,
    ):
        self.state.set_show_scalar_bar(visible)
        self.state.set_legend_preferences(
            title=title,
            orientation=orientation,
            position=position,
            label_count=label_count,
            label_font_size=label_font_size,
            title_font_size=title_font_size,
            show_outline=show_outline,
            show_background=show_background,
        )
        self._notify_window("appearance")
        return self.state.show_scalar_bar

    def set_axes_grid_visible(self, visible: bool):
        self.state.set_axes_grid_visible(visible)
        self._notify_window("appearance")
        return self.state.show_axes_grid

    def set_selection_labels_enabled(self, enabled: bool):
        self.state.set_selection_labels_enabled(enabled)
        self._notify_window("multi_selection")
        return self.state.selection_labels_enabled

    def set_ghost_warp_reference(self, enabled: bool):
        self.state.set_ghost_warp_reference(enabled)
        self._notify_window("warp")
        return self.state.ghost_warp_reference

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
            self._prepare_component_triplet("disp")
        self._invalidate_frame_cache()
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
        self._invalidate_frame_cache()
        self._notify_window("warp")
        return self.state.warp_axes

    def set_warp_scale(self, scale: float | None):
        """Set the displacement exaggeration factor (None = auto)."""
        self.state.set_warp_scale(scale)
        self._invalidate_frame_cache()
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

        # "panel_apply" triggers a full scalar-actor rebuild that picks up every
        # state change above in one render pass.  Data pre-warming with a
        # progress dialog is handled by ViewerMainWindow.on_session_updated.
        self._notify_window("panel_apply")
        return self.state.demand, self.state.component

    def apply_static_color_settings(
        self,
        *,
        color_by: str | None,
        colormap: str,
        vmin: float | None,
        vmax: float | None,
        clamp_enabled: bool,
        wave_blend_enabled: bool = False,
        wave_blend_strength: float = 0.5,
        newmark_T_target: float | None = None,
        newmark_component: str | None = None,
        newmark_data_type: str | None = None,
        newmark_spectral_type: str | None = None,
        newmark_factor: float | None = None,
        newmark_n_jobs: int | None = None,
    ):
        self._static_color_by = self._validate_static_color_by(color_by)
        self._static_color_map = str(colormap).strip() or self._static_color_map
        self._static_user_vmin = None if vmin is None else float(vmin)
        self._static_user_vmax = None if vmax is None else float(vmax)
        self._static_clamp_enabled = bool(clamp_enabled)
        self._wave_blend_enabled = bool(wave_blend_enabled)
        self._wave_blend_strength = max(0.0, min(1.0, float(wave_blend_strength)))
        self._blend_base_rgb_key = None
        self._blend_base_rgb = None
        if newmark_T_target is not None:
            self._newmark_static_T_target = float(newmark_T_target)
        if newmark_component is not None:
            self._newmark_static_component = str(newmark_component).lower()
        if newmark_data_type is not None:
            self._newmark_static_data_type = str(newmark_data_type).lower()
        if newmark_spectral_type is not None:
            self._newmark_static_spectral_type = str(newmark_spectral_type)
        if newmark_factor is not None:
            self._newmark_static_factor = float(newmark_factor)
        if newmark_n_jobs is not None:
            self._newmark_static_n_jobs = int(newmark_n_jobs)
        self._notify_window("static_color")
        return self._static_color_by, self._static_color_map

    def apply_static_color_by(self, color_by: str | None) -> str | None:
        self._static_color_by = self._validate_static_color_by(color_by)
        self._static_color_map = self.state.colormap
        self._blend_base_rgb_key = None
        self._blend_base_rgb = None
        self._notify_window("static_color")
        return self._static_color_by

    def current_color_by(self) -> str:
        return self._static_color_by if self._static_color_by is not None else "field"

    # ── Wave-blend getters ────────────────────────────────────────────────────

    def current_wave_blend_enabled(self) -> bool:
        return self._wave_blend_enabled

    def current_wave_blend_strength(self) -> float:
        return self._wave_blend_strength

    def current_wave_blend_active(self) -> bool:
        return bool(
            self._static_color_by == "elevation_z"
            and self._wave_blend_enabled
            and self.state.is_playing
        )

    def current_display_gf_subfault(self) -> int:
        return int(self._display_gf_subfault)

    def current_static_auto_limits(self) -> tuple[float, float]:
        if self._static_color_by == "elevation_z":
            return self.adapter.elevation_limits()
        if self._static_color_by == "newmark_sa":
            return scalar_limits(self._newmark_surface_scalars(), "resultant")
        if self._static_color_by == "arias":
            return scalar_limits(self._arias_surface_scalars(), "resultant")
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
        if self._static_color_by == "newmark_sa":
            return scalar_limits(scalars, "resultant")
        if self._static_color_by == "arias":
            return scalar_limits(scalars, "resultant")
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

        # Keep an open HDF5 handle while warp is active so every animation
        # frame's displacement_snapshot (3 × scalar_snapshot) uses the
        # persistent handle instead of reopening the file per call.
        if warp_enabled and not was_warp_enabled:
            # Warp just enabled — open handle if disp series not in cache.
            if not all(("disp", _c) in self.adapter._series_cache for _c in ("e", "n", "z")):
                self.adapter.open_playback_handle()
        elif not warp_enabled and not self.state.is_playing:
            # Warp disabled and not playing — nobody needs the handle.
            self.adapter.close_playback_handle()

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
        persistent_static_color = self._static_color_by in ("elevation_z", "newmark_sa", "arias")
        # Elevation is the playback base: keep it alive and let current_scalars()
        # blend the active wave onto it while the animation runs.
        if is_playing and not self._wave_blend_enabled and not persistent_static_color:
            self._static_color_by = None
        if is_playing and not self.state.is_playing:
            if self.state.demand == GF_DEMAND:
                # GF series are compact — warm instantly with no dialog needed.
                try:
                    comp_idx = self.adapter._gf_component_index(self.state.component)
                    self.adapter.warm_gf_series(self._display_gf_subfault, comp_idx)
                except Exception:
                    pass
            else:
                # Regular demands: ViewerMainWindow.on_session_updated("playback")
                # runs _prewarm_demands (with BusyDialog + progress) BEFORE this
                # set_playing call starts the play timer, so the triplet should
                # already be in cache by now.
                # If the series is still not cached (budget too small to hold the
                # triplet), open a persistent HDF5 handle so per-frame reads skip
                # the ~1-5 ms file-open overhead on every animation tick.
                demand = self.state.demand
                if self.window is None:
                    try:
                        self.adapter.scalar_series(demand, self.state.component)
                    except Exception:
                        pass
                    if self.state.disp_warp_enabled:
                        self._prepare_component_triplet("disp")
                if not all((demand, _c) in self.adapter._series_cache for _c in ("e", "n", "z")):
                    self.adapter.open_playback_handle()

        elif not is_playing and self.state.is_playing:
            # Stopping: release the persistent handle only when warp is also
            # off.  When warp is active the handle must stay open so each
            # animation frame's displacement_snapshot uses the fast path
            # instead of reopening the HDF5 file per call.
            if not self.state.disp_warp_enabled:
                self.adapter.close_playback_handle()

        # Only force a scene rebuild for static-color → wave transition when
        # blend mode is OFF (blend mode keeps elevation alive during playback).
        if had_static_color_override and is_playing and not self._wave_blend_enabled and not persistent_static_color:
            self._notify_window("static_color")
        self.state.set_playing(is_playing)
        self._invalidate_frame_cache()
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

    def current_scalars(self, *, state=None):
        st = state if state is not None else self.state
        if self._static_color_by == "elevation_z":
            if self.current_wave_blend_active():
                return self._elevation_wave_rgb_blend()
            return self.adapter.elevation_snapshot()
        if self._static_color_by == "newmark_sa":
            return self._newmark_surface_scalars()
        if self._static_color_by == "arias":
            return self._arias_surface_scalars()
        gf_subfault = self._display_gf_subfault if st.demand == GF_DEMAND else 0
        # time_index is intentionally read from the GLOBAL state — the
        # animation clock is shared across panes (per the per-pane
        # whitelist in pane_state).
        return self.adapter.scalar_snapshot(
            self.state.time_index,
            st.demand,
            st.component,
            subfault_id=gf_subfault,
        )

    def current_wave_scalars(self):
        gf_subfault = self._display_gf_subfault if self.state.demand == GF_DEMAND else 0
        return self.adapter.scalar_snapshot(
            self.state.time_index,
            self.state.demand,
            self.state.component,
            subfault_id=gf_subfault,
        )

    def _elevation_wave_rgb_blend(self):
        """Return per-point RGB blending elevation colours with the current wave.

        ``base_rgb`` carries the static elevation colormap; the wave frame is
        normalised to ``[0, 1]`` by the dataset's global amplitude, gamma is
        applied, and the resulting alpha mixes wave colours over the base.
        With ``strength = 0`` the output equals the pure elevation map; with
        ``strength = 1`` the wave dominates at peak frames.
        """
        import numpy as np

        base_rgb = self._elevation_base_rgb()
        wave = self.current_wave_scalars()
        w_min, w_max = self._current_wave_color_limits(wave)
        wave_rgb = scalars_to_rgb(wave, self._current_wave_colormap(), w_min, w_max)
        w_abs = self._wave_global_maximum(wave)
        amp = np.clip(np.abs(wave) / w_abs, 0.0, 1.0)
        alpha = (float(self._wave_blend_strength) * (amp ** float(self._wave_blend_gamma)))[:, None]
        blended = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * wave_rgb.astype(np.float32)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def _elevation_base_rgb(self):
        elev = self.adapter.elevation_snapshot()
        vmin, vmax = self.current_static_color_limits(elev)
        # ``colormap_inverted`` flips the LUT direction at render time, so
        # toggling it must invalidate the cached RGB — otherwise the user sees
        # stale colours until some unrelated state change drops the cache.
        key = (
            "elevation_z",
            self._static_color_map,
            bool(self.state.colormap_inverted),
            float(vmin),
            float(vmax),
            len(elev),
        )
        if self._blend_base_rgb_key != key or self._blend_base_rgb is None:
            self._blend_base_rgb = scalars_to_rgb(
                elev,
                effective_colormap_name(
                    self._static_color_map,
                    inverted=self.state.colormap_inverted,
                ),
                vmin,
                vmax,
            )
            self._blend_base_rgb_key = key
        return self._blend_base_rgb

    def _wave_global_maximum(self, wave=None) -> float:
        key = (self.state.demand, self.state.component)
        if self._wave_global_max_key == key and self._wave_global_max is not None:
            return self._wave_global_max
        try:
            w_min, w_max = self.adapter.default_scalar_limits(
                self.state.demand,
                self.state.component,
                subfault_id=self._display_gf_subfault if self.state.demand == GF_DEMAND else 0,
            )
            w_abs = max(abs(float(w_min)), abs(float(w_max)), 1.0e-12)
        except Exception:
            import numpy as np

            if wave is None:
                wave = self.current_wave_scalars()
            w_abs = max(float(np.max(np.abs(wave))), 1.0e-12)
        self._wave_global_max_key = key
        self._wave_global_max = w_abs
        return w_abs

    def current_visible_points(self, *, state=None):
        # Visibility filters are global (whitelist in pane_state.py),
        # so we read from self.state directly; ``state`` is accepted
        # for signature symmetry but ignored here.
        return self.adapter.visible_points(
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )

    def current_warped_points(self, *, state=None) -> "np.ndarray":
        """Return visible points displaced by the current displacement field.

        Warp parameters (enabled / axes / scale) are *per-pane* so the
        caller can pass its pane overlay via ``state=``.  When ``state``
        is None we fall back to the global session state — preserving
        the legacy single-pane behaviour for any caller that has not
        been updated.
        """
        import numpy as _np
        st = state if state is not None else self.state

        # Compute the boolean mask without triggering the
        # _visible_node_ids list-comprehension that lives in visible_points().
        # Visibility filters are global so use self.state for them.
        mask = self.adapter.visibility_mask(
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )
        base = self.adapter._display_points[mask]

        if not st.disp_warp_enabled:
            # Per-pane warp state — cache key must include the pane
            # identity through the warp parameters, otherwise two panes
            # with different warp settings would thrash the single
            # cache slot.  Visibility flags are global so they stay in
            # the key for legacy callers.
            key = ("base", self.state.show_internal, self.state.show_external, self.state.show_qa)
            if self._warped_points_cache_key == key and self._warped_points_cache is not None:
                return self._warped_points_cache
            self._warped_points_cache_key = key
            self._warped_points_cache = base
            return base

        scale = st.warp_scale
        if scale is None:
            scale = self.adapter.suggested_warp_scale()
        scale = float(scale)
        key = (
            "warp",
            self.state.time_index,
            scale,
            tuple(st.warp_axes),
            self.state.show_internal,
            self.state.show_external,
            self.state.show_qa,
        )
        if self._warped_points_cache_key == key and self._warped_points_cache is not None:
            return self._warped_points_cache

        t = self.state.time_index
        disp_all = self.adapter.displacement_snapshot(t)   # (N_display, 3)
        disp_all = self.adapter.display_points_from_model_xyz_m(disp_all)
        disp_all[:, 2] *= -1.0
        disp_visible = disp_all[mask]                      # (N_visible, 3) [E, N, Z]

        axes = st.warp_axes                                 # (x_enable, y_enable, z_enable)
        axis_mask = _np.array([float(axes[0]), float(axes[1]), float(axes[2])])
        try:
            warped = base + scale * disp_visible * axis_mask[_np.newaxis, :]
        except Exception:
            warped = base
        self._warped_points_cache_key = key
        self._warped_points_cache = warped
        return warped

    def current_visible_scalars(self, *, state=None):
        # Visibility filters stay global (whitelist in pane_state.py),
        # but the underlying scalar values follow the pane's demand /
        # component / time index combination.
        return self.adapter.visible_scalars(
            self.current_scalars(state=state),
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )

    def default_color_limits(self, *, state=None) -> tuple[float, float]:
        st = state if state is not None else self.state
        if self._static_color_by == "elevation_z":
            return self.adapter.elevation_limits()
        if self._static_color_by == "newmark_sa":
            return scalar_limits(self._newmark_surface_scalars(), "resultant")
        if self._static_color_by == "arias":
            return scalar_limits(self._arias_surface_scalars(), "resultant")
        gf_subfault = self._display_gf_subfault if st.demand == GF_DEMAND else 0
        return self.adapter.default_scalar_limits(
            st.demand,
            st.component,
            subfault_id=gf_subfault,
        )

    def _current_wave_color_limits(self, scalars=None) -> tuple[float, float]:
        if self.state.clamp_enabled and self.state.user_vmin is not None and self.state.user_vmax is not None:
            vmin = float(self.state.user_vmin)
            vmax = float(self.state.user_vmax)
            if vmax <= vmin:
                return vmin, vmin + 1.0
            return vmin, vmax
        try:
            gf_subfault = self._display_gf_subfault if self.state.demand == GF_DEMAND else 0
            return self.adapter.default_scalar_limits(
                self.state.demand,
                self.state.component,
                subfault_id=gf_subfault,
            )
        except Exception:
            if scalars is None:
                scalars = self.current_wave_scalars()
            return scalar_limits(scalars, self.state.component)

    def _current_wave_colormap(self) -> str:
        return effective_colormap_name(
            self.state.colormap or colormap_for_component(self.state.component),
            inverted=self.state.colormap_inverted,
        )

    def current_color_limits(self, scalars=None, *, state=None) -> tuple[float, float]:
        """Return the active color range.

        ``state`` is an optional :class:`PaneDisplayState` (or anything
        with the same attribute names) used to resolve display-only
        attributes — when ``None`` we fall back to the global
        :data:`self.state`.  Callers from a per-pane scene pass their
        pane's overlay so panes with their own ``user_vmin`` / colormap
        inversion / percentile clip still get sane limits.
        """
        st = state if state is not None else self.state
        if self.current_wave_blend_active():
            return self._current_wave_color_limits(None)
        if self._static_color_by is not None:
            return self.current_static_color_limits(scalars)
        if st.clamp_enabled and st.user_vmin is not None and st.user_vmax is not None:
            vmin = float(st.user_vmin)
            vmax = float(st.user_vmax)
            if vmax <= vmin:
                return vmin, vmin + 1.0
            return vmin, vmax
        if scalars is None:
            try:
                lo, hi = self.default_color_limits(state=st)
                if st.symmetric_color_range:
                    vmax = max(abs(float(lo)), abs(float(hi)))
                    return -vmax, vmax
                return lo, hi
            except Exception:
                scalars = self.current_visible_scalars(state=st)
        try:
            import numpy as np

            arr = np.asarray(scalars, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size and st.percentile_clip > 0.0:
                lo = float(np.percentile(arr, st.percentile_clip))
                hi = float(np.percentile(arr, 100.0 - st.percentile_clip))
            else:
                lo, hi = scalar_limits(arr, st.component)
        except Exception:
            lo, hi = scalar_limits(scalars, st.component)
        if st.symmetric_color_range:
            vmax = max(abs(float(lo)), abs(float(hi)))
            return -vmax, vmax
        return lo, hi

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

    def current_background_color(self, *, state=None) -> str:
        st = state if state is not None else self.state
        return BACKGROUND_PRESETS[st.background]

    def current_colormap(self, *, state=None):
        """Return the active colormap.

        Usually a matplotlib name (``str``).  When the user has built a
        custom palette through the Advanced Color Editor (per-stop
        triangle colours), this returns the underlying
        :class:`matplotlib.colors.Colormap` instance instead — pyvista
        accepts either form, and we propagate the object directly so we
        do not need to register it in matplotlib's global cmap registry.
        """
        st = state if state is not None else self.state
        if self.current_wave_blend_active():
            return self._current_wave_colormap()
        if self._static_color_by is not None:
            return effective_colormap_name(
                self._static_color_map,
                inverted=st.colormap_inverted,
            )
        custom = getattr(st, "colormap_object", None)
        if custom is not None:
            return custom
        return effective_colormap_name(
            st.colormap or colormap_for_component(st.component),
            inverted=st.colormap_inverted,
        )

    def current_static_color_by(self) -> str | None:
        return self._static_color_by

    def current_static_colormap(self) -> str:
        return self._static_color_map

    def current_static_clamp_enabled(self) -> bool:
        return self._static_clamp_enabled

    def current_static_user_range(self) -> tuple[float | None, float | None]:
        return self._static_user_vmin, self._static_user_vmax

    def current_newmark_static_settings(self) -> dict[str, object]:
        return {
            "T_target": self._newmark_static_T_target,
            "component": self._newmark_static_component,
            "data_type": self._newmark_static_data_type,
            "spectral_type": self._newmark_static_spectral_type,
            "factor": self._newmark_static_factor,
            "n_jobs": self._newmark_static_n_jobs,
        }

    def current_scalar_bar_title(self, *, state=None) -> str:
        st = state if state is not None else self.state
        if st.legend_title_override:
            return st.legend_title_override
        if self.current_wave_blend_active():
            return f"{st.demand}/{st.component}"
        if self._static_color_by == "elevation_z":
            return "Elevation Z [m]"
        if self._static_color_by == "newmark_sa":
            return (
                f"{self._newmark_static_spectral_type}"
                f"(T={self._newmark_static_T_target:g}s) "
                f"{self._newmark_static_data_type}/{self._newmark_static_component}"
            )
        if self._static_color_by == "arias":
            return f"Arias {self._newmark_static_data_type}/{self._newmark_static_component}"
        return f"{st.demand}/{st.component}"

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
    def _validate_static_color_by(color_by: str | None) -> str | None:
        if color_by is None:
            return None
        color_by = str(color_by).strip().lower()
        if color_by in ("", "default", "field"):
            return None
        if color_by not in VALID_STATIC_COLOR_BY:
            raise KeyError(
                f"Unknown static color source '{color_by}'. "
                f"Use default or one of {', '.join(VALID_STATIC_COLOR_BY)}."
            )
        return color_by

    def _invalidate_frame_cache(self) -> None:
        self._warped_points_cache_key = None
        self._warped_points_cache = None

    def _prepare_component_triplet(self, demand: str) -> None:
        try:
            keys = tuple((demand, comp) for comp in ("e", "n", "z"))
            if all(key in self.adapter._series_cache for key in keys):
                return
            one = self.adapter._estimated_series_bytes_for_demand(demand)
            if one > 0 and one * 3 <= self.adapter.max_cache_bytes:
                self.adapter.prewarm_component_triplet(demand, no_evict=True)
            if not all(key in self.adapter._series_cache for key in keys):
                self.adapter.open_playback_handle()
        except Exception:
            try:
                self.adapter.open_playback_handle()
            except Exception:
                pass

    def _newmark_surface_scalars(self):
        return self.adapter.newmark_surface_snapshot(
            T_target=self._newmark_static_T_target,
            component=self._newmark_static_component,
            data_type=self._newmark_static_data_type,
            spectral_type=self._newmark_static_spectral_type,
            factor=self._newmark_static_factor,
            n_jobs=self._newmark_static_n_jobs,
        )

    def _arias_surface_scalars(self):
        return self.adapter.arias_surface_snapshot(
            component=self._newmark_static_component,
            data_type=self._newmark_static_data_type,
            factor=self._newmark_static_factor,
            n_jobs=self._newmark_static_n_jobs,
        )
