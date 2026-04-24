"""Main orchestration object for an interactive viewer session."""

from __future__ import annotations

from .adapter import ViewerDataAdapter
from .colors import BACKGROUND_PRESETS, colormap_for_component, scalar_limits
from .state import ViewerState


class ViewerSession:
    """Coordinate adapter, state, and optional GUI window."""

    def __init__(
        self,
        model_or_adapter,
        *,
        show: bool = False,
        demand: str = "accel",
        component: str = "resultant",
        time_index: int = 0,
        selected_node=None,
        title: str | None = None,
        cache_time_series: bool = True,
        max_cache_bytes: int = 256 * 1024 * 1024,
        max_cache_entries: int = 6,
    ) -> None:
        if isinstance(model_or_adapter, ViewerDataAdapter):
            self.adapter = model_or_adapter
        else:
            self.adapter = ViewerDataAdapter(
                model_or_adapter,
                cache_time_series=cache_time_series,
                max_cache_bytes=max_cache_bytes,
                max_cache_entries=max_cache_entries,
            )

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

    def set_time_index(self, time_index: int):
        self.state.set_time_index(time_index, max(len(self.adapter.time) - 1, 0))
        self._notify_window("time")
        return self.state.time_index

    def set_demand(self, demand: str):
        self.state.set_demand(demand)
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
        self._notify_window("selection")
        return self.state.selected_node

    def select_nearest_coordinate_m(self, x_m: float, y_m: float, z_m: float):
        node_id, distance_m = self.adapter.nearest_node_from_model_xyz_m([x_m, y_m, z_m])
        self.state.set_selected_node(node_id)
        self._notify_window("selection")
        return node_id, distance_m

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
        self._notify_window("appearance")
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
            # Pre-warm all three displacement component series so every frame
            # after enabling is a pure memory operation.
            try:
                for comp in ("e", "n", "z"):
                    self.adapter.scalar_series("disp", comp)
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
                for comp in ("e", "n", "z"):
                    self.adapter.scalar_series("disp", comp)
            except Exception:
                pass

        self._notify_window("warp")
        return self.state.disp_warp_enabled, self.state.warp_axes, self.state.warp_scale

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
                for comp in ("e", "n", "z"):
                    self.adapter.scalar_series("disp", comp)
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
        if is_playing and not self.state.is_playing:
            # Pre-warm the series cache so every playback frame is a pure
            # memory slice with zero HDF5 I/O.
            try:
                if self.state.component == "resultant":
                    # Cache E, N, Z individually — one contiguous column read
                    # each vs. strided reads.  The resultant fast-path in
                    # scalar_snapshot derives the magnitude from these three
                    # series with no additional I/O.
                    for comp in ("e", "n", "z"):
                        self.adapter.scalar_series(self.state.demand, comp)
                else:
                    self.adapter.scalar_series(self.state.demand, self.state.component)
            except Exception:
                pass  # Large files may exceed cache budget — degrade gracefully
            # Also pre-warm displacement series when warp is active.
            if self.state.disp_warp_enabled:
                try:
                    for comp in ("e", "n", "z"):
                        self.adapter.scalar_series("disp", comp)
                except Exception:
                    pass
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
        return self.adapter.scalar_snapshot(
            self.state.time_index,
            self.state.demand,
            self.state.component,
        )

    def current_visible_points(self):
        return self.adapter.visible_points(
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )

    def current_warped_points(self) -> "np.ndarray":
        """Return visible points displaced by the current displacement field.

        When ``disp_warp_enabled`` is False returns the same array as
        ``current_visible_points`` so callers need not branch on warp state.
        """
        import numpy as _np
        base = self.current_visible_points()
        if not self.state.disp_warp_enabled:
            return base

        scale = self.state.warp_scale
        if scale is None:
            scale = self.adapter.suggested_warp_scale()
        scale = float(scale)

        t = self.state.time_index
        disp_all = self.adapter.displacement_snapshot(t)   # (N_display, 3)

        # Restrict to the visible subset that current_visible_points returned.
        mask = self.adapter.visibility_mask(
            show_internal=self.state.show_internal,
            show_external=self.state.show_external,
            show_qa=self.state.show_qa,
        )
        disp_visible = disp_all[mask]   # (N_visible, 3)  [E, N, Z]

        axes = self.state.warp_axes     # (x_enable, y_enable, z_enable)
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
        return self.adapter.default_scalar_limits(self.state.demand, self.state.component)

    def current_color_limits(self, scalars=None) -> tuple[float, float]:
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
        return self.adapter.trace(node_id, self.state.demand)

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

    def current_time(self) -> float:
        if len(self.adapter.time) == 0:
            return 0.0
        return float(self.adapter.time[self.state.time_index])

    def current_background_color(self) -> str:
        return BACKGROUND_PRESETS[self.state.background]

    def current_colormap(self) -> str:
        return self.state.colormap or colormap_for_component(self.state.component)

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
