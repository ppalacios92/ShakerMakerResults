"""Viewer state container."""

from __future__ import annotations

from dataclasses import dataclass

from .adapter import VALID_COMPONENTS, VALID_DEMANDS
from .colors import BACKGROUND_PRESETS, colormap_for_component


@dataclass
class ViewerState:
    """Mutable state for one viewer session."""

    time_index: int = 0
    demand: str = "accel"
    component: str = "resultant"
    selected_node: int | str | None = None
    background: str = "White"
    colormap: str | None = None
    point_size: float | None = None
    show_scalar_bar: bool = True
    is_playing: bool = False
    playback_speed: float = 1.0

    # Color-range control. When clamp_enabled is False the viewer uses the
    # automatic limits from ShakerMakerData._vmax. When True, user_vmin/user_vmax
    # override the automatic range.
    clamp_enabled: bool = False
    user_vmin: float | None = None
    user_vmax: float | None = None

    # Geometry visibility controls.
    show_internal: bool = True
    show_external: bool = True
    show_qa: bool = True

    # 3-D warp / real-motion displacement visualisation.
    # When disp_warp_enabled is True the scene moves each point by its
    # displacement field × warp_scale at every animation frame.
    # warp_axes controls which spatial axes are warped (E/X, N/Y, Z).
    # warp_scale=None means the adapter's auto-suggestion is used.
    disp_warp_enabled: bool = False
    warp_axes: tuple = (True, True, True)   # (E/X, N/Y, Z)
    warp_scale: float | None = None

    def __post_init__(self) -> None:
        self.demand = self._validate_demand(self.demand)
        self.component = self._validate_component(self.component)
        self.time_index = max(0, int(self.time_index))
        self.background = self._validate_background(self.background)
        if self.colormap is None:
            self.colormap = colormap_for_component(self.component)
        self.point_size = None if self.point_size is None else float(self.point_size)
        self.show_scalar_bar = bool(self.show_scalar_bar)
        self.is_playing = bool(self.is_playing)
        self.playback_speed = self._validate_playback_speed(self.playback_speed)
        self.clamp_enabled = bool(self.clamp_enabled)
        self.user_vmin = None if self.user_vmin is None else float(self.user_vmin)
        self.user_vmax = None if self.user_vmax is None else float(self.user_vmax)
        self.show_internal = bool(self.show_internal)
        self.show_external = bool(self.show_external)
        self.show_qa = bool(self.show_qa)
        self.disp_warp_enabled = bool(self.disp_warp_enabled)
        axes = self.warp_axes
        self.warp_axes = (bool(axes[0]), bool(axes[1]), bool(axes[2]))
        self.warp_scale = None if self.warp_scale is None else float(self.warp_scale)

    def set_time_index(self, time_index: int, max_index: int) -> int:
        self.time_index = max(0, min(int(time_index), int(max_index)))
        return self.time_index

    def set_demand(self, demand: str) -> str:
        self.demand = self._validate_demand(demand)
        return self.demand

    def set_component(self, component: str) -> str:
        self.component = self._validate_component(component)
        if self.colormap is None:
            self.colormap = colormap_for_component(self.component)
        return self.component

    def set_selected_node(self, node_id: int | str | None) -> int | str | None:
        self.selected_node = node_id
        return self.selected_node

    def set_background(self, background: str) -> str:
        self.background = self._validate_background(background)
        return self.background

    def set_colormap(self, colormap: str | None) -> str | None:
        self.colormap = None if colormap in (None, "") else str(colormap)
        return self.colormap

    def set_point_size(self, point_size: float | None) -> float | None:
        self.point_size = None if point_size is None else float(point_size)
        return self.point_size

    def set_show_scalar_bar(self, show_scalar_bar: bool) -> bool:
        self.show_scalar_bar = bool(show_scalar_bar)
        return self.show_scalar_bar

    def set_playing(self, is_playing: bool) -> bool:
        self.is_playing = bool(is_playing)
        return self.is_playing

    def set_playback_speed(self, playback_speed: float) -> float:
        self.playback_speed = self._validate_playback_speed(playback_speed)
        return self.playback_speed

    def set_clamp_enabled(self, enabled: bool) -> bool:
        self.clamp_enabled = bool(enabled)
        return self.clamp_enabled

    def set_user_color_range(self, vmin: float | None, vmax: float | None) -> tuple[float | None, float | None]:
        self.user_vmin = None if vmin is None else float(vmin)
        self.user_vmax = None if vmax is None else float(vmax)
        return self.user_vmin, self.user_vmax

    def set_node_visibility(
        self,
        *,
        show_internal: bool | None = None,
        show_external: bool | None = None,
        show_qa: bool | None = None,
    ) -> tuple[bool, bool, bool]:
        if show_internal is not None:
            self.show_internal = bool(show_internal)
        if show_external is not None:
            self.show_external = bool(show_external)
        if show_qa is not None:
            self.show_qa = bool(show_qa)
        return self.show_internal, self.show_external, self.show_qa

    def set_warp_enabled(self, enabled: bool) -> bool:
        self.disp_warp_enabled = bool(enabled)
        return self.disp_warp_enabled

    def set_warp_axes(self, axes: tuple) -> tuple:
        self.warp_axes = (bool(axes[0]), bool(axes[1]), bool(axes[2]))
        return self.warp_axes

    def set_warp_scale(self, scale: float | None) -> float | None:
        self.warp_scale = None if scale is None else max(0.0, float(scale))
        return self.warp_scale

    @staticmethod
    def _validate_demand(demand: str) -> str:
        demand = demand.lower()
        if demand not in VALID_DEMANDS:
            raise KeyError(
                f"Unknown demand '{demand}'. Use one of {', '.join(VALID_DEMANDS)}."
            )
        return demand

    @staticmethod
    def _validate_component(component: str) -> str:
        component = component.lower()
        if component not in VALID_COMPONENTS:
            raise KeyError(
                "Unknown component "
                f"'{component}'. Use one of {', '.join(VALID_COMPONENTS)}."
            )
        return component

    @staticmethod
    def _validate_background(background: str) -> str:
        if background not in BACKGROUND_PRESETS:
            raise KeyError(
                f"Unknown background '{background}'. Use one of "
                f"{', '.join(BACKGROUND_PRESETS)}."
            )
        return background

    @staticmethod
    def _validate_playback_speed(playback_speed: float) -> float:
        speed = float(playback_speed)
        if speed <= 0.0:
            raise ValueError("playback_speed must be greater than 0.")
        return speed
