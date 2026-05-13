"""
state.py
========
Mutable display state for a single ``ViewerSession``.

This file is plain data + validation. No Qt, no PyVista. Every field has
either a self-evident default or a setter that re-validates and returns
the canonical value.

The fields are grouped (in declaration order) by purpose:

* time / playback (``time_index``, ``is_playing``, ``playback_speed``)
* demand / component / selection (``demand``, ``component``, ``selected_node``)
* appearance basics (``background``, ``colormap``, ``point_size``,
  ``show_scalar_bar``)
* advanced colormap (``colormap_object``, ``colormap_inverted``,
  ``colormap_discrete``, ``colormap_bins``, NaN / below / above range)
* color-range overrides (``clamp_enabled``, ``user_vmin/max``,
  ``symmetric_color_range``, ``percentile_clip``)
* legend / scalar bar (``legend_*``)
* axes grid (``show_axes_grid``, ``axes_grid_color``)
* selection labels and warp ghost (``selection_labels_enabled``,
  ``ghost_warp_reference``)
* geometry visibility (``show_internal/external/qa``)
* displacement warp (``disp_warp_enabled``, ``warp_axes``, ``warp_scale``)
* multi-selection (``multi_selection``, ``selection_visibility``,
  ``node_opacity``)
* vector field overlay (``vector_field_*``)

``PaneDisplayState`` (in :mod:`pane_state`) wraps an instance of this
class so individual viewer panes can override a subset of the fields
without copying the whole state.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapter import VALID_COMPONENTS, VALID_DEMANDS
from .colors import BACKGROUND_PRESETS, colormap_for_component


@dataclass
class ViewerState:
    """Mutable per-session display state.

    Every attribute is a "knob" the user can poke from the side panel,
    a toolbar button or a keyboard shortcut. Setters are typed and
    enforce their valid ranges via the ``_validate_*`` static helpers,
    so external callers can hand us untrusted values (e.g. parsed from
    a QSettings blob) without crashing the viewer.
    """

    time_index: int = 0
    demand: str = "accel"
    component: str = "resultant"
    selected_node: int | str | None = None
    background: str = "White"
    colormap: str | None = None
    #: Optional custom :class:`matplotlib.colors.Colormap` that overrides
    #: the named colormap above when set.  Populated by the Advanced
    #: Color Editor when the user paints individual stops via the preview
    #: triangles; cleared whenever the user picks a different named
    #: preset.  Pyvista's ``add_points(cmap=...)`` accepts a Colormap
    #: instance directly, so no registration with matplotlib's global
    #: cmap registry is necessary.
    colormap_object: object | None = None
    point_size: float | None = None
    show_scalar_bar: bool = True
    colormap_inverted: bool = False
    colormap_discrete: bool = False
    colormap_bins: int = 12
    nan_color: str = "#8c8c8c"
    below_range_color: str = "#15304b"
    above_range_color: str = "#f3b23f"
    use_below_range_color: bool = False
    use_above_range_color: bool = False
    symmetric_color_range: bool = False
    percentile_clip: float = 0.0
    legend_title_override: str = ""
    legend_orientation: str = "vertical"
    legend_position: str = "right"
    legend_label_count: int = 5
    legend_label_font_size: int = 11
    legend_title_font_size: int = 12
    legend_show_outline: bool = True
    legend_show_background: bool = False
    show_axes_grid: bool = False
    axes_grid_color: str = "#8c97a8"
    selection_labels_enabled: bool = False
    ghost_warp_reference: bool = False
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

    # ── Multi-node selection (visualization only — zero data impact) ──────────
    # multi_selection: frozenset of node IDs currently highlighted red.
    # selection_visibility: controls which nodes the main actor renders:
    #   "all"  → normal view (all visible nodes shown)
    #   "only" → only nodes in multi_selection are rendered
    #   "hide" → nodes in multi_selection are hidden, rest shown
    # node_opacity: uniform opacity for the main point actor (0–1).
    #   Does NOT affect the single-node yellow sphere or the red multi-selection
    #   overlay — those always render fully opaque.
    multi_selection: frozenset = frozenset()
    selection_visibility: str = "all"
    node_opacity: float = 1.0

    # ── Arrow / vector-field overlay ──────────────────────────────────────────
    vector_field_enabled: bool = False
    vector_field_demand: str = "disp"   # "accel" | "vel" | "disp"
    vector_field_scale: float = 1.0
    vector_field_colormap: str = "viridis"

    def __post_init__(self) -> None:
        """Normalise every field to its canonical value and bounded range.

        Runs once after dataclass instantiation. Each field that has an
        accepted set of values (e.g. ``demand``, ``component``,
        ``legend_position``) is funnelled through a ``_validate_*``
        helper or clamped with ``min``/``max``. Boolean fields are
        coerced via ``bool(...)`` so callers can pass 0/1/None.
        """
        self.demand = self._validate_demand(self.demand)
        self.component = self._validate_component(self.component)
        self.time_index = max(0, int(self.time_index))
        self.background = self._validate_background(self.background)
        if self.colormap is None:
            self.colormap = colormap_for_component(self.component)
        self.point_size = None if self.point_size is None else float(self.point_size)
        self.show_scalar_bar = bool(self.show_scalar_bar)
        self.colormap_inverted = bool(self.colormap_inverted)
        self.colormap_discrete = bool(self.colormap_discrete)
        self.colormap_bins = max(3, min(64, int(self.colormap_bins)))
        self.nan_color = str(self.nan_color or "#8c8c8c")
        self.below_range_color = str(self.below_range_color or "#15304b")
        self.above_range_color = str(self.above_range_color or "#f3b23f")
        self.use_below_range_color = bool(self.use_below_range_color)
        self.use_above_range_color = bool(self.use_above_range_color)
        self.symmetric_color_range = bool(self.symmetric_color_range)
        self.percentile_clip = max(0.0, min(20.0, float(self.percentile_clip)))
        self.legend_title_override = str(self.legend_title_override or "")
        self.legend_orientation = (
            str(self.legend_orientation)
            if str(self.legend_orientation) in ("vertical", "horizontal")
            else "vertical"
        )
        self.legend_position = (
            str(self.legend_position)
            if str(self.legend_position) in ("right", "left", "top", "bottom")
            else "right"
        )
        self.legend_label_count = max(2, min(12, int(self.legend_label_count)))
        self.legend_label_font_size = max(7, min(24, int(self.legend_label_font_size)))
        self.legend_title_font_size = max(8, min(28, int(self.legend_title_font_size)))
        self.legend_show_outline = bool(self.legend_show_outline)
        self.legend_show_background = bool(self.legend_show_background)
        self.show_axes_grid = bool(self.show_axes_grid)
        self.axes_grid_color = str(self.axes_grid_color or "#8c97a8")
        self.selection_labels_enabled = bool(self.selection_labels_enabled)
        self.ghost_warp_reference = bool(self.ghost_warp_reference)
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
        if not isinstance(self.multi_selection, frozenset):
            self.multi_selection = frozenset(self.multi_selection)
        if self.selection_visibility not in ("all", "only", "hide"):
            self.selection_visibility = "all"
        self.node_opacity = max(0.0, min(1.0, float(self.node_opacity)))
        self.vector_field_enabled = bool(self.vector_field_enabled)
        _vfd = str(self.vector_field_demand).lower()
        self.vector_field_demand = _vfd if _vfd in ("accel", "vel", "disp") else "disp"
        self.vector_field_scale = max(0.01, float(self.vector_field_scale))
        self.vector_field_colormap = str(self.vector_field_colormap) or "viridis"

    def set_time_index(self, time_index: int, max_index: int) -> int:
        """Clamp ``time_index`` to ``[0, max_index]`` and return the stored value."""
        self.time_index = max(0, min(int(time_index), int(max_index)))
        return self.time_index

    def set_demand(self, demand: str) -> str:
        """Set the current demand (``'accel'``, ``'vel'``, ``'disp'``, ``'gf'``)."""
        self.demand = self._validate_demand(demand)
        return self.demand

    def set_component(self, component: str) -> str:
        """Set the current component, refreshing the default colormap if unset."""
        self.component = self._validate_component(component)
        if self.colormap is None:
            self.colormap = colormap_for_component(self.component)
        return self.component

    def set_selected_node(self, node_id: int | str | None) -> int | str | None:
        """Record the active single-node selection (id, ``'QA'``, or ``None``)."""
        self.selected_node = node_id
        return self.selected_node

    def set_background(self, background: str) -> str:
        """Set the 3-D background preset (validated against ``BACKGROUND_PRESETS``)."""
        self.background = self._validate_background(background)
        return self.background

    def set_colormap(self, colormap: str | None) -> str | None:
        """Set the named colormap. Empty / ``None`` resets to the per-component default."""
        self.colormap = None if colormap in (None, "") else str(colormap)
        return self.colormap

    def set_point_size(self, point_size: float | None) -> float | None:
        """Set the scatter point size; ``None`` lets the viewer pick from spacing."""
        self.point_size = None if point_size is None else float(point_size)
        return self.point_size

    def set_show_scalar_bar(self, show_scalar_bar: bool) -> bool:
        """Toggle the scalar bar visibility."""
        self.show_scalar_bar = bool(show_scalar_bar)
        return self.show_scalar_bar

    def set_transfer_function_preferences(
        self,
        *,
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
    ) -> None:
        """Bulk-set every "advanced colormap" knob; all kwargs are required."""
        self.colormap_inverted = bool(inverted)
        self.colormap_discrete = bool(discrete)
        self.colormap_bins = max(3, min(64, int(bins)))
        self.nan_color = str(nan_color or self.nan_color)
        self.below_range_color = str(below_color or self.below_range_color)
        self.above_range_color = str(above_color or self.above_range_color)
        self.use_below_range_color = bool(use_below)
        self.use_above_range_color = bool(use_above)
        self.symmetric_color_range = bool(symmetric_range)
        self.percentile_clip = max(0.0, min(20.0, float(percentile_clip)))

    def set_legend_preferences(
        self,
        *,
        title: str,
        orientation: str,
        position: str,
        label_count: int,
        label_font_size: int,
        title_font_size: int,
        show_outline: bool,
        show_background: bool,
    ) -> None:
        """Bulk-set scalar-bar (legend) preferences. All kwargs are required."""
        self.legend_title_override = str(title or "")
        self.legend_orientation = (
            str(orientation) if str(orientation) in ("vertical", "horizontal") else "vertical"
        )
        self.legend_position = (
            str(position) if str(position) in ("right", "left", "top", "bottom") else "right"
        )
        self.legend_label_count = max(2, min(12, int(label_count)))
        self.legend_label_font_size = max(7, min(24, int(label_font_size)))
        self.legend_title_font_size = max(8, min(28, int(title_font_size)))
        self.legend_show_outline = bool(show_outline)
        self.legend_show_background = bool(show_background)

    def set_axes_grid_visible(self, visible: bool) -> bool:
        """Toggle the world-axes / grid overlay."""
        self.show_axes_grid = bool(visible)
        return self.show_axes_grid

    def set_selection_labels_enabled(self, enabled: bool) -> bool:
        """Toggle the per-node text labels on selected points."""
        self.selection_labels_enabled = bool(enabled)
        return self.selection_labels_enabled

    def set_ghost_warp_reference(self, enabled: bool) -> bool:
        """Toggle the faint undeformed-cloud ghost shown while warp is on."""
        self.ghost_warp_reference = bool(enabled)
        return self.ghost_warp_reference

    def set_playing(self, is_playing: bool) -> bool:
        """Set the playback flag; the actual timer lives in the main window."""
        self.is_playing = bool(is_playing)
        return self.is_playing

    def set_playback_speed(self, playback_speed: float) -> float:
        """Set the playback speed multiplier (validated to 0.05..16.0)."""
        self.playback_speed = self._validate_playback_speed(playback_speed)
        return self.playback_speed

    def set_clamp_enabled(self, enabled: bool) -> bool:
        """Toggle user-defined color-range clamping."""
        self.clamp_enabled = bool(enabled)
        return self.clamp_enabled

    def set_user_color_range(self, vmin: float | None, vmax: float | None) -> tuple[float | None, float | None]:
        """Set the manual color range (each side independently nullable)."""
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
        """Update geometry-visibility toggles. ``None`` keeps the current value."""
        if show_internal is not None:
            self.show_internal = bool(show_internal)
        if show_external is not None:
            self.show_external = bool(show_external)
        if show_qa is not None:
            self.show_qa = bool(show_qa)
        return self.show_internal, self.show_external, self.show_qa

    def set_warp_enabled(self, enabled: bool) -> bool:
        """Toggle displacement-warp visualisation (real-motion preview)."""
        self.disp_warp_enabled = bool(enabled)
        return self.disp_warp_enabled

    def set_warp_axes(self, axes: tuple) -> tuple:
        """Set the per-axis warp mask ``(E/X, N/Y, Z)`` -- 3 bools required."""
        self.warp_axes = (bool(axes[0]), bool(axes[1]), bool(axes[2]))
        return self.warp_axes

    def set_warp_scale(self, scale: float | None) -> float | None:
        """Set the warp scale factor; ``None`` enables the adapter's auto value."""
        self.warp_scale = None if scale is None else max(0.0, float(scale))
        return self.warp_scale

    def set_selection_visibility(self, mode: str) -> str:
        """Set how multi-selection interacts with the main actor.

        Parameters
        ----------
        mode : {'all', 'only', 'hide'}
            * ``'all'``  -- draw every visible node normally (default).
            * ``'only'`` -- draw only nodes in ``multi_selection``.
            * ``'hide'`` -- hide nodes in ``multi_selection``, draw the rest.
            Unknown values fall back to ``'all'``.
        """
        if mode not in ("all", "only", "hide"):
            mode = "all"
        self.selection_visibility = mode
        return self.selection_visibility

    def set_node_opacity(self, opacity: float) -> float:
        """Clamp ``opacity`` to ``[0, 1]`` and store it as the main-actor alpha."""
        self.node_opacity = max(0.0, min(1.0, float(opacity)))
        return self.node_opacity

    @staticmethod
    def _validate_demand(demand: str) -> str:
        """Return lowercase ``demand`` after checking against ``VALID_DEMANDS``."""
        demand = demand.lower()
        if demand not in VALID_DEMANDS:
            raise KeyError(
                f"Unknown demand '{demand}'. Use one of {', '.join(VALID_DEMANDS)}."
            )
        return demand

    @staticmethod
    def _validate_component(component: str) -> str:
        """Return lowercase ``component`` after checking against ``VALID_COMPONENTS``."""
        component = component.lower()
        if component not in VALID_COMPONENTS:
            raise KeyError(
                "Unknown component "
                f"'{component}'. Use one of {', '.join(VALID_COMPONENTS)}."
            )
        return component

    @staticmethod
    def _validate_background(background: str) -> str:
        """Return ``background`` if it is a known preset, else raise ``KeyError``."""
        if background not in BACKGROUND_PRESETS:
            raise KeyError(
                f"Unknown background '{background}'. Use one of "
                f"{', '.join(BACKGROUND_PRESETS)}."
            )
        return background

    @staticmethod
    def _validate_playback_speed(playback_speed: float) -> float:
        """Return ``playback_speed`` as a positive float, raising on non-positive."""
        speed = float(playback_speed)
        if speed <= 0.0:
            raise ValueError("playback_speed must be greater than 0.")
        return speed
