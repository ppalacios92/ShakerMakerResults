"""Multi-viewport central area for the interactive viewer.

Each :class:`ViewPane` (one VTK viewport) is wrapped in a
:class:`~.view_frame.ViewFrame` that paints a thin border, a title-bar with
the pane name, and standard ParaView-style buttons (split horizontal /
split vertical, maximize-toggle, pop-out toggle, close).

:class:`MultiViewArea` owns the splitter tree, the camera rail and the
layout-preset selector.  :class:`TabbedMultiViewArea` wraps several
``MultiViewArea`` instances inside a ``QTabWidget`` so the user can keep
multiple layouts side-by-side and switch between them with a click — the
same model ParaView uses with ``pqTabbedMultiViewWidget``.

The window and the global toolbar always talk to a single
``TabbedMultiViewArea`` instance.  Attribute access (``active_pane``,
``active_plotter``, ``_panes``, ``_current_layout``, ``_layout_n``,
``on_active_pane_changed`` …) is proxied through to the currently visible
tab so the legacy callers do not need to be aware that multiple layouts now
coexist.
"""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .busy_dialog import BusyDialog
from .icons import icon
from .scene import ViewerScene
from .theme import active_palette
from .view_frame import ViewFrame

_, QtInteractor, _, QtCore, _, QtWidgets = require_viewer_dependencies()


# ── Constants ─────────────────────────────────────────────────────────────────

#: Default label assigned to new panes in creation order — used for initial
#: camera orientation and as the title shown in the view-frame title-bar.
_PANE_DEFAULTS: list[str] = [
    "3D NE", "Top", "Front", "Right", "3D NW", "3D SW", "3D SE",
]

#: (layout name, number of panes required) — order matches the toolbar.
LAYOUT_PRESETS: list[tuple[str, int]] = [
    ("1×1", 1),
    ("1×2", 2),
    ("2×1", 2),
    ("2×2", 4),
    ("2+1", 3),
    ("1+2", 3),
    ("GF",  9),   # 3×3 Green Function tensor viewer
]

# Promotion targets used by the per-frame split buttons.  The button never
# shrinks the layout — it only "grows" to the next sensible preset along the
# requested axis, mirroring ParaView's pqViewFrame.SplitHorizontal /
# SplitVertical behaviour without rewriting the splitter tree from scratch.
_SPLIT_H_PROMOTIONS: dict[str, str] = {
    "1×1": "1×2",
    "2×1": "2×2",
    "1+2": "2×2",
}
_SPLIT_V_PROMOTIONS: dict[str, str] = {
    "1×1": "2×1",
    "1×2": "2×2",
    "2+1": "2×2",
}

# GF layout: (adapter component id, viewport label) in row-major order.
_GF_LAYOUT_COMPONENTS: list[tuple[str, str]] = [
    ("g11", "G_11"), ("g12", "G_12"), ("g13", "G_13"),
    ("g21", "G_21"), ("g22", "G_22"), ("g23", "G_23"),
    ("g31", "G_31"), ("g32", "G_32"), ("g33", "G_33"),
]

_CAMERA_VIEW_ENTRIES: list[tuple[str, str, str, str]] = [
    ("ISO", "iso_ne", "NE", "view_iso_ne"),
    ("ISO", "iso_nw", "NW", "view_iso_nw"),
    ("ISO", "iso_sw", "SW", "view_iso_sw"),
    ("ISO", "iso_se", "SE", "view_iso_se"),
    ("ORTHO", "top", "Top", "view_top"),
    ("ORTHO", "bottom", "Bot", "view_bottom"),
    ("ORTHO", "front", "Front", "view_front"),
    ("ORTHO", "back", "Back", "view_back"),
    ("ORTHO", "left", "Left", "view_left"),
    ("ORTHO", "right", "Right", "view_right"),
]

# Azimuth offsets for ISO presets (degrees, applied after view_isometric).
_ISO_AZIMUTH: dict[str, int] = {
    "3D NE": 0,
    "3D NW": 90,
    "3D SW": 180,
    "3D SE": 270,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

class _ClickFilter(QtCore.QObject):
    """Event-filter that fires *callback* on any ``MouseButtonPress`` event."""

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._cb = callback

    def eventFilter(self, obj, event):           # noqa: N802
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self._cb()
        return False                              # never consume the event


class CameraViewRail(QtWidgets.QWidget):
    """Vertical camera preset rail attached to the left of the viewport."""

    def __init__(self, on_view_requested, parent=None):
        super().__init__(parent)
        self.setObjectName("CameraViewRail")
        self.setFixedWidth(54)
        self._on_view_requested = on_view_requested

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(2)

        self._buttons: dict[str, QtWidgets.QToolButton] = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)

        previous_section = None
        for section, key, label, icon_name in _CAMERA_VIEW_ENTRIES:
            if section != previous_section:
                if previous_section is not None:
                    layout.addSpacing(4)
                section_label = QtWidgets.QLabel(section)
                section_label.setObjectName("ViewRailSection")
                section_label.setAlignment(QtCore.Qt.AlignCenter)
                layout.addWidget(section_label)
                previous_section = section

            button = QtWidgets.QToolButton()
            button.setObjectName("ViewRailButton")
            button.setText(label)
            button.setIcon(icon(icon_name, active_palette().navy, 18))
            button.setIconSize(QtCore.QSize(18, 18))
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            button.setCheckable(True)
            button.setToolTip(self._tooltip_for(key))
            button.clicked.connect(lambda _checked, k=key: self._request_view(k))
            self._buttons[key] = button
            group.addButton(button)
            layout.addWidget(button)

        layout.addStretch(1)
        self._buttons["iso_ne"].setChecked(True)

    def refresh_theme(self) -> None:
        navy = active_palette().navy
        for key, button in self._buttons.items():
            icon_name = next(
                (icon_name for _, k, _, icon_name in _CAMERA_VIEW_ENTRIES if k == key),
                None,
            )
            if icon_name is not None:
                button.setIcon(icon(icon_name, navy, 18))

    def _request_view(self, key: str):
        callback = self._on_view_requested
        if callable(callback):
            callback(key)

    @staticmethod
    def _tooltip_for(key: str) -> str:
        tips = {
            "iso_ne": "Isometric NE",
            "iso_nw": "Isometric NW",
            "iso_sw": "Isometric SW",
            "iso_se": "Isometric SE",
            "top": "Top view (+Z)",
            "bottom": "Bottom view (-Z)",
            "front": "Front view (-Y)",
            "back": "Back view (+Y)",
            "left": "Left view (-X)",
            "right": "Right view (+X)",
        }
        return tips.get(key, key)


# ── ViewPane ──────────────────────────────────────────────────────────────────

class ViewPane(QtWidgets.QWidget):
    """One VTK viewport hosted inside a :class:`~.view_frame.ViewFrame`.

    Parameters
    ----------
    session:
        The shared viewer session (data + state).
    label:
        Initial camera orientation key (one of the ``_PANE_DEFAULTS``).
        Applied once at construction; also used as the title shown in the
        wrapping :class:`~.view_frame.ViewFrame` title-bar.
    on_activated:
        Callable ``(pane: ViewPane) -> None`` fired when the user clicks
        anywhere in this pane.  Used by :class:`MultiViewArea` to track the
        active pane and update the frame border / title-bar colour.
    """

    def __init__(
        self,
        session,
        label: str = "3D NE",
        on_activated=None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self._on_activated = on_activated
        self._is_active = False
        self._show_station_tags = True
        self._label = str(label)

        self.setMinimumSize(80, 80)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── VTK interactor ───────────────────────────────────────────────────
        self.plotter = QtInteractor(self)
        outer.addWidget(self.plotter.interactor, 1)

        # ── Scene ────────────────────────────────────────────────────────────
        self.scene = ViewerScene(self.plotter, session)
        try:
            self.scene.build(lightweight=True)
            self.scene.set_station_tags_visible(self._show_station_tags, render=False)
            self._apply_initial_camera(label)
        except Exception:
            pass

        # ── Click → activate ─────────────────────────────────────────────────
        self._click_filter = _ClickFilter(self._fire_activated, self)
        self.plotter.interactor.installEventFilter(self._click_filter)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _apply_initial_camera(self, label: str):
        """Set the camera to the position described by *label* at startup."""
        p = self.plotter
        try:
            if label in _ISO_AZIMUTH:
                p.view_isometric()
                az = _ISO_AZIMUTH[label]
                if az:
                    p.camera.Azimuth(az)
                    p.reset_camera_clipping_range()
            elif label == "Top":
                p.view_xy()
            elif label == "Bottom":
                p.view_xy(negative=True)
            elif label == "Front":
                p.view_xz()
            elif label == "Back":
                p.view_xz(negative=True)
            elif label == "Left":
                p.view_yz()
            elif label == "Right":
                p.view_yz(negative=True)
            p.render()
        except Exception:
            pass

    # ── Active-pane highlight ─────────────────────────────────────────────────

    def _fire_activated(self):
        if self._on_activated is not None:
            self._on_activated(self)

    def set_active(self, active: bool):
        """Track active state.

        Visible highlighting is now driven by the wrapping
        :class:`~.view_frame.ViewFrame`'s border / title-bar; this method
        keeps the boolean state for legacy queries.
        """
        self._is_active = bool(active)

    @property
    def label(self) -> str:
        return self._label

    def set_station_tags_visible(self, visible: bool, render: bool = True):
        self._show_station_tags = bool(visible)
        self.scene.set_station_tags_visible(self._show_station_tags, render=render)

    def station_tags_visible(self) -> bool:
        return bool(self._show_station_tags)

    # ── Scene refresh ─────────────────────────────────────────────────────────

    def on_session_updated(self, reason: str):
        """Apply *reason* to this pane's scene and re-render."""
        if reason == "time":
            self.scene.refresh_scalars(render=False)
            if self.session.state.vector_field_enabled:
                self.scene.update_vector_arrows(render=False)
        elif reason == "selection":
            self.scene.refresh_selection(render=False)
        elif reason in {"stations", "stations_visibility"}:
            self.scene.refresh_station_tags(render=False)
        elif reason == "geometry_transform":
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
            self.scene.refresh_station_tags(render=False)
        elif reason in {"demand", "component"}:
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
        elif reason == "visibility":
            self.scene.rebuild_for_visibility(render=False)
        elif reason == "color_range":
            self.scene.apply_color_range(render=False)
        elif reason in {"appearance", "panel_apply", "static_color"}:
            self.scene.apply_appearance(render=False)
            self.scene.refresh_selection(render=False)
        elif reason == "warp":
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
            self.scene.refresh_ghost_reference(render=False)
        elif reason == "multi_selection":
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_multi_selection(render=False)
        elif reason == "node_opacity":
            self.scene.update_node_opacity(render=False)
        elif reason == "point_size":
            self.scene.update_point_sizes(render=False)
        elif reason == "vector_field":
            self.scene.refresh_vector_field(render=False)
        else:
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
        try:
            self.plotter.render()
        except Exception:
            pass

    def dispose(self) -> None:
        """Tear down this pane's scene and VTK/Qt interactor resources."""
        try:
            if self.scene is not None:
                self.scene.dispose()
        except Exception:
            pass

        iren = getattr(self.plotter, "iren", None)
        vtk_interactor = getattr(iren, "interactor", iren)
        for obj in (vtk_interactor, iren):
            if obj is None:
                continue
            for method_name in ("Finalize", "TerminateApp", "Close"):
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

        interactor_widget = getattr(self.plotter, "interactor", None)
        if interactor_widget is not None:
            try:
                interactor_widget.close()
            except Exception:
                pass
            try:
                interactor_widget.deleteLater()
            except Exception:
                pass

        try:
            self.plotter.close()
        except Exception:
            pass

        self.scene = None
        self.plotter = None


# ── MultiViewArea ─────────────────────────────────────────────────────────────

class MultiViewArea(QtWidgets.QWidget):
    """Central zone that manages 1–N synchronised :class:`ViewPane` instances.

    Each pane is wrapped in a :class:`~.view_frame.ViewFrame`.  Switching
    layouts re-parents the *frames* (which in turn hold their panes) so the
    underlying VTK contexts are preserved across layout changes.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._panes: list[ViewPane] = []
        self._frames: dict[ViewPane, ViewFrame] = {}
        self._popout_windows: dict[ViewPane, QtWidgets.QMainWindow] = {}
        self._maximized_frame: ViewFrame | None = None
        self._active_pane: ViewPane | None = None
        self.on_active_pane_changed = None
        self._current_layout: str = "1×1"
        self._container: QtWidgets.QWidget | None = None
        self._layout_n: dict[str, int] = dict(LAYOUT_PRESETS)
        self._apply_camera_to_all_panes = False
        # Demand + component active before the GF 3×3 layout; both restored on exit.
        self._pre_gf_demand: str | None = None
        self._pre_gf_component: str | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Layout selector bar ───────────────────────────────────────────────
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(30)
        palette = active_palette()
        bar.setStyleSheet(
            f"background: {palette.surface_2}; "
            f"border-bottom: 1px solid {palette.border};"
        )
        blay = QtWidgets.QHBoxLayout(bar)
        blay.setContentsMargins(6, 3, 6, 3)
        blay.setSpacing(3)

        lbl = QtWidgets.QLabel("Layout:")
        lbl.setStyleSheet(f"color: {palette.text_2}; font-size: 11px;")
        blay.addWidget(lbl)

        self._layout_buttons: dict[str, QtWidgets.QPushButton] = {}
        for name, _n in LAYOUT_PRESETS:
            btn = QtWidgets.QPushButton(name)
            w = 36 if name == "GF" else 44
            btn.setFixedSize(w, 22)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c, n=name: self._switch_layout(n))
            self._layout_buttons[name] = btn
            blay.addWidget(btn)
        self._refresh_layout_button_styles()

        blay.addStretch(1)
        root.addWidget(bar)
        self._layout_bar = bar

        # ── Content area ──────────────────────────────────────────────────────
        body = QtWidgets.QWidget()
        body_lay = QtWidgets.QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        self._view_rail = CameraViewRail(self._apply_camera_preset, self)
        body_lay.addWidget(self._view_rail)

        # Stack holds two pages: the splitter grid and the maximised slot.
        self._content_stack = QtWidgets.QStackedWidget()
        body_lay.addWidget(self._content_stack, 1)

        self._grid_host = QtWidgets.QWidget()
        self._grid_lay = QtWidgets.QVBoxLayout(self._grid_host)
        self._grid_lay.setContentsMargins(0, 0, 0, 0)
        self._grid_lay.setSpacing(0)
        self._content_stack.addWidget(self._grid_host)             # index 0

        self._max_host = QtWidgets.QWidget()
        self._max_lay = QtWidgets.QVBoxLayout(self._max_host)
        self._max_lay.setContentsMargins(0, 0, 0, 0)
        self._max_lay.setSpacing(0)
        self._content_stack.addWidget(self._max_host)               # index 1

        root.addWidget(body, 1)

        # Start with a single pane.
        self._switch_layout("1×1")

    # ── Theme ─────────────────────────────────────────────────────────────────

    def refresh_theme(self) -> None:
        """Repaint chrome and frame icons after a palette swap."""
        palette = active_palette()
        self._layout_bar.setStyleSheet(
            f"background: {palette.surface_2}; "
            f"border-bottom: 1px solid {palette.border};"
        )
        self._refresh_layout_button_styles()
        for frame in list(self._frames.values()):
            frame.refresh_theme()
        try:
            self._view_rail.refresh_theme()
        except Exception:
            pass

    def _refresh_layout_button_styles(self) -> None:
        palette = active_palette()
        for name, btn in self._layout_buttons.items():
            if name == "GF":
                btn.setStyleSheet(
                    f"QPushButton {{ border: 1px solid {palette.accent_dark}; "
                    f"border-radius: 3px; background: {palette.surface_3}; "
                    f"color: {palette.accent_dark}; font-size: 11px; "
                    f"font-weight: bold; }}"
                    f"QPushButton:checked {{ background: {palette.surface_3}; "
                    f"color: {palette.navy}; border-color: {palette.accent}; }}"
                    f"QPushButton:hover:!checked {{ background: {palette.surface_3}; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ border: 1px solid {palette.border}; "
                    f"border-radius: 3px; background: {palette.surface}; "
                    f"color: {palette.text}; font-size: 11px; }}"
                    f"QPushButton:checked {{ background: {palette.surface_3}; "
                    f"color: {palette.navy}; border-color: {palette.accent}; }}"
                    f"QPushButton:hover:!checked {{ background: {palette.surface_3}; }}"
                )

    # ── Frame helpers ─────────────────────────────────────────────────────────

    def _frame_for(self, pane: ViewPane) -> ViewFrame:
        """Return the existing wrapper for *pane*, creating it on first use."""
        frame = self._frames.get(pane)
        if frame is None:
            frame = ViewFrame(
                pane,
                pane.label,
                on_split_h=self._on_frame_split_h,
                on_split_v=self._on_frame_split_v,
                on_maximize_toggle=self._on_frame_maximize_toggle,
                on_popout_toggle=self._on_frame_popout_toggle,
                on_close=self._on_frame_close,
            )
            self._frames[pane] = frame
        return frame

    def _frame_close_visibility(self, panes: list[ViewPane]) -> None:
        """Show / hide each frame's close button depending on the current layout.

        Closing the only frame in a 1×1 layout would leave an empty grid, so
        we just hide the button instead of disabling it (matches ParaView).
        """
        allow_close = len(panes) > 1
        for pane in panes:
            frame = self._frames.get(pane)
            if frame is not None:
                frame.set_close_button_visible(allow_close)

    # ── Layout switching ──────────────────────────────────────────────────────

    def _apply_camera_preset(self, key: str) -> None:
        """Apply a camera preset from the left view rail."""
        panes = self._camera_target_panes()
        if not panes:
            return

        for pane in panes:
            plotter = getattr(pane, "plotter", None)
            if plotter is not None:
                self._apply_camera_preset_to_plotter(plotter, key)

    def set_camera_apply_to_all(self, enabled: bool) -> None:
        """Mirror the toolbar's All windows state for the left view rail."""
        self._apply_camera_to_all_panes = bool(enabled)

    def _camera_target_panes(self) -> list[ViewPane]:
        if self._apply_camera_to_all_panes:
            return self.visible_panes()
        return [self._active_pane] if self._active_pane is not None else []

    def visible_panes(self) -> list[ViewPane]:
        needed = self._layout_n.get(self._current_layout, 1)
        return self._panes[: max(0, min(needed, len(self._panes)))]

    def apply_selection_filter(self, mode: str, *, apply_to_all: bool = False):
        panes = self.visible_panes() if apply_to_all else [self._active_pane]
        for pane in panes:
            if pane is None or pane.scene is None:
                continue
            pane.scene.apply_selection_filter(mode, render=False)
            try:
                pane.plotter.render()
            except Exception:
                pass

    @staticmethod
    def _apply_camera_preset_to_plotter(plotter, key: str) -> None:
        try:
            if key.startswith("iso_"):
                plotter.view_isometric()
                azimuth = {
                    "iso_ne": 0,
                    "iso_nw": 90,
                    "iso_sw": 180,
                    "iso_se": 270,
                }.get(key, 0)
                if azimuth:
                    plotter.camera.Azimuth(azimuth)
                    plotter.reset_camera_clipping_range()
            elif key == "top":
                plotter.view_xy()
            elif key == "bottom":
                plotter.view_xy(negative=True)
            elif key == "front":
                plotter.view_xz()
            elif key == "back":
                plotter.view_xz(negative=True)
            elif key == "left":
                plotter.view_yz(negative=True)
            elif key == "right":
                plotter.view_yz()
            plotter.render()
        except Exception:
            pass

    def _switch_layout(self, name: str):
        """Switch to layout *name*, creating panes / frames as needed."""
        if name not in self._layout_n:
            return

        # Exit any maximised view before re-laying-out the grid.
        if self._maximized_frame is not None:
            self._maximized_frame.set_maximized_state(False)
            self._restore_maximised_frame()

        old_layout = self._current_layout
        needed = self._layout_n.get(name, 1)

        # 1. Detach all panes / frames from the current container.
        for pane in self._panes:
            frame = self._frames.get(pane)
            target = frame if frame is not None else pane
            target.setParent(self)
            target.hide()

        # 2. Remove and schedule the old container for deletion.
        if self._container is not None:
            self._grid_lay.removeWidget(self._container)
            self._container.setParent(None)
            self._container.deleteLater()
            self._container = None

        # 3. Create additional panes if the layout needs more than we have.
        while len(self._panes) < needed:
            label = _PANE_DEFAULTS[len(self._panes) % len(_PANE_DEFAULTS)]
            pane = ViewPane(
                self.session,
                label=label,
                on_activated=self._on_pane_activated,
                parent=self,
            )
            self._panes.append(pane)

        # Materialise frames for every pane we are about to show.
        panes_for_layout = self._panes[:needed]
        for pane in panes_for_layout:
            self._frame_for(pane)

        # 4. Build the new splitter/container.
        self._container = self._build_container(name, panes_for_layout)
        self._grid_lay.addWidget(self._container)
        self._container.show()

        for pane in panes_for_layout:
            frame = self._frames[pane]
            frame.show()

        # 5. Ensure a valid active pane.
        if self._active_pane not in panes_for_layout:
            self._set_active_pane(panes_for_layout[0] if panes_for_layout else None)
        else:
            # Refresh the active flag on the (still-active) pane's frame.
            self._set_active_pane(self._active_pane)

        # 6. Sync button states.
        self._current_layout = name
        for n, btn in self._layout_buttons.items():
            btn.setChecked(n == name)

        # 7. Frame close visibility — hidden on the only pane of 1×1.
        self._frame_close_visibility(panes_for_layout)

        # 8. GF-specific configuration.
        if name == "GF":
            self._apply_gf_layout(panes_for_layout)
        elif old_layout == "GF":
            self._clear_gf_layout(panes_for_layout)

        # Show the grid page in the content stack.
        self._content_stack.setCurrentIndex(0)

    def _build_container(
        self, name: str, panes: list[ViewPane]
    ) -> QtWidgets.QWidget:
        """Assemble a splitter tree for *name* using the given pane list.

        The splitter holds *frames* (so the title-bar / border are always
        visible) rather than raw panes.
        """

        def _h(*frames) -> QtWidgets.QSplitter:
            s = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            s.setChildrenCollapsible(False)
            for f in frames:
                s.addWidget(f)
            return s

        def _v(*frames) -> QtWidgets.QSplitter:
            s = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            s.setChildrenCollapsible(False)
            for f in frames:
                s.addWidget(f)
            return s

        frames = [self._frame_for(p) for p in panes]

        if name == "1×1":
            w = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(frames[0])
            return w

        if name == "1×2":
            return _h(frames[0], frames[1])

        if name == "2×1":
            return _v(frames[0], frames[1])

        if name == "2×2":
            return _v(_h(frames[0], frames[1]), _h(frames[2], frames[3]))

        if name == "2+1":
            outer = _v(_h(frames[0], frames[1]), frames[2])
            outer.setSizes([600, 300])
            return outer

        if name == "1+2":
            outer = _v(frames[0], _h(frames[1], frames[2]))
            outer.setSizes([300, 600])
            return outer

        if name == "GF":
            row0 = _h(frames[0], frames[1], frames[2])
            row1 = _h(frames[3], frames[4], frames[5])
            row2 = _h(frames[6], frames[7], frames[8])
            return _v(row0, row1, row2)

        return _h(*frames)

    # ── Frame-button handlers ────────────────────────────────────────────────

    def _on_frame_split_h(self, frame: ViewFrame) -> None:
        target = _SPLIT_H_PROMOTIONS.get(self._current_layout)
        if target is not None:
            self._switch_layout(target)

    def _on_frame_split_v(self, frame: ViewFrame) -> None:
        target = _SPLIT_V_PROMOTIONS.get(self._current_layout)
        if target is not None:
            self._switch_layout(target)

    def _on_frame_maximize_toggle(self, frame: ViewFrame, wants_maximized: bool):
        if wants_maximized:
            # Restore any previous maximised frame first.
            if self._maximized_frame is not None and self._maximized_frame is not frame:
                self._maximized_frame.set_maximized_state(False)
                self._restore_maximised_frame()
            self._maximise_frame(frame)
        else:
            self._restore_maximised_frame()

    def _on_frame_popout_toggle(self, frame: ViewFrame, wants_popout: bool):
        pane = frame.pane
        if not isinstance(pane, ViewPane):
            return
        if wants_popout:
            self._popout_frame(frame)
        else:
            self._dock_frame(frame)

    def _on_frame_close(self, frame: ViewFrame) -> None:
        # "Closing" reduces the current layout by one preset slot.  For the
        # grid presets we walk back through the LAYOUT_PRESETS to the closest
        # smaller compatible one.  This mirrors ParaView's "close pane"
        # behaviour without rewriting splitter trees by hand.
        downgrades = {
            "1×2": "1×1",
            "2×1": "1×1",
            "2×2": "1×2",
            "2+1": "1×2",
            "1+2": "1×2",
            "GF":  "1×1",
        }
        target = downgrades.get(self._current_layout)
        if target is not None:
            self._switch_layout(target)

    # ── Maximise / restore ───────────────────────────────────────────────────

    def _maximise_frame(self, frame: ViewFrame) -> None:
        # Remember its splitter slot so we can put it back later.
        parent = frame.parentWidget()
        index = None
        if isinstance(parent, QtWidgets.QSplitter):
            index = parent.indexOf(frame)
        self._maximize_origin: tuple[QtWidgets.QSplitter | None, int | None] = (
            parent if isinstance(parent, QtWidgets.QSplitter) else None,
            index,
        )
        self._maximized_frame = frame
        # Reparent into the maximise host and show stack[1].
        frame.setParent(self._max_host)
        self._max_lay.addWidget(frame)
        frame.show()
        self._content_stack.setCurrentIndex(1)

    def _restore_maximised_frame(self) -> None:
        frame = self._maximized_frame
        if frame is None:
            return
        origin = getattr(self, "_maximize_origin", (None, None))
        splitter, index = origin
        self._max_lay.removeWidget(frame)
        if isinstance(splitter, QtWidgets.QSplitter) and index is not None:
            splitter.insertWidget(index, frame)
        else:
            # Fallback: rebuild the current layout from scratch.
            self._switch_layout(self._current_layout)
        frame.show()
        self._maximized_frame = None
        self._content_stack.setCurrentIndex(0)

    # ── Pop-out / re-dock ────────────────────────────────────────────────────

    def _popout_frame(self, frame: ViewFrame) -> None:
        pane = frame.pane
        if pane in self._popout_windows:
            return
        # Remember where to put the frame back when re-docked.
        parent = frame.parentWidget()
        origin_splitter = parent if isinstance(parent, QtWidgets.QSplitter) else None
        origin_index = origin_splitter.indexOf(frame) if origin_splitter is not None else None
        frame.setProperty("_origin_splitter", origin_splitter)
        frame.setProperty("_origin_index", origin_index)

        # Build a small top-level window for the frame.
        win = QtWidgets.QMainWindow()
        win.setWindowTitle(f"View — {frame.title()}")
        win.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        win.resize(800, 600)
        win.setCentralWidget(frame)

        def _on_close(_event):
            # User closed the floating window — re-dock instead of disposing.
            frame.set_popout_state(False)
            self._dock_frame(frame)
            _event.accept()

        win.closeEvent = _on_close  # type: ignore[assignment]
        self._popout_windows[pane] = win
        win.show()
        # The slot in the grid is now empty — leave it; QSplitter handles
        # missing children by giving them zero size.  Re-docking restores it.

    def _dock_frame(self, frame: ViewFrame) -> None:
        pane = frame.pane
        win = self._popout_windows.pop(pane, None)
        if win is None:
            return
        win.setCentralWidget(None)
        origin_splitter = frame.property("_origin_splitter")
        origin_index = frame.property("_origin_index")
        if isinstance(origin_splitter, QtWidgets.QSplitter) and origin_index is not None:
            origin_splitter.insertWidget(int(origin_index), frame)
            frame.show()
        else:
            # Could not find an origin slot — rebuild the layout to recover.
            self._switch_layout(self._current_layout)
        try:
            win.close()
            win.deleteLater()
        except Exception:
            pass

    # ── GF layout helpers ─────────────────────────────────────────────────────

    def _apply_gf_layout(self, panes: list) -> None:
        session = self.session
        self._pre_gf_demand = session.state.demand
        self._pre_gf_component = session.state.component

        if session.adapter.has_gf and session.adapter.has_map:
            sf = int(getattr(session, "_display_gf_subfault", 0))
            window = getattr(session, "window", None)
            busy = None
            if window is not None:
                busy = BusyDialog(
                    "Warming Green Function series...\nSeries 0 / 9  -  G_11",
                    window,
                    total_steps=9,
                )
                busy.show()
                QtWidgets.QApplication.processEvents()

            for idx in range(9):
                comp_label = _GF_LAYOUT_COMPONENTS[idx][1]
                if busy is not None:
                    busy.set_message(
                        f"Warming Green Function series...\n"
                        f"Series {idx + 1} / 9  -  {comp_label}"
                    )
                    QtWidgets.QApplication.processEvents()
                try:
                    session.adapter.warm_gf_series(sf, idx)
                except Exception:
                    pass
                if busy is not None:
                    busy.set_step(idx + 1)
                    QtWidgets.QApplication.processEvents()

            if busy is not None:
                busy.close()

        for pane, (comp, label) in zip(panes, _GF_LAYOUT_COMPONENTS):
            try:
                scene = getattr(pane, "scene", None)
                if scene is None:
                    continue
                try:
                    pane.plotter.view_xy()
                    pane.plotter.enable_parallel_projection()
                except Exception:
                    pass
                scene.set_gf_component_pin(comp, label)
                frame = self._frames.get(pane)
                if frame is not None:
                    frame.set_title(label)
            except Exception:
                pass

        if session.adapter.has_gf and session.adapter.has_map:
            if session.state.demand != "gf":
                session.set_demand("gf")
            else:
                for pane in panes:
                    try:
                        if pane.scene is not None:
                            pane.scene.rebuild_scalar_actor(render=False)
                            pane.plotter.render()
                    except Exception:
                        pass
        else:
            for pane in panes:
                try:
                    if pane.scene is not None:
                        pane.scene.rebuild_scalar_actor(render=False)
                        pane.plotter.render()
                except Exception:
                    pass

    def _clear_gf_layout(self, panes_for_new_layout: list) -> None:
        session = self.session

        for pane in self._panes:
            try:
                scene = getattr(pane, "scene", None)
                if scene is not None and scene._gf_component_pin is not None:
                    scene.set_gf_component_pin(None, None)
            except Exception:
                pass
            # Restore the frame title to the pane's original label.
            frame = self._frames.get(pane)
            if frame is not None:
                frame.set_title(pane.label)

        pre_demand = self._pre_gf_demand
        pre_comp   = self._pre_gf_component
        self._pre_gf_demand    = None
        self._pre_gf_component = None

        if pre_demand is not None and pre_demand != session.state.demand:
            if pre_comp is not None:
                try:
                    session.state.set_component(pre_comp)
                except Exception:
                    pass
            session.set_demand(pre_demand)
        else:
            if pre_comp is not None:
                try:
                    session.state.set_component(pre_comp)
                except Exception:
                    pass
            for pane in panes_for_new_layout:
                try:
                    if pane.scene is not None:
                        pane.scene.rebuild_scalar_actor(render=False)
                        pane.plotter.render()
                except Exception:
                    pass

    # ── Active pane ───────────────────────────────────────────────────────────

    def _on_pane_activated(self, pane: ViewPane):
        self._set_active_pane(pane)

    def _set_active_pane(self, pane: ViewPane | None):
        for p in self._panes:
            frame = self._frames.get(p)
            is_active = p is pane
            p.set_active(is_active)
            if frame is not None:
                frame.set_active(is_active)
        self._active_pane = pane
        callback = self.on_active_pane_changed
        if callable(callback):
            try:
                callback(pane)
            except Exception:
                pass

    @property
    def active_pane(self) -> ViewPane | None:
        return self._active_pane

    @property
    def active_plotter(self):
        if self._active_pane is not None:
            return self._active_pane.plotter
        return self._panes[0].plotter if self._panes else None

    # ── Session broadcast ─────────────────────────────────────────────────────

    def on_session_updated(self, reason: str):
        """Broadcast *reason* to every pane that is currently visible."""
        needed = self._layout_n.get(self._current_layout, 1)
        for pane in self._panes[:needed]:
            pane.on_session_updated(reason)

    def dispose(self) -> None:
        """Dispose every pane, frame, popout window and layout container."""
        # Close any still-floating popout windows first.
        for win in list(self._popout_windows.values()):
            try:
                win.setCentralWidget(None)
            except Exception:
                pass
            try:
                win.close()
                win.deleteLater()
            except Exception:
                pass
        self._popout_windows.clear()

        for pane in list(self._panes):
            try:
                pane.dispose()
            except Exception:
                pass
            frame = self._frames.pop(pane, None)
            if frame is not None:
                try:
                    frame.setParent(None)
                    frame.deleteLater()
                except Exception:
                    pass
            try:
                pane.setParent(None)
                pane.deleteLater()
            except Exception:
                pass
        self._panes.clear()
        self._active_pane = None

        if self._container is not None:
            try:
                self._grid_lay.removeWidget(self._container)
            except Exception:
                pass
            try:
                self._container.setParent(None)
                self._container.deleteLater()
            except Exception:
                pass
            self._container = None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close_all_panes(self):
        for pane in self._panes:
            try:
                pane.plotter.close()
            except Exception:
                pass


# ── TabbedMultiViewArea ───────────────────────────────────────────────────────

class TabbedMultiViewArea(QtWidgets.QWidget):
    """ParaView-style tabbed wrapper around several :class:`MultiViewArea`.

    Each tab is an independent :class:`MultiViewArea` with its own grid, its
    own panes and its own active selection.  Attribute access is proxied to
    the currently active tab so the existing toolbar and main window keep
    working without modification.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._on_active_pane_changed_external = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.tabBar().setObjectName("MultiViewTabs")
        self._tabs.setMovable(True)
        self._tabs.setTabsClosable(False)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # "+" corner button — adds a new layout tab on demand.
        self._add_btn = QtWidgets.QToolButton()
        self._add_btn.setText("+")
        self._add_btn.setToolTip("Add a new viewport tab")
        self._add_btn.setAutoRaise(True)
        self._add_btn.setFixedSize(22, 22)
        self._add_btn.clicked.connect(lambda: self.add_tab())
        self._tabs.setCornerWidget(self._add_btn, QtCore.Qt.TopRightCorner)

        root.addWidget(self._tabs)

        # Bootstrap: one Main view tab.
        self.add_tab(label="Main view")

    # ── Tab management ───────────────────────────────────────────────────────

    def add_tab(self, label: str | None = None) -> MultiViewArea:
        area = MultiViewArea(self.session)
        area.on_active_pane_changed = self._proxy_active_pane_changed
        text = label or f"View {self._tabs.count() + 1}"
        idx = self._tabs.addTab(area, text)
        self._tabs.setCurrentIndex(idx)
        self._tabs.setTabsClosable(self._tabs.count() > 1)
        return area

    def _on_tab_close_requested(self, index: int) -> None:
        if self._tabs.count() <= 1:
            return
        area = self._tabs.widget(index)
        self._tabs.removeTab(index)
        if isinstance(area, MultiViewArea):
            try:
                area.dispose()
            except Exception:
                pass
            try:
                area.deleteLater()
            except Exception:
                pass
        self._tabs.setTabsClosable(self._tabs.count() > 1)

    def _on_tab_changed(self, _index: int) -> None:
        self._proxy_active_pane_changed(self.active_pane)

    def _proxy_active_pane_changed(self, pane) -> None:
        cb = self._on_active_pane_changed_external
        if callable(cb):
            try:
                cb(pane)
            except Exception:
                pass

    # ── Proxy attributes (read by the toolbar / window) ──────────────────────

    @property
    def current(self) -> MultiViewArea | None:
        widget = self._tabs.currentWidget()
        return widget if isinstance(widget, MultiViewArea) else None

    @property
    def on_active_pane_changed(self):
        return self._on_active_pane_changed_external

    @on_active_pane_changed.setter
    def on_active_pane_changed(self, callback):
        self._on_active_pane_changed_external = callback

    @property
    def active_pane(self):
        c = self.current
        return c.active_pane if c is not None else None

    @property
    def active_plotter(self):
        c = self.current
        return c.active_plotter if c is not None else None

    @property
    def _panes(self):
        c = self.current
        return c._panes if c is not None else []

    @property
    def _current_layout(self):
        c = self.current
        return c._current_layout if c is not None else "1×1"

    @property
    def _layout_n(self):
        c = self.current
        return c._layout_n if c is not None else dict(LAYOUT_PRESETS)

    def visible_panes(self):
        c = self.current
        return c.visible_panes() if c is not None else []

    def apply_selection_filter(self, mode: str, *, apply_to_all: bool = False):
        c = self.current
        if c is not None:
            c.apply_selection_filter(mode, apply_to_all=apply_to_all)

    def set_camera_apply_to_all(self, enabled: bool):
        c = self.current
        if c is not None:
            c.set_camera_apply_to_all(enabled)

    def on_session_updated(self, reason: str) -> None:
        """Broadcast *reason* to every tab so background tabs stay in sync."""
        for i in range(self._tabs.count()):
            area = self._tabs.widget(i)
            if isinstance(area, MultiViewArea):
                try:
                    area.on_session_updated(reason)
                except Exception:
                    pass

    def dispose(self) -> None:
        while self._tabs.count() > 0:
            area = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if isinstance(area, MultiViewArea):
                try:
                    area.dispose()
                except Exception:
                    pass
                try:
                    area.deleteLater()
                except Exception:
                    pass

    def refresh_theme(self) -> None:
        for i in range(self._tabs.count()):
            area = self._tabs.widget(i)
            if hasattr(area, "refresh_theme"):
                try:
                    area.refresh_theme()
                except Exception:
                    pass


__all__ = ["MultiViewArea", "TabbedMultiViewArea", "ViewPane", "LAYOUT_PRESETS"]
