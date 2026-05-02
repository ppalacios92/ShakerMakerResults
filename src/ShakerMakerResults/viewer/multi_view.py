"""Multi-viewport central area for the interactive viewer.

Provides :class:`ViewPane` (one VTK viewport) and :class:`MultiViewArea`
(a layout-selector bar + a dynamic grid of panes).

Design contract
---------------
* A single :class:`~.session.ViewerSession` is shared by every pane — all
  state (time, demand, selection, warp …) is global.
* Each pane owns an independent :class:`~pyvistaqt.QtInteractor` and a
  :class:`~.scene.ViewerScene`, so cameras and orientations are per-pane.
* On every ``on_session_updated(reason)`` call from the window, the
  ``MultiViewArea`` broadcasts the reason to every visible pane; each pane
  calls the matching scene method and then ``plotter.render()``.
* Clicking anywhere in a pane marks it as *active* (thin blue top-bar).
  The toolbar's view-preset and capture operations act on the active pane.
"""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .busy_dialog import BusyDialog
from .icons import icon
from .scene import ViewerScene
from .theme import LIGHT_PALETTE

_, QtInteractor, _, QtCore, _, QtWidgets = require_viewer_dependencies()


# ── Constants ─────────────────────────────────────────────────────────────────

#: Default label assigned to new panes in creation order (used for initial
#: camera orientation only — there is no visible combo in the pane UI).
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

# GF layout: (adapter component id, viewport label) in row-major order.
# Row 0 → receiver component 1 (Z), Row 1 → 2 (N), Row 2 → 3 (E)
# Col 0 → source component 1,      Col 1 → 2,       Col 2 → 3
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
            button.setIcon(icon(icon_name, LIGHT_PALETTE.navy, 18))
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
    """One viewport: 3 px active-indicator strip + full-height QtInteractor.

    No per-pane menu or combo is shown — camera presets are controlled
    exclusively from the toolbar above.

    Parameters
    ----------
    session:
        The shared viewer session (data + state).
    label:
        Initial camera orientation key (one of the ``_PANE_DEFAULTS``).
        Applied once at construction; not displayed.
    on_activated:
        Callable ``(pane: ViewPane) -> None`` fired when the user clicks
        anywhere in this pane.  Used by :class:`MultiViewArea` to track the
        active pane.
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

        self.setMinimumSize(80, 80)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Active-pane indicator (thin colored strip at the top) ─────────────
        self._indicator = QtWidgets.QWidget()
        self._indicator.setFixedHeight(3)
        self._indicator.setVisible(False)
        outer.addWidget(self._indicator)

        # ── VTK interactor ───────────────────────────────────────────────────
        self.plotter = QtInteractor(self)
        outer.addWidget(self.plotter.interactor, 1)

        # ── Scene ────────────────────────────────────────────────────────────
        self.scene = ViewerScene(self.plotter, session)
        try:
            self.scene.build()
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
        if self._is_active == active:
            return
        self._is_active = active
        self._indicator.setVisible(active)
        if active:
            self._indicator.setStyleSheet("background: #1565C0;")

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
        elif reason == "multi_selection":
            # Node added/removed from selection set, or show/hide filter changed.
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_multi_selection(render=False)
        elif reason == "node_opacity":
            self.scene.update_node_opacity(render=False)
        else:
            # "full" or any unknown reason — full rebuild.
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

    The layout is controlled by a compact selector bar at the top.  Switching
    layouts reparents existing panes into a new splitter tree without destroying
    them, so cameras and rendering state are preserved across layout changes.

    Parameters
    ----------
    session:
        The shared viewer session passed to every new :class:`ViewPane`.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._panes: list[ViewPane] = []
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
        bar.setStyleSheet("background: #f0f0f0; border-bottom: 1px solid #d0d0d0;")
        blay = QtWidgets.QHBoxLayout(bar)
        blay.setContentsMargins(6, 3, 6, 3)
        blay.setSpacing(3)

        lbl = QtWidgets.QLabel("Layout:")
        lbl.setStyleSheet("color: #404040; font-size: 11px;")
        blay.addWidget(lbl)

        self._layout_buttons: dict[str, QtWidgets.QPushButton] = {}
        for name, _n in LAYOUT_PRESETS:
            btn = QtWidgets.QPushButton(name)
            # GF button gets a slightly different style to distinguish it as
            # a "specialised" preset rather than a generic grid selector.
            w = 36 if name == "GF" else 44
            btn.setFixedSize(w, 22)
            btn.setCheckable(True)
            if name == "GF":
                btn.setStyleSheet(
                    "QPushButton { border: 1px solid #1565C0; border-radius: 3px;"
                    "  background: #e8f0fe; color: #1565C0; font-size: 11px;"
                    "  font-weight: bold; }"
                    "QPushButton:checked { background: #e8f0fe; color: #1e3558;"
                    "  border-color: #2a6bc2; }"
                    "QPushButton:hover:!checked { background: #c5d8fa; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { border: 1px solid #bbb; border-radius: 3px;"
                    "  background: #fff; font-size: 11px; }"
                    "QPushButton:checked { background: #e8f0fe; color: #1e3558;"
                    "  border-color: #2a6bc2; }"
                    "QPushButton:hover:!checked { background: #e3eaf6; }"
                )
            btn.clicked.connect(lambda _c, n=name: self._switch_layout(n))
            self._layout_buttons[name] = btn
            blay.addWidget(btn)

        blay.addStretch(1)
        root.addWidget(bar)

        # ── Content area ──────────────────────────────────────────────────────
        body = QtWidgets.QWidget()
        body_lay = QtWidgets.QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        self._view_rail = CameraViewRail(self._apply_camera_preset, self)
        body_lay.addWidget(self._view_rail)

        self._content = QtWidgets.QWidget()
        self._content_lay = QtWidgets.QVBoxLayout(self._content)
        self._content_lay.setContentsMargins(0, 0, 0, 0)
        self._content_lay.setSpacing(0)
        body_lay.addWidget(self._content, 1)
        root.addWidget(body, 1)

        # Start with a single pane.
        self._switch_layout("1×1")

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
        """Switch to layout *name*, creating panes as needed."""
        old_layout = self._current_layout
        needed = self._layout_n.get(name, 1)

        # 1. Detach all panes from the current container (reparent, not delete).
        for pane in self._panes:
            pane.setParent(self)
            pane.hide()

        # 2. Remove and schedule the old container for deletion.
        if self._container is not None:
            self._content_lay.removeWidget(self._container)
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

        # 4. Build the new splitter/container.
        panes_for_layout = self._panes[:needed]
        self._container = self._build_container(name, panes_for_layout)
        self._content_lay.addWidget(self._container)
        self._container.show()

        # addWidget already reparented each pane into the container.
        for pane in panes_for_layout:
            pane.show()

        # 5. Ensure a valid active pane.
        if self._active_pane not in panes_for_layout:
            self._set_active_pane(panes_for_layout[0] if panes_for_layout else None)

        # 6. Sync button states.
        self._current_layout = name
        for n, btn in self._layout_buttons.items():
            btn.setChecked(n == name)

        # 7. GF-specific configuration.
        # Entering GF: auto-switch demand to GF + pin each pane to one component.
        # Leaving GF: clear pins and restore the previous demand.
        if name == "GF":
            self._apply_gf_layout(panes_for_layout)
        elif old_layout == "GF":
            self._clear_gf_layout(panes_for_layout)

    def _build_container(
        self, name: str, panes: list[ViewPane]
    ) -> QtWidgets.QWidget:
        """Assemble a splitter tree for *name* using the given pane list."""

        def _h(*ps) -> QtWidgets.QSplitter:
            s = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            s.setChildrenCollapsible(False)
            for p in ps:
                s.addWidget(p)
            return s

        def _v(*ps) -> QtWidgets.QSplitter:
            s = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            s.setChildrenCollapsible(False)
            for p in ps:
                s.addWidget(p)
            return s

        if name == "1×1":
            w = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(panes[0])
            return w

        if name == "1×2":
            return _h(panes[0], panes[1])

        if name == "2×1":
            return _v(panes[0], panes[1])

        if name == "2×2":
            return _v(_h(panes[0], panes[1]), _h(panes[2], panes[3]))

        if name == "2+1":
            # Two panes on the top row, one full-width on the bottom.
            outer = _v(_h(panes[0], panes[1]), panes[2])
            outer.setSizes([600, 300])
            return outer

        if name == "1+2":
            # One full-width pane on top, two panes on the bottom row.
            outer = _v(panes[0], _h(panes[1], panes[2]))
            outer.setSizes([300, 600])
            return outer

        if name == "GF":
            # 3 × 3 tensor grid — each pane shows one GF component.
            # Row 0: G_11 G_12 G_13
            # Row 1: G_21 G_22 G_23
            # Row 2: G_31 G_32 G_33
            row0 = _h(panes[0], panes[1], panes[2])
            row1 = _h(panes[3], panes[4], panes[5])
            row2 = _h(panes[6], panes[7], panes[8])
            return _v(row0, row1, row2)

        # Fallback — plain horizontal split.
        return _h(*panes)

    # ── GF layout helpers ─────────────────────────────────────────────────────

    def _apply_gf_layout(self, panes: list) -> None:
        """Configure the 9 GF panes: Top view + per-pane component pin.

        Steps
        -----
        1. Save the current demand so it can be restored when leaving GF layout.
        2. Pre-warm all 9 GF component series in one pass (avoids per-frame I/O).
        3. Set Top / orthographic camera on every pane.
        4. Assign each pane its component pin and label overlay.
        5. Switch session demand to "gf" — the normal demand-change broadcast
           rebuilds every pane's scalar actor using its pin.  If demand is
           already "gf" the panes are rebuilt directly.
        """
        session = self.session

        # Step 1 — remember old demand + component so both can be restored on exit.
        self._pre_gf_demand = session.state.demand
        self._pre_gf_component = session.state.component

        # Step 2 — pre-warm all 9 GF series so playback is instant.
        # Show the unified busy dialog while loading all components.
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
                comp_label = _GF_LAYOUT_COMPONENTS[idx][1]   # e.g. "G_11"
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

        # Steps 3 & 4 — camera + pins (before the demand switch so the first
        # rebuild already uses the correct per-pane component).
        for pane, (comp, label) in zip(panes, _GF_LAYOUT_COMPONENTS):
            try:
                scene = getattr(pane, "scene", None)
                if scene is None:
                    continue
                # Top (plan) view + orthographic for a clean 2-D heat map.
                try:
                    pane.plotter.view_xy()
                    pane.plotter.enable_parallel_projection()
                except Exception:
                    pass
                scene.set_gf_component_pin(comp, label)
            except Exception:
                pass

        # Step 5 — switch demand.
        if session.adapter.has_gf and session.adapter.has_map:
            if session.state.demand != "gf":
                # set_demand triggers the normal window broadcast → rebuilds
                # all 9 active panes via on_session_updated("demand").
                session.set_demand("gf")
            else:
                # Demand is already GF — rebuild panes manually (broadcast
                # would be a no-op because the demand did not change).
                for pane in panes:
                    try:
                        if pane.scene is not None:
                            pane.scene.rebuild_scalar_actor(render=False)
                            pane.plotter.render()
                    except Exception:
                        pass
        else:
            # No GF data — rebuild panes so they show whatever demand is active,
            # ignoring the (inactive) pins.
            for pane in panes:
                try:
                    if pane.scene is not None:
                        pane.scene.rebuild_scalar_actor(render=False)
                        pane.plotter.render()
                except Exception:
                    pass

    def _clear_gf_layout(self, panes_for_new_layout: list) -> None:
        """Remove GF component pins from all panes and restore the old demand.

        Called when the user switches *away* from the GF layout.  The pins are
        cleared on every pane (not just the currently visible 9), so stale
        overrides can never bleed into a later layout switch.
        """
        session = self.session

        # Clear pins on ALL panes (some may have been hidden, but they still
        # carry their pin state and could become active again later).
        for pane in self._panes:
            try:
                scene = getattr(pane, "scene", None)
                if scene is not None and scene._gf_component_pin is not None:
                    scene.set_gf_component_pin(None, None)
            except Exception:
                pass

        # Restore the demand + component that were active before GF layout.
        pre_demand = self._pre_gf_demand
        pre_comp   = self._pre_gf_component
        self._pre_gf_demand    = None
        self._pre_gf_component = None

        if pre_demand is not None and pre_demand != session.state.demand:
            # Restore the component first (set_demand auto-corrects it anyway,
            # but setting it explicitly preserves the user's original choice).
            if pre_comp is not None:
                try:
                    session.state.set_component(pre_comp)
                except Exception:
                    pass
            # set_demand fires the normal broadcast → rebuilds the new layout's
            # panes with their cleared pins.
            session.set_demand(pre_demand)
        else:
            # Demand unchanged — restore component manually and rebuild.
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
            p.set_active(p is pane)
        self._active_pane = pane
        callback = self.on_active_pane_changed
        if callable(callback):
            try:
                callback(pane)
            except Exception:
                pass

    @property
    def active_pane(self) -> ViewPane | None:
        """The pane the user last clicked, or the first pane."""
        return self._active_pane

    @property
    def active_plotter(self):
        """Plotter of the active pane — used by the toolbar for capture/views."""
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
        """Dispose every pane and release layout containers."""
        for pane in list(self._panes):
            try:
                pane.dispose()
            except Exception:
                pass
            try:
                pane.setParent(None)
            except Exception:
                pass
            try:
                pane.deleteLater()
            except Exception:
                pass
        self._panes.clear()
        self._active_pane = None

        if self._container is not None:
            try:
                self._content_lay.removeWidget(self._container)
            except Exception:
                pass
            try:
                self._container.setParent(None)
            except Exception:
                pass
            try:
                self._container.deleteLater()
            except Exception:
                pass
            self._container = None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close_all_panes(self):
        """Close every pane's VTK interactor, releasing all OpenGL contexts.

        Called by :class:`~.window.ViewerMainWindow` ``closeEvent`` so that
        subsequent PyVista / matplotlib operations in the same process are
        not blocked by orphaned VTK render windows.
        """
        for pane in self._panes:
            try:
                pane.plotter.close()
            except Exception:
                pass
