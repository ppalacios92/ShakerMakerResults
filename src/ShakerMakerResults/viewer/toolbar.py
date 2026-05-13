"""Themed Qt toolbars for the interactive viewer.

ParaView splits its top toolbar into a handful of small purpose-specific
toolbars (camera, axes, animation time, color, representation, custom
viewpoints …).  Each one is independently floatable, hideable from the
``View → Toolbars`` menu, and reorderable by dragging.

We follow the same pattern: :func:`build_viewer_toolbars` returns a list of
:class:`QToolBar` instances that the :class:`ViewerMainWindow` adds via
``addToolBar``.  Each toolbar carries a class attribute ``windowTitle`` that
shows up in the toggle action of the View menu.

The recording / screenshot toolbar additionally exposes
``write_frame_if_recording()`` so the playback timer in
:class:`ViewerMainWindow` can request one frame per simulation tick during
an active recording.
"""

from __future__ import annotations

import math
import os

from ._imports import require_viewer_dependencies
from .icons import icon as viewer_icon
from .theme import active_palette

import numpy as np

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False


# ── ffmpeg path ───────────────────────────────────────────────────────────────
# Set this to your local ffmpeg.exe if ffmpeg is not on the system PATH.
# Leave as None to rely on the system PATH or imageio-ffmpeg auto-detection.
FFMPEG_EXE: str | None = r"C:\Dropbox\01. Brain\10. Ph.D U ANDES\21. ffmpeg\ffmpeg-2025-09-18-git-c373636f55-essentials_build\bin\ffmpeg.exe"
# ─────────────────────────────────────────────────────────────────────────────


def _icon(name: str, color: str = "#404040"):
    if _HAS_QTA:
        return qta.icon(name, color=color)
    return None


_ISO_PRESETS = [
    ("NE", "Isometric — North-East",  0),
    ("NW", "Isometric — North-West", 90),
    ("SW", "Isometric — South-West", 180),
    ("SE", "Isometric — South-East", 270),
]

_VIEW_PRESETS = [
    ("Top",   "Top view (+Z)",    "mdi.arrow-collapse-up",   "view_xy",  {}),
    ("Bot",   "Bottom view (−Z)", "mdi.arrow-collapse-down", "view_xy",  {"negative": True}),
    ("Front", "Front view (−Y)",  "mdi.arrow-expand-up",     "view_xz",  {}),
    ("Back",  "Back view (+Y)",   "mdi.arrow-expand-down",   "view_xz",  {"negative": True}),
    ("Left",  "Left view (−X)",   "mdi.arrow-expand-left",   "view_yz",  {"negative": True}),
    ("Right", "Right view (+X)",  "mdi.arrow-expand-right",  "view_yz",  {}),
]


# ── Shared helpers (mixed-in by each themed toolbar) ─────────────────────────

_LOCAL_ICON_NAMES = {
    "fit", "ortho", "rotate_left_90", "rotate_right_90",
    "capture_screen", "record_screen", "stop_screen",
}


class _ToolBarBase(QtWidgets.QToolBar):
    """Shared plumbing for every themed viewer toolbar.

    Subclasses inherit:

    * the access pattern to the active plotter / active pane;
    * the "All windows" toggle wired into the multi-view area;
    * helper factories for icon and text buttons.
    """

    def __init__(self, title: str, multi_view, session, parent=None):
        super().__init__(title, parent)
        self._multi_view = multi_view
        self._session = session
        self.setIconSize(QtCore.QSize(18, 18))
        self.setMovable(True)
        self.setFloatable(True)
        # Re-export setWindowTitle as the toolbar's user-visible name.
        self.setWindowTitle(title)

    # ── Pane / plotter access ───────────────────────────────────────────────

    @property
    def _plotter(self):
        return self._multi_view.active_plotter

    def _visible_panes(self):
        # Every entry in MultiViewArea._panes is on screen now that the
        # recursive split / close model owns the splitter tree.  Falling
        # back to ``visible_panes()`` keeps the proxy working through
        # ``TabbedMultiViewArea`` as well.
        target = getattr(self._multi_view, "current", None) or self._multi_view
        visible = getattr(target, "visible_panes", None)
        if callable(visible):
            try:
                return list(visible())
            except Exception:
                pass
        return list(getattr(target, "_panes", []))

    def _target_plotters(self, all_windows: bool):
        if all_windows:
            plotters = [getattr(pane, "plotter", None) for pane in self._visible_panes()]
        else:
            plotters = [self._plotter]
        unique = []
        seen = set()
        for p in plotters:
            if p is None or id(p) in seen:
                continue
            seen.add(id(p))
            unique.append(p)
        return unique

    def _target_panes(self, all_windows: bool):
        if all_windows:
            return self._visible_panes()
        pane = getattr(self._multi_view, "active_pane", None)
        return [pane] if pane is not None else []

    # ── Button factories ────────────────────────────────────────────────────

    def _btn(
        self,
        text: str,
        icon_name: str,
        tooltip: str,
        *,
        checkable: bool = False,
    ) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton(self)
        ic = (
            viewer_icon(icon_name, active_palette().navy, 18)
            if icon_name in _LOCAL_ICON_NAMES
            else _icon(icon_name, active_palette().navy)
        )
        if ic is not None:
            btn.setIcon(ic)
            btn.setIconSize(QtCore.QSize(18, 18))
            if text:
                btn.setText(text)
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            else:
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        else:
            _FALLBACK = {
                "mdi.rotate-3d-variant":       "⬡",
                "mdi.arrow-collapse-up":        "⬆",
                "mdi.arrow-collapse-down":      "⬇",
                "mdi.arrow-expand-up":          "↑",
                "mdi.arrow-expand-down":        "↓",
                "mdi.arrow-expand-left":        "←",
                "mdi.arrow-expand-right":       "→",
                "mdi.fit-to-page-outline":      "⌖",
                "mdi.grid":                     "⊞",
                "mdi.camera-outline":           "📷",
                "mdi.record-circle-outline":    "⏺",
                "mdi.axis-arrow":               "✛",
                "mdi.crop-square":              "▭",
            }
            btn.setText(text or _FALLBACK.get(icon_name, icon_name.split(".")[-1]))
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.setCheckable(checkable)
        btn.setFixedHeight(26)
        return btn

    def _sel_btn(self, label: str, icon_name: str, tooltip: str) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton(self)
        btn.setText(label)
        btn.setIcon(viewer_icon(icon_name, active_palette().navy, 16))
        btn.setIconSize(QtCore.QSize(16, 16))
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.setFixedHeight(26)
        return btn

    # Optional disposer; specific toolbars override if they hold resources.
    def dispose(self) -> None:
        return None


# ── Shared "All windows" coordination ────────────────────────────────────────

class _AllWindowsState:
    """Shared toggle propagated to every themed toolbar.

    The state lives in one object so the user-facing checkbox in the camera
    toolbar and the broadcast logic in the other toolbars all agree on
    whether actions target every pane or just the active one.
    """

    def __init__(self, multi_view):
        self._multi_view = multi_view
        self._enabled = False
        self._subscribers: list = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set(self, value: bool) -> None:
        value = bool(value)
        if value == self._enabled:
            return
        self._enabled = value
        setter = getattr(self._multi_view, "set_camera_apply_to_all", None)
        if callable(setter):
            try:
                setter(value)
            except Exception:
                pass
        for cb in self._subscribers:
            try:
                cb(value)
            except Exception:
                pass

    def subscribe(self, cb) -> None:
        self._subscribers.append(cb)


# ── Global "Apply to" toolbar ─────────────────────────────────────────────────


class ApplyTargetToolBar(_ToolBarBase):
    """Single "Apply to" selector consumed by every editor dock.

    Display, Vector Field, Color By, Warp and any future editor all read
    from this one combo to decide whether their Apply button writes to
    the global ``session.state`` or to the active pane / current tab's
    per-pane override layer.  Putting it in a toolbar (not inside any
    particular dock) makes the routing visible regardless of which dock
    the user has open.
    """

    targetChanged = QtCore.Signal(str)

    def __init__(self, multi_view, session, all_state: "_AllWindowsState", parent=None):
        super().__init__("Apply to", multi_view, session, parent)
        self._all_state = all_state
        label = QtWidgets.QLabel(" Apply to ")
        label.setToolTip(
            "Which viewports the Apply buttons write to.\n"
            "All panes — global change visible everywhere.\n"
            "Current tab — only panes in the visible tab.\n"
            "Active pane — only the pane you last clicked."
        )
        self.addWidget(label)

        self._combo = QtWidgets.QComboBox()
        # Order matters: the first entry is the visible default on launch.
        # "Active pane" is the safest starting point for a fresh session —
        # changes only land on the pane the user explicitly clicked.
        # Users can flip to Current tab / All panes for broader edits.
        self._combo.addItem("Active pane", "pane")
        self._combo.addItem("Current tab", "tab")
        self._combo.addItem("All panes", "all")
        self._combo.setMinimumWidth(140)
        self._combo.currentIndexChanged.connect(
            lambda *_: self.targetChanged.emit(self.target())
        )
        self.addWidget(self._combo)

    def target(self) -> str:
        try:
            return str(self._combo.currentData() or "all")
        except Exception:
            return "all"

    def set_target(self, value: str) -> None:
        idx = self._combo.findData(str(value))
        if idx < 0:
            return
        block = self._combo.blockSignals(True)
        try:
            self._combo.setCurrentIndex(idx)
        finally:
            self._combo.blockSignals(block)


# ── View presets toolbar (replaces the old vertical CameraViewRail) ──────────

#: ``(group, key, label, icon_name, tooltip)`` — drives both the toolbar
#: button layout and the camera-preset callback inside MultiViewArea.
_VIEW_PRESET_ENTRIES: list[tuple[str, str, str, str, str]] = [
    ("ISO",   "iso_ne", "NE", "view_iso_ne", "Isometric NE"),
    ("ISO",   "iso_nw", "NW", "view_iso_nw", "Isometric NW"),
    ("ISO",   "iso_sw", "SW", "view_iso_sw", "Isometric SW"),
    ("ISO",   "iso_se", "SE", "view_iso_se", "Isometric SE"),
    ("ORTHO", "top",    "Top",   "view_top",    "Top view (+Z)"),
    ("ORTHO", "bottom", "Bot",   "view_bottom", "Bottom view (-Z)"),
    ("ORTHO", "front",  "Front", "view_front",  "Front view (-Y)"),
    ("ORTHO", "back",   "Back",  "view_back",   "Back view (+Y)"),
    ("ORTHO", "left",   "Left",  "view_left",   "Left view (-X)"),
    ("ORTHO", "right",  "Right", "view_right",  "Right view (+X)"),
]


class ViewPresetsToolBar(_ToolBarBase):
    """Camera-preset buttons + the GF 3×3 toggle.

    Each preset button calls :class:`~.multi_view.MultiViewArea`
    ``_apply_camera_preset`` on the current tab.  The trailing GF button
    flips :class:`~.multi_view.MultiViewArea.toggle_gf_mode` so the user
    can enter or leave the tensor view from the same toolbar that holds
    the orthographic / iso presets.
    """

    def __init__(self, multi_view, session, all_state: "_AllWindowsState", parent=None):
        super().__init__("View Presets", multi_view, session, parent)
        self._all_state = all_state
        self._gf_button: QtWidgets.QToolButton | None = None

        previous_group: str | None = None
        for group, key, label, icon_name, tooltip in _VIEW_PRESET_ENTRIES:
            if previous_group is not None and group != previous_group:
                self.addSeparator()
            previous_group = group
            btn = QtWidgets.QToolButton(self)
            btn.setText(label)
            btn.setIcon(viewer_icon(icon_name, active_palette().navy, 16))
            btn.setIconSize(QtCore.QSize(16, 16))
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            btn.setToolTip(tooltip)
            btn.setAutoRaise(True)
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda _c=False, k=key: self._apply_preset(k))
            self.addWidget(btn)

        # ── GF tensor view (3×3 grid + auto-warmup + component pins) ────────
        self.addSeparator()
        self._gf_button = QtWidgets.QToolButton(self)
        self._gf_button.setText("GF")
        self._gf_button.setToolTip(
            "Toggle the 3×3 Green Function tensor view.\n"
            "Replaces the current panes with a 9-pane grid pinned to G_ij\n"
            "and switches the active demand to GF.  Press again to restore\n"
            "the previous demand and a single pane."
        )
        self._gf_button.setCheckable(True)
        self._gf_button.setAutoRaise(True)
        self._gf_button.setFixedHeight(26)
        # GF needs a touch more visual weight than a plain preset — use the
        # accent color so it reads as "special action" instead of "axis".
        palette = active_palette()
        self._gf_button.setStyleSheet(
            f"QToolButton {{ color: {palette.accent_dark}; font-weight: bold; "
            f"padding: 0 8px; }}"
            f"QToolButton:checked {{ background: {palette.surface_3}; "
            f"border: 1px solid {palette.accent}; border-radius: 3px; }}"
        )
        self._gf_button.toggled.connect(self._on_gf_toggled)
        self.addWidget(self._gf_button)

    def _apply_preset(self, key: str) -> None:
        # Prefer the per-tab MultiViewArea entry point so the "All windows"
        # state and the GF-layout pin logic still flow through one path.
        c = getattr(self._multi_view, "current", None)
        target = c if c is not None else self._multi_view
        apply = getattr(target, "_apply_camera_preset", None)
        if callable(apply):
            try:
                # Sync the "apply to all" flag before firing so a single click
                # mirrors the user-facing checkbox state.
                set_all = getattr(target, "set_camera_apply_to_all", None)
                if callable(set_all):
                    set_all(self._all_state.enabled)
                apply(key)
                return
            except Exception:
                pass
        # Fallback: drive every active plotter ourselves.
        for p in self._target_plotters(self._all_state.enabled):
            try:
                if key.startswith("iso_"):
                    p.view_isometric()
                    az = {"iso_ne": 0, "iso_nw": 90, "iso_sw": 180, "iso_se": 270}.get(key, 0)
                    if az:
                        p.camera.Azimuth(az)
                        p.reset_camera_clipping_range()
                elif key == "top":
                    p.view_xy()
                elif key == "bottom":
                    p.view_xy(negative=True)
                elif key == "front":
                    p.view_xz()
                elif key == "back":
                    p.view_xz(negative=True)
                elif key == "left":
                    p.view_yz(negative=True)
                elif key == "right":
                    p.view_yz()
                p.render()
            except Exception:
                pass

    def _on_gf_toggled(self, checked: bool) -> None:
        """Forward the checkbox state to the current MultiViewArea."""
        c = getattr(self._multi_view, "current", None)
        target = c if c is not None else self._multi_view
        try:
            if checked:
                if not getattr(target, "in_gf_mode", False):
                    enter = getattr(target, "enter_gf_mode", None)
                    if callable(enter):
                        enter()
            else:
                if getattr(target, "in_gf_mode", False):
                    leave = getattr(target, "exit_gf_mode", None)
                    if callable(leave):
                        leave()
        except Exception:
            # Roll back the button if the action failed so visual state stays honest.
            block = self._gf_button.blockSignals(True)
            self._gf_button.setChecked(getattr(target, "in_gf_mode", False))
            self._gf_button.blockSignals(block)


# ── Camera toolbar (presets + ortho/rotate + All windows toggle) ─────────────

class CameraToolBar(_ToolBarBase):
    """Camera fit / projection / rotate buttons + ``All windows`` checkbox."""

    def __init__(self, multi_view, session, all_state: _AllWindowsState, parent=None):
        super().__init__("Camera", multi_view, session, parent)
        self._all_state = all_state

        # All windows toggle — first item, communicates scope of every action.
        self._all_chk = QtWidgets.QCheckBox("All windows")
        self._all_chk.setToolTip("Apply camera / view actions to every visible pane")
        self._all_chk.setChecked(all_state.enabled)
        self._all_chk.toggled.connect(all_state.set)
        all_state.subscribe(lambda v: self._all_chk.setChecked(v))
        self.addWidget(self._all_chk)
        self.addSeparator()

        fit_btn = self._btn("Fit", "fit", "Fit all — reset camera to show full model")
        fit_btn.clicked.connect(self._fit_all)
        self.addWidget(fit_btn)

        self._ortho_btn = self._btn(
            "Ortho", "ortho",
            "Toggle orthographic / perspective projection",
            checkable=True,
        )
        self._ortho_btn.toggled.connect(self._toggle_ortho)
        self.addWidget(self._ortho_btn)

        self.addSeparator()
        rot_minus_btn = self._btn("-90", "rotate_left_90", "Rotate active view -90 around Z")
        rot_minus_btn.clicked.connect(lambda: self._rotate_camera(-90))
        self.addWidget(rot_minus_btn)
        rot_plus_btn = self._btn("+90", "rotate_right_90", "Rotate active view +90 around Z")
        rot_plus_btn.clicked.connect(lambda: self._rotate_camera(90))
        self.addWidget(rot_plus_btn)

    # ── Internals ───────────────────────────────────────────────────────────

    def _fit_all(self):
        for p in self._target_plotters(self._all_state.enabled):
            try:
                p.reset_camera()
                p.render()
            except Exception:
                pass

    def _toggle_ortho(self, checked: bool):
        for p in self._target_plotters(self._all_state.enabled):
            try:
                if checked:
                    p.enable_parallel_projection()
                else:
                    p.disable_parallel_projection()
                p.render()
            except Exception:
                pass

    def _rotate_camera(self, degrees: float):
        for p in self._target_plotters(self._all_state.enabled):
            camera = getattr(p, "camera", None)
            if camera is None:
                continue
            try:
                angle = math.radians(float(degrees))
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                focal = camera.GetFocalPoint()
                position = camera.GetPosition()
                view_up = camera.GetViewUp()

                rel = [position[i] - focal[i] for i in range(3)]
                rel_rot = (
                    cos_a * rel[0] - sin_a * rel[1],
                    sin_a * rel[0] + cos_a * rel[1],
                    rel[2],
                )
                up_rot = (
                    cos_a * view_up[0] - sin_a * view_up[1],
                    sin_a * view_up[0] + cos_a * view_up[1],
                    view_up[2],
                )

                camera.SetPosition(
                    focal[0] + rel_rot[0],
                    focal[1] + rel_rot[1],
                    focal[2] + rel_rot[2],
                )
                camera.SetViewUp(*up_rot)
                camera.OrthogonalizeViewUp()
                p.reset_camera_clipping_range()
                p.render()
            except Exception:
                pass


# ── Overlays toolbar (axes, bbox, stations) ──────────────────────────────────

class OverlaysToolBar(_ToolBarBase):
    """Toggle visibility of helper overlays (orientation axes, bbox, stations)."""

    def __init__(self, multi_view, session, all_state: _AllWindowsState, parent=None):
        super().__init__("Overlays", multi_view, session, parent)
        self._all_state = all_state
        self._show_bbox = False
        self._show_axes = True

        self._stations_btn = self._btn(
            "Stations",
            "mdi.map-marker-multiple-outline",
            "Toggle station tags visibility",
            checkable=True,
        )
        self._stations_btn.setChecked(True)
        self._stations_btn.toggled.connect(self._toggle_stations)
        self.addWidget(self._stations_btn)
        self._multi_view.on_active_pane_changed = self._sync_from_active_pane

        self._axes_btn = self._btn(
            "Axes", "mdi.axis-arrow",
            "Toggle orientation axes widget",
            checkable=True,
        )
        self._axes_btn.setChecked(True)
        self._axes_btn.toggled.connect(self._toggle_axes)
        self.addWidget(self._axes_btn)

        self._bbox_btn = self._btn(
            "BBox", "mdi.crop-square",
            "Toggle bounding box",
            checkable=True,
        )
        self._bbox_btn.toggled.connect(self._toggle_bbox)
        self.addWidget(self._bbox_btn)

        self._sync_from_active_pane(self._multi_view.active_pane)

    def _toggle_axes(self, checked: bool):
        self._show_axes = checked
        for p in self._target_plotters(self._all_state.enabled):
            try:
                if checked:
                    p.show_axes()
                else:
                    p.hide_axes()
                p.render()
            except Exception:
                pass

    def _toggle_bbox(self, checked: bool):
        self._show_bbox = checked
        for p in self._target_plotters(self._all_state.enabled):
            try:
                if checked:
                    p.add_bounding_box()
                else:
                    p.remove_bounding_box()
                p.render()
            except Exception:
                pass

    def _toggle_stations(self, checked: bool):
        panes = self._target_panes(self._all_state.enabled)
        if not panes:
            return
        for pane in panes:
            if pane is None:
                continue
            try:
                pane.set_station_tags_visible(bool(checked))
            except Exception:
                pass

    def _sync_from_active_pane(self, pane=None):
        if pane is None:
            pane = getattr(self._multi_view, "active_pane", None)
        checked = True if pane is None else pane.station_tags_visible()
        self._stations_btn.blockSignals(True)
        self._stations_btn.setChecked(bool(checked))
        self._stations_btn.blockSignals(False)


# ── Selection toolbar (show all / show selected / hide selected) ─────────────

class SelectionToolBar(_ToolBarBase):
    """Selection-filter buttons (delegates to MultiViewArea.apply_selection_filter)."""

    def __init__(self, multi_view, session, all_state: _AllWindowsState, parent=None):
        super().__init__("Selection", multi_view, session, parent)
        self._all_state = all_state

        all_btn = self._sel_btn(
            "Show All", "sel_all", "Show all visible nodes (reset filter)"
        )
        all_btn.clicked.connect(lambda: self._apply("all"))
        self.addWidget(all_btn)

        only_btn = self._sel_btn(
            "Show Sel.", "sel_show", "Show only selected nodes"
        )
        only_btn.clicked.connect(lambda: self._apply("only"))
        self.addWidget(only_btn)

        hide_btn = self._sel_btn(
            "Hide Sel.", "sel_hide", "Hide selected nodes"
        )
        hide_btn.clicked.connect(lambda: self._apply("hide"))
        self.addWidget(hide_btn)

        reset_btn = self._sel_btn(
            "Reset", "sel_cursor", "Reset selection filter"
        )
        reset_btn.clicked.connect(lambda: self._apply("all"))
        self.addWidget(reset_btn)

    def _apply(self, mode: str):
        fn = getattr(self._multi_view, "apply_selection_filter", None)
        if callable(fn):
            try:
                fn(mode, apply_to_all=self._all_state.enabled)
            except Exception:
                pass


# ── Capture toolbar (screenshot + record) ────────────────────────────────────

class CaptureToolBar(_ToolBarBase):
    """Capture buttons + recording state.

    The :class:`ViewerMainWindow` calls :meth:`write_frame_if_recording` after
    every playback step so the recorded MP4 / GIF stays in sync with the
    animation.
    """

    def __init__(self, multi_view, session, all_state: _AllWindowsState, parent=None):
        super().__init__("Capture", multi_view, session, parent)
        self._all_state = all_state
        self._recording = False
        self._recording_writer = None

        shot_btn = self._btn("", "capture_screen", "Save full viewer window as PNG")
        shot_btn.clicked.connect(self._screenshot)
        self.addWidget(shot_btn)

        self._record_btn = self._btn(
            "Rec", "record_screen",
            "Record full viewer window to MP4 / GIF",
            checkable=True,
        )
        self._record_btn.toggled.connect(self._toggle_record)
        self.addWidget(self._record_btn)

    # ── Recording lifecycle ─────────────────────────────────────────────────

    def _screenshot(self):
        widget = self._capture_widget()
        if widget is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        try:
            self._wait_before_screen_capture()
            pixmap = self._grab_results_pixmap()
            if pixmap is None or pixmap.isNull() or not pixmap.save(path, "PNG"):
                raise RuntimeError("Could not capture the viewer window.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Screenshot failed", str(exc))

    def _toggle_record(self, checked: bool):
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        widget = self._capture_widget()
        if widget is None:
            self._uncheck_record()
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Recording", "animation.mp4",
            "MP4 video (*.mp4);;GIF animation (*.gif);;All files (*)",
        )
        if not path:
            self._uncheck_record()
            return

        try:
            if FFMPEG_EXE:
                os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE
            import imageio.v2 as imageio
            ext = os.path.splitext(path)[1].lower()
            if ext == ".gif":
                self._recording_writer = imageio.get_writer(path, fps=12)
            else:
                self._recording_writer = imageio.get_writer(path, fps=12, quality=5)
            self._recording = True
            self._wait_before_screen_capture()
            self._write_widget_frame()
            self._record_btn.setIcon(viewer_icon("stop_screen", "#cc2020", 18))
            self._record_btn.setToolTip("Stop recording  ● REC")
        except Exception as exc:
            self._close_recording_writer()
            self._recording = False
            self._record_btn.setIcon(viewer_icon("record_screen", active_palette().navy, 18))
            self._uncheck_record()
            QtWidgets.QMessageBox.warning(
                self, "Recording failed",
                f"{exc}\n\nffmpeg may not be installed. "
                "Run:  pip install imageio imageio-ffmpeg",
            )

    def _stop_recording(self):
        if self._recording:
            self._close_recording_writer()
        self._recording = False
        self._record_btn.setIcon(viewer_icon("record_screen", active_palette().navy, 18))
        self._record_btn.setToolTip("Record full viewer window to MP4 / GIF")

    def _uncheck_record(self):
        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(False)
        self._record_btn.blockSignals(False)

    def write_frame_if_recording(self):
        if not self._recording or self._recording_writer is None:
            return
        try:
            self._write_widget_frame()
        except Exception:
            self._stop_recording()
            self._uncheck_record()

    def _capture_widget(self):
        return self.window()

    def _grab_results_pixmap(self):
        widget = self._capture_widget()
        if widget is None:
            return None
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 1)
        try:
            window = widget.window().windowHandle()
            screen = window.screen() if window is not None else QtGui.QGuiApplication.primaryScreen()
            pos = widget.mapToGlobal(QtCore.QPoint(0, 0))
            return screen.grabWindow(0, pos.x(), pos.y(), widget.width(), widget.height())
        except Exception:
            return widget.grab()

    def _wait_before_screen_capture(self):
        deadline = QtCore.QTime.currentTime().addMSecs(1000)
        while QtCore.QTime.currentTime() < deadline:
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.AllEvents,
                50,
            )
            QtCore.QThread.msleep(25)

    def _write_widget_frame(self):
        if self._recording_writer is None:
            return
        pixmap = self._grab_results_pixmap()
        if pixmap is None or pixmap.isNull():
            return
        self._recording_writer.append_data(self._pixmap_to_rgb_array(pixmap))

    def _pixmap_to_rgb_array(self, pixmap):
        image_format = getattr(QtGui.QImage, "Format_RGBA8888", None)
        if image_format is None:
            image_format = QtGui.QImage.Format.Format_RGBA8888
        image = pixmap.toImage().convertToFormat(image_format)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        size = image.sizeInBytes() if hasattr(image, "sizeInBytes") else image.byteCount()
        try:
            ptr.setsize(size)
        except AttributeError:
            ptr = bytes(ptr)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        return arr[:, :, :3].copy()

    def _close_recording_writer(self):
        writer = self._recording_writer
        self._recording_writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def dispose(self) -> None:
        try:
            self._stop_recording()
        except Exception:
            pass


# ── Display toolbar (opacity slider) ─────────────────────────────────────────

class DisplayToolBar(_ToolBarBase):
    """Node opacity slider."""

    def __init__(self, multi_view, session, all_state: _AllWindowsState, parent=None):
        super().__init__("Display", multi_view, session, parent)
        self._all_state = all_state

        label = QtWidgets.QLabel("Opacity")
        label.setToolTip("Uniform opacity for all rendered nodes (0 = transparent, 1 = opaque)")
        self.addWidget(label)

        self._opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setFixedWidth(120)
        self._opacity_slider.setToolTip("Node opacity")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.addWidget(self._opacity_slider)

    def _on_opacity_changed(self, value: int):
        self._session.set_node_opacity(value / 100.0)


# ── Public factory ───────────────────────────────────────────────────────────

def build_viewer_toolbars(multi_view, session, parent) -> list[_ToolBarBase]:
    """Construct every themed toolbar in the order the window expects.

    The "Apply to" selector comes first so it is always the leftmost
    control — every editor dock (Display / Vector Field / Color By /
    Warp) reads from it.  The window groups the first three in row 1
    (Apply-to + Camera + View Presets) and the rest wrap to row 2.
    """
    state = _AllWindowsState(multi_view)
    return [
        ApplyTargetToolBar(multi_view, session, state, parent),
        CameraToolBar(multi_view, session, state, parent),
        ViewPresetsToolBar(multi_view, session, state, parent),
        OverlaysToolBar(multi_view, session, state, parent),
        SelectionToolBar(multi_view, session, state, parent),
        CaptureToolBar(multi_view, session, state, parent),
        DisplayToolBar(multi_view, session, state, parent),
    ]


__all__ = [
    "ApplyTargetToolBar",
    "CameraToolBar",
    "ViewPresetsToolBar",
    "OverlaysToolBar",
    "SelectionToolBar",
    "CaptureToolBar",
    "DisplayToolBar",
    "build_viewer_toolbars",
]
