"""Top toolbar for the interactive viewer — view presets, capture and overlays."""

from __future__ import annotations

import math
import os
import numpy as np

from ._imports import require_viewer_dependencies
from .icons import icon as viewer_icon
from .theme import LIGHT_PALETTE

_, _, _, QtCore, _, QtWidgets = require_viewer_dependencies()

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


# (label, tooltip, azimuth_offset)
_ISO_PRESETS = [
    ("NE", "Isometric — North-East",  0),
    ("NW", "Isometric — North-West", 90),
    ("SW", "Isometric — South-West", 180),
    ("SE", "Isometric — South-East", 270),
]

# (label, tooltip, icon_name, plotter_method, extra_kwargs)
_VIEW_PRESETS = [
    ("Top",   "Top view (+Z)",    "mdi.arrow-collapse-up",   "view_xy",  {}),
    ("Bot",   "Bottom view (−Z)", "mdi.arrow-collapse-down", "view_xy",  {"negative": True}),
    ("Front", "Front view (−Y)",  "mdi.arrow-expand-up",     "view_xz",  {}),
    ("Back",  "Back view (+Y)",   "mdi.arrow-expand-down",   "view_xz",  {"negative": True}),
    ("Left",  "Left view (−X)",   "mdi.arrow-expand-left",   "view_yz",  {"negative": True}),
    ("Right", "Right view (+X)",  "mdi.arrow-expand-right",  "view_yz",  {}),
]


class ViewerToolBar(QtWidgets.QWidget):
    """Horizontal icon toolbar placed below the header bar.

    Parameters
    ----------
    multi_view:
        The :class:`~.multi_view.MultiViewArea` that owns all view panes.
        View-preset and capture operations always target its *active* pane,
        so the toolbar naturally follows whichever pane the user last clicked.
    """

    def __init__(self, multi_view, session, parent=None):
        super().__init__(parent)
        self._multi_view = multi_view
        self._session = session
        self._recording = False
        self._recording_plotters = []    # captured at open_movie() call time
        self._show_bbox = False
        self._show_axes = True

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(1)

        self._all_windows_chk = QtWidgets.QCheckBox("All windows")
        self._all_windows_chk.setToolTip("Apply toolbar actions to all visible panes")
        self._all_windows_chk.setChecked(False)
        self._all_windows_chk.toggled.connect(self._sync_apply_to_all)
        layout.addWidget(self._all_windows_chk)

        # ── Camera ───────────────────────────────────────────────────────────
        self._add_sep(layout)

        fit_btn = self._btn(
            "Fit", "fit", "Fit all — reset camera to show full model"
        )
        fit_btn.clicked.connect(self._fit_all)
        layout.addWidget(fit_btn)
        self._ortho_btn = self._btn(
            "Ortho", "ortho",
            "Toggle orthographic / perspective projection",
            checkable=True,
        )
        self._ortho_btn.toggled.connect(self._toggle_ortho)
        layout.addWidget(self._ortho_btn)
        self._add_sep(layout)
        rot_minus_btn = self._btn("-90", "rotate_left_90", "Rotate active view -90 around Z")
        rot_minus_btn.clicked.connect(lambda: self._rotate_active_camera(-90))
        layout.addWidget(rot_minus_btn)
        rot_plus_btn = self._btn("+90", "rotate_right_90", "Rotate active view +90 around Z")
        rot_plus_btn.clicked.connect(lambda: self._rotate_active_camera(90))
        layout.addWidget(rot_plus_btn)
        self._add_sep(layout)
        self._stations_btn = self._btn(
            "Stations",
            "mdi.map-marker-multiple-outline",
            "Toggle station tags visibility",
            checkable=True,
        )
        self._stations_btn.setChecked(True)
        self._stations_btn.toggled.connect(self._toggle_stations)
        layout.addWidget(self._stations_btn)
        self._multi_view.on_active_pane_changed = self._sync_from_active_pane

        # ── Capture ───────────────────────────────────────────────────────────
        self._add_sep(layout)

        shot_btn = self._btn("", "mdi.camera-outline", "Save screenshot as PNG")
        shot_btn.clicked.connect(self._screenshot)
        layout.addWidget(shot_btn)

        self._record_btn = self._btn(
            "", "mdi.record-circle-outline",
            "Record animation to MP4 / GIF",
            checkable=True,
        )
        self._record_btn.toggled.connect(self._toggle_record)
        layout.addWidget(self._record_btn)

        # ── Overlays ─────────────────────────────────────────────────────────
        self._add_sep(layout)

        self._axes_btn = self._btn(
            "Axes", "mdi.axis-arrow",
            "Toggle orientation axes widget",
            checkable=True,
        )
        self._axes_btn.setChecked(True)
        self._axes_btn.toggled.connect(self._toggle_axes)
        layout.addWidget(self._axes_btn)

        self._bbox_btn = self._btn(
            "BBox", "mdi.crop-square",
            "Toggle bounding box",
            checkable=True,
        )
        self._bbox_btn.toggled.connect(self._toggle_bbox)
        layout.addWidget(self._bbox_btn)

        # ── Selection filters ─────────────────────────────────────────────────
        self._add_sep(layout)

        self._sel_all_btn = self._sel_btn(
            "Show All", "sel_all", "Show all visible nodes (reset filter)"
        )
        self._sel_all_btn.clicked.connect(
            lambda: self._apply_selection_filter("all")
        )
        layout.addWidget(self._sel_all_btn)

        self._sel_only_btn = self._sel_btn(
            "Show Sel.", "sel_show", "Show only selected nodes"
        )
        self._sel_only_btn.clicked.connect(
            lambda: self._apply_selection_filter("only")
        )
        layout.addWidget(self._sel_only_btn)

        self._sel_hide_btn = self._sel_btn(
            "Hide Sel.", "sel_hide", "Hide selected nodes"
        )
        self._sel_hide_btn.clicked.connect(
            lambda: self._apply_selection_filter("hide")
        )
        layout.addWidget(self._sel_hide_btn)

        self._sel_prev_btn = self._sel_btn(
            "Reset", "sel_cursor", "Reset selection filter"
        )
        self._sel_prev_btn.clicked.connect(
            lambda: self._apply_selection_filter("all")
        )
        layout.addWidget(self._sel_prev_btn)

        # ── Node opacity ──────────────────────────────────────────────────────
        self._add_sep(layout)

        opacity_lbl = QtWidgets.QLabel("Opacity")
        opacity_lbl.setToolTip("Uniform opacity for all rendered nodes (0 = transparent, 1 = opaque)")
        layout.addWidget(opacity_lbl)

        self._opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setFixedWidth(80)
        self._opacity_slider.setToolTip("Node opacity")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self._opacity_slider)

        layout.addStretch(1)
        self._add_sep(layout)
        self._transform_widget = self._build_transform_widget()
        layout.addWidget(self._transform_widget)

        self.setStyleSheet("ViewerToolBar { border-bottom: 1px solid #d0d0d0; }")
        self._sync_from_active_pane(self._multi_view.active_pane)
        self._sync_apply_to_all(self._all_windows_chk.isChecked())
        self._load_transform_from_session()

    # ── Active-plotter access ─────────────────────────────────────────────────

    @property
    def _plotter(self):
        """Active pane's plotter — re-evaluated on every access."""
        return self._multi_view.active_plotter

    def _visible_panes(self):
        panes = list(getattr(self._multi_view, "_panes", []))
        current_layout = getattr(self._multi_view, "_current_layout", "1×1")
        layout_n = getattr(self._multi_view, "_layout_n", {})
        needed = int(layout_n.get(current_layout, len(panes) or 1))
        return panes[: max(0, min(needed, len(panes)))]

    def _target_plotters(self):
        if self._all_windows_chk.isChecked():
            plotters = [getattr(pane, "plotter", None) for pane in self._visible_panes()]
        else:
            plotters = [self._plotter]
        unique = []
        seen = set()
        for p in plotters:
            if p is None:
                continue
            key = id(p)
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return unique

    def _target_panes(self):
        if self._all_windows_chk.isChecked():
            return self._visible_panes()
        pane = getattr(self._multi_view, "active_pane", None)
        return [pane] if pane is not None else []

    def _sync_apply_to_all(self, checked: bool) -> None:
        setter = getattr(self._multi_view, "set_camera_apply_to_all", None)
        if callable(setter):
            setter(bool(checked))

    # ── View presets ──────────────────────────────────────────────────────────

    def _view_cb(self, method_name: str, kwargs: dict):
        """Return a callback that calls *method_name* on the active plotter."""
        def _cb():
            plotters = self._target_plotters()
            if not plotters:
                return
            for p in plotters:
                fn = getattr(p, method_name, None)
                if fn is None:
                    continue
                try:
                    fn(**kwargs)
                    p.render()
                except Exception:
                    pass
        return _cb

    def _iso_view_cb(self, azimuth_extra: float):
        """Return a callback for an isometric view rotated by *azimuth_extra*°."""
        def _cb():
            plotters = self._target_plotters()
            if not plotters:
                return
            for p in plotters:
                try:
                    p.view_isometric()
                    if azimuth_extra:
                        p.camera.Azimuth(azimuth_extra)
                        p.reset_camera_clipping_range()
                    p.render()
                except Exception:
                    pass
        return _cb

    def _fit_all(self):
        plotters = self._target_plotters()
        if not plotters:
            return
        for p in plotters:
            try:
                p.reset_camera()
                p.render()
            except Exception:
                pass

    # ── Camera ────────────────────────────────────────────────────────────────

    def _rotate_active_camera(self, degrees: float):
        plotters = self._target_plotters()
        if not plotters:
            return
        for p in plotters:
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

    def _toggle_ortho(self, checked: bool):
        plotters = self._target_plotters()
        if not plotters:
            return
        for p in plotters:
            try:
                if checked:
                    p.enable_parallel_projection()
                else:
                    p.disable_parallel_projection()
                p.render()
            except Exception:
                pass

    # ── Capture ───────────────────────────────────────────────────────────────

    def _screenshot(self):
        plotters = self._target_plotters()
        if not plotters:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        try:
            if len(plotters) == 1:
                plotters[0].screenshot(path, transparent_background=False)
                return

            base, ext = os.path.splitext(path)
            ext = ext or ".png"
            for idx, p in enumerate(plotters, start=1):
                out_path = f"{base}__pane{idx}{ext}"
                p.screenshot(out_path, transparent_background=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Screenshot failed", str(exc))

    def _toggle_record(self, checked: bool):
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        plotters = self._target_plotters()
        if not plotters:
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
            if len(plotters) == 1:
                targets = [(plotters[0], path)]
            else:
                base, ext = os.path.splitext(path)
                ext = ext or ".mp4"
                targets = [
                    (p, f"{base}__pane{idx}{ext}")
                    for idx, p in enumerate(plotters, start=1)
                ]

            opened = []
            for p, out_path in targets:
                p.open_movie(out_path, framerate=12, quality=5)
                opened.append(p)

            self._recording_plotters = opened
            self._recording = True
            if _HAS_QTA:
                self._record_btn.setIcon(
                    qta.icon("mdi.stop-circle-outline", color="#cc2020")
                )
            self._record_btn.setToolTip("Stop recording  ● REC")
        except Exception as exc:
            for p in list(getattr(self, "_recording_plotters", [])):
                try:
                    p.close_movie()
                except Exception:
                    pass
            self._recording_plotters = []
            self._uncheck_record()
            QtWidgets.QMessageBox.warning(
                self, "Recording failed",
                f"{exc}\n\nffmpeg may not be installed. "
                "Run:  pip install imageio-ffmpeg",
            )

    def _stop_recording(self):
        if self._recording and self._recording_plotters:
            for p in list(self._recording_plotters):
                try:
                    p.close_movie()
                except Exception:
                    pass
        self._recording = False
        self._recording_plotters = []
        if _HAS_QTA:
            self._record_btn.setIcon(
                qta.icon("mdi.record-circle-outline", color="#404040")
            )
        self._record_btn.setToolTip("Record animation to MP4 / GIF")

    def _uncheck_record(self):
        """Silently uncheck the record button (user cancelled or error)."""
        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(False)
        self._record_btn.blockSignals(False)

    def write_frame_if_recording(self):
        """Write the current render as one video frame after each playback step."""
        if not self._recording or not self._recording_plotters:
            return
        try:
            for p in list(self._recording_plotters):
                p.write_frame()
        except Exception:
            # Writer failed mid-recording — stop gracefully.
            self._stop_recording()
            self._uncheck_record()

    # ── Overlays ──────────────────────────────────────────────────────────────

    def dispose(self) -> None:
        """Release toolbar runtime state before the viewer closes."""
        try:
            self._stop_recording()
        except Exception:
            pass
        self._multi_view = None

    def _toggle_axes(self, checked: bool):
        plotters = self._target_plotters()
        if not plotters:
            return
        self._show_axes = checked
        for p in plotters:
            try:
                if checked:
                    p.show_axes()
                else:
                    p.hide_axes()
                p.render()
            except Exception:
                pass

    def _toggle_bbox(self, checked: bool):
        plotters = self._target_plotters()
        if not plotters:
            return
        self._show_bbox = checked
        for p in plotters:
            try:
                if checked:
                    p.add_bounding_box()
                else:
                    p.remove_bounding_box()
                p.render()
            except Exception:
                pass

    def _toggle_stations(self, checked: bool):
        panes = self._target_panes()
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_transform_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("Transform")
        title.setStyleSheet(
            f"font-size: 11px; color: {LIGHT_PALETTE.navy}; font-weight: 600;"
        )
        layout.addWidget(title)

        self._transform_spins: dict[str, list[QtWidgets.QDoubleSpinBox]] = {}
        for axis in ("X", "Y", "Z"):
            label = QtWidgets.QLabel(axis)
            label.setStyleSheet(f"font-size: 11px; color: {LIGHT_PALETTE.navy};")
            layout.addWidget(label)
            row_spins: list[QtWidgets.QDoubleSpinBox] = []
            for _ in range(3):
                spin = self._matrix_spin()
                spin.valueChanged.connect(self._mark_transform_dirty)
                layout.addWidget(spin)
                row_spins.append(spin)
            self._transform_spins[axis] = row_spins
            if axis != "Z":
                layout.addSpacing(6)

        self._transform_apply_btn = QtWidgets.QPushButton("Apply")
        self._transform_apply_btn.setFixedHeight(24)
        self._transform_apply_btn.setEnabled(False)
        self._transform_apply_btn.clicked.connect(self._apply_display_transform)
        layout.addWidget(self._transform_apply_btn)
        # TODO: Keep this editor bound to the single global viewer transform state during future geometry debugging.
        return widget

    def _matrix_spin(self) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(0)
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setSingleStep(1.0)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spin.setFixedWidth(50)
        spin.setFixedHeight(24)
        return spin

    def _load_transform_from_session(self) -> None:
        matrix = np.asarray(self._multi_view.session.current_display_transform(), dtype=float)
        for row_idx, axis in enumerate(("X", "Y", "Z")):
            for col_idx, spin in enumerate(self._transform_spins[axis]):
                blocked = spin.blockSignals(True)
                spin.setValue(float(matrix[row_idx, col_idx]))
                spin.blockSignals(blocked)
        self._transform_apply_btn.setEnabled(False)

    def _mark_transform_dirty(self, *_args) -> None:
        self._transform_apply_btn.setEnabled(True)

    def _apply_display_transform(self) -> None:
        matrix = np.array(
            [
                [int(spin.value()) for spin in self._transform_spins["X"]],
                [int(spin.value()) for spin in self._transform_spins["Y"]],
                [int(spin.value()) for spin in self._transform_spins["Z"]],
            ],
            dtype=float,
        )
        try:
            self._multi_view.session.apply_display_transform(matrix)
            self._load_transform_from_session()
        except Exception:
            pass

    # ── Selection helpers ─────────────────────────────────────────────────────

    def _on_opacity_changed(self, value: int):
        self._session.set_node_opacity(value / 100.0)

    def _apply_selection_filter(self, mode: str):
        apply_fn = getattr(self._multi_view, "apply_selection_filter", None)
        if callable(apply_fn):
            apply_fn(mode, apply_to_all=self._all_windows_chk.isChecked())

    def _sel_btn(self, label: str, icon_name: str, tooltip: str) -> QtWidgets.QToolButton:
        """Create a small toolbar button using a viewer SVG icon."""
        btn = QtWidgets.QToolButton()
        btn.setText(label)
        btn.setIcon(viewer_icon(icon_name, LIGHT_PALETTE.navy, 16))
        btn.setIconSize(QtCore.QSize(16, 16))
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.setFixedHeight(26)
        return btn

    def _btn(
        self,
        text: str,
        icon_name: str,
        tooltip: str,
        *,
        checkable: bool = False,
    ) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        local_icon_names = {"fit", "ortho", "rotate_left_90", "rotate_right_90"}
        ic = (
            viewer_icon(icon_name, LIGHT_PALETTE.navy, 18)
            if icon_name in local_icon_names
            else _icon(icon_name)
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

    @staticmethod
    def _add_sep(layout: QtWidgets.QHBoxLayout):
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep.setFixedWidth(8)
        layout.addWidget(sep)
