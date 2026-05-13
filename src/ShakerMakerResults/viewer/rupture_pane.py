"""Dedicated tab + FFSP controls for one rupture source.

The rupture pane is intentionally a self-contained widget that:

* Hosts its own :class:`pyvistaqt.QtInteractor` — it does NOT share scenes,
  actors or pane state with the seismic ``ViewerScene`` so the seismic
  flow stays untouched.
* Renders the subfault cloud and recolours it on every ``time`` /
  ``playback`` broadcast received via :meth:`on_session_updated`.
* Embeds a Display-style control panel on its right edge (Field, Color
  map, Rupture front, Hypocenter) so the user does not need a separate
  dock to drive the visualisation.

Added to :class:`~.multi_view.TabbedMultiViewArea` as a plain ``QWidget``
tab; the tab broadcaster delivers session updates to it without any of
the per-pane plumbing that the seismic :class:`ViewPane` uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ._imports import require_viewer_dependencies
from .cmap_preview import ColormapPreview
from .colors import COLORMAP_OPTIONS, effective_colormap_name
from .rupture_adapter import RuptureSource

pv, QtInteractor, vtk, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


# (key, label, unit, animated?)
_FIELD_OPTIONS: tuple[tuple[str, str, str, bool], ...] = (
    ("slip_inst",    "Slip (instant)",  "m", True),
    ("slip",         "Slip (final)",    "m", False),
    ("rupture_time", "Rupture time",    "s", False),
    ("rise_time",    "Rise time",       "s", False),
    ("peak_time",    "Peak time",       "s", False),
    ("strike",       "Strike",          "deg", False),
    ("dip",          "Dip",             "deg", False),
    ("rake",         "Rake",            "deg", False),
)

_DEFAULT_FIELD = "slip_inst"
_DEFAULT_CMAP = "hot_r"   # white -> yellow -> red, fits ShakerMaker's cmap_white feel
_POINT_SIZE = 7
_MASK_OPACITY = 0.05      # un-ruptured subfaults are nearly invisible


class _FFSPControlPanel(QtWidgets.QWidget):
    """Compact side panel mirroring the Display dock layout."""

    settingsChanged = QtCore.Signal()        # noqa: N815 (Qt-style)
    projectToMainView = QtCore.Signal(bool)  # noqa: N815 (True=attach, False=detach)

    def __init__(self, source: RuptureSource, parent=None):
        super().__init__(parent)
        self.source = source
        self.setFixedWidth(280)
        self.setObjectName("FFSPControlPanel")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Field group ───────────────────────────────────────────────────
        field_box = QtWidgets.QGroupBox("Field")
        field_form = QtWidgets.QFormLayout(field_box)
        field_form.setContentsMargins(6, 4, 6, 6)
        self.field_combo = QtWidgets.QComboBox()
        for key, label, unit, animated in _FIELD_OPTIONS:
            suffix = "  ⏱" if animated else ""
            self.field_combo.addItem(f"{label}{suffix}", key)
        idx = self.field_combo.findData(_DEFAULT_FIELD)
        if idx >= 0:
            self.field_combo.setCurrentIndex(idx)
        self.field_combo.currentIndexChanged.connect(self._emit_changed)
        field_form.addRow("Show", self.field_combo)

        self.animate_cb = QtWidgets.QCheckBox("Animate with global time")
        self.animate_cb.setChecked(True)
        self.animate_cb.setToolTip(
            "When on, the field is masked by the seismic propagation time "
            "so subfaults appear as they rupture.  ‘Slip (instant)’ also "
            "ramps up linearly between rupture_time and +rise_time."
        )
        self.animate_cb.toggled.connect(self._emit_changed)
        field_form.addRow("", self.animate_cb)
        outer.addWidget(field_box)

        # ── Color Map group ───────────────────────────────────────────────
        cmap_box = QtWidgets.QGroupBox("Color Map")
        cmap_form = QtWidgets.QFormLayout(cmap_box)
        cmap_form.setContentsMargins(6, 4, 6, 6)
        self.cmap_combo = QtWidgets.QComboBox()
        for name in COLORMAP_OPTIONS:
            self.cmap_combo.addItem(name, name)
        # Inject hot_r as default even when not in the canonical list.
        if self.cmap_combo.findData(_DEFAULT_CMAP) < 0:
            self.cmap_combo.insertItem(0, _DEFAULT_CMAP, _DEFAULT_CMAP)
        self.cmap_combo.setCurrentIndex(self.cmap_combo.findData(_DEFAULT_CMAP))
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._emit_changed())
        self.cmap_preview = ColormapPreview(self.cmap_combo.currentText())
        self.cmap_combo.currentTextChanged.connect(self.cmap_preview.setColormap)
        cmap_form.addRow("Preset", self.cmap_combo)
        cmap_form.addRow("",       self.cmap_preview)

        self.auto_lbl = QtWidgets.QLabel("-")
        cmap_form.addRow("Auto range", self.auto_lbl)

        self.vmin_spin = self._value_spin()
        self.vmax_spin = self._value_spin()
        self.vmin_spin.valueChanged.connect(lambda *_: self._on_range_edited())
        self.vmax_spin.valueChanged.connect(lambda *_: self._on_range_edited())
        cmap_form.addRow("User min", self.vmin_spin)
        cmap_form.addRow("User max", self.vmax_spin)

        self.clamp_cb = QtWidgets.QCheckBox("Clamp to range")
        self.clamp_cb.toggled.connect(lambda *_: self._emit_changed())
        cmap_form.addRow("", self.clamp_cb)

        self.reset_btn = QtWidgets.QPushButton("Reset to auto")
        self.reset_btn.clicked.connect(self._reset_range)
        cmap_form.addRow("", self.reset_btn)

        outer.addWidget(cmap_box)

        # ── Rupture Front group ───────────────────────────────────────────
        front_box = QtWidgets.QGroupBox("Rupture Front")
        front_form = QtWidgets.QFormLayout(front_box)
        front_form.setContentsMargins(6, 4, 6, 6)
        self.mask_cb = QtWidgets.QCheckBox("Hide un-ruptured subfaults")
        self.mask_cb.setChecked(True)
        self.mask_cb.setToolTip(
            "When on, subfaults whose rupture_time > current time are "
            "rendered with very low opacity so the rupture front is visible."
        )
        self.mask_cb.toggled.connect(self._emit_changed)
        front_form.addRow("", self.mask_cb)
        outer.addWidget(front_box)

        # ── Hypocenter group ──────────────────────────────────────────────
        hypo_box = QtWidgets.QGroupBox("Hypocenter")
        hypo_form = QtWidgets.QFormLayout(hypo_box)
        hypo_form.setContentsMargins(6, 4, 6, 6)
        self.hypo_cb = QtWidgets.QCheckBox("Show hypocenter marker")
        self.hypo_cb.setChecked(True)
        self.hypo_cb.toggled.connect(self._emit_changed)
        hypo_form.addRow("", self.hypo_cb)
        outer.addWidget(hypo_box)

        # ── Project onto main 3-D view ────────────────────────────────────
        proj_box = QtWidgets.QGroupBox("Main 3-D View")
        proj_layout = QtWidgets.QVBoxLayout(proj_box)
        proj_layout.setContentsMargins(6, 4, 6, 6)
        self.project_btn = QtWidgets.QPushButton("Add to main 3-D view")
        self.project_btn.setCheckable(True)
        self.project_btn.setToolTip(
            "Overlay this rupture on the active pane of the seismic Main "
            "view so the fault animates together with the wave field.  The "
            "rupture's metres are scaled to kilometres by default."
        )
        self.project_btn.toggled.connect(self.projectToMainView)
        proj_layout.addWidget(self.project_btn)
        proj_layout.addWidget(QtWidgets.QLabel(
            "Tip: clic en un pane del Main view para fijarlo como activo "
            "antes de presionar este botón."
        ))
        outer.addWidget(proj_box)

        # ── Source info ───────────────────────────────────────────────────
        info_box = QtWidgets.QGroupBox("Source")
        info_layout = QtWidgets.QVBoxLayout(info_box)
        info_layout.setContentsMargins(6, 4, 6, 6)
        path = (source.source_path.name if source.source_path else "<in-memory>")
        info_layout.addWidget(QtWidgets.QLabel(f"File: {path}"))
        info_layout.addWidget(QtWidgets.QLabel(f"Subfaults: {source.n_subfaults}"))
        mag = source.params.get("magnitude")
        if mag is not None:
            info_layout.addWidget(QtWidgets.QLabel(f"Magnitude: {float(mag):.2f}"))
        outer.addWidget(info_box)

        outer.addStretch(1)

        self._user_locked_range = False
        self._syncing = False
        self._refresh_auto_range()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _value_spin() -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(6)
        s.setRange(-1e12, 1e12)
        s.setSingleStep(0.1)
        return s

    def _emit_changed(self) -> None:
        if self._syncing:
            return
        self.settingsChanged.emit()

    def _on_range_edited(self) -> None:
        if self._syncing:
            return
        self._user_locked_range = True
        self._emit_changed()

    def _reset_range(self) -> None:
        self._user_locked_range = False
        self._refresh_auto_range(force_spinboxes=True)
        self._emit_changed()

    def _refresh_auto_range(self, *, force_spinboxes: bool = False) -> None:
        name = self.current_field_key()
        # 'slip_inst' shares the auto-limits of the final slip.
        lo, hi = self.source.field_limits("slip" if name == "slip_inst" else name)
        self.auto_lbl.setText(f"{lo:.4g}  /  {hi:.4g}")
        if not self._user_locked_range or force_spinboxes:
            self._syncing = True
            try:
                self.vmin_spin.setValue(float(lo))
                self.vmax_spin.setValue(float(hi))
            finally:
                self._syncing = False

    # ── Public read-only API consumed by the rupture pane ────────────────

    def current_field_key(self) -> str:
        return str(self.field_combo.currentData() or _DEFAULT_FIELD)

    def animate_enabled(self) -> bool:
        return bool(self.animate_cb.isChecked())

    def colormap_name(self) -> str:
        return str(self.cmap_combo.currentText() or _DEFAULT_CMAP)

    def color_range(self) -> tuple[float, float]:
        return float(self.vmin_spin.value()), float(self.vmax_spin.value())

    def clamp_enabled(self) -> bool:
        return bool(self.clamp_cb.isChecked())

    def mask_enabled(self) -> bool:
        return bool(self.mask_cb.isChecked())

    def hypocenter_enabled(self) -> bool:
        return bool(self.hypo_cb.isChecked())

    def snapshot(self) -> dict:
        """Freeze the current visualisation settings into a plain dict.

        Consumed by :class:`RuptureOverlay` at attach time so the main 3-D
        view shows the same field, colormap, range, mask and hypocenter
        choices the user dialed in on this pane.  The snapshot is a copy —
        later edits in the panel do NOT propagate to overlays already
        attached.
        """
        return {
            "field": self.current_field_key(),
            "animate": self.animate_enabled(),
            "colormap": self.colormap_name(),
            "color_range": self.color_range(),
            "clamp": self.clamp_enabled(),
            "mask": self.mask_enabled(),
            "hypocenter": self.hypocenter_enabled(),
        }

    def on_field_changed(self) -> None:
        """Recompute auto-range when the field selector changes."""
        self._user_locked_range = False
        self._refresh_auto_range(force_spinboxes=True)


class RuptureTab(QtWidgets.QWidget):
    """Self-contained tab widget for one FFSP rupture source.

    Exposes the same surface the :class:`TabbedMultiViewArea` uses:
    :meth:`on_session_updated` is called on every broadcast and
    :meth:`dispose` is called on close — the implementation just
    needs to render the fault and clean up its plotter.
    """

    def __init__(self, session, source: RuptureSource, parent=None):
        super().__init__(parent)
        self.session = session
        self.source = source
        self.setObjectName("RuptureTab")

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 3-D viewport ──────────────────────────────────────────────────
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        try:
            self.plotter.show_axes()
        except Exception:
            pass
        root.addWidget(self.plotter.interactor, 1)

        # ── Side controls ─────────────────────────────────────────────────
        self.controls = _FFSPControlPanel(source, self)
        self.controls.settingsChanged.connect(self._on_settings_changed)
        self.controls.field_combo.currentIndexChanged.connect(
            lambda *_: self.controls.on_field_changed()
        )
        self.controls.projectToMainView.connect(self._on_project_clicked)
        # Tracks the pane(s) where this rupture has been overlaid so the
        # checkable button stays in sync if the user closes the target tab
        # or detaches via some other path.
        self._overlay_panes: list = []
        # Remember the last UTM dialog choice so the user does not have to
        # retype it when re-projecting in the same session.
        self._last_projection: dict | None = None
        root.addWidget(self.controls, 0)

        # ── Actors (lazy) ─────────────────────────────────────────────────
        self._cloud: Optional["pv.PolyData"] = None
        self._point_actor = None
        self._hypo_actor = None

        self._build_scene()
        self._update_scalars()
        try:
            self.plotter.reset_camera()
        except Exception:
            pass

    # ── Scene assembly ───────────────────────────────────────────────────

    def _display_points(self) -> np.ndarray:
        """Points for this tab's standalone viewport.

        :class:`RuptureSource.points` keeps z as **positive depth** (the
        seismic Display Transform consumes them in that form).  The
        dedicated FFSP tab does NOT go through the Display Transform, so
        we apply the same depth-flip here — otherwise the fault would
        render above the origin (PyVista uses z+ up by default).
        """
        pts = np.asarray(self.source.points, dtype=float).copy()
        pts[:, 2] *= -1.0   # depth+ → z- so the fault hangs below the surface
        return pts

    def _build_scene(self) -> None:
        self._cloud = pv.PolyData(self._display_points())
        self._cloud.point_data["active"] = np.zeros(self.source.n_subfaults, dtype=np.float32)
        self._point_actor = self.plotter.add_mesh(
            self._cloud,
            scalars="active",
            cmap=self.controls.colormap_name(),
            point_size=_POINT_SIZE,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": self._scalar_bar_title(),
                "fmt": "%.3g",
                "n_labels": 4,
            },
        )
        self._update_hypocenter()

    def _scalar_bar_title(self) -> str:
        key = self.controls.current_field_key()
        for k, label, unit, _ in _FIELD_OPTIONS:
            if k == key:
                return f"{label} [{unit}]"
        return key

    def _update_hypocenter(self) -> None:
        if self._hypo_actor is not None:
            try:
                self.plotter.remove_actor(self._hypo_actor, render=False)
            except Exception:
                pass
            self._hypo_actor = None
        if not self.controls.hypocenter_enabled():
            return
        x, y, z = self.source.hypocenter_xyz()
        # Same z-flip as :meth:`_display_points` so the marker stays
        # glued to the cloud in the standalone tab.
        sphere = pv.Sphere(radius=self._hypocenter_radius(), center=(x, y, -z))
        self._hypo_actor = self.plotter.add_mesh(
            sphere,
            color="red",
            opacity=1.0,
            specular=0.0,
            render=False,
            show_scalar_bar=False,
        )

    def _hypocenter_radius(self) -> float:
        # 1.5% of the largest fault dimension keeps the marker readable
        # at any zoom level without obscuring the cloud.
        xmin, xmax, ymin, ymax, zmin, zmax = self.source.bounds()
        span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
        return float(0.015 * span)

    # ── Scalars / opacity updates ────────────────────────────────────────

    def _current_time(self) -> float:
        """Best available 'now' for the rupture animation.

        Prefers the seismic propagation time so the rupture animates in
        lock-step with the wave field.  Falls back to a manual interpretation
        of the slider when the seismic adapter is absent.
        """
        try:
            return float(self.session.current_time())
        except Exception:
            pass
        try:
            t_idx = int(self.session.state.time_index)
            times = getattr(self.session.adapter, "time", None)
            if times is not None and len(times) > 0:
                idx = max(0, min(t_idx, len(times) - 1))
                return float(times[idx])
        except Exception:
            pass
        return 0.0

    def _compute_scalars(self) -> np.ndarray:
        key = self.controls.current_field_key()
        t = self._current_time()
        if key == "slip_inst":
            return self.source.instantaneous_slip(t)
        return self.source.field(key)

    def _compute_opacity(self) -> np.ndarray | None:
        if not self.controls.animate_enabled() or not self.controls.mask_enabled():
            return None
        t = self._current_time()
        mask = self.source.rupture_mask(t)
        return np.where(mask, 1.0, _MASK_OPACITY).astype(np.float32)

    def _update_scalars(self, *, render: bool = True) -> None:
        if self._cloud is None or self._point_actor is None:
            return
        scalars = self._compute_scalars()
        self._cloud.point_data["active"] = np.asarray(scalars, dtype=np.float32)

        cmap_name = effective_colormap_name(self.controls.colormap_name(), inverted=False)
        vmin, vmax = self._effective_color_range(scalars)
        mapper = getattr(self._point_actor, "mapper", None)
        if mapper is not None:
            try:
                mapper.scalar_range = (vmin, vmax)
            except Exception:
                pass
            try:
                lut = mapper.lookup_table
                # PyVista stashes the cmap name; safest is to rebuild via add_mesh
                # only when the cmap actually changes — for now we just update
                # the range and rely on the next full rebuild for cmap changes.
                lut.SetTableRange(vmin, vmax)
            except Exception:
                pass

        opacity = self._compute_opacity()
        if opacity is not None:
            # Per-point opacity uses a separate scalar array.  PyVista's
            # ``add_mesh(opacity=array)`` API takes a per-point lookup;
            # to keep the cloud lit we forward through ``opacity`` on the
            # actor's property — but VTK's actor opacity is scalar only,
            # so we approximate by zeroing the alpha of masked points
            # through a colour replacement on the LUT.  Simplest robust
            # workaround: hide un-ruptured points by writing NaN to the
            # scalar so VTK draws them with the NaN colour (transparent).
            masked_scalars = np.where(opacity > 0.5, scalars, np.nan).astype(np.float32)
            self._cloud.point_data["active"] = masked_scalars

        if self._cloud is not None:
            self._cloud.Modified()
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def _effective_color_range(self, scalars: np.ndarray) -> tuple[float, float]:
        vmin, vmax = self.controls.color_range()
        if not self.controls.clamp_enabled():
            arr = scalars[np.isfinite(scalars)]
            if arr.size == 0:
                return float(vmin), float(vmax)
            data_min = float(arr.min())
            data_max = float(arr.max())
            if data_max <= data_min:
                data_max = data_min + 1.0
            return data_min, data_max
        if vmax <= vmin:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_settings_changed(self) -> None:
        # Field / cmap / range / mask toggles — rebuild from scratch when
        # the cmap changed, otherwise just refresh scalars.
        new_cmap = self.controls.colormap_name()
        if (self._point_actor is not None and
                getattr(self, "_active_cmap", new_cmap) != new_cmap):
            self._rebuild_actor()
        else:
            self._active_cmap = new_cmap
        # Scalar-bar title may need updating when the field changes.
        try:
            sb = getattr(self.plotter, "scalar_bar", None)
            if sb is not None:
                sb.SetTitle(self._scalar_bar_title())
        except Exception:
            pass
        self._update_hypocenter()
        self._update_scalars()

    def _rebuild_actor(self) -> None:
        """Recreate the point actor — needed when the colormap changes."""
        if self._point_actor is not None:
            try:
                self.plotter.remove_actor(self._point_actor, render=False)
            except Exception:
                pass
            self._point_actor = None
        if self._cloud is None:
            return
        self._active_cmap = self.controls.colormap_name()
        self._point_actor = self.plotter.add_mesh(
            self._cloud,
            scalars="active",
            cmap=self._active_cmap,
            point_size=_POINT_SIZE,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": self._scalar_bar_title(),
                "fmt": "%.3g",
                "n_labels": 4,
            },
        )

    # ── Project onto main 3-D view ───────────────────────────────────────

    def _find_target_panes(self) -> list:
        """Return the panes that should receive (or release) the overlay.

        Strategy: pick the active pane of the first ``MultiViewArea`` tab
        in the multi-view (i.e. the seismic side), ignoring any rupture
        tabs.  Falls back to every visible pane when no active pane is
        flagged — that way the user does not have to remember to click on
        the seismic pane first.
        """
        window = getattr(self.session, "window", None)
        multi = getattr(window, "multi_view", None) if window is not None else None
        if multi is None:
            return []
        try:
            tabs = multi._tabs
        except Exception:
            return []
        # Lazy import avoids a circular path with multi_view at module load.
        from .multi_view import MultiViewArea
        for i in range(tabs.count()):
            area = tabs.widget(i)
            if not isinstance(area, MultiViewArea):
                continue
            active = getattr(area, "active_pane", None)
            if active is not None:
                return [active]
            panes = getattr(area, "_panes", None) or []
            if panes:
                return list(panes)
        return []

    def _reset_project_button(self) -> None:
        """Uncheck the project button without re-triggering its slot."""
        btn = self.controls.project_btn
        btn.blockSignals(True)
        btn.setChecked(False)
        btn.blockSignals(False)
        btn.setText("Add to main 3-D view")

    def _on_project_clicked(self, attach: bool) -> None:
        if attach:
            targets = self._find_target_panes()
            if not targets:
                QtWidgets.QMessageBox.information(
                    self,
                    "No seismic view found",
                    "Abre la pestaña Main view (o cualquier tab sísmica) "
                    "antes de proyectar la falla.",
                )
                self._reset_project_button()
                return

            dlg = _RuptureProjectionDialog(self.source, self,
                                           previous=self._last_projection)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                self._reset_project_button()
                return
            projection = dlg.values()
            self._last_projection = projection

            # Freeze the FFSP-panel state right before the projection so each
            # overlay rendered in the main view is born with the same field,
            # colormap and range the user picked here.  Re-projecting after
            # editing the controls takes a fresh snapshot — overlays already
            # attached keep the snapshot they were created with.
            display_settings = self.controls.snapshot()

            for pane in targets:
                try:
                    pane.attach_rupture_overlay(
                        self.source,
                        external_coord_km=projection["external_coord_km"],
                        internal_ref_km=projection["internal_ref_km"],
                        strike_deg=projection["strike_deg"],
                        unit_scale=projection["unit_scale"],
                        display_settings=display_settings,
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Overlay failed",
                        f"No pude proyectar la falla:\n\n{exc}",
                    )
                    continue
                self._overlay_panes.append(pane)
            if not self._overlay_panes:
                self._reset_project_button()
                return
            self.controls.project_btn.setText("Remove from main 3-D view")
        else:
            for pane in self._overlay_panes:
                try:
                    pane.detach_rupture_overlay()
                except Exception:
                    pass
            self._overlay_panes.clear()
            self.controls.project_btn.setText("Add to main 3-D view")

    # ── Tab-system contract ──────────────────────────────────────────────

    def on_session_updated(self, reason: str) -> None:
        """Receive broadcasts from :class:`TabbedMultiViewArea`."""
        if reason in ("time", "playback", "full", "init"):
            self._update_scalars()

    def dispose(self) -> None:
        # Drop any overlay still wired into the seismic panes — otherwise
        # closing the rupture tab leaves orphan actors behind.
        for pane in list(self._overlay_panes):
            try:
                pane.detach_rupture_overlay()
            except Exception:
                pass
        self._overlay_panes.clear()
        try:
            self.plotter.close()
        except Exception:
            pass


# ── UTM placement dialog ─────────────────────────────────────────────────────


class _RuptureProjectionDialog(QtWidgets.QDialog):
    """Ask the user where to anchor the fault in UTM (km).

    Same arguments as :meth:`shakermaker.ffspsource.FFSPSource.plot_spacial_distribution`,
    typed verbatim as Python-style 2-vectors:

        internal_ref  = [-8, 0]
        external_coord = [utmx/1000, utmy/1000]

    Strike rotation is read from the HDF5 ``parameters`` group automatically.
    The fault's native metres are scaled to kilometres internally — no
    extra unit field to fill.
    """

    def __init__(
        self,
        source: RuptureSource,
        parent=None,
        previous: dict | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Project rupture onto Main 3-D view")
        self.setModal(True)
        self._source = source

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        strike = float(source.params.get("strike", 0.0))
        intro = QtWidgets.QLabel(
            "Pegá los dos vectores tal cual los usás en <code>plot_spacial_distribution</code>,<br>"
            "pero en <b>metros</b> (sin dividir por 1000 — así no se pierden decimales).<br>"
            "<code>internal_ref</code> sigue la convención de los ejes del plot 2-D:<br>"
            "<b>primer valor = Along Dip</b>, <b>segundo valor = Along Strike</b>.<br>"
            f"Strike del HDF5: <b>{strike:.3f}°</b> — se aplica automáticamente."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        form.setSpacing(6)

        self.internal_edit = QtWidgets.QLineEdit()
        self.internal_edit.setPlaceholderText("-8000, 0   (Along Dip, Along Strike)")
        self.internal_edit.setToolTip(
            "internal_ref [m]: punto en el frame fault-local en convención "
            "del plot 2-D — (Along Dip, Along Strike).  (0, 0) = hipocentro."
        )
        form.addRow("internal_ref [m]", self.internal_edit)

        self.external_edit = QtWidgets.QLineEdit()
        self.external_edit.setPlaceholderText("utmx, utmy   (e.g. 359958.176, 6294124.525)")
        self.external_edit.setToolTip(
            "external_coord [m]: UTM (Easting, Northing) en metros donde "
            "debe quedar el internal_ref."
        )
        form.addRow("external_coord [m]", self.external_edit)

        layout.addLayout(form)

        # Optional override — collapsed under a small "advanced" toggle so
        # the 95 % case stays clean.
        adv_btn = QtWidgets.QToolButton()
        adv_btn.setText("Override strike…")
        adv_btn.setCheckable(True)
        adv_btn.setStyleSheet("QToolButton { border: none; color: #555; }")
        self._adv_box = QtWidgets.QWidget()
        adv_layout = QtWidgets.QFormLayout(self._adv_box)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        self.strike_spin = QtWidgets.QDoubleSpinBox()
        self.strike_spin.setDecimals(3)
        self.strike_spin.setRange(-720.0, 720.0)
        self.strike_spin.setSuffix(" °")
        self.strike_spin.setValue(strike)
        adv_layout.addRow("Strike", self.strike_spin)
        self._adv_box.setVisible(False)
        adv_btn.toggled.connect(self._adv_box.setVisible)
        layout.addWidget(adv_btn)
        layout.addWidget(self._adv_box)

        # Restore previous input on re-projection.  Values are stored in km
        # internally in ShakerMaker convention (ref_x = Along Strike,
        # ref_y = Along Dip); the dialog asks in metres in plot-axes
        # convention (Along Dip, Along Strike), so we multiply by 1000 to
        # convert AND swap the two components so the user sees the same
        # values they last typed.
        if previous:
            try:
                rx, ry = previous["internal_ref_km"]   # (along-strike, along-dip)
                ex, ey = previous["external_coord_km"]
                self.internal_edit.setText(f"{ry * 1000:g}, {rx * 1000:g}")
                self.external_edit.setText(f"{ex * 1000:.6f}, {ey * 1000:.6f}")
                self.strike_spin.setValue(previous.get("strike_deg", strike))
            except Exception:
                pass

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._parsed: dict | None = None

    # ── Parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_vec(text: str) -> tuple[float, float]:
        """Accept ``x, y``, ``[x, y]``, ``(x, y)``, ``x y`` — return ``(x, y)``."""
        s = (text or "").strip().strip("[](){} ").replace(";", ",")
        # Allow comma-or-whitespace separation.
        if "," in s:
            parts = s.split(",")
        else:
            parts = s.split()
        if len(parts) != 2:
            raise ValueError(
                f"Esperaba dos números separados por coma; recibí: {text!r}"
            )
        return float(parts[0].strip()), float(parts[1].strip())

    def _validate_and_accept(self) -> None:
        try:
            internal_m = self._parse_vec(self.internal_edit.text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "internal_ref inválido", str(exc))
            return
        try:
            external_m = self._parse_vec(self.external_edit.text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "external_coord inválido", str(exc))
            return
        # The dialog asks the user in metres so every UTM decimal survives
        # the round-trip.  The overlay's transform expects kilometres, so
        # we divide by 1000 right at the boundary.
        #
        # Convention swap: the user types ``(Along Dip, Along Strike)`` —
        # i.e. the same order as the 2-D plot axes shown by
        # ``plot_spacial_distribution`` (X = Along Dip, Y = Along Strike).
        # ShakerMaker's internal formula expects the OPPOSITE order
        # ``(ref_x = Along Strike, ref_y = Along Dip)``.  We swap on the
        # way in so the user can paste their plot-frame values verbatim
        # and the math still matches the function call.
        self._parsed = {
            "internal_ref_km":   (internal_m[1] / 1000.0, internal_m[0] / 1000.0),
            "external_coord_km": (external_m[0] / 1000.0, external_m[1] / 1000.0),
            "strike_deg":        float(self.strike_spin.value()),
            "unit_scale":        1e-3,   # HDF5 metres → scene kilometres, always.
        }
        self.accept()

    def values(self) -> dict:
        # ``_parsed`` is populated in :meth:`_validate_and_accept` before
        # the dialog closes with ``Accepted``.
        if self._parsed is None:
            raise RuntimeError("dialog values requested before validation")
        return dict(self._parsed)


# ── Rupture overlay (lives inside a seismic ViewPane) ─────────────────────────

class RuptureOverlay:
    """Renders a :class:`RuptureSource` as an additional actor inside a
    foreign :class:`~.multi_view.ViewPane`.

    Owns its own ``PolyData`` and actors but borrows the host pane's
    ``QtInteractor``.  Animates with the global propagation time through
    :meth:`on_session_updated`, which the host pane calls right after its
    own scene refresh in :meth:`ViewPane.on_session_updated`.

    Coordinate transform
    --------------------
    The HDF5 stores subfault positions in **metres**, fault-local frame,
    with the hypocentre at the origin and ``x`` along-strike / ``y``
    along the horizontal projection of dip / ``z`` = depth.  To plant
    the fault inside a seismic mesh that lives in UTM kilometres we
    reproduce :meth:`shakermaker.ffspsource.FFSPSource.plot_spacial_distribution`:

    1. Scale to UTM units (``unit_scale``, default ``1e-3`` for km).
    2. Rotate ``(x, y)`` around the hypocentre by the fault strike so
       along-strike runs in the correct UTM direction (strike measured
       from North, just like the HDF5 ``params['strike']``).
    3. Translate so the user-supplied ``internal_ref`` (a point in
       fault-local km, e.g. ``(0, 0)`` for the hypocentre or ``(-8, 0)``
       for the surface trace tip the user typically uses) lands at the
       user-supplied ``external_coord`` (UTM East, UTM North in km).
    """

    #: Scalar value used for un-ruptured subfaults so VTK draws them
    #: transparent (via the LUT's NaN colour).
    _MASKED = float("nan")

    def __init__(
        self,
        pane,
        source: RuptureSource,
        *,
        external_coord_km: tuple[float, float] = (0.0, 0.0),
        internal_ref_km: tuple[float, float] = (0.0, 0.0),
        strike_deg: float | None = None,
        unit_scale: float = 1e-3,
        z_scale: float | None = None,
        display_settings: dict | None = None,
    ):
        self.pane = pane
        self.source = source
        self.unit_scale = float(unit_scale)
        # Vertical scale defaults to the horizontal one (m → km when
        # ``unit_scale = 1e-3``).  Kept separate so the user can stretch
        # depth independently if their seismic mesh uses different units
        # on z (rare but legal).
        self.z_scale = float(z_scale if z_scale is not None else unit_scale)
        self.external_coord_km = (float(external_coord_km[0]), float(external_coord_km[1]))
        self.internal_ref_km = (float(internal_ref_km[0]), float(internal_ref_km[1]))

        if strike_deg is None:
            strike_deg = float(source.params.get("strike", 0.0))
        self.strike_deg = float(strike_deg)

        # Snapshot of the FFSP-panel state captured at attach time.  This
        # freezes the user's field / colormap / range / mask / hypocenter /
        # animate choices so the overlay in the main 3-D view is born with
        # the same look the importer pane was showing.  No live sync: later
        # edits in the panel only affect future re-projections.
        self._init_display_settings(display_settings)

        # The seismic scene goes through ``adapter._apply_display_transform_m``
        # to swap axes / flip depth (Geographic → Display Transform, e.g.
        # ``[[0,1,0],[1,0,0],[0,0,-1]]``).  We snapshot the same matrix at
        # attach time and apply it to the rupture, so the fault sits in
        # the *display* frame instead of in raw UTM — otherwise the two
        # ride different coordinate systems even when their UTM math is
        # right.
        self._display_transform = self._read_display_transform()

        pts_utm = self._transform_points()

        self._cloud = pv.PolyData(pts_utm)
        self._cloud.point_data["active"] = np.zeros(source.n_subfaults, dtype=np.float32)

        plotter = pane.plotter

        # Initial clim: prefer the user's snapshot range so the overlay
        # opens with the same scale shown in the importer pane.  When the
        # user did not clamp, the per-frame update in :meth:`_update_scalars`
        # will recompute the range from the live data — so this initial
        # value only matters for the first frame.
        vmin0, vmax0 = self._snapshot_clim_default()

        cmap_name = effective_colormap_name(self.colormap, inverted=False)
        self._actor = plotter.add_mesh(
            self._cloud,
            scalars="active",
            cmap=cmap_name,
            clim=(vmin0, vmax0),
            point_size=max(_POINT_SIZE - 1, 4),
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )

        # Hypocentre marker — only when the user kept "Show hypocenter
        # marker" on in the importer.  Same coordinate transform as the
        # point cloud; radius is 1 % of the largest in-plane span so it
        # stays visible regardless of zoom.
        self._hypo_actor = None
        if self.show_hypocenter:
            self._hypo_actor = self._build_hypocenter_actor(pts_utm)

        self._update_scalars()

    # ── Display-settings snapshot ────────────────────────────────────────

    def _init_display_settings(self, display_settings: dict | None) -> None:
        """Read the snapshot dict and store the visual flags on ``self``.

        Missing keys fall back to the legacy hard-coded defaults so older
        callers (or unit tests) that build a :class:`RuptureOverlay`
        without passing ``display_settings`` still get the original look.
        """
        ds = dict(display_settings or {})
        self.field = str(ds.get("field") or "slip")
        self.animate = bool(ds.get("animate", True))
        self.colormap = str(ds.get("colormap") or _DEFAULT_CMAP)
        rng = ds.get("color_range")
        if rng is None:
            lo, hi = self.source.field_limits(
                "slip" if self.field == "slip_inst" else self.field
            )
        else:
            lo, hi = float(rng[0]), float(rng[1])
        if hi <= lo:
            hi = lo + 1.0
        self.color_range = (float(lo), float(hi))
        self.clamp = bool(ds.get("clamp", False))
        self.mask = bool(ds.get("mask", True))
        self.show_hypocenter = bool(ds.get("hypocenter", True))

    def _snapshot_clim_default(self) -> tuple[float, float]:
        return self.color_range

    def _effective_color_range(self, scalars: np.ndarray) -> tuple[float, float]:
        """Mirror :meth:`RuptureTab._effective_color_range`.

        When the user clamped the range, return their (vmin, vmax) verbatim.
        Otherwise fit to the finite values in ``scalars`` so the colormap
        rescales with the rupture front instead of clipping.
        """
        vmin, vmax = self.color_range
        if not self.clamp:
            arr = np.asarray(scalars, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float(vmin), float(vmax)
            data_min = float(arr.min())
            data_max = float(arr.max())
            if data_max <= data_min:
                data_max = data_min + 1.0
            return data_min, data_max
        if vmax <= vmin:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    def _build_hypocenter_actor(self, pts_utm: np.ndarray):
        plotter = getattr(self.pane, "plotter", None)
        if plotter is None:
            return None
        hx, hy, hz = self._transform_point(*self.source.hypocenter_xyz())
        # ``ndarray.ptp()`` was removed in NumPy 2.0 — use ``np.ptp`` so the
        # overlay works on both old and new NumPy.
        span = max(
            float(np.ptp(pts_utm[:, 0])),
            float(np.ptp(pts_utm[:, 1])),
            float(np.ptp(pts_utm[:, 2])),
            1.0,
        )
        radius = max(0.01 * span, 1e-6)
        return plotter.add_mesh(
            pv.Sphere(radius=radius, center=(hx, hy, hz)),
            color="red",
            opacity=1.0,
            show_scalar_bar=False,
        )

    # ── Coordinate transform helpers ─────────────────────────────────────

    def _strike_sin_cos(self) -> tuple[float, float]:
        rad = np.radians(self.strike_deg)
        return float(np.sin(rad)), float(np.cos(rad))

    def _read_display_transform(self) -> np.ndarray:
        """Best-effort snapshot of the Geographic Display Transform.

        Returns identity if the session/adapter is missing or doesn't
        expose the matrix — that keeps the overlay usable even with a
        bare-bones rupture-only session.
        """
        try:
            session = self.pane.session
            mat = session.current_display_transform()
            mat = np.asarray(mat, dtype=float)
            if mat.shape == (3, 3):
                return mat
        except Exception:
            pass
        return np.eye(3, dtype=float)

    def _apply_seismic_frame(self, utm_km: np.ndarray) -> np.ndarray:
        """Take an ``(N, 3)`` array in UTM km and return display-frame metres.

        Mirrors :meth:`ViewerDataAdapter._apply_display_transform_m` so the
        rupture and the seismic mesh share the same coordinate system.
        """
        utm_m = np.asarray(utm_km, dtype=float) * 1000.0
        return utm_m @ self._display_transform

    def _transform_points(self) -> np.ndarray:
        """Vectorised transform of every subfault from fault-local m to the
        seismic display frame (metres).

        Pipeline (matches ``plot_spacial_distribution`` + the Geographic
        Display Transform applied to the seismic mesh):

        1. Scale fault-local metres → kilometres (``unit_scale``).
        2. Rotate ``(x, y)`` around the hypocentre by the fault strike.
        3. Translate so ``internal_ref`` lands at ``external_coord``.
        4. Multiply by 1000 → metres and apply the Display Transform.
        """
        sc = self.unit_scale
        zs = self.z_scale
        sin_s, cos_s = self._strike_sin_cos()

        pts = np.asarray(self.source.points, dtype=float)
        x_local = pts[:, 0] * sc
        y_local = pts[:, 1] * sc

        # Rotate around the hypocentre (which is at the origin in
        # fault-local coords, so no extra centering needed).
        x_rot = x_local * sin_s + y_local * cos_s
        y_rot = x_local * cos_s - y_local * sin_s

        # Translate so internal_ref (after the same rotation) sits at
        # external_coord — reproduces ShakerMaker's offset formula.
        ref_x, ref_y = self.internal_ref_km
        ref_x_rot = ref_x * sin_s + ref_y * cos_s
        ref_y_rot = ref_x * cos_s - ref_y * sin_s
        ext_x, ext_y = self.external_coord_km
        offset_x = ext_x - ref_x_rot
        offset_y = ext_y - ref_y_rot

        x_utm_km = x_rot + offset_x   # UTM Easting (km)
        y_utm_km = y_rot + offset_y   # UTM Northing (km)
        # z stays positive (= depth below surface) at this stage; the
        # Display Transform's ``[0,0,-1]`` row flips it into "below origin"
        # when :meth:`_apply_seismic_frame` runs the matrix multiply.
        z_utm_km = pts[:, 2].astype(float) * zs

        # ShakerMaker stores seismic stations as ``(Northing, Easting, depth)``
        # — geophysics / Liu-Archuleta convention.  The Display Transform
        # ``[[0,1,0],[1,0,0],[0,0,-1]]`` then maps that to display
        # ``(Easting, Northing, -depth)``.  We feed the matrix the same
        # ``(N, E, depth)`` ordering so the rupture lands on top of the
        # corresponding seismic mesh points; the dialog keeps accepting
        # ``external_coord = [Easting, Northing]`` because that is what
        # ``FFSPSource.plot_spacial_distribution`` documents.
        model_km = np.column_stack([y_utm_km, x_utm_km, z_utm_km])
        return self._apply_seismic_frame(model_km)

    def _transform_point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Single-point version of :meth:`_transform_points` for the hypocentre."""
        sc = self.unit_scale
        zs = self.z_scale
        sin_s, cos_s = self._strike_sin_cos()
        x_l = x * sc
        y_l = y * sc
        x_rot = x_l * sin_s + y_l * cos_s   # → UTM Easting
        y_rot = x_l * cos_s - y_l * sin_s   # → UTM Northing
        ref_x, ref_y = self.internal_ref_km
        ref_x_rot = ref_x * sin_s + ref_y * cos_s
        ref_y_rot = ref_x * cos_s - ref_y * sin_s
        ext_x, ext_y = self.external_coord_km
        x_utm_km = x_rot + ext_x - ref_x_rot   # Easting (km)
        y_utm_km = y_rot + ext_y - ref_y_rot   # Northing (km)
        z_utm_km = float(z) * zs
        # Match :meth:`_transform_points`: swap to ``(Northing, Easting, depth)``
        # so the Display Transform produces the same ``(Easting, Northing,
        # -depth)`` ordering as the seismic mesh.
        model_m = np.array([[y_utm_km, x_utm_km, z_utm_km]], dtype=float) * 1000.0
        out = model_m @ self._display_transform
        return float(out[0, 0]), float(out[0, 1]), float(out[0, 2])

    # ── Time-driven updates ──────────────────────────────────────────────

    def _current_time(self) -> float:
        try:
            return float(self.pane.session.current_time())
        except Exception:
            return 0.0

    def _update_scalars(self) -> None:
        """Refresh the point-cloud scalars using the snapshot's field/flags.

        Mirrors :meth:`RuptureTab._update_scalars` so the overlay animates
        identically to its source pane:

        * ``slip_inst`` ramps up between rupture_time and +rise_time.
        * Static fields (slip, rupture_time, ...) stay constant in time.
        * When ``animate`` AND ``mask`` are on, un-ruptured subfaults are
          set to NaN so VTK draws them transparent (the LUT's NaN colour).
        * Colour range stays anchored to the unmasked scalars when the
          user did NOT clamp — so vmax doesn't bounce around as the
          rupture front sweeps the fault.
        """
        if self._cloud is None:
            return
        t = self._current_time()
        if self.field == "slip_inst":
            scalars = np.asarray(self.source.instantaneous_slip(t), dtype=np.float32)
        else:
            scalars = np.asarray(self.source.field(self.field), dtype=np.float32)

        # Range computed BEFORE masking so the LUT stays stable as the
        # rupture front sweeps the fault (same contract as the importer
        # pane).
        vmin, vmax = self._effective_color_range(scalars)

        if self.animate and self.mask:
            rmask = self.source.rupture_mask(t)
            displayed = np.where(rmask, scalars, np.float32(self._MASKED)).astype(np.float32)
        else:
            displayed = scalars

        self._cloud.point_data["active"] = displayed

        mapper = getattr(self._actor, "mapper", None) if self._actor is not None else None
        if mapper is not None:
            try:
                mapper.scalar_range = (vmin, vmax)
            except Exception:
                pass
            try:
                lut = mapper.lookup_table
                lut.SetTableRange(vmin, vmax)
            except Exception:
                pass

        self._cloud.Modified()

    def on_session_updated(self, reason: str) -> None:
        if reason == "geometry_transform":
            # The user just edited the Geographic Display Transform — the
            # seismic mesh has rebuilt under the new matrix, so the overlay
            # has to refresh its point cloud or it sits in the old frame.
            self._rebuild_geometry()
            self._update_scalars()
            return
        if reason in ("time", "playback", "full", "init", "demand", "component"):
            self._update_scalars()

    def _rebuild_geometry(self) -> None:
        """Recompute point positions when the seismic display frame changes.

        Snapshot the new ``display_transform`` and overwrite the cloud's
        ``points`` in place so VTK reuses the same actor (no remove/add,
        which would trip the host pane's actor accounting).  The
        hypocentre sphere is rebuilt because its centre lives in the
        ``vtkPolyData`` topology and is cheaper to replace than to
        translate.
        """
        if self._cloud is None:
            return
        self._display_transform = self._read_display_transform()
        new_pts = self._transform_points()
        try:
            self._cloud.points = new_pts.astype(np.float32, copy=False)
            self._cloud.Modified()
        except Exception:
            pass

        plotter = getattr(self.pane, "plotter", None)
        if plotter is not None and self._hypo_actor is not None:
            try:
                plotter.remove_actor(self._hypo_actor, render=False)
            except Exception:
                pass
            self._hypo_actor = None
            # Only rebuild if the snapshot wanted the marker visible — if
            # the user had it off at attach time, we don't bring it back
            # just because the display frame changed.
            if self.show_hypocenter:
                try:
                    self._hypo_actor = self._build_hypocenter_actor(new_pts)
                except Exception:
                    self._hypo_actor = None

    def detach(self) -> None:
        plotter = getattr(self.pane, "plotter", None)
        for actor in (self._actor, self._hypo_actor):
            if actor is None or plotter is None:
                continue
            try:
                plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._actor = None
        self._hypo_actor = None
        self._cloud = None


__all__ = ["RuptureTab", "RuptureOverlay", "_FFSPControlPanel"]
