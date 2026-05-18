"""
visual_editors.py
=================
Modal dialogs and reusable widgets for advanced appearance controls.

* :class:`TransferFunctionDialog` -- pick / invert / discretise the
  colormap, edit per-stop colours, set NaN / below / above colours,
  configure percentile clipping and the symmetric-range toggle.
* :class:`LegendEditorDialog`     -- scalar-bar title, orientation,
  position, font sizes, outline / background.
* :class:`LegendEditorValues` /
  :class:`TransferEditorValues`   -- frozen dataclasses returned by
  the dialogs and applied via ``session.apply_*_preferences``.
* :class:`_ColorButton`           -- compact colour-swatch button used
  inside the dialogs.
* :func:`icon_button`             -- helper to build a themed
  ``QToolButton`` from one of the SVG icons in :mod:`.icons`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._imports import require_viewer_dependencies
from .cmap_preview import ColormapPreview
from .colors import COLORMAP_OPTIONS
from .icons import icon
from .theme import LIGHT_PALETTE

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()

try:
    _Signal = QtCore.Signal
except AttributeError:
    _Signal = QtCore.pyqtSignal


@dataclass(frozen=True)
class LegendEditorValues:
    visible: bool
    title: str
    orientation: str
    position: str
    label_count: int
    label_font_size: int
    title_font_size: int
    show_outline: bool
    show_background: bool


@dataclass(frozen=True)
class TransferEditorValues:
    colormap: str
    invert: bool
    discrete: bool
    bins: int
    nan_color: str
    below_color: str
    above_color: str
    use_below: bool
    use_above: bool
    symmetric_range: bool
    percentile_clip: float


class _ColorButton(QtWidgets.QToolButton):
    """Small reusable color swatch button."""

    colorChanged = _Signal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = str(color)
        self.setFixedSize(28, 24)
        self.clicked.connect(self._pick_color)
        self._sync_swatch()

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = str(color)
        self._sync_swatch()

    def _sync_swatch(self):
        self.setToolTip(self._color)
        self.setStyleSheet(
            "QToolButton {"
            f" background: {self._color};"
            " border: 1px solid #8c97a8;"
            " border-radius: 4px;"
            "}"
        )

    def _pick_color(self):
        picked = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self._color), self, "Choose color"
        )
        if not picked.isValid():
            return
        self.set_color(picked.name())
        self.colorChanged.emit(self._color)


class TransferFunctionDialog(QtWidgets.QDialog):
    """Professional color-mapping dialog that keeps the viewer layout intact."""

    def __init__(self, session, parent=None, *, initial_colormap: str | None = None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Color Mapping")
        self.setMinimumWidth(520)

        # Resolve the "initial colormap" — the caller (typically a side-
        # panel section) passes the value its own combo currently shows,
        # which may differ from ``session.current_colormap()`` when the
        # user has changed the combo without clicking Apply yet.  Falling
        # back to the session value keeps the legacy single-arg call sites
        # working.
        resolved_initial = (initial_colormap or self.session.current_colormap()).strip()
        # Strip "_r" suffix so the combo + Invert checkbox can drive it
        # independently below.
        if resolved_initial.endswith("_r"):
            base_name = resolved_initial[:-2]
            self._initial_inverted = True
        else:
            base_name = resolved_initial
            self._initial_inverted = bool(self.session.state.colormap_inverted)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QtWidgets.QLabel("Transfer Function")
        title.setStyleSheet("font-size: 15px; font-weight: 600; color: #1e3558;")
        root.addWidget(title)

        preview_box = QtWidgets.QGroupBox("Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        # Live colormap strip with arrow markers.  Same widget the side
        # panel uses, just taller and with the marker count driven by the
        # "Bins" spinbox so the user sees the ramp segmentation update
        # in real time as they tweak Discretize / Bins / Invert / marker
        # colours.
        self.preview = ColormapPreview(base_name)
        self.preview.setMinimumHeight(44)
        preview_layout.addWidget(self.preview)
        root.addWidget(preview_box)

        map_box = QtWidgets.QGroupBox("Palette")
        form = QtWidgets.QFormLayout(map_box)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(7)

        self.cmap_combo = QtWidgets.QComboBox()
        for cmap in COLORMAP_OPTIONS:
            self.cmap_combo.addItem(cmap, cmap)
        # Use the resolved initial cmap (caller's choice) instead of the
        # global session value — fixes the "dialog shows viridis even
        # though the side panel has RdBu_r selected" mismatch.
        self.cmap_combo.setCurrentText(base_name)
        self.cmap_combo.currentTextChanged.connect(self._refresh_preview)

        self.invert_cb = QtWidgets.QCheckBox("Invert ramp")
        # Use the value derived from the caller's initial cmap so a
        # name ending in "_r" lands as Invert=True.
        self.invert_cb.setChecked(bool(self._initial_inverted))
        self.invert_cb.toggled.connect(self._refresh_preview)

        self.discrete_cb = QtWidgets.QCheckBox("Discretize colors")
        self.discrete_cb.setChecked(bool(self.session.state.colormap_discrete))
        self.discrete_cb.toggled.connect(self._refresh_preview)

        self.bin_spin = QtWidgets.QSpinBox()
        self.bin_spin.setRange(3, 64)
        self.bin_spin.setValue(int(self.session.state.colormap_bins))
        self.bin_spin.setEnabled(self.discrete_cb.isChecked())
        self.bin_spin.valueChanged.connect(self._refresh_preview)
        self.discrete_cb.toggled.connect(self.bin_spin.setEnabled)

        self.nan_btn = _ColorButton(self.session.state.nan_color)
        self.below_btn = _ColorButton(self.session.state.below_range_color)
        self.above_btn = _ColorButton(self.session.state.above_range_color)
        self.use_below_cb = QtWidgets.QCheckBox("Enable")
        self.use_below_cb.setChecked(bool(self.session.state.use_below_range_color))
        self.use_above_cb = QtWidgets.QCheckBox("Enable")
        self.use_above_cb.setChecked(bool(self.session.state.use_above_range_color))

        below_row = QtWidgets.QWidget()
        below_lay = QtWidgets.QHBoxLayout(below_row)
        below_lay.setContentsMargins(0, 0, 0, 0)
        below_lay.addWidget(self.use_below_cb)
        below_lay.addWidget(self.below_btn)
        below_lay.addStretch(1)

        above_row = QtWidgets.QWidget()
        above_lay = QtWidgets.QHBoxLayout(above_row)
        above_lay.setContentsMargins(0, 0, 0, 0)
        above_lay.addWidget(self.use_above_cb)
        above_lay.addWidget(self.above_btn)
        above_lay.addStretch(1)

        form.addRow("Preset", self.cmap_combo)
        form.addRow("", self.invert_cb)
        form.addRow("", self.discrete_cb)
        form.addRow("Bins", self.bin_spin)
        form.addRow("NaN color", self.nan_btn)
        form.addRow("Below range", below_row)
        form.addRow("Above range", above_row)
        root.addWidget(map_box)

        range_box = QtWidgets.QGroupBox("Range Strategy")
        range_form = QtWidgets.QFormLayout(range_box)
        range_form.setContentsMargins(8, 8, 8, 8)
        self.symmetric_cb = QtWidgets.QCheckBox("Use symmetric zero-centered range")
        self.symmetric_cb.setChecked(bool(self.session.state.symmetric_color_range))
        self.percentile_spin = QtWidgets.QDoubleSpinBox()
        self.percentile_spin.setRange(0.0, 20.0)
        self.percentile_spin.setDecimals(1)
        self.percentile_spin.setSingleStep(0.5)
        self.percentile_spin.setSuffix(" %")
        self.percentile_spin.setValue(float(self.session.state.percentile_clip))
        range_form.addRow("", self.symmetric_cb)
        range_form.addRow("Clip tails", self.percentile_spin)
        root.addWidget(range_box)

        note = QtWidgets.QLabel(
            "The previewed palette applies to the active viewer color mapping. "
            "Manual min/max values remain controlled from the Display page."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #5a6879;")
        root.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._refresh_preview()

    def values(self) -> TransferEditorValues:
        return TransferEditorValues(
            colormap=str(self.cmap_combo.currentText()),
            invert=bool(self.invert_cb.isChecked()),
            discrete=bool(self.discrete_cb.isChecked()),
            bins=int(self.bin_spin.value()),
            nan_color=self.nan_btn.color(),
            below_color=self.below_btn.color(),
            above_color=self.above_btn.color(),
            use_below=bool(self.use_below_cb.isChecked()),
            use_above=bool(self.use_above_cb.isChecked()),
            symmetric_range=bool(self.symmetric_cb.isChecked()),
            percentile_clip=float(self.percentile_spin.value()),
        )

    def build_custom_colormap(self):
        """Return a custom :class:`matplotlib.colors.Colormap` honouring the
        user-picked marker colours, or ``None`` when no marker was edited.

        The session uses this to override the named preset so the 3-D
        scene renders the same gradient the user is seeing in the
        Preview strip.
        """
        return self.preview.build_custom_colormap()

    def _refresh_preview(self):
        """Push the current dialog state into the live ColormapPreview.

        The preview reflects three things:

        * the selected colormap name (with the ``_r`` suffix toggled on
          when "Invert ramp" is checked);
        * the marker count — N triangles = ``bins`` when Discretize is on,
          otherwise a fixed 5 markers as a visual reference;
        * (implicitly) the discretize state — the marker grid hints at the
          discrete-bin segmentation the actual scene will render.

        Drag-to-edit-individual-stops is queued for a follow-up; this
        commit focuses on showing a live, correct preview that responds
        to every change in the dialog without the old QGradient stop
        warnings (which came from the previous CSS-percent gradient).
        """
        cmap = str(self.cmap_combo.currentText() or "viridis")
        if self.invert_cb.isChecked() and not cmap.endswith("_r"):
            cmap = f"{cmap}_r"
        elif not self.invert_cb.isChecked() and cmap.endswith("_r"):
            cmap = cmap[:-2]
        try:
            if cmap != self.preview._cmap:  # noqa: SLF001 — internal sentinel
                # Switching presets invalidates any per-marker overrides
                # the user had painted onto the previous ramp.
                self.preview.clearMarkerOverrides()
            self.preview.setColormap(cmap)
        except Exception:
            pass
        try:
            count = int(self.bin_spin.value()) if self.discrete_cb.isChecked() else 5
            self.preview.setMarkerCount(max(2, count))
        except Exception:
            pass


class LegendEditorDialog(QtWidgets.QDialog):
    """Edit scalar-bar presentation without changing viewer geometry."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Legend Settings")
        self.setMinimumWidth(460)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        heading = QtWidgets.QLabel("Scalar Legend")
        heading.setStyleSheet("font-size: 15px; font-weight: 600; color: #1e3558;")
        root.addWidget(heading)

        box = QtWidgets.QGroupBox("Presentation")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(8, 8, 8, 8)

        self.visible_cb = QtWidgets.QCheckBox("Show scalar legend")
        self.visible_cb.setChecked(bool(self.session.state.show_scalar_bar))
        self.title_edit = QtWidgets.QLineEdit(self.session.state.legend_title_override or "")
        self.title_edit.setPlaceholderText("Use automatic legend title")

        self.orientation_combo = QtWidgets.QComboBox()
        self.orientation_combo.addItems(["vertical", "horizontal"])
        self.orientation_combo.setCurrentText(self.session.state.legend_orientation)

        self.position_combo = QtWidgets.QComboBox()
        self.position_combo.addItems(["right", "left", "top", "bottom"])
        self.position_combo.setCurrentText(self.session.state.legend_position)

        self.label_count_spin = QtWidgets.QSpinBox()
        self.label_count_spin.setRange(2, 12)
        self.label_count_spin.setValue(int(self.session.state.legend_label_count))

        self.label_font_spin = QtWidgets.QSpinBox()
        self.label_font_spin.setRange(7, 24)
        self.label_font_spin.setValue(int(self.session.state.legend_label_font_size))

        self.title_font_spin = QtWidgets.QSpinBox()
        self.title_font_spin.setRange(8, 28)
        self.title_font_spin.setValue(int(self.session.state.legend_title_font_size))

        self.outline_cb = QtWidgets.QCheckBox("Show outline")
        self.outline_cb.setChecked(bool(self.session.state.legend_show_outline))
        self.background_cb = QtWidgets.QCheckBox("Show translucent background")
        self.background_cb.setChecked(bool(self.session.state.legend_show_background))

        form.addRow("", self.visible_cb)
        form.addRow("Custom title", self.title_edit)
        form.addRow("Orientation", self.orientation_combo)
        form.addRow("Position", self.position_combo)
        form.addRow("Labels", self.label_count_spin)
        form.addRow("Label font", self.label_font_spin)
        form.addRow("Title font", self.title_font_spin)
        form.addRow("", self.outline_cb)
        form.addRow("", self.background_cb)
        root.addWidget(box)

        helper = QtWidgets.QLabel(
            "These settings affect the scalar legend generated in every active viewport."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #5a6879;")
        root.addWidget(helper)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def values(self) -> LegendEditorValues:
        return LegendEditorValues(
            visible=bool(self.visible_cb.isChecked()),
            title=str(self.title_edit.text()).strip(),
            orientation=str(self.orientation_combo.currentText()),
            position=str(self.position_combo.currentText()),
            label_count=int(self.label_count_spin.value()),
            label_font_size=int(self.label_font_spin.value()),
            title_font_size=int(self.title_font_spin.value()),
            show_outline=bool(self.outline_cb.isChecked()),
            show_background=bool(self.background_cb.isChecked()),
        )


def icon_button(icon_name: str, tooltip: str, parent=None) -> QtWidgets.QToolButton:
    """Create a compact icon button used by the advanced editors."""
    button = QtWidgets.QToolButton(parent)
    button.setObjectName("IconButton")
    button.setIcon(icon(icon_name, LIGHT_PALETTE.navy, 16))
    button.setIconSize(QtCore.QSize(16, 16))
    button.setToolTip(tooltip)
    button.setAutoRaise(True)
    button.setFixedSize(28, 28)
    return button


__all__ = [
    "LegendEditorDialog",
    "LegendEditorValues",
    "TransferEditorValues",
    "TransferFunctionDialog",
    "icon_button",
]
