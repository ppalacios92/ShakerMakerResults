"""Appearance dock — global rendering preferences for the active scene.

Bundles the small toggles and pickers that used to live in the header's
"Appearance" popover, plus the convenience buttons to open the legacy
:class:`LegendEditorDialog` and :class:`TransferFunctionDialog`.

This dock writes through :class:`ViewerSession` setters only; no data is
mutated and no rendering is triggered directly — the session broadcast
loop in :class:`ViewerMainWindow` rerenders on each ``"appearance"`` /
``"static_color"`` reason.
"""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .colors import BACKGROUND_PRESETS
from .theme import active_palette

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


class AppearancePanel(QtWidgets.QWidget):
    """Compact global-rendering options panel."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._syncing = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Background + point size + scalar-bar toggle ─────────────────────
        scene_box = QtWidgets.QGroupBox("Scene")
        scene_form = QtWidgets.QFormLayout(scene_box)
        scene_form.setContentsMargins(8, 6, 8, 8)
        scene_form.setHorizontalSpacing(8)

        self.background_combo = QtWidgets.QComboBox()
        for name in BACKGROUND_PRESETS:
            self.background_combo.addItem(name, name)
        self.background_combo.currentTextChanged.connect(self._on_background_changed)
        scene_form.addRow("Background", self.background_combo)

        self.point_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_size_slider.setRange(2, 24)
        self.point_size_slider.valueChanged.connect(self._on_point_size_changed)
        self.point_size_label = QtWidgets.QLabel("—")
        size_row = QtWidgets.QWidget()
        size_lay = QtWidgets.QHBoxLayout(size_row)
        size_lay.setContentsMargins(0, 0, 0, 0)
        size_lay.setSpacing(6)
        size_lay.addWidget(self.point_size_slider, 1)
        size_lay.addWidget(self.point_size_label)
        scene_form.addRow("Point size", size_row)

        self.scalar_bar_cb = QtWidgets.QCheckBox("Show scalar bar")
        self.scalar_bar_cb.toggled.connect(self._on_scalar_bar_toggled)
        scene_form.addRow("", self.scalar_bar_cb)

        outer.addWidget(scene_box)

        # ── Helpers (axes grid / ghost reference / selection labels) ────────
        helpers_box = QtWidgets.QGroupBox("Helpers")
        helpers_lay = QtWidgets.QVBoxLayout(helpers_box)
        helpers_lay.setContentsMargins(8, 6, 8, 8)
        helpers_lay.setSpacing(4)

        self.axes_grid_cb = QtWidgets.QCheckBox("Axes grid")
        self.axes_grid_cb.setToolTip(
            "Show a labelled axes grid around the rendered domain."
        )
        self.axes_grid_cb.toggled.connect(self._on_axes_grid_toggled)
        helpers_lay.addWidget(self.axes_grid_cb)

        self.ghost_cb = QtWidgets.QCheckBox("Ghost reference under warp")
        self.ghost_cb.setToolTip(
            "Overlay a translucent copy of the un-warped domain so it is\n"
            "easy to compare deformed vs reference geometry."
        )
        self.ghost_cb.toggled.connect(self._on_ghost_toggled)
        helpers_lay.addWidget(self.ghost_cb)

        self.selection_labels_cb = QtWidgets.QCheckBox("Selection labels")
        self.selection_labels_cb.setToolTip(
            "Draw node IDs as small labels next to every selected node."
        )
        self.selection_labels_cb.toggled.connect(self._on_selection_labels_toggled)
        helpers_lay.addWidget(self.selection_labels_cb)

        outer.addWidget(helpers_box)

        # ── Advanced editors (legacy dialogs) ──────────────────────────────
        editors_box = QtWidgets.QGroupBox("Advanced editors")
        editors_lay = QtWidgets.QVBoxLayout(editors_box)
        editors_lay.setContentsMargins(8, 6, 8, 8)
        editors_lay.setSpacing(4)

        self.legend_btn = QtWidgets.QPushButton("Legend settings…")
        self.legend_btn.setToolTip(
            "Open the legend editor — position, orientation, fonts,\n"
            "label count, outline and background."
        )
        self.legend_btn.clicked.connect(self._open_legend_editor)
        editors_lay.addWidget(self.legend_btn)

        self.transfer_btn = QtWidgets.QPushButton("Advanced color editor…")
        self.transfer_btn.setToolTip(
            "Open the transfer-function editor — discrete bins,\n"
            "NaN colour, above/below range colours."
        )
        self.transfer_btn.clicked.connect(self._open_transfer_editor)
        editors_lay.addWidget(self.transfer_btn)

        outer.addWidget(editors_box)

        outer.addStretch(1)
        self.refresh("init")

    # ── Sync from session state ─────────────────────────────────────────────

    def refresh(self, reason: str = "full") -> None:
        if reason == "time":
            return
        self._syncing = True
        try:
            state = self.session.state
            self._set_combo(self.background_combo, getattr(state, "background", "White"))

            point_size = self.session.suggested_point_size()
            block = self.point_size_slider.blockSignals(True)
            self.point_size_slider.setValue(int(round(point_size)))
            self.point_size_slider.blockSignals(block)
            self.point_size_label.setText(str(int(round(point_size))))

            block = self.scalar_bar_cb.blockSignals(True)
            self.scalar_bar_cb.setChecked(bool(getattr(state, "show_scalar_bar", True)))
            self.scalar_bar_cb.blockSignals(block)

            for cb, attr in (
                (self.axes_grid_cb,        "show_axes_grid"),
                (self.ghost_cb,            "ghost_warp_reference"),
                (self.selection_labels_cb, "selection_labels_enabled"),
            ):
                block = cb.blockSignals(True)
                cb.setChecked(bool(getattr(state, attr, False)))
                cb.blockSignals(block)
        finally:
            self._syncing = False

    # ── Slots ───────────────────────────────────────────────────────────────

    def _on_background_changed(self, name: str) -> None:
        if self._syncing:
            return
        self.session.set_background(str(name))

    def _on_point_size_changed(self, value: int) -> None:
        if self._syncing:
            return
        self.session.set_point_size(float(value))
        self.point_size_label.setText(str(int(value)))

    def _on_scalar_bar_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        self.session.set_show_scalar_bar(bool(checked))

    def _on_axes_grid_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        setter = getattr(self.session, "set_axes_grid_visible", None)
        if callable(setter):
            setter(bool(checked))

    def _on_ghost_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        setter = getattr(self.session, "set_ghost_warp_reference", None)
        if callable(setter):
            setter(bool(checked))

    def _on_selection_labels_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        setter = getattr(self.session, "set_selection_labels_enabled", None)
        if callable(setter):
            setter(bool(checked))

    def _open_legend_editor(self) -> None:
        try:
            from .visual_editors import LegendEditorDialog
        except Exception:
            return
        dialog = LegendEditorDialog(self.session, self.window())
        dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()

    def _open_transfer_editor(self) -> None:
        try:
            from .visual_editors import TransferFunctionDialog
        except Exception:
            return
        dialog = TransferFunctionDialog(self.session, self.window())
        dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _set_combo(combo: QtWidgets.QComboBox, value) -> None:
        block = combo.blockSignals(True)
        combo.setCurrentText(str(value))
        combo.blockSignals(block)


class AppearanceDock(QtWidgets.QDockWidget):
    """:class:`QDockWidget` shell around :class:`AppearancePanel`."""

    def __init__(self, session, parent=None):
        super().__init__("Appearance", parent)
        self.setObjectName("Dock_appearance")
        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.panel = AppearancePanel(session, self)
        self.setWidget(self.panel)

    def refresh(self, reason: str = "full") -> None:
        self.panel.refresh(reason)


__all__ = ["AppearancePanel", "AppearanceDock"]
