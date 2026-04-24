"""Qt controls used by the viewer window."""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .colors import BACKGROUND_PRESETS, COLORMAP_OPTIONS

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


class HeaderBar(QtWidgets.QWidget):
    """Minimal top bar with title and appearance controls."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel(session.title)
        font = title.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        title.setFont(font)

        self.settings_button = AppearanceButton(session)

        layout.addWidget(title, 1)
        layout.addWidget(self.settings_button, 0)

    def sync_from_state(self):
        self.settings_button.sync_from_state()


class AppearanceButton(QtWidgets.QToolButton):
    """Tool button that exposes appearance settings in a popover menu."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.setText("⚙")
        self.setToolTip("Appearance")
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setAutoRaise(True)

        self.menu_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.menu_widget)
        form.setContentsMargins(8, 8, 8, 8)

        self.background_combo = QtWidgets.QComboBox()
        for name in BACKGROUND_PRESETS:
            self.background_combo.addItem(name, name)
        self.background_combo.currentTextChanged.connect(session.set_background)

        self.colormap_combo = QtWidgets.QComboBox()
        for cmap in COLORMAP_OPTIONS:
            self.colormap_combo.addItem(cmap, cmap)
        self.colormap_combo.currentTextChanged.connect(session.set_colormap)

        self.point_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_size_slider.setMinimum(2)
        self.point_size_slider.setMaximum(24)
        self.point_size_slider.valueChanged.connect(
            lambda value: session.set_point_size(float(value))
        )

        self.point_size_label = QtWidgets.QLabel()
        self.point_size_slider.valueChanged.connect(
            lambda value: self.point_size_label.setText(str(int(value)))
        )

        point_size_row = QtWidgets.QWidget()
        point_size_layout = QtWidgets.QHBoxLayout(point_size_row)
        point_size_layout.setContentsMargins(0, 0, 0, 0)
        point_size_layout.addWidget(self.point_size_slider, 1)
        point_size_layout.addWidget(self.point_size_label, 0)

        self.scalar_bar_checkbox = QtWidgets.QCheckBox("Visible")
        self.scalar_bar_checkbox.toggled.connect(session.set_show_scalar_bar)

        form.addRow("Background", self.background_combo)
        form.addRow("Colormap", self.colormap_combo)
        form.addRow("Point size", point_size_row)
        form.addRow("Scalar bar", self.scalar_bar_checkbox)

        menu = QtWidgets.QMenu(self)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(self.menu_widget)
        menu.addAction(action)
        self.setMenu(menu)

        self.sync_from_state()

    def sync_from_state(self):
        self._set_combo_value(self.background_combo, self.session.state.background)
        self._set_combo_value(self.colormap_combo, self.session.current_colormap())
        point_size = self.session.suggested_point_size()
        block = self.point_size_slider.blockSignals(True)
        self.point_size_slider.setValue(int(round(point_size)))
        self.point_size_slider.blockSignals(block)
        self.point_size_label.setText(str(int(round(point_size))))
        block = self.scalar_bar_checkbox.blockSignals(True)
        self.scalar_bar_checkbox.setChecked(self.session.state.show_scalar_bar)
        self.scalar_bar_checkbox.blockSignals(block)

    @staticmethod
    def _set_combo_value(combo, value):
        block = combo.blockSignals(True)
        combo.setCurrentText(str(value))
        combo.blockSignals(block)


class TimeControls(QtWidgets.QWidget):
    """Bottom transport bar with stepping, slider and playback."""

    def __init__(self, session, on_play_toggled=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.on_play_toggled = on_play_toggled
        self._pending_value = session.state.time_index
        self._dispatch_timer = QtCore.QTimer(self)
        self._dispatch_timer.setSingleShot(True)
        self._dispatch_timer.setInterval(25)
        self._dispatch_timer.timeout.connect(self._flush_pending_value)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.jump_back_button = self._make_button("◀◀", lambda: session.jump_time(-10))
        self.step_back_button = self._make_button("◀", lambda: session.step_time(-1))
        self.step_forward_button = self._make_button("▶", lambda: session.step_time(1))
        self.jump_forward_button = self._make_button("▶▶", lambda: session.jump_time(10))

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(len(session.adapter.time) - 1, 0))
        self.slider.valueChanged.connect(self._queue_time_index)
        self.slider.sliderReleased.connect(self._flush_pending_value)

        self.time_label = QtWidgets.QLabel()
        self.play_button = self._make_button("▶ Play", self._toggle_play)
        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setDecimals(1)
        self.speed_spin.setMinimum(0.1)
        self.speed_spin.setMaximum(100.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setSuffix("x")
        self.speed_spin.valueChanged.connect(session.set_playback_speed)

        layout.addWidget(self.jump_back_button)
        layout.addWidget(self.step_back_button)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.step_forward_button)
        layout.addWidget(self.jump_forward_button)
        layout.addWidget(self.time_label)
        layout.addWidget(QtWidgets.QLabel("Speed"))
        layout.addWidget(self.speed_spin)
        layout.addWidget(self.play_button)

        self.sync_from_state()

    def sync_from_state(self):
        block = self.slider.blockSignals(True)
        self.slider.setMaximum(max(len(self.session.adapter.time) - 1, 0))
        self.slider.setValue(self.session.state.time_index)
        self.slider.blockSignals(block)
        self.time_label.setText(f"{self.session.current_time():.3f} s")
        self._pending_value = self.session.state.time_index
        self.play_button.setText("❚❚ Pause" if self.session.state.is_playing else "▶ Play")
        block = self.speed_spin.blockSignals(True)
        self.speed_spin.setValue(float(self.session.state.playback_speed))
        self.speed_spin.blockSignals(block)

    def _queue_time_index(self, value):
        self._pending_value = int(value)
        time = self.session.adapter.time
        if len(time) > 0:
            idx = max(0, min(self._pending_value, len(time) - 1))
            self.time_label.setText(f"{float(time[idx]):.3f} s")
        self._dispatch_timer.start()

    def _flush_pending_value(self):
        if self._dispatch_timer.isActive():
            self._dispatch_timer.stop()
        if int(self._pending_value) != int(self.session.state.time_index):
            self.session.set_time_index(self._pending_value)

    def _toggle_play(self):
        self.session.toggle_playing()
        if self.on_play_toggled is not None:
            self.on_play_toggled(self.session.state.is_playing)

    @staticmethod
    def _make_button(text, callback):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        button.setAutoDefault(False)
        return button


class StatusChipBar(QtWidgets.QWidget):
    """Compact status presentation using pill-like labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._chips = []
        for _ in range(5):
            label = QtWidgets.QLabel()
            label.setStyleSheet(
                "QLabel {"
                "padding: 2px 8px;"
                "border-radius: 9px;"
                "background: #e7eaf0;"
                "color: #202630;"
                "}"
            )
            layout.addWidget(label)
            self._chips.append(label)
        layout.addStretch(1)

    def update_values(self, values):
        for label, text in zip(self._chips, values):
            label.setText(str(text))
