"""Right-side panels for viewer properties, traces, spectra, and Arias intensity."""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .colors import COLORMAP_OPTIONS

_, _, _, _, _, QtWidgets = require_viewer_dependencies()

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:  # pragma: no cover - compatibility fallback
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


COMPONENT_COLORS = {
    "Z": "tab:blue",
    "E": "tab:orange",
    "N": "tab:green",
    "z": "tab:blue",
    "e": "tab:orange",
    "n": "tab:green",
}


class TracePanel(QtWidgets.QWidget):
    """Embedded matplotlib panel showing node traces."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self._trace_lines = []
        self._time_cursors = []
        self._current_key = None

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)

        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time" and self._current_key is not None:
            self._update_time_cursor()
            self.canvas.draw_idle()
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_key = None
            self._trace_lines = []
            self._time_cursors = []
            for ax in self.axes:
                ax.clear()
            self.title_label.setText("No node selected")
            for ax, label in zip(self.axes, ("Z", "E", "N")):
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.25)
            self.axes[-1].set_xlabel("Time [s]")
            self.canvas.draw_idle()
            return

        trace = self.session.current_trace()
        time = self.session.adapter.time
        labels = ("Z", "E", "N")
        node_key = (node_id, self.session.state.demand)

        if self._current_key != node_key:
            self._current_key = node_key
            self._trace_lines = []
            self._time_cursors = []
            for ax in self.axes:
                ax.clear()

            self.title_label.setText(
                f"Node {node_id} | {self.session.state.demand} traces"
            )
            for ax, values, label in zip(self.axes, trace, labels):
                line, = ax.plot(
                    time,
                    values,
                    linewidth=1.2,
                    color=COMPONENT_COLORS[label],
                    label=label,
                )
                cursor = ax.axvline(self.session.current_time(), color="tab:red", alpha=0.35)
                self._trace_lines.append(line)
                self._time_cursors.append(cursor)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="upper right")
            self.axes[-1].set_xlabel("Time [s]")
        else:
            self.title_label.setText(
                f"Node {node_id} | {self.session.state.demand} traces"
            )
            for line, values in zip(self._trace_lines, trace):
                line.set_ydata(values)

        self._update_time_cursor()
        self.canvas.draw_idle()

    def _update_time_cursor(self):
        current_time = self.session.current_time()
        for cursor in self._time_cursors:
            cursor.set_xdata([current_time, current_time])


class SpectrumPanel(QtWidgets.QWidget):
    """Panel that computes and shows Newmark PSa on demand."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)

        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time":
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_node = None
            for ax in self.axes:
                ax.clear()
                ax.grid(True, alpha=0.25)
            self.title_label.setText("No node selected")
            self.axes[-1].set_xlabel("Period [s]")
            self.canvas.draw_idle()
            return

        if self._current_node == node_id and reason not in {"selection", "full", "init"}:
            return

        self._current_node = node_id
        for ax in self.axes:
            ax.clear()

        try:
            spectrum = self.session.current_spectrum()
        except Exception as exc:  # pragma: no cover - runtime-only when deps are missing
            self.title_label.setText(f"Spectrum unavailable: {exc}")
            for ax in self.axes:
                ax.grid(True, alpha=0.25)
            self.canvas.draw_idle()
            return

        self.title_label.setText(f"Node {node_id} | Newmark PSa")
        for ax, label in zip(self.axes, ("z", "e", "n")):
            ax.plot(
                spectrum["T"],
                spectrum[f"PSa_{label}"],
                linewidth=1.2,
                color=COMPONENT_COLORS[label],
                label=label.upper(),
            )
            ax.set_ylabel(label.upper())
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
        self.axes[-1].set_xlabel("Period [s]")
        self.canvas.draw_idle()


class AriasIntensityPanel(QtWidgets.QWidget):
    """Panel showing Arias intensity curves for the selected node."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time":
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_node = None
            for ax in self.axes:
                ax.clear()
                ax.grid(True, alpha=0.25)
            self.title_label.setText("No node selected")
            self.axes[-1].set_xlabel("Time [s]")
            self.canvas.draw_idle()
            return

        if self._current_node == node_id and reason not in {"selection", "full", "init"}:
            return

        self._current_node = node_id
        for ax in self.axes:
            ax.clear()

        try:
            arias = self.session.current_arias()
        except Exception as exc:  # pragma: no cover - optional dependency/runtime
            self.title_label.setText(f"Arias unavailable: {exc}")
            for ax in self.axes:
                ax.grid(True, alpha=0.25)
            self.canvas.draw_idle()
            return

        time = arias["time"]
        self.title_label.setText(f"Node {node_id} | Arias Intensity")
        for ax, label in zip(self.axes, ("z", "e", "n")):
            item = arias["components"][label]
            ax.plot(
                time,
                item["IA_pct"],
                linewidth=1.2,
                color=COMPONENT_COLORS[label],
                label=f"{label.upper()} | Ia={item['ia_total']:.3f} m/s",
            )
            ax.axvline(item["t_start"], color=COMPONENT_COLORS[label], linestyle="--", linewidth=1, alpha=0.45)
            ax.axvline(item["t_end"], color=COMPONENT_COLORS[label], linestyle="--", linewidth=1, alpha=0.45)
            ax.axhline(5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.axhline(95, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_ylabel(label.upper())
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left")
        self.axes[-1].set_xlabel("Time [s]")
        self.canvas.draw_idle()


class ViewerPropertiesPanel(QtWidgets.QWidget):
    """Upper-right properties and controls panel."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._syncing = False
        self._dirty_sections = {
            "data": False,
            "color": False,
            "visibility": False,
            "warp": False,
        }

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        form_layout = QtWidgets.QVBoxLayout(content)
        form_layout.setContentsMargins(4, 4, 4, 4)

        form_layout.addWidget(self._build_node_box())
        form_layout.addWidget(self._build_data_box())
        form_layout.addWidget(self._build_color_box())
        form_layout.addWidget(self._build_visibility_box())
        form_layout.addWidget(self._build_warp_box())
        form_layout.addStretch(1)

        layout.addWidget(scroll, 1)
        self.refresh("init")

    def _build_node_box(self):
        box = QtWidgets.QGroupBox("Selected Node / Coordinate Search")
        layout = QtWidgets.QFormLayout(box)

        self.node_id_spin = QtWidgets.QSpinBox()
        self.node_id_spin.setMinimum(0)
        self.node_id_spin.setMaximum(max(len(self.session.adapter.node_ids) - 1, 0))
        self.node_select_button = QtWidgets.QPushButton("Select node")
        self.node_select_button.clicked.connect(self._select_node_id)

        node_row = QtWidgets.QWidget()
        node_row_layout = QtWidgets.QHBoxLayout(node_row)
        node_row_layout.setContentsMargins(0, 0, 0, 0)
        node_row_layout.addWidget(self.node_id_spin, 1)
        node_row_layout.addWidget(self.node_select_button)

        self.x_spin = self._coord_spin()
        self.y_spin = self._coord_spin()
        self.z_spin = self._coord_spin()
        self.find_nearest_button = QtWidgets.QPushButton("Find nearest")
        self.find_nearest_button.clicked.connect(self._find_nearest)
        self.nearest_label = QtWidgets.QLabel("-")

        self.node_label = QtWidgets.QLabel("None")
        self.type_label = QtWidgets.QLabel("-")
        self.xyz_model_label = QtWidgets.QLabel("-")
        self.xyz_view_label = QtWidgets.QLabel("-")
        self.gf_slot_label = QtWidgets.QLabel("-")

        layout.addRow("Node ID", node_row)
        layout.addRow("X [m]", self.x_spin)
        layout.addRow("Y [m]", self.y_spin)
        layout.addRow("Z [m]", self.z_spin)
        layout.addRow("", self.find_nearest_button)
        layout.addRow("Nearest", self.nearest_label)
        layout.addRow("Selected", self.node_label)
        layout.addRow("Type", self.type_label)
        layout.addRow("XYZ model [m]", self.xyz_model_label)
        layout.addRow("XYZ view [m]", self.xyz_view_label)
        layout.addRow("GF slot s0", self.gf_slot_label)
        return box

    def _build_data_box(self):
        box = QtWidgets.QGroupBox("Data")
        outer = QtWidgets.QVBoxLayout(box)
        layout = QtWidgets.QFormLayout()

        self.demand_combo = QtWidgets.QComboBox()
        for demand in self.session.adapter.available_demands:
            self.demand_combo.addItem(demand, demand)
        self.demand_combo.currentTextChanged.connect(
            lambda _value: self._on_data_changed("data")
        )

        self.component_combo = QtWidgets.QComboBox()
        for component in self.session.adapter.available_components:
            self.component_combo.addItem(component, component)
        self.component_combo.currentTextChanged.connect(
            lambda _value: self._on_data_changed("data")
        )

        layout.addRow("Demand", self.demand_combo)
        layout.addRow("Component", self.component_combo)
        outer.addLayout(layout)
        outer.addWidget(self._make_section_apply_button("data", self._apply_data_settings))
        return box

    def _build_color_box(self):
        box = QtWidgets.QGroupBox("Color Map")
        outer = QtWidgets.QVBoxLayout(box)
        layout = QtWidgets.QFormLayout()

        self.colormap_combo = QtWidgets.QComboBox()
        for cmap in COLORMAP_OPTIONS:
            self.colormap_combo.addItem(cmap, cmap)
        self.colormap_combo.currentTextChanged.connect(
            lambda *_args: self._mark_dirty("color")
        )

        self.vmin_spin = self._value_spin()
        self.vmax_spin = self._value_spin()
        self.vmin_spin.valueChanged.connect(lambda *_args: self._mark_dirty("color"))
        self.vmax_spin.valueChanged.connect(lambda *_args: self._mark_dirty("color"))

        self.clamp_checkbox = QtWidgets.QCheckBox("Clamp")
        self.clamp_checkbox.toggled.connect(lambda *_args: self._mark_dirty("color"))

        self.reset_range_button = QtWidgets.QPushButton("Reset to auto")
        self.reset_range_button.clicked.connect(self._reset_range)
        self.auto_range_label = QtWidgets.QLabel("-")

        layout.addRow("Preset", self.colormap_combo)
        layout.addRow("Auto range", self.auto_range_label)
        layout.addRow("User min", self.vmin_spin)
        layout.addRow("User max", self.vmax_spin)
        layout.addRow("", self.clamp_checkbox)
        layout.addRow("", self.reset_range_button)
        outer.addLayout(layout)
        outer.addWidget(self._make_section_apply_button("color", self._apply_color_settings))
        return box

    def _build_visibility_box(self):
        box = QtWidgets.QGroupBox("Visualization")
        layout = QtWidgets.QVBoxLayout(box)

        self.show_internal_checkbox = QtWidgets.QCheckBox("Show internal nodes")
        self.show_external_checkbox = QtWidgets.QCheckBox("Show external nodes")
        self.show_qa_checkbox = QtWidgets.QCheckBox("Show QA")
        self.show_internal_checkbox.toggled.connect(lambda *_args: self._mark_dirty("visibility"))
        self.show_external_checkbox.toggled.connect(lambda *_args: self._mark_dirty("visibility"))
        self.show_qa_checkbox.toggled.connect(lambda *_args: self._mark_dirty("visibility"))

        layout.addWidget(self.show_internal_checkbox)
        layout.addWidget(self.show_external_checkbox)
        layout.addWidget(self.show_qa_checkbox)
        layout.addWidget(self._make_section_apply_button("visibility", self._apply_visibility_settings))
        return box

    def _build_warp_box(self):
        """Build the 3-D Warp / Real Motion controls section."""
        box = QtWidgets.QGroupBox("3D Warp — Real Motion")
        outer = QtWidgets.QVBoxLayout(box)
        outer.setSpacing(4)

        # ── Enable toggle ──────────────────────────────────────────────
        self.warp_enabled_checkbox = QtWidgets.QCheckBox("Enable displacement warp")
        self.warp_enabled_checkbox.toggled.connect(lambda *_args: self._mark_dirty("warp"))
        outer.addWidget(self.warp_enabled_checkbox)

        # ── Axis toggles ───────────────────────────────────────────────
        axes_widget = QtWidgets.QWidget()
        axes_layout = QtWidgets.QHBoxLayout(axes_widget)
        axes_layout.setContentsMargins(0, 0, 0, 0)
        axes_layout.setSpacing(6)
        axes_layout.addWidget(QtWidgets.QLabel("Axes:"))
        self.warp_x_checkbox = QtWidgets.QCheckBox("X (E)")
        self.warp_y_checkbox = QtWidgets.QCheckBox("Y (N)")
        self.warp_z_checkbox = QtWidgets.QCheckBox("Z")
        for cb in (self.warp_x_checkbox, self.warp_y_checkbox, self.warp_z_checkbox):
            cb.setChecked(True)
            cb.toggled.connect(lambda *_args: self._mark_dirty("warp"))
            axes_layout.addWidget(cb)
        axes_layout.addStretch()
        outer.addWidget(axes_widget)

        # ── Scale row: presets + spin + Auto button ────────────────────
        scale_widget = QtWidgets.QWidget()
        scale_layout = QtWidgets.QHBoxLayout(scale_widget)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(4)

        scale_layout.addWidget(QtWidgets.QLabel("Scale:"))

        for label, value in (("×10", 10), ("×100", 100), ("×1k", 1_000), ("×10k", 10_000)):
            btn = QtWidgets.QPushButton(label)
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _checked, v=value: self._apply_warp_scale_preset(v))
            scale_layout.addWidget(btn)

        self.warp_scale_spin = QtWidgets.QDoubleSpinBox()
        self.warp_scale_spin.setDecimals(1)
        self.warp_scale_spin.setRange(1.0, 1_000_000.0)
        self.warp_scale_spin.setSingleStep(10.0)
        self.warp_scale_spin.setValue(100.0)
        self.warp_scale_spin.setSuffix("×")
        self.warp_scale_spin.valueChanged.connect(lambda *_args: self._mark_dirty("warp"))
        scale_layout.addWidget(self.warp_scale_spin, 1)

        self.warp_auto_button = QtWidgets.QPushButton("Auto")
        self.warp_auto_button.setToolTip(
            "Suggest a scale so peak displacement fills ~5 % of the domain."
        )
        self.warp_auto_button.clicked.connect(self._suggest_warp_scale)
        scale_layout.addWidget(self.warp_auto_button)

        outer.addWidget(scale_widget)
        outer.addWidget(self._make_section_apply_button("warp", self._apply_warp_settings))
        return box

    def refresh(self, reason: str = "full"):
        self._syncing = True
        try:
            if reason in {"init", "full", "demand"} or not self._dirty_sections["data"]:
                self._sync_data_from_session()
            if reason in {"init", "full", "appearance", "color_range", "demand"} or not self._dirty_sections["color"]:
                self._sync_color_from_session()
            if reason in {"init", "full", "visibility"} or not self._dirty_sections["visibility"]:
                self._sync_visibility_from_session()
            if reason in {"init", "full", "warp"} or not self._dirty_sections["warp"]:
                self._sync_warp_from_session()

            info = self.session.current_node_info()
            if info is None:
                self.node_label.setText("None")
                self.type_label.setText("-")
                self.xyz_model_label.setText("-")
                self.xyz_view_label.setText("-")
                self.gf_slot_label.setText("-")
                return

            node_id = info["node_id"]
            if node_id not in ("QA", "qa"):
                self.node_id_spin.setValue(int(node_id))
            xyz_model = info.get("xyz_model_m")
            xyz_view = info.get("xyz_m")
            self.node_label.setText(str(node_id))
            self.type_label.setText(str(info["type"]))
            self.xyz_model_label.setText(self._format_xyz(xyz_model))
            self.xyz_view_label.setText(self._format_xyz(xyz_view))
            self.gf_slot_label.setText("-" if info["gf_slot_s0"] is None else str(info["gf_slot_s0"]))
            if xyz_model is not None:
                self.x_spin.setValue(float(xyz_model[0]))
                self.y_spin.setValue(float(xyz_model[1]))
                self.z_spin.setValue(float(xyz_model[2]))
        finally:
            self._syncing = False
            self._update_apply_buttons()

    def _select_node_id(self):
        if self._syncing:
            return
        self.session.select_node(int(self.node_id_spin.value()))

    def _find_nearest(self):
        if self._syncing:
            return
        node_id, distance_m = self.session.select_nearest_coordinate_m(
            self.x_spin.value(),
            self.y_spin.value(),
            self.z_spin.value(),
        )
        self.nearest_label.setText(f"{node_id} | dist={distance_m:.3f} m")

    def _on_data_changed(self, section: str):
        if self._syncing:
            return
        self._update_auto_range_label()
        self._mark_dirty(section)

    def _mark_dirty(self, section: str, *_args):
        if self._syncing:
            return
        self._dirty_sections[section] = True
        self._update_apply_buttons()

    def _reset_range(self):
        auto_vmin, auto_vmax = self._draft_default_color_limits()
        block_vmin = self.vmin_spin.blockSignals(True)
        block_vmax = self.vmax_spin.blockSignals(True)
        block_clamp = self.clamp_checkbox.blockSignals(True)
        self.vmin_spin.setValue(auto_vmin)
        self.vmax_spin.setValue(auto_vmax)
        self.clamp_checkbox.setChecked(False)
        self.vmin_spin.blockSignals(block_vmin)
        self.vmax_spin.blockSignals(block_vmax)
        self.clamp_checkbox.blockSignals(block_clamp)
        self._mark_dirty("color")

    # ── 3-D Warp callbacks ────────────────────────────────────────────────

    def _apply_warp_scale_preset(self, value: float):
        if self._syncing:
            return
        block = self.warp_scale_spin.blockSignals(True)
        self.warp_scale_spin.setValue(float(value))
        self.warp_scale_spin.blockSignals(block)
        self._mark_dirty("warp")

    def _suggest_warp_scale(self):
        scale = self.session.suggested_warp_scale()
        block = self.warp_scale_spin.blockSignals(True)
        self.warp_scale_spin.setValue(scale)
        self.warp_scale_spin.blockSignals(block)
        self._mark_dirty("warp")

    def _apply_data_settings(self):
        if self._syncing:
            return
        self.session.apply_data_settings(
            demand=self.demand_combo.currentText(),
            component=self.component_combo.currentText(),
        )
        self._dirty_sections["data"] = False
        self._update_apply_buttons()

    def _apply_color_settings(self):
        if self._syncing:
            return
        self.session.apply_color_settings(
            colormap=self.colormap_combo.currentText(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            clamp_enabled=self.clamp_checkbox.isChecked(),
        )
        self._dirty_sections["color"] = False
        self._update_apply_buttons()

    def _apply_visibility_settings(self):
        if self._syncing:
            return
        self.session.apply_visibility_settings(
            show_internal=self.show_internal_checkbox.isChecked(),
            show_external=self.show_external_checkbox.isChecked(),
            show_qa=self.show_qa_checkbox.isChecked(),
        )
        self._dirty_sections["visibility"] = False
        self._update_apply_buttons()

    def _apply_warp_settings(self):
        if self._syncing:
            return
        self.session.apply_warp_settings(
            warp_enabled=self.warp_enabled_checkbox.isChecked(),
            warp_axes=(
                self.warp_x_checkbox.isChecked(),
                self.warp_y_checkbox.isChecked(),
                self.warp_z_checkbox.isChecked(),
            ),
            warp_scale=self.warp_scale_spin.value(),
        )
        self._dirty_sections["warp"] = False
        self._update_apply_buttons()

    def _sync_data_from_session(self):
        self._set_combo_value(self.demand_combo, self.session.state.demand)
        self._set_combo_value(self.component_combo, self.session.state.component)
        self._dirty_sections["data"] = False

    def _sync_color_from_session(self):
        self._set_combo_value(self.colormap_combo, self.session.current_colormap())
        auto_vmin, auto_vmax = self._draft_default_color_limits()
        self.auto_range_label.setText(f"{auto_vmin:.4g}  /  {auto_vmax:.4g}")
        if self.session.state.user_vmin is None or self.session.state.user_vmax is None:
            self.session.state.set_user_color_range(auto_vmin, auto_vmax)
        self.vmin_spin.setValue(float(self.session.state.user_vmin))
        self.vmax_spin.setValue(float(self.session.state.user_vmax))
        self.clamp_checkbox.setChecked(self.session.state.clamp_enabled)
        self._dirty_sections["color"] = False

    def _sync_visibility_from_session(self):
        self.show_internal_checkbox.setChecked(self.session.state.show_internal)
        self.show_external_checkbox.setChecked(self.session.state.show_external)
        self.show_qa_checkbox.setChecked(self.session.state.show_qa)
        self._dirty_sections["visibility"] = False

    def _sync_warp_from_session(self):
        self.warp_enabled_checkbox.setChecked(self.session.state.disp_warp_enabled)
        axes = self.session.state.warp_axes
        self.warp_x_checkbox.setChecked(axes[0])
        self.warp_y_checkbox.setChecked(axes[1])
        self.warp_z_checkbox.setChecked(axes[2])
        self.warp_scale_spin.setValue(
            float(self.session.state.warp_scale)
            if self.session.state.warp_scale is not None
            else float(self.session.suggested_warp_scale())
        )
        self._dirty_sections["warp"] = False

    def _draft_default_color_limits(self):
        demand = self.demand_combo.currentText() or self.session.state.demand
        component = self.component_combo.currentText() or self.session.state.component
        return self.session.adapter.default_scalar_limits(demand, component)

    def _update_auto_range_label(self):
        auto_vmin, auto_vmax = self._draft_default_color_limits()
        self.auto_range_label.setText(f"{auto_vmin:.4g}  /  {auto_vmax:.4g}")

    def _make_section_apply_button(self, section: str, callback):
        button = QtWidgets.QPushButton("Apply")
        button.setEnabled(False)
        button.clicked.connect(callback)
        if not hasattr(self, "_apply_buttons"):
            self._apply_buttons = {}
        self._apply_buttons[section] = button
        return button

    def _update_apply_buttons(self):
        for section, button in getattr(self, "_apply_buttons", {}).items():
            button.setEnabled(bool(self._dirty_sections.get(section, False)))

    @staticmethod
    def _coord_spin():
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(-1.0e12, 1.0e12)
        spin.setSingleStep(1.0)
        return spin

    @staticmethod
    def _value_spin():
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(8)
        spin.setRange(-1.0e20, 1.0e20)
        spin.setSingleStep(0.1)
        return spin

    @staticmethod
    def _format_xyz(xyz):
        if xyz is None:
            return "-"
        return f"{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}"

    @staticmethod
    def _set_combo_value(combo, value):
        block = combo.blockSignals(True)
        combo.setCurrentText(str(value))
        combo.blockSignals(block)


# Backwards-compatible alias. Older window.py versions imported InfoPanel.
InfoPanel = ViewerPropertiesPanel
