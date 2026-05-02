"""Custom VTK interaction styles for the viewer."""

from __future__ import annotations

from ._imports import require_viewer_dependencies

_, _, vtk, QtCore, _, QtWidgets = require_viewer_dependencies()


class RevitInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera variant with Revit-like navigation.

    Interaction map
    ---------------
    - Left click: pick/select a node.
    - Left drag: rubber-band area selection.
    - Mouse wheel and Ctrl + mouse wheel: zoom toward the cursor.
    - Right mouse drag: default VTK dolly/zoom.
    - Middle mouse drag: pan.
    - Shift + middle mouse drag: orbit.
    - Esc: clear current selection.
    """

    def __init__(
        self,
        plotter,
        picker,
        on_point_picked=None,
        on_point_double_clicked=None,
        on_area_selected=None,
        on_clear_selection=None,
    ):
        super().__init__()
        self._plotter = plotter
        self._picker = picker
        self._on_point_picked = on_point_picked
        self._on_point_double_clicked = on_point_double_clicked
        self._on_area_selected = on_area_selected
        self._on_clear_selection = on_clear_selection
        self._sel_drag_start = None
        self._sel_dragging = False
        self._rubber_band = None
        self._shift_orbiting = False

        self.AddObserver("LeftButtonPressEvent", self._on_left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self._on_left_button_release)
        self.AddObserver("MiddleButtonPressEvent", self._on_middle_button_press)
        self.AddObserver("MiddleButtonReleaseEvent", self._on_middle_button_release)
        self.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.AddObserver("LeftButtonDoubleClickEvent", self._on_left_button_double_click)
        self.AddObserver("MouseWheelForwardEvent", self._on_mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._on_mouse_wheel_backward)
        self.AddObserver("KeyPressEvent", self._on_key_press)

    def _on_left_button_press(self, _obj, _event):
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        self._sel_drag_start = (x, y)
        self._sel_dragging = False
        self._show_rubber_band((x, y), (x, y))

    def _on_left_button_release(self, _obj, _event):
        if self._sel_drag_start is None:
            return

        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        if self._sel_dragging:
            if self._on_area_selected is not None:
                self._on_area_selected(self._sel_drag_start, (x, y))
        else:
            point_id, pick_pos = self._pick_from_event_position(interactor)
            if point_id >= 0 and self._on_point_picked is not None:
                self._on_point_picked(point_id, pick_pos)

        self._clear_rubber_band()
        self._sel_drag_start = None
        self._sel_dragging = False

    def _on_mouse_move(self, _obj, _event):
        if self._sel_drag_start is not None:
            interactor = self.GetInteractor()
            x, y = interactor.GetEventPosition()
            if abs(x - self._sel_drag_start[0]) > 4 or abs(y - self._sel_drag_start[1]) > 4:
                self._sel_dragging = True
                self._show_rubber_band(self._sel_drag_start, (x, y))
            return
        self.OnMouseMove()

    def _on_middle_button_press(self, _obj, _event):
        interactor = self.GetInteractor()
        self._shift_orbiting = bool(interactor.GetShiftKey())
        if self._shift_orbiting:
            self.OnLeftButtonDown()
        else:
            self.OnMiddleButtonDown()

    def _on_middle_button_release(self, _obj, _event):
        if self._shift_orbiting:
            self.OnLeftButtonUp()
            self._shift_orbiting = False
        else:
            self.OnMiddleButtonUp()

    def _on_key_press(self, _obj, _event):
        interactor = self.GetInteractor()
        key = str(interactor.GetKeySym() or "").lower()
        if key in {"escape", "esc"} and callable(self._on_clear_selection):
            self._on_clear_selection()

    def _interactor_widget(self):
        widget = getattr(self._plotter, "interactor", None)
        return widget if widget is not None else getattr(self._plotter, "iren", None)

    def _to_qt_point(self, pos):
        widget = self._interactor_widget()
        x, y = pos
        if widget is not None:
            try:
                y = widget.height() - int(y)
            except Exception:
                pass
        return QtCore.QPoint(int(x), int(y))

    def _show_rubber_band(self, start, end):
        widget = self._interactor_widget()
        if widget is None:
            return
        if self._rubber_band is None:
            self._rubber_band = QtWidgets.QRubberBand(
                QtWidgets.QRubberBand.Rectangle,
                widget,
            )
            self._rubber_band.setStyleSheet(
                "QRubberBand {"
                "border: 1px solid rgba(42, 107, 194, 210);"
                "background-color: rgba(42, 107, 194, 128);"
                "}"
            )
        rect = QtCore.QRect(self._to_qt_point(start), self._to_qt_point(end)).normalized()
        self._rubber_band.setGeometry(rect)
        self._rubber_band.show()

    def _clear_rubber_band(self):
        if self._rubber_band is None:
            return
        try:
            self._rubber_band.hide()
            self._rubber_band.deleteLater()
        except Exception:
            pass
        self._rubber_band = None

    def _on_mouse_wheel_forward(self, _obj, _event):
        self._focus_under_cursor()
        self.OnMouseWheelForward()

    def _on_mouse_wheel_backward(self, _obj, _event):
        self._focus_under_cursor()
        self.OnMouseWheelBackward()

    def _on_left_button_double_click(self, _obj, _event):
        interactor = self.GetInteractor()
        point_id, pick_pos = self._pick_from_event_position(interactor)
        if point_id >= 0 and self._on_point_double_clicked is not None:
            self._on_point_double_clicked(point_id, pick_pos)

    def _focus_under_cursor(self):
        interactor = self.GetInteractor()
        point_id, pick_pos = self._pick_from_event_position(interactor)
        if point_id >= 0:
            self._set_focal_point(pick_pos)

    def _pick_from_event_position(self, interactor):
        renderer = getattr(self._plotter, "renderer", None)
        if renderer is None:
            return -1, None

        x, y = interactor.GetEventPosition()
        self._picker.Pick(float(x), float(y), 0.0, renderer)
        point_id = int(self._picker.GetPointId())
        if point_id < 0:
            return -1, None

        return point_id, self._picker.GetPickPosition()

    def _set_focal_point(self, pick_pos):
        camera = getattr(self._plotter, "camera", None)
        if camera is None or pick_pos is None:
            return

        camera.SetFocalPoint(*pick_pos)
        self._plotter.render()
