"""Custom VTK interaction styles for the viewer."""

from __future__ import annotations

from ._imports import require_viewer_dependencies

_, _, vtk, _, _, _ = require_viewer_dependencies()


class RevitInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera variant with Revit-like focus behavior.

    Features
    --------
    - Shift + left click: orbit around the picked point by updating the
      camera focal point before delegating to the default rotation logic.
    - Left click: keep node picking enabled through a callback.
    - Mouse wheel: zoom toward the point under the cursor when possible.
    - Middle mouse: preserved from ``vtkInteractorStyleTrackballCamera``
      for pan behavior.
    """

    def __init__(
        self,
        plotter,
        picker,
        on_point_picked=None,
        on_point_double_clicked=None,
    ):
        super().__init__()
        self._plotter = plotter
        self._picker = picker
        self._on_point_picked = on_point_picked
        self._on_point_double_clicked = on_point_double_clicked

        self.AddObserver("LeftButtonPressEvent", self._on_left_button_press)
        self.AddObserver("LeftButtonDoubleClickEvent", self._on_left_button_double_click)
        self.AddObserver("MouseWheelForwardEvent", self._on_mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._on_mouse_wheel_backward)

    def _on_left_button_press(self, _obj, _event):
        interactor = self.GetInteractor()
        point_id, pick_pos = self._pick_from_event_position(interactor)
        if point_id >= 0 and self._on_point_picked is not None:
            self._on_point_picked(point_id, pick_pos)

        if interactor.GetShiftKey() and point_id >= 0:
            self._set_focal_point(pick_pos)

        self.OnLeftButtonDown()

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
