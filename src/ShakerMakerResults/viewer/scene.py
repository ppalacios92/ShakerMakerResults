"""PyVista scene management for the interactive viewer."""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .interaction import RevitInteractorStyle

pv, _, vtk, _, _, _ = require_viewer_dependencies()


class ViewerScene:
    """Build and refresh the 3D PyVista representation."""

    def __init__(self, plotter, session):
        self.plotter = plotter
        self.session = session
        self.point_cloud = None
        self.point_actor = None
        self.selection_actor = None
        self._picker = vtk.vtkPointPicker()
        self._picker.SetTolerance(0.02)
        self._interactor_style = None

    def build(self):
        self._rebuild_point_cloud()
        self.plotter.set_background(self.session.current_background_color())
        self.point_actor = self._add_point_actor()
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self._install_picking()
        self.refresh_selection(render=False)
        self._add_branding()

    def refresh_scalars(self, render: bool = True):
        scalars = self.session.current_visible_scalars()
        if self.point_cloud is None or len(scalars) != self.point_cloud.n_points:
            self.rebuild_scalar_actor(render=render)
            return
        self.point_cloud.point_data["active_scalars"] = scalars
        self.point_cloud.Modified()
        if self.point_actor is not None:
            # Skip color-range recalculation during playback — the range was
            # fixed when the actor was built, so every frame is free.
            if not self.session.state.is_playing:
                self.point_actor.mapper.scalar_range = self.session.current_color_limits(scalars)
        # Update point geometry for warp mode — pure NumPy when cache is warm.
        if self.session.state.disp_warp_enabled:
            self.refresh_geometry(render=False)
        if render:
            self.plotter.render()

    def refresh_geometry(self, render: bool = True):
        """Move each visible point by its displacement field × warp scale.

        Called every animation frame when 3-D warp is active.  The update is
        O(N) NumPy arithmetic with no I/O when the displacement series is
        pre-warmed in the cache.
        """
        if not self.session.state.disp_warp_enabled:
            if render:
                self.plotter.render()
            return
        try:
            new_pts = self.session.current_warped_points()
            if self.point_cloud is not None and len(new_pts) == self.point_cloud.n_points:
                self.point_cloud.points = new_pts
                self.point_cloud.Modified()
        except Exception:
            pass
        if render:
            self.plotter.render()

    def rebuild_for_visibility(self, render: bool = True):
        if self.point_actor is not None:
            self.plotter.remove_actor(self.point_actor, render=False)
            self.point_actor = None
        self._rebuild_point_cloud()
        self.point_actor = self._add_point_actor()
        self.refresh_selection(render=False)
        if render:
            self.plotter.render()

    def rebuild_scalar_actor(self, render: bool = True):
        self._rebuild_point_cloud()
        if self.point_actor is not None:
            self.plotter.remove_actor(self.point_actor, render=False)
        self.point_actor = self._add_point_actor()
        if render:
            self.plotter.render()

    def refresh_selection(self, render: bool = True):
        if self.selection_actor is not None:
            self.plotter.remove_actor(self.selection_actor, render=False)
            self.selection_actor = None

        node_id = self.session.state.selected_node
        if node_id is None:
            if render:
                self.plotter.render()
            return

        point = self.session.adapter.point_for_node(node_id)
        selected = pv.PolyData(point[None, :])
        self.selection_actor = self.plotter.add_points(
            selected,
            color="yellow",
            point_size=max(self._point_size() + 8, 18),
            render_points_as_spheres=True,
            render=False,
        )
        if render:
            self.plotter.render()

    def _install_picking(self):
        vtk_interactor = self._vtk_interactor()
        if vtk_interactor is None:
            return
        self._interactor_style = RevitInteractorStyle(
            self.plotter,
            self._picker,
            on_point_picked=self._handle_point_pick,
            on_point_double_clicked=self._handle_point_double_click,
        )
        vtk_interactor.SetInteractorStyle(self._interactor_style)

    def _handle_point_pick(self, point_id, _pick_pos=None):
        if point_id >= 0:
            node_id = self.session.adapter.node_id_from_visible_index(point_id)
            self.session.select_node(node_id)

    def _handle_point_double_click(self, point_id, _pick_pos=None):
        if point_id >= 0:
            node_id = self.session.adapter.node_id_from_visible_index(point_id)
            self.session.select_node(node_id)
            self.center_on_node(node_id)

    def apply_appearance(self, render: bool = True):
        self.plotter.set_background(self.session.current_background_color())
        self.rebuild_scalar_actor(render=render)
        self.refresh_selection(render=False)

    def apply_color_range(self, render: bool = True):
        scalars = self.session.current_visible_scalars()
        if self.point_actor is not None:
            self.point_actor.mapper.scalar_range = self.session.current_color_limits(scalars)
        if render:
            self.plotter.render()

    def center_on_node(self, node_id, render: bool = True):
        point = self.session.adapter.point_for_node(node_id)
        camera = getattr(self.plotter, "camera", None)
        if camera is None:
            return

        focal = camera.GetFocalPoint()
        position = camera.GetPosition()
        direction = [position[i] - focal[i] for i in range(3)]
        scale = 0.35
        new_position = [point[i] + direction[i] * scale for i in range(3)]
        camera.SetFocalPoint(*point)
        camera.SetPosition(*new_position)
        self.plotter.reset_camera_clipping_range()
        if render:
            self.plotter.render()

    def _rebuild_point_cloud(self):
        # Use warped positions when warp is active; falls back to base positions
        # transparently when warp is disabled.
        points = self.session.current_warped_points()
        scalars = self.session.current_visible_scalars()
        self.point_cloud = pv.PolyData(points)
        self.point_cloud.point_data["active_scalars"] = scalars

    def _add_point_actor(self):
        scalars = self.point_cloud.point_data["active_scalars"]
        return self.plotter.add_points(
            self.point_cloud,
            scalars="active_scalars",
            cmap=self.session.current_colormap(),
            clim=self.session.current_color_limits(scalars),
            render_points_as_spheres=True,
            point_size=self._point_size(),
            show_scalar_bar=self.session.state.show_scalar_bar,
            scalar_bar_args={"title": self.session.state.demand},
            render=False,
        )

    def _add_branding(self):
        """Add a small title block to the upper-left corner of the 3-D viewport."""
        lines = [
            "ShakerMaker Results",
            "By: Ladruno Team",
            "2026 - V 1.0.0",
            "An Interactive View for ShakerMaker Tool",
            ".h5drm files supported",
        ]
        try:
            self.plotter.add_text(
                "\n".join(lines),
                position="upper_left",
                font_size=9,
                color="#555555",
                shadow=False,
                name="branding",
                render=False,
            )
        except Exception:
            pass


    def _vtk_interactor(self):
        iren = getattr(self.plotter, "iren", None)
        if iren is None:
            return None
        return getattr(iren, "interactor", iren)

    def _point_size(self) -> float:
        return self.session.suggested_point_size()
