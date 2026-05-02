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
        self.multi_selection_actor = None   # red spheres for selection-mode set
        self.station_actor = None
        self.station_label_actor = None
        self.show_station_tags = True
        self._picker = vtk.vtkPointPicker()
        self._picker.SetTolerance(0.02)
        self._interactor_style = None
        self._selection_visibility = "all"
        self._domain_node_ids = None
        # When the visual filter is active, this maps rendered point_id to node_id.
        self._visual_node_ids_list = None
        # Per-pane GF component pin used by the GF 3x3 layout.
        # When set and session demand is GF, this pane renders *this* component
        # instead of session.state.component, giving each pane its own tensor
        # component independently of the global side-panel selection.
        self._gf_component_pin: str | None = None
        self._gf_label_actor = None    # text actor showing the component label

    # GF component pin.

    def set_gf_component_pin(self, component: str | None, label: str | None = None) -> None:
        """Pin this pane to *component* when demand is GF (GF 3x3 layout).

        Pass ``component=None`` to release the pin and restore normal behaviour.
        *label* is rendered as a text overlay in the upper-right corner of this
        viewport so the user can always see which tensor component is displayed.
        """
        self._gf_component_pin = component
        # Remove the old label actor (if any).
        if self._gf_label_actor is not None:
            try:
                self.plotter.remove_actor(self._gf_label_actor, render=False)
            except Exception:
                pass
            self._gf_label_actor = None
        # Add a new label when a pin is being set.
        if label:
            try:
                self._gf_label_actor = self.plotter.add_text(
                    label,
                    position="upper_right",
                    font_size=7,
                    color="#1565C0",
                    render=False,
                )
            except Exception:
                pass

    def _gf_pin_active(self) -> bool:
        """Return True when this pane should render its own GF component."""
        from .adapter import GF_DEMAND
        return (
            self.session.current_static_color_by() is None
            and
            self._gf_component_pin is not None
            and self.session.state.demand == GF_DEMAND
        )

    def _scalars_for_gf_pin(self):
        """Compute visible scalars using the pinned GF component (no I/O when warm)."""
        comp = self._gf_component_pin
        demand = self.session.state.demand
        sf = int(getattr(self.session, "_display_gf_subfault", 0))
        raw = self.session.adapter.scalar_snapshot(
            self.session.state.time_index, demand, comp, subfault_id=sf
        )
        return self.session.adapter.visible_scalars(
            raw,
            show_internal=self.session.state.show_internal,
            show_external=self.session.state.show_external,
            show_qa=self.session.state.show_qa,
        )

    def _color_limits_for_gf_pin(self, scalars=None) -> tuple:
        """Color limits for the pinned GF component."""
        comp = self._gf_component_pin
        state = self.session.state
        if state.clamp_enabled and state.user_vmin is not None and state.user_vmax is not None:
            vmin = float(state.user_vmin)
            vmax = float(state.user_vmax)
            return (vmin, vmin + 1.0) if vmax <= vmin else (vmin, vmax)
        sf = int(getattr(self.session, "_display_gf_subfault", 0))
        try:
            return self.session.adapter.default_scalar_limits(
                self.session.state.demand, comp, subfault_id=sf
            )
        except Exception:
            pass
        if scalars is None:
            try:
                scalars = self._scalars_for_gf_pin()
            except Exception:
                return (-1.0, 1.0)
        try:
            from .colors import scalar_limits
            return scalar_limits(scalars, comp)
        except Exception:
            return (-1.0, 1.0)

    # Scene build.

    def build(self):
        self._rebuild_point_cloud()
        self.plotter.set_background(self.session.current_background_color())
        self.point_actor = self._add_point_actor()
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self._install_picking()
        self.refresh_selection(render=False)
        self.refresh_multi_selection(render=False)
        self.refresh_station_tags(render=False)
        self._add_branding()

    def refresh_scalars(self, render: bool = True):
        # Hot path: respect the per-pane GF component pin when active.
        if self._gf_pin_active():
            scalars = self._scalars_for_gf_pin()
        else:
            scalars = self.session.current_visible_scalars()
        domain_idx = self._domain_index_array()
        if domain_idx is not None:
            scalars = scalars[domain_idx]
        if self.point_cloud is None or len(scalars) != self.point_cloud.n_points:
            self.rebuild_scalar_actor(render=render)
            return
        self.point_cloud.point_data["active_scalars"] = scalars
        self.point_cloud.Modified()
        if self.point_actor is not None:
            # Skip color-range recalculation during playback; the range was
            # fixed when the actor was built, so every frame is free.
            if not self.session.state.is_playing:
                if self._gf_pin_active():
                    self.point_actor.mapper.scalar_range = self._color_limits_for_gf_pin(scalars)
                else:
                    self.point_actor.mapper.scalar_range = self.session.current_color_limits(scalars)
        # Update point geometry for warp mode; pure NumPy when cache is warm.
        if self.session.state.disp_warp_enabled:
            self.refresh_geometry(render=False)
        if render:
            self.plotter.render()

    def refresh_geometry(self, render: bool = True):
        """Move each visible point by its displacement field times warp scale.

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
            domain_idx = self._domain_index_array()
            if domain_idx is not None:
                new_pts = new_pts[domain_idx]
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
        if node_id is None or not self._domain_contains(node_id):
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
        # Must NOT be pickable; vtkPointPicker would otherwise pick the yellow
        # sphere and feed its point_id (always 0) to node_id_from_visible_index,
        # causing every subsequent click to select node 0.
        try:
            self.selection_actor.SetPickable(0)
        except Exception:
            pass
        if render:
            self.plotter.render()

    def refresh_multi_selection(self, render: bool = True):
        """Render selected nodes (selection mode) as red spheres."""
        if self.multi_selection_actor is not None:
            self.plotter.remove_actor(self.multi_selection_actor, render=False)
            self.multi_selection_actor = None

        state = self.session.state
        if not state.multi_selection:
            if render:
                self.plotter.render()
            return

        import numpy as np
        pts = []
        for node_id in state.multi_selection:
            if not self._domain_contains(node_id):
                continue
            try:
                pts.append(self.session.adapter.point_for_node(node_id))
            except Exception:
                pass
        if not pts:
            if render:
                self.plotter.render()
            return

        sel_cloud = pv.PolyData(np.array(pts, dtype=float))
        self.multi_selection_actor = self.plotter.add_points(
            sel_cloud,
            color="#e53935",
            point_size=max(self._point_size() + 6, 16),
            render_points_as_spheres=True,
            render=False,
        )
        try:
            self.multi_selection_actor.SetPickable(0)
        except Exception:
            pass
        if render:
            self.plotter.render()

    def update_node_opacity(self, render: bool = True):
        """Apply node_opacity to the main point actor without a full rebuild."""
        if self.point_actor is not None:
            try:
                self.point_actor.GetProperty().SetOpacity(
                    self.session.state.node_opacity
                )
            except Exception:
                pass
        if render:
            self.plotter.render()

    def refresh_station_tags(self, render: bool = True):
        if self.station_actor is not None:
            try:
                self.plotter.remove_actor(self.station_actor, render=False)
            except Exception:
                pass
            self.station_actor = None
        if self.station_label_actor is not None:
            try:
                self.plotter.remove_actor(self.station_label_actor, render=False)
            except Exception:
                pass
            self.station_label_actor = None

        stations = self.session.current_station_tags() if self.show_station_tags else []
        if stations:
            pts = [entry["xyz_display_m"] for entry in stations]
            labels = [str(entry["name"]) for entry in stations]
            try:
                station_poly = pv.PolyData(pts)
                self.station_actor = self.plotter.add_points(
                    station_poly,
                    color="#d32f2f",
                    point_size=max(self._point_size() + 3, 12),
                    render_points_as_spheres=True,
                    render=False,
                )
                # Station markers must NOT be pickable; they sit on top of
                # simulation nodes and vtkPointPicker would pick them first,
                # returning a point_id within the station actor (0..M-1) that
                # gets incorrectly mapped to a simulation node index.
                try:
                    self.station_actor.SetPickable(0)
                except Exception:
                    pass
                self.station_label_actor = self.plotter.add_point_labels(
                    station_poly,
                    labels,
                    point_size=0,
                    font_size=10,
                    text_color="#d32f2f",
                    fill_shape=False,
                    always_visible=True,
                    render=False,
                )
            except Exception:
                self.station_actor = None
                self.station_label_actor = None

        if render:
            self.plotter.render()

    def set_station_tags_visible(self, visible: bool, render: bool = True):
        self.show_station_tags = bool(visible)
        self.refresh_station_tags(render=render)

    def _install_picking(self):
        vtk_interactor = self._vtk_interactor()
        if vtk_interactor is None:
            return
        self._interactor_style = RevitInteractorStyle(
            self.plotter,
            self._picker,
            on_point_picked=self._handle_point_pick,
            on_point_double_clicked=self._handle_point_double_click,
            on_area_selected=self._handle_area_selection,
            on_clear_selection=self.session.clear_selection,
        )
        vtk_interactor.SetInteractorStyle(self._interactor_style)

    def _handle_area_selection(self, start: tuple, end: tuple):
        """Select all rendered nodes whose screen projection falls in the drag rect."""
        if self.point_cloud is None or self.point_cloud.n_points == 0:
            return
        renderer = getattr(self.plotter, "renderer", None)
        if renderer is None:
            return

        x0 = min(start[0], end[0])
        x1 = max(start[0], end[0])
        y0 = min(start[1], end[1])
        y1 = max(start[1], end[1])
        if x1 - x0 < 2 and y1 - y0 < 2:
            return

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        node_ids_in_rect = []
        pts = self.point_cloud.points
        for i in range(len(pts)):
            coord.SetValue(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
            dx, dy = coord.GetComputedDisplayValue(renderer)
            if x0 <= dx <= x1 and y0 <= dy <= y1:
                node_id = self._resolve_node_id(i)
                if node_id is not None:
                    node_ids_in_rect.append(node_id)

        if node_ids_in_rect:
            self.session.add_nodes_to_multi_selection(node_ids_in_rect)

    def apply_selection_filter(self, mode: str, render: bool = True):
        if mode not in ("all", "only", "hide"):
            mode = "all"
        self._selection_visibility = mode
        if mode == "all":
            self._domain_node_ids = None
        else:
            current_selection = self._current_selection_ids()
            if current_selection:
                visible_ids = self.session.current_visible_node_ids()
                current_domain = (
                    set(self._domain_node_ids)
                    if self._domain_node_ids is not None
                    else set(visible_ids)
                )
                selected_in_domain = current_domain.intersection(current_selection)
                if mode == "only":
                    self._domain_node_ids = selected_in_domain
                else:  # "hide"
                    self._domain_node_ids = current_domain.difference(current_selection)
        self.rebuild_scalar_actor(render=False)
        self.refresh_selection(render=False)
        self.refresh_multi_selection(render=False)
        if render:
            self.plotter.render()

    def _current_selection_ids(self) -> set:
        selected = set(self.session.state.multi_selection)
        node_id = self.session.state.selected_node
        if node_id is not None:
            selected.add(node_id)
        return selected

    def _domain_contains(self, node_id) -> bool:
        return self._domain_node_ids is None or node_id in self._domain_node_ids

    def _domain_index_array(self):
        if self._domain_node_ids is None:
            return None
        import numpy as np
        visible_ids = self.session.current_visible_node_ids()
        mask = [
            i for i, nid in enumerate(visible_ids)
            if nid in self._domain_node_ids
        ]
        return np.array(mask, dtype=int)

    def _resolve_node_id(self, point_id: int):
        """Map a picker point_id to a node_id, respecting visual filters."""
        if self._visual_node_ids_list is not None:
            if point_id < len(self._visual_node_ids_list):
                return self._visual_node_ids_list[point_id]
            return None
        return self.session.adapter.node_id_from_visible_index(point_id)

    def _handle_point_pick(self, point_id, _pick_pos=None):
        if point_id < 0:
            return
        node_id = self._resolve_node_id(point_id)
        if node_id is None:
            return
        self.session.select_node(node_id)

    def _handle_point_double_click(self, point_id, _pick_pos=None):
        if point_id < 0:
            return
        node_id = self._resolve_node_id(point_id)
        if node_id is None:
            return
        self.session.select_node(node_id)
        self.center_on_node(node_id)

    def apply_appearance(self, render: bool = True):
        self.plotter.set_background(self.session.current_background_color())
        self.rebuild_scalar_actor(render=render)
        self.refresh_selection(render=False)

    def apply_color_range(self, render: bool = True):
        if self.point_cloud is not None and "active_scalars" in self.point_cloud.point_data:
            scalars = self.point_cloud.point_data["active_scalars"]
        else:
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
        # Update _visible_node_ids so that node-picking resolves correctly after
        # this rebuild.  current_warped_points() was refactored to skip this
        # O(N) Python list-comprehension on every animation frame, so we call
        # current_visible_points() explicitly here (rebuild is not a hot path).
        self.session.current_visible_points()
        points = self.session.current_warped_points()
        if self._gf_pin_active():
            scalars = self._scalars_for_gf_pin()
        else:
            scalars = self.session.current_visible_scalars()

        # Per-pane visual domain (selection filters).
        self._visual_node_ids_list = None
        if self._domain_node_ids is not None:
            visible_ids = self.session.current_visible_node_ids()
            arr = self._domain_index_array()
            points = points[arr]
            scalars = scalars[arr]
            self._visual_node_ids_list = [visible_ids[i] for i in arr]

        self.point_cloud = pv.PolyData(points)
        self.point_cloud.point_data["active_scalars"] = scalars

    def _add_point_actor(self):
        if self.point_cloud is None or self.point_cloud.n_points == 0:
            return None
        scalars = self.point_cloud.point_data["active_scalars"]
        if self._gf_pin_active():
            clim = self._color_limits_for_gf_pin(scalars)
            bar_title = str(self._gf_component_pin).upper()   # e.g. "G11"
        else:
            clim = self.session.current_color_limits(scalars)
            bar_title = self.session.current_scalar_bar_title()
        actor = self.plotter.add_points(
            self.point_cloud,
            scalars="active_scalars",
            cmap=self.session.current_colormap(),
            clim=clim,
            render_points_as_spheres=True,
            point_size=self._point_size(),
            show_scalar_bar=self.session.state.show_scalar_bar,
            scalar_bar_args={"title": bar_title},
            render=False,
        )
        try:
            actor.GetProperty().SetOpacity(self.session.state.node_opacity)
        except Exception:
            pass
        return actor

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
                font_size=7,
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

    def dispose(self) -> None:
        """Release actors, VTK helpers and geometry owned by this scene."""
        try:
            if self.station_label_actor is not None:
                self.plotter.remove_actor(self.station_label_actor, render=False)
        except Exception:
            pass
        try:
            if self.station_actor is not None:
                self.plotter.remove_actor(self.station_actor, render=False)
        except Exception:
            pass
        try:
            if self.selection_actor is not None:
                self.plotter.remove_actor(self.selection_actor, render=False)
        except Exception:
            pass
        try:
            if self.point_actor is not None:
                self.plotter.remove_actor(self.point_actor, render=False)
        except Exception:
            pass
        try:
            if self._gf_label_actor is not None:
                self.plotter.remove_actor(self._gf_label_actor, render=False)
        except Exception:
            pass
        try:
            self.plotter.clear()
        except Exception:
            pass

        self.point_cloud = None
        self.point_actor = None
        self.selection_actor = None
        self.station_actor = None
        self.station_label_actor = None
        self._gf_label_actor = None
        self._interactor_style = None
        self._picker = None
