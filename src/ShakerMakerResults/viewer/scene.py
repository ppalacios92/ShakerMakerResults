"""PyVista scene management for the interactive viewer."""

from __future__ import annotations

import math

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
        self._point_actor_rgb = False
        self.selection_actor = None
        self.multi_selection_actor = None   # red spheres for selection-mode set
        self.multi_selection_label_actor = None
        self.vector_actor = None            # line overlay for vector field
        self._vec_pv = None                 # pv.PolyData live source (lines)
        self._vec_factor = None             # cached scale factor (float)
        self._vec_N = 0                     # number of points in current field
        self._vec_max_mag = 1.0             # scalar range upper bound
        self._vec_pts_buf = None            # pre-allocated (4*N, 3) points buffer
        self.station_actor = None
        self.station_label_actor = None
        self.ghost_actor = None
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
        self._gf_label_text: str | None = None  # remembered for re-render on theme switch

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
        self._gf_label_text = label if component is not None else None
        # Add a new label when a pin is being set.
        if label:
            try:
                self._gf_label_actor = self.plotter.add_text(
                    label,
                    position="upper_right",
                    font_size=9,
                    color=self._foreground_color(),
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

    def build(self, *, lightweight: bool = False):
        self._rebuild_point_cloud(lightweight=lightweight)
        self.plotter.set_background(self.session.current_background_color())
        self.point_actor = self._add_point_actor()
        self.plotter.add_axes()
        self._sync_axes_grid()
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
        rgb_scalars = self._scalars_are_rgb(scalars)
        if self.point_actor is not None and rgb_scalars != self._point_actor_rgb:
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

    def refresh_ghost_reference(self, render: bool = True):
        if self.ghost_actor is not None:
            try:
                self.plotter.remove_actor(self.ghost_actor, render=False)
            except Exception:
                pass
            self.ghost_actor = None
        if not (
            self.session.state.ghost_warp_reference
            and self.session.state.disp_warp_enabled
        ):
            if render:
                self.plotter.render()
            return
        try:
            import numpy as np

            base_points = self.session.current_visible_points()
            domain_idx = self._domain_index_array()
            if domain_idx is not None:
                base_points = base_points[domain_idx]
            cloud = pv.PolyData(np.asarray(base_points, dtype=float))
            self.ghost_actor = self.plotter.add_points(
                cloud,
                color="#6b7280",
                opacity=0.22,
                point_size=max(self._point_size() - 1, 2),
                render_points_as_spheres=True,
                render=False,
            )
            try:
                self.ghost_actor.SetPickable(0)
            except Exception:
                pass
        except Exception:
            self.ghost_actor = None
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
        if self.session.state.vector_field_enabled:
            self.refresh_vector_field(render=False)
        # Branding doubles as the in-scene HUD (field / range / warp);
        # rebuild it so a demand-or-warp switch reflects in the corner
        # text without waiting for a separate appearance broadcast.
        self._refresh_branding()
        if render:
            self.plotter.render()

    def _refresh_branding(self) -> None:
        """Remove + re-add the branding/HUD actor with the current state."""
        try:
            self.plotter.remove_actor("branding", render=False)
        except Exception:
            pass
        self._add_branding()

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
        if self.multi_selection_label_actor is not None:
            try:
                self.plotter.remove_actor(self.multi_selection_label_actor, render=False)
            except Exception:
                pass
            self.multi_selection_label_actor = None

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
        if state.selection_labels_enabled:
            try:
                labels = [
                    str(node_id)
                    for node_id in state.multi_selection
                    if self._domain_contains(node_id)
                ]
                if labels:
                    self.multi_selection_label_actor = self.plotter.add_point_labels(
                        sel_cloud,
                        labels,
                        point_size=0,
                        font_size=9,
                        text_color="#b71c1c",
                        fill_shape=False,
                        always_visible=True,
                        render=False,
                    )
            except Exception:
                self.multi_selection_label_actor = None
        if render:
            self.plotter.render()

    @staticmethod
    def _build_vec_lut(cmap_name: str, max_mag: float) -> "vtk.vtkLookupTable":
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        lut.SetTableRange(0.0, max(max_mag, 1e-30))
        try:
            import matplotlib.cm as _cm
            cmap = _cm.get_cmap(cmap_name, 256)
            for i in range(256):
                r, g, b, a = cmap(i / 255.0)
                lut.SetTableValue(i, r, g, b, 1.0)
        except Exception:
            lut.SetHueRange(0.667, 0.0)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
        lut.Build()
        return lut

    def refresh_vector_field(self, render: bool = True):
        """Build a live all-lines overlay: shaft + V-tip, coloured by magnitude."""
        self._teardown_vector_pipeline()

        if not self.session.state.vector_field_enabled:
            if render:
                self.plotter.render()
            return

        try:
            import numpy as np
            points, vectors = self.session.current_vector_data()
            if points is None or len(points) == 0:
                return

            # Apply the per-scene selection/domain filter so "Show only selected"
            # also masks the vector field to match the main point cloud.
            domain_idx = self._domain_index_array()
            if domain_idx is not None:
                if len(domain_idx) == 0:
                    return
                points = points[domain_idx]
                vectors = vectors[domain_idx]

            magnitudes = np.linalg.norm(vectors, axis=1)
            max_mag = float(magnitudes.max())
            if max_mag < 1e-30:
                return

            bounds = self.plotter.bounds
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            domain_diag = max((dx**2 + dy**2 + dz**2) ** 0.5, 1.0)
            n = max(len(points), 1)
            factor = (
                self.session.state.vector_field_scale
                * domain_diag
                * 0.05
                / (n ** 0.33 * max_mag)
            )
            self._vec_factor = factor
            self._vec_N = len(points)
            self._vec_max_mag = max_mag

            lut = self._build_vec_lut(self.session.state.vector_field_colormap, max_mag)
            all_pts, lines, scalars = self._build_vec_geometry(
                points, vectors, magnitudes, factor
            )
            self._vec_pts_buf = all_pts.copy()

            self._vec_pv = pv.PolyData()
            self._vec_pv.points = self._vec_pts_buf
            self._vec_pv.lines = lines
            self._vec_pv["magnitude"] = scalars

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self._vec_pv)
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToMapScalars()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(0.0, max_mag)

            self.vector_actor = vtk.vtkActor()
            self.vector_actor.SetMapper(mapper)
            self.vector_actor.GetProperty().SetLineWidth(2.0)
            self.vector_actor.SetPickable(0)

            renderer = getattr(self.plotter, "renderer", None)
            if renderer is not None:
                renderer.AddActor(self.vector_actor)
        except Exception:
            self._teardown_vector_pipeline()

        if render:
            self.plotter.render()

    def update_vector_arrows(self, render: bool = True):
        """Hot-path: update line points in-place without rebuilding topology or allocs."""
        if self._vec_pv is None or self._vec_factor is None or self._vec_pts_buf is None:
            return
        try:
            import numpy as np
            points, vectors = self.session.current_vector_data()
            domain_idx = self._domain_index_array()
            if domain_idx is not None:
                if len(domain_idx) == 0:
                    return
                points = points[domain_idx]
                vectors = vectors[domain_idx]
            if vectors is None or len(vectors) != self._vec_N:
                return

            N = self._vec_N
            factor = self._vec_factor

            magnitudes = np.linalg.norm(vectors, axis=1)
            safe_mag = np.where(magnitudes > 1e-30, magnitudes, 1.0)
            dirs = vectors / safe_mag[:, None]

            zup = np.array([0.0, 0.0, 1.0])
            perps = np.cross(dirs, zup)
            bad = np.linalg.norm(perps, axis=1) < 1e-6
            if bad.any():
                perps[bad] = np.cross(dirs[bad], [1.0, 0.0, 0.0])
            perps /= np.maximum(np.linalg.norm(perps, axis=1, keepdims=True), 1e-30)

            ends = points + vectors * factor
            tip_len = (magnitudes * factor * 0.25)[:, None]

            # Write directly into the pre-allocated buffer — no vstack, no lines recompute
            buf = self._vec_pts_buf
            buf[:N]    = points
            buf[N:2*N] = ends
            buf[2*N:3*N] = ends - dirs * tip_len + perps * tip_len * 0.5
            buf[3*N:]    = ends - dirs * tip_len - perps * tip_len * 0.5

            self._vec_pv.points = buf
            self._vec_pv["magnitude"] = np.tile(magnitudes, 4).astype(float)
            self._vec_pv.Modified()
        except Exception:
            pass
        if render:
            self.plotter.render()

    @staticmethod
    def _build_vec_geometry(points, vectors, magnitudes, factor):
        """Return (all_pts, lines, scalars) for shaft + V-tip lines.

        Point layout: [starts | ends | left-wings | right-wings]  (4×N)
        Lines:        shaft (N) + left-wing (N) + right-wing (N)  (3×N)
        """
        import numpy as np
        N = len(points)
        safe_mag = np.where(magnitudes > 1e-30, magnitudes, 1.0)
        dirs = vectors / safe_mag[:, None]           # unit shaft direction

        # Perpendicular for the V — cross with Z, fall back to X
        zup = np.array([0.0, 0.0, 1.0])
        perps = np.cross(dirs, zup)
        bad = np.linalg.norm(perps, axis=1) < 1e-6
        if bad.any():
            perps[bad] = np.cross(dirs[bad], [1.0, 0.0, 0.0])
        perp_n = np.linalg.norm(perps, axis=1, keepdims=True)
        perps /= np.maximum(perp_n, 1e-30)

        ends = (points + vectors * factor).astype(float)
        tip_len = (magnitudes * factor * 0.25)[:, None]  # 25 % of shaft length
        left_pts  = ends - dirs * tip_len + perps * tip_len * 0.5
        right_pts = ends - dirs * tip_len - perps * tip_len * 0.5

        all_pts = np.vstack([points, ends, left_pts, right_pts]).astype(float)

        idx = np.arange(N)
        lines = np.empty((3 * N, 3), dtype=np.int_)
        lines[0*N:1*N] = np.column_stack([np.full(N, 2), idx,     idx + N])
        lines[1*N:2*N] = np.column_stack([np.full(N, 2), idx + N, idx + 2*N])
        lines[2*N:3*N] = np.column_stack([np.full(N, 2), idx + N, idx + 3*N])

        scalars = np.tile(magnitudes, 4).astype(float)
        return all_pts, lines, scalars

    def _teardown_vector_pipeline(self):
        """Remove vector actor and release pipeline objects."""
        if self.vector_actor is not None:
            renderer = getattr(self.plotter, "renderer", None)
            try:
                if renderer is not None:
                    renderer.RemoveActor(self.vector_actor)
                else:
                    self.plotter.remove_actor(self.vector_actor, render=False)
            except Exception:
                try:
                    self.plotter.remove_actor(self.vector_actor, render=False)
                except Exception:
                    pass
            self.vector_actor = None
        self._vec_pv = None
        self._vec_factor = None
        self._vec_N = 0
        self._vec_max_mag = 1.0
        self._vec_pts_buf = None

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

    def update_point_sizes(self, render: bool = True):
        """Apply point-size changes to existing actors without rebuilding domains."""
        base_size = self._point_size()
        updates = (
            (self.point_actor, base_size),
            (self.selection_actor, max(base_size + 8, 18)),
            (self.multi_selection_actor, max(base_size + 6, 16)),
            (self.station_actor, max(base_size + 3, 12)),
        )
        for actor, point_size in updates:
            if actor is None:
                continue
            try:
                actor.GetProperty().SetPointSize(float(point_size))
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

    def _handle_point_pick(self, point_id, _pick_pos=None, additive: bool = False):
        if point_id < 0:
            return
        node_id = self._resolve_node_id(point_id)
        if node_id is None:
            return
        if additive:
            self.session.toggle_node_in_multi_selection(node_id)
        else:
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
        # VTK text actors bake their colour at creation time — the branding
        # block and the GF component label keep their *original* foreground
        # colour even after the renderer background changes.  Re-create the
        # branding so it reads against the current scene.  ``rebuild_scalar_actor``
        # below will refresh it again (with up-to-date demand / range / warp
        # text) but doing it here first guarantees the right colour even on
        # paths that skip the rebuild.
        self._refresh_branding()
        if self._gf_component_pin is not None and self._gf_label_actor is not None:
            try:
                self.plotter.remove_actor(self._gf_label_actor, render=False)
            except Exception:
                pass
            self._gf_label_actor = None
            try:
                self._gf_label_actor = self.plotter.add_text(
                    self._gf_label_text or self._gf_component_pin,
                    position="upper_right",
                    font_size=9,
                    color=self._foreground_color(),
                    render=False,
                )
            except Exception:
                pass
        self.rebuild_scalar_actor(render=render)
        self.refresh_selection(render=False)
        self.refresh_multi_selection(render=False)
        self.refresh_ghost_reference(render=False)
        self._sync_axes_grid()

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

    def _rebuild_point_cloud(self, *, lightweight: bool = False):
        # Update _visible_node_ids so that node-picking resolves correctly after
        # this rebuild.  current_warped_points() was refactored to skip this
        # O(N) Python list-comprehension on every animation frame, so we call
        # current_visible_points() explicitly here (rebuild is not a hot path).
        self.session.current_visible_points()
        points = self.session.current_warped_points()
        if lightweight:
            scalars = self.session.adapter.visible_scalars(
                self.session.adapter.elevation_snapshot(),
                show_internal=self.session.state.show_internal,
                show_external=self.session.state.show_external,
                show_qa=self.session.state.show_qa,
            )
        elif self._gf_pin_active():
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
        rgb_scalars = self._scalars_are_rgb(scalars)
        if self._gf_pin_active():
            clim = self._color_limits_for_gf_pin(scalars)
            bar_title = str(self._gf_component_pin).upper()   # e.g. "G11"
        else:
            clim = self.session.current_color_limits(scalars)
            bar_title = self._scalar_bar_title()
        # PyVista keeps every scalar bar it has ever added in a per-plotter
        # registry keyed by title.  Calling ``.clear()`` only empties the
        # *registry* dict — the underlying VTK 2-D actors stay parented to
        # the renderer, so a demand switch (accel → vel etc.) ends up
        # overlaying two titles + two sets of labels on top of each other.
        # The fix is to ``remove_scalar_bar(title)`` per known title so the
        # actors actually leave the renderer, then drop the registry.
        try:
            bars = getattr(self.plotter, "scalar_bars", None)
            if bars is not None:
                try:
                    titles = list(bars.keys()) if hasattr(bars, "keys") else []
                except Exception:
                    titles = []
                for title in titles:
                    try:
                        self.plotter.remove_scalar_bar(title)
                    except Exception:
                        pass
                if hasattr(bars, "clear"):
                    try:
                        bars.clear()
                    except Exception:
                        pass
        except Exception:
            pass
        kwargs = {
            "scalars": "active_scalars",
            "render_points_as_spheres": True,
            "point_size": self._point_size(),
            "render": False,
        }
        if rgb_scalars:
            kwargs["rgb"] = True
            kwargs["show_scalar_bar"] = False
        else:
            kwargs["cmap"] = self.session.current_colormap()
            kwargs["clim"] = clim
            kwargs["n_colors"] = (
                self.session.state.colormap_bins
                if self.session.state.colormap_discrete
                else 256
            )
            kwargs["nan_color"] = self.session.state.nan_color
            if self.session.state.use_below_range_color:
                kwargs["below_color"] = self.session.state.below_range_color
            if self.session.state.use_above_range_color:
                kwargs["above_color"] = self.session.state.above_range_color
            kwargs["show_scalar_bar"] = self.session.state.show_scalar_bar
            # Build the scalar-bar configuration from the user-controllable
            # state (position, orientation, fonts, outline …) and inject
            # sane defaults (numeric format, no shadow, theme-aware text
            # colour) so the bar reads against both light and dark scenes.
            scalar_bar_args = {
                "title": bar_title,
                "title_font_size": self.session.state.legend_title_font_size,
                "label_font_size": self.session.state.legend_label_font_size,
                "n_labels": self.session.state.legend_label_count,
                "vertical": self.session.state.legend_orientation == "vertical",
                "outline": self.session.state.legend_show_outline,
                "fmt": "%.3g",
                "shadow": False,
                "italic": False,
                "color": self._foreground_color(),
            }
            if self.session.state.legend_position == "left":
                scalar_bar_args.update({"position_x": 0.03, "position_y": 0.18})
            elif self.session.state.legend_position == "top":
                scalar_bar_args.update({"position_x": 0.28, "position_y": 0.88})
            elif self.session.state.legend_position == "bottom":
                scalar_bar_args.update({"position_x": 0.28, "position_y": 0.04})
            else:
                scalar_bar_args.update({"position_x": 0.86, "position_y": 0.18})
            if self.session.state.legend_orientation == "horizontal":
                scalar_bar_args.update({"width": 0.42, "height": 0.08})
            else:
                scalar_bar_args.update({"width": 0.08, "height": 0.56})
            if self.session.state.legend_show_background:
                # Use a colour that contrasts with the renderer background
                # without being pure white in dark mode (which used to leave
                # a glaring white box around the colormap).  Light bg → off
                # white panel surface; dark bg → palette surface so the bar
                # sits a touch lighter than the scene.
                from .theme import active_palette
                scalar_bar_args["background_color"] = active_palette().surface
            kwargs["scalar_bar_args"] = scalar_bar_args
        actor = self.plotter.add_points(self.point_cloud, **kwargs)
        self._point_actor_rgb = rgb_scalars
        try:
            actor.GetProperty().SetOpacity(self.session.state.node_opacity)
        except Exception:
            pass
        return actor

    def _sync_axes_grid(self):
        try:
            if self.session.state.show_axes_grid:
                self.plotter.show_grid(
                    color=self.session.state.axes_grid_color,
                    render=False,
                )
            else:
                self.plotter.remove_bounds_axes()
        except Exception:
            pass

    @staticmethod
    def _scalars_are_rgb(scalars) -> bool:
        return getattr(scalars, "ndim", 1) == 2 and scalars.shape[1] in (3, 4)

    def _scalar_bar_title(self) -> str:
        title = self.session.current_scalar_bar_title()
        state = self.session.state
        if not state.disp_warp_enabled:
            return title
        components = [
            label
            for enabled, label in zip(state.warp_axes, ("e", "n", "z"))
            if enabled
        ]
        component_text = ", ".join(components) if components else "none"
        scale = state.warp_scale
        if scale is None:
            scale = self.session.suggested_warp_scale()
        return f"{title}  + warp / {component_text} x{self._format_warp_scale(scale)}"

    @staticmethod
    def _format_warp_scale(scale) -> str:
        try:
            value = float(scale)
        except (TypeError, ValueError):
            return "auto"
        if not math.isfinite(value):
            return "auto"
        sign = "-" if value < 0 else ""
        value = abs(value)
        if value >= 1000.0:
            scaled = value / 1000.0
            return f"{sign}{scaled:g}k"
        return f"{sign}{value:g}"

    def _add_branding(self):
        """Add a compact branding + state HUD to the upper-left corner.

        Two short lines:

        1. ``ShakerMaker Results · <project name>`` — the sobrio header.
        2. ``<demand>/<component> · range [vmin, vmax] · warp on/off`` —
           a status HUD so the user always knows what the renderer is
           showing without consulting any side dock.

        Old ``By: Ladruno Team / Version / Description`` lines moved out
        to ``Help → About`` (paper / poster captures stay clean).
        """
        project_name = self.session.adapter.summary().name or "model"
        state = self.session.state
        demand = state.demand or "—"
        component = state.component or "—"
        try:
            vmin, vmax = self.session.current_color_limits()
            range_text = f"range [{vmin:.3g}, {vmax:.3g}]"
        except Exception:
            range_text = "range —"
        warp_text = "warp on" if getattr(state, "disp_warp_enabled", False) else "warp off"

        header = f"ShakerMaker Results  ·  {project_name}"
        status = f"{demand} / {component}    {range_text}    {warp_text}"

        try:
            self.plotter.add_text(
                f"{header}\n{status}",
                position="upper_left",
                font_size=8,
                color=self._foreground_color(secondary=True),
                shadow=False,
                name="branding",
                render=False,
            )
        except Exception:
            pass

    def _foreground_color(self, *, secondary: bool = False) -> str:
        """Return a text color that reads against the active renderer background.

        Uses the perceived luminance of the plotter's clear color (or the
        background named in the session state when the plotter cannot report
        its color) to pick light text on dark scenes and vice-versa.  The
        ``secondary`` flag dims the result so the branding text never competes
        with the scalar bar labels for visual weight.
        """

        from .colors import BACKGROUND_PRESETS

        def _hex_to_rgb(value: str) -> tuple[float, float, float]:
            value = value.lstrip("#")
            if len(value) == 3:
                value = "".join(ch * 2 for ch in value)
            if len(value) != 6:
                return 1.0, 1.0, 1.0
            return (
                int(value[0:2], 16) / 255.0,
                int(value[2:4], 16) / 255.0,
                int(value[4:6], 16) / 255.0,
            )

        rgb: tuple[float, float, float] | None = None
        try:
            renderer = getattr(self.plotter, "renderer", None)
            if renderer is not None and hasattr(renderer, "GetBackground"):
                rgb = tuple(float(c) for c in renderer.GetBackground())  # type: ignore[assignment]
        except Exception:
            rgb = None
        if rgb is None:
            bg_name = getattr(self.session.state, "background", "White")
            rgb = _hex_to_rgb(BACKGROUND_PRESETS.get(bg_name, "#ffffff"))

        # Rec. 709 luminance.  Dark renderer background → bone-tinted text
        # to match the "hueso" theme; light background → near-black text.
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        if luminance < 0.45:
            return "#aea495" if secondary else "#e8dfca"   # dim bone / bone
        return "#555555" if secondary else "#172033"


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
            if self.multi_selection_label_actor is not None:
                self.plotter.remove_actor(self.multi_selection_label_actor, render=False)
        except Exception:
            pass
        try:
            if self.ghost_actor is not None:
                self.plotter.remove_actor(self.ghost_actor, render=False)
        except Exception:
            pass
        try:
            if self.selection_actor is not None:
                self.plotter.remove_actor(self.selection_actor, render=False)
        except Exception:
            pass
        try:
            self._teardown_vector_pipeline()
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
        self.multi_selection_label_actor = None
        self.ghost_actor = None
        self._gf_label_actor = None
        self._interactor_style = None
        self._picker = None
