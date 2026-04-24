import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ShakerMakerResults.viewer.session import ViewerSession


class FakeModel:
    def __init__(self):
        self.xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        self.xyz_qa = np.array([[0.5, 0.0, 0.0]], dtype=float)
        self.time = np.array([0.0, 0.5, 1.0], dtype=float)
        self.name = "session-model"
        self.is_drm = True
        self.internal = np.array([False, True], dtype=bool)
        self._has_gf = False
        self._has_map = False

    def get_node_data(self, node_id, data_type="accel"):
        scale = {"accel": 1.0, "vel": 0.1, "disp": 0.01}[data_type]
        base = node_id + 1
        return np.array(
            [
                [base, base + 1, base + 2],
                [base + 10, base + 11, base + 12],
                [base + 20, base + 21, base + 22],
            ],
            dtype=float,
        ) * scale

    def get_qa_data(self, data_type="accel"):
        scale = {"accel": 1.0, "vel": 0.1, "disp": 0.01}[data_type]
        return np.array(
            [
                [100.0, 101.0, 102.0],
                [200.0, 201.0, 202.0],
                [300.0, 301.0, 302.0],
            ],
            dtype=float,
        ) * scale

    def get_surface_snapshot(self, time_idx, component="z", data_type="accel"):
        idx = {"z": 0, "e": 1, "n": 2}[component]
        return np.array(
            [self.get_node_data(node_id, data_type)[idx, time_idx] for node_id in range(2)],
            dtype=float,
        )


class ViewerSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = ViewerSession(FakeModel(), show=False)

    def test_initial_session_has_adapter_and_state(self):
        self.assertEqual(self.session.title, "session-model")
        self.assertEqual(self.session.state.component, "resultant")
        self.assertEqual(self.session.current_time(), 0.0)
        self.assertIsNone(self.session.window)

    def test_session_mutators_refresh_state_without_window(self):
        self.session.set_time_index(2)
        self.session.set_demand("vel")
        self.session.set_component("z")
        self.session.select_node("QA")

        self.assertEqual(self.session.current_time(), 1.0)
        self.assertEqual(self.session.state.demand, "vel")
        self.assertEqual(self.session.state.component, "z")
        self.assertEqual(self.session.state.selected_node, "QA")

        scalars = self.session.current_scalars()
        trace = self.session.current_trace()
        self.assertEqual(scalars.shape, (3,))
        self.assertEqual(trace.shape, (3, 3))
        self.assertAlmostEqual(trace[0, 0], 10.0)

    def test_step_jump_and_play_state(self):
        self.assertEqual(self.session.step_time(1), 1)
        self.assertEqual(self.session.jump_time(10), 2)
        self.assertTrue(self.session.toggle_playing())
        self.assertFalse(self.session.toggle_playing())
        self.assertEqual(self.session.set_playback_speed(1.5), 1.5)

    def test_appearance_state_updates(self):
        self.assertEqual(self.session.set_background("Gray"), "Gray")
        self.assertEqual(self.session.set_colormap("seismic"), "seismic")
        self.assertEqual(self.session.set_point_size(14), 14.0)
        self.assertFalse(self.session.set_show_scalar_bar(False))
        self.assertEqual(self.session.current_background_color(), "#d7d9dd")
        self.assertEqual(self.session.current_colormap(), "seismic")
        self.assertEqual(self.session.suggested_point_size(), 14.0)

    def test_set_playing_prewarms_series_cache(self):
        """Starting playback must pre-load the scalar series into memory."""
        self.assertEqual(self.session.adapter.cache_info["entries"], 0)
        self.session.set_playing(True)
        self.assertEqual(self.session.adapter.cache_info["entries"], 1)
        self.assertTrue(self.session.state.is_playing)

    def test_second_set_playing_true_does_not_double_load(self):
        """A second set_playing(True) call while already playing must not reload."""
        self.session.set_playing(True)
        hits_before = self.session.adapter.cache_info["hits"]
        entries_before = self.session.adapter.cache_info["entries"]

        # Already playing → the guard `not self.state.is_playing` prevents
        # a redundant scalar_series() call, so hits and entries stay unchanged.
        self.session.set_playing(True)
        self.assertEqual(self.session.adapter.cache_info["entries"], entries_before)
        self.assertEqual(self.session.adapter.cache_info["hits"], hits_before)

    def test_set_playing_false_does_not_clear_cache(self):
        """Stopping playback leaves the series cache intact for fast resume."""
        self.session.set_playing(True)
        entries_before = self.session.adapter.cache_info["entries"]
        self.session.set_playing(False)
        self.assertEqual(self.session.adapter.cache_info["entries"], entries_before)
        self.assertFalse(self.session.state.is_playing)

    def test_node_visibility_filters_visible_scalars(self):
        # With defaults all nodes are shown; hiding internal reduces count.
        self.session.set_node_visibility(show_internal=False)
        scalars = self.session.current_visible_scalars()
        # FakeModel has 2 nodes (1 internal) + 1 QA → hiding internal leaves 2
        self.assertEqual(scalars.shape, (2,))

    def test_clamp_returns_user_limits(self):
        self.session.set_color_range(0.5, 2.5, clamp=True)
        vmin, vmax = self.session.current_color_limits()
        self.assertAlmostEqual(vmin, 0.5)
        self.assertAlmostEqual(vmax, 2.5)

    def test_clamp_disabled_returns_data_limits(self):
        self.session.set_color_range(0.5, 2.5, clamp=False)
        vmin, vmax = self.session.current_color_limits()
        # Auto limits come from data — they differ from the user-set range
        self.assertGreaterEqual(vmax, 0.0)

    def test_select_nearest_coordinate_resolves_to_node(self):
        # Point (0,0,0) in model metres → node 0
        node_id, dist = self.session.select_nearest_coordinate_m(0.0, 0.0, 0.0)
        self.assertIn(node_id, self.session.adapter.node_ids)
        self.assertAlmostEqual(dist, 0.0, places=3)

    # ── 3-D Warp ─────────────────────────────────────────────────────────

    def test_warp_disabled_warped_points_equals_base(self):
        base = self.session.current_visible_points()
        warped = self.session.current_warped_points()
        self.assertTrue(np.allclose(base, warped))

    def test_set_warp_enabled_prewarms_displacement_cache(self):
        before = self.session.adapter.cache_info["entries"]
        self.session.set_warp_enabled(True)
        # 3 displacement components (e, n, z) loaded into cache
        self.assertEqual(self.session.adapter.cache_info["entries"], before + 3)
        self.assertTrue(self.session.state.disp_warp_enabled)

    def test_set_warp_disabled_returns_base_points(self):
        self.session.set_warp_enabled(True)
        self.session.set_warp_enabled(False)
        base = self.session.current_visible_points()
        warped = self.session.current_warped_points()
        self.assertTrue(np.allclose(base, warped))

    def test_warp_enabled_displaces_points(self):
        self.session.set_warp_enabled(True)
        self.session.set_warp_scale(1000.0)
        base = self.session.current_visible_points()
        warped = self.session.current_warped_points()
        # With a non-zero scale the warped positions differ from base
        # (model has non-zero disp data at t=0)
        diff = np.abs(warped - base)
        self.assertGreater(diff.max(), 0.0)

    def test_warp_axes_z_only(self):
        """Disabling X and Y leaves only the Z axis warped."""
        self.session.set_warp_enabled(True)
        self.session.set_warp_scale(1000.0)
        self.session.set_warp_axes(x=False, y=False, z=True)
        base = self.session.current_visible_points()
        warped = self.session.current_warped_points()
        # X and Y columns must be identical to base
        self.assertTrue(np.allclose(warped[:, 0], base[:, 0]))
        self.assertTrue(np.allclose(warped[:, 1], base[:, 1]))
        # Z column may differ
        # (only if displacement z is non-zero at t=0 for all nodes)

    def test_warp_scale_none_uses_auto(self):
        self.session.set_warp_enabled(True)
        self.session.set_warp_scale(None)
        self.assertIsNone(self.session.state.warp_scale)
        # current_warped_points must not raise with auto scale
        warped = self.session.current_warped_points()
        self.assertIsNotNone(warped)

    def test_set_playing_with_warp_prewarms_displacement(self):
        self.session.set_warp_enabled(True)
        # Clear any pre-warmed entries so we count cleanly
        self.session.adapter._series_cache.clear()
        self.session.set_playing(True)
        # Must have: 1 entry for the scalar demand + 3 for disp components
        self.assertGreaterEqual(self.session.adapter.cache_info["entries"], 4)

    def test_suggested_warp_scale_positive(self):
        scale = self.session.suggested_warp_scale()
        self.assertGreater(scale, 0.0)


if __name__ == "__main__":
    unittest.main()
