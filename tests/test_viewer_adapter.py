import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ShakerMakerResults.viewer.adapter import ViewerDataAdapter


class FakeModel:
    def __init__(self):
        self.xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        self.xyz_qa = np.array([[0.5, 0.5, 0.0]], dtype=float)
        self.time = np.array([0.0, 1.0, 2.0], dtype=float)
        self.name = "fake-model"
        self.is_drm = True
        self.internal = np.array([False, True, False], dtype=bool)
        self._has_gf = True
        self._has_map = True
        self.surface_calls = 0

    def get_node_data(self, node_id, data_type="accel"):
        scale = {"accel": 1.0, "vel": 0.1, "disp": 0.01}[data_type]
        z = np.array([node_id + 1, node_id + 2, node_id + 3], dtype=float) * scale
        e = np.array([10 + node_id, 11 + node_id, 12 + node_id], dtype=float) * scale
        n = np.array([20 + node_id, 21 + node_id, 22 + node_id], dtype=float) * scale
        return np.vstack([z, e, n])

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
        self.surface_calls += 1
        idx = {"z": 0, "e": 1, "n": 2}[component]
        return np.array(
            [self.get_node_data(node_id, data_type)[idx, time_idx] for node_id in range(3)],
            dtype=float,
        )

    def get_gf(self, node_id, subfault_id, component):
        if component == "tdata":
            return np.ones((3, 9))
        node_value = 999 if node_id in ("QA", "qa") else float(node_id)
        return np.array([node_value, float(subfault_id), 1.5], dtype=float)


class FakeDataset:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape

    def __getitem__(self, key):
        return self.data[key]


class FakeH5File:
    def __init__(self, mapping):
        self.mapping = mapping

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getitem__(self, key):
        return self.mapping[key]


class ViewerDataAdapterTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.adapter = ViewerDataAdapter(self.model)

    def test_summary_reflects_model_features(self):
        summary = self.adapter.summary()
        self.assertEqual(summary.name, "fake-model")
        self.assertEqual(summary.dataset_type, "DRM")
        self.assertEqual(summary.node_count, 3)
        self.assertEqual(summary.display_node_count, 4)
        self.assertTrue(summary.has_qa)
        self.assertTrue(summary.has_gf)
        self.assertTrue(summary.has_map)

    def test_scalar_snapshot_resultant_includes_qa(self):
        values = self.adapter.scalar_snapshot(1, demand="accel", component="resultant")
        self.assertEqual(values.shape, (4,))

        node0 = np.linalg.norm([2.0, 11.0, 21.0])
        qa = np.linalg.norm([101.0, 201.0, 301.0])
        self.assertAlmostEqual(values[0], node0, places=5)
        self.assertAlmostEqual(values[-1], qa, places=4)

    def test_scalar_snapshot_component_uses_existing_surface_data(self):
        values = self.adapter.scalar_snapshot(2, demand="vel", component="z")
        self.assertTrue(np.allclose(values[:3], [0.3, 0.4, 0.5]))
        self.assertAlmostEqual(values[-1], 10.2, places=5)

    def test_trace_returns_zen_order(self):
        trace = self.adapter.trace(2, "accel")
        self.assertTrue(np.allclose(trace[0], [3.0, 4.0, 5.0]))
        self.assertTrue(np.allclose(trace[1], [12.0, 13.0, 14.0]))
        self.assertTrue(np.allclose(trace[2], [22.0, 23.0, 24.0]))

    def test_nearest_node_id_resolves_qa_point(self):
        qa_node = self.adapter.nearest_node_id(self.adapter.points[-1] + 1e-9)
        self.assertEqual(qa_node, "QA")

    def test_gf_trace_proxies_to_model(self):
        values = self.adapter.gf_trace("QA", 7, "z")
        self.assertTrue(np.allclose(values, [999.0, 7.0, 1.5]))

    def test_node_info_reports_coordinates_and_type(self):
        info = self.adapter.node_info(1)
        self.assertEqual(info["node_id"], 1)
        self.assertEqual(info["type"], "internal")
        self.assertTrue(np.allclose(info["xyz_km"], [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(info["xyz_m"], [1000.0, 0.0, 0.0]))

    def test_scalar_series_uses_cache_after_first_build(self):
        # Warm the series explicitly — _build_scalar_series loops over all 3
        # time steps calling get_surface_snapshot once per step for component "z".
        series = self.adapter.scalar_series("accel", "z")
        first_calls = self.model.surface_calls  # should be 3 (one per timestep)

        # Both snapshots should now be pure memory slices — zero extra I/O.
        first = self.adapter.scalar_snapshot(0, demand="accel", component="z")
        second = self.adapter.scalar_snapshot(2, demand="accel", component="z")

        self.assertEqual(first_calls, 3)
        self.assertEqual(self.model.surface_calls, first_calls)

        # Numerical correctness: node-data for node 0 at t=0 is [1,2,3]*1.0
        self.assertTrue(np.allclose(first[:3], [1.0, 2.0, 3.0]))
        # Node-data for node 0 at t=2 is [3,4,5]*1.0
        self.assertTrue(np.allclose(second[:3], [3.0, 4.0, 5.0]))
        self.assertGreaterEqual(self.adapter.cache_info["hits"], 2)

    def test_resultant_cache_warms_once(self):
        # Warming resultant requires 3 surface-snapshot calls per timestep (z, e, n)
        # across 3 timesteps → 9 total.
        self.adapter.scalar_series("vel", "resultant")
        first_calls = self.model.surface_calls

        # Subsequent snapshots must hit the cache with no additional I/O.
        self.adapter.scalar_snapshot(1, demand="vel", component="resultant")
        self.adapter.scalar_snapshot(2, demand="vel", component="resultant")

        self.assertEqual(first_calls, 9)
        self.assertEqual(self.model.surface_calls, first_calls)

    def test_scalar_snapshot_single_frame_without_cache(self):
        # scalar_snapshot without a pre-warmed series should call get_surface_snapshot
        # exactly once for a single-component demand.
        before = self.model.surface_calls
        self.adapter.scalar_snapshot(1, demand="accel", component="z")
        self.assertEqual(self.model.surface_calls, before + 1)

    def test_cache_fast_path_increments_hit_counter(self):
        # Pre-warm via scalar_series, then verify the hit counter tracks reuse.
        self.adapter.scalar_series("accel", "z")
        before_hits = self.adapter.cache_info["hits"]

        for _ in range(5):
            self.adapter.scalar_snapshot(0, demand="accel", component="z")

        self.assertEqual(self.adapter.cache_info["hits"], before_hits + 5)

    def test_visibility_mask_all_shown(self):
        mask = self.adapter.visibility_mask(
            show_internal=True, show_external=True, show_qa=True
        )
        self.assertEqual(mask.sum(), 4)  # 3 nodes + 1 QA

    def test_visibility_mask_exclude_internal(self):
        mask = self.adapter.visibility_mask(
            show_internal=False, show_external=True, show_qa=True
        )
        # Node 1 is internal, so 3 display points remain (nodes 0, 2, QA)
        self.assertEqual(mask.sum(), 3)
        self.assertFalse(mask[1])  # internal node filtered

    def test_visibility_mask_exclude_qa(self):
        mask = self.adapter.visibility_mask(
            show_internal=True, show_external=True, show_qa=False
        )
        self.assertFalse(mask[-1])  # last entry is QA
        self.assertEqual(mask.sum(), 3)

    def test_visible_scalars_subset_matches_mask(self):
        values = self.adapter.scalar_snapshot(0, "accel", "resultant")
        visible = self.adapter.visible_scalars(
            values, show_internal=False, show_external=True, show_qa=True
        )
        # Only external (node 0, 2) + QA remain → 3 values
        self.assertEqual(visible.shape, (3,))

    def test_cache_info_returns_bytes_and_entries(self):
        info_before = self.adapter.cache_info
        self.assertEqual(info_before["entries"], 0)
        self.assertEqual(info_before["bytes"], 0)

        self.adapter.scalar_series("accel", "z")
        info_after = self.adapter.cache_info
        self.assertEqual(info_after["entries"], 1)
        self.assertGreater(info_after["bytes"], 0)

    def test_default_scalar_limits_returns_valid_range(self):
        vmin, vmax = self.adapter.default_scalar_limits("accel", "resultant")
        self.assertLessEqual(vmin, vmax)
        self.assertGreaterEqual(vmax, 0.0)

    # ── 3-D Warp / Real Motion ────────────────────────────────────────────

    def test_displacement_snapshot_shape(self):
        disp = self.adapter.displacement_snapshot(1)
        # (N_display, 3) — 3 nodes + 1 QA = 4 rows, columns = E, N, Z
        self.assertEqual(disp.shape, (4, 3))

    def test_displacement_snapshot_correct_values(self):
        # At t=1, node 0: get_node_data(0, "disp") returns
        #   z=[0.01, 0.02, 0.03], e=[0.10, 0.11, 0.12], n=[0.20, 0.21, 0.22]
        # Column order returned by displacement_snapshot: [E, N, Z]
        disp = self.adapter.displacement_snapshot(1)
        # Node 0 E-col at t=1: 0.11 (index 1 of e-series for node 0)
        self.assertAlmostEqual(disp[0, 0], 0.11, places=5)   # E
        self.assertAlmostEqual(disp[0, 1], 0.21, places=5)   # N
        self.assertAlmostEqual(disp[0, 2], 0.02, places=5)   # Z

    def test_displacement_snapshot_returns_zeros_on_invalid_demand(self):
        adapter = ViewerDataAdapter(self.model)
        # Temporarily patch so "disp" raises
        original = adapter.scalar_snapshot
        def broken(*args, **kwargs):
            raise RuntimeError("no disp data")
        adapter.scalar_snapshot = broken
        disp = adapter.displacement_snapshot(0)
        self.assertEqual(disp.shape, (4, 3))
        self.assertTrue(np.all(disp == 0.0))
        adapter.scalar_snapshot = original

    def test_suggested_warp_scale_positive(self):
        scale = self.adapter.suggested_warp_scale()
        self.assertGreater(scale, 0.0)
        self.assertIsInstance(scale, float)

    def test_suggested_warp_scale_uses_model_vmax(self):
        # Inject a _vmax dict — should use it without any I/O.
        self.model._vmax = {"disp": {"resultant": 0.05, "z": 0.04}}
        calls_before = self.model.surface_calls
        scale = self.adapter.suggested_warp_scale()
        self.assertEqual(self.model.surface_calls, calls_before)
        self.assertGreater(scale, 0.0)

    def test_window_mask_uses_direct_hdf5_path(self):
        model = FakeModel()
        model.filename = "fake.h5"
        model._data_grp = "Data"
        model._qa_grp = "QA"
        model._window_mask = np.array([False, True, True, False], dtype=bool)
        model.time = np.array([1.0, 2.0], dtype=float)

        data = np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
                [40.0, 41.0, 42.0, 43.0],
                [50.0, 51.0, 52.0, 53.0],
                [60.0, 61.0, 62.0, 63.0],
                [70.0, 71.0, 72.0, 73.0],
                [80.0, 81.0, 82.0, 83.0],
                [90.0, 91.0, 92.0, 93.0],
            ],
            dtype=np.float32,
        )
        qa = np.array(
            [
                [100.0, 101.0, 102.0, 103.0],
                [200.0, 201.0, 202.0, 203.0],
                [300.0, 301.0, 302.0, 303.0],
            ],
            dtype=np.float32,
        )
        fake_handle = FakeH5File(
            {
                "Data/acceleration": FakeDataset(data),
                "QA/acceleration": FakeDataset(qa),
            }
        )
        fake_h5py = type("FakeH5py", (), {"File": lambda *args, **kwargs: fake_handle})

        adapter = ViewerDataAdapter(model)
        with mock.patch.dict(sys.modules, {"h5py": fake_h5py}):
            values = adapter.scalar_snapshot(0, demand="accel", component="z")

        self.assertEqual(model.surface_calls, 0)
        self.assertTrue(np.allclose(values[:3], [31.0, 61.0, 91.0]))
        self.assertAlmostEqual(values[-1], 301.0)


if __name__ == "__main__":
    unittest.main()
