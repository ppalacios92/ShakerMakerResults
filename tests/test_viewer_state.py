import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ShakerMakerResults.viewer.state import ViewerState


class ViewerStateTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        state = ViewerState()
        self.assertEqual(state.time_index, 0)
        self.assertEqual(state.demand, "accel")
        self.assertEqual(state.component, "resultant")
        self.assertIsNone(state.selected_node)
        self.assertEqual(state.background, "White")
        self.assertTrue(state.show_scalar_bar)
        self.assertFalse(state.is_playing)
        self.assertEqual(state.playback_speed, 1.0)

    def test_setters_validate_and_clamp(self):
        state = ViewerState()
        self.assertEqual(state.set_time_index(9, max_index=4), 4)
        self.assertEqual(state.set_demand("vel"), "vel")
        self.assertEqual(state.set_component("z"), "z")
        self.assertEqual(state.set_selected_node("QA"), "QA")
        self.assertEqual(state.set_background("Gray"), "Gray")
        self.assertEqual(state.set_colormap("seismic"), "seismic")
        self.assertEqual(state.set_point_size(12), 12.0)
        self.assertFalse(state.set_show_scalar_bar(False))
        self.assertTrue(state.set_playing(True))
        self.assertEqual(state.set_playback_speed(1.5), 1.5)

    def test_invalid_demand_raises(self):
        with self.assertRaises(KeyError):
            ViewerState(demand="pressure")

    def test_invalid_component_raises(self):
        with self.assertRaises(KeyError):
            ViewerState(component="x")

    def test_invalid_background_raises(self):
        with self.assertRaises(KeyError):
            ViewerState(background="Blue")

    def test_invalid_playback_speed_raises(self):
        with self.assertRaises(ValueError):
            ViewerState(playback_speed=0)


if __name__ == "__main__":
    unittest.main()
