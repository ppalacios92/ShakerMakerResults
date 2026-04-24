import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ShakerMakerResults as smr


class PackageExportTests(unittest.TestCase):
    def test_viewer_exports_are_available_lazily(self):
        self.assertTrue(hasattr(smr, "ViewerSession"))
        self.assertTrue(hasattr(smr, "ViewerState"))
        self.assertTrue(hasattr(smr, "ViewerDataAdapter"))


if __name__ == "__main__":
    unittest.main()
