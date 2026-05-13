"""
station_data.py  (root-level shim)
==================================
Backwards-compatible re-export so legacy notebooks keep working unchanged::

    from ShakerMakerResults.station_data import StationData

The canonical home is :mod:`ShakerMakerResults.core.station_data`. New code
should prefer the top-level lazy export
(``from ShakerMakerResults import StationData``).

The smoke test verifies that the shim and the canonical class are the same
object so ``isinstance(...)`` keeps behaving.
"""

from .core.station_data import StationData

__all__ = ["StationData"]
