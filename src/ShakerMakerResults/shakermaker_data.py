"""
shakermaker_data.py  (root-level shim)
======================================
Backwards-compatible re-export of the core readers so legacy notebooks keep
working without rewrites::

    from ShakerMakerResults.shakermaker_data import ShakerMakerData
    from ShakerMakerResults.shakermaker_data import DRMData, SurfaceData

The canonical home of these classes is
:mod:`ShakerMakerResults.core.shakermaker_data`. New code should prefer the
top-level lazy export (``from ShakerMakerResults import ShakerMakerData``) or
the canonical submodule -- not this shim.

``DRMData`` and ``SurfaceData`` are intentionally **aliases** of
``ShakerMakerData`` (same class object). The HDF5 layout decides the role at
runtime; the alias just keeps notebook code self-documenting.

The smoke test guarantees that all three names resolve to the same class
across the shim, the canonical module, and the lazy package facade.
"""

from .core.shakermaker_data import DRMData, ShakerMakerData, SurfaceData

__all__ = ["ShakerMakerData", "DRMData", "SurfaceData"]
