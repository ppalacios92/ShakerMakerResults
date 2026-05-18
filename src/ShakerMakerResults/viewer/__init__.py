"""
ShakerMakerResults.viewer
=========================
Interactive Qt + PyVista viewer for ``ShakerMakerData``.

The viewer is **optional** -- the rest of the package works without
PyQt / PyVista installed. We import the public classes at package level
so that ``from ShakerMakerResults.viewer import ViewerSession`` is a
clean entry point, but each class only pulls in the heavy GUI deps when
its ``__init__`` actually needs them (see :func:`._imports.require_viewer_dependencies`).

Public surface
--------------
``ViewerSession``
    The orchestrator. A session owns a ``ViewerDataAdapter`` (data layer)
    and a ``ViewerState`` (display layer) and, when ``show=True``, drives
    a ``ViewerMainWindow`` (Qt). Notebook users typically create one via
    ``model.viewer()`` rather than instantiating directly.

``ViewerState``
    Mutable dataclass holding every per-display knob (demand, component,
    colormap, warp, visibility, playback ...). Validated through
    typed setters.

``ViewerDataAdapter``
    Read / cache layer between the GUI and the underlying
    ``ShakerMakerData``. Owns the HDF5 handle for fast playback, manages
    LRU caches of scalar series, and exposes the (cheap) lookup helpers
    used by the scene / panels.

Internal layout (high-level)
----------------------------
::

    ViewerSession                <-- orchestration / public API
      |
      |-- ViewerDataAdapter      <-- data + cache (always present)
      |-- ViewerState            <-- per-session knobs
      |-- ViewerMainWindow       <-- Qt window (only if show=True)
           |
           |-- TabbedMultiViewArea
           |     |-- ViewPane    <-- one PyVista plotter per pane
           |          |-- ViewerScene    (point cloud, picking, overlays)
           |          |-- PaneDisplayState  (per-pane overrides on top of ViewerState)
           |
           |-- side_panel.ViewerSidePanel  (Apply-to per-section editors)
           |-- toolbars (apply-target, camera, overlays, capture ...)
           |-- docks   (Properties, Appearance, Information, Pipeline)
           |-- trace_panel.* (Spectrum / Arias / Responses / GF)
           |-- rupture_pane.RuptureTab    (FFSP rupture viewer tab)

See ``viewer/README.md`` for the architecture writeup.
"""

from .adapter import ViewerDataAdapter
from .session import ViewerSession
from .state import ViewerState

__all__ = ["ViewerDataAdapter", "ViewerSession", "ViewerState"]
