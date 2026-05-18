"""
Comparison plotting helpers.

This package groups multi-model plots:

* :mod:`response_plots` -- time-history and Newmark spectrum overlays.
* :mod:`gf_plots`       -- Green-function and 9-component tensor overlays.
* :mod:`domain_plots`   -- multi-domain 3-D scatter.
* :mod:`arias_plots`    -- Arias intensity overlays.
* :mod:`_common`        -- shared label / accessor helpers.

All public functions accept a list of readers (``ShakerMakerData`` and/or
``StationData`` mixed) and a parallel list / list-of-lists of node ids or
positions. The lazy :mod:`ShakerMakerResults.plotting` facade exposes
them under their flat ``plot_models_*`` names.
"""
