"""
Single-model plotting helpers.

Each submodule attaches helpers that take a single ``ShakerMakerData``
instance (passed as the first positional argument, called ``self`` for
historical reasons) and produce a matplotlib figure:

* :mod:`node_plots`      -- per-node time histories, GFs and spectra.
* :mod:`domain_plots`    -- 3-D domain overview and GF-coverage plots.
* :mod:`surface_plots`   -- surface PGA / spectral / Arias maps.
* :mod:`animation_plots` -- per-frame PNG sequences + ffmpeg stitching.
* :mod:`map_plots`       -- folium / tile-basemap overlays.

The reader class binds these as methods (``model.plot_node_response`` etc.)
through the trampolines defined in ``core.shakermaker_data``.
"""
