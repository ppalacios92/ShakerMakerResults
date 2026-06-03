# API reference

This section is generated automatically from the source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/). Improve a docstring in the
`.py` file and this reference improves with it.

## The public surface

`import ShakerMakerResults` exposes a lazy public API — names are only imported
on first access, so the import stays cheap even without the viewer extras.

| Area | Names |
|---|---|
| **Readers** | `ShakerMakerData`, `DRMData`, `SurfaceData`, `StationData` |
| **Analysis** | `NewmarkSpectrumAnalyzer`, `compute_vmax` |
| **Single-model plotting** | `plot_node_response`, `plot_node_gf`, `plot_node_tensor_gf`, `plot_node_newmark`, `plot_node_arias`, `plot_domain`, `plot_domain_calculated_t0`, `plot_gf_connections`, `plot_calculated_vs_reused`, `plot_surface`, `plot_surface_newmark`, `plot_surface_arias`, `plot_surface_on_map`, `create_animation`, `create_animation_plane`, `create_animation_map` |
| **Comparison** | `plot_models_response`, `plot_models_newmark`, `plot_models_gf`, `plot_models_tensor_gf`, `plot_models_domain`, `plot_models_arias`, `compare_node_response`, `compare_spectra` |
| **Viewer** | `ViewerSession`, `ViewerState`, `ViewerDataAdapter` |

## Pages

<div class="grid cards" markdown>

-   __[Core readers](core.md)__ — `ShakerMakerData` & `StationData`
-   __[Core services](services.md)__ — query / GF / window / filter
-   __[Analysis](analysis.md)__ — Newmark, Arias, vmax
-   __[Plotting](plotting.md)__ — single-model & comparison figures
-   __[Comparison](comparison.md)__ — similarity metrics
-   __[I/O & export](io.md)__ — writing `.h5drm`
-   __[Interactive viewer](viewer.md)__ — session, state, adapter

</div>
