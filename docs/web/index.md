---
hide:
  - navigation
---

<!-- HERO: the wordmark is an <h1> so Material does NOT inject an ugly
     auto "Home" title above it. -->
<div class="sm-hero" markdown>
<img class="sm-hero__mark" src="assets/logo.svg" alt="ShakerMakerResults mark" />
<div>
  <h1 class="sm-hero__word">ShakerMakerResults</h1>
  <div class="sm-hero__sub">Read · Query · Plot · Compare · Visualize</div>
</div>
</div>

**ShakerMakerResults** is a Python package for reading, querying, plotting,
comparing, and interactively visualizing the HDF5 results produced by
[**ShakerMaker**](https://github.com/ppalacios92/ShakerMaker) — a physics-based
ground-motion simulation tool.

It is, deliberately, a **results reader and visualization toolkit**. It does
**not** generate `.h5drm` files or Green's Function databases: those come from a
ShakerMaker workflow associated with the work of **Prof. José Abell** and
collaborators. ShakerMakerResults picks up where the simulation ends.

!!! info "What it reads"
    The package targets three families of files: DRM-style HDF5 results
    (`.h5drm`), Green's Function databases (`*_gf.h5`), and Green's Function
    mapping files (`*_map.h5`). The HDF5 layout (DRM box vs. surface grid vs.
    real station recording) is detected automatically.

## Key features

- Reads **ShakerMaker** HDF5 outputs into a single `ShakerMakerData` object.
- Queries node histories, QA histories, surface snapshots, and Green's Functions.
- Builds *lazy* derived models: time windows, resampling, and band-pass filtering.
- Produces domain, node, surface, spectral, Arias, GF, and animation plots.
- Compares multiple models in a common plotting and metrics interface.
- Computes Newmark response spectra and Arias intensity quantities.
- Ships an optional interactive **Qt + PyVista** viewer for 3-D inspection.

## How it works

| Stage | You do | Tool |
|---|---|---|
| **Load** | open an `.h5drm` (and optionally GF + map) | `ShakerMakerData(...)` |
| **Derive** | window, resample, or filter the signals | `get_window` · `resample` · `apply_filter` |
| **Query** | pull node / QA / surface / GF arrays | `get_node_data` · `get_qa_data` · `get_gf` |
| **Plot** | domain, node, surface, animation figures | `plot_*` · `create_animation*` |
| **Compare** | overlay several models + metrics | `plot_models_*` · `compare_*` |
| **Visualize** | interactive 3-D scene | `model.viewer()` |

## A minimal example

```python
from ShakerMakerResults import ShakerMakerData

# Load a result and (optionally) its Green's Function database + map
model = ShakerMakerData("surface_case.h5drm")
model.load_gf_database("greensfunctions_database_surface_gf.h5")
model.load_map("greensfunctions_database_surface_map.h5")

# A lazy 0–40 s window, then plot one node's velocity history
model_window = model.get_window(t_start=0.0, t_end=40.0)
model_window.plot_node_response(node_id=10, data_type="vel")
```

![Node velocity response](assets/images/node_response_vel.png){ width="560" }

---

<!-- The "Where do you want to start?" cards go AT THE END — a friendly
     launchpad after the reader knows what the project is. -->

## Where do you want to start?

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } &nbsp; __[Getting started](guides/getting_started.md)__

    ---

    *I'm new — orient me.*

    Install the package and load your first result file.

-   :material-rocket-launch:{ .lg .middle } &nbsp; __[Examples](examples/index.md)__

    ---

    *Show me a working model.*

    Notebooks and scripts from real DRM / surface workflows.

-   :material-chart-bell-curve:{ .lg .middle } &nbsp; __[Single-model plotting](guides/single_model_plotting.md)__

    ---

    *I want figures fast.*

    Domain, node, surface, spectra, Arias, and animations.

-   :material-cube-scan:{ .lg .middle } &nbsp; __[Interactive viewer](guides/viewer.md)__

    ---

    *Let me explore in 3-D.*

    The optional Qt + PyVista scene driven from `model.viewer()`.

-   :material-vector-difference:{ .lg .middle } &nbsp; __[Compare models](guides/comparison.md)__

    ---

    *How close are two runs?*

    Overlay traces and spectra, and compute similarity metrics.

-   :material-book-open-variant:{ .lg .middle } &nbsp; __[API reference](api/index.md)__

    ---

    *Look up a class or method.*

    Auto-generated from the source docstrings.

</div>

---

**Authors:** Patricio Palacios B. · Nicolas Mora Bowen ·
[@ppalacios92](https://github.com/ppalacios92) ·
[@nmorabowen](https://github.com/nmorabowen)
