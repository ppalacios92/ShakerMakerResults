# ShakerMakerResults

`ShakerMakerResults` is a Python package for reading, querying, plotting, comparing, and interactively visualizing HDF5 results produced by **ShakerMaker**.

This project is a **results reader and visualization toolkit**. It does **not** generate `.h5drm` files or Green Function databases. Those files must come from a ShakerMaker workflow associated with the work of **Prof. Jose Abell** and collaborators.

The package is designed to work with:
- DRM-style HDF5 results (`.h5drm`)
- Green Function databases (`*_gf.h5`)
- Green Function mapping files (`*_map.h5`)

---

## What This Package Does

- Reads **ShakerMaker** HDF5 outputs into a single `ShakerMakerData` object
- Queries node histories, QA histories, surface snapshots, and Green Functions
- Creates time-windowed and resampled derived models
- Produces domain, node, surface, spectral, Arias, GF, and animation plots
- Compares multiple models in a common plotting interface
- Provides an optional interactive Qt/PyVista viewer

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/ppalacios92/ShakerMakerResults.git
cd ShakerMakerResults
pip install -e .
```

To install the optional interactive viewer:

```bash
pip install -e .[viewer]
```

(pyvista, vtk, pyvistaqt, qtpy dependencies)

### Core dependencies

Installed by default:
- `h5py`
- `matplotlib`
- `numpy`
- `scipy`
- `numba`
- `psutil`

### Optional runtime tools

Some features rely on additional tools outside the base dependency set:
- `ffmpeg` for video export
- `EarthquakeSignal` for Arias intensity routines
Repo: https://github.com/ppalacios92/EarthquakeSignal
- `folium` and geospatial utilities for map-based plotting

---

## Input Files

`ShakerMakerResults` expects files generated elsewhere, typically from a ShakerMaker workflow:

```text
surface_case.h5drm
greensfunctions_database_surface_gf.h5
greensfunctions_database_surface_map.h5
```

Typical usage starts by loading a result file and, when needed, the GF and map files:

```python
from ShakerMakerResults import ShakerMakerData

model = ShakerMakerData("surface_case.h5drm")
model.load_gf_database("greensfunctions_database_surface_gf.h5")
model.load_map("greensfunctions_database_surface_map.h5")
```

---

## Recommended Naming Convention

Use a simple naming pattern throughout your scripts:

- original object: `model`
- time-windowed object: `model_window`
- resampled object: `model_resample`

Examples:

```python
model = ShakerMakerData("surface_case.h5drm")

model_window = model.get_window(t_start=0.0, t_end=40.0)

model_resample = model.resample(dt=0.01)
```

For the plotting examples below, use `model_window` unless a different object is explicitly needed.

---

## Quick Start

```python
from ShakerMakerResults import ShakerMakerData

model = ShakerMakerData("surface_case.h5drm")
model.load_gf_database("greensfunctions_database_surface_gf.h5")
model.load_map("greensfunctions_database_surface_map.h5")

model_window = model.get_window(t_start=0.0, t_end=40.0)
model_resample = model.resample(dt=0.01)

acc_node = model.get_node_data(node_id=10, data_type="accel")
vel_qa = model.get_qa_data(data_type="vel")
gf_z = model.get_gf(node_id=10, subfault_id=0, component="z")
snapshot = model.get_surface_snapshot(time_idx=100, component="z", data_type="vel")
```

---

## Core Data API

### Create a model

```python
from ShakerMakerResults import ShakerMakerData

model = ShakerMakerData("surface_case.h5drm")
```

### Load GF and map

```python
model.load_gf_database("greensfunctions_database_surface_gf.h5")
model.load_map("greensfunctions_database_surface_map.h5")
```

### Build derived models

```python
model_window = model.get_window(t_start=0.0, t_end=40.0)

model_resample = model.resample(dt=0.01)
```

### Direct data access

```python
acc_node = model.get_node_data(node_id=10, data_type="accel")

vel_qa = model.get_qa_data(data_type="vel")

gf_z = model.get_gf(node_id=10, subfault_id=0, component="z")

```

### Export

```python
model.write_h5drm(name="exported_case")
```
Still working on it!

---

## Single-Model Plotting

All methods below are available directly from `ShakerMakerData`.

### Domain methods

`plot_domain`: Plot the domain geometry, QA node, and optional calculated-node overlay.

```python
model_window.plot_domain(
    xyz_origin=None,
    label_nodes=False,
    show_calculated=False,
    figsize=(8, 6),
    axis_equal=False,
)
```
| <img src="docs/images/domain.png" width="500"/> |
|:---:|

`plot_domain_calculated_t0`: Plot the full domain colored by GF `t0` for a selected subfault.

```python
model_window.plot_domain_calculated_t0(
    subfault=0,
    xyz_origin=None,
    show_calculated_only=False,
    figsize=(8, 6),
    axis_equal=True,
    cmap="viridis",
)
```
| <img src="docs/images/t0.png" width="500"/> |
|:---:|

`plot_gf_connections`: Show donor/receiver GF relationships for one node.

```python
model_window.plot_gf_connections(
    node_id=10,
    xyz_origin=None,
    label_nodes=False,
    figsize=(8, 6),
    axis_equal=False,
)
```
| <img src="docs/images/gf_con.png" width="500"/> |
|:---:|

### Node methods

`plot_node_response`: Plot time histories for one or more nodes.

```python
model_window.plot_node_response(
    node_id=None,
    target_pos=None,
    xlim=None,
    data_type="vel",
    figsize=(10, 8),
    factor=1.0,
    filtered=False,
)
```
| <img src="docs/images/node_response_vel.png" width="500"/> |
|:---:|

`plot_node_gf`: Plot Green Function time histories, optionally rotated to physical components.

```python
model_window.plot_node_gf(
    node_id=None,
    target_pos=None,
    xlim=None,
    subfault=0,
    figsize=(8, 10),
    ffsp_source=None,
    strikes=None,
    dips=None,
    rakes=None,
    src_x=None,
    src_y=None,
    internal_ref=None,
    external_coord=None,
)
```

`plot_node_tensor_gf`: Plot the full 9-component GF tensor.

```python
model_window.plot_node_tensor_gf(
    node_id=None,
    target_pos=None,
    xlim=None,
    subfault=0,
    figsize=(10, 8),
)
```
| <img src="docs/images/tensor.png" width="500"/> |
|:---:|


`plot_node_newmark`: Plot Newmark response spectra for one or more nodes.

```python
model_window.plot_node_newmark(
    node_id=None,
    target_pos=None,
    xlim=None,
    data_type="accel",
    figsize=(8, 10),
    factor=1.0,
    filtered=False,
    spectral_type="PSa",
)
```
| <img src="docs/images/newmark.png" width="500"/> |
|:---:|


`plot_node_arias`: Plot Arias intensity curves for one or more nodes.

```python
model_window.plot_node_arias(
    node_id=None,
    target_pos=None,
    data_type="accel",
    xlim=None,
    figsize=(10, 8),
    factor=1.0,
)
```
| <img src="docs/images/arias.png" width="500"/> |
|:---:|

### Surface methods

`plot_surface`: Plot a 3D scatter snapshot of the domain at a selected time.

```python
model_window.plot_surface(
    time=0.0,
    component="z",
    data_type="vel",
    cmap="RdBu_r",
    figsize=(12, 8),
    elev=30,
    azim=-60,
    s=20,
    alpha=0.85,
    axis_equal=False,
    interpolate=False,
    interp_method="linear",
    interp_resolution=300,
)
```
| <img src="docs/images/surface.png" width="500"/> |
|:---:|

`plot_surface_newmark`: Plot a 3D map of spectral values at one target period.

```python
model_window.plot_surface_newmark(
    T_target=0.0,
    component="z",
    data_type="accel",
    spectral_type="PSa",
    factor=1.0,
    cmap="hot_r",
    figsize=(12, 8),
    elev=30,
    azim=-60,
    s=20,
    alpha=0.85,
    axis_equal=False,
    n_jobs=-1,
)
```
| <img src="docs/images/surface_newmark.png" width="500"/> |
|:---:|

`plot_surface_arias`: Plot a 3D map of Arias intensity across the domain.

```python
model_window.plot_surface_arias(
    component="z",
    data_type="accel",
    factor=1.0,
    cmap="hot_r",
    figsize=(12, 8),
    elev=30,
    azim=-60,
    s=20,
    alpha=0.85,
    axis_equal=False,
    n_jobs=-1,
)
```
| <img src="docs/images/surface_arias.png" width="500"/> |
|:---:|

### Animation methods

`create_animation`: Create a full-domain 3D animation.

```python
model_window.create_animation(
    time_start=0.0,
    time_end=None,
    n_frames=50,
    component="z",
    data_type="vel",
    cmap="RdBu_r",
    figsize=(12, 8),
    dpi=100,
    fps=10,
    elev=30,
    azim=-60,
    s=20,
    alpha=0.85,
    ffmpeg_path=None,
    output_dir="animation",
    output_video="animation.mp4",
    axis_equal=True,
    vmax_from_range=False,
)
```

`create_animation_plane`: Create a 3D animation for a planar slice.

```python
model_window.create_animation_plane(
    plane="xy",
    plane_value=0.0,
    time_start=0.0,
    time_end=None,
    n_frames=50,
    component="z",
    data_type="vel",
    cmap="RdBu_r",
    figsize=(12, 8),
    dpi=100,
    fps=10,
    elev=30,
    azim=-60,
    s=50,
    alpha=0.85,
    ffmpeg_path=None,
    output_dir="animation_plane",
    output_video="animation_plane.mp4",
    vmax_from_range=False,
    axis_equal=True,
)
```


---

## Multi-Model Comparison

These functions compare several models in one figure.

```python
from ShakerMakerResults import (
    plot_models_response,
    plot_models_newmark,
    plot_models_gf,
    plot_models_tensor_gf,
    plot_models_domain,
    plot_models_arias,
    compare_node_response,
    compare_spectra,
)
```

`plot_models_response`: Compare time histories from multiple models.

```python
plot_models_response(
    models=[model_window, model_window],
    node_ids=[10, 10],
    target_pos=None,
    data_type="vel",
    xlim=None,
    figsize=(10, 8),
    factor=1.0,
)
```

`plot_models_newmark`: Compare response spectra from multiple models.

```python
plot_models_newmark(
    models=[model_window, model_window],
    node_ids=[10, 10],
    target_pos=None,
    data_type="accel",
    spectral_type="PSa",
    xlim=None,
    figsize=(8, 10),
    factor=1.0,
)
```

`plot_models_gf`: Compare Green Function traces from multiple models.

```python
plot_models_gf(
    models=[model_window, model_window],
    node_ids=[10, 10],
    target_pos=None,
    subfault=0,
    xlim=None,
    figsize=(8, 10),
    factor=1.0,
    ffsp_source=None,
    strikes=None,
    dips=None,
    rakes=None,
    src_x=None,
    src_y=None,
    internal_ref=None,
    external_coord=None,
)
```

`plot_models_tensor_gf`: Compare the full GF tensor from multiple models.

```python
plot_models_tensor_gf(
    models=[model_window, model_window],
    node_ids=[10, 10],
    target_pos=None,
    subfault=0,
    xlim=None,
    figsize=(12, 10),
    factor=1.0,
)
```

`plot_models_domain`: Compare multiple domains in one 3D plot.

```python
plot_models_domain(
    models=[model_window, model_window],
    xlim=None,
    ylim=None,
    zlim=None,
    label_nodes=False,
    show="all",
    show_nodes=True,
    show_cubes=True,
    axis_equal=True,
    figsize=(10, 8),
)
```

`plot_models_arias`: Compare Arias intensity curves from multiple models.

```python
plot_models_arias(
    models=[model_window, model_window],
    node_ids=[10, 10],
    target_pos=None,
    data_type="accel",
    xlim=None,
    figsize=(10, 8),
    factor=1.0,
)
```

`compare_node_response`: Compute time-history similarity metrics against a reference model.

```python
compare_node_response(
    models=[model_window, model_window],
    node_id=[10, 10],
    data_type="vel",
    reference_index=0,
    filtered=False,
)
```

`compare_spectra`: Compute spectral similarity metrics against a reference model.

```python
compare_spectra(
    models=[model_window, model_window],
    node_id=[10, 10],
    data_type="accel",
    spectral_type="PSa",
    reference_index=0,
    filtered=False,
)
```

---

## Analysis Helpers

`NewmarkSpectrumAnalyzer`: Compute spectra directly from a signal array.

```python
from ShakerMakerResults import NewmarkSpectrumAnalyzer

spectrum = NewmarkSpectrumAnalyzer.compute(acc, dt)
```

`compute_vmax`: Compute or refresh the cached global maxima used by plotting utilities.

```python
from ShakerMakerResults import compute_vmax

compute_vmax(model)
```

---

## Minimal Viewer Note

The package also includes an optional interactive viewer.

Install the viewer extras first:

```bash
pip install -e .[viewer]
```

Then launch it from a model object:

```python
model_window.viewer()
```

| <img src="docs/images/viewer.png" width="500"/> |
|:---:|

The viewer is intended for interactive inspection of the same data already accessible from the plotting API. A dedicated viewer section can be expanded later.

---

## Practical Notes

- Use `model` for the original object.
- Use `model_window` for time-windowed analysis and plotting.
- Use `model_resample` when a different `dt` is needed.
- Load both `GF` and `MAP` before using GF-dependent tools.
- `plot_domain_calculated_t0` requires GF `t0` and the GF map.
- Video export requires `ffmpeg`.
- Arias intensity routines require `EarthquakeSignal` at runtime.

---

## License

This project is distributed under the MIT License.
