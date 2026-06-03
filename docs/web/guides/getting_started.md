# Getting started

This page takes you from a clean environment to your first plot.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/ppalacios92/ShakerMakerResults.git
cd ShakerMakerResults
pip install -e .
```

To add the optional interactive viewer (PyVista / Qt stack):

```bash
pip install -e .[viewer]
```

### Dependencies

| Group | Packages | Installed when |
|---|---|---|
| **Core** | `h5py`, `matplotlib`, `numpy`, `scipy`, `numba`, `psutil` | always |
| **Viewer** | `pyvista`, `vtk`, `pyvistaqt`, `qtpy`, `PyQt5` | `.[viewer]` |
| **Runtime tools** | `ffmpeg` (video), [`EarthquakeSignal`](https://github.com/ppalacios92/EarthquakeSignal) (Arias), `folium` + geo utils (maps) | as needed |

!!! note "Optional, lazy imports"
    The top-level package imports lazily: `import ShakerMakerResults` stays cheap
    even if PyVista / Qt are not installed. The viewer dependencies are only
    pulled in the moment you touch a viewer name (e.g. `model.viewer()`).

## Input files

ShakerMakerResults expects files generated elsewhere, typically by a ShakerMaker
workflow:

```text
surface_case.h5drm                              # DRM-style result
greensfunctions_database_surface_gf.h5          # Green's Function database
greensfunctions_database_surface_map.h5         # GF subfault → slot mapping
```

## Your first model

```python
from ShakerMakerResults import ShakerMakerData

model = ShakerMakerData("surface_case.h5drm")
model.load_gf_database("greensfunctions_database_surface_gf.h5")
model.load_map("greensfunctions_database_surface_map.h5")
```

The file type (DRM box, surface grid, or real station recording) is detected
automatically. GF and map files are optional — load them only when you need
Green's-Function-dependent tools.

## Quick start

```python
# Derived models (lazy — no signal data is read yet)
model_window   = model.get_window(t_start=0.0, t_end=40.0)
model_resample = model.resample(dt=0.01)

# Direct data access
acc_node = model.get_node_data(node_id=10, data_type="accel")   # (3, Nt) [Z,E,N]
vel_qa   = model.get_qa_data(data_type="vel")                   # (3, Nt) [Z,E,N]
gf_z     = model.get_gf(node_id=10, subfault_id=0, component="z")
snapshot = model.get_surface_snapshot(time_idx=100, component="z", data_type="vel")

# A plot straight off the model
model_window.plot_node_response(node_id=10, data_type="vel")
```

## Recommended naming convention

A consistent naming pattern keeps scripts readable:

| Name | Meaning |
|---|---|
| `model` | the original object |
| `model_window` | a time-windowed view |
| `model_resample` | a resampled copy |

```python
model          = ShakerMakerData("surface_case.h5drm")
model_window   = model.get_window(t_start=0.0, t_end=40.0)
model_resample = model.resample(dt=0.01)
```

## Where to go next

- **[The data model](data_model.md)** — what `ShakerMakerData` is and how files map onto it.
- **[Querying data](querying_data.md)** — pulling node, QA, surface, and GF arrays.
- **[Single-model plotting](single_model_plotting.md)** — the full figure catalog.
