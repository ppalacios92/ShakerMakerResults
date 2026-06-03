# The data model

Everything in ShakerMakerResults revolves around one reader class and a small
family of delegated *services*.

## `ShakerMakerData`

The single entry point. It opens a ShakerMaker HDF5 file, auto-detects its
layout, and exposes querying, derivation, plotting, export, and viewer methods.

```python
from ShakerMakerResults import ShakerMakerData

model = ShakerMakerData(filename, dt=None)
```

| Parameter | Type | Meaning |
|---|---|---|
| `filename` | `str` | path to the `.h5drm` / HDF5 result file |
| `dt` | `float`, optional | override the time step stored in the file |

### One class, three roles

`DRMData` and `SurfaceData` are **aliases** of `ShakerMakerData` — the same class
object. The HDF5 layout decides the role at runtime; the alias names exist for
readability and backwards compatibility.

```python
from ShakerMakerResults import ShakerMakerData, DRMData, SurfaceData
# DRMData is ShakerMakerData  ->  True
```

### Loading auxiliary files

### `load_gf_database(h5_path)`
Attach an external Green's Function HDF5 database to the model.

### `load_map(h5_path)`
Attach the GF subfault-to-slot mapping file. Required by `plot_domain_calculated_t0`
and other GF-aware tools.

```python
model.load_gf_database("..._gf.h5")
model.load_map("..._map.h5")
```

!!! tip "Load order"
    Load **both** the GF database and the map before using GF-dependent tools.
    `plot_domain_calculated_t0` additionally needs the GF `t0` values.

## The service layer

`ShakerMakerData` stays thin: most behavior is implemented in focused service
modules under `core/` and called through delegating methods. You normally use the
methods on the model, but the services are public too.

| Service | Module | Responsibility |
|---|---|---|
| Query | `core.query_service` | node / QA / surface array access, cache clearing |
| Green's Functions | `core.gf_service` | GF database + map loading, GF traces and tensors |
| Windowing | `core.window_service` | lazy time windows and resampling |
| Filtering | `core.filter_service` | lazy band-pass filtering |
| Export | `io.export_service` | writing `.h5drm` files back out |
| Analysis | `analysis.*` | Newmark spectra, Arias intensity, vmax caches |

## Caches and memory

### `clear_cache()`
Drop all in-memory caches (node, GF, spectrum) and run a manual garbage
collection pass. Useful in long notebook sessions or batch loops.

```python
model.clear_cache()
```

## Reference

See the [Core readers API](../api/core.md) and [Core services API](../api/services.md).
