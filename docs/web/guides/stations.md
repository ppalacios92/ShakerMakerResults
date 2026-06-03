# Real station recordings

`StationData` reads **recorded** seismic station data (NPZ or HDF5) and exposes a
plotting / analysis surface that mirrors `ShakerMakerData`. This lets you put a
real record side by side with a simulation in the comparison tools.

```python
from ShakerMakerResults import StationData

sta = StationData(filepath, station_index=0, name=None)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `filepath` | `str` | — | path to an NPZ or HDF5 recording |
| `station_index` | `int` | `0` | which station to read from a multi-station file |
| `name` | `str`, optional | `None` | display name |

## Signal components

Each is a `(z, e, n)` tuple, derived on demand:

| Property | Meaning |
|---|---|
| `velocity` | raw velocity (Z, E, N) |
| `acceleration` | finite-difference acceleration |
| `displacement` | cumulative-trapezoid displacement |
| `velocity_filtered` | filtered velocity (falls back to raw) |
| `acceleration_filtered` | filtered acceleration |
| `displacement_filtered` | filtered displacement |

## Windowing and filtering

### `get_window(tmin, tmax)`
Return a new `StationData` trimmed to `[tmin, tmax]`.

### `apply_filter(filter_type="bandpass", freqmin=0.25, freqmax=50.0, corners=4, zerophase=True)`
Apply an ObsPy filter to the velocity components in place.

## Spectra and Fourier

### `get_fourier(component="velocity", filtered=False)`
Return `(freqs, z_amp, e_amp, n_amp)` for the chosen component.

### `get_newmark(filtered=False)`
Compute (or return cached) Newmark response spectra.

## Plotting

Time-series:

| Method | Default factor |
|---|---|
| `plot_velocity(xlim=None, factor=1.0, figsize=(10, 8))` | 1.0 |
| `plot_acceleration(xlim=None, factor=9.81, figsize=(10, 8))` | 9.81 |
| `plot_displacement(xlim=None, factor=1.0, figsize=(10, 8))` | 1.0 |
| `plot_velocity_filtered(...)` · `plot_acceleration_filtered(...)` · `plot_displacement_filtered(...)` | filtered variants |

Fourier and spectra:

| Method | Purpose |
|---|---|
| `plot_fourier(component="acceleration", xlim=None, factor=9.81, figsize=(12, 4))` | raw amplitude spectrum |
| `plot_fourier_filtered(...)` | filtered amplitude spectrum |
| `plot_fourier_comparison(...)` | raw vs filtered side by side |
| `plot_newmark(xlim=None, filtered=False, figsize=(12, 4), factor=1.0, spectral_type="PSa")` | Newmark spectra (Z / E / N) |

## Comparing a record with a simulation

```python
from ShakerMakerResults import compare_node_response

metrics = compare_node_response(
    models=[sta, model_window],   # recorded vs simulated
    node_id=["QA", 10],
    data_type="vel",
    reference_index=0,            # the record is the reference
)
```

## Reference

See the [Core readers API](../api/core.md).
