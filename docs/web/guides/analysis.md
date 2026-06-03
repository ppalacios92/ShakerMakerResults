# Analysis helpers

Beyond plotting, the package exposes the numerical engines directly so you can
run them on bare signal arrays.

## Newmark response spectra

### `NewmarkSpectrumAnalyzer.compute(ag, dt, zeta=0.05, max_period=5.01, intervals=0.01)`
Compute the full SDOF response spectrum for a ground motion. Returns a dict with
keys `'T'`, `'PSa'`, `'PSv'`, `'Sd'`, `'Sv'`, `'Sa'`, `'u'`, `'v'`, `'a'`, `'at'`.

```python
from ShakerMakerResults import NewmarkSpectrumAnalyzer

acc = model.get_node_data(node_id=10, data_type="accel")[0]   # Z component
spectrum = NewmarkSpectrumAnalyzer.compute(acc, dt=0.005)
```

| Parameter | Default | Meaning |
|---|---|---|
| `ag` | — | ground acceleration array |
| `dt` | — | time step (s) |
| `zeta` | `0.05` | damping ratio |
| `max_period` | `5.01` | longest period in the spectrum (s) |
| `intervals` | `0.01` | period spacing (s) |

Under the hood the per-period SDOF solver is the Numba-compiled
`solve_newmark(ag, dt, zeta, Tj)`.

## Arias intensity

### `AriasIntensityAnalyzer.compute(signal, dt)`
Compute Arias intensity and 5–95% significant duration, returning
`(IA_percent, t_start, t_end, ia_total, pot_dest)` (the last being the
Araya–Saragoni destructiveness potential).

!!! note "Optional runtime dependency"
    Some Arias routines used by the plotting layer rely on the
    [`EarthquakeSignal`](https://github.com/ppalacios92/EarthquakeSignal) package
    at runtime.

## Color-limit caches (vmax)

### `compute_vmax(model)`
Compute (and persist to a sidecar JSON) the per-component color limits used by
every surface / animation plot, so colorbars stay consistent across frames.

```python
from ShakerMakerResults import compute_vmax

compute_vmax(model)   # populates model._vmax and writes the cache file
```

## Reference

See the [Analysis API](../api/analysis.md).
