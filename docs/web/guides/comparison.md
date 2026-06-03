# Multi-model comparison

These functions take a **list of models** and either overlay them in one figure
or compute quantitative similarity metrics against a reference. They are
module-level functions (not methods). Shared inputs are defined in the
[Common parameters glossary](parameters.md).

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

## Overlay plots

### `plot_models_response(models, node_ids=None, target_pos=None, data_type="vel", xlim=None, figsize=(10, 8), factor=1.0)`
Overlay time-history traces from several models on one `3×1` figure.

```python
plot_models_response(models=[m1, m2], node_ids=[10, 10], data_type="vel")
```

### `plot_models_newmark(models, node_ids=None, target_pos=None, data_type="accel", spectral_type="PSa", xlim=None, figsize=(8, 10), factor=1.0)`
Overlay Newmark response spectra from several models.

### `plot_models_gf(models, node_ids=None, target_pos=None, subfault=0, xlim=None, figsize=(8, 10), factor=1.0, ffsp_source=None, strikes=None, dips=None, rakes=None, src_x=None, src_y=None, internal_ref=None, external_coord=None)`
Overlay Green's Function traces from several models.

### `plot_models_tensor_gf(models, node_ids=None, target_pos=None, subfault=0, xlim=None, figsize=(12, 10), factor=1.0)`
Overlay the full 9-component GF tensor on a `3×3` figure.

### `plot_models_domain(models, xlim=None, ylim=None, zlim=None, label_nodes=False, show="all", show_nodes=True, show_cubes=True, axis_equal=True, figsize=(10, 8))`
Overlay several model domains in one 3-D plot.

### `plot_models_arias(models, node_ids=None, target_pos=None, data_type="accel", xlim=None, figsize=(10, 8), factor=1.0)`
Overlay Arias intensity curves from several models.

## Similarity metrics

Both metric functions compare each model against a reference model and return a
nested dict `{name: {component: {metric: value}}}`. The four metrics are a
goodness-of-fit score, peak error (%), Pearson correlation, and RMSE.

### `compare_node_response(models, node_id, data_type="vel", reference_index=0, filtered=False)`
Compare time-history signals against a reference.

### `compare_spectra(models, node_id, data_type="accel", spectral_type="PSa", reference_index=0, filtered=False)`
Compare Newmark response spectra against a reference.

```python
metrics = compare_node_response(
    models=[reference, candidate],
    node_id=[10, 10],
    data_type="vel",
    reference_index=0,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `models` | — | list of `ShakerMakerData` / `StationData` |
| `node_id` | — | scalar, per-model list, or list of lists |
| `data_type` | `"vel"` / `"accel"` | signal type |
| `spectral_type` | `"PSa"` | `"PSa"`, `"Sa"`, `"PSv"`, `"Sv"`, `"Sd"` |
| `reference_index` | `0` | index of the reference model |
| `filtered` | `False` | use filtered signals |

!!! tip "Comparing against a real record"
    `StationData` objects can sit in the same `models` list, so a simulation can
    be compared directly against a recorded station. See
    [Real station recordings](stations.md).

## Reference

See the [Comparison API](../api/comparison.md) and [Plotting API](../api/plotting.md).
