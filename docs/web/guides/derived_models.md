# Windowing, resampling & filtering

These methods return **new** `ShakerMakerData` objects that are *lazy*: they
describe a transformation of the original data without reading or copying the
signals until you query or plot them. Chain them freely.

## Time windows

### `get_window(t_start, t_end)`
Return a lazy time-windowed copy of the model.

| Parameter | Type | Meaning |
|---|---|---|
| `t_start` | `float` | window start time (s) |
| `t_end` | `float` | window end time (s) |

```python
model_window = model.get_window(t_start=0.0, t_end=40.0)
```

The window is the recommended object for most plotting and analysis — it keeps
figures focused on the part of the record that matters.

## Resampling

### `resample(dt)`
Return a copy of the model with a different effective time step `dt`.

```python
model_resample = model.resample(dt=0.01)
```

## Band-pass filtering

### `apply_filter(mode="all", freqmin=0.25, freqmax=10.0, corners=4, zerophase=True, apply_gf=False)`
Return a lazy band-pass-filtered copy of the model (Butterworth, via ObsPy).

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `str` | `"all"` | which signals to filter |
| `freqmin` | `float` | `0.25` | low corner frequency (Hz) |
| `freqmax` | `float` | `10.0` | high corner frequency (Hz) |
| `corners` | `int` | `4` | filter order |
| `zerophase` | `bool` | `True` | apply forward + reverse (zero phase) |
| `apply_gf` | `bool` | `False` | also filter Green's Functions |

```python
model_filt = model.apply_filter(freqmin=0.1, freqmax=5.0)
```

## Composing transformations

Because each call returns a model, they compose:

```python
view = (
    model
    .get_window(t_start=0.0, t_end=40.0)
    .resample(dt=0.01)
    .apply_filter(freqmin=0.1, freqmax=5.0)
)
view.plot_node_response(node_id=10, data_type="vel")
```

## Reference

See the [Core services API](../api/services.md) — `window_service` and
`filter_service`.
