# Common parameters glossary

The same handful of inputs appears across most plotting, query, and analysis
methods. Rather than repeat them on every page, they are defined **once** here.
When a method's table just says "see glossary", this is the page it means.

## Selecting what to read

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `node_id` | `int` or `list[int]` | `None` | One or more node **indices** in the DRM box / surface grid. Mutually exclusive with `target_pos`. When both are `None`, a small default set is used. |
| `target_pos` | `(x, y, z)` or `list` of them | `None` | One or more physical positions; the **nearest node** to each is selected automatically. Mutually exclusive with `node_id`. |
| `subfault_id` / `subfault` | `int` | `0` | Which fault subfault to use for Green's-Function queries. |
| `time_idx` | `int` | — | A time-**step index** (integer position in the time vector). |
| `time` | `float` | `0.0` | A wall-clock **time in seconds** (resolved to the nearest step). |

!!! tip "Index vs. position, step vs. seconds"
    `node_id` is an integer index; `target_pos` is a coordinate. Likewise
    `time_idx` is an integer step while `time` is seconds. Pick whichever you
    have on hand — the library resolves the other.

## Signal type & component

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `data_type` | `{"accel", "vel", "disp"}` | varies | Which signal family to use: acceleration, velocity, or displacement. Plot methods default to `"vel"`; spectra/Arias default to `"accel"`. |
| `component` | `{"z", "e", "n"}` (+ `"resultant"`, `g11`–`g33` in the viewer) | `"z"` / `"resultant"` | Which directional component to read. Node/QA arrays are always returned in `[Z, E, N]` order. |
| `spectral_type` | `{"PSa", "Sa", "PSv", "Sv", "Sd"}` | `"PSa"` | Which response-spectrum quantity to plot (pseudo-acceleration, acceleration, pseudo-velocity, velocity, displacement). |
| `filtered` | `bool` | `False` | When `True`, use the band-pass-filtered signals instead of the raw ones. See [filtering](derived_models.md#band-pass-filtering). |

## Scaling & axis limits

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `factor` | `float` | `1.0` | Multiplicative scale applied to every trace before plotting (e.g. unit conversion or visual exaggeration). |
| `xlim` | `(tmin, tmax)` | `None` | Limits for the x-axis (time in seconds, or period for spectra). `None` = auto. |
| `ylim` / `zlim` | `(lo, hi)` | `None` | Axis limits for 3-D domain plots. |

## Figure appearance

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `figsize` | `(w, h)` | per method | Matplotlib figure size, in inches. |
| `cmap` | `str` | per method | Matplotlib colormap name (e.g. `"RdBu_r"`, `"hot_r"`, `"viridis"`). |
| `elev`, `azim` | `float` | `30`, `-60` | 3-D view elevation and azimuth angles, in degrees. |
| `s` | `float` | `20` | Scatter marker size for 3-D point clouds. |
| `alpha` | `float` | `0.85` | Marker opacity (0 transparent → 1 opaque). |
| `axis_equal` | `bool` | varies | Force equal aspect ratio on all axes. |
| `label_nodes` | `bool` | `False` | Draw the node id next to each plotted node. |

## Surface interpolation & parallelism

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `interpolate` | `bool` | `False` | Interpolate scattered node values onto a regular grid (`plot_surface`). |
| `interp_method` | `str` | `"linear"` | Grid interpolation method passed to SciPy. |
| `interp_resolution` | `int` | `300` | Grid resolution per axis when interpolating. |
| `n_jobs` | `int` | `-1` | Parallel workers for per-node spectra/Arias maps. `-1` = all cores. |

## Animation & video

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `time_start`, `time_end` | `float` | `0.0`, `None` | Time range to animate (seconds). `None` end = last frame. |
| `n_frames` | `int` | `50` | Number of frames rendered. |
| `fps` | `int` | `10` | Frames per second in the output video. |
| `dpi` | `int` | `100` | Output resolution. |
| `ffmpeg_path` | `str` | `None` | Explicit path to the `ffmpeg` binary (auto-detected if `None`). |
| `output_dir`, `output_video` | `str` | per method | Where frames and the final MP4 are written. |
| `vmax_from_range` | `bool` | `False` | Derive color limits from the animated range instead of the global cache. |

## Comparison-specific

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `models` | `list` | — | List of `ShakerMakerData` (and/or `StationData`) objects to overlay or compare. |
| `node_ids` | `list[int]` | `None` | One node per model (parallel to `models`). |
| `reference_index` | `int` | `0` | Index in `models` treated as the reference for similarity metrics. |

!!! info "Returned metrics"
    `compare_node_response` and `compare_spectra` return a nested dict
    `{name: {component: {metric: value}}}`, where the metrics are a
    goodness-of-fit score, peak error (%), Pearson correlation, and RMSE.
