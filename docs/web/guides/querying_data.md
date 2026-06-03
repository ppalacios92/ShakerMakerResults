# Querying data

Once a model is loaded, these methods return raw NumPy arrays. All node/QA
arrays are shaped `(3, Nt)` and ordered **`[Z, E, N]`**.

## Node and QA histories

### `get_node_data(node_id, data_type="accel")`
Return a `(3, Nt)` array for one node, ordered `[Z, E, N]`.

### `get_qa_data(data_type="accel")`
Return a `(3, Nt)` array for the QA (quality-assurance) station.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `node_id` | `int` | — | node index in the DRM box / surface grid |
| `data_type` | `str` | `"accel"` | one of `"accel"`, `"vel"`, `"disp"` |

```python
acc = model.get_node_data(node_id=10, data_type="accel")   # (3, Nt)
vel_qa = model.get_qa_data(data_type="vel")                 # (3, Nt)
z, e, n = acc                                               # unpack components
```

## Surface snapshots

### `get_surface_snapshot(time_idx, component="z", data_type="vel")`
Return one signal component for **every** node at a single time step — the basis
of surface maps and animations.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `time_idx` | `int` | — | time-step index |
| `component` | `str` | `"z"` | `"z"`, `"e"`, `"n"` |
| `data_type` | `str` | `"vel"` | `"accel"`, `"vel"`, `"disp"` |

```python
snap = model.get_surface_snapshot(time_idx=100, component="z", data_type="vel")
```

## Green's Functions

These require a loaded GF database (and usually the map).

### `get_gf(node_id, subfault_id, component="z")`
Return the Green's Function trace for a single `(node, subfault)` pair.

```python
gf_z = model.get_gf(node_id=10, subfault_id=0, component="z")
```

Lower-level GF helpers live in `core.gf_service` and are also reachable directly:

| Function | Purpose |
|---|---|
| `get_gf(model, node_id, subfault_id, component="z")` | single-component GF trace |
| `get_gf_tensor(model, node_id, subfault_id)` | full `(nt, 9)` GF tensor + time metadata |
| `get_gf_time(model, slot)` | GF time vector for a slot (window/resample aware) |
| `load_gf_database(model, h5_path)` | attach a GF database |
| `load_map(model, h5_path)` | attach the GF mapping file |

## Cache management

### `clear_cache()`
Drop node / GF / spectrum caches and collect garbage.

## Reference

See the [Core services API](../api/services.md).
