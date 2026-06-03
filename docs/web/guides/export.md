# Exporting results

You can write a (possibly windowed) model back out to a self-contained `.h5drm`
file — handy for sharing a trimmed record or feeding a downstream solver.

## `write_h5drm(name=None)`
Write the active time window of the model to a new `.h5drm` file. Returns the
absolute path of the written file.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str`, optional | `<stem>_t<start>_<end>.h5drm` | output file name |

```python
model_window = model.get_window(t_start=0.0, t_end=40.0)
path = model_window.write_h5drm(name="exported_case")
print("wrote", path)
```

!!! note "Window-aware"
    The export captures the **current** time window, so window first, then export
    to write only the slice you care about.

## Reference

See the [I/O & export API](../api/io.md).
