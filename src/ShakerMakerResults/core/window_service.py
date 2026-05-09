"""Windowing helpers for :class:`ShakerMakerData`."""

from __future__ import annotations

import numpy as np


def get_window(model, t_start, t_end):
    """Return a time-windowed copy without reading signal data eagerly."""
    new = model.__class__.__new__(model.__class__)
    for a, v in model.__dict__.items():
        setattr(new, a, v)

    t_start = float(t_start)
    t_end = float(t_end)

    n_time_data_initial = len(model.time)
    n_time_gf_initial = len(model.gf_time) if hasattr(model, "gf_time") else 0

    new._gf_window_range = (t_start, t_end)

    mask = (model.time >= t_start) & (model.time <= t_end)
    new._window_mask = mask
    new._n_time_data = int(mask.sum())
    new.time = model.time[mask]

    if hasattr(model, "gf_time"):
        gf_time = np.asarray(model.gf_time)
        if gf_time.size > 0:
            gf_mask = (gf_time >= t_start) & (gf_time <= t_end)
            new._gf_window_mask = gf_mask
            new._n_time_gf = int(gf_mask.sum())
            new.gf_time = gf_time[gf_mask]
        else:
            new._gf_window_mask = np.zeros(0, dtype=bool)
            new._n_time_gf = 0
            new.gf_time = gf_time.copy()

    new.name = model.name
    new.window_label = f"{t_start}-{t_end}s"
    new._node_cache = {}
    new._gf_cache = {}
    new._spectrum_cache = {}
    new._vmax = model._vmax

    print("--" * 50)
    print(f"Window   : {model.name}")
    print(f"  Range  : [{t_start:.3f}, {t_end:.3f}] s")
    print(f"  Data   : {n_time_data_initial} -> {new._n_time_data} samples")

    if new._n_time_data > 0:
        print(f"  Time   : [{model.time[0]:.3f}, {model.time[-1]:.3f}] -> [{new.time[0]:.3f}, {new.time[-1]:.3f}] s")
    else:
        print(f"  Time   : [{model.time[0]:.3f}, {model.time[-1]:.3f}] -> empty")

    if getattr(new, "_has_gf", False):
        print(f"  GF     : {n_time_gf_initial} -> {int(getattr(new, '_n_time_gf', 0))} samples")

    print("--" * 50 + "\n")

    return new


def resample(model, dt):
    """Return a lazily resampled copy of the model."""
    new = model.__class__.__new__(model.__class__)
    for a, v in model.__dict__.items():
        setattr(new, a, v)

    dt = float(dt)

    t_orig = np.arange(model._n_time_data) * model._dt_orig + model._tstart
    gf_orig = np.arange(model._n_time_gf) * model._dt_orig

    new.dt = dt
    new.time = np.arange(t_orig[0], t_orig[-1], dt)

    if len(gf_orig) > 0:
        new.gf_time = np.arange(gf_orig[0], gf_orig[-1], dt)
    else:
        new.gf_time = np.array([])

    new._resample_cache = {"time_orig": t_orig, "gf_time_orig": gf_orig}
    new._node_cache = {}
    new._gf_cache = {}
    new._spectrum_cache = {}
    new._vmax = model._vmax

    print("--" * 50)
    print("Resample")
    print(f"  Data dt  : {model._dt_orig:.6f}s -> {dt:.6f}s")
    print(f"  Data     : {model._n_time_data} -> {len(new.time)} samples")
    print(f"  Time     : [{t_orig[0]:.3f}, {t_orig[-1]:.3f}] s unchanged")

    if len(gf_orig) > 0:
        print(f"  GF       : {model._n_time_gf} -> {len(new.gf_time)} samples")
        print(f"  GF Time  : [{gf_orig[0]:.3f}, {gf_orig[-1]:.3f}] s unchanged")
    else:
        print("  GF       : not loaded")

    print("--" * 50 + "\n")

    return new
