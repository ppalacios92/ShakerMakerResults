"""
query_service.py
================
Data-access helpers for :class:`ShakerMakerData`.

These three functions all honour:

* ``model._window_mask`` set by :meth:`get_window`.
* ``model._resample_cache`` set by :meth:`resample` (linear interpolation
  along the time axis).

The output convention for all multi-component readers is **(Z, E, N)** --
the on-disk layout is (E, N, Z) and we reorder it to ``[2, 0, 1]`` before
returning. Plots and the viewer assume Z-up.
"""

from __future__ import annotations

import gc
import warnings

import h5py
import numpy as np
from scipy.interpolate import interp1d

_DATA_PATHS = {
    "accel": "acceleration",
    "vel": "velocity",
    "disp": "displacement",
}


def _read_node_zen(model, node_id, data_type):
    """Read one node's ``(3, Nt)`` trace in ``[Z, E, N]`` order, post-mask/resample."""
    idx = 3 * node_id
    path = f"{model._data_grp}/{_DATA_PATHS[data_type]}"
    with h5py.File(model.filename, "r") as f:
        data = f[path][idx : idx + 3, :]
    if hasattr(model, "_window_mask"):
        data = data[:, model._window_mask]
    elif hasattr(model, "_resample_cache"):
        rs = np.zeros((3, len(model.time)))
        for i in range(3):
            rs[i] = interp1d(
                model._resample_cache["time_orig"],
                data[i],
                kind="linear",
                fill_value="extrapolate",
            )(model.time)
        data = rs
    return data[[2, 0, 1], :]


def _read_qa_zen(model, data_type):
    """Read the QA station ``(3, Nt)`` trace in ``[Z, E, N]`` order, post-mask/resample.

    The QA dataset can be longer than the DRM node datasets (e.g. a QA
    station that was never truncated when the node traces were cut). We
    therefore build the window mask / resample source grid from the QA's
    *own* length using the shared ``dt`` / ``tstart``, instead of reusing
    the node-grid mask in ``model._window_mask`` (which would mismatch).
    """
    path = f"{model._qa_grp}/{_DATA_PATHS[data_type]}"
    with h5py.File(model.filename, "r") as f:
        data = f[path][:]
    qa_time = np.arange(data.shape[1]) * model._dt_orig + model._tstart
    if hasattr(model, "_resample_cache"):
        rs = np.zeros((3, len(model.time)))
        for i in range(3):
            rs[i] = interp1d(
                qa_time,
                data[i],
                kind="linear",
                fill_value="extrapolate",
            )(model.time)
        data = rs
    else:
        # Base or windowed read: QA shares the node dt grid, so select the
        # samples falling inside the active time vector's [t0, t1] span.
        t0, t1 = model.time[0], model.time[-1]
        mask = (qa_time >= t0) & (qa_time <= t1)
        data = data[:, mask]
    return data[[2, 0, 1], :]


def _filtered_trace(model, requested_type, reader):
    """Apply ``model._filter`` to a (3, Nt) trace from ``reader(data_type)``.

    ``reader`` is a callable taking a ``data_type`` string and returning
    the ``(3, Nt)`` ``[Z, E, N]`` array. Used for both nodes and QA.
    """
    from .filter_service import bandpass, derive

    filt = model._filter
    mode = filt["mode"]
    base_type = requested_type if mode == "all" else mode
    base = reader(base_type)
    base = bandpass(base, float(model.dt), filt, time_axis=-1)
    if base_type == requested_type:
        return base
    return derive(base, float(model.dt), base_type, requested_type)


def get_node_data(model, node_id, data_type="accel"):
    """Return a ``(3, Nt)`` array for one node, in ``[Z, E, N]`` order.

    Parameters
    ----------
    model : ShakerMakerData
    node_id : int
        Index in ``model.xyz``.
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``

    Returns
    -------
    np.ndarray, shape (3, Nt)
        Cached in ``model._node_cache`` keyed by ``(node_id, data_type)``.
    """
    key = (node_id, data_type)
    if key not in model._node_cache:
        if hasattr(model, "_filter"):
            data = _filtered_trace(
                model,
                data_type,
                lambda dt: _read_node_zen(model, node_id, dt),
            )
        else:
            data = _read_node_zen(model, node_id, data_type)
        model._node_cache[key] = data
    return model._node_cache[key]


def get_qa_data(model, data_type="accel"):
    """Return a ``(3, Nt)`` array for the QA station, in ``[Z, E, N]`` order.

    Parameters
    ----------
    model : ShakerMakerData
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``

    Returns
    -------
    np.ndarray, shape (3, Nt)

    Raises
    ------
    AttributeError
        If the file is a station-only output (no QA group).
    """
    if model._qa_grp is None:
        raise AttributeError("QA station only available in DRM output files.")
    key = ("qa", data_type)
    if key not in model._node_cache:
        if hasattr(model, "_filter"):
            data = _filtered_trace(
                model,
                data_type,
                lambda dt: _read_qa_zen(model, dt),
            )
        else:
            data = _read_qa_zen(model, data_type)
        model._node_cache[key] = data
    return model._node_cache[key]


def get_surface_snapshot(model, time_idx, component="z", data_type="vel"):
    """Return values of one component for every node at a single time index.

    Parameters
    ----------
    model : ShakerMakerData
    time_idx : int
        Column index into the on-disk dataset (not the resampled ``time``
        vector). Callers normally compute it via
        ``int(np.argmin(np.abs(self.time - t)))``.
    component : {'z', 'e', 'n'}, default ``'z'``
    data_type : {'accel', 'vel', 'disp'}, default ``'vel'``

    Returns
    -------
    np.ndarray, shape (n_nodes,)
        Not cached -- the typical caller iterates over time, so caching
        every snapshot would blow up RAM.
    """
    if hasattr(model, "_filter"):
        warnings.warn(
            "get_surface_snapshot ignores the active _filter: a single "
            "time-step has no time series to band-pass. Use get_node_data "
            "if you need the filtered trace.",
            RuntimeWarning,
            stacklevel=2,
        )
    row = {"e": 0, "n": 1, "z": 2}[component.lower()]
    path = {
        "accel": f"{model._data_grp}/acceleration",
        "vel": f"{model._data_grp}/velocity",
        "disp": f"{model._data_grp}/displacement",
    }[data_type]
    with h5py.File(model.filename, "r") as f:
        return f[path][row::3, time_idx]


def clear_cache(model):
    """Drop all in-memory caches and run a manual garbage collection pass.

    Affects ``_node_cache``, ``_gf_cache`` and ``_spectrum_cache``. Useful
    before opening a second large file in the same notebook session.
    """
    model._node_cache.clear()
    model._gf_cache.clear()
    model._spectrum_cache.clear()
    gc.collect()
    print("Cache cleared.")
