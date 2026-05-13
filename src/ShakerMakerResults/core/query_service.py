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

import h5py
import numpy as np
from scipy.interpolate import interp1d


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
        idx = 3 * node_id
        path = {
            "accel": f"{model._data_grp}/acceleration",
            "vel": f"{model._data_grp}/velocity",
            "disp": f"{model._data_grp}/displacement",
        }[data_type]
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
        data = data[[2, 0, 1], :]
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
        path = {
            "accel": f"{model._qa_grp}/acceleration",
            "vel": f"{model._qa_grp}/velocity",
            "disp": f"{model._qa_grp}/displacement",
        }[data_type]
        with h5py.File(model.filename, "r") as f:
            data = f[path][:]
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
        data = data[[2, 0, 1], :]
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
