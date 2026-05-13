"""
gf_service.py
=============
Green-Function read / mapping helpers for :class:`ShakerMakerData`.

These functions live outside the reader class on purpose -- the reader
keeps the trampoline methods (``model.load_gf_database(...)``) and we
keep the HDF5 logic here so the class stays focused on metadata and
display geometry.

Two GF layouts are recognised:

* **OP pipeline** -- ``tdata`` dataset (``(slot, time, 9)``), optional
  ``t0``, plus a mapping file with ``pair_to_slot`` for O(1) lookup.
* **Legacy** -- ``GF/sta_N/sub_M/{z,e,n,t,tdata,t0}`` + ``Node_Mapping``.

All public helpers honour :meth:`get_window` (``_window_mask``) and
:meth:`resample` (``_resample_cache``) when attached to ``model``.
"""

from __future__ import annotations

import h5py
import numpy as np
from scipy.interpolate import interp1d


def load_gf_database(model, h5_path):
    """Attach an OP-format GF HDF5 database to ``model``.

    Parameters
    ----------
    model : ShakerMakerData
        Target reader. Mutated in-place: gains ``_gf_h5_path``,
        ``_tdata_shape``, ``_nt_gf``, ``_t0_available``, ``_has_gf=True``
        and a rebuilt ``gf_time``.
    h5_path : str
        Path to the GF file. The dataset ``tdata`` is required; ``t0`` is
        optional (per-slot time offset).

    Returns
    -------
    None
        The function only mutates ``model`` and prints a one-page summary.

    Raises
    ------
    ValueError
        If the file does not contain a ``tdata`` dataset (unsupported
        GF format).
    """
    # sep = "--" * 50
    model._gf_h5_path = h5_path

    with h5py.File(h5_path, "r") as f:
        if "tdata" not in f:
            raise ValueError("Unsupported GF format: expected dataset 'tdata'")

        model._tdata_shape = f["tdata"].shape
        model._nt_gf = model._tdata_shape[1]
        model._t0_available = "t0" in f

    full_gf_time = np.arange(model._nt_gf, dtype=float) * float(model._dt_orig)
    model._n_time_gf = int(model._nt_gf)

    if hasattr(model, "_resample_cache"):
        model._resample_cache["gf_time_orig"] = full_gf_time
        if full_gf_time.size > 0:
            model.gf_time = np.arange(full_gf_time[0], full_gf_time[-1], float(model.dt))
        else:
            model.gf_time = np.array([], dtype=float)
    else:
        model.gf_time = full_gf_time

    if hasattr(model, "_gf_window_range"):
        t_start, t_end = model._gf_window_range
        gf_mask = (full_gf_time >= t_start) & (full_gf_time <= t_end)
        model._gf_window_mask = gf_mask
        model.gf_time = full_gf_time[gf_mask]
        model._n_time_gf = int(gf_mask.sum())

    model._has_gf = True

    gf_slots = int(model._tdata_shape[0])
    gf_steps = int(model._nt_gf)
    gf_components = int(model._tdata_shape[2]) if len(model._tdata_shape) > 2 else None

    print( "--" * 50)
    print(f"Green Functions : {h5_path}")
    print(f"  Shape    : {model._tdata_shape}")
    print(f"  GF       : steps={gf_steps}  |  slots={gf_slots}")
    if gf_components is not None:
        print(f"  Tensor   : components={gf_components}")
    print(f"  Time     : dt={model._dt_orig}s  |  t=[{full_gf_time[0]:.3f}, {full_gf_time[-1]:.3f}]s")
    print(f"  t0       : {'yes' if model._t0_available else 'no'}")
    print(f"  Map      : not loaded yet (subfault count unavailable)")
    print( "--" * 50 + "\n")


def load_map(model, h5_path):
    """Attach the OP-format GF mapping file to ``model``.

    Parameters
    ----------
    model : ShakerMakerData
        Target reader. Mutated in-place: gains ``pairs_to_compute``,
        ``pair_to_slot``, the three ``(dh|zsrc|zrec)_of_pairs`` arrays,
        the ``delta_*`` tolerances, ``_nsources(_db)`` and
        ``_has_map=True``.
    h5_path : str
        Path to the mapping HDF5 file.

    Returns
    -------
    None
        Mutates ``model`` and prints a one-page summary including the
        GF-evaluation reduction percentage (``1 - n_slots / n_pairs``).
    """
    # sep = "--" * 50
    model._gf_map_h5_path = h5_path

    with h5py.File(h5_path, "r") as f:
        model._pairs_to_compute = f["pairs_to_compute"][:]
        model._pair_to_slot = f["pair_to_slot"][:]
        model._dh_of_pairs = f["dh_of_pairs"][:]
        model._zsrc_of_pairs = f["zsrc_of_pairs"][:]
        model._zrec_of_pairs = f["zrec_of_pairs"][:]
        model._delta_h = float(f["delta_h"][()])
        model._delta_v_src = float(f["delta_v_src"][()])
        model._delta_v_rec = float(f["delta_v_rec"][()])
        model._nsources = int(f["nsources"][()])
        model._nsources_db = int(f["nsources"][()])

    model._has_map = True
    model._use_pair_to_slot = True
    model._gf_loaded = True

    n_pairs = len(model._pair_to_slot)
    n_slots = len(model._pairs_to_compute)
    reduction = 100.0 * (1.0 - n_slots / n_pairs) if n_pairs > 0 else 0.0

    print("--" * 50)
    print(f"GF Map   : {h5_path}")
    print(f"  Format : OP")
    print(f"  Map    : subfaults={model._nsources_db}  |  pairs={n_pairs}  |  slots={n_slots}")
    print(f"  Reduce : {reduction:.2f}% fewer GF evaluations")
    print(f"  Tol      _delta_h={model._delta_h}  |  delta_v_src={model._delta_v_src}  |  delta_v_rec={model._delta_v_rec}")
    print("--" * 50 + "\n")


def _resolve_gf_slot(model, node_id, subfault_id):
    """Translate ``(node_id, subfault_id)`` into ``(numeric_node_id, slot_index)``.

    Used by every public GF query so QA-handling lives in one place.
    """
    node_id_num = model._n_nodes if node_id in ("QA", "qa") else node_id
    return node_id_num, model._get_slot(node_id_num, subfault_id)


def _read_raw_gf_slot(model, slot):
    """Read a single GF slot from disk, returning ``(tdata, t0)``.

    Parameters
    ----------
    model : ShakerMakerData
    slot : int
        Row index in the ``tdata`` dataset.

    Returns
    -------
    tuple
        ``(np.ndarray shape (Nt, 9), float)``. ``t0`` is 0 when the file
        has no ``t0`` dataset.
    """
    with h5py.File(model._gf_h5_path, "r") as f:
        tdata = np.asarray(f["tdata"][slot])
        t0 = float(f["t0"][slot]) if getattr(model, "_t0_available", False) else 0.0
    return tdata, t0


def _apply_gf_time_transform(model, tdata, t0=0.0):
    """Apply window mask and/or resampling to a raw GF slot.

    The transform mirrors what ``query_service`` does for node data so
    that GF time and signal time stay aligned after windowing or
    resampling.

    Returns
    -------
    tuple
        ``(np.ndarray, float)`` -- the transformed tensor and the input
        ``t0`` unchanged (returned for caller convenience).
    """
    if hasattr(model, "_window_mask"):
        t_start   = float(model.time[0])
        t_end     = float(model.time[-1])
        full_time = np.arange(tdata.shape[0]) * float(model._dt_orig) + float(t0)
        gf_mask   = (full_time >= t_start) & (full_time <= t_end)
        tdata     = tdata[gf_mask, :]

    if hasattr(model, "_resample_cache"):
        gf_time_orig = np.asarray(model._resample_cache.get("gf_time_orig", []), dtype=float)
        gf_time_new  = np.asarray(getattr(model, "gf_time", []), dtype=float)
        if gf_time_orig.size == tdata.shape[0] and gf_time_new.size > 0:
            rs = np.empty((gf_time_new.size, tdata.shape[1]), dtype=float)
            for j in range(tdata.shape[1]):
                rs[:, j] = interp1d(
                    gf_time_orig,
                    tdata[:, j],
                    kind="linear",
                    fill_value="extrapolate",
                )(gf_time_new)
            return rs, float(t0)

    return np.asarray(tdata), float(t0)


def get_gf_time(model, slot):
    """Return the GF time vector for a given slot, respecting window/resample.

    Parameters
    ----------
    model : ShakerMakerData
    slot : int

    Returns
    -------
    np.ndarray, shape (Nt,)
        Time in seconds, including the slot's ``t0`` offset when the file
        provides it.

    Raises
    ------
    RuntimeError
        If no GF database has been attached (``model._has_gf`` is False).
    """
    if not getattr(model, "_has_gf", False):
        raise RuntimeError("GF not loaded. Call load_gf_database() first.")

    nt = int(getattr(model, "_nt_gf", getattr(model, "_n_time_gf", 0)))
    t0 = 0.0
    if getattr(model, "_t0_available", False):
        with h5py.File(model._gf_h5_path, "r") as f:
            t0 = float(f["t0"][slot])
            if nt <= 0:
                nt = int(f["tdata"].shape[1])

    time = np.arange(nt, dtype=float) * float(model._dt_orig) + float(t0)

    if hasattr(model, "_window_mask"):
        t_start = float(model.time[0])
        t_end   = float(model.time[-1])
        gf_mask = (time >= t_start) & (time <= t_end)
        return time[gf_mask]

    if hasattr(model, "_resample_cache"):
        gf_time_new = np.asarray(getattr(model, "gf_time", []), dtype=float)
        if gf_time_new.size > 0:
            return gf_time_new + float(t0)

    return time
    


def get_gf_tensor(model, node_id, subfault_id):
    """Return the full ``(nt, 9)`` GF tensor plus time metadata for one pair.

    Parameters
    ----------
    model : ShakerMakerData
    node_id : int or {'QA', 'qa'}
    subfault_id : int

    Returns
    -------
    dict
        Keys:

        * ``'node_id_num'`` -- ``int`` resolved node index.
        * ``'slot'``        -- ``int`` row index in ``tdata``.
        * ``'t0'``          -- ``float`` per-slot time offset.
        * ``'time'``        -- ``np.ndarray`` time vector after window/resample.
        * ``'tdata'``       -- ``np.ndarray`` shape ``(nt, 9)``, post-transform.

    Raises
    ------
    RuntimeError
        If GF database and/or mapping are not loaded.
    """
    if not model._has_gf:
        raise RuntimeError("GF not loaded. Call load_gf_database() first.")
    if not model._has_map:
        raise RuntimeError("Map not loaded. Call load_map() first.")

    node_id_num, slot = _resolve_gf_slot(model, node_id, subfault_id)
    tdata_raw, t0 = _read_raw_gf_slot(model, slot)
    tdata, _ = _apply_gf_time_transform(model, tdata_raw, t0)
    time = get_gf_time(model, slot)
    return {
        "node_id_num": node_id_num,
        "slot": int(slot),
        "t0": float(t0),
        "time": np.asarray(time),
        "tdata": np.asarray(tdata),
    }


def get_gf(model, node_id, subfault_id, component="z"):
    """Return a single-component GF time series for a node/subfault pair.

    Parameters
    ----------
    model : ShakerMakerData
    node_id : int or {'QA', 'qa'}
    subfault_id : int
    component : {'z', 'e', 'n', 'tdata'}, default ``'z'``
        ``'tdata'`` returns the full ``(nt, 9)`` tensor (same as
        ``get_gf_tensor(...)['tdata']``).

    Returns
    -------
    np.ndarray
        Shape ``(nt,)`` for a single component, ``(nt, 9)`` for ``'tdata'``.

    Notes
    -----
    Results are memoised on ``model._gf_cache`` keyed by
    ``(node_id, subfault_id, component)``.
    """
    if not model._has_gf:
        raise RuntimeError("GF not loaded. Call load_gf_database() first.")
    if not model._has_map:
        raise RuntimeError("Map not loaded. Call load_map() first.")

    key = (node_id, subfault_id, component)

    if key not in model._gf_cache:
        comp_map = {"z": 0, "e": 1, "n": 2}
        gf_data = get_gf_tensor(model, node_id, subfault_id)
        tdata = gf_data["tdata"]

        if component == "tdata":
            model._gf_cache[key] = tdata
        elif component in comp_map:
            model._gf_cache[key] = tdata[:, comp_map[component]]
        else:
            raise KeyError(f"Unknown component '{component}'. Use 'z','e','n','tdata'.")

    return model._gf_cache[key]
