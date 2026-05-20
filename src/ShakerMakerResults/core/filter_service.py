"""
filter_service.py
=================
Lazy band-pass filter helper for :class:`ShakerMakerData`.

Mirrors the design of :mod:`window_service`: ``apply_filter`` does **not**
read or transform any signal — it returns a new ``ShakerMakerData`` with
a ``_filter`` dict attached and reset caches. The actual ObsPy call
happens lazily inside :mod:`query_service` and (optionally)
:mod:`gf_service` the next time a per-node trace is requested.

The ``mode`` parameter controls what gets filtered on disk and how the
other two data types are produced:

* ``'all'``  -- read whichever ``data_type`` the caller requested
  (``vel`` / ``accel`` / ``disp``) directly from disk and filter it.
* ``'vel'``  -- always read velocity, filter it, then integrate
  (trapezoidal cumulative sum) to displacement or differentiate
  (central differences) to acceleration.
* ``'accel'``-- same idea, base = acceleration; integrate once for
  velocity, twice for displacement.
* ``'disp'`` -- base = displacement; differentiate once for velocity,
  twice for acceleration.

QA station data is always filtered with the same ``_filter`` settings.
Green Function tensors are only filtered when ``apply_gf=True`` (off by
default — GFs are intermediate quantities and shouldn't normally be
band-passed).
"""

from __future__ import annotations

import numpy as np

_VALID_MODES = ("all", "vel", "accel", "disp")


def apply_filter(model, mode="all", freqmin=0.25, freqmax=10.0,
                 corners=4, zerophase=True, apply_gf=False):
    """Return a lazy band-pass-filtered copy of ``model``.

    Parameters
    ----------
    model : ShakerMakerData
    mode : {'all', 'vel', 'accel', 'disp'}, default ``'all'``
        Which dataset to filter as the "base". When ``'all'``, every
        ``data_type`` is filtered independently from disk. When set to a
        specific type, that type is read & filtered, and the other two
        are derived from it by integration / differentiation.
    freqmin, freqmax : float
        Band edges in Hz. Validated against the Nyquist frequency
        (``1 / (2 * model.dt)``).
    corners : int, default ``4``
        Number of Butterworth filter corners (passed to ObsPy).
    zerophase : bool, default ``True``
        Zero-phase forward+backward pass (passed to ObsPy).
    apply_gf : bool, default ``False``
        If ``True``, GF tensors are also filtered when read via
        :func:`gf_service.get_gf` / :func:`gf_service.get_gf_tensor`.

    Returns
    -------
    ShakerMakerData
        New instance sharing on-disk arrays with the original but with
        a ``_filter`` dict attached. ``_node_cache``, ``_gf_cache`` and
        ``_spectrum_cache`` are reset.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Use one of {_VALID_MODES}."
        )

    dt = float(getattr(model, "dt", model._dt_orig))
    nyquist = 0.5 / dt
    fmin = float(freqmin)
    fmax = float(freqmax)
    if fmin <= 0:
        raise ValueError(f"freqmin must be > 0 (got {fmin}).")
    if fmax <= fmin:
        raise ValueError(f"freqmax ({fmax}) must be > freqmin ({fmin}).")
    if fmax >= nyquist:
        raise ValueError(
            f"freqmax ({fmax} Hz) must be < Nyquist ({nyquist:.4f} Hz) "
            f"for dt={dt}s."
        )

    new = model.__class__.__new__(model.__class__)
    for a, v in model.__dict__.items():
        setattr(new, a, v)

    new._filter = {
        "mode": mode,
        "freqmin": fmin,
        "freqmax": fmax,
        "corners": int(corners),
        "zerophase": bool(zerophase),
        "apply_gf": bool(apply_gf),
    }
    new._node_cache = {}
    new._gf_cache = {}
    new._spectrum_cache = {}
    new._vmax = model._vmax

    print("--" * 50)
    print(f"Filter   : {model.name}")
    print(f"  Mode   : {mode}  |  bandpass [{fmin}, {fmax}] Hz")
    print(f"  ObsPy  : corners={int(corners)}  |  zerophase={bool(zerophase)}")
    print(f"  Nyquist: {nyquist:.4f} Hz  (dt={dt}s)")
    print(f"  QA     : filtered (always)")
    print(f"  GF     : {'filtered' if apply_gf else 'not filtered'}")
    print("--" * 50 + "\n")

    return new


def bandpass(data, dt, filt, time_axis=-1):
    """Apply a Butterworth bandpass via ObsPy along ``time_axis``.

    Parameters
    ----------
    data : np.ndarray
        Any shape; commonly ``(3, Nt)`` for node traces (time_axis=-1)
        or ``(Nt, 9)`` for GF tensors (time_axis=0).
    dt : float
        Sample spacing in seconds.
    filt : dict
        Filter parameters with keys ``freqmin``, ``freqmax``, ``corners``
        and ``zerophase`` (the dict stored on ``model._filter``).
    time_axis : int, default ``-1``
        Axis along which time runs.

    Returns
    -------
    np.ndarray, same shape as ``data``, dtype float64
    """
    from obspy import Stream, Trace

    arr = np.moveaxis(np.asarray(data), time_axis, -1)
    flat = arr.reshape(-1, arr.shape[-1])

    traces = []
    for row in flat:
        tr = Trace(data=np.ascontiguousarray(row, dtype=np.float32))
        tr.stats.delta = float(dt)
        traces.append(tr)
    st = Stream(traces)
    st.filter(
        "bandpass",
        freqmin=filt["freqmin"],
        freqmax=filt["freqmax"],
        corners=filt["corners"],
        zerophase=filt["zerophase"],
    )

    out_flat = np.vstack([tr.data for tr in st]).astype(np.float64)
    out = out_flat.reshape(arr.shape)
    return np.moveaxis(out, -1, time_axis)


def _integrate(arr, dt):
    """Trapezoidal cumulative integral along axis 1, output shape preserved."""
    out = np.zeros_like(arr)
    out[:, 1:] = np.cumsum(0.5 * (arr[:, 1:] + arr[:, :-1]) * dt, axis=1)
    return out


def _differentiate(arr, dt):
    """Central differences along axis 1, with forward/backward at boundaries."""
    out = np.empty_like(arr)
    out[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2.0 * dt)
    out[:, 0] = (arr[:, 1] - arr[:, 0]) / dt
    out[:, -1] = (arr[:, -1] - arr[:, -2]) / dt
    return out


_ORDER = {"disp": 0, "vel": 1, "accel": 2}


def derive(base, dt, base_type, requested_type):
    """Convert a filtered ``base_type`` trace to ``requested_type``.

    Differentiation order = order(requested) - order(base):
    disp=0, vel=1, accel=2. Positive -> differentiate that many times,
    negative -> integrate.
    """
    if base_type == requested_type:
        return base
    delta = _ORDER[requested_type] - _ORDER[base_type]
    out = base
    if delta > 0:
        for _ in range(delta):
            out = _differentiate(out, dt)
    else:
        for _ in range(-delta):
            out = _integrate(out, dt)
    return out
