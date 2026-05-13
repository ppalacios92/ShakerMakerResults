"""
utils.py
========
Shared utility functions used across the plotting and comparison modules.

We keep these here -- and not on the reader classes -- so the readers stay
free of cross-dependencies (``ShakerMakerData`` does not need to know about
``StationData`` and vice versa). The plotting helpers reach into both
through duck-typing checks defined below.

Contents
--------
``_R``
    Rotation matrix that maps ShakerMaker coordinates (km, ENU) to a
    right-handed display frame (Z-up).
``_rotate(xyz_km)``
    Apply ``_R`` and convert kilometres to metres.
``_is_station(obj)``
    Duck-typed check -- ``True`` for ``StationData``-like objects.
``_resolve_node(node_id, model_index, n_models)``
    Pull a scalar node index for a given model out of flexible input
    (scalar / per-model list / list of lists).
``_get_signals(obj, node_idx, data_type, filtered)``
    Unified accessor returning ``[Z, E, N]`` regardless of the object type.
``_get_time(obj)``
    Time vector accessor for either reader type.
``_get_name(obj)``
    Display name accessor with sensible fallbacks.
``_fk_tensor_rotation(...)``
    Rotate a 9-component FK Green-Function tensor into physical Z/E/N
    components. Used by ``plot_node_gf`` / ``plot_models_gf``.
"""

import numpy as np

# Historical: we used to build _R explicitly from two ENU basis vectors --
# left here as a comment so the rotation choice is traceable.
#
# _R = np.column_stack([
#     np.array([0, 1, 0]),
#     np.array([1, 0, 0]),
#     np.cross(np.array([0, 1, 0]), np.array([1, 0, 0]))
# ])

_R = np.array([[0, 1,  0],
               [1, 0,  0],
               [0, 0, -1]])

def _rotate(xyz_km):
    """Apply the ShakerMaker display rotation and convert km to m.

    Parameters
    ----------
    xyz_km : np.ndarray, shape (N, 3)
        ShakerMaker coordinates in kilometres.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Same points expressed in the viewer / plotting frame, in metres.
    """
    return xyz_km * 1000 @ _R


# ---------------------------------------------------------------------------
# Duck-typed accessors -- used by plotting / comparison helpers that need to
# treat ``ShakerMakerData`` and ``StationData`` interchangeably without
# importing either class (the import would create a cycle).
# ---------------------------------------------------------------------------

def _is_station(obj):
    """Return ``True`` for ``StationData``-like objects.

    A ``StationData`` instance has ``z_v`` and lacks ``internal``; a
    ``ShakerMakerData`` instance has both attributes.
    """
    return hasattr(obj, 'z_v') and not hasattr(obj, 'internal')

def _resolve_node(node_id, model_index, n_models):
    """Pick a single node index for ``model_index`` out of flexible input.

    Parameters
    ----------
    node_id : int, str, list, or list of lists
        Accepted shapes:

        * scalar (``int`` / ``'QA'``)         -- used for every model.
        * list of length ``n_models``         -- one entry per model.
        * list of lists                       -- inner list is picked by
          model index, the first element of the inner list is returned.
    model_index : int
    n_models : int

    Returns
    -------
    int or str
        Scalar usable by ``get_node_data`` / ``get_qa_data``.
    """
    if not isinstance(node_id, list):
        return node_id
    if isinstance(node_id[0], list):
        return node_id[model_index][0]
    if len(node_id) == n_models:
        return node_id[model_index]
    return node_id[0]



def _get_signals(obj, node_idx, data_type, filtered=False):
    """Return ``[Z, E, N]`` signal arrays from either reader type.

    Parameters
    ----------
    obj : ShakerMakerData or StationData
    node_idx : int or {'QA', 'qa'}
        Ignored for ``StationData`` (a station only has one time series).
    data_type : {'accel', 'vel', 'disp'}
    filtered : bool, default ``False``
        Only honoured for ``StationData`` (the simulation reader doesn't
        have a filtered view).

    Returns
    -------
    list of np.ndarray
        Three component traces ordered ``[Z, E, N]``.
    """
    if _is_station(obj):
        if data_type in ('accel', 'acceleration'):
            z, e, n = obj.acceleration_filtered if filtered else obj.acceleration
        elif data_type in ('vel', 'velocity'):
            z, e, n = obj.velocity_filtered if filtered else obj.velocity
        else:
            z, e, n = obj.displacement_filtered if filtered else obj.displacement
        return [z, e, n]

    if node_idx in ('QA', 'qa') or (
            isinstance(node_idx, int) and node_idx >= len(obj.xyz)):
        d = obj.get_qa_data(data_type)
    else:
        d = obj.get_node_data(node_idx, data_type)
    return [d[2], d[0], d[1]]   # Z, E, N


def _get_time(obj):
    """Return the active time vector for either reader type.

    ``StationData`` stores it as ``t``; ``ShakerMakerData`` as ``time``.
    """
    return obj.t if _is_station(obj) else obj.time


def _get_name(obj):
    """Return a short display name with sensible fallbacks.

    ``StationData`` returns ``obj.name`` or ``"Station"`` when unset.
    ``ShakerMakerData`` returns ``obj.name`` or ``"Model"`` when unset.
    """
    if _is_station(obj):
        return obj.name if obj.name else "Station"
    return obj.name if getattr(obj, "name", None) else "Model"


def _fk_tensor_rotation(gf_tensor, strike, dip, rake, azimuth):
    """Rotate a 9-component FK tensor into physical Z/E/N components.

    Parameters
    ----------
    gf_tensor : np.ndarray
        Array of shape ``(nt, 9)`` ordered as stored in the GF database.
    strike, dip, rake, azimuth : float
        Angles in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Rotated ``(z, e, n)`` time histories.
    """
    pf = strike
    df = dip
    lf = rake
    p = azimuth

    f1 = np.cos(lf) * np.cos(pf) + np.sin(lf) * np.cos(df) * np.sin(pf)
    f2 = np.cos(lf) * np.sin(pf) - np.sin(lf) * np.cos(df) * np.cos(pf)
    f3 = -np.sin(lf) * np.sin(df)
    n1 = -np.sin(pf) * np.sin(df)
    n2 = np.cos(pf) * np.sin(df)
    n3 = -np.cos(df)

    a = (f1 * n1 - f2 * n2) * np.cos(2 * p) + (f1 * n2 + f2 * n1) * np.sin(2 * p)
    b = (f1 * n3 + f3 * n1) * np.cos(p) + (f2 * n3 + f3 * n2) * np.sin(p)
    c = f3 * n3
    a_t = (f1 * n1 - f2 * n2) * np.sin(2 * p) - (f1 * n2 + f2 * n1) * np.cos(2 * p)
    b_t = (f1 * n3 + f3 * n1) * np.sin(p) - (f2 * n3 + f3 * n2) * np.cos(p)

    z_gf = gf_tensor[:, 6] * a + gf_tensor[:, 3] * b + gf_tensor[:, 0] * c
    r_gf = gf_tensor[:, 7] * a + gf_tensor[:, 4] * b + gf_tensor[:, 1] * c
    t_gf = gf_tensor[:, 8] * a_t + gf_tensor[:, 5] * b_t

    e_gf = -r_gf * np.sin(p) - t_gf * np.cos(p)
    n_gf = -r_gf * np.cos(p) + t_gf * np.sin(p)
    return z_gf, e_gf, n_gf
