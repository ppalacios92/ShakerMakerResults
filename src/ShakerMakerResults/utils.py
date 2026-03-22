"""
utils.py
========
Shared utility functions and constants used across plotting.py and comparison.py.

Centralises helpers that would otherwise be duplicated between modules,
keeping shakermaker_data.py and station_data.py free of cross-dependencies.

Contents
--------
_R
    Rotation matrix that maps ShakerMaker coordinates (km, ENU) to a
    right-handed display frame.
_rotate(xyz_km)
    Apply _R and convert kilometres to metres.
_is_station(obj)
    Duck-type check, returns True for StationData objects.
_resolve_node(node_id, model_index, n_models)
    Extract a scalar node index for a given model from flexible input.
_get_signals(obj, node_idx, data_type, filtered)
    Unified signal accessor, returns [Z, E, N] regardless of object type.
_get_time(obj)
    Return the time vector from either object type.
_get_name(obj)
    Return a short display name from either object type.

"""

import numpy as np

# _R = np.column_stack([
#     np.array([0, 1, 0]),
#     np.array([1, 0, 0]),
#     np.cross(np.array([0, 1, 0]), np.array([1, 0, 0]))
# ])

_R = np.array([[1, 0,  0],
               [0, 1,  0],
               [0, 0, -1]])

def _rotate(xyz_km):
    """Apply the ShakerMaker display rotation and convert km to m."""
    return xyz_km * 1000 @ _R


# Comparasion

def _is_station(obj):
    """Return True if obj is a StationData instance.

    Uses duck-typing to avoid importing StationData directly.
    """
    return hasattr(obj, 'z_v') and not hasattr(obj, 'internal')

def _resolve_node(node_id, model_index, n_models):
    """Extract a scalar node index for a given model from flexible input.

    Supports scalar, flat list (one per model), and list of lists.
    """
    if not isinstance(node_id, list):
        return node_id
    if isinstance(node_id[0], list):
        return node_id[model_index][0]
    if len(node_id) == n_models:
        return node_id[model_index]
    return node_id[0]



def _get_signals(obj, node_idx, data_type, filtered=False):
    """Return [Z, E, N] arrays from ShakerMakerData or StationRead."""
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
    """Return the time vector from any supported model object."""
    return obj.t if _is_station(obj) else obj.time


def _get_name(obj):
    """Return a short display name from any supported model object.

    StationData returns obj.name or 'Station'.
    ShakerMakerData returns obj.model_name.
    """
    if _is_station(obj):
        return obj.name if obj.name else "Station"
    return obj.model_name
