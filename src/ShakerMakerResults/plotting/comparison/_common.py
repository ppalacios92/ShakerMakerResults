"""
_common.py
==========
Tiny helpers shared by the multi-model comparison plots.

Used by ``response_plots`` / ``arias_plots`` / ``gf_plots`` so the
per-plot files stay focused on the plotting itself.
"""

from __future__ import annotations


def _build_label(obj, node_id):
    """Build a compact "<model> | N<id> | dt=..." label for a legend entry.

    Parameters
    ----------
    obj : ShakerMakerData or StationData
        Used for ``name`` and ``dt``.
    node_id : int or {'QA', 'qa'}

    Returns
    -------
    str
    """
    node_part = "QA" if node_id in ("QA", "qa") else f"N{node_id}"
    return f"{obj.name} | {node_part} | dt={obj.dt:.4f}s"


def _get_node_data(obj, node_id, data_type):
    """Return ``(z, e, n)`` for a node, dispatching QA on the reader.

    Parameters
    ----------
    obj : ShakerMakerData
    node_id : int or {'QA', 'qa'}
    data_type : {'accel', 'vel', 'disp'}

    Returns
    -------
    tuple of np.ndarray
        Three component traces, already in Z / E / N order.
    """
    if node_id in ("QA", "qa"):
        data = obj.get_qa_data(data_type)
    else:
        data = obj.get_node_data(node_id, data_type)
    return data[0], data[1], data[2]
