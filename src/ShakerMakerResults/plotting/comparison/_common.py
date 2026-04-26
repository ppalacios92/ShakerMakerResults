"""Shared helpers for multi-model comparison plots."""

from __future__ import annotations

import numpy as np

from ...core.gf_service import get_gf_time


def _build_label(obj, node_id):
    """Build a compact display label for a model + node combination."""
    node_part = "QA" if node_id in ("QA", "qa") else f"N{node_id}"
    return f"{obj.model_name} | {node_part} | dt={obj.dt:.4f}s"


def _get_node_data(obj, node_id, data_type):
    """Return ``(z, e, n)`` signal tuple for a node, handling QA transparently."""
    if node_id in ("QA", "qa"):
        data = obj.get_qa_data(data_type)
    else:
        data = obj.get_node_data(node_id, data_type)
    return data[0], data[1], data[2]


def _get_gf_time(obj, slot):
    """Return the GF time vector for a given slot, respecting window/resample."""
    return get_gf_time(obj, slot)
