"""Comparison plots for time histories and spectra."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ...analysis.newmark import NewmarkSpectrumAnalyzer
from ._common import _build_label, _get_node_data

__all__ = ["plot_models_response", "plot_models_newmark"]


def plot_models_response(
    models,
    node_ids=None,
    target_pos=None,
    data_type="vel",
    xlim=None,
    figsize=(10, 8),
    factor=1.0,
):
    """Overlay time-history traces from multiple models on one 3x1 figure.

    Parameters
    ----------
    models : list of ShakerMakerData or StationData
    node_ids : list, optional
    target_pos : list, optional
        Same shape rules as :func:`plot_models_arias`.
    data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
    xlim : tuple, optional
    figsize : tuple, default ``(10, 8)``
    factor : float, default ``1.0``
        Multiplier applied before plotting.

    Returns
    -------
    None
        Calls ``plt.show()`` directly.
    """
    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    n = len(models)
    nids_list = node_ids if node_ids else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    ylabel = {"accel": "Acceleration", "vel": "Velocity", "disp": "Displacement"}[data_type]

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        for nid in nids:
            z, e, n_comp = _get_node_data(obj, nid, data_type)
            lbl = _build_label(obj, nid)
            for ax, sig in zip(axes, (z, e, n_comp)):
                ax.plot(obj.time, sig * factor, linewidth=1, label=lbl)

    comp_titles = ("Vertical (Z)", "East (E)", "North (N)")
    for ax, comp in zip(axes, comp_titles):
        ax.set_title(f"{comp} - {ylabel}", fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def plot_models_newmark(
    models,
    node_ids=None,
    target_pos=None,
    data_type="accel",
    spectral_type="PSa",
    xlim=None,
    figsize=(8, 10),
    factor=1.0,
):
    """Overlay Newmark response spectra from multiple models on one 3x1 figure.

    Parameters
    ----------
    models : list of ShakerMakerData or StationData
    node_ids : list, optional
    target_pos : list, optional
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
        Source signal. When ``'accel'`` the input is converted from m/s^2
        to g before the Newmark solver runs (``scale = 1 / 9.81``).
    spectral_type : {'PSa', 'Sa', 'PSv', 'Sv', 'Sd'}, default ``'PSa'``
        Which spectrum to plot. The solver returns all five, so we just
        pick the right key.
    xlim : tuple, default ``[0, 5]``
        Period bounds in seconds.
    figsize : tuple, default ``(8, 10)``
    factor : float, default ``1.0``

    Returns
    -------
    None
        Calls ``plt.show()`` directly.
    """
    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    if xlim is None:
        xlim = [0, 5]

    n = len(models)
    nids_list = node_ids if node_ids else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    scale = 1.0 / 9.81 if data_type == "accel" else 1.0
    ylabel = {
        "PSa": "PSa (g)",
        "Sa": "Sa (g)",
        "PSv": "PSv (m/s)",
        "Sv": "Sv (m/s)",
        "Sd": "Sd (m)",
    }.get(spectral_type, spectral_type)

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        dt = obj.time[1] - obj.time[0]
        for nid in nids:
            z, e, n_comp = _get_node_data(obj, nid, data_type)
            lbl = _build_label(obj, nid)
            specs = [
                NewmarkSpectrumAnalyzer.compute(sig * factor * scale, dt)
                for sig in (z, e, n_comp)
            ]
            periods = specs[0]["T"]
            for ax, sp in zip(axes, specs):
                ax.plot(periods, sp[spectral_type], linewidth=2, label=lbl)

    comp_titles = ("Vertical (Z)", "East (E)", "North (N)")
    for ax, comp in zip(axes, comp_titles):
        ax.set_title(f"{comp} - {spectral_type}", fontweight="bold")
        ax.set_xlabel("T (s)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()
