"""Comparison plots for Green's functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...core.gf_service import get_gf_tensor
from ...utils import _fk_tensor_rotation

__all__ = ["plot_models_gf", "plot_models_tensor_gf"]


def plot_models_gf(
    models,
    node_ids=None,
    target_pos=None,
    subfault=0,
    xlim=None,
    figsize=(8, 10),
    factor=1.0,
    ffsp_source=None,
    strikes=None,
    dips=None,
    rakes=None,
    src_x=None,
    src_y=None,
    internal_ref=None,
    external_coord=None,
):
    """Plot Green's-function time series for multiple models."""
    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")

    n = len(models)
    if len(node_ids if node_ids else target_pos) != n:
        raise ValueError("models and node_ids / target_pos must have the same length.")

    nids_list = node_ids if node_ids else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    def _norm(val):
        return val if val is not None else [None] * n

    ffsp_list = _norm(ffsp_source)
    str_list = _norm(strikes)
    dip_list = _norm(dips)
    rak_list = _norm(rakes)
    srcx_list = _norm(src_x)
    srcy_list = _norm(src_y)
    iref_list = _norm(internal_ref)
    ecoord_list = _norm(external_coord)

    sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj, nids, tpos, ffsp, st, di, ra, sx, sy, iref, ecoord in zip(
        models,
        nids_list,
        tpos_list,
        ffsp_list,
        str_list,
        dip_list,
        rak_list,
        srcx_list,
        srcy_list,
        iref_list,
        ecoord_list,
    ):
        if not obj._has_gf or not obj._has_map:
            print(f"  Warning: {obj.model_name} has no GFs loaded - skipped.")
            continue

        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        use_physical = ffsp is not None or st is not None

        offset_x = offset_y = 0.0
        strike_rad_fault = 0.0
        if ffsp is not None:
            strike_rad_fault = np.radians(ffsp.params["strike"])
            if iref is not None and ecoord is not None:
                ref_x, ref_y = iref
                ext_x, ext_y = ecoord
                ref_x_rot = ref_x * np.sin(strike_rad_fault) + ref_y * np.cos(strike_rad_fault)
                ref_y_rot = ref_x * np.cos(strike_rad_fault) - ref_y * np.sin(strike_rad_fault)
                offset_x = ext_x - ref_x_rot
                offset_y = ext_y - ref_y_rot

        for nid in nids:
            nid_num = obj._n_nodes if nid in ("QA", "qa") else nid
            nid_label = "QA" if nid in ("QA", "qa") else f"N{nid}"

            if nid in ("QA", "qa"):
                rx, ry = obj.xyz_qa[0, 0], obj.xyz_qa[0, 1]
            else:
                rx, ry = obj.xyz[nid, 0], obj.xyz[nid, 1]

            for sid in sub_ids:
                gf_data = get_gf_tensor(obj, nid, sid)
                tdata = gf_data["tdata"]
                time = gf_data["time"]
                lbl = f"{obj.model_name} | {nid_label} | S{sid}"

                if use_physical:
                    if ffsp is not None:
                        sf = ffsp.subfaults
                        strike = np.radians(sf["strike"][sid])
                        dip = np.radians(sf["dip"][sid])
                        rake = np.radians(sf["rake"][sid])
                        sx_ = sf["x"][sid] / 1e3 * np.sin(strike_rad_fault) + offset_x
                        sy_ = sf["y"][sid] / 1e3 * np.cos(strike_rad_fault) + offset_y
                    else:
                        strike = np.radians(st[sid])
                        dip = np.radians(di[sid])
                        rake = np.radians(ra[sid])
                        sx_ = sx[sid]
                        sy_ = sy[sid]

                    azimuth = np.arctan2(ry - sy_, rx - sx_)
                    z_gf, e_gf, n_gf = _fk_tensor_rotation(tdata, strike, dip, rake, azimuth)
                    components = (z_gf * factor, e_gf * factor, n_gf * factor)
                else:
                    components = (tdata[:, 0] * factor, tdata[:, 1] * factor, tdata[:, 2] * factor)

                for ax, sig in zip(axes, components):
                    ax.plot(time, sig, linewidth=1, label=lbl)

    comp_titles = ("Vertical (Z)", "East (E)", "North (N)")
    for ax, title in zip(axes, comp_titles):
        ax.set_title(f"{title} - Green Function", fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def plot_models_tensor_gf(
    models,
    node_ids=None,
    target_pos=None,
    subfault=0,
    xlim=None,
    figsize=(12, 10),
    factor=1.0,
):
    """Plot the 9-component tensor Green's functions for multiple models."""
    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
    labels = [f"G_{i+1}{j+1}" for i in range(3) for j in range(3)]
    n = len(models)
    nids_list = node_ids if node_ids else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        if not obj._gf_loaded:
            print(f"  Warning: {obj.model_name} has no GFs loaded - skipped.")
            continue

        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        for nid in nids:
            if nid in ("QA", "qa"):
                nid_num = obj._n_nodes
                nid_label = "QA"
            else:
                nid_num = nid
                nid_label = f"N{nid}"
            for sid in sub_ids:
                slot = obj._get_slot(nid_num, sid)
                donor = obj._pairs_to_compute[slot, 0]
                if donor != nid_num:
                    print(f"  {obj.model_name} | {nid_label}/S{sid} -> slot {slot} (donor {donor})")
                lbl = f"{obj.model_name} | {nid_label} | S{sid}"
                gf_data = get_gf_tensor(obj, nid, sid)
                time = gf_data["time"]
                tdata = gf_data["tdata"] * factor
                for j in range(9):
                    axes[j // 3, j % 3].plot(time, tdata[:, j], linewidth=0.8, label=lbl)

    for j, lbl in enumerate(labels):
        ax = axes[j // 3, j % 3]
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [s]", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(xlim)

    axes[0, 2].legend(fontsize=8)
    plt.suptitle("Tensor Green Functions - Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
