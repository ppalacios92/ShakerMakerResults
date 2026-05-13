"""Standalone domain-level plotting helpers."""

from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ...utils import _rotate

def plot_domain(self, 
                xyz_origin=None, 
                label_nodes=False, 
                show_calculated=False,
                figsize=(8,6),
                axis_equal=False):

    """Plot the 3-D node domain of the DRM or SurfaceGrid object.

    Renders internal nodes, external (boundary) nodes, and the QA station
    in a 3-D scatter plot, overlaid with the bounding-box wireframe.
    Optionally highlights computational donor nodes when GF data is loaded.

    Parameters
    ----------
    xyz_origin : array-like (3,), optional
        If provided, shifts all coordinates so that the QA station is
        placed at this position [x, y, z] in metres.
    label_nodes : bool or str, default ``False``
        Node labelling mode:

        - ``False``             : no labels
        - ``True``              : all nodes
        - ``'corners'``         : corner nodes only
        - ``'corners_edges'``   : corners and edge nodes
        - ``'corners_half'``    : corners and edge midpoints
        - ``'calculated'``      : computational donor nodes only

    show_calculated : bool, default ``False``
        If ``True`` and a GF database is loaded, donor nodes (actually
        computed GFs) are highlighted in blue and reused nodes in light
        blue.  Requires ``load_gf_database()`` to have been called first.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes3D
    """
    xyz_t    = _rotate(self.xyz)
    xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None
    if xyz_origin is not None and xyz_qa_t is not None:
        t = np.asarray(xyz_origin) - xyz_qa_t[0]
        xyz_t += t; xyz_qa_t += t

    xyz_int = xyz_t[self.internal]; xyz_ext = xyz_t[~self.internal]
    # SurfaceGrid has no internal nodes — use all for bounding box
    bbox = xyz_int if len(xyz_int) > 0 else xyz_t
    _, faces, bounds = self._build_cube_faces(bbox)
    fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111,projection='3d')

    comp_donors = None
    if show_calculated and self._gf_loaded and self._pairs_to_compute is not None:
        comp_donors = set(np.unique(self._pairs_to_compute[:,0]))
        all_idx = np.arange(len(xyz_t))
        calc_mask = np.isin(all_idx, list(comp_donors))
        ax.scatter(xyz_t[~calc_mask,0],xyz_t[~calc_mask,1],xyz_t[~calc_mask,2],
                   c='lightblue',s=30,alpha=0.3)
        if calc_mask.any():
            ax.scatter(xyz_t[calc_mask,0],xyz_t[calc_mask,1],xyz_t[calc_mask,2],
                       c='blue',s=50,alpha=0.5,edgecolors='darkblue',linewidths=1.5)
    elif len(xyz_int) > 0:
        ax.scatter(xyz_ext[:,0],xyz_ext[:,1],xyz_ext[:,2],c='blue',marker='o',s=50,alpha=0.1)
        ax.scatter(xyz_int[:,0],xyz_int[:,1],xyz_int[:,2],c='red',marker='s',s=30,alpha=0.4)
    else:
        ax.scatter(xyz_t[:,0],xyz_t[:,1],xyz_t[:,2],c='blue',marker='s',s=30,alpha=0.4)

    if xyz_qa_t is not None:
        ax.scatter(xyz_qa_t[:,0],xyz_qa_t[:,1],xyz_qa_t[:,2],c='green',marker='*',
                   s=300,label='QA',zorder=10,edgecolors='black',linewidths=2)
    ax.add_collection3d(Poly3DCollection(faces,alpha=0.15,facecolor='red',
                                         edgecolor='darkred',linewidths=1.5))
    if label_nodes: 
        self._label_nodes_on_ax(ax,xyz_t,bounds,label_nodes,comp_donors)

    ax.set_xlabel("X' (m)")
    ax.set_ylabel("Y' (m)")
    ax.set_zlabel("Z' (m)")
    ax.legend(); ax.grid(False); 

    if axis_equal is True:
        ax.axis('equal')
    plt.tight_layout(); 
    plt.show()

    if xyz_qa_t is not None: 
        print(f"QA position: {xyz_qa_t[0]}")
    return fig, ax


def plot_domain_calculated_t0(
    self,
    subfault=0,
    show_calculated_only=False,
    xyz_origin=None,
    figsize=(8, 6),
    axis_equal=True,
    cmap="viridis",
):
    """Plot domain nodes coloured by GF t0.

    Parameters
    ----------
    subfault : int or 'all', default 0
    show_calculated_only : bool, default False
    xyz_origin : array-like (3,), optional
        If provided, shifts all coordinates so that the QA station is
        placed at this position [x, y, z] in metres.
    figsize : tuple, default (8, 6)
    axis_equal : bool, default True
    cmap : str, default 'viridis'
    """
    if not getattr(self, "_has_gf", False):
        print("GF not loaded."); return
    if not getattr(self, "_has_map", False):
        print("Map not loaded."); return
    if not getattr(self, "_t0_available", False):
        print("GF file does not contain t0."); return

    nsources = int(getattr(self, "_nsources_db", getattr(self, "_nsources", 0)))
    use_all  = (subfault == "all")
    if not use_all:
        subfault = int(subfault)

    with h5py.File(self._gf_h5_path, "r") as f:
        t0_all = np.asarray(f["t0"][:], dtype=float)

    # Rotated coordinates — same as plot_domain
    xyz_t    = _rotate(self.xyz)
    xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None
    if xyz_origin is not None and xyz_qa_t is not None:
        shift     = np.asarray(xyz_origin, dtype=float) - xyz_qa_t[0]
        xyz_t    += shift
        xyz_qa_t += shift

    xyz_all_t   = np.vstack([xyz_t, xyz_qa_t]) if xyz_qa_t is not None else xyz_t.copy()
    n_nodes_all = len(xyz_all_t)
    x = xyz_all_t[:, 0]
    y = xyz_all_t[:, 1]
    z = xyz_all_t[:, 2]

    pair_to_slot = np.asarray(self._pair_to_slot, dtype=int)

    if use_all:
        calc_node_ids = np.unique(self._pairs_to_compute[:, 0].astype(int))
        t0_of_node    = np.full(n_nodes_all, np.inf)
        for sf in range(nsources):
            flat_ids   = np.arange(n_nodes_all, dtype=int) * nsources + sf
            slots      = pair_to_slot[flat_ids]
            t0_of_node = np.minimum(t0_of_node, t0_all[slots])
        title = "all subfaults"
    else:
        flat_ids      = np.arange(n_nodes_all, dtype=int) * nsources + subfault
        slots         = pair_to_slot[flat_ids]
        t0_of_node    = t0_all[slots]
        rep_nodes     = self._pairs_to_compute[:, 0].astype(int)
        src_nodes     = self._pairs_to_compute[:, 1].astype(int)
        calc_node_ids = np.unique(rep_nodes[src_nodes == subfault])
        title         = f"subfault {subfault}"

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    if show_calculated_only:
        mask = np.isin(np.arange(n_nodes_all), calc_node_ids)
        sc   = ax.scatter(x[mask], y[mask], z[mask],
                          c=t0_of_node[mask], cmap=cmap, s=50)
    else:
        calc_mask  = np.isin(np.arange(n_nodes_all), calc_node_ids)
        adopt_mask = ~calc_mask
        ax.scatter(x[adopt_mask], y[adopt_mask], z[adopt_mask],
                   c=t0_of_node[adopt_mask], cmap=cmap,
                   s=20, alpha=0.25, label=f"Adopted ({adopt_mask.sum()})")
        sc = ax.scatter(x[calc_mask], y[calc_mask], z[calc_mask],
                        c=t0_of_node[calc_mask], cmap=cmap,
                        s=60, alpha=0.95,
                        edgecolors="black", linewidths=0.8,
                        label=f"Calculated ({calc_mask.sum()})")
        ax.legend()

    fig.colorbar(sc, ax=ax, shrink=0.75, label="t0 [s]")
    ax.set_xlabel("X' (m)"); ax.set_ylabel("Y' (m)"); ax.set_zlabel("Z' (m)")
    ax.set_title(f"GF t0 — {title}", fontweight="bold")
    if axis_equal:
        ax.axis("equal")
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    print(f"Calculated: {len(calc_node_ids)}  |  t0 range: {t0_of_node.min():.3f} .. {t0_of_node.max():.3f} s")
    return fig, ax


def plot_calculated_vs_reused(self, 
                                db_filename=None, 
                                xyz_origin=None,
                                label_nodes=False):

    """Visualise computed vs donor-reused GF nodes."""
    # Get pairs from OP pipeline or legacy
    if self._gf_loaded and self._pairs_to_compute is not None:
        unique_calc = np.unique(self._pairs_to_compute[:,0])
    elif self.gf_db_pairs is not None:
        unique_calc = np.unique(self.gf_db_pairs[:,0])
    else:
        print("No GF database info. Call load_gf_database() first."); return

    xyz_t    = _rotate(self.xyz)
    xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None
    if xyz_origin is not None and xyz_qa_t is not None:
        t = np.asarray(xyz_origin)-xyz_qa_t[0]; xyz_t+=t; xyz_qa_t+=t

    all_idx   = np.arange(len(xyz_t))
    calc_mask = np.isin(all_idx, unique_calc)

    # Bounding box — use all nodes if no internal
    bbox = xyz_t[self.internal] if self.internal.any() else xyz_t
    _,faces,bounds = self._build_cube_faces(bbox)

    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111,projection='3d')
    ax.scatter(xyz_t[~calc_mask,0],xyz_t[~calc_mask,1],xyz_t[~calc_mask,2],
               c='lightblue',marker='o',s=30,alpha=0.3,label='Reused')
    if calc_mask.any():
        ax.scatter(xyz_t[calc_mask,0],xyz_t[calc_mask,1],xyz_t[calc_mask,2],
                   c='blue',marker='o',alpha=0.5,edgecolors='darkblue',
                   linewidths=1.5,label='Calculated')
    if xyz_qa_t is not None:
        ax.scatter(*xyz_qa_t[0],c='green',marker='*',s=400,label='QA',
                   zorder=10,edgecolors='black',linewidths=2)
    ax.add_collection3d(Poly3DCollection(faces,alpha=0.1,facecolor='red',
                                         edgecolor='darkred',linewidths=2))
    if label_nodes: self._label_nodes_on_ax(ax,xyz_t,bounds,label_nodes,set(unique_calc))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(); ax.grid(True,alpha=0.3); plt.tight_layout(); plt.show()
    n = len(xyz_t)
    print("="*60)
    print(f"Calculated: {len(unique_calc)}/{n} ({len(unique_calc)/n*100:.1f}%)")
    print(f"Reused:     {n-len(unique_calc)}/{n}")
    print("="*60)
    return fig, ax


def plot_gf_connections(self,
                        node_id,
                        xyz_origin=None,
                        label_nodes=False,
                        figsize=(8, 6),
                        axis_equal=False):
    """Visualise donor-recipient GF connections for a single node.

    Prints a full node classification summary (super donors, solitary
    donors, pure receivers) and highlights the donor-recipient
    relationships for the requested node in a 3-D scatter plot.

    Parameters
    ----------
    node_id : int
        Node index to analyse.
    xyz_origin : array-like (3,), optional
        If provided, shifts all coordinates so that the QA station
        is placed at this position [x, y, z] in metres.
    label_nodes : bool or str, default ``False``
        Node labelling mode:

        - ``False``              : no labels
        - ``True``               : all nodes
        - ``'corners'``          : corner nodes only
        - ``'corners_edges'``    : corners and edge nodes
        - ``'corners_half'``     : corners and edge midpoints
        - ``'calculated'``       : computational donor nodes only
    figsize : tuple, default ``(8, 6)``
    """
    if not self._gf_loaded:
        print("No GFs. Call load_gf_database() first.")
        return

    # Translate 'QA' / 'qa' into the numeric index it occupies.
    if node_id in ('QA', 'qa'):
        node_id_num = self._n_nodes
        node_id_label = 'QA'
    else:
        node_id_num = node_id
        node_id_label = str(node_id)

    # Classification
    comp_donors = set(np.unique(self._pairs_to_compute[:, 0]))
    super_donors = set()
    # Include QA in the analysis when present.
    total_nodes = self._n_nodes + (1 if self.xyz_qa is not None else 0)
    for node in range(total_nodes):
        donor = self._donor_of_op(node, 0)
        if donor != node:
            super_donors.add(donor)
    solitary = comp_donors - super_donors
    all_nodes = set(range(total_nodes))
    pure_receivers = all_nodes - comp_donors

    sep = '--' * 50
    print(sep)
    print("GF NODE CLASSIFICATION")
    print(f"  Super Donors    ({len(super_donors)})  :  "
          f"{sorted(int(x) for x in super_donors)}")
    print(f"  Solitary Donors ({len(solitary)})  :  "
          f"{sorted(int(x) for x in solitary)}")
    print(f"  Pure Receivers  ({len(pure_receivers)})  :  "
          f"{sorted(int(x) for x in pure_receivers)}")
    print(sep)
    print(f"  Analyzing Node : {node_id_label}")
    print('--' * 50)

    if node_id_num in super_donors:
        recs = [n for n in range(total_nodes)
                if n != node_id_num and self._donor_of_op(n, 0) == node_id_num]
        dtp, rtp = node_id_num, recs
        print(f"  Node {node_id_label}  →  SUPER DONOR  |  donates to {len(recs)} nodes")
    elif node_id_num in solitary:
        dtp, rtp = node_id_num, []
        print(f"  Node {node_id_label}  →  SOLITARY DONOR  |  uses its own GFs only")
    else:
        dtp = self._donor_of_op(node_id_num, 0)
        rtp = [node_id_num]
        print(f"  Node {node_id_label}  →  RECEIVER  ←  donor {dtp}")
    print(sep + '\n')

    # Geometry -- include the QA point at the end so indexing matches
    # the donor / receiver ids we collected above.
    xyz_t = _rotate(self.xyz)
    xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None

    # Full xyz array (regular nodes + QA) used by the donor / receiver lookup.
    if xyz_qa_t is not None:
        xyz_all_t = np.vstack([xyz_t, xyz_qa_t])
    else:
        xyz_all_t = xyz_t
    
    if xyz_origin is not None and xyz_qa_t is not None:
        t = np.asarray(xyz_origin) - xyz_qa_t[0]
        xyz_t += t
        xyz_qa_t += t
        xyz_all_t = np.vstack([xyz_t, xyz_qa_t])
    
    bbox = xyz_t[self.internal] if self.internal.any() else xyz_t
    _, faces, bounds = self._build_cube_faces(bbox)

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2], marker='s',
               c='blue', s=30, alpha=0.1)

    # Donor point -- read from xyz_all_t so QA (last row) is reachable.
    dp = xyz_all_t[dtp]
    ax.scatter(*dp, c='red', marker='s', s=100,
               edgecolors='darkred', linewidths=2, zorder=10, alpha=0.5)

    # Receiver points
    for rec in rtp:
        rp = xyz_all_t[rec]
        ax.scatter(*rp, c='orange', marker='o', s=80,
                   edgecolors='darkorange', linewidths=1.5, alpha=0.5)
        ax.plot([dp[0], rp[0]], [dp[1], rp[1]], [dp[2], rp[2]],
                color='darkorange', linestyle='--', alpha=0.5, linewidth=2)

    # QA marker (siempre mostrar si existe)
    if xyz_qa_t is not None:
        ax.scatter(*xyz_qa_t[0], c='green', marker='*', s=300,
                   label='QA', zorder=10, edgecolors='black', linewidths=2)

    ax.add_collection3d(Poly3DCollection(faces, alpha=0.10, facecolor='red',
                                          edgecolor='darkred', linewidths=1.5))

    if label_nodes == 'donor_receivers':
        # Label donor
        x, y, z = xyz_all_t[dtp]
        dtp_label = 'QA' if dtp == self._n_nodes else str(dtp)
        ax.text(x, y, z, dtp_label, fontsize=10,
                color='darkred', fontweight='bold')
        # Label receivers
        for rec in rtp:
            x, y, z = xyz_all_t[rec]
            rec_label = 'QA' if rec == self._n_nodes else str(rec)
            ax.text(x, y, z, rec_label, fontsize=9,
                    color='darkblue', fontweight='bold')
    elif label_nodes:
        self._label_nodes_on_ax(ax, xyz_t, bounds, label_nodes, comp_donors)

    ax.set_xlabel("X' (m)")
    ax.set_ylabel("Y' (m)")
    ax.set_zlabel("Z' (m)")
    ax.legend()
    ax.grid(False)
    if axis_equal is True:
        ax.axis('equal')
    plt.tight_layout()
    plt.show()
