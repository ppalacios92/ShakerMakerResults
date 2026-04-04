"""
plotting.py
===========
Multi-model comparison plots for ShakerMakerData objects.

All functions accept a list of ShakerMakerData instances and a parallel
list of node-ID lists (one sub-list per model). This allows comparing
any combination of nodes across models in a single figure.

Typical usage
-------------
>>> from ShakerMakerResults import plot_models_response, plot_models_newmark, plot_models_gf

>>> plot_models_response(
...     models   = [surf1, surf2, drm1],
...     node_ids = [['QA', 0], ['QA', 5], [217]],
...     data_type = 'vel',
...     xlim      = [0, 30],
... )

Notes
-----
- ``node_ids`` must be a list of lists, one sub-list per model.
- Each sub-list may contain any mix of integer node indices and ``'QA'``.
- ``factor`` is always a multiplier applied to signals before plotting.
- Functions that depend on Station objects (compare_newmark, compare_fourier,
  plot_combined_response, plot_arias) are reserved for a future release.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

from .newmark import NewmarkSpectrumAnalyzer
from .utils import _rotate

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_label(obj, node_id):
    """Build a compact display label for a model + node combination.

    Parameters
    ----------
    obj : ShakerMakerData
    node_id : int or 'QA'

    Returns
    -------
    str
    """
    node_part = 'QA' if node_id in ('QA', 'qa') else f'N{node_id}'
    return f'{obj.model_name} | {node_part} | dt={obj.dt:.4f}s'


def _get_node_data(obj, node_id, data_type):
    """Return (z, e, n) signal tuple for a node, handling QA transparently.

    Parameters
    ----------
    obj : ShakerMakerData
    node_id : int or 'QA'
    data_type : {'vel', 'accel', 'disp'}

    Returns
    -------
    tuple of np.ndarray
        (z, e, n) each of shape (Nt,)
    """
    if node_id in ('QA', 'qa'):
        data = obj.get_qa_data(data_type)
    else:
        data = obj.get_node_data(node_id, data_type)
    return data[0], data[1], data[2]


def _get_gf_time(obj, slot):
    """Return the GF time vector for a given slot, accounting for t0.

    Parameters
    ----------
    obj : ShakerMakerData
    slot : int

    Returns
    -------
    np.ndarray
    """
    t0 = 0.0
    with h5py.File(obj._gf_h5_path, 'r') as f:
        t0_path = f'tdata_dict/{slot}_t0'
        if t0_path in f:
            t0 = float(f[t0_path][()])
    n = obj._n_time_gf
    return np.arange(n) * obj._dt_orig + t0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def plot_models_response(models,
                         node_ids=None,
                         target_pos=None,
                         data_type='vel',
                         xlim=None,
                         figsize=(10, 8),
                         factor=1.0):
    """Plot time-history response for multiple models, overlaid in one figure.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_ids : list of list, optional
        One sub-list per model with node indices (int or ``'QA'``).
        Example: ``[['QA', 0], ['QA', 5], [217]]``
    target_pos : list of array-like, optional
        One position per model ``[x, y, z]`` in km. Overrides ``node_ids``.
        Example: ``[[6,8,0], [6,8,0], [6,8,0]]``
    data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
    xlim : list of float, optional
    figsize : tuple of float, default ``(10, 8)``
    factor : float, default ``1.0``
    """
    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    n         = len(models)
    nids_list = node_ids   if node_ids   else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    ylabel = {'accel': 'Acceleration', 'vel': 'Velocity',
              'disp': 'Displacement'}[data_type]

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        for nid in nids:
            z, e, n = _get_node_data(obj, nid, data_type)
            lbl = _build_label(obj, nid)
            for ax, sig in zip(axes, (z, e, n)):
                ax.plot(obj.time, sig * factor, linewidth=1, label=lbl)

    comp_titles = ('Vertical (Z)', 'East (E)', 'North (N)')
    for ax, comp in zip(axes, comp_titles):
        ax.set_title(f'{comp} — {ylabel}', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()
    


def plot_models_newmark(models,
                        node_ids=None,
                        target_pos=None,
                        data_type='accel',
                        spectral_type='PSa',
                        xlim=None,
                        figsize=(8, 10),
                        factor=1.0):
    """Plot Newmark response spectra for multiple models, overlaid in one figure.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_ids : list of list, optional
        One sub-list per model with node indices (int or ``'QA'``).
    target_pos : list of array-like, optional
        One position per model ``[x, y, z]`` in km. Overrides ``node_ids``.
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    spectral_type : {'PSa', 'Sa', 'PSv', 'Sv', 'Sd'}, default ``'PSa'``
    xlim : list of float, optional
    figsize : tuple of float, default ``(8, 10)``
    factor : float, default ``1.0``
    """
    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    if xlim is None:
        xlim = [0, 5]

    n         = len(models)
    nids_list = node_ids   if node_ids   else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    scale  = 1.0 / 9.81 if data_type == 'accel' else 1.0
    ylabel = {'PSa': 'PSa (g)', 'Sa': 'Sa (g)', 'PSv': 'PSv (m/s)',
              'Sv': 'Sv (m/s)', 'Sd': 'Sd (m)'}.get(spectral_type, spectral_type)

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        dt = obj.time[1] - obj.time[0]
        for nid in nids:
            z, e, n = _get_node_data(obj, nid, data_type)
            lbl = _build_label(obj, nid)
            specs = [NewmarkSpectrumAnalyzer.compute(sig * factor * scale, dt)
                     for sig in (z, e, n)]
            T = specs[0]['T']
            for ax, sp in zip(axes, specs):
                ax.plot(T, sp[spectral_type], linewidth=2, label=lbl)

    comp_titles = ('Vertical (Z)', 'East (E)', 'North (N)')
    for ax, comp in zip(axes, comp_titles):
        ax.set_title(f'{comp} — {spectral_type}', fontweight='bold')
        ax.set_xlabel('T (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()



def plot_models_gf(models,
                   node_ids=None,
                   target_pos=None,
                   subfault=0,
                   xlim=None,
                   figsize=(8, 10),
                   factor=1.0):
    """Plot Green's function time series for multiple models, overlaid in one figure.

    Only models that have a GF database loaded (via ``load_gf_database()``)
    will contribute curves. Models without GFs are skipped with a warning.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_ids : list of list, optional
        One sub-list per model with node indices (int or ``'QA'``).
        Example: ``[[0, 5], [217], ['QA']]``
    target_pos : list of array-like, optional
        One position per model ``[x, y, z]`` in km. Overrides ``node_ids``
        when provided. Use ``None`` entries to fall back to ``node_ids``
        for specific models. Example: ``[[6,8,0], None, [6,8,0]]``
    subfault : int or list of int, default ``0``
    xlim : list of float, optional
    figsize : tuple of float, default ``(8, 10)``
    factor : float, default ``1.0``
    """
    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    sub_ids   = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
    n         = len(models)
    nids_list = node_ids  if node_ids   else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        if not obj._gf_loaded:
            print(f"  Warning: {obj.model_name} has no GFs loaded — skipped.")
            continue

        # Resolve node IDs
        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        for nid in nids:
            if nid in ('QA', 'qa'):
                nid_num   = obj._n_nodes
                nid_label = 'QA'
            else:
                nid_num   = nid
                nid_label = f'N{nid}'
            for sid in sub_ids:
                slot = obj._get_slot(nid_num, sid)
                time = _get_gf_time(obj, slot)
                lbl  = f'{obj.model_name} | {nid_label} | S{sid} | dt={obj.dt:.4f}s'
                for ax, comp in zip(axes, ('z', 'e', 'n')):
                    gf = obj.get_gf(nid, sid, comp) * factor
                    ax.plot(time, gf, linewidth=1, label=lbl)

    comp_titles = ('Vertical (Z)', 'East (E)', 'North (N)')
    for ax, comp in zip(axes, comp_titles):
        ax.set_title(f'{comp} — Green Function', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def plot_models_tensor_gf(models,
                          node_ids=None,
                          target_pos=None,
                          subfault=0,
                          xlim=None,
                          figsize=(12, 10),
                          factor=1.0):
    """Plot the 9-component tensor Green's functions for multiple models.

    Only models with a GF database loaded are plotted. Each component
    ``G_ij`` occupies one subplot in a 3x3 grid.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_ids : list of list, optional
        One sub-list per model with node indices (int or ``'QA'``).
    target_pos : list of array-like, optional
        One position per model ``[x, y, z]`` in km. Overrides ``node_ids``
        when provided. Use ``None`` entries to fall back to ``node_ids``
        for specific models.
    subfault : int or list of int, default ``0``
    xlim : list of float, optional
    figsize : tuple of float, default ``(12, 10)``
    factor : float, default ``1.0``
    """
    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    sub_ids   = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
    labels    = [f'G_{i+1}{j+1}' for i in range(3) for j in range(3)]
    n         = len(models)
    nids_list = node_ids   if node_ids   else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        if not obj._gf_loaded:
            print(f"  Warning: {obj.model_name} has no GFs loaded — skipped.")
            continue

        # Resolve node IDs
        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        for nid in nids:
            if nid in ('QA', 'qa'):
                nid_num   = obj._n_nodes
                nid_label = 'QA'
            else:
                nid_num   = nid
                nid_label = f'N{nid}'
            for sid in sub_ids:
                slot  = obj._get_slot(nid_num, sid)
                donor = obj._pairs_to_compute[slot, 0]
                if donor != nid_num:
                    print(f"  {obj.model_name} | {nid_label}/S{sid} "
                          f"→ slot {slot} (donor {donor})")
                time = _get_gf_time(obj, slot)
                lbl  = f'{obj.model_name} | {nid_label} | S{sid}'
                with h5py.File(obj._gf_h5_path, 'r') as f:
                    tp = f'tdata_dict/{slot}_tdata'
                    if tp not in f:
                        print(f"  tdata not found: {tp} — skipped.")
                        continue
                    tdata = f[tp][:] * factor
                for j in range(9):
                    axes[j // 3, j % 3].plot(time, tdata[:, j],
                                              linewidth=0.8, label=lbl)

    for j, lbl in enumerate(labels):
        ax = axes[j // 3, j % 3]
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(xlim)

    axes[0, 2].legend(fontsize=8)
    plt.suptitle('Tensor Green Functions — Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_models_domain(models,
                    xlim=None,
                    ylim=None,
                    zlim=None,
                    label_nodes=False,
                    show='all',
                    show_nodes=True,
                    show_cubes=True,
                    axis_equal=True,
                    figsize=(10, 8)):
    """Plot multiple ShakerMakerData domains in one 3-D figure.

    Each model is rendered in a distinct colour. The QA station is marked
    with a star and the bounding box with a translucent cube.

    Parameters
    ----------
    models : list of ShakerMakerData
    xlim, ylim, zlim : list of float, optional
        Axis limits in metres.
    label_nodes : bool or str, default ``False``
        Node labelling mode passed to ``_label_nodes_on_ax``.
    show : {'all', 'internal', 'boundary'}, default ``'all'``
        Which node subsets to scatter.
    show_nodes : bool, default ``True``
    show_cubes : bool, default ``True``
    figsize : tuple of float, default ``(10, 8)``
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    for obj, color in zip(models, colors):
        xyz_t    = _rotate(obj.xyz)
        xyz_qa_t = _rotate(obj.xyz_qa) if obj.xyz_qa is not None else None
        xyz_int  = xyz_t[obj.internal]
        xyz_ext  = xyz_t[~obj.internal]

        if show_nodes:
            if show in ('all', 'boundary') and len(xyz_ext) > 0:
                ax.scatter(xyz_ext[:, 0], xyz_ext[:, 1], xyz_ext[:, 2],
                           c=[color], marker='o', s=50, alpha=0.3)
            if show in ('all', 'internal') and len(xyz_int) > 0:
                ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2],
                           c=[color], marker='s', s=30, alpha=0.6)
            if show == 'all' and len(xyz_int) == 0:
                # SurfaceGrid — all nodes are external
                ax.scatter(xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2],
                           c=[color], marker='o', s=30, alpha=0.4)

        if xyz_qa_t is not None:
            ax.scatter(xyz_qa_t[:, 0], xyz_qa_t[:, 1], xyz_qa_t[:, 2],
                       c=[color], marker='*', s=300,
                       edgecolors='black', linewidths=2,
                       label=obj.name)

        if show_cubes:
            bbox = xyz_int if len(xyz_int) > 0 else xyz_t
            x0, x1 = bbox[:, 0].min(), bbox[:, 0].max()
            y0, y1 = bbox[:, 1].min(), bbox[:, 1].max()
            z0, z1 = bbox[:, 2].min(), bbox[:, 2].max()
            c = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                           [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
            faces = [[c[4],c[5],c[6],c[7]],[c[0],c[1],c[5],c[4]],
                     [c[2],c[3],c[7],c[6]],[c[0],c[3],c[7],c[4]],
                     [c[1],c[2],c[6],c[5]]]
            ax.add_collection3d(
                Poly3DCollection(faces, alpha=0.15,
                                 facecolor=color, edgecolor=color,
                                 linewidths=2))

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)
    ax.legend()
    ax.grid(False)
    if axis_equal is True:
        ax.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_models_arias(models,
                      node_ids=None,
                      target_pos=None,
                      data_type='accel',
                      xlim=None,
                      figsize=(10, 8),
                      factor=1.0):
    """Plot Arias intensity curves for multiple models, overlaid in one figure.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_ids : list of list, optional
        One sub-list per model with node indices (int or ``'QA'``).
        Example: ``[['QA'], ['QA', 0], [217]]``
    target_pos : list of array-like, optional
        One position per model ``[x, y, z]`` in km. Overrides ``node_ids``.
        Example: ``[[6,8,0], [6,8,0], [6,8,0]]``
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    xlim : list of float, optional
    figsize : tuple of float, default ``(10, 8)``
    factor : float, default ``1.0``
        Multiplier applied to every signal before computing Arias intensity.
    """
    from EarthquakeSignal.core.arias_intensity import AriasIntensityAnalyzer

    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    n         = len(models)
    nids_list = node_ids   if node_ids   else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        dt = obj.time[1] - obj.time[0]

        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        for nid in nids:
            data, lbl = obj._resolve_node(nid, 'accel')
            for ax, sig in zip(axes, (data[0], data[1], data[2])):
                IA_pct, t_start, t_end, ia_total, _ = AriasIntensityAnalyzer.compute(
                    sig * factor / 9.81, dt)
                t = np.arange(len(IA_pct)) * dt
                line, = ax.plot(t, IA_pct, linewidth=1.5,
                                label=f"{lbl} | Ia={ia_total:.3f} m/s")
                ax.axvline(t_start, color=line.get_color(),
                           linestyle='--', linewidth=1, alpha=0.5)
                ax.axvline(t_end, color=line.get_color(),
                           linestyle='--', linewidth=1, alpha=0.5)

    for ax, comp in zip(axes, ('Vertical (Z)', 'East (E)', 'North (N)')):
        ax.axhline(5,  color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(95, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.set_title(f'{comp} — Arias Intensity', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('IA (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()
    

# ---------------------------------------------------------------------------
# Station + DRM mixed helpers — reserved for future release
# ---------------------------------------------------------------------------

# def plot_combined_response(...)  : coming soon
# def compare_newmark(...)         : coming soon
# def compare_fourier(...)         : coming soon
# def plot_arias(...)              : coming soon



# # ---------------------------------------------------------------------------
# # Mixed DRM + Station helpers
# # ---------------------------------------------------------------------------

# def plot_combined_response(models,
#                             data_type='vel',
#                             drm_node='QA',
#                             factor=1.0,
#                             xlim=None,
#                             filtered=False,
#                             figsize=(10, 8)):

#     """Plot time-history response for a mixed list of ShakerMakerData and StationRead.

#     Parameters
#     ----------
#     models : list
#         Mix of ``ShakerMakerData`` and ``StationRead`` objects.
#     data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
#     drm_node : int or 'QA', default ``'QA'``
#         Node used for ShakerMakerData models.
#     factor : float, default ``1.0``
#         Scale factor applied to all signals.
#     xlim : list, optional
#     filtered : bool, default ``False``
#         Use filtered data for StationRead objects.
#     figsize : tuple, default ``(10, 8)``
#     """
#     ylabel = {'accel': 'Acceleration', 'vel': 'Velocity',
#               'disp': 'Displacement'}.get(data_type, data_type)

#     fig, axes = plt.subplots(3, 1, figsize=figsize)

#     for obj in models:
#         if _is_station(obj):
#             z, e, n = _get_signals(obj, None, data_type, filtered)
#             lbl  = obj.name if obj.name else 'Station'
#             time = obj.t
#         else:
#             z, e, n = _get_signals(obj, drm_node, data_type)
#             node_part = '_QA' if drm_node in ('QA', 'qa') else f'_N{drm_node}'
#             lbl  = f'{obj.model_name}{node_part}'
#             time = obj.time

#         for ax, sig in zip(axes, (z, e, n)):
#             ax.plot(time, sig / factor, linewidth=1, label=lbl)

#     for ax, comp in zip(axes, ('Vertical (Z)', 'East (E)', 'North (N)')):
#         ax.set_title(f'{comp} — {ylabel}', fontweight='bold')
#         ax.set_xlabel('Time [s]')
#         ax.set_ylabel('Amplitude')
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc='upper right')
#         if xlim:
#             ax.set_xlim(xlim)

#     plt.tight_layout()
#     plt.show()


# def compare_newmark(models, 
#                     data_type='accel', 
#                     spectral_type='PSa',
#                     drm_node='QA', 
#                     factor=1.0, 
#                     xlim=None,
#                     filtered=False, 
#                     figsize=(6, 10)):

#     """Plot Newmark spectra for a mixed list of ShakerMakerData and StationRead.

#     Parameters
#     ----------
#     models : list
#     data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
#     spectral_type : {'PSa', 'Sa', 'PSv', 'Sv', 'Sd'}, default ``'PSa'``
#     drm_node : int or 'QA', default ``'QA'``
#     factor : float, default ``1.0``
#     xlim : list, default ``[0, 5]``
#     filtered : bool, default ``False``
#     figsize : tuple, default ``(6, 10)``
#     """
#     if xlim is None:
#         xlim = [0, 5]
#     ylabel = {'PSa': 'PSa (g)', 'Sa': 'Sa (g)', 'PSv': 'PSv (m/s)',
#               'Sv': 'Sv (m/s)', 'Sd': 'Sd (m)'}.get(spectral_type, spectral_type)

#     fig, axes = plt.subplots(3, 1, figsize=figsize)

#     for obj in models:
#         if _is_station(obj):
#             z, e, n = _get_signals(obj, None, data_type, filtered)
#             lbl = obj.name if obj.name else 'Station'
#             dt  = obj.dt
#         else:
#             z, e, n = _get_signals(obj, drm_node, data_type)
#             node_part = '_QA' if drm_node in ('QA', 'qa') else f'_N{drm_node}'
#             lbl = f'{obj.model_name}{node_part}'
#             dt  = obj.time[1] - obj.time[0]

#         specs = [NewmarkSpectrumAnalyzer.compute(sig / factor, dt)
#                  for sig in (z, e, n)]
#         T = specs[0]['T']

#         for ax, sp in zip(axes, specs):
#             ax.plot(T, sp[spectral_type], linewidth=2, label=lbl)

#     for ax, comp in zip(axes, ('Vertical (Z)', 'East (E)', 'North (N)')):
#         ax.set_title(f'{comp} — {spectral_type}', fontweight='bold')
#         ax.set_xlabel('T (s)')
#         ax.set_ylabel(ylabel)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#         ax.set_xlim(xlim)

#     plt.tight_layout()
#     plt.show()


# def compare_fourier(models, 
#                     data_type='accel', 
#                     drm_node='QA',
#                     xlim=None, 
#                     filtered=False, 
#                     factor=1.0, 
#                     figsize=(8, 10)):

#     """Plot Fourier spectra for a mixed list of ShakerMakerData and StationRead.

#     Parameters
#     ----------
#     models : list
#     data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
#     drm_node : int or 'QA', default ``'QA'``
#     xlim : list, optional
#     filtered : bool, default ``False``
#     factor : float, default ``1.0``
#     figsize : tuple, default ``(8, 10)``
#     """
#     fig = plt.figure(figsize=figsize)

#     for obj in models:
#         if _is_station(obj):
#             dtype_map = {'accel': 'acceleration', 'vel': 'velocity',
#                          'disp': 'displacement'}
#             freqs, z_amp, e_amp, n_amp = obj.get_fourier(
#                 dtype_map[data_type], filtered=filtered)
#             lbl = obj.name if obj.name else 'Station'
#         else:
#             z, e, n = _get_signals(obj, drm_node, data_type)
#             dt    = obj.time[1] - obj.time[0]
#             freqs = np.fft.rfftfreq(len(obj.time), dt)
#             z_amp = np.abs(np.fft.rfft(z)) * dt
#             e_amp = np.abs(np.fft.rfft(e)) * dt
#             n_amp = np.abs(np.fft.rfft(n)) * dt
#             node_part = '_QA' if drm_node in ('QA', 'qa') else f'_N{drm_node}'
#             lbl = f'{obj.model_name}{node_part}'

#         for k, amp in enumerate((z_amp, e_amp, n_amp), 1):
#             plt.subplot(3, 1, k)
#             plt.semilogx(freqs, amp / factor, linewidth=1, label=lbl)

#     for k, comp in enumerate(('Vertical (Z)', 'East (E)', 'North (N)'), 1):
#         ax = plt.subplot(3, 1, k)
#         ax.set_title(f'{comp} — Fourier Spectrum', fontweight='bold')
#         ax.set_xlabel('Frequency [Hz]')
#         ax.set_ylabel('Amplitude')
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#         if xlim:
#             ax.set_xlim(xlim)

#     plt.tight_layout()
#     plt.show()


# # -------------------------

#     def plot_models_arias(models,
#                           node_ids,
#                           data_type='accel',
#                           xlim=None,
#                           figsize=(10, 8),
#                           factor=1.0):
#         """Plot Arias intensity curves for multiple models, overlaid in one figure.

#         Parameters
#         ----------
#         models : list of ShakerMakerData
#             Models to compare.
#         node_ids : list of list
#             One sub-list per model with node indices (int or ``'QA'``).
#             Example: ``[['QA'], ['QA', 0], [217]]``
#         data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
#             Signal type. Arias intensity is computed from acceleration.
#         xlim : list of float, optional
#             Time axis limits ``[t_min, t_max]`` in seconds.
#         figsize : tuple of float, default ``(10, 8)``
#         factor : float, default ``1.0``
#             Multiplier applied to every signal before computing Arias intensity.
#         """
#         from EarthquakeSignal.core.arias_intensity import AriasIntensityAnalyzer

#         if len(models) != len(node_ids):
#             raise ValueError("models and node_ids must have the same length.")

#         fig, axes = plt.subplots(3, 1, figsize=figsize)

#         for obj, nids in zip(models, node_ids):
#             dt = obj.time[1] - obj.time[0]
#             for nid in nids:
#                 z, e, n = _get_node_data(obj, nid, data_type)
#                 lbl = _build_label(obj, nid)
#                 for ax, sig in zip(axes, (z, e, n)):
#                     IA_pct, t_start, t_end, ia_total, _ = AriasIntensityAnalyzer.compute(
#                         sig * factor / 9.81, dt)
#                     t = np.arange(len(IA_pct)) * dt
#                     line, = ax.plot(t, IA_pct, linewidth=1.5,
#                                     label=f"{lbl} | Ia={ia_total:.3f} m/s")
#                     ax.axvline(t_start, color=line.get_color(),
#                                linestyle='--', linewidth=1, alpha=0.5)
#                     ax.axvline(t_end, color=line.get_color(),
#                                linestyle='--', linewidth=1, alpha=0.5)

#         for ax, comp in zip(axes, ('Vertical (Z)', 'East (E)', 'North (N)')):
#             ax.axhline(5,  color='gray', linestyle=':', linewidth=1, alpha=0.7)
#             ax.axhline(95, color='gray', linestyle=':', linewidth=1, alpha=0.7)
#             ax.set_title(f'{comp} — Arias Intensity', fontweight='bold')
#             ax.set_xlabel('Time [s]')
#             ax.set_ylabel('IA (%)')
#             ax.set_ylim(0, 100)
#             ax.grid(True, alpha=0.3)
#             ax.legend(loc='upper left')
#             if xlim:
#                 ax.set_xlim(xlim)

#         plt.tight_layout()
#         plt.show()