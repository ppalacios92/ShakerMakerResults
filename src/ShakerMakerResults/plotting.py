"""
plotting.py
===========
Plotting functions for comparing multiple ShakerMakerData objects.

All functions accept a list of ``ShakerMakerData`` instances (or a mix of
``ShakerMakerData`` and ``StationRead`` for the combined helpers) and produce
matplotlib figures.

Author: Patricio Palacios B.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .newmark import NewmarkSpectrumAnalyzer

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_R = np.column_stack([
    np.array([0, 1, 0]),
    np.array([1, 0, 0]),
    np.cross(np.array([0, 1, 0]), np.array([1, 0, 0]))
])


def _rotate(xyz_km):
    return xyz_km * 1000 @ _R


def _is_station(obj):
    """Return True if obj is a StationRead (not a ShakerMakerData)."""
    return hasattr(obj, 'z_v') and not hasattr(obj, 'internal')


def _resolve_node(node_id, model_index, n_models):
    """Extract a single node index for a given model from flexible input."""
    if not isinstance(node_id, list):
        return node_id
    if isinstance(node_id[0], list):
        return node_id[model_index][0]
    if len(node_id) == n_models:
        return node_id[model_index]
    return node_id[0]


def _get_data_drm(obj, node_idx, data_type):
    """Return (Z, E, N) arrays from a ShakerMakerData object."""
    if node_idx in ('QA', 'qa') or (
            isinstance(node_idx, int) and node_idx >= len(obj.xyz)):
        d = obj.get_qa_data(data_type)
    else:
        d = obj.get_node_data(node_idx, data_type)
    return d[2], d[0], d[1]   # Z, E, N


def _get_data_station(obj, data_type, filtered=False):
    """Return (Z, E, N) arrays from a StationRead object."""
    if data_type in ('accel', 'acceleration'):
        z, e, n = obj.acceleration_filtered if filtered else obj.acceleration
    elif data_type in ('vel', 'velocity'):
        z, e, n = obj.velocity_filtered if filtered else obj.velocity
    else:
        z, e, n = obj.displacement_filtered if filtered else obj.displacement
    return z, e, n


def _label(obj, node_idx=None):
    """Short display label for a model object."""
    if _is_station(obj):
        return obj.name if obj.name else "Station"
    node_part = f'_N{node_idx}' if node_idx not in (None, 'QA', 'qa') else '_QA'
    return f'{obj.model_name}{node_part}_dt={obj.dt:.4f}s'


# ---------------------------------------------------------------------------
# Multi-model comparison helpers
# ---------------------------------------------------------------------------

def plot_models_response(models, node_id, xlim=None, data_type='vel'):
    """Plot time-history response for multiple ShakerMakerData models.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_id : int, str, or list
        Single value applied to all models, or a list (one per model), or
        a list of lists. Use ``'QA'`` for the QA station.
    xlim : list, optional
        Time axis limits ``[tmin, tmax]``.
    data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
    """
    n      = len(models)
    ylabel = {'accel': 'Acceleration', 'vel': 'Velocity',
              'disp': 'Displacement'}[data_type]

    fig = plt.figure(figsize=(8, 8))

    for i, obj in enumerate(models):
        nid  = _resolve_node(node_id, i, n)
        nodes = nid if isinstance(nid, (list, np.ndarray)) else [nid]
        for node_idx in nodes:
            z, e, nn = _get_data_drm(obj, node_idx, data_type)
            lbl = _label(obj, node_idx)
            for k, data in enumerate((z, e, nn), 1):
                plt.subplot(3, 1, k)
                plt.plot(obj.time, data, linewidth=1, label=lbl)

    for k, comp in enumerate(('Vertical (Z)', 'X', 'Y'), 1):
        ax = plt.subplot(3, 1, k)
        ax.set_title(f'{comp} — {ylabel}', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(xlim)

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -1), ncol=2)
    plt.tight_layout()
    plt.show()


def plot_models_gf(models, node_id, subfault, xlim=None):
    """Plot Green's function time series for multiple ShakerMakerData models.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_id : int or list
    subfault : int or list
    xlim : list, optional
    """
    n       = len(models)
    sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]

    fig = plt.figure(figsize=(8, 10))

    for i, obj in enumerate(models):
        nid   = _resolve_node(node_id, i, n)
        nodes = nid if isinstance(nid, (list, np.ndarray)) else [nid]
        for node_idx in nodes:
            if node_idx in ('QA', 'qa'):
                continue
            for sid in sub_ids:
                lbl = f'{obj.model_name}_N{node_idx}_S{sid}_dt={obj.dt:.4f}s'
                for k, comp in enumerate(('z', 'e', 'n'), 1):
                    plt.subplot(3, 1, k)
                    plt.plot(obj.gf_time, obj.get_gf(node_idx, sid, comp),
                             linewidth=1, label=lbl)

    for k, title in enumerate(('Vertical (Z)', 'East (E)', 'North (N)'), 1):
        ax = plt.subplot(3, 1, k)
        ax.set_title(f'{title} — Green Function', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def plot_models_f_spectrum(models, node_id, subfault, xlim=None):
    """Plot Fourier magnitude spectra for multiple ShakerMakerData models.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_id : int or list
    subfault : int or list
    xlim : list, optional
        Frequency axis limits ``[fmin, fmax]``.
    """
    n       = len(models)
    sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]

    fig = plt.figure(figsize=(8, 10))

    for i, obj in enumerate(models):
        nid   = _resolve_node(node_id, i, n)
        nodes = nid if isinstance(nid, (list, np.ndarray)) else [nid]
        for node_idx in nodes:
            if node_idx in ('QA', 'qa'):
                continue
            for sid in sub_ids:
                try:
                    mags = [np.sqrt(obj.get_spectrum(node_idx, sid, c, 'real') ** 2 +
                                    obj.get_spectrum(node_idx, sid, c, 'imag') ** 2)
                            for c in ('z', 'e', 'n')]
                    lbl = f'{obj.model_name}_N{node_idx}_S{sid}'
                    for k, mag in enumerate(mags, 1):
                        plt.subplot(3, 1, k)
                        plt.loglog(obj.freqs, mag, linewidth=1, alpha=0.7, label=lbl)
                except KeyError:
                    print(f"  ! No spectrum for N{node_idx} S{sid}")

    for k, title in enumerate(('Vertical (Z)', 'East (E)', 'North (N)'), 1):
        ax = plt.subplot(3, 1, k)
        ax.set_title(f'{title} — Fourier Spectrum', fontweight='bold')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude (log scale)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def plot_models_newmark_spectra(models, node_id=None, target_pos=None,
                                xlim=None, data_type='accel', figsize=(8, 10)):
    """Plot Newmark response spectra for multiple ShakerMakerData models.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_id : int, str, or list, optional
    target_pos : array-like (3,), optional
        Find the nearest node in each model to this km-coordinate.
    xlim : list, default ``[0, 5]``
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    figsize : tuple, default ``(8, 10)``
    """
    if xlim is None:
        xlim = [0, 5]
    if node_id is None and target_pos is None:
        raise ValueError("Provide node_id or target_pos.")

    n      = len(models)
    ylabel = 'Sa (g)' if data_type == 'accel' else 'Spectral Response'
    scale  = 1.0 / 9.81 if data_type == 'accel' else 1.0

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for i, obj in enumerate(models):
        if target_pos is not None:
            dist = np.linalg.norm(obj.xyz_all - np.asarray(target_pos), axis=1)
            node_idx = int(np.argmin(dist))
        else:
            node_idx = _resolve_node(node_id, i, n)

        z, e, nn = _get_data_drm(obj, node_idx, data_type)
        dt  = obj.time[1] - obj.time[0]
        lbl = _label(obj, node_idx)

        specs = [NewmarkSpectrumAnalyzer.compute(sig * scale, dt)
                 for sig in (z, e, nn)]
        T = specs[0]['T']
        for ax, sp in zip(axes, specs):
            ax.plot(T, sp['PSa'], linewidth=2, label=lbl)

    for ax, comp in zip(axes, ('Vertical (Z)', 'X', 'Y')):
        ax.set_title(f'{comp} — Newmark Spectrum', fontweight='bold')
        ax.set_xlabel('T (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_models_DRM(models, xlim=None, ylim=None, zlim=None,
                    label_nodes=False, show='all',
                    show_nodes=True, show_cubes=True):
    """Plot multiple ShakerMakerData domains in one 3-D figure.

    Parameters
    ----------
    models : list of ShakerMakerData
    xlim, ylim, zlim : list, optional
        Axis limits.
    label_nodes : bool or str, optional
        ``'corners'``, ``'corners_edges'``, ``'corners_half'``, or ``False``.
    show : {'all', 'internal', 'boundary'}, default ``'all'``
    show_nodes : bool, default ``True``
    show_cubes : bool, default ``True``
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    for i, obj in enumerate(models):
        color    = colors[i]
        xyz_t    = _rotate(obj.xyz)
        xyz_qa_t = _rotate(obj.xyz_qa) if obj.xyz_qa is not None else None
        xyz_int  = xyz_t[obj.internal]
        xyz_ext  = xyz_t[~obj.internal]

        if show_nodes:
            if show in ('all', 'boundary'):
                ax.scatter(xyz_ext[:, 0], xyz_ext[:, 1], xyz_ext[:, 2],
                           c=[color], marker='o', s=50, alpha=0.3)
            if show in ('all', 'internal'):
                ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2],
                           c=[color], marker='s', s=30, alpha=0.6)

        if xyz_qa_t is not None:
            ax.scatter(xyz_qa_t[:, 0], xyz_qa_t[:, 1], xyz_qa_t[:, 2],
                       c=[color], marker='*', s=300,
                       edgecolors='black', linewidths=2,
                       label=obj.model_name)

        if show_cubes:
            x_min, x_max = xyz_int[:, 0].min(), xyz_int[:, 0].max()
            y_min, y_max = xyz_int[:, 1].min(), xyz_int[:, 1].max()
            z_min, z_max = xyz_int[:, 2].min(), xyz_int[:, 2].max()
            corners = np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max],
            ])
            faces = [
                [corners[4], corners[5], corners[6], corners[7]],
                [corners[0], corners[1], corners[5], corners[4]],
                [corners[2], corners[3], corners[7], corners[6]],
                [corners[0], corners[3], corners[7], corners[4]],
                [corners[1], corners[2], corners[6], corners[5]],
            ]
            ax.add_collection3d(
                Poly3DCollection(faces, alpha=0.15, facecolor=color,
                                 edgecolor=color, linewidths=2))

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_models_tensor_gf(models, node_id, subfault, xlim=None):
    """Plot 9-component tensor Green's functions for multiple models.

    Parameters
    ----------
    models : list of ShakerMakerData
    node_id : list of int (one per model)
    subfault : int or list
    xlim : list, optional
    """
    n       = len(models)
    sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
    labels  = [f'G_{i+1}{j+1}' for i in range(3) for j in range(3)]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for i, obj in enumerate(models):
        nid   = _resolve_node(node_id, i, n)
        nodes = nid if isinstance(nid, (list, np.ndarray)) else [nid]
        for node_idx in nodes:
            if node_idx in ('QA', 'qa'):
                continue
            for sid in sub_ids:
                if obj.node_mapping is not None:
                    mask  = ((obj.node_mapping[:, 0] == node_idx) &
                             (obj.node_mapping[:, 1] == sid))
                    idx   = np.where(mask)[0]
                    if not len(idx):
                        continue
                    ipair = obj.node_mapping[idx[0], 2]
                    src_n, src_s = obj.pairs_mapping[ipair]
                else:
                    src_n, src_s = node_idx, sid

                with h5py.File(obj.filename, 'r') as f:
                    path = f'GF/sta_{src_n}/sub_{src_s}/tdata'
                    if path not in f:
                        continue
                    tdata = f[path][:]
                    t0    = f[f'GF/sta_{src_n}/sub_{src_s}/t0'][()]

                time  = np.arange(tdata.shape[0]) * obj.dt + t0
                lbl   = f'{obj.model_name}_N{node_idx}_S{sid}'
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

    axes[0, 0].legend(fontsize=8)
    plt.suptitle('Tensor Green Functions Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Mixed DRM + Station helpers
# ---------------------------------------------------------------------------

def plot_combined_response(models, data_type='vel', drm_node='QA',
                            factor=1.0, xlim=None, filtered=False):
    """Plot time-history response for a mixed list of ShakerMakerData and StationRead.

    Parameters
    ----------
    models : list
        Mix of ``ShakerMakerData`` and ``StationRead`` objects.
    data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
    drm_node : int or 'QA', default ``'QA'``
        Node used for ShakerMakerData models.
    factor : float, default ``1.0``
        Scale factor applied to all signals.
    xlim : list, optional
    filtered : bool, default ``False``
        Use filtered data for StationRead objects.
    """
    ylabel = {'accel': 'Acceleration', 'vel': 'Velocity',
              'disp': 'Displacement'}.get(data_type, data_type)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    for obj in models:
        if _is_station(obj):
            z, e, n = _get_data_station(obj, data_type, filtered)
            lbl  = obj.name if obj.name else 'Station'
            time = obj.t
        else:
            z, e, n = _get_data_drm(obj, drm_node, data_type)
            node_part = '_QA' if drm_node in ('QA', 'qa') else f'_N{drm_node}'
            lbl  = f'{obj.model_name}{node_part}'
            time = obj.time

        for ax, sig in zip(axes, (z, e, n)):
            ax.plot(time, sig / factor, linewidth=1, label=lbl)

    for ax, comp in zip(axes, ('Vertical (Z)', 'East (E)', 'North (N)')):
        ax.set_title(f'{comp} — {ylabel}', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def compare_newmark(models, data_type='accel', spectral_type='PSa',
                    drm_node='QA', factor=1.0, xlim=None,
                    filtered=False, figsize=(6, 10)):
    """Plot Newmark spectra for a mixed list of ShakerMakerData and StationRead.

    Parameters
    ----------
    models : list
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    spectral_type : {'PSa', 'Sa', 'PSv', 'Sv', 'Sd'}, default ``'PSa'``
    drm_node : int or 'QA', default ``'QA'``
    factor : float, default ``1.0``
    xlim : list, default ``[0, 5]``
    filtered : bool, default ``False``
    figsize : tuple, default ``(6, 10)``
    """
    if xlim is None:
        xlim = [0, 5]
    ylabel = {'PSa': 'PSa (g)', 'Sa': 'Sa (g)', 'PSv': 'PSv (m/s)',
              'Sv': 'Sv (m/s)', 'Sd': 'Sd (m)'}.get(spectral_type, spectral_type)

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj in models:
        if _is_station(obj):
            z, e, n = _get_data_station(obj, data_type, filtered)
            lbl = obj.name if obj.name else 'Station'
            dt  = obj.dt
        else:
            z, e, n = _get_data_drm(obj, drm_node, data_type)
            node_part = '_QA' if drm_node in ('QA', 'qa') else f'_N{drm_node}'
            lbl = f'{obj.model_name}{node_part}'
            dt  = obj.time[1] - obj.time[0]

        specs = [NewmarkSpectrumAnalyzer.compute(sig / factor, dt)
                 for sig in (z, e, n)]
        T = specs[0]['T']

        for ax, sp in zip(axes, specs):
            ax.plot(T, sp[spectral_type], linewidth=2, label=lbl)

    for ax, comp in zip(axes, ('Vertical (Z)', 'East (E)', 'North (N)')):
        ax.set_title(f'{comp} — {spectral_type}', fontweight='bold')
        ax.set_xlabel('T (s)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def compare_fourier(models, data_type='accel', drm_node='QA',
                    xlim=None, filtered=False, factor=1.0, figsize=(8, 10)):
    """Plot Fourier spectra for a mixed list of ShakerMakerData and StationRead.

    Parameters
    ----------
    models : list
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    drm_node : int or 'QA', default ``'QA'``
    xlim : list, optional
    filtered : bool, default ``False``
    factor : float, default ``1.0``
    figsize : tuple, default ``(8, 10)``
    """
    fig = plt.figure(figsize=figsize)

    for obj in models:
        if _is_station(obj):
            dtype_map = {'accel': 'acceleration', 'vel': 'velocity',
                         'disp': 'displacement'}
            freqs, z_amp, e_amp, n_amp = obj.get_fourier(
                dtype_map[data_type], filtered=filtered)
            lbl = obj.name if obj.name else 'Station'
        else:
            z, e, n = _get_data_drm(obj, drm_node, data_type)
            dt    = obj.time[1] - obj.time[0]
            freqs = np.fft.rfftfreq(len(obj.time), dt)
            z_amp = np.abs(np.fft.rfft(z)) * dt
            e_amp = np.abs(np.fft.rfft(e)) * dt
            n_amp = np.abs(np.fft.rfft(n)) * dt
            node_part = '_QA' if drm_node in ('QA', 'qa') else f'_N{drm_node}'
            lbl = f'{obj.model_name}{node_part}'

        for k, amp in enumerate((z_amp, e_amp, n_amp), 1):
            plt.subplot(3, 1, k)
            plt.semilogx(freqs, amp / factor, linewidth=1, label=lbl)

    for k, comp in enumerate(('Vertical (Z)', 'East (E)', 'North (N)'), 1):
        ax = plt.subplot(3, 1, k)
        ax.set_title(f'{comp} — Fourier Spectrum', fontweight='bold')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()


def plot_arias(models, data_type='accel', drm_node='QA',
               xlim=None, figsize=(8, 10)):
    """Plot Arias intensity curves for a mixed list of models.

    Requires ``AriasIntensityAnalyzer`` from ``EarthquakeSignal``.

    Parameters
    ----------
    models : list
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    drm_node : int or 'QA', default ``'QA'``
    xlim : list, optional
    figsize : tuple, default ``(8, 10)``
    """
    from EarthquakeSignal.core.arias_intensity import AriasIntensityAnalyzer

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj in models:
        if _is_station(obj):
            z, e, n = obj.acceleration
            dt   = obj.dt
            time = obj.t
            lbl  = obj.name if obj.name else 'Station'
        else:
            z, e, n = _get_data_drm(obj, drm_node, 'accel')
            dt   = obj.time[1] - obj.time[0]
            time = obj.time
            node_part = '_QA' if drm_node in ('QA', 'qa') else f'_N{drm_node}'
            lbl  = f'{obj.model_name}{node_part}'

        for ax, sig in zip(axes, (z, e, n)):
            IA_pct, t_start, t_end, ia_total, _ = AriasIntensityAnalyzer.compute(
                sig / 9.81, dt)
            t = np.arange(len(IA_pct)) * dt
            ax.plot(t, IA_pct, linewidth=1.5,
                    label=f"{lbl} | Ia={ia_total:.3f} m/s")
            ax.axvline(t_start, linestyle='--', linewidth=1, alpha=0.6)
            ax.axvline(t_end,   linestyle='--', linewidth=1, alpha=0.6)

    for ax, comp in zip(axes, ('Vertical (Z)', 'East (E)', 'North (N)')):
        ax.axhline(5,  color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(95, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.set_title(f'{comp} — Arias Intensity', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('IA (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()
