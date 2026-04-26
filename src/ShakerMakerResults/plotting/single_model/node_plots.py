"""Standalone node-level plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...analysis.newmark import NewmarkSpectrumAnalyzer
from ...core.gf_service import get_gf_tensor
from ...utils import _fk_tensor_rotation, _rotate

def plot_node_response(self,
                        node_id=None,
                        target_pos=None,
                        xlim=None,
                        data_type='vel',
                        figsize=(10, 8),
                        factor=1.0,
                        filtered=False):
    """Plot time-history for one or more nodes.

    Parameters
    ----------
    node_id : int, str, or list, optional
        Single node index, ``'QA'``, or list of indices/``'QA'``.
    target_pos : array-like (3,) or list of array-like, optional
        Single position ``[x, y, z]`` or list of positions
        ``[[x1,y1,z1], [x2,y2,z2], ...]`` in km.  The nearest node
        to each position is resolved automatically.
    xlim : list, optional
    data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
    figsize : tuple, default ``(10, 8)``
    factor : float, default ``1.0``
    filtered : bool, default ``False``
    """
    # Resolve node IDs — handle list of positions
    if target_pos is not None:
        target_pos = np.asarray(target_pos)
        if target_pos.ndim == 1:
            # Single position [x, y, z]
            nids = self._collect_node_ids(target_pos=target_pos)
        else:
            # Multiple positions [[x1,y1,z1], [x2,y2,z2], ...]
            nids = []
            for pos in target_pos:
                nids += self._collect_node_ids(target_pos=pos, print_info=True)
    else:
        nids = self._collect_node_ids(node_id=node_id)

    ylabel = {'accel': 'Acceleration', 'vel': 'Velocity',
              'disp': 'Displacement'}[data_type]
    fig = plt.figure(figsize=figsize)
    for nid in nids:
        data, lbl = self._resolve_node(nid, data_type)
        for k in range(1, 4):
            plt.subplot(3, 1, k)
            plt.plot(self.time, data[k-1] * factor, linewidth=1, label=lbl)
    for k, comp in enumerate(('Vertical (Z)', 'East (E)', 'North (N)'), 1):
        ax = plt.subplot(3, 1, k)
        ax.set_title(f'{comp} — {ylabel}', fontweight='bold')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        if xlim:
            ax.set_xlim(xlim)
    plt.tight_layout()
    plt.show()


def plot_node_gf(self,
                 node_id=None,
                 target_pos=None,
                 xlim=None,
                 subfault=0,
                 figsize=(8, 10),
                 ffsp_source=None,
                 strikes=None, dips=None, rakes=None,
                 src_x=None, src_y=None,
                 internal_ref=None,
                 external_coord=None):
    """Plot Green's function time series for one or more nodes.

    If ffsp_source or strikes/dips/rakes are provided, computes physical
    z/e/n components from the FK tensor. Otherwise plots raw FK kernels.

    Parameters
    ----------
    node_id : int, str, or list, optional
    target_pos : array-like (3,), optional
    xlim : list, optional
    subfault : int or list, default 0
    figsize : tuple, default (8, 10)
    ffsp_source : FFSPSource, optional
        If provided, extracts strike/dip/rake/position from this object.
    strikes, dips, rakes : list of float, optional
        Mechanism angles in degrees. One value per subfault index.
    src_x, src_y : list of float, optional
        Source positions in km (ShakerMaker coords). One per subfault.
    internal_ref : list [x, y], optional
        Reference point in FFSP local coords (km). Required with ffsp_source.
    external_coord : list [x, y], optional
        Target position in ShakerMaker coords (km). Required with ffsp_source.
    """
    if not self._has_gf or not self._has_map:
        print("No GFs available. Call load_gf_database() and load_map() first.")
        return

    nids    = self._collect_node_ids(node_id, target_pos)
    sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]

    # --- Resolve mechanism mode ---
    use_physical = ffsp_source is not None or strikes is not None

    # Precompute coordinate offset if using ffsp_source
    offset_x = offset_y = 0.0
    strike_rad_fault = 0.0
    if ffsp_source is not None:
        strike_rad_fault = np.radians(ffsp_source.params['strike'])
        if internal_ref is not None and external_coord is not None:
            ref_x, ref_y   = internal_ref
            ext_x, ext_y   = external_coord
            ref_x_rot      = ref_x * np.sin(strike_rad_fault) + ref_y * np.cos(strike_rad_fault)
            ref_y_rot      = ref_x * np.cos(strike_rad_fault) - ref_y * np.sin(strike_rad_fault)
            offset_x       = ext_x - ref_x_rot
            offset_y       = ext_y - ref_y_rot

    fig = plt.figure(figsize=figsize)

    for nid in nids:
        nid_num   = self._n_nodes if nid in ('QA', 'qa') else nid
        nid_label = 'QA'        if nid in ('QA', 'qa') else f'N{nid}'

        # Receiver position
        if nid in ('QA', 'qa'):
            rx, ry = self.xyz_qa[0, 0], self.xyz_qa[0, 1]
        else:
            rx, ry = self.xyz[nid, 0], self.xyz[nid, 1]

        for sid in sub_ids:
            gf_data = get_gf_tensor(self, nid, sid)
            tdata = gf_data['tdata']
            time = gf_data['time']
            lbl  = f'{nid_label}_S{sid}'

            if use_physical:
                # --- Resolve mechanism for this subfault ---
                if ffsp_source is not None:
                    sf   = ffsp_source.subfaults
                    pf   = np.radians(sf['strike'][sid])
                    df   = np.radians(sf['dip'][sid])
                    lf   = np.radians(sf['rake'][sid])
                    # Transform subfault position to ShakerMaker coords
                    sx_  = sf['x'][sid] / 1e3 * np.sin(strike_rad_fault) + offset_x
                    sy_  = sf['y'][sid] / 1e3 * np.cos(strike_rad_fault) + offset_y
                else:
                    pf   = np.radians(strikes[sid])
                    df   = np.radians(dips[sid])
                    lf   = np.radians(rakes[sid])
                    sx_  = src_x[sid]
                    sy_  = src_y[sid]

                # --- Azimuth source → receiver ---
                p = np.arctan2(ry - sy_, rx - sx_)

                # --- Slip and normal vectors ---
                f1 =  np.cos(lf)*np.cos(pf) + np.sin(lf)*np.cos(df)*np.sin(pf)
                f2 =  np.cos(lf)*np.sin(pf) - np.sin(lf)*np.cos(df)*np.cos(pf)
                f3 = -np.sin(lf)*np.sin(df)
                n1 = -np.sin(pf)*np.sin(df)
                n2 =  np.cos(pf)*np.sin(df)
                n3 = -np.cos(df)

                A  = (f1*n1 - f2*n2)*np.cos(2*p) + (f1*n2 + f2*n1)*np.sin(2*p)
                B  = (f1*n3 + f3*n1)*np.cos(p)   + (f2*n3 + f3*n2)*np.sin(p)
                C  = f3 * n3


                # z_gf =  tdata[:, 6]*A + tdata[:, 3]*B + tdata[:, 0]*C
                # r_gf =  tdata[:, 7]*A + tdata[:, 4]*B + tdata[:, 1]*C
                # t_gf =  tdata[:, 8]*A + tdata[:, 5]*B

                A_t  = (f1*n1 - f2*n2)*np.sin(2*p) - (f1*n2 + f2*n1)*np.cos(2*p)
                B_t  = (f1*n3 + f3*n1)*np.sin(p)   - (f2*n3 + f3*n2)*np.cos(p)

                z_gf =  tdata[:, 6]*A   + tdata[:, 3]*B   + tdata[:, 0]*C
                r_gf =  tdata[:, 7]*A   + tdata[:, 4]*B   + tdata[:, 1]*C
                t_gf =  tdata[:, 8]*A_t + tdata[:, 5]*B_t



                e_gf = -r_gf*np.sin(p) - t_gf*np.cos(p)
                n_gf = -r_gf*np.cos(p) + t_gf*np.sin(p)

                components = (z_gf, e_gf, n_gf)

            else:
                # Raw FK kernels (debug)
                components = (tdata[:, 0], tdata[:, 1], tdata[:, 2])

            for k, sig in enumerate(components, 1):
                plt.subplot(3, 1, k)
                plt.plot(time, sig, linewidth=1, label=lbl)

    comp_titles = ('Vertical (Z)', 'East (E)', 'North (N)')
    for k, title in enumerate(comp_titles, 1):
        ax = plt.subplot(3, 1, k)
        ax.set_title(f'{title} — Green Function', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()
    return {'time': time, 'z': z_gf, 'e': e_gf, 'n': n_gf} 


def plot_node_tensor_gf(self,
                        node_id=None,
                        target_pos=None,
                        xlim=None,
                        subfault=0,
                        figsize=(10, 8)):
    """Plot the 9-component tensor Green's functions."""
    if not self._gf_loaded:
        print("No GFs. Call load_gf_database() first.")
        return

    # Resolve node IDs — support list of positions
    if target_pos is not None:
        target_pos = np.asarray(target_pos)
        if target_pos.ndim == 1:
            # Single position [x, y, z]
            nids = self._collect_node_ids(target_pos=target_pos)
        else:
            # Multiple positions [[x1,y1,z1], [x2,y2,z2], ...]
            nids = []
            for pos in target_pos:
                nids += self._collect_node_ids(target_pos=pos, print_info=True)
    elif node_id is not None:
        if isinstance(node_id, (list, np.ndarray)):
            nids = []
            for nid in node_id:
                nids += self._collect_node_ids(node_id=nid, print_info=True)
        else:
            nids = self._collect_node_ids(node_id=node_id)
    else:
        raise ValueError("Provide node_id or target_pos.")

    sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
    labels  = [f'G_{i+1}{j+1}' for i in range(3) for j in range(3)]
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    results = {}

    for nid in nids:
        if nid in ('QA', 'qa'):
            nid_num   = self._n_nodes
            nid_label = 'QA'
        else:
            nid_num   = nid
            nid_label = f'N{nid}'

        for sid in sub_ids:
            slot  = self._get_slot(nid_num, sid)
            donor = self._pairs_to_compute[slot, 0]
            if donor != nid_num:
                print(f"Node {nid_label}/sub {sid} → slot {slot} (donor {donor})")

            gf_data = get_gf_tensor(self, nid, sid)
            tdata_out = gf_data['tdata']
            time = gf_data['time']
            lbl  = f'{nid_label}_S{sid}'

            for j in range(9):
                axes[j // 3, j % 3].plot(time, tdata_out[:, j],
                                          linewidth=0.8, label=lbl)

            results[f'{nid_label}_S{sid}'] = {
                'tdata'  : tdata_out,
                'time'   : time,
                't0'     : gf_data['t0'],
                'node_id': nid_num
            }

    for j, lbl in enumerate(labels):
        ax = axes[j // 3, j % 3]
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(xlim)

    axes[0, 2].legend(fontsize=8)
    plt.suptitle('Tensor Green Functions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return results


def plot_node_newmark(self,
                      node_id=None,
                      target_pos=None,
                      xlim=None,
                      data_type='accel',
                      figsize=(8, 10),
                      factor=1.0,
                      filtered=False,
                      spectral_type='PSa'):

    """Plot Newmark response spectra for one or more nodes.

    Parameters
    ----------
    node_id : int, str, or list, optional
    target_pos : array-like (3,), optional
    xlim : list, optional. Default ``[0, 5]``
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    figsize : tuple, default ``(8, 10)``
    factor : float, default ``1.0``
        Scale factor applied to all signals before computing spectra.
    filtered : bool, default ``False``
        Use filtered data (only applies when obj is StationData).
    spectral_type : {'PSa', 'Sa', 'PSv', 'Sv', 'Sd'}, default ``'PSa'``
    """
    if xlim is None: xlim = [0, 5]
    nids   = self._collect_node_ids(node_id, target_pos)

    dt     = self.time[1] - self.time[0]
    scale  = 1.0 / 9.81 if data_type == 'accel' else 1.0
    ylabel = {'PSa': 'PSa (g)', 'Sa': 'Sa (g)', 'PSv': 'PSv (m/s)',
              'Sv': 'Sv (m/s)', 'Sd': 'Sd (m)'}.get(spectral_type, spectral_type)
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    for nid in nids:
        data, lbl = self._resolve_node(nid, data_type)
        specs = [NewmarkSpectrumAnalyzer.compute(data[i] * scale * factor, dt) for i in range(3)]
        T = specs[0]['T']
        for ax, sp in zip(axes, specs):
            ax.plot(T, sp[spectral_type], linewidth=2, label=lbl)
    for ax, comp in zip(axes, ('Vertical (Z)', 'East (E)', 'North (N)')):
        ax.set_title(f'{comp} — {spectral_type}', fontweight='bold')
        ax.set_xlabel('T (s)', fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(xlim); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.show()


def plot_node_arias(self,
                    node_id=None,
                    target_pos=None,
                    data_type='accel',
                    xlim=None,
                    figsize=(10, 8),
                    factor=1.0):
    """Plot Arias intensity curves for one or more nodes.

    Parameters
    ----------
    node_id : int, str, or list, optional
    target_pos : array-like (3,), optional
    data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
    xlim : list, optional
    figsize : tuple, default ``(10, 8)``
    factor : float, default ``1.0``
        Multiplier applied to acceleration before computing Arias intensity.
    """
    from EarthquakeSignal.core.arias_intensity import AriasIntensityAnalyzer

    nids = self._collect_node_ids(node_id, target_pos)

    dt   = self.time[1] - self.time[0]

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for nid in nids:
        data, lbl = self._resolve_node(nid, data_type)
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
