"""
shakermaker_data.py
===================
Unified reader for ShakerMaker HDF5 output files.

Supports DRM outputs (DRMHDF5StationListWriter + DRMBox / SurfaceGrid /
PointCloudDRMReceiver) and plain station outputs (HDF5StationListWriter).
The format is detected automatically from the HDF5 file structure.

HDF5 layouts recognised
------------------------
DRM layout  (DRMHDF5StationListWriter):
    /DRM_Data/{xyz, internal, velocity, acceleration, displacement}
    /DRM_QA_Data/{xyz, velocity, acceleration, displacement}
    /DRM_Metadata/{dt, tstart, tend, name, ...}
    /GF_tdata/{slot}_tdata, {slot}_t0                 (OP pipeline GFs)
    /GF_Database_Info/{pairs_to_compute, pair_to_slot, ...}
    /GF/sta_N/sub_M/{z, e, n, t, tdata, t0}          (legacy GFs)
    /Node_Mapping/...                                 (legacy mapping)

Station layout  (HDF5StationListWriter):
    /Data/{xyz, internal, velocity, acceleration, displacement}
    /Metadata/{dt, tstart, tend, name, ...}

Author: Patricio Palacios B.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from .newmark import NewmarkSpectrumAnalyzer

_R = np.column_stack([
    np.array([0, 1, 0]),
    np.array([1, 0, 0]),
    np.cross(np.array([0, 1, 0]), np.array([1, 0, 0]))
])


def _rotate(xyz_km):
    """Apply the ShakerMaker display rotation and convert km to m."""
    return xyz_km * 1000 @ _R


class ShakerMakerData:
    """Unified reader for ShakerMaker HDF5 output files.

    Automatically detects DRM layout (/DRM_Data) or station layout (/Data).
    Supports both the OP pipeline GF format (GF_tdata + pair_to_slot) and
    the legacy format (GF/sta_N/sub_M + Node_Mapping).

    Parameters
    ----------
    filename : str
    dt : float, optional
        Resample to this time step. Defaults to original dt in file.
    """

    def __init__(self, filename, dt=None):
        self.filename = filename

        with h5py.File(filename, 'r') as f:
            self.is_drm = 'DRM_Data' in f
            if self.is_drm:
                data_grp, meta_grp, qa_grp = 'DRM_Data', 'DRM_Metadata', 'DRM_QA_Data'
            else:
                data_grp, meta_grp, qa_grp = 'Data', 'Metadata', None

            self.xyz      = f[f'{data_grp}/xyz'][:]
            self.internal = f[f'{data_grp}/internal'][:]

            if qa_grp and f'{qa_grp}/xyz' in f:
                self.xyz_qa = f[f'{qa_grp}/xyz'][:]
            else:
                self.xyz_qa = None

            dt_orig = float(f[f'{meta_grp}/dt'][()])
            tstart  = float(f[f'{meta_grp}/tstart'][()])

            try:
                raw = f[f'{meta_grp}/name'][()]
                self.name = raw.decode() if isinstance(raw, bytes) else str(raw)
            except KeyError:
                self.name = filename

            n_nodes     = len(self.xyz)
            n_time_data = f[f'{data_grp}/velocity'].shape[1]

            self.freqs = None
            if 'GF_Spectrum/sta_0/sub_0/freqs' in f:
                self.freqs = f['GF_Spectrum/sta_0/sub_0/freqs'][:]

            # Detect GF time steps — OP format first, then legacy
            n_subfaults = 0; n_time_gf = 0
            if 'GF_tdata/0_tdata' in f:
                n_time_gf = f['GF_tdata/0_tdata'].shape[0]
            elif 'GF/sta_0' in f:
                n_subfaults = len(list(f['GF/sta_0'].keys()))
                n_time_gf   = len(f['GF/sta_0/sub_0/z'][:])
            elif 'GF_Spectrum/sta_0' in f:
                n_subfaults = len([k for k in f['GF_Spectrum/sta_0'].keys()
                                   if k.startswith('sub_')])

        self.xyz_all = np.vstack([self.xyz, self.xyz_qa]) if self.xyz_qa is not None else self.xyz

        xyz_t = _rotate(self.xyz)
        h_x = np.diff(np.sort(np.unique(np.round(xyz_t[:, 0], 6))))[0]
        h_y = np.diff(np.sort(np.unique(np.round(xyz_t[:, 1], 6))))[0]
        h_z = np.diff(np.sort(np.unique(np.round(xyz_t[:, 2], 6))))[0]
        self.spacing    = (h_x, h_y, h_z)
        self.model_name = f"{h_x:.1f}m"

        self._dt_orig    = dt_orig; self._tstart     = tstart
        self._n_nodes    = n_nodes; self._n_subfaults = n_subfaults
        self._n_time_gf  = n_time_gf; self._n_time_data = n_time_data
        self._data_grp   = data_grp; self._meta_grp   = meta_grp
        self._qa_grp     = qa_grp

        self._node_cache = {}; self._gf_cache = {}; self._spectrum_cache = {}

        # GF state — OP pipeline
        self._gf_loaded        = False
        self._pair_to_slot     = None
        self._pairs_to_compute = None
        self._nsources_db      = 1
        self._use_pair_to_slot = False
        self._ktree            = None
        self._delta_h          = None
        self._delta_v_src      = None
        self._delta_v_rec      = None
        self._dh_slots         = None
        self._zsrc_slots       = None

        # Legacy GF state
        self.gf_db_pairs = None
        self.node_mapping = None; self.pairs_mapping = None

        # Auto-detect GF info already embedded in the file
        self._try_load_gf_from_file()

        if dt is None:
            self.dt      = dt_orig
            self.time    = np.arange(n_time_data) * dt_orig + tstart
            self.gf_time = np.arange(n_time_gf)  * dt_orig
        else:
            self.dt   = dt
            t_orig    = np.arange(n_time_data) * dt_orig + tstart
            gf_orig   = np.arange(n_time_gf)   * dt_orig
            self.time    = np.arange(t_orig[0],  t_orig[-1],  dt)
            self.gf_time = np.arange(gf_orig[0], gf_orig[-1], dt)
            self._resample_cache = {'time_orig': t_orig, 'gf_time_orig': gf_orig}

        # Precompute vmax for animation colour limits
        self._vmax = {}
        with h5py.File(filename, 'r') as f:
            for dtype, pkey in [('accel','acceleration'),('vel','velocity'),('disp','displacement')]:
                path = f'{data_grp}/{pkey}'
                if path in f:
                    d = f[path][:]
                    e_d = d[0::3,:]; n_d = d[1::3,:]; z_d = d[2::3,:]
                    self._vmax[dtype] = {
                        'e': float(np.abs(e_d).max()),
                        'n': float(np.abs(n_d).max()),
                        'z': float(np.abs(z_d).max()),
                        'resultant': float(np.sqrt(e_d**2+n_d**2+z_d**2).max()),
                    }

        print("=" * 60)
        print(f"FILE : {filename}")
        is_surface = self.is_drm and not np.any(self.internal)
        type_str   = 'SurfaceGrid' if is_surface else ('DRM' if self.is_drm else 'Station')
        print(f"TYPE : {type_str}")
        print(f"NODES: {n_nodes}  |  QA: {'yes' if self.xyz_qa is not None else 'no'}")
        if self.xyz_qa is not None:
            print(f"QA position: {self.xyz_qa[0] * 1000} m")
        print(f"SPACING : {h_x:.1f}m x {h_y:.1f}m x {h_z:.1f}m  |  model: {self.model_name}")
        print(f"TIME    : dt={dt_orig}s  |  steps={n_time_data}  "
              f"|  t=[{tstart:.3f}, {tstart + n_time_data*dt_orig:.3f}]s")
        print(f"GF      : steps={n_time_gf}"
              + (f"  |  nsources={self._nsources_db}" if self._gf_loaded else "  |  not loaded"))
        print(f"INTERNAL: {self.internal.sum()}  |  EXTERNAL: {(~self.internal).sum()}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # GF database — OP pipeline
    # ------------------------------------------------------------------

    def _try_load_gf_from_file(self):
        """Auto-detect and load GF metadata already embedded in the HDF5 file.

        Checks for OP pipeline layout (GF_Database_Info + GF_tdata) first,
        then falls back to legacy layout (Node_Mapping + GF/sta_N).
        """
        with h5py.File(self.filename, 'r') as f:

            # ── OP pipeline: GF_Database_Info ──────────────────────────
            if 'GF_Database_Info/pairs_to_compute' in f:
                grp = f['GF_Database_Info']
                self._pairs_to_compute = grp['pairs_to_compute'][:]
                self._delta_h     = float(grp.attrs['delta_h'])
                self._delta_v_src = float(grp.attrs['delta_v_src'])
                self._delta_v_rec = float(grp.attrs['delta_v_rec'])

                if 'pair_to_slot' in grp:
                    self._pair_to_slot     = grp['pair_to_slot'][:]
                    self._nsources_db      = int(grp.attrs.get('nsources', 1))
                    self._use_pair_to_slot = True
                else:
                    # KDTree fallback
                    dh   = grp['dh_of_pairs'][:]
                    zsrc = grp['zsrc_of_pairs'][:]
                    zrec = grp['zrec_of_pairs'][:]
                    self._dh_slots   = dh
                    self._zsrc_slots = zsrc
                    pts = np.column_stack([dh   / self._delta_h,
                                           zsrc / self._delta_v_src,
                                           zrec / self._delta_v_rec])
                    self._ktree = cKDTree(pts)
                    self._use_pair_to_slot = False

                # Store for plot_calculated_vs_reused
                self.gf_db_pairs     = self._pairs_to_compute
                self.gf_db_dh        = grp['dh_of_pairs'][:]
                self.gf_db_zrec      = grp['zrec_of_pairs'][:]
                self.gf_db_zsrc      = grp['zsrc_of_pairs'][:]
                self.gf_db_delta_h   = self._delta_h
                self.gf_db_delta_v_rec = self._delta_v_rec
                self.gf_db_delta_v_src = self._delta_v_src

                self._gf_loaded = True
                unique = np.unique(self._pairs_to_compute[:, 0])
                mode   = "O(1) pair_to_slot" if self._use_pair_to_slot else "KDTree"
                print(f"  GF DB ({mode}): {len(unique)}/{self._n_nodes} computed "
                      f"({(1-len(unique)/self._n_nodes)*100:.1f}% reduction)")
                return

            # ── Legacy: Node_Mapping ────────────────────────────────────
            if 'Node_Mapping/node_to_pair_mapping' in f:
                self.node_mapping  = f['Node_Mapping/node_to_pair_mapping'][:]
                self.pairs_mapping = f['Node_Mapping/pairs_to_compute'][:]
                print("  GF mapping loaded (legacy Node_Mapping).")

    def load_gf_database(self, h5_path):
        """Merge a GF database (.h5) produced by Stage 0/1 into the .h5drm file.

        Copies GF_Database_Info metadata and GF timeseries (tdata_dict →
        GF_tdata) into the .h5drm so the file becomes self-contained.
        After merging, GF methods are activated automatically.

        Parameters
        ----------
        h5_path : str
            Path to the GF database file produced by Stage 0/1.

        Examples
        --------
        >>> surf = SurfaceData("surface_400m_hollow.h5drm")
        >>> surf.load_gf_database("gf_database_surface_400m_hollow")
        >>> surf.plot_node_gf(node_id=0, subfault=0)
        """
        print(f"Merging GF database: {h5_path} → {self.filename}")

        with h5py.File(h5_path, 'r') as src, h5py.File(self.filename, 'a') as dst:

            # ── GF_Database_Info ──────────────────────────────────────
            if 'GF_Database_Info' in dst:
                del dst['GF_Database_Info']
            grp = dst.create_group('GF_Database_Info')

            for key in ['pairs_to_compute', 'dh_of_pairs', 'dv_of_pairs',
                        'zsrc_of_pairs', 'zrec_of_pairs']:
                if key in src:
                    grp.create_dataset(key, data=src[key][:])

            grp.attrs['delta_h']     = float(src['delta_h'][()])
            grp.attrs['delta_v_src'] = float(src['delta_v_src'][()])
            grp.attrs['delta_v_rec'] = float(src['delta_v_rec'][()])
            grp.attrs['nstations']   = int(src['nstations'][()])
            grp.attrs['nsources']    = int(src['nsources'][()])

            if 'pair_to_slot' in src:
                grp.create_dataset('pair_to_slot', data=src['pair_to_slot'][:])

            # ── GF timeseries: tdata_dict → GF_tdata ─────────────────
            if 'GF_tdata' in dst:
                del dst['GF_tdata']

            if 'tdata_dict' in src:
                src.copy('tdata_dict', dst, name='GF_tdata')
                n_slots = len([k for k in src['tdata_dict'].keys()
                                if k.endswith('_tdata')])
                print(f"  GF timeseries copied: {n_slots} slots")
            else:
                print("  Warning: no tdata_dict found in source file")

        print(f"Done. File updated: {self.filename}")

        # Rebuild GF state from newly written data
        self._gf_cache = {}
        self._gf_loaded = False
        self._pair_to_slot = None
        self._pairs_to_compute = None
        self._ktree = None
        self._try_load_gf_from_file()

        # Update gf_time if needed
        with h5py.File(self.filename, 'r') as f:
            if 'GF_tdata/0_tdata' in f:
                n_time_gf = f['GF_tdata/0_tdata'].shape[0]
                self._n_time_gf = n_time_gf
                self.gf_time    = np.arange(n_time_gf) * self._dt_orig

    def _get_slot(self, node_id, subfault_id):
        """Return GF slot index for (node_id, subfault_id).

        Primary: O(1) flat array via pair_to_slot[node * nsources + subfault].
        Fallback: KDTree lookup for legacy databases.
        """
        if not self._gf_loaded:
            raise RuntimeError(
                "GFs not loaded. Call load_gf_database('file.h5') first.")
        if self._use_pair_to_slot:
            if subfault_id >= self._nsources_db:
                raise ValueError(
                    f"subfault_id={subfault_id} out of range. "
                    f"This file has nsources={self._nsources_db}.")
            flat = node_id * self._nsources_db + subfault_id
            return int(self._pair_to_slot[flat])
        else:
            zrec  = float(self.xyz[node_id][2])
            zsrc  = float(self._zsrc_slots[subfault_id % len(self._zsrc_slots)])
            dh    = float(self._dh_slots[subfault_id  % len(self._dh_slots)])
            q     = np.array([[dh   / self._delta_h,
                                zsrc / self._delta_v_src,
                                zrec / self._delta_v_rec]])
            _, si = self._ktree.query(q)
            return int(si[0])
    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_node_data(self, node_id, data_type='accel'):
        """Return signal array (3, Nt) for a single node.

        Parameters
        ----------
        node_id : int
        data_type : {'accel','vel','disp'}
        """
        key = (node_id, data_type)
        if key not in self._node_cache:
            idx  = 3 * node_id
            path = {'accel': f'{self._data_grp}/acceleration',
                    'vel':   f'{self._data_grp}/velocity',
                    'disp':  f'{self._data_grp}/displacement'}[data_type]
            with h5py.File(self.filename, 'r') as f:
                data = f[path][idx:idx+3, :]
            if hasattr(self, '_window_mask'):
                data = data[:, self._window_mask]
            elif hasattr(self, '_resample_cache'):
                rs = np.zeros((3, len(self.time)))
                for i in range(3):
                    rs[i] = interp1d(self._resample_cache['time_orig'], data[i],
                                     kind='linear', fill_value='extrapolate')(self.time)
                data = rs
            self._node_cache[key] = data
        return self._node_cache[key]

    def get_qa_data(self, data_type='accel'):
        """Return signal array (3, Nt) for the QA station (DRM only)."""
        if self._qa_grp is None:
            raise AttributeError("QA station only available in DRM output files.")
        key = ('qa', data_type)
        if key not in self._node_cache:
            path = {'accel': f'{self._qa_grp}/acceleration',
                    'vel':   f'{self._qa_grp}/velocity',
                    'disp':  f'{self._qa_grp}/displacement'}[data_type]
            with h5py.File(self.filename, 'r') as f:
                data = f[path][:]
            if hasattr(self, '_window_mask'):
                data = data[:, self._window_mask]
            elif hasattr(self, '_resample_cache'):
                rs = np.zeros((3, len(self.time)))
                for i in range(3):
                    rs[i] = interp1d(self._resample_cache['time_orig'], data[i],
                                     kind='linear', fill_value='extrapolate')(self.time)
                data = rs
            self._node_cache[key] = data
        return self._node_cache[key]

    def get_gf(self, node_id, subfault_id, component='z'):
        """Return Green's function time series for a node/subfault pair.

        Supports both OP pipeline (GF_tdata/{slot}_tdata) and legacy
        (GF/sta_N/sub_M/{z,e,n}) formats, detected automatically.

        Parameters
        ----------
        node_id : int
        subfault_id : int
        component : {'z','e','n','tdata'}, default 'z'
        """
        key = (node_id, subfault_id, component)
        if key not in self._gf_cache:

            # ── OP pipeline ────────────────────────────────────────────
            if self._gf_loaded:
                slot = self._get_slot(node_id, subfault_id)
                donor_n, donor_s = self._pairs_to_compute[slot]
                if donor_n != node_id:
                    print(f"  Node {node_id}/sub {subfault_id} → "
                          f"slot {slot} (donor: {donor_n})")
                with h5py.File(self.filename, 'r') as f:
                    tdata_path = f'GF_tdata/{slot}_tdata'
                    if tdata_path not in f:
                        raise KeyError(
                            f"GF not found: {tdata_path}. "
                            "Call load_gf_database() first.")
                    tdata = f[tdata_path][:]   # (Nt, 9)

                comp_map = {'z': 0, 'e': 1, 'n': 2, 'tdata': None}
                if component == 'tdata':
                    self._gf_cache[key] = tdata
                elif component in comp_map:
                    self._gf_cache[key] = tdata[:, comp_map[component]]
                else:
                    raise KeyError(f"Unknown component '{component}'.")

            # ── Legacy Node_Mapping ────────────────────────────────────
            elif self.node_mapping is not None:
                mask  = ((self.node_mapping[:,0]==node_id) &
                         (self.node_mapping[:,1]==subfault_id))
                idx   = np.where(mask)[0]
                if not len(idx):
                    raise KeyError(f"Node {node_id}, subfault {subfault_id} "
                                   "not in mapping")
                ipair = self.node_mapping[idx[0], 2]
                src_n, src_s = self.pairs_mapping[ipair]
                if src_n != node_id:
                    print(f"  Node {node_id}/sub {subfault_id} → donor {src_n}")
                path = f'GF/sta_{src_n}/sub_{src_s}/{component}'
                with h5py.File(self.filename, 'r') as f:
                    if path not in f:
                        raise KeyError(f"GF not found: {path}")
                    self._gf_cache[key] = f[path][:]
            else:
                raise RuntimeError(
                    "No GF data available. Call load_gf_database() first.")

        return self._gf_cache[key]

    def get_spectrum(self, node_id, subfault_id, component='z', part='real'):
        """Return a GF frequency-domain spectrum component (legacy format)."""
        key = (node_id, subfault_id, component, part)
        if key not in self._spectrum_cache:
            if self.node_mapping is not None:
                pi = self.node_mapping[node_id, subfault_id]
                if pi == -1:
                    raise KeyError(f"Node {node_id}, subfault {subfault_id} "
                                   "not computed.")
                src_n, src_s = self.pairs_mapping[pi]
            else:
                src_n, src_s = node_id, subfault_id
            path = (f'GF_Spectrum/sta_{src_n}/sub_{src_s}/'
                    f'spectrum_{component}_{part}')
            with h5py.File(self.filename, 'r') as f:
                self._spectrum_cache[key] = f[path][:]
        return self._spectrum_cache[key]

    def get_surface_snapshot(self, time_idx, component='z', data_type='vel'):
        """Return signal values for all nodes at a single time index.

        Reads directly from HDF5 without caching.

        Parameters
        ----------
        time_idx : int
        component : {'z','e','n'}
        data_type : {'vel','accel','disp'}

        Returns
        -------
        np.ndarray, shape (N,)
        """
        row  = {'e': 0, 'n': 1, 'z': 2}[component.lower()]
        path = {'accel': f'{self._data_grp}/acceleration',
                'vel':   f'{self._data_grp}/velocity',
                'disp':  f'{self._data_grp}/displacement'}[data_type]
        with h5py.File(self.filename, 'r') as f:
            return f[path][row::3, time_idx]

    def clear_cache(self):
        """Release all in-memory cached data."""
        import gc
        self._node_cache.clear(); self._gf_cache.clear()
        self._spectrum_cache.clear(); gc.collect()
        print("Cache cleared.")

    # ------------------------------------------------------------------
    # Windowing / resampling
    # ------------------------------------------------------------------

    def get_window(self, t_start, t_end):
        """Return a time-windowed copy (lazy — no data loaded).

        Parameters
        ----------
        t_start, t_end : float
        """
        new = ShakerMakerData.__new__(ShakerMakerData)
        for a, v in self.__dict__.items(): setattr(new, a, v)
        mask = (self.time >= t_start) & (self.time <= t_end)
        new._window_mask = mask; new._n_time_data = int(mask.sum())
        new.time = self.time[mask]; new.name = f"{self.name} [{t_start}-{t_end}s]"
        new._node_cache = {}; new._gf_cache = {}; new._spectrum_cache = {}
        print(f"Window [{t_start}, {t_end}]s → {new._n_time_data} samples")
        return new

    def resample(self, dt):
        """Return a copy configured for a new time step (lazy).

        Parameters
        ----------
        dt : float
        """
        new = ShakerMakerData.__new__(ShakerMakerData)
        for a, v in self.__dict__.items(): setattr(new, a, v)
        t_orig  = np.arange(self._n_time_data) * self._dt_orig + self._tstart
        gf_orig = np.arange(self._n_time_gf)   * self._dt_orig
        new.dt = dt
        new.time    = np.arange(t_orig[0],  t_orig[-1],  dt)
        new.gf_time = np.arange(gf_orig[0], gf_orig[-1], dt)
        new._resample_cache = {'time_orig': t_orig, 'gf_time_orig': gf_orig}
        new._node_cache = {}; new._gf_cache = {}; new._spectrum_cache = {}
        print(f"Resampled: {len(new.time)} steps at dt={dt}s")
        return new

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_node(self, node_id, data_type):
        if node_id in ('QA','qa') or (isinstance(node_id,int) and node_id>=len(self.xyz)):
            return self.get_qa_data(data_type), 'QA'
        return self.get_node_data(node_id, data_type), f'Node {node_id}'

    @staticmethod
    def _build_cube_faces(xyz_nodes):
        x0,x1 = xyz_nodes[:,0].min(), xyz_nodes[:,0].max()
        y0,y1 = xyz_nodes[:,1].min(), xyz_nodes[:,1].max()
        z0,z1 = xyz_nodes[:,2].min(), xyz_nodes[:,2].max()
        c = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                      [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        faces = [[c[4],c[5],c[6],c[7]],[c[0],c[1],c[5],c[4]],
                 [c[2],c[3],c[7],c[6]],[c[0],c[3],c[7],c[4]],
                 [c[1],c[2],c[6],c[5]]]
        return c, faces, (x0,x1,y0,y1,z0,z1)

    def _label_nodes_on_ax(self, ax, xyz_t, bounds, label_nodes, comp_donors=None):
        x0,x1,y0,y1,z0,z1 = bounds
        xe0,xe1 = xyz_t[:,0].min(), xyz_t[:,0].max()
        ye0,ye1 = xyz_t[:,1].min(), xyz_t[:,1].max()
        ze0,ze1 = xyz_t[:,2].min(), xyz_t[:,2].max()

        def oi(x,y,z,n=2):
            return sum([abs(x-x0)<1e-3 or abs(x-x1)<1e-3,
                        abs(y-y0)<1e-3 or abs(y-y1)<1e-3,
                        abs(z-z0)<1e-3 or abs(z-z1)<1e-3])>=n
        def oe(x,y,z,n=2):
            return sum([abs(x-xe0)<1e-3 or abs(x-xe1)<1e-3,
                        abs(y-ye0)<1e-3 or abs(y-ye1)<1e-3,
                        abs(z-ze0)<1e-3 or abs(z-ze1)<1e-3])>=n

        for i,(x,y,z) in enumerate(xyz_t):
            col = 'darkred' if self.internal[i] else 'darkblue'
            if label_nodes is True:
                ax.text(x,y,z,str(i),fontsize=8,color=col)
            elif label_nodes=='corners':
                if oi(x,y,z,3) or oe(x,y,z,3):
                    ax.text(x,y,z,str(i),fontsize=8,color=col,fontweight='bold')
            elif label_nodes=='corners_edges':
                if oi(x,y,z) or oe(x,y,z):
                    ax.text(x,y,z,str(i),fontsize=9,color=col)
            elif label_nodes=='corners_half':
                xm=(x0+x1)/2; ym=(y0+y1)/2; zm=(z0+z1)/2
                corner = oi(x,y,z,3)
                mid = any([oi(x,y,zm,2) and abs(z-zm)<1e-3,
                           oi(x,ym,z,2) and abs(y-ym)<1e-3,
                           oi(xm,y,z,2) and abs(x-xm)<1e-3])
                if corner or mid:
                    ax.text(x,y,z,str(i),fontsize=9,color=col)
            elif label_nodes=='calculated' and comp_donors is not None:
                if i in comp_donors:
                    ax.text(x,y,z,str(i),fontsize=8,color=col)

    def _collect_node_ids(self, node_id, target_pos):
        if node_id is not None:
            return node_id if isinstance(node_id,(list,np.ndarray)) else [node_id]
        if target_pos is not None:
            dist = np.linalg.norm(self.xyz_all - np.asarray(target_pos), axis=1)
            idx  = int(np.argmin(dist))
            print(f"Nearest node: {idx}  distance={dist[idx]:.6f} km")
            return [idx]
        raise ValueError("Provide node_id or target_pos.")

    def _donor_of_op(self, node_id, subfault_id):
        """Return donor node for OP pipeline."""
        slot = self._get_slot(node_id, subfault_id)
        return int(self._pairs_to_compute[slot, 0])

    # ------------------------------------------------------------------
    # Plotting — single-object methods
    # ------------------------------------------------------------------

    def plot_domain(self, xyz_origin=None, label_nodes=False, show_calculated=False):
        """Plot the 3-D node domain."""
        xyz_t    = _rotate(self.xyz)
        xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None
        if xyz_origin is not None and xyz_qa_t is not None:
            t = np.asarray(xyz_origin) - xyz_qa_t[0]
            xyz_t += t; xyz_qa_t += t

        xyz_int = xyz_t[self.internal]; xyz_ext = xyz_t[~self.internal]
        # SurfaceGrid has no internal nodes — use all for bounding box
        bbox = xyz_int if len(xyz_int) > 0 else xyz_t
        _, faces, bounds = self._build_cube_faces(bbox)
        fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111,projection='3d')

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
            ax.scatter(xyz_t[:,0],xyz_t[:,1],xyz_t[:,2],c='blue',marker='o',s=50,alpha=0.6)

        if xyz_qa_t is not None:
            ax.scatter(xyz_qa_t[:,0],xyz_qa_t[:,1],xyz_qa_t[:,2],c='green',marker='*',
                       s=300,label='QA',zorder=10,edgecolors='black',linewidths=2)
        ax.add_collection3d(Poly3DCollection(faces,alpha=0.15,facecolor='red',
                                             edgecolor='darkred',linewidths=1.5))
        if label_nodes: self._label_nodes_on_ax(ax,xyz_t,bounds,label_nodes,comp_donors)
        ax.set_xlabel("X' (m)"); ax.set_ylabel("Y' (m)"); ax.set_zlabel("Z' (m)")
        ax.legend(); ax.grid(False); plt.tight_layout(); plt.show()
        if xyz_qa_t is not None: print(f"QA position: {xyz_qa_t[0]}")
        return fig, ax

    def plot_node_response(self, node_id=None, target_pos=None, xlim=None, data_type='vel'):
        """Plot time-history for one or more nodes."""
        nids   = self._collect_node_ids(node_id, target_pos)
        ylabel = {'accel':'Acceleration','vel':'Velocity','disp':'Displacement'}[data_type]
        fig = plt.figure(figsize=(8,8))
        for nid in nids:
            data, lbl = self._resolve_node(nid, data_type)
            for k in range(1,4):
                plt.subplot(3,1,k); plt.plot(self.time,data[k-1],linewidth=1,label=lbl)
        for k,comp in enumerate(('Vertical (Z)','East (E)','North (N)'),1):
            ax = plt.subplot(3,1,k)
            ax.set_title(f'{comp} — {ylabel}',fontweight='bold')
            ax.set_xlabel('Time [s]'); ax.set_ylabel('Amplitude')
            ax.grid(True,alpha=0.3); ax.legend(loc='lower left')
            if xlim: ax.set_xlim(xlim)
        plt.tight_layout(); plt.show()

    def plot_node_gf(self, node_id=None, target_pos=None, xlim=None, subfault=0):
        """Plot Green's function time series for one or more nodes."""
        if not self._gf_loaded and self.node_mapping is None:
            print("No GFs available. Call load_gf_database() first."); return
        nids    = self._collect_node_ids(node_id, target_pos)
        sub_ids = subfault if isinstance(subfault,(list,np.ndarray)) else [subfault]
        fig = plt.figure(figsize=(8,10))
        for nid in nids:
            if nid in ('QA','qa'): print("GFs not available for QA."); continue
            for sid in sub_ids:
                lbl = f'N{nid}_S{sid}'
                for k,comp in enumerate(('z','e','n'),1):
                    plt.subplot(3,1,k)
                    plt.plot(self.gf_time,self.get_gf(nid,sid,comp),linewidth=1,label=lbl)
        for k,t in enumerate(('Vertical (Z)','East (E)','North (N)'),1):
            ax = plt.subplot(3,1,k)
            ax.set_title(f'{t} — Green Function',fontweight='bold')
            ax.set_xlabel('Time [s]'); ax.set_ylabel('Amplitude')
            ax.grid(True,alpha=0.3); ax.legend()
            if xlim: ax.set_xlim(xlim)
        plt.tight_layout(); plt.show()

    def plot_node_tensor_gf(self, node_id=None, target_pos=None, xlim=None, subfault=0):
        """Plot the 9-component tensor Green's functions."""
        if not self._gf_loaded:
            print("No GFs. Call load_gf_database() first."); return
        nids    = self._collect_node_ids(node_id, target_pos)
        sub_ids = subfault if isinstance(subfault,(list,np.ndarray)) else [subfault]
        labels  = [f'G_{i+1}{j+1}' for i in range(3) for j in range(3)]
        fig, axes = plt.subplots(3,3,figsize=(10,8))
        for nid in nids:
            if nid in ('QA','qa'): continue
            for sid in sub_ids:
                slot   = self._get_slot(nid, sid)
                donor  = self._pairs_to_compute[slot,0]
                if donor != nid:
                    print(f"Node {nid}/sub {sid} → slot {slot} (donor {donor})")
                with h5py.File(self.filename,'r') as f:
                    tp = f'GF_tdata/{slot}_tdata'
                    if tp not in f: continue
                    tdata = f[tp][:]
                    t0    = f[f'GF_tdata/{slot}_t0'][()] if f'GF_tdata/{slot}_t0' in f else 0.0
                time = np.arange(tdata.shape[0])*self._dt_orig+t0
                lbl  = f'N{nid}_S{sid}'
                for j in range(9):
                    axes[j//3,j%3].plot(time,tdata[:,j],linewidth=0.8,label=lbl)
        for j,lbl in enumerate(labels):
            ax = axes[j//3,j%3]
            ax.set_title(lbl,fontsize=11,fontweight='bold')
            ax.set_xlabel('Time [s]',fontsize=9); ax.set_ylabel('Amplitude',fontsize=9)
            ax.grid(True,alpha=0.3)
            if xlim: ax.set_xlim(xlim)
        axes[0,0].legend(fontsize=8)
        plt.suptitle('Tensor Green Functions',fontsize=14,fontweight='bold')
        plt.tight_layout(); plt.show()

    def plot_node_f_spectrum(self, node_id=None, target_pos=None, xlim=None, subfault=0):
        """Plot Fourier magnitude spectrum (legacy GF_Spectrum format)."""
        nids    = self._collect_node_ids(node_id, target_pos)
        sub_ids = subfault if isinstance(subfault,(list,np.ndarray)) else [subfault]
        fig = plt.figure(figsize=(8,10))
        for nid in nids:
            if nid in ('QA','qa'): continue
            for sid in sub_ids:
                try:
                    mags = [np.sqrt(self.get_spectrum(nid,sid,c,'real')**2+
                                    self.get_spectrum(nid,sid,c,'imag')**2)
                            for c in ('z','e','n')]
                    lbl = f'N{nid}_S{sid}'
                    for k,mag in enumerate(mags,1):
                        plt.subplot(3,1,k); plt.loglog(self.freqs,mag,linewidth=1,label=lbl)
                except KeyError:
                    print(f"  ! No spectrum for node {nid}, subfault {sid}")
        for k,comp in enumerate(('Vertical (Z)','East (E)','North (N)'),1):
            ax = plt.subplot(3,1,k)
            ax.set_title(f'{comp} — Fourier Spectrum',fontweight='bold')
            ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel('Magnitude (log scale)')
            ax.grid(True,alpha=0.3); ax.legend()
            if xlim: ax.set_xlim(xlim)
        plt.tight_layout(); plt.show()

    def plot_node_newmark(self, node_id=None, target_pos=None, xlim=None, data_type='accel'):
        """Plot Newmark response spectra for one or more nodes."""
        if xlim is None: xlim = [0,5]
        nids   = self._collect_node_ids(node_id, target_pos)
        dt     = self.time[1]-self.time[0]
        scale  = 1.0/9.81 if data_type=='accel' else 1.0
        ylabel = 'Sa (g)' if data_type=='accel' else 'Spectral Response'
        fig, axes = plt.subplots(3,1,figsize=(8,10))
        for nid in nids:
            data, lbl = self._resolve_node(nid, data_type)
            specs = [NewmarkSpectrumAnalyzer.compute(data[i]*scale,dt) for i in range(3)]
            T = specs[0]['T']
            for ax,sp in zip(axes,specs):
                ax.plot(T,sp['PSa'],linewidth=2,label=lbl)
        for ax,comp in zip(axes,('Vertical (Z)','X','Y')):
            ax.set_title(f'{comp} — Newmark Spectrum',fontweight='bold')
            ax.set_xlabel('T (s)',fontsize=12); ax.set_ylabel(ylabel,fontsize=12)
            ax.set_xlim(xlim); ax.grid(True,alpha=0.3); ax.legend()
        plt.tight_layout(); plt.show()

    def plot_calculated_vs_reused(self, db_filename=None, xyz_origin=None, label_nodes=False):
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
                            label_nodes=False):
        """Visualise donor–recipient GF connections for a single node."""
        if not self._gf_loaded:
            print("No GFs. Call load_gf_database() first."); return

        comp_donors = set(np.unique(self._pairs_to_compute[:,0]))
        super_donors = set()
        for node in range(len(self.xyz)):
            donor = self._donor_of_op(node, 0)
            if donor != node: super_donors.add(donor)
        solitary = comp_donors - super_donors

        if node_id in super_donors:
            recs = [n for n in range(len(self.xyz))
                    if n!=node_id and self._donor_of_op(n,0)==node_id]
            dtp,rtp = node_id, recs
            print(f"Node {node_id}: SUPER DONOR → {len(recs)} recipients")
        elif node_id in solitary:
            dtp,rtp = node_id, []
            print(f"Node {node_id}: SOLITARY DONOR")
        else:
            dtp = self._donor_of_op(node_id,0); rtp = [node_id]
            print(f"Node {node_id}: RECEIVER ← donor {dtp}")

        xyz_t    = _rotate(self.xyz)
        xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None
        if xyz_origin is not None and xyz_qa_t is not None:
            t = np.asarray(xyz_origin)-xyz_qa_t[0]; xyz_t+=t; xyz_qa_t+=t
        bbox = xyz_t[self.internal] if self.internal.any() else xyz_t
        _,faces,bounds = self._build_cube_faces(bbox)

        fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111,projection='3d')
        ax.scatter(xyz_t[:,0],xyz_t[:,1],xyz_t[:,2],c='blue',s=50,alpha=0.1)
        dp = xyz_t[dtp]
        ax.scatter(*dp,c='red',marker='s',s=100,edgecolors='darkred',linewidths=2,zorder=10,alpha=0.5)
        for rec in rtp:
            rp = xyz_t[rec]
            ax.scatter(*rp,c='orange',marker='o',s=80,edgecolors='darkorange',linewidths=1.5,alpha=0.5)
            ax.plot([dp[0],rp[0]],[dp[1],rp[1]],[dp[2],rp[2]],
                    color='darkorange',linestyle='--',alpha=0.5,linewidth=2)
        if xyz_qa_t is not None:
            ax.scatter(*xyz_qa_t[0],c='green',marker='*',s=300,label='QA',
                       zorder=10,edgecolors='black',linewidths=2)
        ax.add_collection3d(Poly3DCollection(faces,alpha=0.10,facecolor='red',
                                             edgecolor='darkred',linewidths=1.5))
        if label_nodes: self._label_nodes_on_ax(ax,xyz_t,bounds,label_nodes,comp_donors)
        ax.set_xlabel("X' (m)"); ax.set_ylabel("Y' (m)"); ax.set_zlabel("Z' (m)")
        ax.legend(); ax.grid(False); plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # Surface / animation methods  (primarily for SurfaceGrid outputs)
    # ------------------------------------------------------------------

    def plot_surface(self, time=0.0, component='z', data_type='vel',
                     cmap='RdBu_r', figsize=(12,8),
                     elev=30, azim=45, s=20, alpha=0.85):
        """Plot a 3-D scatter snapshot of the domain at a given time."""
        it = int(np.argmin(np.abs(self.time - time)))
        actual_t = self.time[it]
        if component.lower() == 'resultant':
            mag  = np.sqrt(self.get_surface_snapshot(it,'e',data_type)**2 +
                           self.get_surface_snapshot(it,'n',data_type)**2 +
                           self.get_surface_snapshot(it,'z',data_type)**2)
            vmax = self._vmax[data_type]['resultant']; vmin = 0; clbl = 'Resultant'
        else:
            mag  = self.get_surface_snapshot(it, component, data_type)
            vmax = self._vmax[data_type][component.lower()]; vmin = -vmax
            clbl = {'z':'Vertical (Z)','e':'East (E)','n':'North (N)'}[component.lower()]
        x=self.xyz[:,0]*1000; y=self.xyz[:,1]*1000; z=self.xyz[:,2]*1000
        fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111,projection='3d')
        ax.scatter(x,y,z,c='lightgray',s=s,alpha=0.3)
        active = np.abs(mag) >= vmax*0.01
        if active.any():
            sc = ax.scatter(x[active],y[active],z[active],c=mag[active],
                            cmap=cmap,s=s,alpha=alpha,vmin=vmin,vmax=vmax)
            fig.colorbar(sc,ax=ax,shrink=0.5)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.invert_zaxis()
        ax.set_title(f'{self.name} | t={actual_t:.3f}s | {clbl}',fontweight='bold')
        ax.view_init(elev=elev,azim=azim)
        plt.tight_layout(); plt.show()

    def create_animation(self, time_start=0.0, time_end=None, n_frames=50,
                         component='z', data_type='vel', cmap='RdBu_r',
                         figsize=(12,8), dpi=100, fps=10,
                         elev=30, azim=45, s=20, alpha=0.85,
                         output_dir='animation', output_video='animation.mp4'):
        """Create a 3-D scatter animation of the full domain."""
        import subprocess
        os.makedirs(output_dir, exist_ok=True)
        if time_end is None: time_end = self.time[-1]
        if component.lower()=='resultant':
            vmax=self._vmax[data_type]['resultant']; vmin=0
        else:
            vmax=self._vmax[data_type][component.lower()]; vmin=-vmax
        x=self.xyz[:,0]*1000; y=self.xyz[:,1]*1000; z=self.xyz[:,2]*1000
        for i,t in enumerate(np.linspace(time_start,time_end,n_frames)):
            it = int(np.argmin(np.abs(self.time-t)))
            if component.lower()=='resultant':
                mag = np.sqrt(self.get_surface_snapshot(it,'e',data_type)**2+
                              self.get_surface_snapshot(it,'n',data_type)**2+
                              self.get_surface_snapshot(it,'z',data_type)**2)
            else:
                mag = self.get_surface_snapshot(it,component,data_type)
            fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111,projection='3d')
            ax.scatter(x,y,z,c='lightgray',s=s,alpha=0.3)
            active = np.abs(mag)>=vmax*0.01
            if active.any():
                ax.scatter(x[active],y[active],z[active],c=mag[active],
                           cmap=cmap,s=s,alpha=alpha,vmin=vmin,vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=vmin,vmax=vmax))
            sm.set_array([]); fig.colorbar(sm,ax=ax,shrink=0.5)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            ax.set_title(f't = {self.time[it]:.3f} s',fontsize=14,fontweight='bold')
            ax.view_init(elev=elev,azim=azim)
            plt.tight_layout(); plt.savefig(f'{output_dir}/frame_{i:03d}.png',dpi=dpi); plt.close()
            print(f'Frame {i+1}/{n_frames}')
        try:
            subprocess.run(['ffmpeg','-y','-framerate',str(fps),
                            '-i',f'{output_dir}/frame_%03d.png',
                            '-c:v','libx264','-pix_fmt','yuv420p',
                            '-crf','18',output_video],check=True,capture_output=True)
            print(f'Video saved: {output_video}')
        except Exception as e:
            print(f'ffmpeg error — frames in {output_dir}: {e}')

    def create_animation_plane(self, plane='xy', plane_value=0.0,
                                time_start=0.0, time_end=None, n_frames=50,
                                component='z', data_type='vel', cmap='RdBu_r',
                                figsize=(12,8), dpi=100, fps=10,
                                elev=30, azim=45, s=50, alpha=0.85,
                                output_dir='animation_plane',
                                output_video='animation_plane.mp4',
                                vmax_from_range=False):
        """Create a 3-D animation of a planar slice through the domain."""
        import subprocess
        os.makedirs(output_dir, exist_ok=True)
        if time_end is None: time_end = self.time[-1]
        x=self.xyz[:,0]*1000; y=self.xyz[:,1]*1000; z=self.xyz[:,2]*1000
        tol = self.spacing[0]*0.1 if self.spacing[0]>0 else 1.0
        if plane.lower()=='xy':
            pmask = np.abs(z-plane_value)<tol; tpl = f'Z = {plane_value:.1f} m'
        elif plane.lower()=='xz':
            pmask = np.abs(y-plane_value)<tol; tpl = f'Y = {plane_value:.1f} m'
        elif plane.lower()=='yz':
            pmask = np.abs(x-plane_value)<tol; tpl = f'X = {plane_value:.1f} m'
        else:
            raise ValueError("plane must be 'xy','xz','yz'")
        if not pmask.any():
            print(f'No nodes found for {tpl}'); return
        pidx = np.where(pmask)[0]
        xp=x[pmask]; yp=y[pmask]; zp=z[pmask]
        if vmax_from_range:
            i0=int(np.argmin(np.abs(self.time-time_start)))
            i1=int(np.argmin(np.abs(self.time-time_end)))
            path={'accel':f'{self._data_grp}/acceleration',
                  'vel':f'{self._data_grp}/velocity',
                  'disp':f'{self._data_grp}/displacement'}[data_type]
            with h5py.File(self.filename,'r') as f:
                d = f[path][:,i0:i1+1]
            if component.lower()=='resultant':
                e=d[0::3,:][pidx]; n=d[1::3,:][pidx]; zc=d[2::3,:][pidx]
                vmax=float(np.sqrt(e**2+n**2+zc**2).max()); vmin=0
            else:
                row={'e':0,'n':1,'z':2}[component.lower()]
                vmax=float(np.abs(d[row::3,:][pidx]).max()); vmin=-vmax
        else:
            if component.lower()=='resultant':
                vmax=self._vmax[data_type]['resultant']; vmin=0
            else:
                vmax=self._vmax[data_type][component.lower()]; vmin=-vmax
        for i,t in enumerate(np.linspace(time_start,time_end,n_frames)):
            it = int(np.argmin(np.abs(self.time-t)))
            if component.lower()=='resultant':
                mag = np.sqrt(self.get_surface_snapshot(it,'e',data_type)[pidx]**2+
                              self.get_surface_snapshot(it,'n',data_type)[pidx]**2+
                              self.get_surface_snapshot(it,'z',data_type)[pidx]**2)
            else:
                mag = self.get_surface_snapshot(it,component,data_type)[pidx]
            fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111,projection='3d')
            ax.scatter(x,y,z,c='lightgray',s=5,alpha=0.05)
            active = np.abs(mag)>=vmax*0.01
            if active.any():
                ax.scatter(xp[active],yp[active],zp[active],c=mag[active],
                           cmap=cmap,s=s,alpha=alpha,vmin=vmin,vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=vmin,vmax=vmax))
            sm.set_array([]); fig.colorbar(sm,ax=ax,shrink=0.5)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.invert_xaxis(); ax.invert_yaxis(); ax.invert_zaxis()
            ax.set_title(f'{tpl} | t = {self.time[it]:.3f} s',fontsize=14,fontweight='bold')
            ax.view_init(elev=elev,azim=azim)
            plt.tight_layout(); plt.savefig(f'{output_dir}/frame_{i:03d}.png',dpi=dpi); plt.close()
            print(f'Frame {i+1}/{n_frames}')
        try:
            subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),
                            '-i',f'{output_dir}/frame_%03d.png',
                            '-c:v','libx264','-pix_fmt','yuv420p','-crf','18',output_video],
                           check=True)
            print(f'Video saved: {output_video}')
        except Exception as e:
            print(f'ffmpeg error — frames in {output_dir}: {e}')


# ---------------------------------------------------------------------------
# Semantic aliases
# ---------------------------------------------------------------------------
DRMData     = ShakerMakerData   # DRMBox / PointCloudDRMReceiver
SurfaceData = ShakerMakerData   # SurfaceGrid