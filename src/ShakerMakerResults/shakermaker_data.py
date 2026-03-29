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

"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import shutil

from .newmark import NewmarkSpectrumAnalyzer

from .utils import _rotate




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
        # h_x = np.diff(np.sort(np.unique(np.round(xyz_t[:, 0], 6))))[0]
        # h_y = np.diff(np.sort(np.unique(np.round(xyz_t[:, 1], 6))))[0]
        # h_z = np.diff(np.sort(np.unique(np.round(xyz_t[:, 2], 6))))[0]
        def _spacing(arr): d = np.diff(np.sort(np.unique(np.round(arr, 6)))); return float(d[0]) if len(d) > 0 else 0.0
        h_x = _spacing(xyz_t[:, 0])
        h_y = _spacing(xyz_t[:, 1])
        h_z = _spacing(xyz_t[:, 2])
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

        # ------------------------------------------------------------------
        # RAM awareness — compute bytes per node and set large-file flag.
        # Methods that need all-node data use self._large_file to decide
        # between fast (pre-load RAM) and safe (chunk HDF5) modes.
        # ------------------------------------------------------------------
        import psutil as _psutil
        self._bytes_per_node  = int(3 * n_time_data * 8)   # 3 components, float64
        _total_data_bytes     = self._bytes_per_node * n_nodes * 3  # vel+accel+disp
        _mem_available        = _psutil.virtual_memory().available
        self._large_file      = _total_data_bytes > _mem_available * 0.5

        # ------------------------------------------------------------------
        # _vmax — lazy loading with sidecar cache.
        # On first use (plot_surface, create_animation, etc.) _compute_vmax()
        # is called, computes by chunks, stores in self._vmax, and writes a
        # small JSON sidecar next to the HDF5 file so future sessions load
        # it instantly without touching the data.
        # ------------------------------------------------------------------
        self._vmax             = None   # computed on demand
        self._vmax_cache_path  = filename + '.vmax.json'
        self._data_grp_for_vmax = data_grp

        # Try loading from sidecar cache — instantaneous
        import json as _json
        if os.path.exists(self._vmax_cache_path):
            try:
                with open(self._vmax_cache_path, 'r') as _cf:
                    self._vmax = _json.load(_cf)
                print(f"  vmax cache loaded from sidecar.")
            except Exception:
                self._vmax = None   # corrupted cache — recompute on demand
        sep = '--' * 50
        is_surface = self.is_drm and not np.any(self.internal)
        type_str   = 'SurfaceGrid' if is_surface else ('DRM' if self.is_drm else 'Station')

        xyz_t_print = _rotate(self.xyz)
        Lx = xyz_t_print[:,0].max() - xyz_t_print[:,0].min()
        Ly = xyz_t_print[:,1].max() - xyz_t_print[:,1].min()
        Lz = xyz_t_print[:,2].max() - xyz_t_print[:,2].min()

        print(sep)
        print(f"ShakerMakerData  :  {filename}")
        print(f"  Type     : {type_str}")
        print(f"  Model    : {self.model_name}  |  Spacing: {h_x:.1f}m x {h_y:.1f}m x {h_z:.1f}m")
        print(f"  Domain   : Lx={Lx:.1f}m  Ly={Ly:.1f}m  Lz={Lz:.1f}m")
        print(f"  Nodes    : {n_nodes}  |  Internal: {self.internal.sum()}  |  External: {(~self.internal).sum()}")
        print(f"  QA       : {'yes  ->  ' + str(self.xyz_qa[0] * 1000) + ' m' if self.xyz_qa is not None else 'no'}")
        print(f"  Time     : dt={dt_orig}s  |  steps={n_time_data}  |  t=[{tstart:.3f}, {tstart + n_time_data*dt_orig:.3f}]s")
        print(f"  GF       : steps={n_time_gf}" + (f"  |  nsources={self._nsources_db}" if self._gf_loaded else "  |  not loaded"))
        with h5py.File(filename, 'r') as f:
            if 'DRM_Metadata/program_used' in f:
                _ver = f['DRM_Metadata/program_used'][()].decode()
                _dat = f['DRM_Metadata/created_on'][()].decode()
                print(f"  Version  : {_ver}  |  {_dat}")
        import psutil
        mem = psutil.virtual_memory()
        print(f"  RAM      : {mem.used/1e9:.1f} GB used  |  "
              f"{mem.available/1e9:.1f} GB free  |  {mem.percent:.1f}%")
        with h5py.File(filename, 'r') as f:
            total_size = 0
            print(f"  File size:")
            for key in f[data_grp].keys():
                ds = f[f'{data_grp}/{key}']
                if hasattr(ds, 'shape') and len(ds.shape) > 1:
                    size_gb = ds.nbytes / 1e9
                    total_size += size_gb
                    print(f"    {key:<20} {ds.shape}  {size_gb:.2f} GB")
            print(f"    {'TOTAL':<20}              {total_size:.2f} GB")

        if self._large_file:
            print(f"  WARNING  : File data ({_total_data_bytes/1e9:.1f} GB) exceeds "
                  f"50% of available RAM ({_mem_available/1e9:.1f} GB). "
                  f"Surface methods will use safe chunk mode automatically.")
        print(sep + '\n')

    # ------------------------------------------------------------------
    # GF database — OP pipeline
    # ------------------------------------------------------------------

    def _try_load_gf_from_file(self):
        """Auto-detect and load GF metadata already embedded in the HDF5 file.

        Checks for OP pipeline layout (GF_Database_Info + GF_tdata) first,
        then falls back to legacy layout (Node_Mapping + GF/sta_N).
        """
        with h5py.File(self.filename, 'r') as f:

            #  OP pipeline: GF_Database_Info 
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

            #  Legacy: Node_Mapping 
            if 'Node_Mapping/node_to_pair_mapping' in f:
                self.node_mapping  = f['Node_Mapping/node_to_pair_mapping'][:]
                self.pairs_mapping = f['Node_Mapping/pairs_to_compute'][:]
                print("  GF mapping loaded (legacy Node_Mapping).")

    def load_gf_database(self, h5_path):
        """Load GF database (.h5) produced by Stage 0/1 into memory.

        Reads GF_Database_Info metadata and GF timeseries directly from
        the source file without modifying any file on disk.

        Parameters
        ----------
        h5_path : str
            Path to the GF database file produced by Stage 0/1.
        """
        print(f"Loading GF database: {h5_path}")
        self._gf_h5_path = h5_path

        with h5py.File(h5_path, 'r') as f:
            self._pairs_to_compute = f['pairs_to_compute'][:]
            self._delta_h          = float(f['delta_h'][()])
            self._delta_v_src      = float(f['delta_v_src'][()])
            self._delta_v_rec      = float(f['delta_v_rec'][()])
            self._nsources_db      = int(f['nsources'][()])

            if 'pair_to_slot' in f:
                self._pair_to_slot     = f['pair_to_slot'][:]
                self._use_pair_to_slot = True
            else:
                dh   = f['dh_of_pairs'][:]
                zsrc = f['zsrc_of_pairs'][:]
                zrec = f['zrec_of_pairs'][:]
                self._dh_slots   = dh
                self._zsrc_slots = zsrc
                pts = np.column_stack([dh   / self._delta_h,
                                       zsrc / self._delta_v_src,
                                       zrec / self._delta_v_rec])
                self._ktree = cKDTree(pts)
                self._use_pair_to_slot = False

            self.gf_db_pairs       = self._pairs_to_compute
            self.gf_db_dh          = f['dh_of_pairs'][:]
            self.gf_db_zrec        = f['zrec_of_pairs'][:]
            self.gf_db_zsrc        = f['zsrc_of_pairs'][:]
            self.gf_db_delta_h     = self._delta_h
            self.gf_db_delta_v_rec = self._delta_v_rec
            self.gf_db_delta_v_src = self._delta_v_src

            n_slots = len([k for k in f['tdata_dict'].keys()
                           if k.endswith('_tdata')])

        self._gf_cache  = {}
        self._gf_loaded = True

        n_time_gf = 0
        with h5py.File(h5_path, 'r') as f:
            if 'tdata_dict/0_tdata' in f:
                n_time_gf = f['tdata_dict/0_tdata'].shape[0]
        self._n_time_gf = n_time_gf
        self.gf_time    = np.arange(n_time_gf) * self._dt_orig

        unique = np.unique(self._pairs_to_compute[:, 0])
        mode   = "O(1) pair_to_slot" if self._use_pair_to_slot else "KDTree"
        print(f"  GF DB ({mode}): {n_slots} slots  |  {len(unique)}/{self._n_nodes} computed "
              f"({(1-len(unique)/self._n_nodes)*100:.1f}% reduction)")

        with h5py.File(h5_path, 'r') as f:
            total_size = 0
            # Detect available data types from first slot
            slot_keys = [k for k in f['tdata_dict'].keys() if not k.endswith('_t0')]
            data_keys = sorted(set(k.split('_', 1)[1] for k in slot_keys))
            print(f"  GF file contents:")
            print(f"    slots          : {n_slots}")
            print(f"    data per slot  : {data_keys}")
            for dk in data_keys:
                example = f[f'tdata_dict/0_{dk}']
                size_total = example.nbytes * n_slots / 1e9
                total_size += size_total
                print(f"    {dk:<20} shape={example.shape}  "
                      f"dtype={example.dtype}  total={size_total:.2f} GB")
            print(f"    {'TOTAL':<20}              {total_size:.2f} GB")

        import psutil
        mem = psutil.virtual_memory()
        print(f"  RAM      : {mem.used/1e9:.1f} GB used  |  "
              f"{mem.available/1e9:.1f} GB free  |  {mem.percent:.1f}%")
        print(f"Done.")



    def _compute_vmax(self):
        """Compute and cache vmax for animation colour limits.

        Reads the HDF5 file in chunks to avoid loading everything into RAM.
        Results are stored in ``self._vmax`` (in-session RAM cache) and
        written to a small JSON sidecar file next to the HDF5 so that
        future sessions load it instantly without touching the data.

        Called automatically by ``plot_surface``, ``create_animation``, and
        ``create_animation_plane`` the first time they need colour limits.
        After the first call in a session, ``self._vmax`` is already set and
        this method is never called again.
        """
        import json
        data_grp    = self._data_grp_for_vmax
        _chunk_rows = 120   # 120 * n_time_data * 8 bytes per chunk
        print(f"  Computing vmax (chunk mode, {_chunk_rows} rows/chunk)...")
        vmax = {}
        with h5py.File(self.filename, 'r') as f:
            for dtype, pkey in [('accel', 'acceleration'),
                                 ('vel',   'velocity'),
                                 ('disp',  'displacement')]:
                path = f'{data_grp}/{pkey}'
                if path not in f:
                    continue
                ds     = f[path]
                n_rows = ds.shape[0]
                e_max = n_max = z_max = r_max = 0.0
                for _s in range(0, n_rows, _chunk_rows):
                    _e  = min(_s + _chunk_rows, n_rows)
                    d   = ds[_s:_e, :]
                    e_d = d[0::3, :]; n_d = d[1::3, :]; z_d = d[2::3, :]
                    e_max = max(e_max, float(np.abs(e_d).max()))
                    n_max = max(n_max, float(np.abs(n_d).max()))
                    z_max = max(z_max, float(np.abs(z_d).max()))
                    r_max = max(r_max,
                                float(np.sqrt(e_d**2 + n_d**2 + z_d**2).max()))
                vmax[dtype] = {
                    'e': e_max, 'n': n_max,
                    'z': z_max, 'resultant': r_max,
                }

        self._vmax = vmax

        # Write sidecar cache
        try:
            with open(self._vmax_cache_path, 'w') as cf:
                json.dump(vmax, cf)
            print(f"  vmax cached to: {self._vmax_cache_path}")
        except Exception as e:
            print(f"  vmax cache write failed (read-only filesystem?): {e}")

    def _get_slot(self, node_id, subfault_id):
        """Return GF slot index for (node_id, subfault_id).

        Primary: O(1) flat array via pair_to_slot[node * nsources + subfault].
        Fallback: KDTree lookup for legacy databases.
        """
        if not self._gf_loaded:
            raise RuntimeError(
                "GFs not loaded. Call load_gf_database('file.h5') first.")
        
        # Convertir 'QA' al índice real
        if node_id in ('QA', 'qa'):
            node_id = self._n_nodes  # QA está en índice nstations
        
        if self._use_pair_to_slot:
            if subfault_id >= self._nsources_db:
                raise ValueError(
                    f"subfault_id={subfault_id} out of range. "
                    f"This file has nsources={self._nsources_db}.")
            flat = node_id * self._nsources_db + subfault_id
            return int(self._pair_to_slot[flat])
        else:
            # KDTree fallback - para QA usar xyz_qa
            if node_id == self._n_nodes and self.xyz_qa is not None:
                zrec = float(self.xyz_qa[0][2])
            else:
                zrec = float(self.xyz[node_id][2])
            zsrc = float(self._zsrc_slots[subfault_id % len(self._zsrc_slots)])
            dh   = float(self._dh_slots[subfault_id % len(self._dh_slots)])
            q    = np.array([[dh   / self._delta_h,
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
            # data[0]=e, data[1]=n, data[2]=z → reordenar a [z, e, n]
            data = data[[2, 0, 1], :]
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
            data = data[[2, 0, 1], :]
            self._node_cache[key] = data
        return self._node_cache[key]


    def get_gf(self, node_id, subfault_id, component='z'):
        """Return Green's function time series for a node/subfault pair."""
        
        # Normalizar node_id para QA
        node_id_key = node_id
        if node_id in ('QA', 'qa'):
            node_id_key = 'QA'
            node_id_num = self._n_nodes
        else:
            node_id_num = node_id
        
        key = (node_id_key, subfault_id, component)
        if key not in self._gf_cache:

            # OP pipeline
            if self._gf_loaded:
                slot = self._get_slot(node_id_num, subfault_id)
                donor_n, donor_s = self._pairs_to_compute[slot]
                if donor_n != node_id_num:
                    print(f"  Node {node_id_key}/sub {subfault_id} → "
                          f"slot {slot} (donor: {donor_n})")
                with h5py.File(self._gf_h5_path, 'r') as f:
                    tdata_path = f'tdata_dict/{slot}_tdata'
                    if tdata_path not in f:
                        raise KeyError(
                            f"GF not found: {tdata_path}. "
                            "Call load_gf_database() first.")
                    if component == 'tdata':
                        self._gf_cache[key] = f[tdata_path][:]
                    elif component in ('z', 'e', 'n'):
                        comp_path = f'tdata_dict/{slot}_{component}'
                        if comp_path in f:
                            self._gf_cache[key] = f[comp_path][:]
                        else:
                            comp_map = {'z': 0, 'e': 1, 'n': 2}
                            self._gf_cache[key] = f[tdata_path][:, comp_map[component]]
                    else:
                        raise KeyError(f"Unknown component '{component}'.")

            # Legacy Node_Mapping
            elif self.node_mapping is not None:
                if node_id_key == 'QA':
                    raise KeyError("QA GFs not available in legacy format.")
                mask = ((self.node_mapping[:, 0] == node_id_num) &
                        (self.node_mapping[:, 1] == subfault_id))
                idx = np.where(mask)[0]
                if not len(idx):
                    raise KeyError(f"Node {node_id_num}, subfault {subfault_id} "
                                   "not in mapping")
                ipair = self.node_mapping[idx[0], 2]
                src_n, src_s = self.pairs_mapping[ipair]
                if src_n != node_id_num:
                    print(f"  Node {node_id_num}/sub {subfault_id} → donor {src_n}")
                path = f'GF/sta_{src_n}/sub_{src_s}/{component}'
                with h5py.File(self.filename, 'r') as f:
                    if path not in f:
                        raise KeyError(f"GF not found: {path}")
                    self._gf_cache[key] = f[path][:]
            else:
                raise RuntimeError(
                    "No GF data available. Call load_gf_database() first.")

        return self._gf_cache[key]



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
            Target time step [s].
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

        sep = '--' * 50
        print(sep)
        print("Resample")
        print(f"  dt       :  {self._dt_orig}s  →  {dt}s")
        print(f"  Steps    :  {self._n_time_data}  →  {len(new.time)}")
        print(f"  Duration :  {t_orig[0]:.3f}s  —  {t_orig[-1]:.3f}s  (unchanged)")
        print(sep + '\n')

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

    # def _collect_node_ids(self, node_id, target_pos):
    #     if node_id is not None:
    #         if isinstance(node_id, (list, np.ndarray)):
    #             return node_id
    #         # Manejar 'QA' como caso especial
    #         if node_id in ('QA', 'qa'):
    #             return ['QA']
    #         return [node_id]
    #     if target_pos is not None:
    #         dist = np.linalg.norm(self.xyz_all - np.asarray(target_pos), axis=1)
    #         idx = int(np.argmin(dist))
    #         print(f"Nearest node: {idx}  distance={dist[idx]:.6f} km")
    #         return [idx]
    #     raise ValueError("Provide node_id or target_pos.")

    def _collect_node_ids(self, node_id=None, target_pos=None, print_info=True):
        """Resolve node IDs from node_id or target_pos and optionally print info.
        
        Parameters
        ----------
        node_id : int, str, list, or None
        target_pos : array-like (3,) or None
        print_info : bool, default True
            Print node info (position, QA match, etc.)
        
        Returns
        -------
        list of int or 'QA'
            Resolved node IDs
        """
        nids = []
        
        if node_id is not None:
            if isinstance(node_id, (list, np.ndarray)):
                nids = list(node_id)
            elif node_id in ('QA', 'qa'):
                nids = ['QA']
            else:
                nids = [node_id]
        elif target_pos is not None:
            target = np.asarray(target_pos)
            dist = np.linalg.norm(self.xyz_all - target, axis=1)
            idx = int(np.argmin(dist))
            # Verificar si es el QA
            if self.xyz_qa is not None and idx == len(self.xyz):
                nids = ['QA']
            else:
                nids = [idx]
        else:
            raise ValueError("Provide node_id or target_pos.")
        
        if print_info:
            sep = '-' * 50
            print(sep)
            print("NODE INFO")
            for nid in nids:
                if nid in ('QA', 'qa'):
                    pos = self.xyz_qa[0] if self.xyz_qa is not None else None
                    if pos is not None:
                        print(f"  {'QA':<8} │ pos = [{pos[0]*1000:>10.2f}, {pos[1]*1000:>10.2f}, {pos[2]*1000:>10.2f}] m")
                    else:
                        print(f"  {'QA':<8} │ position not available")
                else:
                    pos = self.xyz[nid]
                    is_internal = self.internal[nid]
                    node_type = "internal" if is_internal else "external"
                    # Verificar si coincide con QA
                    qa_match = ""
                    if self.xyz_qa is not None:
                        dist_to_qa = np.linalg.norm(pos - self.xyz_qa[0])
                        if dist_to_qa < 1e-6:
                            qa_match = "  ★ COINCIDES WITH QA"
                    print(f"  N{nid:<6} │ pos = [{pos[0]*1000:>10.2f}, {pos[1]*1000:>10.2f}, {pos[2]*1000:>10.2f}] m │ {node_type}{qa_match}")
            
            # Si se dio target_pos, mostrar distancia
            if target_pos is not None:
                target = np.asarray(target_pos)
                for nid in nids:
                    if nid in ('QA', 'qa'):
                        pos = self.xyz_qa[0]
                    else:
                        pos = self.xyz[nid]
                    dist = np.linalg.norm(pos - target) * 1000  # a metros
                    print(f"  Target   │ pos = [{target[0]*1000:>10.2f}, {target[1]*1000:>10.2f}, {target[2]*1000:>10.2f}] m │ dist = {dist:.2f} m")
            print(sep)
        
        return nids


    def _donor_of_op(self, node_id, subfault_id):
        """Return donor node for OP pipeline."""
        # Convertir 'QA' al índice numérico
        if node_id in ('QA', 'qa'):
            node_id = self._n_nodes
        slot = self._get_slot(node_id, subfault_id)
        return int(self._pairs_to_compute[slot, 0])


    # ------------------------------------------------------------------
    # Interpolation fucntions 
    # ------------------------------------------------------------------


    def _interpolate_to_grid(self, x, y, z, mag, resolution=300, method='linear'):
        """Interpolate scattered node data onto a regular 2-D grid.

        Automatically detects the active plane (XY, XZ, or YZ) by finding
        which axis has no variation. Returns the two active coordinate arrays,
        the interpolated grid, and axis labels.

        Parameters
        ----------
        x, y, z : np.ndarray, shape (N,)
            Node coordinates in metres (already rotated).
        mag : np.ndarray, shape (N,)
            Scalar field to interpolate (velocity, acceleration, etc.).
        resolution : int, default ``300``
            Number of grid points along each axis.
        method : {'linear', 'cubic', 'nearest'}, default ``'linear'``

        Returns
        -------
        A, B : np.ndarray, shape (resolution, resolution)
            Meshgrid of the two active axes.
        Zi : np.ndarray, shape (resolution, resolution)
            Interpolated field values.
        albl, blbl : str
            Axis labels for the two active axes.
        """
        from scipy.interpolate import griddata

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        z_range = z.max() - z.min()

        if x_range < 1e-3:
            a, b, albl, blbl = y, z, 'Y (m)', 'Z (m)'
        elif y_range < 1e-3:
            a, b, albl, blbl = x, z, 'X (m)', 'Z (m)'
        else:
            a, b, albl, blbl = x, y, 'X (m)', 'Y (m)'

        ai = np.linspace(a.min(), a.max(), resolution)
        bi = np.linspace(b.min(), b.max(), resolution)
        Ai, Bi = np.meshgrid(ai, bi)
        Zi = griddata((a, b), mag, (Ai, Bi), method=method)
        return Ai, Bi, Zi, albl, blbl



    # ------------------------------------------------------------------
    # Plotting — single-object methods
    # ------------------------------------------------------------------

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
                     figsize=(8, 10)):
        """Plot Green's function time series for one or more nodes."""
        if not self._gf_loaded and self.node_mapping is None:
            print("No GFs available. Call load_gf_database() first.")
            return
        nids = self._collect_node_ids(node_id, target_pos)
        sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
        fig = plt.figure(figsize=figsize)
        
        for nid in nids:
            # Convertir 'QA' al índice numérico para _get_slot
            if nid in ('QA', 'qa'):
                nid_num = self._n_nodes
                nid_label = 'QA'
            else:
                nid_num = nid
                nid_label = f'N{nid}'
            
            for sid in sub_ids:
                slot = self._get_slot(nid_num, sid)
                with h5py.File(self._gf_h5_path, 'r') as f:
                    t0_path = f'tdata_dict/{slot}_t0'
                    t0 = f[t0_path][()] if t0_path in f else 0.0
                lbl = f'{nid_label}_S{sid}'
                for k, comp in enumerate(('z', 'e', 'n'), 1):
                    gf_data = self.get_gf(nid, sid, comp)
                    time = np.arange(len(gf_data)) * self._dt_orig + t0
                    plt.subplot(3, 1, k)
                    plt.plot(time, gf_data, linewidth=1, label=lbl)
        
        for k, t in enumerate(('Vertical (Z)', 'East (E)', 'North (N)'), 1):
            ax = plt.subplot(3, 1, k)
            ax.set_title(f'{t} — Green Function', fontweight='bold')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            if xlim:
                ax.set_xlim(xlim)
        plt.tight_layout()
        plt.show()


    
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
        nids = self._collect_node_ids(node_id, target_pos)
        sub_ids = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
        labels = [f'G_{i+1}{j+1}' for i in range(3) for j in range(3)]
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        for nid in nids:
            # Convertir 'QA' al índice numérico
            if nid in ('QA', 'qa'):
                nid_num = self._n_nodes
                nid_label = 'QA'
            else:
                nid_num = nid
                nid_label = f'N{nid}'
            
            for sid in sub_ids:
                slot = self._get_slot(nid_num, sid)
                donor = self._pairs_to_compute[slot, 0]
                if donor != nid_num:
                    print(f"Node {nid_label}/sub {sid} → slot {slot} (donor {donor})")
                with h5py.File(self._gf_h5_path, 'r') as f:
                    tp = f'tdata_dict/{slot}_tdata'
                    if tp not in f:
                        continue
                    tdata = f[tp][:]
                    t0_path = f'tdata_dict/{slot}_t0'
                    t0 = f[t0_path][()] if t0_path in f else 0.0
                time = np.arange(tdata.shape[0]) * self._dt_orig + t0
                lbl = f'{nid_label}_S{sid}'
                for j in range(9):
                    axes[j // 3, j % 3].plot(time, tdata[:, j], linewidth=0.8, label=lbl)
        
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

        # Convertir 'QA' al índice numérico
        if node_id in ('QA', 'qa'):
            node_id_num = self._n_nodes
            node_id_label = 'QA'
        else:
            node_id_num = node_id
            node_id_label = str(node_id)

        # Classification
        comp_donors = set(np.unique(self._pairs_to_compute[:, 0]))
        super_donors = set()
        # Incluir QA en el análisis
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

        # Geometry - incluir QA en xyz_t
        xyz_t = _rotate(self.xyz)
        xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None
        
        # Crear array completo incluyendo QA
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

        # Donor point - usar xyz_all_t para incluir QA
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



    # ------------------------------------------------------------------
    # Surface / animation methods  (primarily for SurfaceGrid outputs)
    # ------------------------------------------------------------------
    
    def plot_surface(self, 
                    time=0.0, 
                    component='z', 
                    data_type='vel',
                    cmap='RdBu_r', 
                    figsize=(12,8),
                    elev=30, azim=-60, s=20, alpha=0.85,
                    axis_equal=False,
                    interpolate=False,
                    interp_method='linear',
                    interp_resolution=300):
        """Plot a 3-D scatter snapshot of the domain at a given time."""
        # Ensure vmax is computed
        if self._vmax is None:
            self._compute_vmax()
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

        xyz_t = _rotate(self.xyz)
        x=xyz_t[:,0]; y=xyz_t[:,1]; z=xyz_t[:,2]

        fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='lightgray', s=s, alpha=0.05)

        if interpolate:
            Ai, Bi, Zi, albl, blbl = self._interpolate_to_grid(
                x, y, z, mag, resolution=interp_resolution, method=interp_method)
            # Flatten interpolated grid back to scatter on 3D axes
            Zi_flat = Zi.ravel()
            valid   = ~np.isnan(Zi_flat)
            # Reconstruct third coordinate (constant plane value)
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()
            if x_range < 1e-3:        # YZ plane
                xs = np.full(Zi_flat.shape, x.mean())
                ys = Ai.ravel(); zs = Bi.ravel()
            elif y_range < 1e-3:      # XZ plane
                xs = Ai.ravel(); ys = np.full(Zi_flat.shape, y.mean()); zs = Bi.ravel()
            else:                     # XY plane
                xs = Ai.ravel(); ys = Bi.ravel(); zs = np.full(Zi_flat.shape, z.mean())
            sc = ax.scatter(xs[valid], ys[valid], zs[valid],
                            c=Zi_flat[valid], cmap=cmap, s=s,
                            alpha=alpha, vmin=vmin, vmax=vmax)
        else:
            active = np.abs(mag) >= vmax * 0.01
            if active.any():
                sc = ax.scatter(x[active], y[active], z[active], c=mag[active],
                                cmap=cmap, s=s, alpha=alpha, vmin=vmin, vmax=vmax)

        if 'sc' in dir():
            fig.colorbar(sc, ax=ax, shrink=0.5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.grid(False)
        if axis_equal is True:
            ax.axis('equal')
        ax.set_title(f'{self.name} | t={actual_t:.3f}s | {clbl}', fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()


    def create_animation(self, time_start=0.0, time_end=None, n_frames=50,
                         component='z', data_type='vel', cmap='RdBu_r',
                         figsize=(12,8), dpi=100, fps=10,
                         elev=30, azim=-60, s=20, alpha=0.85,
                         ffmpeg_path=None, output_dir='animation', output_video='animation.mp4',
                         axis_equal=True, vmax_from_range=False):

        """Create a 3-D scatter animation of the full domain."""
        # Ensure vmax is computed
        if self._vmax is None:
            self._compute_vmax()
        import subprocess
        os.makedirs(output_dir, exist_ok=True)
        if time_end is None: time_end = self.time[-1]

        if vmax_from_range:
            i0   = int(np.argmin(np.abs(self.time - time_start)))
            i1   = int(np.argmin(np.abs(self.time - time_end)))
            path = {'accel': f'{self._data_grp}/acceleration',
                    'vel':   f'{self._data_grp}/velocity',
                    'disp':  f'{self._data_grp}/displacement'}[data_type]
            _chunk_rows = 600
            vmax = 0.0
            with h5py.File(self.filename, 'r') as f:
                n_rows = f[path].shape[0]
                for _s in range(0, n_rows, _chunk_rows):
                    _e  = min(_s + _chunk_rows, n_rows)
                    _d  = f[path][_s:_e, i0:i1+1]
                    if component.lower() == 'resultant':
                        _ed = _d[0::3,:]; _nd = _d[1::3,:]; _zd = _d[2::3,:]
                        vmax = max(vmax,
                                   float(np.sqrt(_ed**2+_nd**2+_zd**2).max()))
                    else:
                        _row = {'e': 0, 'n': 1, 'z': 2}[component.lower()]
                        vmax = max(vmax, float(np.abs(_d[_row::3,:]).max()))
            vmin = 0 if component.lower() == 'resultant' else -vmax
        else:
            if component.lower() == 'resultant':
                vmax = self._vmax[data_type]['resultant']; vmin = 0
            else:
                vmax = self._vmax[data_type][component.lower()]; vmin = -vmax

        xyz_t = _rotate(self.xyz)
        x=xyz_t[:,0]; y=xyz_t[:,1]; z=xyz_t[:,2]

        for i,t in enumerate(np.linspace(time_start, time_end, n_frames)):
            it = int(np.argmin(np.abs(self.time - t)))
            if component.lower() == 'resultant':
                mag = np.sqrt(self.get_surface_snapshot(it,'e',data_type)**2+
                              self.get_surface_snapshot(it,'n',data_type)**2+
                              self.get_surface_snapshot(it,'z',data_type)**2)
            else:
                mag = self.get_surface_snapshot(it, component, data_type)
            fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c='lightgray', s=s, alpha=0.05)
            active = np.abs(mag) >= vmax * 0.01
            if active.any():
                ax.scatter(x[active], y[active], z[active], c=mag[active],
                           cmap=cmap, s=s, alpha=alpha, vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([]); fig.colorbar(sm, ax=ax, shrink=0.5)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f't = {self.time[it]:.3f} s', fontsize=14, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)
            ax.grid(False)
            if axis_equal:
                ax.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/frame_{i:03d}.png', dpi=dpi)
            plt.close()
            print(f'Frame {i+1}/{n_frames}')
        try:
            ffmpeg_exe = ffmpeg_path or shutil.which('ffmpeg') or 'ffmpeg'
            subprocess.run([ffmpeg_exe, '-y', '-framerate', str(fps),
                            '-i', f'{output_dir}/frame_%03d.png',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '18', output_video], check=True, capture_output=True)
            print(f'Video saved: {output_video}')
        except Exception as e:
            print(f'ffmpeg error — frames in {output_dir}: {e}')



    def create_animation_plane(self, plane='xy', plane_value=0.0,
                                time_start=0.0, time_end=None, n_frames=50,
                                component='z', data_type='vel', cmap='RdBu_r',
                                figsize=(12,8), dpi=100, fps=10,
                                elev=30, azim=-60, s=50, alpha=0.85,
                                ffmpeg_path= None,
                                output_dir='animation_plane',
                                output_video='animation_plane.mp4',
                                vmax_from_range=False,
                                axis_equal=True):

        """Create a 3-D animation of a planar slice through the domain."""
        # Ensure vmax is computed
        if self._vmax is None:
            self._compute_vmax()
        import subprocess
        os.makedirs(output_dir, exist_ok=True)
        if time_end is None: time_end = self.time[-1]
        x=self.xyz[:,0]*1000; y=self.xyz[:,1]*1000; z=self.xyz[:,2]*1000

        xyz_t = _rotate(self.xyz)
        x=xyz_t[:,0]; y=xyz_t[:,1]; z=xyz_t[:,2]

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
            _chunk_rows = 600
            vmax = 0.0
            with h5py.File(self.filename,'r') as f:
                n_rows = f[path].shape[0]
                for _s in range(0, n_rows, _chunk_rows):
                    _e  = min(_s + _chunk_rows, n_rows)
                    _d  = f[path][_s:_e, i0:i1+1]
                    _pidx_chunk = pidx[(pidx >= _s) & (pidx < _e)] - _s
                    if len(_pidx_chunk) == 0:
                        continue
                    if component.lower()=='resultant':
                        _ed=_d[0::3,:][_pidx_chunk]
                        _nd=_d[1::3,:][_pidx_chunk]
                        _zd=_d[2::3,:][_pidx_chunk]
                        vmax=max(vmax,
                                 float(np.sqrt(_ed**2+_nd**2+_zd**2).max()))
                    else:
                        _row={'e':0,'n':1,'z':2}[component.lower()]
                        vmax=max(vmax,
                                 float(np.abs(_d[_row::3,:][_pidx_chunk]).max()))
            vmin = 0 if component.lower()=='resultant' else -vmax
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

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            # ax.invert_xaxis()
            # ax.invert_yaxis()
            # ax.invert_zaxis()
            ax.set_title(f'{tpl} | t = {self.time[it]:.3f} s',fontsize=14,fontweight='bold')
            ax.view_init(elev=elev,azim=azim)
            ax.grid(False)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/frame_{i:03d}.png',dpi=dpi)
            plt.close()
            print(f'Frame {i+1}/{n_frames}')
        try:
            ffmpeg_exe = ffmpeg_path or shutil.which('ffmpeg') or 'ffmpeg'
            subprocess.run([ffmpeg_exe, '-y', '-framerate', str(fps),
                            '-i', f'{output_dir}/frame_%03d.png',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '18', output_video], check=True, capture_output=True)
            print(f'Video saved: {output_video}')
        except Exception as e:
            print(f'ffmpeg error — frames in {output_dir}: {e}')

# -------------------------
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












#### PLOT SURFACES WITH PARALLEL POOL
    def plot_surface_newmark(self,
                             T_target=0.0,
                             component='z',
                             data_type='accel',
                             spectral_type='PSa',
                             factor=1.0,
                             cmap='hot_r',
                             figsize=(12, 8),
                             elev=30, azim=-60,
                             s=20, alpha=0.85,
                             axis_equal=False,
                             n_jobs=-1):
        """Plot a 3-D scatter map of spectral values at a given period T.

        Full spectra (Z, E, N) for all spectral quantities are computed once
        and cached per ``data_type``.  Subsequent calls with the same
        ``data_type`` are instantaneous regardless of changes to ``T_target``,
        ``component``, ``spectral_type``, ``factor``, or any plot parameter.
        Only changing ``data_type`` triggers a full recomputation.

        The method automatically selects the loading strategy based on
        available RAM:

        - **fast/preload** — all node data is loaded into RAM before the
          parallel pool starts.  Used when the data fits comfortably in RAM.
        - **safe/chunk**   — each parallel worker opens the HDF5 file and
          reads only its own node (~0.5 MB).  Peak RAM is proportional to
          ``n_jobs``, not to the number of nodes.  Used for large files.

        Window masks (``get_window``) and resampling (``resample``) are
        respected in both modes.

        Parameters
        ----------
        T_target : float, default ``0.0``
            Target period in seconds.  Use ``0.0`` to obtain the PGA map.
        component : {'z', 'e', 'n', 'resultant'}, default ``'z'``
            Signal component used to compute the spectrum.  ``'resultant'``
            averages the spectra of Z, E, and N.
        data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
        spectral_type : {'PSa', 'Sa', 'PSv', 'Sv', 'Sd'}, default ``'PSa'``
        factor : float, default ``1.0``
            Multiplier applied to every spectral value before plotting.
        cmap : str, default ``'hot_r'``
        figsize : tuple of float, default ``(12, 8)``
        elev, azim : float
            3-D view angles.
        s : int, default ``20``
            Scatter marker size.
        alpha : float, default ``0.85``
        axis_equal : bool, default ``False``
        n_jobs : int, default ``-1``
            Number of parallel workers.  ``-1`` uses all CPUs,
            ``-2`` uses all minus one.
        """
        from joblib import Parallel, delayed

        dt        = self.time[1] - self.time[0]
        n         = self._n_nodes
        comp      = component.lower()
        cache_key = (data_type,)

        # Cache check — keyed only on data_type so all spectral quantities
        # and components are available without recomputing.
        if not hasattr(self, '_newmark_cache'):
            self._newmark_cache = {}

        if cache_key in self._newmark_cache:
            print(f"  Cache hit — using stored spectra for {data_type}")
            T_array, sa_full = self._newmark_cache[cache_key]
        else:
            import psutil as _psutil
            mem_available = _psutil.virtual_memory().available
            data_needed   = self._bytes_per_node * n
            use_safe_mode = self._large_file or (data_needed > mem_available * 0.6)

            print(f"Computing spectra for {n} nodes  n_jobs={n_jobs}")
            print(f"  Mode     : {'safe/chunk' if use_safe_mode else 'fast/preload'}"
                  f"  ({data_needed/1e9:.1f} GB needed  |  "
                  f"{mem_available/1e9:.1f} GB available)")

            # Capture everything workers need — no self references inside
            # worker functions to avoid serialisation issues with joblib.
            _filename       = self.filename
            _data_grp       = self._data_grp
            _hdf5_path      = {'accel': f'{_data_grp}/acceleration',
                               'vel':   f'{_data_grp}/velocity',
                               'disp':  f'{_data_grp}/displacement'}[data_type]
            _window_mask    = getattr(self, '_window_mask',    None)
            _resample_cache = getattr(self, '_resample_cache', None)
            _time_len       = len(self.time)

            if use_safe_mode:
                # Each worker reads its own node directly from HDF5.
                # Peak RAM = n_jobs * bytes_per_node (a few MB at most).
                def _compute_spectrum(i):
                    with h5py.File(_filename, 'r') as _f:
                        _d = _f[_hdf5_path][3*i : 3*i+3, :]
                    _d = _d[[2, 0, 1], :]          # reorder E,N,Z -> Z,E,N
                    if _window_mask is not None:
                        _d = _d[:, _window_mask]
                    elif _resample_cache is not None:
                        _t_orig = _resample_cache['time_orig']
                        _rs = np.zeros((3, _time_len))
                        for _k in range(3):
                            _rs[_k] = interp1d(_t_orig, _d[_k],
                                               kind='linear',
                                               fill_value='extrapolate')(
                                np.linspace(_t_orig[0], _t_orig[-1], _time_len))
                        _d = _rs
                    specs = [NewmarkSpectrumAnalyzer.compute(_d[k], dt)
                             for k in range(3)]
                    T  = specs[0]['T']
                    sa = {qty: np.array([sp[qty] for sp in specs])
                          for qty in ('PSa', 'Sa', 'PSv', 'Sv', 'Sd')}
                    return T, sa
            else:
                # Pre-load all data into RAM; workers read from the array.
                print("  Loading data into memory...")
                all_data = np.zeros((n, 3, len(self.time)))
                for i in range(n):
                    all_data[i] = self.get_node_data(i, data_type)
                print("  Data loaded. Computing spectra...")

                def _compute_spectrum(i):
                    data  = all_data[i]
                    specs = [NewmarkSpectrumAnalyzer.compute(data[k], dt)
                             for k in range(3)]
                    T  = specs[0]['T']
                    sa = {qty: np.array([sp[qty] for sp in specs])
                          for qty in ('PSa', 'Sa', 'PSv', 'Sv', 'Sd')}
                    return T, sa

            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_compute_spectrum)(i) for i in range(n))

            T_array = results[0][0]
            # sa_full[qty] shape: (n_nodes, 3, n_periods)
            sa_full = {qty: np.array([r[1][qty] for r in results])
                       for qty in ('PSa', 'Sa', 'PSv', 'Sv', 'Sd')}

            self._newmark_cache[cache_key] = (T_array, sa_full)
            print(f"Done. All spectral quantities cached for {data_type}")

        # Apply component, T_target, spectral_type and factor.
        # This is pure numpy interpolation — instantaneous.
        sp_data = sa_full[spectral_type]   # (n_nodes, 3, n_periods)

        if comp == 'resultant':
            sa_map = np.array([
                np.mean([np.interp(T_target, T_array, sp_data[i][k])
                         for k in range(3)])
                for i in range(n)]) * factor
        else:
            k = {'z': 0, 'e': 1, 'n': 2}[comp]
            sa_map = np.array([
                np.interp(T_target, T_array, sp_data[i][k])
                for i in range(n)]) * factor

        print(f"  {spectral_type}(T={T_target}s) | {comp} | factor={factor}  "
              f"Max={sa_map.max():.4f}  Min={sa_map.min():.4f}")

        # Plot
        xyz_t = _rotate(self.xyz)
        x = xyz_t[:, 0]; y = xyz_t[:, 1]; z = xyz_t[:, 2]
        clbl  = {'z': 'Vertical (Z)', 'e': 'East (E)',
                 'n': 'North (N)', 'resultant': 'Resultant'}[comp]

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        sa_map = np.nan_to_num(sa_map, nan=0.0)
        sc  = ax.scatter(x, y, z, c=sa_map, cmap=cmap, s=s, alpha=alpha,
                 vmin=0, vmax=np.nanmax(sa_map))
        fig.colorbar(sc, ax=ax, shrink=0.5,
                     label=f'{spectral_type}(T={T_target}s)')

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.grid(False)
        if axis_equal:
            ax.axis('equal')
        ax.set_title(f'{self.name} | {spectral_type}(T={T_target}s) | {clbl}',
                     fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()


    def plot_surface_arias(self,
                           component='z',
                           data_type='accel',
                           factor=1.0,
                           cmap='hot_r',
                           figsize=(12, 8),
                           elev=30, azim=-60,
                           s=20, alpha=0.85,
                           axis_equal=False,
                           n_jobs=-1):
        """Plot a 3-D scatter map of Arias intensity for every node.

        Arias intensity (Z, E, N) is computed once and cached per
        ``data_type``.  Subsequent calls with the same ``data_type`` are
        instantaneous.  Only changing ``data_type`` triggers recomputation.

        The method automatically selects the loading strategy based on
        available RAM (same logic as ``plot_surface_newmark``):

        - **fast/preload** — all node data loaded into RAM before the pool.
        - **safe/chunk**   — each worker reads only its own node from HDF5.

        Window masks (``get_window``) and resampling (``resample``) are
        respected in both modes.

        Parameters
        ----------
        component : {'z', 'e', 'n', 'resultant'}, default ``'z'``
            Component to display.  ``'resultant'`` averages Z, E, N.
        data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
        factor : float, default ``1.0``
            Multiplier applied to every Arias value before plotting.
        cmap : str, default ``'hot_r'``
        figsize : tuple of float, default ``(12, 8)``
        elev, azim : float
            3-D view angles.
        s : int, default ``20``
            Scatter marker size.
        alpha : float, default ``0.85``
        axis_equal : bool, default ``False``
        n_jobs : int, default ``-1``
            Number of parallel workers.  ``-1`` uses all CPUs.
        """
        from joblib import Parallel, delayed
        from EarthquakeSignal.core.arias_intensity import AriasIntensityAnalyzer

        dt        = self.time[1] - self.time[0]
        n         = self._n_nodes
        comp      = component.lower()
        cache_key = (data_type, 'arias')

        # Cache check — keyed on (data_type, 'arias').
        if not hasattr(self, '_newmark_cache'):
            self._newmark_cache = {}

        if cache_key in self._newmark_cache:
            print(f"  Cache hit — using stored Arias for {data_type}")
            ia_full = self._newmark_cache[cache_key]
        else:
            import psutil as _psutil
            mem_available = _psutil.virtual_memory().available
            data_needed   = self._bytes_per_node * n
            use_safe_mode = self._large_file or (data_needed > mem_available * 0.6)

            print(f"Computing Arias intensity for {n} nodes  n_jobs={n_jobs}")
            print(f"  Mode     : {'safe/chunk' if use_safe_mode else 'fast/preload'}"
                  f"  ({data_needed/1e9:.1f} GB needed  |  "
                  f"{mem_available/1e9:.1f} GB available)")

            # Capture state for workers — no self references.
            _filename       = self.filename
            _data_grp       = self._data_grp
            _hdf5_path      = {'accel': f'{_data_grp}/acceleration',
                               'vel':   f'{_data_grp}/velocity',
                               'disp':  f'{_data_grp}/displacement'}[data_type]
            _window_mask    = getattr(self, '_window_mask',    None)
            _resample_cache = getattr(self, '_resample_cache', None)
            _time_len       = len(self.time)

            if use_safe_mode:
                # Each worker reads its own node directly from HDF5.
                # AriasIntensityAnalyzer imported inside worker to avoid
                # serialisation issues with joblib LokyBackend.
                def _compute_arias(i):
                    from EarthquakeSignal.core.arias_intensity import \
                        AriasIntensityAnalyzer as _AIA
                    with h5py.File(_filename, 'r') as _f:
                        _d = _f[_hdf5_path][3*i : 3*i+3, :]
                    _d = _d[[2, 0, 1], :]          # reorder E,N,Z -> Z,E,N
                    if _window_mask is not None:
                        _d = _d[:, _window_mask]
                    elif _resample_cache is not None:
                        _t_orig = _resample_cache['time_orig']
                        _rs = np.zeros((3, _time_len))
                        for _k in range(3):
                            _rs[_k] = interp1d(_t_orig, _d[_k],
                                               kind='linear',
                                               fill_value='extrapolate')(
                                np.linspace(_t_orig[0], _t_orig[-1], _time_len))
                        _d = _rs
                    ia = np.zeros(3)
                    for k in range(3):
                        _, _, _, ia_total, _ = _AIA.compute(_d[k] / 9.81, dt)
                        ia[k] = ia_total
                    return ia
            else:
                # Pre-load all data into RAM; workers read from the array.
                print("  Loading data into memory...")
                all_data = np.zeros((n, 3, len(self.time)))
                for i in range(n):
                    all_data[i] = self.get_node_data(i, data_type)
                print("  Data loaded. Computing Arias intensity...")

                def _compute_arias(i):
                    data = all_data[i]
                    ia   = np.zeros(3)
                    for k in range(3):
                        _, _, _, ia_total, _ = AriasIntensityAnalyzer.compute(
                            data[k] / 9.81, dt)
                        ia[k] = ia_total
                    return ia

            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_compute_arias)(i) for i in range(n))

            # ia_full shape: (n_nodes, 3) — columns: Z, E, N
            ia_full = np.array(results)
            self._newmark_cache[cache_key] = ia_full
            print(f"Done. Arias intensity cached for {data_type}")

        # Apply component and factor — instantaneous.
        if comp == 'resultant':
            ia_map = np.mean(ia_full, axis=1) * factor
        else:
            k      = {'z': 0, 'e': 1, 'n': 2}[comp]
            ia_map = ia_full[:, k] * factor

        print(f"  Arias | {comp} | factor={factor}  "
              f"Max={ia_map.max():.4f}  Min={ia_map.min():.4f}")

        # Plot
        xyz_t = _rotate(self.xyz)
        x = xyz_t[:, 0]; y = xyz_t[:, 1]; z = xyz_t[:, 2]
        clbl  = {'z': 'Vertical (Z)', 'e': 'East (E)',
                 'n': 'North (N)', 'resultant': 'Resultant'}[comp]

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        ia_map = np.nan_to_num(ia_map, nan=0.0)
        sc  = ax.scatter(x, y, z, c=ia_map, cmap=cmap, s=s, alpha=alpha,
                 vmin=0, vmax=np.nanmax(ia_map))
        fig.colorbar(sc, ax=ax, shrink=0.5, label='Arias Intensity [m/s]')

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.grid(False)
        if axis_equal:
            ax.axis('equal')
        ax.set_title(f'{self.name} | Arias Intensity | {clbl}', fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()


    
# # ---------------------------------------------------------------------------
# # Semantic aliases
# # ---------------------------------------------------------------------------
# DRMData     = ShakerMakerData   # DRMBox / PointCloudDRMReceiver
# SurfaceData = ShakerMakerData   # SurfaceGrid