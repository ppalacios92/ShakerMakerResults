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
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from ..analysis.newmark import NewmarkSpectrumAnalyzer
from ..utils import _rotate



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

        # --- NEW: GF + MAP paths ---
        self._gf_h5_path = None
        self._gf_map_h5_path = None

        # --- NEW: flags ---
        self._has_gf = False
        self._has_map = False

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
            gf_orig   = np.arange(n_time_gf) * dt_orig
            self.time    = np.arange(t_orig[0],  t_orig[-1],  dt)
            self.gf_time = np.arange(gf_orig[0], gf_orig[-1], dt) if len(gf_orig) > 0 else np.array([])
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
        gf_steps = int(getattr(self, "_n_time_gf", n_time_gf))
        gf_slots = None
        tdata_shape = getattr(self, "_tdata_shape", None)
        if tdata_shape is not None and len(tdata_shape) >= 1:
            try:
                gf_slots = int(tdata_shape[0])
            except (TypeError, ValueError):
                gf_slots = None
        if self._has_gf and self._has_map:
            gf_msg = f"steps={gf_steps}"
            if gf_slots is not None:
                gf_msg += f"  |  slots={gf_slots}"
            gf_msg += f"  |  subfaults={int(getattr(self, '_nsources_db', 0))}"
        elif self._has_gf:
            gf_msg = f"steps={gf_steps}"
            if gf_slots is not None:
                gf_msg += f"  |  slots={gf_slots}"
            gf_msg += "  |  map not loaded (subfaults unavailable)"
        else:
            gf_msg = f"steps={gf_steps}  |  not loaded"
        # TODO: Keep the GF summary focused on user-facing concepts:
        # time steps, unique slots, and subfault count from the map.
        print(f"  GF       : {gf_msg}")
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
                self._has_map   = True 
                unique = np.unique(self._pairs_to_compute[:, 0])
                mode   = "O(1) pair_to_slot" if self._use_pair_to_slot else "KDTree"
                print(f"  GF DB ({mode}): ...")
                return

            #  Legacy: Node_Mapping 
            if 'Node_Mapping/node_to_pair_mapping' in f:
                self.node_mapping  = f['Node_Mapping/node_to_pair_mapping'][:]
                self.pairs_mapping = f['Node_Mapping/pairs_to_compute'][:]
                print("  GF mapping loaded (legacy Node_Mapping).")







    # def _compute_vmax(self):
    #     import json
    #     print(f"  Computing vmax (hardcoded)...")
    #     vmax = {
    #         'accel': {'e': 10.0, 'n': 10.0, 'z': 10.0, 'resultant': 10.0},
    #         'vel':   {'e': 10.0, 'n': 10.0, 'z': 10.0, 'resultant': 10.0},
    #         'disp':  {'e': 10.0, 'n': 10.0, 'z': 10.0, 'resultant': 10.0},
    #     }
    #     self._vmax = vmax
    #     try:
    #         with open(self._vmax_cache_path, 'w') as cf:
    #             json.dump(vmax, cf)
    #         print(f"  vmax cached to: {self._vmax_cache_path}")
    #     except Exception as e:
    #         print(f"  vmax cache write failed: {e}")


    def _compute_vmax(self):
        from ..analysis.vmax_service import compute_vmax

        return compute_vmax(self)




    def _get_slot(self, node_id, subfault_id):
        """Return GF slot index for (node_id, subfault_id).

        Primary: O(1) flat array via pair_to_slot[node * nsources + subfault].
        Fallback: KDTree lookup for legacy databases.
        """
        if not self._has_map:
            raise RuntimeError("Map not loaded. Call load_map('file_map.h5') first.")
        
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












    def viewer(self, show=True, **kwargs):
        """Open an interactive viewer session for this model.

        Parameters
        ----------
        show : bool, default ``True``
            When ``True``, build and show the Qt/PyVista window
            immediately. When ``False``, return the session object
            without opening the GUI yet.
        **kwargs
            Forwarded to :class:`ShakerMakerResults.viewer.ViewerSession`.

        Returns
        -------
        ViewerSession
            Interactive session bound to the current model instance.
        """
        from ..viewer import ViewerSession

        return ViewerSession(self, show=show, **kwargs)

    # ------------------------------------------------------------------
    # Windowing / resampling
    # ------------------------------------------------------------------





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







    

    








    # ------------------------------------------------------------------
    # Surface / animation methods  (primarily for SurfaceGrid outputs)
    # ------------------------------------------------------------------
    






# -------------------------





#### PLOT SURFACES WITH PARALLEL POOL






    
    
    
    





# # ---------------------------------------------------------------------------
# # Semantic aliases
# # ---------------------------------------------------------------------------
# DRMData     = ShakerMakerData   # DRMBox / PointCloudDRMReceiver
# SurfaceData = ShakerMakerData   # SurfaceGrid


    
    
    

    # ------------------------------------------------------------------
    # Delegated public API
    # ------------------------------------------------------------------

    def load_gf_database(self, h5_path):
        from .gf_service import load_gf_database

        return load_gf_database(self, h5_path)

    def load_map(self, h5_path):
        from .gf_service import load_map

        return load_map(self, h5_path)

    def get_node_data(self, node_id, data_type='accel'):
        from .query_service import get_node_data

        return get_node_data(self, node_id, data_type)

    def get_qa_data(self, data_type='accel'):
        from .query_service import get_qa_data

        return get_qa_data(self, data_type)

    def get_gf(self, node_id, subfault_id, component='z'):
        from .gf_service import get_gf

        return get_gf(self, node_id, subfault_id, component)

    def get_surface_snapshot(self, time_idx, component='z', data_type='vel'):
        from .query_service import get_surface_snapshot

        return get_surface_snapshot(self, time_idx, component, data_type)

    def clear_cache(self):
        from .query_service import clear_cache

        return clear_cache(self)

    def get_window(self, t_start, t_end):
        from .window_service import get_window

        return get_window(self, t_start, t_end)

    def resample(self, dt):
        from .window_service import resample

        return resample(self, dt)

    def plot_domain(self, **kwargs):
        from ..plotting.single_model.domain_plots import plot_domain

        return plot_domain(self, **kwargs)

    def plot_node_response(self, node_id=None, **kwargs):
        from ..plotting.single_model.node_plots import plot_node_response

        return plot_node_response(self, node_id=node_id, **kwargs)

    def plot_node_gf(self, node_id=None, **kwargs):
        from ..plotting.single_model.node_plots import plot_node_gf

        return plot_node_gf(self, node_id=node_id, **kwargs)

    def plot_node_tensor_gf(self, node_id=None, **kwargs):
        from ..plotting.single_model.node_plots import plot_node_tensor_gf

        return plot_node_tensor_gf(self, node_id=node_id, **kwargs)

    def plot_node_newmark(self, node_id=None, **kwargs):
        from ..plotting.single_model.node_plots import plot_node_newmark

        return plot_node_newmark(self, node_id=node_id, **kwargs)

    def plot_calculated_vs_reused(self, **kwargs):
        from ..plotting.single_model.domain_plots import plot_calculated_vs_reused

        return plot_calculated_vs_reused(self, **kwargs)

    def plot_gf_connections(self, **kwargs):
        from ..plotting.single_model.domain_plots import plot_gf_connections

        return plot_gf_connections(self, **kwargs)

    def plot_surface(self, **kwargs):
        from ..plotting.single_model.surface_plots import plot_surface

        return plot_surface(self, **kwargs)

    def create_animation(self, **kwargs):
        from ..plotting.single_model.animation_plots import create_animation

        return create_animation(self, **kwargs)

    def create_animation_plane(self, **kwargs):
        from ..plotting.single_model.animation_plots import create_animation_plane

        return create_animation_plane(self, **kwargs)

    def plot_node_arias(self, node_id=None, **kwargs):
        from ..plotting.single_model.node_plots import plot_node_arias

        return plot_node_arias(self, node_id=node_id, **kwargs)

    def plot_surface_newmark(self, **kwargs):
        from ..plotting.single_model.surface_plots import plot_surface_newmark

        return plot_surface_newmark(self, **kwargs)

    def plot_surface_arias(self, **kwargs):
        from ..plotting.single_model.surface_plots import plot_surface_arias

        return plot_surface_arias(self, **kwargs)

    def write_h5drm(self, name=None):
        from ..io.export_service import write_h5drm

        return write_h5drm(self, name=name)

    def plot_surface_on_map(self, mapa, **kwargs):
        from ..plotting.single_model.map_plots import plot_surface_on_map

        return plot_surface_on_map(self, mapa=mapa, **kwargs)

    def create_animation_map(self, **kwargs):
        from ..plotting.single_model.map_plots import create_animation_map

        return create_animation_map(self, **kwargs)


DRMData = ShakerMakerData
SurfaceData = ShakerMakerData

