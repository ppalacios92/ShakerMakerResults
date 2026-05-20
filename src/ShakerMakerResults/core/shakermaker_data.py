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

    Detects the layout from the HDF5 root group:

    * ``/DRM_Data``  -- DRM box / SurfaceGrid / PointCloud output
    * ``/Data``      -- plain station-list output

    Also auto-loads Green-Function metadata when present, in either the
    OP-pipeline format (``GF_Database_Info`` + ``GF_tdata`` + optional
    ``pair_to_slot`` for O(1) lookup) or the legacy ``Node_Mapping`` layout.

    Parameters
    ----------
    filename : str
        Path to a ShakerMaker HDF5 file.
    dt : float, optional
        Target time step. When given, the time vector is rebuilt at this
        resolution and per-node reads are resampled on demand using a linear
        interpolator. ``None`` keeps the native ``dt`` stored in the file.

    Attributes
    ----------
    filename : str
    name : str
        Stem of ``filename`` (no path, no extension). Used as the display
        label in plots and the viewer.
    is_drm : bool
        ``True`` when the file uses the DRM layout, ``False`` for the
        station-only layout.
    xyz : np.ndarray, shape (n_nodes, 3)
        Node positions in **kilometres**, in ShakerMaker's native frame.
    xyz_qa : np.ndarray or None
        Optional QA / control-station position. ``None`` for non-DRM files
        and DRM files without QA.
    xyz_all : np.ndarray
        ``xyz`` stacked with ``xyz_qa`` when QA exists; otherwise just
        ``xyz``. Convenient for nearest-neighbour searches.
    internal : np.ndarray of bool, shape (n_nodes,)
        ``True`` for internal DRM nodes, ``False`` for boundary nodes.
    spacing : tuple of float
        Grid spacing along ``(x', y', z')`` in **metres**, computed from
        the rotated coordinates.
    time : np.ndarray
        Data time vector in seconds, after optional resampling.
    gf_time : np.ndarray
        Green-Function time vector in seconds (empty if no GF loaded).
    dt : float
        Active data time step in seconds.

    Notes
    -----
    The constructor only reads metadata + the (small) coordinate arrays, so
    instantiation is cheap even on very large files. Heavy reads are done
    on demand through the delegated methods (``get_node_data``,
    ``get_surface_snapshot``, etc.) and cached in ``_node_cache`` /
    ``_gf_cache``. A ``_vmax`` sidecar (``<filename>.vmax.json``) is loaded
    when present so surface-colour limits are available without scanning
    the whole file.
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

            self.name = os.path.splitext(os.path.basename(filename))[0]

            n_nodes     = len(self.xyz)
            n_time_data = f[f'{data_grp}/velocity'].shape[1]

            # self.freqs = None
            # if 'GF_Spectrum/sta_0/sub_0/freqs' in f:
            #     self.freqs = f['GF_Spectrum/sta_0/sub_0/freqs'][:]

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

        # Estimate the regular grid spacing along each rotated axis. Two
        # near-coincident nodes are treated as one (rounded to 1 µm); the
        # smallest gap that remains is taken as the spacing.
        def _spacing(arr): 
            d = np.diff(np.sort(np.unique(np.round(arr, 6))))
            if len(d) > 0:
                return float(d[0])
            else:
                return 0.0

        h_x = _spacing(xyz_t[:, 0])
        h_y = _spacing(xyz_t[:, 1])
        h_z = _spacing(xyz_t[:, 2])

        self.spacing    = (h_x, h_y, h_z)

        self._dt_orig    = dt_orig; self._tstart     = tstart
        self._n_nodes    = n_nodes; self._n_subfaults = n_subfaults
        self._n_time_gf  = n_time_gf; self._n_time_data = n_time_data
        self._data_grp   = data_grp; self._meta_grp   = meta_grp
        self._qa_grp     = qa_grp

        self._node_cache = {}
        self._gf_cache = {}
        self._spectrum_cache = {}

        # GF + MAP paths 
        self._gf_h5_path = None
        self._gf_map_h5_path = None

        # flags
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
        # sep = '--' * 50
        is_surface = self.is_drm and not np.any(self.internal)
        type_str   = 'SurfaceGrid' if is_surface else ('DRM' if self.is_drm else 'Station')

        xyz_t_print = _rotate(self.xyz)
        Lx = xyz_t_print[:,0].max() - xyz_t_print[:,0].min()
        Ly = xyz_t_print[:,1].max() - xyz_t_print[:,1].min()
        Lz = xyz_t_print[:,2].max() - xyz_t_print[:,2].min()

        print('--' * 50)
        print(f"ShakerMakerData  :  {filename}")
        print(f"  Type     : {type_str}")
        print(f"  Model    : {self.name}  |  Spacing: {h_x:.1f}m x {h_y:.1f}m x {h_z:.1f}m")
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
        print('--' * 50 + '\n')

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


    def _compute_vmax(self):
        """Compute (and cache to a sidecar JSON) the per-component vmax map."""
        from ..analysis.vmax_service import compute_vmax

        return compute_vmax(self)



    def _get_slot(self, node_id, subfault_id):
        """Return GF slot index for a ``(node_id, subfault_id)`` pair.

        Parameters
        ----------
        node_id : int or {'QA', 'qa'}
            Node index. ``'QA'`` maps to the index that lives one past the
            last regular node so the same flat lookup table covers it.
        subfault_id : int
            Subfault index inside the source.

        Returns
        -------
        int
            Slot index in the GF ``tdata`` dataset.

        Notes
        -----
        Primary path: O(1) flat lookup via
        ``pair_to_slot[node * nsources + subfault]``.

        Fallback: KDTree query against the ``(dh, zsrc, zrec)`` cloud --
        used only for legacy databases that don't ship a ``pair_to_slot``
        array.
        """
        if not self._has_map:
            raise RuntimeError("Map not loaded. Call load_map('file_map.h5') first.")
        
        # Translate 'QA' / 'qa' to its numeric index, which lives one
        # past the last regular node (so QA shares the same flat array).
        if node_id in ('QA', 'qa'):
            node_id = self._n_nodes
        
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_node(self, node_id, data_type):
        """Return ``(data, label)`` for a node id, treating QA-like ids transparently.

        Parameters
        ----------
        node_id : int or {'QA', 'qa'}
            Node index. Anything ``>= len(self.xyz)`` is treated as QA.
        data_type : {'accel', 'vel', 'disp'}

        Returns
        -------
        tuple
            ``(np.ndarray shape (3, Nt), str)`` -- the signal triplet and a
            short label suitable for plot legends ("Node 12" / "QA").
        """
        if node_id in ('QA','qa') or (isinstance(node_id,int) and node_id>=len(self.xyz)):
            return self.get_qa_data(data_type), 'QA'
        return self.get_node_data(node_id, data_type), f'Node {node_id}'

    @staticmethod
    def _build_cube_faces(xyz_nodes):
        """Return cube vertices + polygon faces enclosing a 3-D point set.

        Parameters
        ----------
        xyz_nodes : np.ndarray, shape (N, 3)
            Points whose axis-aligned bounding box defines the cube.

        Returns
        -------
        tuple
            ``(corners, faces, bounds)`` where:

            * ``corners`` is a ``(8, 3)`` array of the eight cube vertices.
            * ``faces`` is a list of 5 polygons (top + 4 lateral). The
              bottom face is intentionally omitted to keep the 3-D scatter
              readable from below.
            * ``bounds`` is ``(x0, x1, y0, y1, z0, z1)``.
        """
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
        """Draw per-node text labels on a matplotlib 3-D axes object.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D
        xyz_t : np.ndarray, shape (n_nodes, 3)
            Rotated node coordinates in metres.
        bounds : tuple
            ``(x0, x1, y0, y1, z0, z1)`` from :meth:`_build_cube_faces` --
            used to detect "corner / edge / mid" classes.
        label_nodes : bool or str
            * ``False``           : nothing drawn (caller usually skips us).
            * ``True``            : label every node.
            * ``'corners'``       : only true cube corners.
            * ``'corners_edges'`` : corners and edge nodes.
            * ``'corners_half'``  : corners and edge midpoints.
            * ``'calculated'``    : only nodes in ``comp_donors``.
        comp_donors : set of int, optional
            Donor node ids; required when ``label_nodes='calculated'``.
        """
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
            # ``xyz_all`` has QA appended at the end, so an index past the
            # last regular row means the nearest point IS the QA station.
            if self.xyz_qa is not None and idx == len(self.xyz):
                nids = ['QA']
            else:
                nids = [idx]
        else:
            raise ValueError("Provide node_id or target_pos.")
        
        if print_info:
            # sep = '-' * 50
            print('-' * 50)
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
                    # Flag the rare case where a regular node sits on top
                    # of the QA station (mostly useful while debugging FFSP
                    # alignment).
                    qa_match = ""
                    if self.xyz_qa is not None:
                        dist_to_qa = np.linalg.norm(pos - self.xyz_qa[0])
                        if dist_to_qa < 1e-6:
                            qa_match = "  ★ COINCIDES WITH QA"
                    print(f"  N{nid:<6} │ pos = [{pos[0]*1000:>10.2f}, {pos[1]*1000:>10.2f}, {pos[2]*1000:>10.2f}] m │ {node_type}{qa_match}")

            # When the caller supplied a target_pos, print the distance to
            # the resolved node so they can sanity-check the snap.
            if target_pos is not None:
                target = np.asarray(target_pos)
                for nid in nids:
                    if nid in ('QA', 'qa'):
                        pos = self.xyz_qa[0]
                    else:
                        pos = self.xyz[nid]
                    dist = np.linalg.norm(pos - target) * 1000  # a metros
                    print(f"  Target   │ pos = [{target[0]*1000:>10.2f}, {target[1]*1000:>10.2f}, {target[2]*1000:>10.2f}] m │ dist = {dist:.2f} m")
            print('-' * 50)
        
        return nids


    def _donor_of_op(self, node_id, subfault_id):
        """Return the donor node id that owns the GF for one (node, subfault) pair.

        Parameters
        ----------
        node_id : int or {'QA', 'qa'}
        subfault_id : int

        Returns
        -------
        int
            Index in ``self.xyz`` of the node whose GF this pair reuses
            (equal to ``node_id`` when the pair is its own donor).
        """
        # Map 'QA' / 'qa' to the numeric index that lives at len(xyz).
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
    # Delegated public API
    # ------------------------------------------------------------------
    # Each method here is a one-line passthrough to a helper in
    # ``core.*_service``, ``plotting.*`` or ``io.export_service``. We keep
    # the trampolines (rather than attaching the helpers as bound methods)
    # so the heavy submodules stay import-deferred -- nothing in the
    # plotting stack is loaded until the user actually calls a plot method.

    def load_gf_database(self, h5_path):
        """Attach an external Green-Function HDF5 database to this model.

        Parameters
        ----------
        h5_path : str
            Path to a GF file in OP format (``tdata`` dataset + optional
            ``t0``). The handle path is stored; data is read lazily by
            :meth:`get_gf` / :meth:`get_gf_tensor`.

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.gf_service.load_gf_database`.
        """
        from .gf_service import load_gf_database

        return load_gf_database(self, h5_path)

    def load_map(self, h5_path):
        """Attach the GF subfault-to-slot mapping HDF5 file.

        Parameters
        ----------
        h5_path : str
            Path to a mapping file with ``pairs_to_compute``, ``pair_to_slot``,
            ``dh/zsrc/zrec_of_pairs``, ``delta_h/v_src/v_rec`` and ``nsources``.

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.gf_service.load_map`.
        Required before any GF query when the GF file does not embed its
        own mapping.
        """
        from .gf_service import load_map

        return load_map(self, h5_path)

    def get_node_data(self, node_id, data_type='accel'):
        """Return ``(3, Nt)`` array for one node, ordered as ``[Z, E, N]``.

        Parameters
        ----------
        node_id : int
            Node index in ``self.xyz``.
        data_type : {'accel', 'vel', 'disp'}, default ``'accel'``

        Returns
        -------
        np.ndarray, shape (3, Nt)
            Three component traces. Results are cached in ``self._node_cache``
            keyed by ``(node_id, data_type)``.

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.query_service.get_node_data`.
        Honours :meth:`get_window` masks and :meth:`resample` interpolation.
        """
        from .query_service import get_node_data

        return get_node_data(self, node_id, data_type)

    def get_qa_data(self, data_type='accel'):
        """Return ``(3, Nt)`` array for the QA station ordered as ``[Z, E, N]``.

        Parameters
        ----------
        data_type : {'accel', 'vel', 'disp'}, default ``'accel'``

        Returns
        -------
        np.ndarray, shape (3, Nt)

        Raises
        ------
        AttributeError
            If the file is a station-only output (no QA group).

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.query_service.get_qa_data`.
        """
        from .query_service import get_qa_data

        return get_qa_data(self, data_type)

    def get_gf(self, node_id, subfault_id, component='z'):
        """Return the Green-Function trace for a single (node, subfault) pair.

        Parameters
        ----------
        node_id : int or {'QA', 'qa'}
        subfault_id : int
        component : {'z', 'e', 'n', 'tdata'}, default ``'z'``
            ``'tdata'`` returns the full ``(Nt, 9)`` tensor; the others slice
            a single column.

        Returns
        -------
        np.ndarray
            Shape ``(Nt,)`` for a single component, ``(Nt, 9)`` for ``'tdata'``.

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.gf_service.get_gf`. Results
        are cached in ``self._gf_cache`` keyed by ``(node, subfault, comp)``.
        """
        from .gf_service import get_gf

        return get_gf(self, node_id, subfault_id, component)

    def get_surface_snapshot(self, time_idx, component='z', data_type='vel'):
        """Return one signal component for every node at a single time step.

        Parameters
        ----------
        time_idx : int
            Index into ``self.time``.
        component : {'z', 'e', 'n'}, default ``'z'``
        data_type : {'accel', 'vel', 'disp'}, default ``'vel'``

        Returns
        -------
        np.ndarray, shape (n_nodes,)

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.query_service.get_surface_snapshot`.
        Reads directly from disk -- no caching, since callers typically iterate
        across time.
        """
        from .query_service import get_surface_snapshot

        return get_surface_snapshot(self, time_idx, component, data_type)

    def clear_cache(self):
        """Drop all in-memory caches (node, GF, spectrum) and run ``gc.collect()``.

        Returns
        -------
        None
        """
        from .query_service import clear_cache

        return clear_cache(self)

    def get_window(self, t_start, t_end):
        """Return a *lazy* time-windowed copy of this model.

        Parameters
        ----------
        t_start, t_end : float
            Window bounds in seconds.

        Returns
        -------
        ShakerMakerData
            New instance sharing the original ``xyz`` / ``internal`` arrays
            but with a ``_window_mask`` that future reads honour. No signal
            data is read at this point.

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.window_service.get_window`.
        The cache attributes (``_node_cache``, ``_gf_cache``, etc.) are
        reset on the returned object.
        """
        from .window_service import get_window

        return get_window(self, t_start, t_end)

    def resample(self, dt):
        """Return a copy of this model with a different effective ``dt``.

        Parameters
        ----------
        dt : float
            Target time step in seconds.

        Returns
        -------
        ShakerMakerData
            New instance whose ``time`` / ``gf_time`` are rebuilt at the
            target ``dt``. Subsequent per-node reads are linearly
            interpolated on demand.

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.window_service.resample`.
        """
        from .window_service import resample

        return resample(self, dt)

    def apply_filter(self, mode="all", freqmin=0.25, freqmax=10.0,
                     corners=4, zerophase=True, apply_gf=False):
        """Return a *lazy* band-pass filtered copy of this model.

        Parameters
        ----------
        mode : {'all', 'vel', 'accel', 'disp'}, default ``'all'``
            ``'all'`` filters each ``data_type`` independently from disk.
            ``'vel'`` / ``'accel'`` / ``'disp'`` reads only that base type,
            filters it and derives the other two by integration /
            differentiation.
        freqmin, freqmax : float
            Band edges in Hz. Validated against the Nyquist frequency.
        corners : int, default ``4``
        zerophase : bool, default ``True``
        apply_gf : bool, default ``False``
            If ``True``, Green Function tensors are also filtered on read.

        Returns
        -------
        ShakerMakerData
            New instance with ``_filter`` set. No signal data is read at
            this point; ObsPy is called lazily inside ``get_node_data`` /
            ``get_qa_data`` (and optionally ``get_gf``).

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.core.filter_service.apply_filter`.
        Composable with :meth:`get_window` and :meth:`resample`.
        """
        from .filter_service import apply_filter

        return apply_filter(self, mode, freqmin, freqmax, corners, zerophase,
                            apply_gf)

    # -- plotting trampolines ----------------------------------------
    # All ``plot_*`` and ``create_animation*`` methods forward to the
    # standalone helpers under ``plotting.single_model.*``. We import them
    # inside the method so a user who never plots never pays the matplotlib
    # import cost.

    def plot_domain(self, **kwargs):
        """Plot the 3-D node domain. See :func:`plotting.single_model.domain_plots.plot_domain`."""
        from ..plotting.single_model.domain_plots import plot_domain

        return plot_domain(self, **kwargs)

    def plot_domain_calculated_t0(self, **kwargs):
        """Plot domain nodes coloured by GF ``t0``. See :func:`plotting.single_model.domain_plots.plot_domain_calculated_t0`."""
        from ..plotting.single_model.domain_plots import plot_domain_calculated_t0

        return plot_domain_calculated_t0(self, **kwargs)

    def plot_node_response(self, node_id=None, **kwargs):
        """Plot time histories for one or more nodes. See :func:`plotting.single_model.node_plots.plot_node_response`."""
        from ..plotting.single_model.node_plots import plot_node_response

        return plot_node_response(self, node_id=node_id, **kwargs)

    def plot_node_gf(self, node_id=None, **kwargs):
        """Plot Green-Function traces for one or more nodes. See :func:`plotting.single_model.node_plots.plot_node_gf`."""
        from ..plotting.single_model.node_plots import plot_node_gf

        return plot_node_gf(self, node_id=node_id, **kwargs)

    def plot_node_tensor_gf(self, node_id=None, **kwargs):
        """Plot the 9-component GF tensor. See :func:`plotting.single_model.node_plots.plot_node_tensor_gf`."""
        from ..plotting.single_model.node_plots import plot_node_tensor_gf

        return plot_node_tensor_gf(self, node_id=node_id, **kwargs)

    def plot_node_newmark(self, node_id=None, **kwargs):
        """Plot Newmark response spectra for one or more nodes. See :func:`plotting.single_model.node_plots.plot_node_newmark`."""
        from ..plotting.single_model.node_plots import plot_node_newmark

        return plot_node_newmark(self, node_id=node_id, **kwargs)

    def plot_calculated_vs_reused(self, **kwargs):
        """Show which nodes are computed donors vs reused receivers. See :func:`plotting.single_model.domain_plots.plot_calculated_vs_reused`."""
        from ..plotting.single_model.domain_plots import plot_calculated_vs_reused

        return plot_calculated_vs_reused(self, **kwargs)

    def plot_gf_connections(self, **kwargs):
        """Visualise donor-receiver GF links for one node. See :func:`plotting.single_model.domain_plots.plot_gf_connections`."""
        from ..plotting.single_model.domain_plots import plot_gf_connections

        return plot_gf_connections(self, **kwargs)

    def plot_surface(self, **kwargs):
        """3-D scatter snapshot at a given time. See :func:`plotting.single_model.surface_plots.plot_surface`."""
        from ..plotting.single_model.surface_plots import plot_surface

        return plot_surface(self, **kwargs)

    def create_animation(self, **kwargs):
        """Render a full-domain 3-D scatter animation. See :func:`plotting.single_model.animation_plots.create_animation`."""
        from ..plotting.single_model.animation_plots import create_animation

        return create_animation(self, **kwargs)

    def create_animation_plane(self, **kwargs):
        """Render an animation of a planar slice through the domain. See :func:`plotting.single_model.animation_plots.create_animation_plane`."""
        from ..plotting.single_model.animation_plots import create_animation_plane

        return create_animation_plane(self, **kwargs)

    def plot_node_arias(self, node_id=None, **kwargs):
        """Plot Arias intensity curves per node. See :func:`plotting.single_model.node_plots.plot_node_arias`."""
        from ..plotting.single_model.node_plots import plot_node_arias

        return plot_node_arias(self, node_id=node_id, **kwargs)

    def plot_surface_newmark(self, **kwargs):
        """Spectral map over the surface at a given period. See :func:`plotting.single_model.surface_plots.plot_surface_newmark`."""
        from ..plotting.single_model.surface_plots import plot_surface_newmark

        return plot_surface_newmark(self, **kwargs)

    def plot_surface_arias(self, **kwargs):
        """Arias-intensity map over the surface. See :func:`plotting.single_model.surface_plots.plot_surface_arias`."""
        from ..plotting.single_model.surface_plots import plot_surface_arias

        return plot_surface_arias(self, **kwargs)

    def write_h5drm(self, name=None):
        """Write the current time window back out to an HDF5 file.

        Parameters
        ----------
        name : str, optional
            Output filename. Defaults to ``<orig_stem>_t<start>_<end>.h5drm``
            next to the original file.

        Returns
        -------
        str
            Absolute path to the written file.

        Notes
        -----
        Delegates to :func:`ShakerMakerResults.io.export_service.write_h5drm`.
        """
        from ..io.export_service import write_h5drm

        return write_h5drm(self, name=name)

    def plot_surface_on_map(self, mapa, **kwargs):
        """Overlay a time snapshot on a folium map. See :func:`plotting.single_model.map_plots.plot_surface_on_map`."""
        from ..plotting.single_model.map_plots import plot_surface_on_map

        return plot_surface_on_map(self, mapa=mapa, **kwargs)

    def create_animation_map(self, **kwargs):
        """Render an animation on a tile basemap. See :func:`plotting.single_model.map_plots.create_animation_map`."""
        from ..plotting.single_model.map_plots import create_animation_map

        return create_animation_map(self, **kwargs)


# DRMData / SurfaceData are *not* subclasses -- they are aliases of the same
# class. The runtime layout (DRM box vs SurfaceGrid) is detected inside the
# constructor, so notebooks can pick whichever alias reads best.
DRMData = ShakerMakerData
SurfaceData = ShakerMakerData
