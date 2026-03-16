"""
shakermaker_data.py
===================
Unified reader for ShakerMaker HDF5 output files.

Supports both DRM (Domain Reduction Method) outputs written by
DRMHDF5StationListWriter and station outputs written by
HDF5StationListWriter. The format is detected automatically from the
HDF5 file structure.

HDF5 layouts recognised
------------------------
DRM layout  (DRMHDF5StationListWriter):
    /DRM_Data/{xyz, internal, velocity, acceleration, displacement}
    /DRM_QA_Data/{xyz, velocity, acceleration, displacement}
    /DRM_Metadata/{dt, tstart, tend, name, ...}
    /GF/sta_N/sub_M/{z, e, n, t, tdata, t0}          (optional)
    /GF_Database_Info/...                             (optional)
    /Node_Mapping/...                                 (optional)

Station layout  (HDF5StationListWriter):
    /Data/{xyz, internal, velocity, acceleration, displacement}
    /Metadata/{dt, tstart, tend, name, ...}

Author: Patricio Palacios B.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d

from .newmark import NewmarkSpectrumAnalyzer

# ---------------------------------------------------------------------------
# Rotation matrix shared by all plotting methods (ShakerMaker convention)
# ---------------------------------------------------------------------------
_R = np.column_stack([
    np.array([0, 1, 0]),
    np.array([1, 0, 0]),
    np.cross(np.array([0, 1, 0]), np.array([1, 0, 0]))
])


def _rotate(xyz_km):
    """Apply the ShakerMaker → display rotation and convert km → m."""
    return xyz_km * 1000 @ _R


class ShakerMakerData:
    """Unified reader for ShakerMaker HDF5 output files.

    Automatically detects whether the file is a DRM output
    (``/DRM_Data`` group present) or a station output (``/Data`` group).

    Parameters
    ----------
    filename : str
        Path to the HDF5 file produced by ShakerMaker.
    dt : float, optional
        If provided, time series will be resampled to this time step.
        Defaults to the original ``dt`` stored in the file.

    Attributes
    ----------
    filename : str
    is_drm : bool
        ``True`` when the file contains DRM data (internal/external nodes,
        QA station).  ``False`` for plain station output.
    xyz : np.ndarray, shape (N, 3)
        Node coordinates in km.
    internal : np.ndarray, shape (N,), dtype bool
        ``True`` for interior DRM nodes. All ``False`` for station files.
    xyz_qa : np.ndarray, shape (1, 3) or None
        QA station coordinate. ``None`` for station files.
    xyz_all : np.ndarray
        ``np.vstack([xyz, xyz_qa])`` for DRM; same as ``xyz`` otherwise.
    time : np.ndarray
        Output time vector in seconds.
    dt : float
        Time step of the output time vector.
    name : str
        Simulation name stored in the file metadata.
    model_name : str
        Short label derived from node spacing (e.g. ``"5.0m"``).
    spacing : tuple of float
        ``(h_x, h_y, h_z)`` in metres.

    Examples
    --------
    >>> result = ShakerMakerData("DRM_5m_H1_s0.h5drm")
    >>> data = result.get_node_data(0, 'accel')   # shape (3, Nt)
    >>> result.plot_node_response(node_id=0)

    >>> station = ShakerMakerData("station_output.h5")
    >>> station.plot_node_response(node_id=0)
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, filename, dt=None):
        self.filename = filename

        with h5py.File(filename, 'r') as f:
            # Detect format
            self.is_drm = 'DRM_Data' in f

            if self.is_drm:
                data_grp  = 'DRM_Data'
                meta_grp  = 'DRM_Metadata'
                qa_grp    = 'DRM_QA_Data'
            else:
                data_grp  = 'Data'
                meta_grp  = 'Metadata'
                qa_grp    = None

            # Geometry
            self.xyz      = f[f'{data_grp}/xyz'][:]
            self.internal = f[f'{data_grp}/internal'][:]

            if qa_grp and f'{qa_grp}/xyz' in f:
                self.xyz_qa = f[f'{qa_grp}/xyz'][:]
            else:
                self.xyz_qa = None

            # Metadata
            dt_orig    = float(f[f'{meta_grp}/dt'][()])
            tstart     = float(f[f'{meta_grp}/tstart'][()])

            try:
                raw_name = f[f'{meta_grp}/name'][()]
                self.name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            except KeyError:
                self.name = filename

            # Sizes
            n_nodes      = len(self.xyz)
            n_time_data  = f[f'{data_grp}/velocity'].shape[1]

            # Optional GF info
            self.freqs = None
            if 'GF_Spectrum/sta_0/sub_0/freqs' in f:
                self.freqs = f['GF_Spectrum/sta_0/sub_0/freqs'][:]

            n_subfaults = 0
            n_time_gf   = 0
            if 'GF/sta_0' in f:
                n_subfaults = len(list(f['GF/sta_0'].keys()))
                n_time_gf   = len(f['GF/sta_0/sub_0/z'][:])
            elif 'GF_Spectrum/sta_0' in f:
                n_subfaults = len([k for k in f['GF_Spectrum/sta_0'].keys()
                                   if k.startswith('sub_')])

        # xyz_all
        if self.xyz_qa is not None:
            self.xyz_all = np.vstack([self.xyz, self.xyz_qa])
        else:
            self.xyz_all = self.xyz

        # Spacing and model_name
        xyz_t    = _rotate(self.xyz)
        h_x = np.diff(np.sort(np.unique(np.round(xyz_t[:, 0], 6))))[0]
        h_y = np.diff(np.sort(np.unique(np.round(xyz_t[:, 1], 6))))[0]
        h_z = np.diff(np.sort(np.unique(np.round(xyz_t[:, 2], 6))))[0]
        self.spacing    = (h_x, h_y, h_z)
        self.model_name = f"{h_x:.1f}m"

        # Store private metadata
        self._dt_orig      = dt_orig
        self._tstart       = tstart
        self._n_nodes      = n_nodes
        self._n_subfaults  = n_subfaults
        self._n_time_gf    = n_time_gf
        self._n_time_data  = n_time_data
        self._data_grp     = data_grp
        self._meta_grp     = meta_grp
        self._qa_grp       = qa_grp

        # Caches
        self._node_cache     = {}
        self._gf_cache       = {}
        self._spectrum_cache = {}

        # GF Database Info
        with h5py.File(filename, 'r') as f:
            if 'GF_Database_Info/pairs_to_compute' in f:
                self.gf_db_pairs     = f['GF_Database_Info/pairs_to_compute'][:]
                self.gf_db_dh        = f['GF_Database_Info/dh_of_pairs'][:]
                self.gf_db_zrec      = f['GF_Database_Info/zrec_of_pairs'][:]
                self.gf_db_zsrc      = f['GF_Database_Info/zsrc_of_pairs'][:]
                self.gf_db_delta_h   = f['GF_Database_Info'].attrs['delta_h']
                self.gf_db_delta_v_rec = f['GF_Database_Info'].attrs['delta_v_rec']
                self.gf_db_delta_v_src = f['GF_Database_Info'].attrs['delta_v_src']

                unique = np.unique(self.gf_db_pairs[:, 0])
                pct    = (1 - len(unique) / n_nodes) * 100
                print(f"  GF DB: {len(unique)}/{n_nodes} computed ({pct:.1f}% reduction)")
            else:
                self.gf_db_pairs = None

            # Node mapping
            if 'Node_Mapping/node_to_pair_mapping' in f:
                self.node_mapping  = f['Node_Mapping/node_to_pair_mapping'][:]
                self.pairs_mapping = f['Node_Mapping/pairs_to_compute'][:]
            else:
                self.node_mapping  = None
                self.pairs_mapping = None

        # Time vectors
        if dt is None:
            self.dt       = dt_orig
            self.time     = np.arange(n_time_data) * dt_orig + tstart
            self.gf_time  = np.arange(n_time_gf)  * dt_orig
        else:
            self.dt       = dt
            time_orig     = np.arange(n_time_data) * dt_orig + tstart
            gf_orig       = np.arange(n_time_gf)   * dt_orig
            self.time     = np.arange(time_orig[0], time_orig[-1], dt)
            self.gf_time  = np.arange(gf_orig[0],  gf_orig[-1],   dt)
            self._resample_cache = {'time_orig': time_orig, 'gf_time_orig': gf_orig}

        print("----------------------------------------")
        print(f"{filename} loaded ({'DRM' if self.is_drm else 'Station'})")
        print(f"  nodes={n_nodes}  dt={dt_orig}s  "
              f"spacing={h_x:.1f}m x {h_y:.1f}m x {h_z:.1f}m")
        print(f"  time steps={n_time_data}  GF steps={n_time_gf}")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_node_data(self, node_id, data_type='accel'):
        """Return the signal array for a single node.

        Parameters
        ----------
        node_id : int
            Zero-based node index.
        data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
            Signal type to retrieve.

        Returns
        -------
        np.ndarray, shape (3, Nt)
            Rows are [E, N, Z] components (ShakerMaker convention).
        """
        key = (node_id, data_type)
        if key not in self._node_cache:
            idx  = 3 * node_id
            path = {
                'accel': f'{self._data_grp}/acceleration',
                'vel':   f'{self._data_grp}/velocity',
                'disp':  f'{self._data_grp}/displacement',
            }[data_type]

            with h5py.File(self.filename, 'r') as f:
                data = f[path][idx:idx + 3, :]

            if hasattr(self, '_window_mask'):
                data = data[:, self._window_mask]
            elif hasattr(self, '_resample_cache'):
                resampled = np.zeros((3, len(self.time)))
                for i in range(3):
                    resampled[i] = interp1d(
                        self._resample_cache['time_orig'], data[i],
                        kind='linear', fill_value='extrapolate')(self.time)
                data = resampled

            self._node_cache[key] = data
        return self._node_cache[key]

    def get_qa_data(self, data_type='accel'):
        """Return the signal array for the QA station (DRM files only).

        Parameters
        ----------
        data_type : {'accel', 'vel', 'disp'}, default ``'accel'``

        Returns
        -------
        np.ndarray, shape (3, Nt)

        Raises
        ------
        AttributeError
            If the file is not a DRM output.
        """
        if self._qa_grp is None:
            raise AttributeError("QA station only available in DRM output files.")

        key = ('qa', data_type)
        if key not in self._node_cache:
            path = {
                'accel': f'{self._qa_grp}/acceleration',
                'vel':   f'{self._qa_grp}/velocity',
                'disp':  f'{self._qa_grp}/displacement',
            }[data_type]

            with h5py.File(self.filename, 'r') as f:
                data = f[path][:]

            if hasattr(self, '_window_mask'):
                data = data[:, self._window_mask]
            elif hasattr(self, '_resample_cache'):
                resampled = np.zeros((3, len(self.time)))
                for i in range(3):
                    resampled[i] = interp1d(
                        self._resample_cache['time_orig'], data[i],
                        kind='linear', fill_value='extrapolate')(self.time)
                data = resampled

            self._node_cache[key] = data
        return self._node_cache[key]

    def get_gf(self, node_id, subfault_id, component='z'):
        """Return the Green's function time series for a node/subfault pair.

        Parameters
        ----------
        node_id : int
        subfault_id : int
        component : {'z', 'e', 'n'}, default ``'z'``

        Returns
        -------
        np.ndarray, shape (Nt_gf,)
        """
        key = (node_id, subfault_id, component)
        if key not in self._gf_cache:
            if self.node_mapping is not None:
                mask = ((self.node_mapping[:, 0] == node_id) &
                        (self.node_mapping[:, 1] == subfault_id))
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    raise KeyError(f"Node {node_id}, subfault {subfault_id} not in mapping")
                ipair  = self.node_mapping[idx[0], 2]
                src_n, src_s = self.pairs_mapping[ipair]
                if src_n != node_id:
                    print(f"  Node {node_id}/sub {subfault_id} → donor node {src_n}")
            else:
                src_n, src_s = node_id, subfault_id

            path = f'GF/sta_{src_n}/sub_{src_s}/{component}'
            with h5py.File(self.filename, 'r') as f:
                if path not in f:
                    raise KeyError(f"GF not found: {path}")
                self._gf_cache[key] = f[path][:]

        return self._gf_cache[key]

    def get_spectrum(self, node_id, subfault_id, component='z', part='real'):
        """Return a GF frequency-domain spectrum component.

        Parameters
        ----------
        node_id : int
        subfault_id : int
        component : {'z', 'e', 'n'}
        part : {'real', 'imag'}

        Returns
        -------
        np.ndarray
        """
        key = (node_id, subfault_id, component, part)
        if key not in self._spectrum_cache:
            if self.node_mapping is not None:
                pair_idx = self.node_mapping[node_id, subfault_id]
                if pair_idx == -1:
                    raise KeyError(f"Node {node_id}, subfault {subfault_id} not computed.")
                src_n, src_s = self.pairs_mapping[pair_idx]
            else:
                src_n, src_s = node_id, subfault_id

            path = (f'GF_Spectrum/sta_{src_n}/sub_{src_s}/'
                    f'spectrum_{component}_{part}')
            with h5py.File(self.filename, 'r') as f:
                self._spectrum_cache[key] = f[path][:]

        return self._spectrum_cache[key]

    def clear_cache(self):
        """Release all in-memory cached data and run garbage collection."""
        import gc
        self._node_cache.clear()
        self._gf_cache.clear()
        self._spectrum_cache.clear()
        gc.collect()
        print("Cache cleared.")

    # ------------------------------------------------------------------
    # Windowing / resampling
    # ------------------------------------------------------------------

    def get_window(self, t_start, t_end):
        """Return a time-windowed copy of this object (no data is loaded).

        Parameters
        ----------
        t_start : float
            Window start time in seconds.
        t_end : float
            Window end time in seconds.

        Returns
        -------
        ShakerMakerData
            New object with ``time`` restricted to ``[t_start, t_end]``.
        """
        new = ShakerMakerData.__new__(ShakerMakerData)
        # Copy all attributes
        for attr, val in self.__dict__.items():
            setattr(new, attr, val)

        mask                = (self.time >= t_start) & (self.time <= t_end)
        new._window_mask    = mask
        new._n_time_data    = int(np.sum(mask))
        new.time            = self.time[mask]
        new._node_cache     = {}
        new._gf_cache       = {}
        new._spectrum_cache = {}
        new.name            = f"{self.name} [{t_start}-{t_end}s]"
        print(f"Window [{t_start}, {t_end}]s → {new._n_time_data} samples")
        return new

    def resample(self, dt):
        """Return a copy configured to resample all data to a new time step.

        No data is loaded until ``get_node_data`` / ``get_qa_data`` is called.

        Parameters
        ----------
        dt : float
            Target time step in seconds.

        Returns
        -------
        ShakerMakerData
        """
        new = ShakerMakerData.__new__(ShakerMakerData)
        for attr, val in self.__dict__.items():
            setattr(new, attr, val)

        time_orig      = np.arange(self._n_time_data) * self._dt_orig + self._tstart
        gf_orig        = np.arange(self._n_time_gf)   * self._dt_orig
        new.dt         = dt
        new.time       = np.arange(time_orig[0], time_orig[-1], dt)
        new.gf_time    = np.arange(gf_orig[0],  gf_orig[-1],   dt)
        new._resample_cache = {'time_orig': time_orig, 'gf_time_orig': gf_orig}
        new._node_cache     = {}
        new._gf_cache       = {}
        new._spectrum_cache = {}
        print(f"Resampled: {len(new.time)} time steps at dt={dt}s")
        return new

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_node(self, node_id, data_type):
        """Return (data, label) for a node_id that may be int or 'QA'."""
        if node_id in ('QA', 'qa') or (
                isinstance(node_id, int) and node_id >= len(self.xyz)):
            return self.get_qa_data(data_type), 'QA'
        data = self.get_node_data(node_id, data_type)
        return data, f'Node {node_id}'

    @staticmethod
    def _setup_3comp_axes(ylabel, title_prefix, xlim, components=('Z', 'E', 'N')):
        fig = plt.figure(figsize=(8, 8))
        for k, comp in enumerate(components, 1):
            ax = plt.subplot(3, 1, k)
            ax.set_title(f'{comp} - {title_prefix}', fontweight='bold')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if xlim:
                ax.set_xlim(xlim)
        return fig

    @staticmethod
    def _build_cube_faces(xyz_int):
        """Return corner array and faces list for a bounding-box cube."""
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
        return corners, faces, (x_min, x_max, y_min, y_max, z_min, z_max)

    def _label_nodes_on_ax(self, ax, xyz_t, bounds_int, label_nodes, comp_donors=None):
        """Add node-ID text labels to a 3D axes according to ``label_nodes``."""
        x_min, x_max, y_min, y_max, z_min, z_max = bounds_int
        x_min_e, x_max_e = xyz_t[:, 0].min(), xyz_t[:, 0].max()
        y_min_e, y_max_e = xyz_t[:, 1].min(), xyz_t[:, 1].max()
        z_min_e, z_max_e = xyz_t[:, 2].min(), xyz_t[:, 2].max()

        def on_edge_int(x, y, z, n=2):
            return sum([abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3,
                        abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3,
                        abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3]) >= n

        def on_edge_ext(x, y, z, n=2):
            return sum([abs(x - x_min_e) < 1e-3 or abs(x - x_max_e) < 1e-3,
                        abs(y - y_min_e) < 1e-3 or abs(y - y_max_e) < 1e-3,
                        abs(z - z_min_e) < 1e-3 or abs(z - z_max_e) < 1e-3]) >= n

        for i in range(len(xyz_t)):
            x, y, z = xyz_t[i]
            color = 'darkred' if self.internal[i] else 'darkblue'

            if label_nodes is True:
                ax.text(x, y, z, str(i), fontsize=8, color=color)
            elif label_nodes == 'corners':
                if on_edge_int(x, y, z, 3) or on_edge_ext(x, y, z, 3):
                    ax.text(x, y, z, str(i), fontsize=8, color=color, fontweight='bold')
            elif label_nodes == 'corners_edges':
                if on_edge_int(x, y, z) or on_edge_ext(x, y, z):
                    ax.text(x, y, z, str(i), fontsize=9, color=color)
            elif label_nodes == 'corners_half':
                x_mid = (x_min + x_max) / 2
                y_mid = (y_min + y_max) / 2
                z_mid = (z_min + z_max) / 2
                is_corner = on_edge_int(x, y, z, 3)
                is_mid = any([
                    on_edge_int(x, y, z_mid, 2) and abs(z - z_mid) < 1e-3,
                    on_edge_int(x, y_mid, z, 2) and abs(y - y_mid) < 1e-3,
                    on_edge_int(x_mid, y, z, 2) and abs(x - x_mid) < 1e-3,
                ])
                if is_corner or is_mid:
                    ax.text(x, y, z, str(i), fontsize=9, color=color)
            elif label_nodes == 'calculated' and comp_donors is not None:
                if i in comp_donors:
                    ax.text(x, y, z, str(i), fontsize=8, color=color)

    # ------------------------------------------------------------------
    # Plotting — single-object methods
    # ------------------------------------------------------------------

    def plot_domain(self, xyz_origin=None, label_nodes=False,
                    show_calculated=False):
        """Plot the 3-D node domain.

        Parameters
        ----------
        xyz_origin : array-like (3,), optional
            Translate the domain so that the QA station coincides with
            this point (in metres).
        label_nodes : bool or str, optional
            ``False`` — no labels.
            ``True`` — all nodes.
            ``'corners'``, ``'corners_edges'``, ``'corners_half'`` — subsets.
            ``'calculated'`` — only GF-donor nodes.
        show_calculated : bool, default ``False``
            Highlight GF-donor nodes with distinct colours.

        Returns
        -------
        fig, ax
        """
        xyz_t    = _rotate(self.xyz)
        xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None

        if xyz_origin is not None and xyz_qa_t is not None:
            translation = np.asarray(xyz_origin) - xyz_qa_t[0]
            xyz_t    += translation
            xyz_qa_t += translation

        xyz_int = xyz_t[self.internal]
        xyz_ext = xyz_t[~self.internal]
        _, faces, bounds = self._build_cube_faces(xyz_int)

        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection='3d')

        comp_donors = None
        if show_calculated and self.gf_db_pairs is not None:
            comp_donors = set(np.unique(self.gf_db_pairs[:, 0]))
            calc_int = np.isin(np.where(self.internal)[0],  list(comp_donors))
            calc_ext = np.isin(np.where(~self.internal)[0], list(comp_donors))
            ax.scatter(xyz_ext[~calc_ext, 0], xyz_ext[~calc_ext, 1],
                       xyz_ext[~calc_ext, 2], c='lightblue', s=30, alpha=0.3)
            ax.scatter(xyz_int[~calc_int, 0], xyz_int[~calc_int, 1],
                       xyz_int[~calc_int, 2], c='pink', s=20, alpha=0.3)
            if calc_ext.any():
                ax.scatter(xyz_ext[calc_ext, 0], xyz_ext[calc_ext, 1],
                           xyz_ext[calc_ext, 2], c='blue', s=50, alpha=0.5)
            if calc_int.any():
                ax.scatter(xyz_int[calc_int, 0], xyz_int[calc_int, 1],
                           xyz_int[calc_int, 2], c='red', s=30, alpha=0.5)
        else:
            ax.scatter(xyz_ext[:, 0], xyz_ext[:, 1], xyz_ext[:, 2],
                       c='blue', marker='o', s=50, alpha=0.1)
            ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2],
                       c='red', marker='s', s=30, alpha=0.4)

        if xyz_qa_t is not None:
            ax.scatter(xyz_qa_t[:, 0], xyz_qa_t[:, 1], xyz_qa_t[:, 2],
                       c='green', marker='*', s=300, label='QA', zorder=10,
                       edgecolors='black', linewidths=2)

        cube = Poly3DCollection(faces, alpha=0.15, facecolor='red',
                                edgecolor='darkred', linewidths=1.5)
        ax.add_collection3d(cube)

        if label_nodes:
            self._label_nodes_on_ax(ax, xyz_t, bounds, label_nodes, comp_donors)

        ax.set_xlabel("X' (km)")
        ax.set_ylabel("Y' (km)")
        ax.set_zlabel("Z' (km)")
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        plt.show()
        if xyz_qa_t is not None:
            print(f"QA position: {xyz_qa_t[0]}")
        return fig, ax

    def plot_node_response(self, node_id=None, target_pos=None,
                           xlim=None, data_type='vel'):
        """Plot the time-history response for one or more nodes.

        Parameters
        ----------
        node_id : int, list, or 'QA', optional
            Node index or list of indices. ``'QA'`` for the QA station.
        target_pos : array-like (3,), optional
            Find the nearest node to this km-coordinate.
        xlim : list, optional
            Time axis limits ``[tmin, tmax]``.
        data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
        """
        node_ids = self._collect_node_ids(node_id, target_pos)
        ylabel   = {'accel': 'Acceleration', 'vel': 'Velocity',
                    'disp': 'Displacement'}[data_type]

        fig = plt.figure(figsize=(8, 8))
        for nid in node_ids:
            data, label = self._resolve_node(nid, data_type)
            for k, comp in enumerate(('Z', 'E', 'N'), 1):
                plt.subplot(3, 1, k)
                plt.plot(self.time, data[k - 1], linewidth=1, label=label)

        for k, comp in enumerate(('Vertical (Z)', 'East (E)', 'North (N)'), 1):
            ax = plt.subplot(3, 1, k)
            ax.set_title(f'{comp} — {ylabel}', fontweight='bold')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left')
            if xlim:
                ax.set_xlim(xlim)

        plt.tight_layout()
        plt.show()

    def plot_node_gf(self, node_id=None, target_pos=None,
                     xlim=None, subfault=0):
        """Plot Green's function time series for one or more nodes.

        Parameters
        ----------
        node_id : int, list, or 'QA', optional
        target_pos : array-like (3,), optional
        xlim : list, optional
        subfault : int or list, default ``0``
        """
        node_ids   = self._collect_node_ids(node_id, target_pos)
        sub_ids    = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]

        fig = plt.figure(figsize=(8, 10))
        for nid in node_ids:
            if nid in ('QA', 'qa'):
                print("GFs not available for QA node.")
                continue
            for sid in sub_ids:
                label = f'N{nid}_S{sid}'
                for k, comp in enumerate(('z', 'e', 'n'), 1):
                    plt.subplot(3, 1, k)
                    plt.plot(self.gf_time, self.get_gf(nid, sid, comp),
                             linewidth=1, label=label)

        titles = ('Vertical (Z)', 'East (E)', 'North (N)')
        for k, title in enumerate(titles, 1):
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

    def plot_node_tensor_gf(self, node_id=None, target_pos=None,
                            xlim=None, subfault=0):
        """Plot the 9-component tensor Green's functions.

        Parameters
        ----------
        node_id : int, list, optional
        target_pos : array-like (3,), optional
        xlim : list, optional
        subfault : int or list, default ``0``
        """
        node_ids = self._collect_node_ids(node_id, target_pos)
        sub_ids  = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
        labels   = [f'G_{i+1}{j+1}' for i in range(3) for j in range(3)]

        fig, axes = plt.subplots(3, 3, figsize=(10, 8))

        for nid in node_ids:
            if nid in ('QA', 'qa'):
                print("Tensor GFs not available for QA.")
                continue
            for sid in sub_ids:
                if self.node_mapping is not None:
                    mask  = ((self.node_mapping[:, 0] == nid) &
                             (self.node_mapping[:, 1] == sid))
                    idx   = np.where(mask)[0]
                    if not len(idx):
                        continue
                    ipair = self.node_mapping[idx[0], 2]
                    src_n, src_s = self.pairs_mapping[ipair]
                else:
                    src_n, src_s = nid, sid

                with h5py.File(self.filename, 'r') as f:
                    path = f'GF/sta_{src_n}/sub_{src_s}/tdata'
                    if path not in f:
                        continue
                    tdata = f[path][:]
                    t0    = f[f'GF/sta_{src_n}/sub_{src_s}/t0'][()]

                time  = np.arange(tdata.shape[0]) * self._dt_orig + t0
                label = f'N{nid}_S{sid}'
                for j in range(9):
                    axes[j // 3, j % 3].plot(time, tdata[:, j],
                                             linewidth=0.8, label=label)

        for j, lbl in enumerate(labels):
            ax = axes[j // 3, j % 3]
            ax.set_title(lbl, fontsize=11, fontweight='bold')
            ax.set_xlabel('Time [s]', fontsize=9)
            ax.set_ylabel('Amplitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            if xlim:
                ax.set_xlim(xlim)

        axes[0, 0].legend(fontsize=8)
        plt.suptitle('Tensor Green Functions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_node_f_spectrum(self, node_id=None, target_pos=None,
                             xlim=None, subfault=0):
        """Plot the Fourier magnitude spectrum for one or more nodes.

        Parameters
        ----------
        node_id : int, list, optional
        target_pos : array-like (3,), optional
        xlim : list, optional
        subfault : int or list, default ``0``
        """
        node_ids = self._collect_node_ids(node_id, target_pos)
        sub_ids  = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]

        fig = plt.figure(figsize=(8, 10))

        for nid in node_ids:
            if nid in ('QA', 'qa'):
                print("Fourier spectrum not available for QA node.")
                continue
            for sid in sub_ids:
                try:
                    mag_z = np.sqrt(self.get_spectrum(nid, sid, 'z', 'real') ** 2 +
                                    self.get_spectrum(nid, sid, 'z', 'imag') ** 2)
                    mag_e = np.sqrt(self.get_spectrum(nid, sid, 'e', 'real') ** 2 +
                                    self.get_spectrum(nid, sid, 'e', 'imag') ** 2)
                    mag_n = np.sqrt(self.get_spectrum(nid, sid, 'n', 'real') ** 2 +
                                    self.get_spectrum(nid, sid, 'n', 'imag') ** 2)
                    label = f'N{nid}_S{sid}'
                    for k, mag in enumerate((mag_z, mag_e, mag_n), 1):
                        plt.subplot(3, 1, k)
                        plt.loglog(self.freqs, mag, linewidth=1, label=label)
                except KeyError:
                    print(f"  ! No spectrum for node {nid}, subfault {sid}")

        for k, comp in enumerate(('Vertical (Z)', 'East (E)', 'North (N)'), 1):
            ax = plt.subplot(3, 1, k)
            ax.set_title(f'{comp} — Fourier Spectrum', fontweight='bold')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Magnitude (log scale)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            if xlim:
                ax.set_xlim(xlim)

        plt.tight_layout()
        plt.show()

    def plot_node_newmark(self, node_id=None, target_pos=None,
                          xlim=None, data_type='accel'):
        """Plot Newmark response spectra for one or more nodes.

        Parameters
        ----------
        node_id : int, list, or 'QA', optional
        target_pos : array-like (3,), optional
        xlim : list, default ``[0, 5]``
        data_type : {'accel', 'vel', 'disp'}, default ``'accel'``
        """
        if xlim is None:
            xlim = [0, 5]
        node_ids = self._collect_node_ids(node_id, target_pos)
        dt       = self.time[1] - self.time[0]
        scale    = 1.0 / 9.81 if data_type == 'accel' else 1.0
        ylabel   = 'Sa (g)' if data_type == 'accel' else 'Spectral Response'

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        for nid in node_ids:
            data, label = self._resolve_node(nid, data_type)
            specs = [NewmarkSpectrumAnalyzer.compute(data[i] * scale, dt)
                     for i in range(3)]
            T = specs[0]['T']
            for k, (ax, sp, comp) in enumerate(
                    zip(axes, specs, ('Vertical (Z)', 'X', 'Y'))):
                ax.plot(T, sp['PSa'], linewidth=2, label=label)

        for ax, comp in zip(axes, ('Vertical (Z)', 'X', 'Y')):
            ax.set_title(f'{comp} — Newmark Spectrum', fontweight='bold')
            ax.set_xlabel('T (s)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlim(xlim)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_calculated_vs_reused(self, db_filename=None,
                                  xyz_origin=None, label_nodes=False):
        """Visualise which nodes have computed GFs vs. reused from a donor.

        Parameters
        ----------
        db_filename : str, optional
            External GF database HDF5 file. If ``None``, the info is read
            from the current file's ``/GF_Database_Info`` group.
        xyz_origin : array-like (3,), optional
            Translate so the QA station sits at this point (metres).
        label_nodes : bool or str, optional
        """
        if db_filename is not None:
            with h5py.File(db_filename, 'r') as f:
                pairs_calc = f['pairs_to_compute'][:]
            unique_db = np.unique(pairs_calc[:, 0])
        elif self.gf_db_pairs is not None:
            unique_db = np.unique(self.gf_db_pairs[:, 0])
        else:
            print("No GF database info available.")
            return

        with h5py.File(self.filename, 'r') as f:
            unique_calc = (np.array([int(k.replace('sta_', ''))
                                     for k in f['GF'].keys()])
                           if 'GF' in f else np.array([]))

        xyz_t    = _rotate(self.xyz)
        xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None

        if xyz_origin is not None and xyz_qa_t is not None:
            translation = np.asarray(xyz_origin) - xyz_qa_t[0]
            xyz_t    += translation
            xyz_qa_t += translation

        xyz_int = xyz_t[self.internal]
        xyz_ext = xyz_t[~self.internal]
        int_idx = np.where(self.internal)[0]
        ext_idx = np.where(~self.internal)[0]

        c_int = np.isin(int_idx, unique_calc)
        c_ext = np.isin(ext_idx, unique_calc)

        _, faces, bounds = self._build_cube_faces(xyz_int)

        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection='3d')

        for data, mask, color, marker in [
            (xyz_ext, ~c_ext, 'lightblue', 'o'),
            (xyz_int, ~c_int, 'pink',      's'),
            (xyz_ext,  c_ext, 'blue',      'o'),
            (xyz_int,  c_int, 'red',       's'),
        ]:
            sub = data[mask]
            if len(sub):
                ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
                           c=color, marker=marker, alpha=0.5)

        if xyz_qa_t is not None:
            ax.scatter(*xyz_qa_t[0], c='green', marker='*', s=400,
                       label='QA', zorder=10, edgecolors='black', linewidths=2)

        ax.add_collection3d(
            Poly3DCollection(faces, alpha=0.1, facecolor='red',
                             edgecolor='darkred', linewidths=2))

        if label_nodes:
            self._label_nodes_on_ax(ax, xyz_t, bounds, label_nodes,
                                    set(unique_calc))

        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        n_int = len(int_idx); n_ext = len(ext_idx)
        print("=" * 60)
        print(f"Internal : {c_int.sum()}/{n_int} with GFs "
              f"({c_int.sum()/n_int*100:.1f}%)")
        print(f"Boundary : {c_ext.sum()}/{n_ext} with GFs "
              f"({c_ext.sum()/n_ext*100:.1f}%)")
        print(f"Total    : {len(unique_calc)}/{len(self.xyz)} "
              f"({len(unique_calc)/len(self.xyz)*100:.1f}%)")
        print("=" * 60)
        return fig, ax

    def plot_gf_connections(self, node_id, xyz_origin=None, label_nodes=False):
        """Visualise the donor–recipient GF connections for a single node.

        Parameters
        ----------
        node_id : int
            The node to analyse.
        xyz_origin : array-like (3,), optional
            Translate coordinate origin (metres).
        label_nodes : bool or str, optional
        """
        if self.node_mapping is None:
            print("No node mapping available.")
            return

        with h5py.File(self.filename, 'r') as f:
            if 'GF_Database_Info/pairs_to_compute' not in f:
                print("No GF Database info.")
                return
            pairs_db   = f['GF_Database_Info/pairs_to_compute'][:]
            comp_donors = set(np.unique(pairs_db[:, 0]))

        super_donors = set()
        for node in range(len(self.xyz)):
            mask  = ((self.node_mapping[:, 0] == node) &
                     (self.node_mapping[:, 1] == 0))
            idx   = np.where(mask)[0]
            if len(idx):
                donor = self.pairs_mapping[self.node_mapping[idx[0], 2], 0]
                if donor != node:
                    super_donors.add(donor)

        solitary_donors = comp_donors - super_donors

        if node_id in super_donors:
            recipients = [n for n in range(len(self.xyz))
                          if n != node_id and self._donor_of(n, 0) == node_id]
            donor_to_plot      = node_id
            recipients_to_plot = recipients
            print(f"Node {node_id}: SUPER DONOR → {len(recipients)} recipients")
        elif node_id in solitary_donors:
            donor_to_plot      = node_id
            recipients_to_plot = []
            print(f"Node {node_id}: SOLITARY DONOR")
        else:
            donor_to_plot      = self._donor_of(node_id, 0)
            recipients_to_plot = [node_id]
            print(f"Node {node_id}: RECEIVER ← donor {donor_to_plot}")

        xyz_t    = _rotate(self.xyz)
        xyz_qa_t = _rotate(self.xyz_qa) if self.xyz_qa is not None else None

        if xyz_origin is not None and xyz_qa_t is not None:
            translation = np.asarray(xyz_origin) - xyz_qa_t[0]
            xyz_t    += translation
            xyz_qa_t += translation

        xyz_int = xyz_t[self.internal]
        _, faces, bounds = self._build_cube_faces(xyz_int)

        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection='3d')

        ax.scatter(xyz_t[~self.internal, 0], xyz_t[~self.internal, 1],
                   xyz_t[~self.internal, 2], c='blue', s=50, alpha=0.1)
        ax.scatter(xyz_t[self.internal, 0], xyz_t[self.internal, 1],
                   xyz_t[self.internal, 2], c='red', s=30, alpha=0.3)

        dp = xyz_t[donor_to_plot]
        ax.scatter(*dp, c='red', marker='s', s=100,
                   edgecolors='darkred', linewidths=2, zorder=10, alpha=0.5)

        for rec in recipients_to_plot:
            rp = xyz_t[rec]
            ax.scatter(*rp, c='orange', marker='o', s=80,
                       edgecolors='darkorange', linewidths=1.5, alpha=0.5)
            ax.plot([dp[0], rp[0]], [dp[1], rp[1]], [dp[2], rp[2]],
                    color='darkorange', linestyle='--', alpha=0.5, linewidth=2)

        if xyz_qa_t is not None:
            ax.scatter(*xyz_qa_t[0], c='green', marker='*', s=300,
                       label='QA', zorder=10, edgecolors='black', linewidths=2)

        ax.add_collection3d(
            Poly3DCollection(faces, alpha=0.10, facecolor='red',
                             edgecolor='darkred', linewidths=1.5))

        if label_nodes:
            self._label_nodes_on_ax(ax, xyz_t, bounds, label_nodes, comp_donors)

        ax.set_xlabel("X' (km)"); ax.set_ylabel("Y' (km)"); ax.set_zlabel("Z' (km)")
        ax.legend(); ax.grid(False)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _donor_of(self, node_id, subfault_id):
        """Return the donor node index for a given node/subfault pair."""
        mask = ((self.node_mapping[:, 0] == node_id) &
                (self.node_mapping[:, 1] == subfault_id))
        idx  = np.where(mask)[0]
        if not len(idx):
            return node_id
        return int(self.pairs_mapping[self.node_mapping[idx[0], 2], 0])

    def _collect_node_ids(self, node_id, target_pos):
        """Normalise node_id / target_pos into a list of ids."""
        if node_id is not None:
            return node_id if isinstance(node_id, (list, np.ndarray)) else [node_id]
        if target_pos is not None:
            dist = np.linalg.norm(self.xyz_all - np.asarray(target_pos), axis=1)
            idx  = int(np.argmin(dist))
            print(f"Nearest node: {idx}  distance={dist[idx]:.6f} km")
            return [idx]
        raise ValueError("Provide node_id or target_pos.")
