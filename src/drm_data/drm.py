import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
from .newmark import NewmarkSpectrumAnalyzer



class DRM:
    def __init__(self, filename, dt=None, gf_database=None):
        with h5py.File(filename, 'r') as f:
            # Geometry
            self.xyz = f['DRM_Data/xyz'][:]
            self.internal = f['DRM_Data/internal'][:]
            self.xyz_qa = f['DRM_QA_Data/xyz'][:]
            # Metadata
            dt_orig = f['DRM_Metadata/dt'][()]
            self.tstart = f['DRM_Metadata/tstart'][()]
            self.name = f['DRM_Metadata/name'][()].decode()
            
            # Frequencies
            self.freqs = None
            if 'GF_Spectrum/sta_0/sub_0/freqs' in f:
                self.freqs = f['GF_Spectrum/sta_0/sub_0/freqs'][:]
            
            # Dimensions
            n_nodes = len(self.xyz)
            n_freqs = len(self.freqs) if self.freqs is not None else 0
            
            # Subfaults
            n_subfaults = 0
            n_time = 0
            if 'GF/sta_0' in f:
                n_subfaults = len(list(f['GF/sta_0'].keys()))
                n_time = len(f['GF/sta_0/sub_0/z'][:])
            elif 'GF_Spectrum/sta_0' in f:
                n_subfaults = len([k for k in f['GF_Spectrum/sta_0'].keys() if k.startswith('sub_')])
                n_time = 0 
            n_time_drm = f['DRM_Data/acceleration'].shape[1]

        self.filename = filename
        self.xyz_all = np.vstack([self.xyz, self.xyz_qa])
    
        # Calculate spacing for model name
        u_x = np.array([0, 1, 0])
        u_y = np.array([1, 0, 0])
        u_z = np.cross(u_x, u_y)
        R = np.column_stack([u_x, u_y, u_z])
        xyz_t = self.xyz * 1000 @ R
        xyz_int = xyz_t[self.internal]
        
        h_x = np.diff(np.sort(np.unique(np.round(xyz_t[:, 0], 6))))[0]
        h_y = np.diff(np.sort(np.unique(np.round(xyz_t[:, 1], 6))))[0]
        h_z = np.diff(np.sort(np.unique(np.round(xyz_t[:, 2], 6))))[0]
        
        dim_x = xyz_int[:, 0].max() - xyz_int[:, 0].min()
        dim_y = xyz_int[:, 1].max() - xyz_int[:, 1].min()
        dim_z = xyz_int[:, 2].max() - xyz_int[:, 2].min()
        
        nx = int(dim_x / h_x)
        ny = int(dim_y / h_y)
        nz = int(dim_z / h_z)

        self.spacing = (h_x, h_y, h_z)
        self.model_name = f"{h_x:.1f}m"

        print("----------------------------------------")
        print(f"{filename} loaded: {n_nodes} nodes, dt={dt_orig}s")
        print(f"spacing={h_x:.1f}m x {h_y:.1f}m x {h_z:.1f}m")
        print(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")
        print(f"time steps: {n_time_drm}")
        print(f"GF time steps: {n_time}")

        # Store metadata
        self._dt_orig = dt_orig
        self._n_nodes = n_nodes
        self._n_subfaults = n_subfaults
        self._n_time = n_time
        self._n_time_drm = n_time_drm
        self._n_freqs = n_freqs

        # Caches
        self._node_cache = {}
        self._gf_cache = {}
        self._spectrum_cache = {}

        # Load GF database info if available
        with h5py.File(filename, 'r') as f:
            if 'GF_Database_Info/pairs_to_compute' in f:
                self.gf_db_pairs = f['GF_Database_Info/pairs_to_compute'][:]
                self.gf_db_dh = f['GF_Database_Info/dh_of_pairs'][:]
                self.gf_db_zrec = f['GF_Database_Info/zrec_of_pairs'][:]
                self.gf_db_zsrc = f['GF_Database_Info/zsrc_of_pairs'][:]
                self.gf_db_delta_h = f['GF_Database_Info'].attrs['delta_h']
                self.gf_db_delta_v_rec = f['GF_Database_Info'].attrs['delta_v_rec']
                self.gf_db_delta_v_src = f['GF_Database_Info'].attrs['delta_v_src']
                
                unique_stations = np.unique(self.gf_db_pairs[:, 0])
                reduction_pct = (1 - len(unique_stations) / self._n_nodes) * 100
                
                print(f"GF Database Info loaded:")
                print(f"  Calculated nodes: {len(unique_stations)}/{self._n_nodes} ({reduction_pct:.1f}% reduction)")
                print(f"  Tolerances: dh={self.gf_db_delta_h*1000:.1f}m, dv_rec={self.gf_db_delta_v_rec*1000:.1f}m, dv_src={self.gf_db_delta_v_src*1000:.1f}m")
            else:
                self.gf_db_pairs = None
                print("No GF Database info (file created with standard run())")

        # Setup time
        if dt is None:
            self.dt = dt_orig
            self.time = np.arange(n_time_drm) * dt_orig + self.tstart
            self.gf_time = np.arange(n_time) * dt_orig
        else:
            self.dt = dt
            time_orig = np.arange(n_time_drm) * dt_orig + self.tstart
            gf_time_orig = np.arange(n_time) * dt_orig
            self.time = np.arange(time_orig[0], time_orig[-1], dt)
            self.gf_time = np.arange(gf_time_orig[0], gf_time_orig[-1], dt)
            self._needs_resample = True
            self._resample_dt = dt

        # Load node mapping if available
        with h5py.File(filename, 'r') as f:
            if 'Node_Mapping/node_to_pair_mapping' in f:
                self.node_mapping = f['Node_Mapping/node_to_pair_mapping'][:]
                self.pairs_mapping = f['Node_Mapping/pairs_to_compute'][:]
            else:
                self.node_mapping = None

    # ----------------------------------------------------------------
    
    def get_node_data(self, node_id, data_type='accel'):
        key = (node_id, data_type)
        
        if key not in self._node_cache:
            idx = 3 * node_id
            
            with h5py.File(self.filename, 'r') as f:
                if data_type == 'accel':
                    path = 'DRM_Data/acceleration'
                elif data_type == 'vel':
                    path = 'DRM_Data/velocity'
                else:
                    path = 'DRM_Data/displacement'
                
                data_orig = f[path][idx:idx+3, :]
            
            # Apply window mask if exists
            if hasattr(self, '_window_mask'):
                data_orig = data_orig[:, self._window_mask]
            # Apply resampling if exists
            elif hasattr(self, '_resample_cache'):
                data_resampled = np.zeros((3, len(self.time)))
                for i in range(3):
                    data_resampled[i] = interp1d(
                        self._resample_cache['time_orig'], 
                        data_orig[i], 
                        kind='linear', 
                        fill_value='extrapolate'
                    )(self.time)
                data_orig = data_resampled
            
            self._node_cache[key] = data_orig
        
        return self._node_cache[key]

    # ----------------------------------------------------------------
    
    def get_qa_data(self, data_type='accel'):
        key = ('qa', data_type)
        
        if key not in self._node_cache:
            with h5py.File(self.filename, 'r') as f:
                if data_type == 'accel':
                    path = 'DRM_QA_Data/acceleration'
                elif data_type == 'vel':
                    path = 'DRM_QA_Data/velocity'
                else:
                    path = 'DRM_QA_Data/displacement'
                
                data_orig = f[path][:]
            
            # Apply window mask if exists
            if hasattr(self, '_window_mask'):
                data_orig = data_orig[:, self._window_mask]
            # Apply resampling if exists
            elif hasattr(self, '_resample_cache'):
                data_resampled = np.zeros((3, len(self.time)))
                for i in range(3):
                    data_resampled[i] = interp1d(
                        self._resample_cache['time_orig'], 
                        data_orig[i], 
                        kind='linear', 
                        fill_value='extrapolate'
                    )(self.time)
                data_orig = data_resampled
            
            self._node_cache[key] = data_orig
        
        return self._node_cache[key]

    # ----------------------------------------------------------------
    def get_gf(self, node_id, subfault_id, component='z'):
        key = (node_id, subfault_id, component)
        
        if key not in self._gf_cache:
            # Find donor from mapping
            if self.node_mapping is not None:
                mask = (self.node_mapping[:, 0] == node_id) & (self.node_mapping[:, 1] == subfault_id)
                idx = np.where(mask)[0]
                
                if len(idx) == 0:
                    raise KeyError(f"Node {node_id}, subfault {subfault_id} not in mapping")
                
                ipair_target = self.node_mapping[idx[0], 2]
                source_node, source_subfault = self.pairs_mapping[ipair_target]
                # Inform if using donor node
                if source_node != node_id or source_subfault != subfault_id:
                    print(f"Node {node_id}, subfault {subfault_id} -> using donor: node {source_node}, subfault {source_subfault}")
            else:
                source_node, source_subfault = node_id, subfault_id
            
            with h5py.File(self.filename, 'r') as f:
                path = f'GF/sta_{source_node}/sub_{source_subfault}/{component}'
                if path not in f:
                    raise KeyError(f"GF not found at {path}")
                self._gf_cache[key] = f[path][:]
        
        return self._gf_cache[key]


    # ----------------------------------------------------------------
    def get_spectrum(self, node_id, subfault_id, component='z', part='real'):
        key = (node_id, subfault_id, component, part)
        
        if key not in self._spectrum_cache:
            if self.node_mapping is not None:
                pair_idx = self.node_mapping[node_id, subfault_id]
                
                if pair_idx == -1:
                    raise KeyError(
                        f"Node {node_id}, subfault {subfault_id} not computed (excluded during optimization)."
                    )
                
                source_node, source_subfault = self.pairs_mapping[pair_idx]
                # Inform if using donor node
                if source_node != node_id or source_subfault != subfault_id:
                    print(f"Node {node_id}, subfault {subfault_id} -> using donor: node {source_node}, subfault {source_subfault}")
            else:
                source_node, source_subfault = node_id, subfault_id
            
            with h5py.File(self.filename, 'r') as f:
                path = f'GF_Spectrum/sta_{source_node}/sub_{source_subfault}/spectrum_{component}_{part}'
                self._spectrum_cache[key] = f[path][:]
        
        return self._spectrum_cache[key]

    # ----------------------------------------------------------------
    def clear_cache(self):
        self._node_cache.clear()
        self._gf_cache.clear()
        self._spectrum_cache.clear()
        import gc
        gc.collect()
        print("Cache cleared")

    # ----------------------------------------------------------------

    def resample(self, dt):
        new_drm = DRM.__new__(DRM)
        
        new_drm.xyz = self.xyz
        new_drm.internal = self.internal
        new_drm.xyz_qa = self.xyz_qa
        new_drm.xyz_all = self.xyz_all
        new_drm.filename = self.filename
        new_drm.tstart = self.tstart
        new_drm.name = self.name
        new_drm.freqs = self.freqs
        new_drm.spacing = self.spacing
        new_drm.model_name = self.model_name
        
        new_drm._dt_orig = self._dt_orig
        new_drm._n_nodes = self._n_nodes
        new_drm._n_subfaults = self._n_subfaults
        new_drm._n_time = self._n_time
        new_drm._n_time_drm = self._n_time_drm
        new_drm._n_freqs = self._n_freqs
        
        new_drm._node_cache = {}
        new_drm._gf_cache = {}
        new_drm._spectrum_cache = {}
        
        # Time vectors
        time_orig = np.arange(self._n_time_drm) * self._dt_orig + self.tstart
        gf_time_orig = np.arange(self._n_time) * self._dt_orig
        
        new_drm.dt = dt
        new_drm.time = np.arange(time_orig[0], time_orig[-1], dt)
        new_drm.gf_time = np.arange(gf_time_orig[0], gf_time_orig[-1], dt)
        
        # AGREGA ESTO:
        new_drm._resample_cache = {
            'time_orig': time_orig,
            'gf_time_orig': gf_time_orig
        }
        
        print(f"time steps RESAMPLE: {len(new_drm.time)}")
        print(f"GF_Spectrum Fz. steps RESAMPLE: {len(new_drm.gf_time)}")
        
        return new_drm
    
    # ----------------------------------------------------------------
    def plot_DRM(self, xyz_origin=None, label_nodes=False, show_calculated=False):
        """
        Plot the DRM domain with optional visualization of nodes with calculated GFs.
        
        Parameters
        ----------
        xyz_origin : array-like, optional
            Origin for coordinate translation [x, y, z]
        label_nodes : bool or str, optional
            Node labeling mode:
            - False: no labels
            - True: all nodes
            - 'corners': corner nodes only
            - 'corners_edges': corners and edges
            - 'corners_half': corners and edge midpoints
            - 'calculated': only computational donor nodes
        show_calculated : bool, optional
            If True, highlight computational donor nodes (actually calculated GFs).
            Default: False
        """
        import h5py
        
        u_x = np.array([0, 1, 0])
        u_y = np.array([1, 0, 0])
        u_z = np.cross(u_x, u_y)
        R = np.column_stack([u_x, u_y, u_z])
        
        xyz_t = self.xyz * 1000 @ R
        xyz_qa_t = self.xyz_qa * 1000 @ R
        
        if xyz_origin is not None:
            translation = xyz_origin - xyz_qa_t[0]
            xyz_t += translation
            xyz_qa_t += translation
        
        xyz_int = xyz_t[self.internal]
        xyz_ext = xyz_t[~self.internal]
        
        x_min, x_max = xyz_int[:, 0].min(), xyz_int[:, 0].max()
        y_min, y_max = xyz_int[:, 1].min(), xyz_int[:, 1].max()
        z_min, z_max = xyz_int[:, 2].min(), xyz_int[:, 2].max()
        
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        
        faces = [
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[0], corners[3], corners[7], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]]
        ]
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_calculated:
            # Find computational donors (nodes that actually calculated GFs)
            with h5py.File(self.filename, 'r') as f:
                if 'GF_Database_Info/pairs_to_compute' in f:
                    pairs_db = f['GF_Database_Info/pairs_to_compute'][:]
                    computational_donors = np.unique(pairs_db[:, 0])
                    
                    int_idx = np.where(self.internal)[0]
                    ext_idx = np.where(~self.internal)[0]
                    
                    calc_int = np.isin(int_idx, computational_donors)
                    calc_ext = np.isin(ext_idx, computational_donors)
                    
                    xyz_int_calc = xyz_int[calc_int]
                    xyz_int_reuse = xyz_int[~calc_int]
                    xyz_ext_calc = xyz_ext[calc_ext]
                    xyz_ext_reuse = xyz_ext[~calc_ext]
                    
                    if len(xyz_ext_reuse) > 0:
                        ax.scatter(xyz_ext_reuse[:, 0], xyz_ext_reuse[:, 1], xyz_ext_reuse[:, 2],
                                c='lightblue', marker='o', s=30, alpha=0.3)
                    if len(xyz_int_reuse) > 0:
                        ax.scatter(xyz_int_reuse[:, 0], xyz_int_reuse[:, 1], xyz_int_reuse[:, 2],
                                c='pink', marker='s', s=20, alpha=0.3)
                    
                    if len(xyz_ext_calc) > 0:
                        ax.scatter(xyz_ext_calc[:, 0], xyz_ext_calc[:, 1], xyz_ext_calc[:, 2],
                                c='blue', marker='o', s=50, alpha=0.4, 
                                edgecolors='darkblue', linewidths=1.5)
                    if len(xyz_int_calc) > 0:
                        ax.scatter(xyz_int_calc[:, 0], xyz_int_calc[:, 1], xyz_int_calc[:, 2],
                                c='red', marker='s', s=30, alpha=0.4,
                                edgecolors='darkred', linewidths=1.5)
                    
                    print("="*60)
                    print("COMPUTATIONAL DONORS (nodes that calculated GFs)")
                    print("="*60)
                    print(f"Internal:  {calc_int.sum()}/{len(int_idx)} donors ({calc_int.sum()/len(int_idx)*100:.1f}%)")
                    print(f"Boundary:  {calc_ext.sum()}/{len(ext_idx)} donors ({calc_ext.sum()/len(ext_idx)*100:.1f}%)")
                    print(f"Total:     {len(computational_donors)}/{len(self.xyz)} donors ({len(computational_donors)/len(self.xyz)*100:.1f}%)")
                    print(f"Donor nodes: {sorted(computational_donors)}")
                    print("="*60)
                    
                else:
                    print("="*60)
                    print("NO GF DATABASE INFO")
                    print("="*60)
                    ax.scatter(xyz_ext[:, 0], xyz_ext[:, 1], xyz_ext[:, 2],
                            c='blue', marker='o', s=50, alpha=0.1)
                    ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2],
                            c='red', marker='s', s=30, alpha=0.4)
        else:
            ax.scatter(xyz_ext[:, 0], xyz_ext[:, 1], xyz_ext[:, 2], 
                    c='blue', marker='o', s=50, alpha=0.1)
            ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2], 
                    c='red', marker='s', s=30, alpha=0.4)
        
        ax.scatter(xyz_qa_t[:, 0], xyz_qa_t[:, 1], xyz_qa_t[:, 2], 
                c='green', marker='*', s=300, label='QA', zorder=10, 
                edgecolors='black', linewidths=2)
        
        cube = Poly3DCollection(faces, alpha=0.15, facecolor='red', 
                                edgecolor='darkred', linewidths=1.5)
        ax.add_collection3d(cube)

        if label_nodes == True:
            for i in range(len(xyz_t)):
                ax.text(xyz_t[i, 0], xyz_t[i, 1], xyz_t[i, 2], str(i), fontsize=8)

        elif label_nodes == 'corners':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                
                is_corner_int = (abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3) and \
                                (abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3) and \
                                (abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3)
                
                is_corner_ext = (abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3) and \
                                (abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3) and \
                                (abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3)
                
                if is_corner_int or is_corner_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=8, color=color, fontweight='bold')

        elif label_nodes == 'corners_edges':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                
                on_x_int = abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3
                on_y_int = abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3
                on_z_int = abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3
                
                on_x_ext = abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3
                on_y_ext = abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3
                on_z_ext = abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3
                
                is_on_int = sum([on_x_int, on_y_int, on_z_int]) >= 2
                is_on_ext = sum([on_x_ext, on_y_ext, on_z_ext]) >= 2
                
                if is_on_int or is_on_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=9, color=color)

        elif label_nodes == 'corners_half':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            z_mid = (z_min + z_max) / 2
            x_mid_ext = (x_min_ext + x_max_ext) / 2
            y_mid_ext = (y_min_ext + y_max_ext) / 2
            z_mid_ext = (z_min_ext + z_max_ext) / 2
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                
                on_x_int = abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3
                on_y_int = abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3
                on_z_int = abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3
                is_corner_int = on_x_int and on_y_int and on_z_int
                
                on_x_mid_int = abs(x - x_mid) < 1e-3
                on_y_mid_int = abs(y - y_mid) < 1e-3
                on_z_mid_int = abs(z - z_mid) < 1e-3
                is_mid_int = sum([on_x_int and on_y_int and on_z_mid_int,
                                on_x_int and on_y_mid_int and on_z_int,
                                on_x_mid_int and on_y_int and on_z_int]) > 0
                
                on_x_ext = abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3
                on_y_ext = abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3
                on_z_ext = abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3
                is_corner_ext = on_x_ext and on_y_ext and on_z_ext
                
                on_x_mid_ext = abs(x - x_mid_ext) < 1e-3
                on_y_mid_ext = abs(y - y_mid_ext) < 1e-3
                on_z_mid_ext = abs(z - z_mid_ext) < 1e-3
                is_mid_ext = sum([on_x_ext and on_y_ext and on_z_mid_ext,
                                on_x_ext and on_y_mid_ext and on_z_ext,
                                on_x_mid_ext and on_y_ext and on_z_ext]) > 0
                
                if is_corner_int or is_mid_int or is_corner_ext or is_mid_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=9, color=color)

        elif label_nodes == 'calculated':
            with h5py.File(self.filename, 'r') as f:
                if 'GF_Database_Info/pairs_to_compute' in f:
                    pairs_db = f['GF_Database_Info/pairs_to_compute'][:]
                    computational_donors = np.unique(pairs_db[:, 0])
                    
                    for i in computational_donors:
                        x, y, z = xyz_t[i]
                        color = 'darkred' if self.internal[i] else 'darkblue'
                        ax.text(x, y, z, str(i), fontsize=8, color=color)
                else:
                    print("No GF Database info")

        ax.set_xlabel("X' (km)")
        ax.set_ylabel("Y' (km)")
        ax.set_zlabel("Z' (km)")
        ax.legend()
        ax.grid(False, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"QA final position: {xyz_qa_t[0]}")
        return fig, ax


    # ----------------------------------------------------------------
    def plot_node_response(self, 
                        node_id=None, 
                        target_pos=None, 
                        xlim=[0,40], 
                        data_type='velocity'):
        """
        Plot time history response for specified nodes.
        
        Parameters
        ----------
        node_id : int, list, or 'QA', optional
            Node ID(s) to plot. Can be:
            - Single integer: node_id=5
            - List of integers: node_id=[0, 5, 10]
            - 'QA' string: node_id='QA'
            - Mixed list: node_id=[0, 'QA', 10]
        target_pos : array-like, optional
            [x, y, z] position to find nearest node
        xlim : list, default=[0, 40]
            Time limits [tmin, tmax]
        data_type : str, default='velocity'
            'accel', 'vel', or 'disp'
        """
        if node_id is not None:
            node_indices = node_id if isinstance(node_id, (list, np.ndarray)) else [node_id]
        elif target_pos is not None:
            distances = np.linalg.norm(self.xyz_all - target_pos, axis=1)
            node_idx = np.argmin(distances)
            node_indices = [node_idx]
            print(f"Distance: {distances[node_idx]:.6f} km")
        else:
            raise ValueError("Provide node_id or target_pos")
        
        fig = plt.figure(figsize=(8, 8))
        
        for node_idx in node_indices:
            # Handle 'QA' string
            if node_idx == 'QA' or node_idx == 'qa':
                print(f"Node: QA, Position: {self.xyz_qa[0]}")
                qa_data = self.get_qa_data(data_type)
                data_x, data_y, data_z = qa_data[0], qa_data[1], qa_data[2]
                label = 'QA'
            elif node_idx < len(self.xyz):
                print(f"Node: {node_idx}, Position: {self.xyz_all[node_idx]}")
                data = self.get_node_data(node_idx, data_type)
                data_x, data_y, data_z = data[0], data[1], data[2]
                label = f'Node {node_idx}'
            else:
                print(f"Node: {node_idx}, Position: {self.xyz_all[node_idx]}")
                qa_data = self.get_qa_data(data_type)
                data_x, data_y, data_z = qa_data[0], qa_data[1], qa_data[2]
                label = 'QA'
            
            plt.subplot(3, 1, 1)
            plt.plot(self.time, data_z, linewidth=1, label=label)
            
            plt.subplot(3, 1, 2)
            plt.plot(self.time, data_x, linewidth=1, label=label)
            
            plt.subplot(3, 1, 3)
            plt.plot(self.time, data_y, linewidth=1, label=label)
        
        ylabel = {'accel': 'Acceleration', 'vel': 'Velocity', 'disp': 'Displacement'}[data_type]
        
        plt.subplot(3, 1, 1)
        plt.title(f'Vertical (Z) - {ylabel}', fontweight='bold')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')
        
        plt.subplot(3, 1, 2)
        plt.title(f'East (E) - {ylabel}', fontweight='bold')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')
        
        plt.subplot(3, 1, 3)
        plt.title(f'North (N) - {ylabel}', fontweight='bold')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')
        
        plt.tight_layout()
        plt.show()
    # ----------------------------------------------------------------
    def plot_node_gf(self, 
                    node_id=None, 
                    target_pos=None, 
                    xlim=[0,40], 
                    subfault=0):
        """
        Plot Green's functions for specified nodes.
        
        Parameters
        ----------
        node_id : int, list, or 'QA', optional
            Node ID(s) to plot. Can include integers or 'QA' string.
        target_pos : array-like, optional
            [x, y, z] position to find nearest node
        xlim : list, default=[0, 40]
            Time limits [tmin, tmax]
        subfault : int or list, default=0
            Subfault ID(s) to plot
        """
        if node_id is not None:
            node_indices = node_id if isinstance(node_id, (list, np.ndarray)) else [node_id]
        elif target_pos is not None:
            distances = np.linalg.norm(self.xyz_all - target_pos, axis=1)
            node_idx = np.argmin(distances)
            node_indices = [node_idx]
            print(f"Distance: {distances[node_idx]:.6f} km")
        else:
            raise ValueError("Provide node_id or target_pos")
        
        subfault_indices = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
        
        fig = plt.figure(figsize=(8, 10))
        
        for node_idx in node_indices:
            for sub_idx in subfault_indices:
                if node_idx == 'QA' or node_idx == 'qa':
                    print(f"! GFs not available for QA node")
                    continue
                
                print(f"Node: {node_idx}, Subfault: {sub_idx}, Position: {self.xyz_all[node_idx]}")
                # Show donor info if using mapping
                if self.node_mapping is not None:
                    mask = (self.node_mapping[:, 0] == node_idx) & (self.node_mapping[:, 1] == sub_idx)
                    idx = np.where(mask)[0]
                    if len(idx) > 0:
                        ipair_target = self.node_mapping[idx[0], 2]
                        donor_node, donor_sub = self.pairs_mapping[ipair_target]
                        if donor_node != node_idx or donor_sub != sub_idx:
                            print(f"  -> Using GF from donor: node {donor_node}, subfault {donor_sub}")
                
                gf_z = self.get_gf(node_idx, sub_idx, 'z')
                gf_e = self.get_gf(node_idx, sub_idx, 'e')
                gf_n = self.get_gf(node_idx, sub_idx, 'n')
                
                label = f'N{node_idx}_S{sub_idx}'
                
                plt.subplot(3, 1, 1)
                plt.plot(self.gf_time, gf_z, linewidth=1, alpha=1, label=label)
                
                plt.subplot(3, 1, 2)
                plt.plot(self.gf_time, gf_e, linewidth=1, alpha=1, label=label)
                
                plt.subplot(3, 1, 3)
                plt.plot(self.gf_time, gf_n, linewidth=1, alpha=1, label=label)
        
        plt.subplot(3, 1, 1)
        plt.title('Vertical (Z) - Green Function', fontweight='bold')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.title('East (E) - Green Function', fontweight='bold')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.title('North (N) - Green Function', fontweight='bold')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------

    def plot_node_tensor_gf(self, 
                            node_id=None, 
                            target_pos=None, 
                            xlim=[0,40], 
                            subfault=0):
        """
        Plot tensor Green's functions (9 components) for specified nodes.
        Uses node mapping to find the actual donor node when GFs are not directly available.
        """
        if node_id is not None:
            node_indices = node_id if isinstance(node_id, (list, np.ndarray)) else [node_id]
        elif target_pos is not None:
            distances = np.linalg.norm(self.xyz_all - target_pos, axis=1)
            node_idx = np.argmin(distances)
            node_indices = [node_idx]
            print(f"Distance: {distances[node_idx]:.6f} km")
        else:
            raise ValueError("Provide node_id or target_pos")
        
        subfault_indices = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
        
        # Check which subfaults have same GF source for all requested nodes
        if len(node_indices) >= 2 and self.node_mapping is not None:
            equal_subfaults = []
            different_subfaults = []
            
            for sub in range(self._n_subfaults):
                sources = []
                for node_idx in node_indices:
                    mask = (self.node_mapping[:, 0] == node_idx) & (self.node_mapping[:, 1] == sub)
                    idx = np.where(mask)[0]
                    if len(idx) > 0:
                        ipair = self.node_mapping[idx[0], 2]
                        source = tuple(self.pairs_mapping[ipair])
                        sources.append(source)
                
                if len(sources) == len(node_indices) and len(set(sources)) == 1:
                    equal_subfaults.append(sub)
                else:
                    different_subfaults.append(sub)
            
            print(f"\nNodes {node_indices}:")
            print(f"  Equal GFs: {equal_subfaults}")
            print(f"  Different: {different_subfaults}\n")
        
        # Component labels
        component_labels = [
            'G_11', 'G_12', 'G_13',
            'G_21', 'G_22', 'G_23',
            'G_31', 'G_32', 'G_33'
        ]
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        
        for node_idx in node_indices:
            for sub_idx in subfault_indices:
                if node_idx == 'QA' or node_idx == 'qa':
                    print(f"! Tensor GFs not available for QA node")
                    continue
                
                # Find the donor node using mapping
                if self.node_mapping is not None:
                    mask = (self.node_mapping[:, 0] == node_idx) & (self.node_mapping[:, 1] == sub_idx)
                    idx = np.where(mask)[0]
                    
                    if len(idx) == 0:
                        print(f"Node {node_idx}, subfault {sub_idx}: not in mapping")
                        continue
                    
                    ipair = self.node_mapping[idx[0], 2]
                    source_node, source_subfault = self.pairs_mapping[ipair]
                    
                    if source_node != node_idx:
                        print(f"Node {node_idx}, subfault {sub_idx} -> using donor: node {source_node}, subfault {source_subfault}")
                    else:
                        print(f"Node {node_idx}, subfault {sub_idx} -> uses own GFs")
                else:
                    source_node, source_subfault = node_idx, sub_idx
                
                # Load tdata from the donor node
                with h5py.File(self.filename, 'r') as f:
                    path = f'GF/sta_{source_node}/sub_{source_subfault}/tdata'
                    if path not in f:
                        print(f"  ! GF not found at {path}")
                        continue
                    tdata = f[path][:]
                    t0 = f[f'GF/sta_{source_node}/sub_{source_subfault}/t0'][()]
                
                # Time vector
                time = np.arange(tdata.shape[0]) * self._dt_orig + t0
                label = f'N{node_idx}_S{sub_idx}'
                
                # Plot 9 components
                for i in range(9):
                    row = i // 3
                    col = i % 3
                    axes[row, col].plot(time, tdata[:, i], linewidth=0.8, label=label)
                    axes[row, col].set_title(f'{component_labels[i]}', fontsize=11, fontweight='bold')
                    axes[row, col].set_xlabel('Time [s]', fontsize=9)
                    axes[row, col].set_ylabel('Amplitude', fontsize=9)
                    axes[row, col].grid(True, alpha=0.3)
                    axes[row, col].set_xlim(xlim)
        
        axes[0, 0].legend(fontsize=8)
        plt.suptitle(f'Tensor Green Functions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------
    def plot_node_f_spectrum(self, 
                        node_id=None, 
                        target_pos=None, 
                        xlim=[0,100], 
                        subfault=0):
        """
        Plot Fourier spectrum for specified nodes.
        
        Parameters
        ----------
        node_id : int, list, or 'QA', optional
            Node ID(s) to plot. Can include integers or 'QA' string.
        target_pos : array-like, optional
            [x, y, z] position to find nearest node
        xlim : list, default=[0, 100]
            Frequency limits [fmin, fmax]
        subfault : int or list, default=0
            Subfault ID(s) to plot
        """
        if node_id is not None:
            node_indices = node_id if isinstance(node_id, (list, np.ndarray)) else [node_id]
        elif target_pos is not None:
            distances = np.linalg.norm(self.xyz_all - target_pos, axis=1)
            node_idx = np.argmin(distances)
            node_indices = [node_idx]
            print(f"Distance: {distances[node_idx]:.6f} km")
        else:
            raise ValueError("Provide node_id or target_pos")
        
        subfault_indices = subfault if isinstance(subfault, (list, np.ndarray)) else [subfault]
        
        fig = plt.figure(figsize=(8, 10))
        
        for node_idx in node_indices:
            for sub_idx in subfault_indices:
                if node_idx == 'QA' or node_idx == 'qa':
                    print(f"! Spectrum not available for QA node")
                    continue
                
                print(f"Node: {node_idx}, Subfault: {sub_idx}, Position: {self.xyz_all[node_idx]}")
                # Show donor info if using mapping
                if self.node_mapping is not None:
                    mask = (self.node_mapping[:, 0] == node_idx) & (self.node_mapping[:, 1] == sub_idx)
                    idx = np.where(mask)[0]
                    if len(idx) > 0:
                        ipair_target = self.node_mapping[idx[0], 2]
                        donor_node, donor_sub = self.pairs_mapping[ipair_target]
                        if donor_node != node_idx or donor_sub != sub_idx:
                            print(f"  -> Using GF from donor: node {donor_node}, subfault {donor_sub}")

                try:
                    real_z = self.get_spectrum(node_idx, sub_idx, 'z', 'real')
                    imag_z = self.get_spectrum(node_idx, sub_idx, 'z', 'imag')
                    real_e = self.get_spectrum(node_idx, sub_idx, 'e', 'real')
                    imag_e = self.get_spectrum(node_idx, sub_idx, 'e', 'imag')
                    real_n = self.get_spectrum(node_idx, sub_idx, 'n', 'real')
                    imag_n = self.get_spectrum(node_idx, sub_idx, 'n', 'imag')
                    
                    mag_z = np.sqrt(real_z**2 + imag_z**2)
                    mag_e = np.sqrt(real_e**2 + imag_e**2)
                    mag_n = np.sqrt(real_n**2 + imag_n**2)
                    
                    label = f'N{node_idx}_S{sub_idx}'
                    
                    plt.subplot(3, 1, 1)
                    plt.loglog(self.freqs, mag_z, linewidth=1, alpha=1, label=label)
                    
                    plt.subplot(3, 1, 2)
                    plt.loglog(self.freqs, mag_e, linewidth=1, alpha=1, label=label)
                    
                    plt.subplot(3, 1, 3)
                    plt.loglog(self.freqs, mag_n, linewidth=1, alpha=1, label=label)
                    
                except KeyError:
                    print(f"  ! No spectrum")
        
        plt.subplot(3, 1, 1)
        plt.title('Vertical (Z) - Fourier Spectrum', fontweight='bold')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude (log scale)')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.title('East (E) - Fourier Spectrum', fontweight='bold')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude (log scale)')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.title('North (N) - Fourier Spectrum', fontweight='bold')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude (log scale)')
        plt.xlim(xlim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    # ----------------------------------------------------------------
    def plot_calculated_vs_reused(self, db_filename='GF_database_pairs.h5', 
                                xyz_origin=None, 
                                label_nodes=False):
        """
        Parameters
        ----------
        db_filename : str, optional
            Path to GF database file. Default: 'GF_database_pairs.h5'
        xyz_origin : array-like, optional
            Origin for coordinate translation [x, y, z]
        label_nodes : bool or str, optional
            Node labeling mode:
            - False: no labels
            - True: all nodes
            - 'corners': corner nodes only
            - 'corners_edges': corners and edges
            - 'corners_half': corners and edge midpoints
            - 'calculated_nodes': only nodes with saved GFs
        """
        
        import h5py
        
        with h5py.File(db_filename, 'r') as f:
            pairs_calc = f['pairs_to_compute'][:]
            unique_db_pairs = np.unique(pairs_calc[:, 0])
        
        with h5py.File(self.filename, 'r') as f:
            if 'GF' in f:
                stations_with_gf = [int(k.replace('sta_', '')) for k in f['GF'].keys()]
                unique_stations_calc = np.array(stations_with_gf)
            else:
                unique_stations_calc = np.array([])
        
        u_x = np.array([0, 1, 0])
        u_y = np.array([1, 0, 0])
        u_z = np.cross(u_x, u_y)
        R = np.column_stack([u_x, u_y, u_z])
        
        xyz_t = self.xyz * 1000 @ R
        xyz_qa_t = self.xyz_qa * 1000 @ R
        
        if xyz_origin is not None:
            translation = xyz_origin - xyz_qa_t[0]
            xyz_t += translation
            xyz_qa_t += translation
        
        xyz_int = xyz_t[self.internal]
        xyz_ext = xyz_t[~self.internal]
        
        int_idx = np.where(self.internal)[0]
        ext_idx = np.where(~self.internal)[0]
        
        calc_int = np.isin(int_idx, unique_stations_calc)
        calc_ext = np.isin(ext_idx, unique_stations_calc)
        
        xyz_int_calc = xyz_int[calc_int]
        xyz_int_reuse = xyz_int[~calc_int]
        xyz_ext_calc = xyz_ext[calc_ext]
        xyz_ext_reuse = xyz_ext[~calc_ext]
        
        x_min, x_max = xyz_int[:, 0].min(), xyz_int[:, 0].max()
        y_min, y_max = xyz_int[:, 1].min(), xyz_int[:, 1].max()
        z_min, z_max = xyz_int[:, 2].min(), xyz_int[:, 2].max()
        
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        
        faces = [
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[0], corners[3], corners[7], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]]
        ]
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(xyz_ext_reuse) > 0:
            ax.scatter(xyz_ext_reuse[:, 0], xyz_ext_reuse[:, 1], xyz_ext_reuse[:, 2],
                    c='lightblue', marker='o', s=30, alpha=0.3)
        
        if len(xyz_int_reuse) > 0:
            ax.scatter(xyz_int_reuse[:, 0], xyz_int_reuse[:, 1], xyz_int_reuse[:, 2],
                    c='pink', marker='s', s=20, alpha=0.3)
        
        if len(xyz_ext_calc) > 0:
            ax.scatter(xyz_ext_calc[:, 0], xyz_ext_calc[:, 1], xyz_ext_calc[:, 2],
                    c='blue', marker='o', alpha=0.5,
                    edgecolors='darkblue', linewidths=1.5)
        
        if len(xyz_int_calc) > 0:
            ax.scatter(xyz_int_calc[:, 0], xyz_int_calc[:, 1], xyz_int_calc[:, 2],
                    c='red', marker='s', alpha=0.5,
                    edgecolors='darkred', linewidths=1.5)
        
        ax.scatter(xyz_qa_t[:, 0], xyz_qa_t[:, 1], xyz_qa_t[:, 2],
                c='green', marker='*', s=400, label='QA', zorder=10,
                edgecolors='black', linewidths=2)
        
        cube = Poly3DCollection(faces, alpha=0.1, facecolor='red',
                                edgecolor='darkred', linewidths=2)
        ax.add_collection3d(cube)
        
        if label_nodes == True:
            for i in range(len(xyz_t)):
                ax.text(xyz_t[i, 0], xyz_t[i, 1], xyz_t[i, 2], str(i), fontsize=8)

        elif label_nodes == 'corners':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                is_corner_int = (abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3) and \
                                (abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3) and \
                                (abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3)
                is_corner_ext = (abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3) and \
                                (abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3) and \
                                (abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3)
                if is_corner_int or is_corner_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=8, color=color)

        elif label_nodes == 'corners_edges':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                on_x_int = abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3
                on_y_int = abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3
                on_z_int = abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3
                on_x_ext = abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3
                on_y_ext = abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3
                on_z_ext = abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3
                is_on_int = sum([on_x_int, on_y_int, on_z_int]) >= 2
                is_on_ext = sum([on_x_ext, on_y_ext, on_z_ext]) >= 2
                if is_on_int or is_on_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=9, color=color)

        elif label_nodes == 'corners_half':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            z_mid = (z_min + z_max) / 2
            x_mid_ext = (x_min_ext + x_max_ext) / 2
            y_mid_ext = (y_min_ext + y_max_ext) / 2
            z_mid_ext = (z_min_ext + z_max_ext) / 2
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                on_x_int = abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3
                on_y_int = abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3
                on_z_int = abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3
                is_corner_int = on_x_int and on_y_int and on_z_int
                on_x_mid_int = abs(x - x_mid) < 1e-3
                on_y_mid_int = abs(y - y_mid) < 1e-3
                on_z_mid_int = abs(z - z_mid) < 1e-3
                is_mid_int = sum([on_x_int and on_y_int and on_z_mid_int,
                                on_x_int and on_y_mid_int and on_z_int,
                                on_x_mid_int and on_y_int and on_z_int]) > 0
                on_x_ext = abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3
                on_y_ext = abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3
                on_z_ext = abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3
                is_corner_ext = on_x_ext and on_y_ext and on_z_ext
                on_x_mid_ext = abs(x - x_mid_ext) < 1e-3
                on_y_mid_ext = abs(y - y_mid_ext) < 1e-3
                on_z_mid_ext = abs(z - z_mid_ext) < 1e-3
                is_mid_ext = sum([on_x_ext and on_y_ext and on_z_mid_ext,
                                on_x_ext and on_y_mid_ext and on_z_ext,
                                on_x_mid_ext and on_y_ext and on_z_ext]) > 0
                if is_corner_int or is_mid_int or is_corner_ext or is_mid_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=9, color=color)

        elif label_nodes == 'calculated_nodes':
            for i in unique_stations_calc:
                x, y, z = xyz_t[i]
                color = 'darkred' if self.internal[i] else 'darkblue'
                ax.text(x, y, z, str(i), fontsize=8, color=color)

        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_zlabel("Z (m)", fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("="*60)
        print("GFs SAVED IN FILE")
        print("="*60)
        print(f"Internal:  {calc_int.sum()}/{len(int_idx)} with GFs ({calc_int.sum()/len(int_idx)*100:.1f}%)")
        print(f"Boundary:  {calc_ext.sum()}/{len(ext_idx)} with GFs ({calc_ext.sum()/len(ext_idx)*100:.1f}%)")
        print(f"Total:     {len(unique_stations_calc)}/{len(self.xyz)} with GFs ({len(unique_stations_calc)/len(self.xyz)*100:.1f}%)")
        print("\n" + "-"*60)
        print("DATABASE PAIRS (Stage 1)")
        print("-"*60)
        print(f"Total pairs in database: {len(unique_db_pairs)}/{len(self.xyz)} ({len(unique_db_pairs)/len(self.xyz)*100:.1f}%)")
        print(f"Difference: {len(unique_db_pairs) - len(unique_stations_calc)} pairs computed but GFs not saved")
        print("="*60)
        
        return fig, ax
        
    # ----------------------------------------------------------------
    def plot_newmark_spectra(self, 
                            node_id=None, 
                            target_pos=None, 
                            xlim=[0, 5], 
                            data_type='accel'):
        """
        Plot Newmark response spectra for specified nodes.
        
        Parameters
        ----------
        node_id : int, list, or 'QA', optional
            Node ID(s) to plot. Can include integers or 'QA' string.
        target_pos : array-like, optional
            [x, y, z] position to find nearest node
        xlim : list, default=[0, 5]
            Period limits [Tmin, Tmax]
        data_type : str, default='accel'
            'accel', 'vel', or 'disp'
        """
        # Get node indices
        if node_id is not None:
            node_indices = node_id if isinstance(node_id, (list, np.ndarray)) else [node_id]
        elif target_pos is not None:
            distances = np.linalg.norm(self.xyz_all - target_pos, axis=1)
            node_idx = np.argmin(distances)
            node_indices = [node_idx]
            print(f"Distance: {distances[node_idx]:.6f} km")
        else:
            raise ValueError("Provide node_id or target_pos")
        
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        
        for node_idx in node_indices:
            # Get data
            if node_idx == 'QA' or node_idx == 'qa':
                print(f"Node: QA, Position: {self.xyz_qa[0]}")
                data = self.get_qa_data(data_type)
                label = 'QA'
            elif node_idx < len(self.xyz):
                print(f"Node: {node_idx}, Position: {self.xyz_all[node_idx]}")
                data = self.get_node_data(node_idx, data_type)
                label = f'N-{node_idx}'
            else:
                print(f"Node: {node_idx}, Position: {self.xyz_all[node_idx]}")
                data = self.get_qa_data(data_type)
                label = 'QA'
            
            data_x, data_y, data_z = data[0], data[1], data[2]
            dt = self.time[1] - self.time[0]
            
            # Compute spectra
            if data_type == 'accel':
                spec_z = NewmarkSpectrumAnalyzer.compute(data_z / 9.81, dt)
                spec_x = NewmarkSpectrumAnalyzer.compute(data_x / 9.81, dt)
                spec_y = NewmarkSpectrumAnalyzer.compute(data_y / 9.81, dt)
            else:
                spec_z = NewmarkSpectrumAnalyzer.compute(data_z, dt)
                spec_x = NewmarkSpectrumAnalyzer.compute(data_x, dt)
                spec_y = NewmarkSpectrumAnalyzer.compute(data_y, dt)
            
            T = spec_z['T']
            PSa_z = spec_z['PSa']
            PSa_x = spec_x['PSa']
            PSa_y = spec_y['PSa']
            
            # Plot
            axes[0].plot(T, PSa_z, linewidth=2, label=label)
            axes[1].plot(T, PSa_x, linewidth=2, label=label)
            axes[2].plot(T, PSa_y, linewidth=2, label=label)
        
        # Configure
        ylabel = 'Sa (g)' if data_type == 'accel' else 'Spectral Response'
        
        axes[0].set_title('Vertical (Z) - Newmark Spectrum', fontweight='bold')
        axes[0].set_xlabel('T (s)', fontsize=12)
        axes[0].set_ylabel(ylabel, fontsize=12)
        axes[0].set_xlim(xlim)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_title('X - Newmark Spectrum', fontweight='bold')
        axes[1].set_xlabel('T (s)', fontsize=12)
        axes[1].set_ylabel(ylabel, fontsize=12)
        axes[1].set_xlim(xlim)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].set_title('Y - Newmark Spectrum', fontweight='bold')
        axes[2].set_xlabel('T (s)', fontsize=12)
        axes[2].set_ylabel(ylabel, fontsize=12)
        axes[2].set_xlim(xlim)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
    # ----------------------------------------------------------------
    def plot_gf_connections(self, 
                            node_id, 
                            xyz_origin=None, 
                            label_nodes=False):
        """
        Plot GF donor-recipient connections for a specific node.
        
        Parameters
        ----------
        node_id : int
            Node ID to analyze
        xyz_origin : array-like, optional
            Origin for coordinate translation [x, y, z]
        label_nodes : bool or str, optional
            Node labeling mode:
            - False: no labels
            - True: all nodes
            - 'corners': corner nodes only
            - 'corners_edges': corners and edges
            - 'corners_half': corners and edge midpoints
            - 'calculated': only computational donor nodes
            - 'donor_receivers': only donor and its receivers
        """
        import h5py
        
        if self.node_mapping is None:
            print("No node mapping available")
            return
        
        # Get computational donors
        with h5py.File(self.filename, 'r') as f:
            if 'GF_Database_Info/pairs_to_compute' in f:
                pairs_db = f['GF_Database_Info/pairs_to_compute'][:]
                comp_donors = set(np.unique(pairs_db[:, 0]))
            else:
                print("No GF Database info")
                return
        
        # Classify ALL nodes
        super_donors = set()
        recipients_from_others = set()
        
        for node in range(len(self.xyz)):
            mask = (self.node_mapping[:, 0] == node) & (self.node_mapping[:, 1] == 0)
            idx = np.where(mask)[0]
            
            if len(idx) > 0:
                ipair = self.node_mapping[idx[0], 2]
                donor_node = self.pairs_mapping[ipair, 0]
                
                if donor_node != node:
                    recipients_from_others.add(node)
                    super_donors.add(donor_node)
        
        solitary_donors = comp_donors - super_donors
        recipients_pure = recipients_from_others - comp_donors
        
        # Print classification
        print("NODE CLASSIFICATION")
        print("="*70)
        print(f"Super Donors (9):      {sorted(super_donors)}")
        print(f"Solitary Donors (27):  {sorted(solitary_donors)}")
        print(f"Pure Receivers (62):   {sorted(recipients_pure)}")
        
        # Analyze target node
        print(f"\nANALYZING NODE {node_id}:")
        print("-"*70)
        
        if node_id in super_donors:
            print(f"Node {node_id} is a SUPER DONOR")
            
            # Count recipients per subfault
            n_subfaults = self._n_subfaults
            
            if n_subfaults <= 10:  # Solo si no son demasiadas
                print(f"  Recipients per subfault:")
                for sub in range(min(5, n_subfaults)):  # Mostrar máximo 5
                    recipients = []
                    for node in range(len(self.xyz)):
                        mask = (self.node_mapping[:, 0] == node) & (self.node_mapping[:, 1] == sub)
                        idx = np.where(mask)[0]
                        if len(idx) > 0:
                            ipair = self.node_mapping[idx[0], 2]
                            donor_node = self.pairs_mapping[ipair, 0]
                            if donor_node == node_id and node != node_id:
                                recipients.append(node)
                    print(f"    Subfault {sub}: {len(recipients)} recipients")
            else:
                # Para muchas subfallas, solo mostrar subfault 0
                recipients = []
                for node in range(len(self.xyz)):
                    mask = (self.node_mapping[:, 0] == node) & (self.node_mapping[:, 1] == 0)
                    idx = np.where(mask)[0]
                    if len(idx) > 0:
                        ipair = self.node_mapping[idx[0], 2]
                        donor_node = self.pairs_mapping[ipair, 0]
                        if donor_node == node_id and node != node_id:
                            recipients.append(node)
                print(f"  Donates to {len(recipients)} nodes (checked for subfault 0)")
                print(f"  Note: Mapping is per (node, subfault) pair - may vary across subfaults")


            
            donor_to_plot = node_id
            recipients_to_plot = recipients
            
        elif node_id in solitary_donors:
            print(f"Node {node_id} is a SOLITARY DONOR")
            print(f"  Only uses its own GFs")
            
            donor_to_plot = node_id
            recipients_to_plot = []
            
        else:
            print(f"Node {node_id} is a PURE RECEIVER")
            
            # Find donor
            mask = (self.node_mapping[:, 0] == node_id) & (self.node_mapping[:, 1] == 0)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                ipair = self.node_mapping[idx[0], 2]
                donor_node = self.pairs_mapping[ipair, 0]
                print(f"  Uses GFs from donor: {donor_node}")
            else:
                print(f"  ERROR: No donor found")
                return
            
            donor_to_plot = donor_node
            recipients_to_plot = [node_id]
        
        print("="*70)
        
        # === PLOTTING (same format as plot_DRM) ===
        u_x = np.array([0, 1, 0])
        u_y = np.array([1, 0, 0])
        u_z = np.cross(u_x, u_y)
        R = np.column_stack([u_x, u_y, u_z])
        
        xyz_t = self.xyz * 1000 @ R
        xyz_qa_t = self.xyz_qa * 1000 @ R
        
        if xyz_origin is not None:
            translation = xyz_origin - xyz_qa_t[0]
            xyz_t += translation
            xyz_qa_t += translation
        
        xyz_int = xyz_t[self.internal]
        xyz_ext = xyz_t[~self.internal]
        
        x_min, x_max = xyz_int[:, 0].min(), xyz_int[:, 0].max()
        y_min, y_max = xyz_int[:, 1].min(), xyz_int[:, 1].max()
        z_min, z_max = xyz_int[:, 2].min(), xyz_int[:, 2].max()
        
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        
        faces = [
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[0], corners[3], corners[7], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]]
        ]
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Background nodes (faded)
        ax.scatter(xyz_ext[:, 0], xyz_ext[:, 1], xyz_ext[:, 2],
                c='blue', marker='o', s=50, alpha=0.1)
        ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2],
                c='red', marker='s', s=30, alpha=0.3)
        
        # Donor node (red square)
        donor_pos = xyz_t[donor_to_plot]
        ax.scatter(*donor_pos, c='red', marker='s', s=100,
                edgecolors='darkred', linewidths=2, zorder=10, alpha=0.5)
        
        # Recipients and connection rays
        for rec in recipients_to_plot:
            rec_pos = xyz_t[rec]
            
            # Recipient node (orange)
            ax.scatter(*rec_pos, c='orange', marker='o', s=80,
                    edgecolors='darkorange', linewidths=1.5, alpha=0.5)
            
            # Ray from donor to recipient
            ax.plot([donor_pos[0], rec_pos[0]],
                    [donor_pos[1], rec_pos[1]],
                    [donor_pos[2], rec_pos[2]],
                    color='darkorange', linestyle='--', alpha=0.5, linewidth=2)
        
        # QA point
        ax.scatter(xyz_qa_t[:, 0], xyz_qa_t[:, 1], xyz_qa_t[:, 2],
                c='green', marker='*', s=300, label='QA', zorder=10,
                edgecolors='black', linewidths=2)
        
        # Cube
        cube = Poly3DCollection(faces, alpha=0.10, facecolor='red',
                                edgecolor='darkred', linewidths=1.5)
        ax.add_collection3d(cube)
        
        # Node labels
        if label_nodes == True:
            for i in range(len(xyz_t)):
                ax.text(xyz_t[i, 0], xyz_t[i, 1], xyz_t[i, 2], str(i), fontsize=8)
        
        elif label_nodes == 'corners':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                
                is_corner_int = (abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3) and \
                                (abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3) and \
                                (abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3)
                
                is_corner_ext = (abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3) and \
                                (abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3) and \
                                (abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3)
                
                if is_corner_int or is_corner_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=8, color=color, fontweight='bold')
        
        elif label_nodes == 'corners_edges':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                
                on_x_int = abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3
                on_y_int = abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3
                on_z_int = abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3
                
                on_x_ext = abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3
                on_y_ext = abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3
                on_z_ext = abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3
                
                is_on_int = sum([on_x_int, on_y_int, on_z_int]) >= 2
                is_on_ext = sum([on_x_ext, on_y_ext, on_z_ext]) >= 2
                
                if is_on_int or is_on_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=9, color=color)
        
        elif label_nodes == 'corners_half':
            x_min_ext = xyz_t[:, 0].min()
            x_max_ext = xyz_t[:, 0].max()
            y_min_ext = xyz_t[:, 1].min()
            y_max_ext = xyz_t[:, 1].max()
            z_min_ext = xyz_t[:, 2].min()
            z_max_ext = xyz_t[:, 2].max()
            
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            z_mid = (z_min + z_max) / 2
            x_mid_ext = (x_min_ext + x_max_ext) / 2
            y_mid_ext = (y_min_ext + y_max_ext) / 2
            z_mid_ext = (z_min_ext + z_max_ext) / 2
            
            for i in range(len(xyz_t)):
                x, y, z = xyz_t[i]
                
                on_x_int = abs(x - x_min) < 1e-3 or abs(x - x_max) < 1e-3
                on_y_int = abs(y - y_min) < 1e-3 or abs(y - y_max) < 1e-3
                on_z_int = abs(z - z_min) < 1e-3 or abs(z - z_max) < 1e-3
                is_corner_int = on_x_int and on_y_int and on_z_int
                
                on_x_mid_int = abs(x - x_mid) < 1e-3
                on_y_mid_int = abs(y - y_mid) < 1e-3
                on_z_mid_int = abs(z - z_mid) < 1e-3
                is_mid_int = sum([on_x_int and on_y_int and on_z_mid_int,
                                on_x_int and on_y_mid_int and on_z_int,
                                on_x_mid_int and on_y_int and on_z_int]) > 0
                
                on_x_ext = abs(x - x_min_ext) < 1e-3 or abs(x - x_max_ext) < 1e-3
                on_y_ext = abs(y - y_min_ext) < 1e-3 or abs(y - y_max_ext) < 1e-3
                on_z_ext = abs(z - z_min_ext) < 1e-3 or abs(z - z_max_ext) < 1e-3
                is_corner_ext = on_x_ext and on_y_ext and on_z_ext
                
                on_x_mid_ext = abs(x - x_mid_ext) < 1e-3
                on_y_mid_ext = abs(y - y_mid_ext) < 1e-3
                on_z_mid_ext = abs(z - z_mid_ext) < 1e-3
                is_mid_ext = sum([on_x_ext and on_y_ext and on_z_mid_ext,
                                on_x_ext and on_y_mid_ext and on_z_ext,
                                on_x_mid_ext and on_y_ext and on_z_ext]) > 0
                
                if is_corner_int or is_mid_int or is_corner_ext or is_mid_ext:
                    color = 'darkred' if self.internal[i] else 'darkblue'
                    ax.text(x, y, z, str(i), fontsize=9, color=color)
        
        elif label_nodes == 'calculated':
            for i in comp_donors:
                x, y, z = xyz_t[i]
                color = 'darkred' if self.internal[i] else 'darkblue'
                ax.text(x, y, z, str(i), fontsize=8, color=color)
        
        elif label_nodes == 'donor_receivers':
            # Label donor in red
            x, y, z = xyz_t[donor_to_plot]
            ax.text(x, y, z, str(donor_to_plot), fontsize=10, 
                    color='darkred', fontweight='bold')
            
            # Label receivers in blue
            for rec in recipients_to_plot:
                x, y, z = xyz_t[rec]
                ax.text(x, y, z, str(rec), fontsize=9, 
                        color='darkblue', fontweight='bold')
        
        # Title
        if node_id in super_donors:
            title = f"Node {node_id} (SUPER DONOR) → {len(recipients_to_plot)} recipients"
        elif node_id in solitary_donors:
            title = f"Node {node_id} (SOLITARY DONOR)"
        else:
            title = f"Node {node_id} (RECEIVER) ← Donor {donor_to_plot}"
        
        ax.set_xlabel("X' (km)")
        ax.set_ylabel("Y' (km)")
        ax.set_zlabel("Z' (km)")
        ax.legend()
        ax.grid(False, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------

    def get_window(self, t_start, t_end):
        """
        Get a windowed copy of the DRM object.
        
        Parameters
        ----------
        t_start : float
            Start time of window (seconds)
        t_end : float
            End time of window (seconds)
        
        Returns
        -------
        DRM
            New DRM object with windowed data
        """
        new_drm = DRM.__new__(DRM)
        
        # Copy geometry and metadata
        new_drm.xyz = self.xyz
        new_drm.internal = self.internal
        new_drm.xyz_qa = self.xyz_qa
        new_drm.xyz_all = self.xyz_all
        new_drm.filename = self.filename
        new_drm.tstart = t_start
        new_drm.name = f"{self.name} [{t_start}-{t_end}s]"
        new_drm.freqs = self.freqs
        new_drm.spacing = self.spacing
        new_drm.model_name = self.model_name
        
        # Copy internal metadata
        new_drm._dt_orig = self._dt_orig
        new_drm._n_nodes = self._n_nodes
        new_drm._n_subfaults = self._n_subfaults
        new_drm._n_time = self._n_time
        new_drm._n_freqs = self._n_freqs
        
        # Find window indices
        mask = (self.time >= t_start) & (self.time <= t_end)
        new_drm._window_mask = mask
        new_drm._n_time_drm = np.sum(mask)
        
        # Windowed time vectors
        new_drm.dt = self.dt
        new_drm.time = self.time[mask]
        new_drm.gf_time = self.gf_time  # GF time unchanged
        
        # Store original time for cache interpolation
        new_drm._original_time = self.time
        new_drm._source_drm = self  # Reference to original for data access
        
        # Empty caches
        new_drm._node_cache = {}
        new_drm._gf_cache = {}
        new_drm._spectrum_cache = {}
        
        # Copy mapping if exists
        new_drm.node_mapping = self.node_mapping if hasattr(self, 'node_mapping') else None
        new_drm.pairs_mapping = self.pairs_mapping if hasattr(self, 'pairs_mapping') else None
        
        # Copy GF database info if exists
        for attr in ['gf_db_pairs', 'gf_db_dh', 'gf_db_zrec', 'gf_db_zsrc',
                    'gf_db_delta_h', 'gf_db_delta_v_rec', 'gf_db_delta_v_src']:
            if hasattr(self, attr):
                setattr(new_drm, attr, getattr(self, attr))
        
        print(f"Window: [{t_start}, {t_end}]s -> {new_drm._n_time_drm} samples")
        
        return new_drm

