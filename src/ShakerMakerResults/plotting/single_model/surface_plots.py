"""
surface_plots.py
================
Standalone surface plotting helpers (PGA / spectral / Arias maps).

These functions are heavy:

* :func:`plot_surface_newmark` computes a full SDOF Newmark spectrum per
  node (joblib-parallel) and caches the result on ``self._newmark_cache``.
  The cache key is the ``data_type`` -- changing ``T_target``,
  ``spectral_type`` or ``factor`` reuses the same cached spectra.
* :func:`plot_surface_arias` mirrors the same caching strategy for Arias
  intensity.

Both helpers pick a *safe* mode (chunked HDF5 reads inside each worker)
when the data does not fit comfortably in RAM, and a *fast* mode
(everything preloaded) otherwise.

Note: the module-level ``time`` import is shadowed inside ``plot_surface``
by the ``time`` parameter; we leave it as-is for backward compatibility
with existing notebooks. Use ``import time as _time`` upstream if you
need the module from inside a ``plot_surface`` call.
"""

from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import numpy as np
import time

from ...utils import _rotate


def _format_parallel_elapsed(seconds: float) -> str:
    """Format a seconds count as ``"  5.4s"`` or ``"  3.2min"`` (right-padded)."""
    if seconds >= 60.0:
        return f"{seconds / 60.0:5.1f}min"
    return f"{seconds:6.1f}s"


def _parallel_with_total(*, n_jobs: int, total_tasks: int, verbose: int = 5):
    """Build a joblib ``Parallel`` that prints "Done N/total" every 500 tasks.

    Parameters
    ----------
    n_jobs : int
        Forwarded to ``Parallel`` (``-1`` for all CPUs).
    total_tasks : int
        Used to print a ratio in the progress line.
    verbose : int, default ``5``

    Returns
    -------
    Parallel
        A subclass of ``joblib.Parallel`` with a custom ``print_progress``
        that limits the report frequency.
    """
    from joblib import Parallel

    class _ParallelWithTotal(Parallel):
        """Joblib pool that throttles progress prints to every 500 tasks."""
        def __init__(self, *args, total_tasks: int, **kwargs):
            super().__init__(*args, **kwargs)
            self._total_tasks = int(total_tasks)
            self._next_progress_report = 500

        def print_progress(self):
            completed = int(self.n_completed_tasks)
            if self._total_tasks <= 0 or completed <= 0:
                return super().print_progress()
            if completed < self._total_tasks and completed < self._next_progress_report:
                return
            while self._next_progress_report <= completed:
                self._next_progress_report += 500
            start_time = getattr(self, "_start_time", time.time())
            elapsed = time.time() - start_time
            self._print(
                f"Done {completed:5d} tasks of {self._total_tasks}"
                f" | elapsed: {_format_parallel_elapsed(elapsed)}"
            )

    return _ParallelWithTotal(
        n_jobs=n_jobs,
        verbose=verbose,
        total_tasks=total_tasks,
    )


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
    """3-D scatter snapshot of the domain at a single simulation time.

    Parameters
    ----------
    self : ShakerMakerData
    time : float, default ``0.0``
        Simulation time in seconds. The closest available step is used.
    component : {'z', 'e', 'n', 'resultant'}, default ``'z'``
        ``'resultant'`` plots the magnitude of the (E, N, Z) vector.
    data_type : {'vel', 'accel', 'disp'}, default ``'vel'``
    cmap : str, default ``'RdBu_r'``
    figsize : tuple, default ``(12, 8)``
    elev, azim : float
        3-D view angles in degrees.
    s : float, default ``20``
        Scatter marker size.
    alpha : float, default ``0.85``
    axis_equal : bool, default ``False``
        Force matplotlib's "equal" aspect ratio (slower for huge clouds).
    interpolate : bool, default ``False``
        When ``True`` the scattered nodes are interpolated onto a regular
        2-D grid (using :meth:`ShakerMakerData._interpolate_to_grid`) and
        rendered as a dense scatter on the active plane. Useful for thin
        slabs or surface grids.
    interp_method : {'linear', 'cubic', 'nearest'}, default ``'linear'``
    interp_resolution : int, default ``300``
        Grid resolution along each axis.

    Returns
    -------
    None
        Calls ``plt.show()`` directly.
    """
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


    from joblib import delayed

    dt        = self.time[1] - self.time[0]
    n         = self._n_nodes
    cache_key = (data_type,)

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

        _filename       = self.filename
        _data_grp       = self._data_grp
        _hdf5_path      = {'accel': f'{_data_grp}/acceleration',
                           'vel':   f'{_data_grp}/velocity',
                           'disp':  f'{_data_grp}/displacement'}[data_type]
        _window_mask    = getattr(self, '_window_mask',    None)
        _resample_cache = getattr(self, '_resample_cache', None)
        _time_len       = len(self.time)

        if use_safe_mode:
            def _compute_spectrum(i):
                # Local imports — each loky worker process starts fresh,
                # so every symbol used here must be imported inside the function.
                from scipy.interpolate import interp1d as _interp1d
                from ShakerMakerResults.analysis.newmark import (
                    NewmarkSpectrumAnalyzer as _NSA,
                )
                import h5py as _h5py
                import numpy as _np

                with _h5py.File(_filename, 'r') as _f:
                    _d = _f[_hdf5_path][3*i : 3*i+3, :]
                _d = _d[[2, 0, 1], :]
                if _window_mask is not None:
                    _d = _d[:, _window_mask]
                elif _resample_cache is not None:
                    _t_orig = _resample_cache['time_orig']
                    _rs = _np.zeros((3, _time_len))
                    for _k in range(3):
                        _rs[_k] = _interp1d(_t_orig, _d[_k],
                                            kind='linear',
                                            fill_value='extrapolate')(
                            _np.linspace(_t_orig[0], _t_orig[-1], _time_len))
                    _d = _rs
                specs = [_NSA.compute(_d[k], dt) for k in range(3)]
                T  = specs[0]['T']
                sa = {qty: _np.array([sp[qty] for sp in specs])
                      for qty in ('PSa', 'Sa', 'PSv', 'Sv', 'Sd')}
                return T, sa
        else:
            print("  Loading data into memory...")
            all_data = np.zeros((n, 3, len(self.time)))
            for i in range(n):
                all_data[i] = self.get_node_data(i, data_type)
            print("  Data loaded. Computing spectra...")

            def _compute_spectrum(i):
                # Local import — needed in every worker process spawned by loky.
                from ShakerMakerResults.analysis.newmark import (
                    NewmarkSpectrumAnalyzer as _NSA,
                )
                import numpy as _np

                _data = all_data[i]
                specs = [_NSA.compute(_data[k], dt) for k in range(3)]
                T  = specs[0]['T']
                sa = {qty: _np.array([sp[qty] for sp in specs])
                      for qty in ('PSa', 'Sa', 'PSv', 'Sv', 'Sd')}
                return T, sa

        results = _parallel_with_total(n_jobs=n_jobs, total_tasks=n, verbose=5)(
            delayed(_compute_spectrum)(i) for i in range(n))

        T_array = results[0][0]
        sa_full = {qty: np.array([r[1][qty] for r in results])
                   for qty in ('PSa', 'Sa', 'PSv', 'Sv', 'Sd')}

        self._newmark_cache[cache_key] = (T_array, sa_full)
        print(f"Done. All spectral quantities cached for {data_type}")

    sp_data    = sa_full[spectral_type]
    comp_lbl   = {'z': 'Vertical (Z)', 'e': 'East (E)',
                  'n': 'North (N)', 'resultant': 'Resultant'}
    xyz_t      = _rotate(self.xyz)
    x = xyz_t[:, 0]; y = xyz_t[:, 1]; z = xyz_t[:, 2]

    # Support list or single string for component
    components = component if isinstance(component, list) else [component]
    n_comp     = len(components)
    fig        = plt.figure(figsize=(figsize[0] * n_comp, figsize[1]))

    for idx, comp in enumerate(components):
        comp = comp.lower()
        if comp == 'resultant':
            sa_map = np.array([
                np.mean([np.interp(T_target, T_array, sp_data[i][k])
                         for k in range(3)])
                for i in range(n)]) * factor
        else:
            k_idx  = {'z': 0, 'e': 1, 'n': 2}[comp]
            sa_map = np.array([
                np.interp(T_target, T_array, sp_data[i][k_idx])
                for i in range(n)]) * factor

        sa_map = np.nan_to_num(sa_map, nan=0.0)
        clbl   = comp_lbl[comp]
        print(f"  {spectral_type}(T={T_target}s) | {comp} | "
              f"Max={sa_map.max():.4f}  Min={sa_map.min():.4f}")

        ax = fig.add_subplot(1, n_comp, idx + 1, projection='3d')
        sc = ax.scatter(x, y, z, c=sa_map, cmap=cmap, s=s, alpha=alpha,
                        vmin=0, vmax=np.nanmax(sa_map))
        fig.colorbar(sc, ax=ax, shrink=0.5,
                     label=f'{spectral_type}(T={T_target}s)')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.grid(False)
        if axis_equal:
            ax.axis('equal')
        ax.set_title(f'{spectral_type}(T={T_target}s) | {clbl}',
                     fontweight='bold')
        ax.view_init(elev=elev, azim=azim)

    plt.suptitle(self.name, fontsize=13, fontweight='bold')
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

    Parameters
    ----------
    component : str or list of str, default 'z'
        One or more of {'z', 'e', 'n', 'resultant'}.
    data_type : {'accel', 'vel', 'disp'}, default 'accel'
    factor : float, default 1.0
    cmap : str, default 'hot_r'
    figsize : tuple, default (12, 8)
    elev, azim : float
    s : int, default 20
    alpha : float, default 0.85
    axis_equal : bool, default False
    n_jobs : int, default -1
    """
    from joblib import delayed
    from ...analysis.arias_intensity import AriasIntensityAnalyzer

    dt        = self.time[1] - self.time[0]
    n         = self._n_nodes
    cache_key = (data_type, 'arias')

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

        _filename       = self.filename
        _data_grp       = self._data_grp
        _hdf5_path      = {'accel': f'{_data_grp}/acceleration',
                           'vel':   f'{_data_grp}/velocity',
                           'disp':  f'{_data_grp}/displacement'}[data_type]
        _window_mask    = getattr(self, '_window_mask',    None)
        _resample_cache = getattr(self, '_resample_cache', None)
        _time_len       = len(self.time)

        if use_safe_mode:
            def _compute_arias(i):
                # Loky spawns fresh worker processes, so every symbol we touch
                # has to be importable from inside the function -- relying on
                # module-level globals breaks the pickled closure.
                from scipy.interpolate import interp1d as _interp1d
                from ShakerMakerResults.analysis.arias_intensity import (
                    AriasIntensityAnalyzer as _AIA,
                )
                import h5py as _h5py
                import numpy as _np

                with _h5py.File(_filename, 'r') as _f:
                    _d = _f[_hdf5_path][3*i : 3*i+3, :]
                _d = _d[[2, 0, 1], :]
                if _window_mask is not None:
                    _d = _d[:, _window_mask]
                elif _resample_cache is not None:
                    _t_orig = _resample_cache['time_orig']
                    _rs = _np.zeros((3, _time_len))
                    for _k in range(3):
                        _rs[_k] = _interp1d(_t_orig, _d[_k],
                                            kind='linear',
                                            fill_value='extrapolate')(
                            _np.linspace(_t_orig[0], _t_orig[-1], _time_len))
                    _d = _rs
                ia = _np.zeros(3)
                for k in range(3):
                    _, _, _, ia_total, _ = _AIA.compute(_d[k] / 9.81, dt)
                    ia[k] = ia_total
                return ia
        else:
            print("  Loading data into memory...")
            all_data = np.zeros((n, 3, len(self.time)))
            for i in range(n):
                all_data[i] = self.get_node_data(i, data_type)
            print("  Data loaded. Computing Arias intensity...")

            def _compute_arias(i):
                # Fast path: loky workers still need a self-contained closure.
                from ShakerMakerResults.analysis.arias_intensity import (
                    AriasIntensityAnalyzer as _AIA,
                )
                import numpy as _np

                data = all_data[i]
                ia   = _np.zeros(3)
                for k in range(3):
                    _, _, _, ia_total, _ = _AIA.compute(data[k] / 9.81, dt)
                    ia[k] = ia_total
                return ia

        results = _parallel_with_total(n_jobs=n_jobs, total_tasks=n, verbose=5)(
            delayed(_compute_arias)(i) for i in range(n))

        ia_full = np.array(results)
        self._newmark_cache[cache_key] = ia_full
        print(f"Done. Arias intensity cached for {data_type}")

    # --- Plot ---
    components = component if isinstance(component, list) else [component]
    n_comps    = len(components)

    xyz_t = _rotate(self.xyz)
    x = xyz_t[:, 0]; y = xyz_t[:, 1]; z = xyz_t[:, 2]

    comp_labels = {'z': 'Vertical (Z)', 'e': 'East (E)',
                   'n': 'North (N)', 'resultant': 'Resultant'}

    fig = plt.figure(figsize=(figsize[0] * n_comps, figsize[1]))

    for idx, comp in enumerate(components):
        comp = comp.lower()
        if comp == 'resultant':
            ia_map = np.mean(ia_full, axis=1) * factor
        else:
            k      = {'z': 0, 'e': 1, 'n': 2}[comp]
            ia_map = ia_full[:, k] * factor

        ia_map = np.nan_to_num(ia_map, nan=0.0)
        clbl   = comp_labels[comp]

        print(f"  Arias | {comp} | factor={factor}  "
              f"Max={ia_map.max():.4f}  Min={ia_map.min():.4f}")

        ax = fig.add_subplot(1, n_comps, idx + 1, projection='3d')
        sc = ax.scatter(x, y, z, c=ia_map, cmap=cmap, s=s, alpha=alpha,
                        vmin=0, vmax=np.nanmax(ia_map))
        fig.colorbar(sc, ax=ax, shrink=0.5, label='Arias Intensity [m/s]')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.grid(False)
        if axis_equal:
            ax.axis('equal')
        ax.set_title(f'{self.name} | Arias | {clbl}', fontweight='bold')
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()
