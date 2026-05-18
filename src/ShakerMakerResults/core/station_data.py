"""
station_data.py
===============
Reader for real seismic station recordings stored in NPZ or HDF5 format.

This class is intended for field recordings or any external velocity time
series that need to be compared against ShakerMaker simulation outputs
(``DRMData``, ``SurfaceData``).

Supported formats
-----------------
NPZ  (.npz)
    Arrays: ``_t``, ``_z``, ``_e``, ``_n``
    Optional: ``_x`` (coordinates), ``_metadata`` (dict with ``'name'``)

HDF5 (.h5 / .hdf5)
    Written by ``HDF5StationListWriter`` (ShakerMaker):
    ``/Data/velocity``  shape (3*N, Nt) — rows: E, N, Z per station
    ``/Metadata/{dt, tstart, tend}``

Author: Patricio Palacios B.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..analysis.newmark import NewmarkSpectrumAnalyzer


class StationData:
    """Reader for real seismic station recordings (NPZ or HDF5).

    Provides velocity, acceleration and displacement as lazy-computed
    properties, optional ObsPy-based filtering, Fourier spectra, and
    Newmark response spectra — all using the same interface expected by
    ``comparison.compare_node_response`` and the plotting helpers in
    ``plotting.py``.

    Parameters
    ----------
    filepath : str
        Path to ``.npz`` or ``.h5``/``.hdf5`` file.
    station_index : int, default ``0``
        Station index within the HDF5 file (ignored for NPZ).
    name : str, optional
        Display name.  If ``None``, inferred from file metadata or filename.

    Attributes
    ----------
    t : np.ndarray
        Time vector in seconds.
    dt : float
        Time step in seconds.
    z_v, e_v, n_v : np.ndarray
        Raw velocity components (Z-down, E, N).
    name : str

    Examples
    --------
    >>> sta = StationData("station_H1.npz", name="H1 field")
    >>> z_a, e_a, n_a = sta.acceleration
    >>> sta.plot_velocity()

    >>> sta.apply_filter('bandpass', freqmin=0.1, freqmax=10.0)
    >>> sta.plot_acceleration_filtered()
    """

    def __init__(self, filepath, station_index=0, name=None):
        self.filepath      = filepath
        self.station_index = station_index
        self.name          = name
        self._load_data()
        self._init_cache()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Dispatch to the right loader based on the file extension."""
        if self.filepath.endswith('.npz'):
            self._load_npz()
        elif self.filepath.endswith(('.h5', '.hdf5')):
            self._load_hdf5()
        else:
            raise ValueError(
                f"Unsupported format: {self.filepath}. Use .npz or .h5/.hdf5")

    def _load_npz(self):
        """Load from NPZ file (ShakerMaker station save format)."""
        data     = np.load(self.filepath, allow_pickle=True)
        self.t   = data['_t']
        self.dt  = float(self.t[1] - self.t[0])
        self.z_v = data['_z']
        self.e_v = data['_e']
        self.n_v = data['_n']

        if hasattr(self, 'x') is False and '_x' in data.files:
            self.x = data['_x']

        if self.name is None and '_metadata' in data.files:
            meta = data['_metadata']
            if meta.shape == ():
                meta = meta.item()
            if isinstance(meta, dict):
                self.name = meta.get('name', None)

    def _load_hdf5(self):
        """Load from HDF5 file written by HDF5StationListWriter."""
        with h5py.File(self.filepath, 'r') as hf:
            dt     = float(hf['Metadata/dt'][()])
            tstart = float(hf['Metadata/tstart'][()])
            tend   = float(hf['Metadata/tend'][()])

            self.t  = np.arange(tstart, tend, dt)
            self.dt = dt

            vel = hf['Data/velocity'][:]
            idx = self.station_index

            # Layout: rows 3*idx=E, 3*idx+1=N, 3*idx+2=Z
            self.e_v = vel[3 * idx,     :]
            self.n_v = vel[3 * idx + 1, :]
            self.z_v = vel[3 * idx + 2, :]

            # Trim to equal length
            n = min(len(self.t), len(self.z_v))
            self.t   = self.t[:n]
            self.z_v = self.z_v[:n]
            self.e_v = self.e_v[:n]
            self.n_v = self.n_v[:n]

            if 'Data/xyz' in hf:
                self.x = hf['Data/xyz'][idx, :]

        if self.name is None:
            self.name = self.filepath.split('/')[-1].replace('.h5', '')

    # ------------------------------------------------------------------
    # Cache initialisation
    # ------------------------------------------------------------------

    def _init_cache(self):
        """Reset all lazily-computed derived caches (filtered + unfiltered)."""
        # Unfiltered derived quantities
        self._z_a = self._e_a = self._n_a = None
        self._z_d = self._e_d = self._n_d = None

        # Filtered data
        self._filtered    = False
        self._z_v_filt    = None
        self._e_v_filt    = None
        self._n_v_filt    = None
        self._z_a_filt    = self._e_a_filt = self._n_a_filt = None
        self._z_d_filt    = self._e_d_filt = self._n_d_filt = None

        # Fourier caches
        self._fourier_cache      = {}
        self._fourier_filt_cache = {}

        # Newmark caches
        self._newmark      = None
        self._newmark_filt = None

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def get_window(self, tmin, tmax):
        """Return a new ``StationData`` trimmed to ``[tmin, tmax]``.

        Parameters
        ----------
        tmin, tmax : float
            Time window bounds in seconds.

        Returns
        -------
        StationData
        """
        mask = (self.t >= tmin) & (self.t <= tmax)
        if not np.any(mask):
            raise ValueError(
                f"No data in [{tmin}, {tmax}]s. "
                f"Data range: [{self.t[0]}, {self.t[-1]}]s")

        new               = object.__new__(StationData)
        new.filepath      = self.filepath
        new.name          = self.name
        new.station_index = self.station_index
        new.t             = self.t[mask].copy()
        new.dt            = self.dt
        new.z_v           = self.z_v[mask].copy()
        new.e_v           = self.e_v[mask].copy()
        new.n_v           = self.n_v[mask].copy()

        if hasattr(self, 'x'):
            new.x = self.x.copy()

        new._init_cache()

        if self._filtered:
            new._filtered  = True
            new._z_v_filt  = self._z_v_filt[mask].copy()
            new._e_v_filt  = self._e_v_filt[mask].copy()
            new._n_v_filt  = self._n_v_filt[mask].copy()

        return new

    # ------------------------------------------------------------------
    # Velocity / Acceleration / Displacement
    # ------------------------------------------------------------------

    def _compute_acceleration(self):
        """Fill the acceleration cache by forward-differencing the raw velocity."""
        if self._z_a is None:
            n          = len(self.t)
            self._z_a  = np.zeros(n)
            self._e_a  = np.zeros(n)
            self._n_a  = np.zeros(n)
            self._z_a[1:] = (self.z_v[1:] - self.z_v[:-1]) / self.dt
            self._e_a[1:] = (self.e_v[1:] - self.e_v[:-1]) / self.dt
            self._n_a[1:] = (self.n_v[1:] - self.n_v[:-1]) / self.dt

    def _compute_displacement(self):
        """Fill the displacement cache by trapezoidal integration of velocity."""
        if self._z_d is None:
            self._z_d = cumulative_trapezoid(self.z_v, self.t, initial=0.)
            self._e_d = cumulative_trapezoid(self.e_v, self.t, initial=0.)
            self._n_d = cumulative_trapezoid(self.n_v, self.t, initial=0.)

    @property
    def velocity(self):
        """Raw velocity (Z, E, N)."""
        return self.z_v, self.e_v, self.n_v

    @property
    def acceleration(self):
        """Acceleration derived by finite differences (Z, E, N)."""
        self._compute_acceleration()
        return self._z_a, self._e_a, self._n_a

    @property
    def displacement(self):
        """Displacement derived by cumulative trapezoidal integration (Z, E, N)."""
        self._compute_displacement()
        return self._z_d, self._e_d, self._n_d

    # ------------------------------------------------------------------
    # Filtering  (requires ObsPy)
    # ------------------------------------------------------------------

    def apply_filter(self, filter_type='bandpass', freqmin=0.25, freqmax=50.0,
                     corners=4, zerophase=True):
        """Apply an ObsPy filter to the velocity components.

        Parameters
        ----------
        filter_type : {'bandpass', 'lowpass', 'highpass', 'bandstop'}
        freqmin : float, default ``0.25``
        freqmax : float, default ``50.0``
        corners : int, default ``4``
        zerophase : bool, default ``True``
        """
        import obspy

        st = obspy.Stream()
        for vel in (self.z_v, self.e_v, self.n_v):
            tr             = obspy.Trace(data=vel.copy())
            tr.stats.delta = self.dt
            st.append(tr)

        if filter_type == 'bandpass':
            st.filter('bandpass', freqmin=freqmin, freqmax=freqmax,
                      corners=corners, zerophase=zerophase)
        elif filter_type == 'lowpass':
            st.filter('lowpass', freq=freqmax,
                      corners=corners, zerophase=zerophase)
        elif filter_type == 'highpass':
            st.filter('highpass', freq=freqmin,
                      corners=corners, zerophase=zerophase)
        elif filter_type == 'bandstop':
            st.filter('bandstop', freqmin=freqmin, freqmax=freqmax,
                      corners=corners, zerophase=zerophase)
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        self._z_v_filt = st[0].data
        self._e_v_filt = st[1].data
        self._n_v_filt = st[2].data
        self._filtered = True

        # Reset filtered derived caches
        self._z_a_filt = self._e_a_filt = self._n_a_filt = None
        self._z_d_filt = self._e_d_filt = self._n_d_filt = None
        self._fourier_filt_cache = {}
        self._newmark_filt       = None

    def _compute_acceleration_filtered(self):
        """Forward-difference the filtered velocity into the filtered accel cache."""
        if self._z_a_filt is None:
            n              = len(self.t)
            self._z_a_filt = np.zeros(n)
            self._e_a_filt = np.zeros(n)
            self._n_a_filt = np.zeros(n)
            self._z_a_filt[1:] = (self._z_v_filt[1:] - self._z_v_filt[:-1]) / self.dt
            self._e_a_filt[1:] = (self._e_v_filt[1:] - self._e_v_filt[:-1]) / self.dt
            self._n_a_filt[1:] = (self._n_v_filt[1:] - self._n_v_filt[:-1]) / self.dt

    def _compute_displacement_filtered(self):
        """Trapezoidal-integrate the filtered velocity into the filtered displ cache."""
        if self._z_d_filt is None:
            self._z_d_filt = cumulative_trapezoid(self._z_v_filt, self.t, initial=0.)
            self._e_d_filt = cumulative_trapezoid(self._e_v_filt, self.t, initial=0.)
            self._n_d_filt = cumulative_trapezoid(self._n_v_filt, self.t, initial=0.)

    @property
    def velocity_filtered(self):
        """Filtered velocity. Falls back to raw if no filter applied."""
        if self._filtered:
            return self._z_v_filt, self._e_v_filt, self._n_v_filt
        return self.velocity

    @property
    def acceleration_filtered(self):
        """Filtered acceleration. Falls back to raw if no filter applied."""
        if self._filtered:
            self._compute_acceleration_filtered()
            return self._z_a_filt, self._e_a_filt, self._n_a_filt
        return self.acceleration

    @property
    def displacement_filtered(self):
        """Filtered displacement. Falls back to raw if no filter applied."""
        if self._filtered:
            self._compute_displacement_filtered()
            return self._z_d_filt, self._e_d_filt, self._n_d_filt
        return self.displacement

    # ------------------------------------------------------------------
    # Fourier
    # ------------------------------------------------------------------

    def get_fourier(self, component='velocity', filtered=False):
        """Return (freqs, z_amp, e_amp, n_amp) for the given component.

        Parameters
        ----------
        component : {'velocity', 'acceleration', 'displacement'}
        filtered : bool, default ``False``

        Returns
        -------
        tuple of np.ndarray
        """
        cache = self._fourier_filt_cache if filtered else self._fourier_cache
        if component in cache:
            return cache[component]

        use_filt = filtered and self._filtered
        if component == 'velocity':
            z, e, n = self.velocity_filtered if use_filt else self.velocity
        elif component == 'acceleration':
            z, e, n = self.acceleration_filtered if use_filt else self.acceleration
        elif component == 'displacement':
            z, e, n = self.displacement_filtered if use_filt else self.displacement
        else:
            raise ValueError(f"Unknown component: {component}")

        freqs = np.fft.rfftfreq(len(self.t), self.dt)
        z_amp = np.abs(np.fft.rfft(z)) * self.dt
        e_amp = np.abs(np.fft.rfft(e)) * self.dt
        n_amp = np.abs(np.fft.rfft(n)) * self.dt

        cache[component] = (freqs, z_amp, e_amp, n_amp)
        return cache[component]

    # ------------------------------------------------------------------
    # Newmark
    # ------------------------------------------------------------------

    def get_newmark(self, filtered=False):
        """Compute (or return cached) Newmark response spectra.

        Parameters
        ----------
        filtered : bool, default ``False``

        Returns
        -------
        dict with keys: ``'T'``, ``'PSa_z'``, ``'PSa_e'``, ``'PSa_n'``
        """
        attr = '_newmark_filt' if filtered else '_newmark'
        if getattr(self, attr) is not None:
            return getattr(self, attr)

        z_a, e_a, n_a = self.acceleration_filtered if filtered else self.acceleration

        spec_z = NewmarkSpectrumAnalyzer.compute(z_a / 9.81, self.dt)
        spec_e = NewmarkSpectrumAnalyzer.compute(e_a / 9.81, self.dt)
        spec_n = NewmarkSpectrumAnalyzer.compute(n_a / 9.81, self.dt)

        result = {
            'T':     spec_z['T'],
            'PSa_z': spec_z['PSa'],
            'PSa_e': spec_e['PSa'],
            'PSa_n': spec_n['PSa'],
        }
        setattr(self, attr, result)
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _label(self):
        """Return ``self.name`` or the placeholder ``"Station"`` for plot legends."""
        return self.name if self.name else "Station"

    def _plot_3comp(self, z, e, n, ylabel, title, xlim, figsize=(10, 8)):
        """Stacked 3-row plot (Z / E / N) of a single quantity.

        Parameters
        ----------
        z, e, n : np.ndarray
            Component traces sharing ``self.t``.
        ylabel : str
            Y axis label template (LaTeX is fine).
        title : str
            Figure title.
        xlim : tuple of float or None
            Time bounds, ``None`` keeps autoscaling.
        figsize : tuple, default ``(10, 8)``
        """
        lbl = self._label()
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        for ax, data, comp in zip(axes, (z, e, n), ('Z', 'E', 'N')):
            ax.plot(self.t, data, label=f"{comp} ({lbl})")
            ax.set_ylabel(f"{ylabel}$_{comp.lower()}$")
            ax.grid(True)
            ax.legend()
            if xlim:
                ax.set_xlim(xlim)
        axes[0].set_title(title, fontweight='bold')
        axes[-1].set_xlabel("Time, $t$ (s)")
        plt.tight_layout()
        plt.show()

    def plot_velocity(self, xlim=None, factor=1.0, figsize=(10, 8)):
        """Plot raw velocity time series.

        Parameters
        ----------
        xlim : tuple of float, optional
            Time bounds in seconds.
        factor : float, default ``1.0``
            Divides the signal before plotting (use 100.0 to plot in cm/s).
        figsize : tuple, default ``(10, 8)``
        """
        z, e, n = self.velocity
        self._plot_3comp(z/factor, e/factor, n/factor,
                         r"$\dot{u}$", "Velocity", xlim, figsize)

    def plot_acceleration(self, xlim=None, factor=9.81, figsize=(10, 8)):
        """Plot acceleration time series.

        Parameters
        ----------
        xlim : tuple of float, optional
        factor : float, default ``9.81``
            Divisor applied before plotting. Default ``9.81`` converts m/s^2
            to g; pass ``1.0`` to keep m/s^2.
        figsize : tuple, default ``(10, 8)``
        """
        z, e, n = self.acceleration
        ylabel = r"$\ddot{u}$ (g)" if factor == 9.81 else r"$\ddot{u}$"
        self._plot_3comp(z/factor, e/factor, n/factor, ylabel, "Acceleration", xlim, figsize)

    def plot_displacement(self, xlim=None, factor=1.0, figsize=(10, 8)):
        """Plot displacement time series.

        Parameters
        ----------
        xlim : tuple of float, optional
        factor : float, default ``1.0``
        figsize : tuple, default ``(10, 8)``
        """
        z, e, n = self.displacement
        self._plot_3comp(z/factor, e/factor, n/factor,
                         r"$u$", "Displacement", xlim, figsize)

    def plot_velocity_filtered(self, xlim=None, factor=1.0, figsize=(10, 8)):
        """Plot the filtered velocity. Falls back to raw if no filter is set."""
        z, e, n = self.velocity_filtered
        self._plot_3comp(z/factor, e/factor, n/factor,
                         r"$\dot{u}$", "Velocity (Filtered)", xlim, figsize)

    def plot_acceleration_filtered(self, xlim=None, factor=9.81, figsize=(10, 8)):
        """Plot the filtered acceleration. Falls back to raw if no filter is set."""
        z, e, n = self.acceleration_filtered
        ylabel = r"$\ddot{u}$ (g)" if factor == 9.81 else r"$\ddot{u}$"
        self._plot_3comp(z/factor, e/factor, n/factor,
                         ylabel, "Acceleration (Filtered)", xlim, figsize)

    def plot_displacement_filtered(self, xlim=None, factor=1.0, figsize=(10, 8)):
        """Plot the filtered displacement. Falls back to raw if no filter is set."""
        z, e, n = self.displacement_filtered
        self._plot_3comp(z/factor, e/factor, n/factor,
                         r"$u$", "Displacement (Filtered)", xlim, figsize)

    def _plot_fourier_internal(self, freqs, z_amp, e_amp, n_amp, title, xlim, figsize=(12, 4)):
        """Render a 1x3 Fourier amplitude plot. Helper for ``plot_fourier*``."""
        lbl = self._label()
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, amp, comp in zip(axes, (z_amp, e_amp, n_amp), ('Z', 'E', 'N')):
            ax.plot(freqs, amp, label=lbl)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_xscale('log')
            ax.set_title(f'{comp} Spectrum')
            ax.legend()
            ax.grid(True)
            if xlim:
                ax.set_xlim(xlim)
        fig.suptitle(title, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_fourier(self, component='acceleration', xlim=None, factor=9.81, figsize=(12, 4)):
        """Plot the raw-signal Fourier amplitude spectrum.

        Parameters
        ----------
        component : {'velocity', 'acceleration', 'displacement'}, default ``'acceleration'``
        xlim : tuple of float, optional
            Frequency bounds in Hz.
        factor : float, default ``9.81``
            Divides the amplitude before plotting. The default (g) is
            consistent with the default of :meth:`plot_acceleration`.
        figsize : tuple, default ``(12, 4)``
        """
        freqs, z_amp, e_amp, n_amp = self.get_fourier(component, filtered=False)
        self._plot_fourier_internal(
            freqs, z_amp/factor, e_amp/factor, n_amp/factor,
            f"Fourier - {component.capitalize()}", xlim, figsize)

    def plot_fourier_filtered(self, component='acceleration', xlim=None, factor=9.81, figsize=(12, 4)):
        """Plot the filtered-signal Fourier amplitude spectrum. Same parameters as :meth:`plot_fourier`."""
        freqs, z_amp, e_amp, n_amp = self.get_fourier(component, filtered=True)
        self._plot_fourier_internal(
            freqs, z_amp/factor, e_amp/factor, n_amp/factor,
            f"Fourier - {component.capitalize()} (Filtered)", xlim, figsize)

    def plot_fourier_comparison(self, component='acceleration',
                                xlim=None, factor=9.81, figsize=(12, 4)):
        """Plot original vs filtered Fourier side by side.

        Raises
        ------
        ValueError
            If no filter has been applied yet.
        """
        if not self._filtered:
            raise ValueError("No filter applied. Call apply_filter() first.")

        freqs,   z_a,  e_a,  n_a  = self.get_fourier(component, filtered=False)
        freqs_f, z_af, e_af, n_af = self.get_fourier(component, filtered=True)
        lbl = self._label()

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, amp, ampf, comp in zip(
                axes, (z_a, e_a, n_a), (z_af, e_af, n_af), ('Z', 'E', 'N')):
            ax.plot(freqs,   amp  / factor, '--', label=f'{lbl} Original')
            ax.plot(freqs_f, ampf / factor,       label=f'{lbl} Filtered')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_xscale('log')
            ax.set_title(f'{comp} Spectrum')
            ax.legend()
            ax.grid(True)
            if xlim:
                ax.set_xlim(xlim)

        fig.suptitle(f'Fourier Comparison — {component.capitalize()}',
                     fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_newmark(self, xlim=None, filtered=False, figsize=(12, 4),
                     factor=1.0, spectral_type='PSa'):
        """Plot Newmark response spectra (1x3, Z / E / N).

        Parameters
        ----------
        xlim : tuple of float, optional
            Period bounds in seconds.
        filtered : bool, default ``False``
            Use filtered acceleration as input (requires ``apply_filter()``).
        figsize : tuple, default ``(12, 4)``
        factor : float, default ``1.0``
            Divides the spectrum values before plotting.
        spectral_type : {'PSa', 'Sa', 'PSv', 'Sv', 'Sd'}, default ``'PSa'``
            Currently only PSa is cached by :meth:`get_newmark`; the other
            spectral types fall back to the PSa data with their own y-axis
            label, kept for API compatibility with the multi-model plots.

        Notes
        -----
        The y-axis label uses pretty units (``"PSa (g)"`` etc.) but the
        actual data plotted is always the PSa series cached on this
        instance. If you need true Sa/PSv/Sv/Sd you have to compute them
        from acceleration via
        :func:`ShakerMakerResults.analysis.newmark.NewmarkSpectrumAnalyzer.compute`.
        """
        spec  = self.get_newmark(filtered)
        lbl   = self._label()
        title = "Newmark Spectrum (Filtered)" if filtered else "Newmark Spectrum"
        ylabel = {'PSa': 'PSa (g)', 'Sa': 'Sa (g)', 'PSv': 'PSv (m/s)',
                  'Sv': 'Sv (m/s)', 'Sd': 'Sd (m)'}.get(spectral_type, spectral_type)

        # get_newmark() only caches PSa today, so any spectral_type request
        # falls back to PSa data. The legend / axis label still respects the
        # user's choice so multi-model overlays line up.
        key_map = {'PSa': ('PSa_z', 'PSa_e', 'PSa_n'),
                   'Sa':  ('PSa_z', 'PSa_e', 'PSa_n')}
        keys = key_map.get(spectral_type, ('PSa_z', 'PSa_e', 'PSa_n'))

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, psa, comp in zip(axes, (spec[keys[0]], spec[keys[1]], spec[keys[2]]),
                                  ('Z', 'E', 'N')):
            ax.plot(spec['T'], psa / factor, label=lbl, linewidth=2)
            ax.set_xlabel('T (s)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{comp} — {spectral_type}')
            ax.legend()
            ax.grid(True)
            if xlim:
                ax.set_xlim(xlim)

        fig.suptitle(title, fontweight='bold')
        plt.tight_layout()
        plt.show()
