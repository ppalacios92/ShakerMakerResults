"""
StationRead: Class for reading and processing seismic station data.
Supports NPZ and HDF5 formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import obspy
import h5py
from .newmark import NewmarkSpectrumAnalyzer

class StationRead:
    
    def __init__(self, filepath, station_index=0):
        """
        Load station data from NPZ or HDF5 file.
        
        Parameters
        ----------
        filepath : str
            Path to .npz or .h5/.hdf5 file
        station_index : int
            Station index to load (only used for HDF5 with multiple stations)
        """
        self.filepath = filepath
        self.name = None
        self.station_index = station_index
        self._load_data()
        self._init_cache()
    
    def _load_data(self):
        if self.filepath.endswith('.npz'):
            self._load_npz()
        elif self.filepath.endswith('.h5') or self.filepath.endswith('.hdf5'):
            self._load_hdf5()
        else:
            raise ValueError(f"Unsupported format: {self.filepath}. Use .npz or .h5")
    
    def _load_npz(self):
        """Load data from NPZ file."""
        data = np.load(self.filepath, allow_pickle=True)
        self.t = data['_t']
        self.dt = self.t[1] - self.t[0]
        self.z_v = data['_z']
        self.e_v = data['_e']
        self.n_v = data['_n']
        
        # Optional metadata
        if '_x' in data.files:
            self.x = data['_x']
        if '_metadata' in data.files:
            meta = data['_metadata']
            if meta.shape == ():
                meta = meta.item()
            if isinstance(meta, dict) and 'name' in meta:
                self.name = meta['name']
    
    def _load_hdf5(self):
        """Load data from HDF5 file."""
        with h5py.File(self.filepath, 'r') as hf:
            # Get metadata
            dt = hf['Metadata/dt'][()]
            tstart = hf['Metadata/tstart'][()]
            tend = hf['Metadata/tend'][()]
            
            # Build time vector
            self.t = np.arange(tstart, tend, dt)
            self.dt = dt
            
            # Get velocity data
            # HDF5 format: velocity[3*index, :] = E, velocity[3*index+1, :] = N, velocity[3*index+2, :] = Z
            velocity = hf['Data/velocity'][:]
            idx = self.station_index
            
            self.e_v = velocity[3 * idx, :]
            self.n_v = velocity[3 * idx + 1, :]
            self.z_v = velocity[3 * idx + 2, :]
            
            # Ensure time and velocity have same length
            min_len = min(len(self.t), len(self.z_v))
            self.t = self.t[:min_len]
            self.z_v = self.z_v[:min_len]
            self.e_v = self.e_v[:min_len]
            self.n_v = self.n_v[:min_len]
            
            # Optional: coordinates
            if 'Data/xyz' in hf:
                self.x = hf['Data/xyz'][idx, :]
    
    def get_window(self, tmin, tmax):
        """
        Create a new StationRead object with data trimmed to time window.
        
        Parameters
        ----------
        tmin : float
            Start time of window
        tmax : float
            End time of window
            
        Returns
        -------
        StationRead
            New object with trimmed data
        """
        import copy
        
        # Find indices for time window
        mask = (self.t >= tmin) & (self.t <= tmax)
        
        if not np.any(mask):
            raise ValueError(f"No data in time window [{tmin}, {tmax}]. Data range: [{self.t[0]}, {self.t[-1]}]")
        
        # Create new object without loading file
        new_station = object.__new__(StationRead)
        
        # Copy basic attributes
        new_station.filepath = self.filepath
        new_station.name = self.name
        new_station.station_index = self.station_index
        
        # Trim time and velocity data
        new_station.t = self.t[mask].copy()
        new_station.dt = self.dt
        new_station.z_v = self.z_v[mask].copy()
        new_station.e_v = self.e_v[mask].copy()
        new_station.n_v = self.n_v[mask].copy()
        
        # Copy coordinates if exists
        if hasattr(self, 'x'):
            new_station.x = self.x.copy()
        
        # Initialize fresh cache
        new_station._init_cache()
        
        # Copy filtered data if exists
        if self._filtered:
            new_station._filtered = True
            new_station._z_v_filt = self._z_v_filt[mask].copy()
            new_station._e_v_filt = self._e_v_filt[mask].copy()
            new_station._n_v_filt = self._n_v_filt[mask].copy()
        
        return new_station
    
    def _init_cache(self):
        # Derived quantities cache
        self._z_a = None
        self._e_a = None
        self._n_a = None
        self._z_d = None
        self._e_d = None
        self._n_d = None
        
        # Filtered data cache
        self._filtered = False
        self._z_v_filt = None
        self._e_v_filt = None
        self._n_v_filt = None
        self._z_a_filt = None
        self._e_a_filt = None
        self._n_a_filt = None
        self._z_d_filt = None
        self._e_d_filt = None
        self._n_d_filt = None
        
        # Fourier cache
        self._freqs = None
        self._fourier_cache = {}
        self._fourier_filt_cache = {}
        
        # Newmark cache
        self._newmark = None
        self._newmark_filt = None
    
    # ==================== DERIVED QUANTITIES ====================
    
    def _compute_acceleration(self):
        if self._z_a is None:
            n = len(self.t)
            self._z_a = np.zeros(n)
            self._e_a = np.zeros(n)
            self._n_a = np.zeros(n)
            self._z_a[1:] = (self.z_v[1:] - self.z_v[:-1]) / self.dt
            self._e_a[1:] = (self.e_v[1:] - self.e_v[:-1]) / self.dt
            self._n_a[1:] = (self.n_v[1:] - self.n_v[:-1]) / self.dt
    
    def _compute_displacement(self):
        if self._z_d is None:
            self._z_d = cumulative_trapezoid(self.z_v, self.t, initial=0.)
            self._e_d = cumulative_trapezoid(self.e_v, self.t, initial=0.)
            self._n_d = cumulative_trapezoid(self.n_v, self.t, initial=0.)
    
    @property
    def acceleration(self):
        self._compute_acceleration()
        return self._z_a, self._e_a, self._n_a
    
    @property
    def displacement(self):
        self._compute_displacement()
        return self._z_d, self._e_d, self._n_d
    
    @property
    def velocity(self):
        return self.z_v, self.e_v, self.n_v
    
    # ==================== FILTERING ====================
    
    def apply_filter(self, filter_type='bandpass', freqmin=0.25, freqmax=50.0, corners=4, zerophase=True):
        st = obspy.Stream()
        for vel in [self.z_v, self.e_v, self.n_v]:
            tr = obspy.Trace(data=vel.copy())
            tr.stats.delta = self.dt
            st.append(tr)
        
        if filter_type == 'bandpass':
            st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=corners, zerophase=zerophase)
        elif filter_type == 'lowpass':
            st.filter('lowpass', freq=freqmax, corners=corners, zerophase=zerophase)
        elif filter_type == 'highpass':
            st.filter('highpass', freq=freqmin, corners=corners, zerophase=zerophase)
        elif filter_type == 'bandstop':
            st.filter('bandstop', freqmin=freqmin, freqmax=freqmax, corners=corners, zerophase=zerophase)
        
        self._z_v_filt = st[0].data
        self._e_v_filt = st[1].data
        self._n_v_filt = st[2].data
        self._filtered = True
        
        # Reset filtered derived quantities
        self._z_a_filt = None
        self._e_a_filt = None
        self._n_a_filt = None
        self._z_d_filt = None
        self._e_d_filt = None
        self._n_d_filt = None
        self._fourier_filt_cache = {}
        self._newmark_filt = None
    
    def _compute_acceleration_filtered(self):
        if self._z_a_filt is None:
            n = len(self.t)
            self._z_a_filt = np.zeros(n)
            self._e_a_filt = np.zeros(n)
            self._n_a_filt = np.zeros(n)
            self._z_a_filt[1:] = (self._z_v_filt[1:] - self._z_v_filt[:-1]) / self.dt
            self._e_a_filt[1:] = (self._e_v_filt[1:] - self._e_v_filt[:-1]) / self.dt
            self._n_a_filt[1:] = (self._n_v_filt[1:] - self._n_v_filt[:-1]) / self.dt
    
    def _compute_displacement_filtered(self):
        if self._z_d_filt is None:
            self._z_d_filt = cumulative_trapezoid(self._z_v_filt, self.t, initial=0.)
            self._e_d_filt = cumulative_trapezoid(self._e_v_filt, self.t, initial=0.)
            self._n_d_filt = cumulative_trapezoid(self._n_v_filt, self.t, initial=0.)
    
    @property
    def acceleration_filtered(self):
        if self._filtered:
            self._compute_acceleration_filtered()
            return self._z_a_filt, self._e_a_filt, self._n_a_filt
        else:
            return self.acceleration
    
    @property
    def displacement_filtered(self):
        if self._filtered:
            self._compute_displacement_filtered()
            return self._z_d_filt, self._e_d_filt, self._n_d_filt
        else:
            return self.displacement
    
    @property
    def velocity_filtered(self):
        if self._filtered:
            return self._z_v_filt, self._e_v_filt, self._n_v_filt
        else:
            return self.velocity
    
    # ==================== FOURIER ====================
    
    def _compute_fourier(self, component='velocity', filtered=False):
        cache = self._fourier_filt_cache if filtered else self._fourier_cache
        
        if component in cache:
            return cache[component]
        
        # If filtered requested but no filter applied, use original data
        use_filtered = filtered and self._filtered
        
        if component == 'velocity':
            z, e, n = self.velocity_filtered if use_filtered else self.velocity
        elif component == 'acceleration':
            z, e, n = self.acceleration_filtered if use_filtered else self.acceleration
        elif component == 'displacement':
            z, e, n = self.displacement_filtered if use_filtered else self.displacement
        
        freqs = np.fft.rfftfreq(len(self.t), self.dt)
        z_amp = np.abs(np.fft.rfft(z)) * self.dt
        e_amp = np.abs(np.fft.rfft(e)) * self.dt
        n_amp = np.abs(np.fft.rfft(n)) * self.dt
        
        cache[component] = (freqs, z_amp, e_amp, n_amp)
        return cache[component]
    
    def get_fourier(self, component='velocity', filtered=False):
        return self._compute_fourier(component, filtered)
    
    # ==================== NEWMARK ====================
    
    def _compute_newmark(self, filtered=False):
        cache_attr = '_newmark_filt' if filtered else '_newmark'
        
        if getattr(self, cache_attr) is not None:
            return getattr(self, cache_attr)
        
        from EarthquakeSignal.core.newmark_spectrum_analyzer import NewmarkSpectrumAnalyzer
        
        z_a, e_a, n_a = self.acceleration_filtered if filtered else self.acceleration
        
        spec_z = NewmarkSpectrumAnalyzer.compute(z_a / 9.81, self.dt)
        spec_e = NewmarkSpectrumAnalyzer.compute(e_a / 9.81, self.dt)
        spec_n = NewmarkSpectrumAnalyzer.compute(n_a / 9.81, self.dt)
        
        result = {
            'T': spec_z['T'],
            'PSa_z': spec_z['PSa'],
            'PSa_e': spec_e['PSa'],
            'PSa_n': spec_n['PSa']
        }
        
        setattr(self, cache_attr, result)
        return result
    
    def get_newmark(self, filtered=False):
        return self._compute_newmark(filtered)
    
    # ==================== PLOTTING ====================
    
    def _get_label(self):
        return self.name if self.name else "Station"
    
    def plot_velocity(self, xlim=None, factor=1.0):
        z, e, n = self.velocity
        self._plot_time_series(z/factor, e/factor, n/factor, 
                               r"$\dot{u}$", "Velocity", xlim)
    
    def plot_acceleration(self, xlim=None, factor=9.81):
        z, e, n = self.acceleration
        self._plot_time_series(z/factor, e/factor, n/factor,
                               r"$\ddot{u}$ (g)" if factor==9.81 else r"$\ddot{u}$", 
                               "Acceleration", xlim)
    
    def plot_displacement(self, xlim=None, factor=1.0):
        z, e, n = self.displacement
        self._plot_time_series(z/factor, e/factor, n/factor,
                               r"$u$", "Displacement", xlim)
    
    def plot_velocity_filtered(self, xlim=None, factor=1.0):
        z, e, n = self.velocity_filtered
        self._plot_time_series(z/factor, e/factor, n/factor,
                               r"$\dot{u}$", "Velocity (Filtered)", xlim)
    
    def plot_acceleration_filtered(self, xlim=None, factor=9.81):
        z, e, n = self.acceleration_filtered
        self._plot_time_series(z/factor, e/factor, n/factor,
                               r"$\ddot{u}$ (g)" if factor==9.81 else r"$\ddot{u}$",
                               "Acceleration (Filtered)", xlim)
    
    def plot_displacement_filtered(self, xlim=None, factor=1.0):
        z, e, n = self.displacement_filtered
        self._plot_time_series(z/factor, e/factor, n/factor,
                               r"$u$", "Displacement (Filtered)", xlim)
    
    def _plot_time_series(self, z, e, n, ylabel, title, xlim):
        label = self._get_label()
        fig, axes = plt.subplots(3, 1, figsize=(10, 6))
        
        for ax, data, comp in zip(axes, [z, e, n], ['Z', 'E', 'N']):
            ax.plot(self.t, data, label=f"{comp} ({label})")
            ax.set_ylabel(f"{ylabel}$_{comp.lower()}$")
            ax.grid(True)
            ax.legend()
            if xlim:
                ax.set_xlim(xlim)
        
        axes[0].set_title(title, fontweight='bold')
        axes[-1].set_xlabel("Time, $t$ (s)")
        plt.tight_layout()
        plt.show()
    
    def plot_fourier(self, component='acceleration', xlim=None, factor=9.81):
        freqs, z_amp, e_amp, n_amp = self.get_fourier(component, filtered=False)
        self._plot_fourier_internal(freqs, z_amp/factor, e_amp/factor, n_amp/factor,
                                    f"Fourier - {component.capitalize()}", xlim)
    
    def plot_fourier_filtered(self, component='acceleration', xlim=None, factor=9.81):
        freqs, z_amp, e_amp, n_amp = self.get_fourier(component, filtered=True)
        self._plot_fourier_internal(freqs, z_amp/factor, e_amp/factor, n_amp/factor,
                                    f"Fourier - {component.capitalize()} (Filtered)", xlim)
    
    def _plot_fourier_internal(self, freqs, z_amp, e_amp, n_amp, title, xlim):
        label = self._get_label()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for ax, amp, comp in zip(axes, [z_amp, e_amp, n_amp], ['Z', 'E', 'N']):
            ax.plot(freqs, amp, label=label)
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
    
    def plot_newmark(self, xlim=None, filtered=False):
        spec = self.get_newmark(filtered)
        label = self._get_label()
        title = "Newmark Spectrum (Filtered)" if filtered else "Newmark Spectrum"
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for ax, psa, comp in zip(axes, [spec['PSa_z'], spec['PSa_e'], spec['PSa_n']], ['Z', 'E', 'N']):
            ax.plot(spec['T'], psa, label=label, linewidth=2)
            ax.set_xlabel('T (s)')
            ax.set_ylabel('Sa (g)')
            ax.set_title(f'{comp} Component')
            ax.legend()
            ax.grid(True)
            if xlim:
                ax.set_xlim(xlim)
        
        fig.suptitle(title, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_fourier_comparison(self, component='acceleration', xlim=None, factor=9.81):
        """Plot original vs filtered Fourier."""
        if not self._filtered:
            raise ValueError("No filter applied. Call apply_filter() first.")
        
        freqs, z_amp, e_amp, n_amp = self.get_fourier(component, filtered=False)
        freqs_f, z_amp_f, e_amp_f, n_amp_f = self.get_fourier(component, filtered=True)
        
        label = self._get_label()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for ax, amp, amp_f, comp in zip(axes, [z_amp, e_amp, n_amp], 
                                         [z_amp_f, e_amp_f, n_amp_f], ['Z', 'E', 'N']):
            ax.plot(freqs, amp/factor, '--', label=f'{label} Original')
            ax.plot(freqs_f, amp_f/factor, label=f'{label} Filtered')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_xscale('log')
            ax.set_title(f'{comp} Spectrum')
            ax.legend()
            ax.grid(True)
            if xlim:
                ax.set_xlim(xlim)
        
        fig.suptitle(f'Fourier Comparison - {component.capitalize()}', fontweight='bold')
        plt.tight_layout()
        plt.show()


# ==================== COMPARISON FUNCTIONS ====================

def compare_velocity(stations, xlim=None, factor=1.0):
    _compare_time_series(stations, 'velocity', xlim, factor, r"$\dot{u}$", "Velocity Comparison")

def compare_acceleration(stations, xlim=None, factor=9.81):
    ylabel = r"$\ddot{u}$ (g)" if factor == 9.81 else r"$\ddot{u}$"
    _compare_time_series(stations, 'acceleration', xlim, factor, ylabel, "Acceleration Comparison")

def compare_displacement(stations, xlim=None, factor=1.0):
    _compare_time_series(stations, 'displacement', xlim, factor, r"$u$", "Displacement Comparison")

def _compare_time_series(stations, data_type, xlim, factor, ylabel, title):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    
    for station in stations:
        label = station.name if station.name else "Station"
        if data_type == 'velocity':
            z, e, n = station.velocity
        elif data_type == 'acceleration':
            z, e, n = station.acceleration
        elif data_type == 'displacement':
            z, e, n = station.displacement
        
        for ax, data, comp in zip(axes, [z, e, n], ['Z', 'E', 'N']):
            ax.plot(station.t, data/factor, label=f"{comp} ({label})")
    
    for ax, comp in zip(axes, ['Z', 'E', 'N']):
        ax.set_ylabel(f"{ylabel}$_{comp.lower()}$")
        ax.grid(True)
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)
    
    axes[0].set_title(title, fontweight='bold')
    axes[-1].set_xlabel("Time, $t$ (s)")
    plt.tight_layout()
    plt.show()

def compare_fourier(stations, component='acceleration', xlim=None, factor=9.81):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for station in stations:
        label = station.name if station.name else "Station"
        freqs, z_amp, e_amp, n_amp = station.get_fourier(component, filtered=False)
        
        for ax, amp, comp in zip(axes, [z_amp, e_amp, n_amp], ['Z', 'E', 'N']):
            ax.plot(freqs, amp/factor, label=label)
    
    for ax, comp in zip(axes, ['Z', 'E', 'N']):
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_xscale('log')
        ax.set_title(f'{comp} Spectrum')
        ax.legend()
        ax.grid(True)
        if xlim:
            ax.set_xlim(xlim)
    
    fig.suptitle(f'Fourier Comparison - {component.capitalize()}', fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_newmark(stations, xlim=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for station in stations:
        label = station.name if station.name else "Station"
        spec = station.get_newmark(filtered=False)
        
        for ax, psa, comp in zip(axes, [spec['PSa_z'], spec['PSa_e'], spec['PSa_n']], ['Z', 'E', 'N']):
            ax.plot(spec['T'], psa, label=label, linewidth=2)
    
    for ax, comp in zip(axes, ['Z', 'E', 'N']):
        ax.set_xlabel('T (s)')
        ax.set_ylabel('Sa (g)')
        ax.set_title(f'{comp} Component')
        ax.legend()
        ax.grid(True)
        if xlim:
            ax.set_xlim(xlim)
    
    fig.suptitle('Newmark Spectrum Comparison', fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_velocity_filtered(stations, xlim=None, factor=1.0):
    _compare_time_series_filtered(stations, 'velocity', xlim, factor, r"$\dot{u}$", "Velocity Comparison (Filtered)")

def compare_acceleration_filtered(stations, xlim=None, factor=9.81):
    ylabel = r"$\ddot{u}$ (g)" if factor == 9.81 else r"$\ddot{u}$"
    _compare_time_series_filtered(stations, 'acceleration', xlim, factor, ylabel, "Acceleration Comparison (Filtered)")

def compare_displacement_filtered(stations, xlim=None, factor=1.0):
    _compare_time_series_filtered(stations, 'displacement', xlim, factor, r"$u$", "Displacement Comparison (Filtered)")

def _compare_time_series_filtered(stations, data_type, xlim, factor, ylabel, title):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    
    for station in stations:
        label = station.name if station.name else "Station"
        if data_type == 'velocity':
            z, e, n = station.velocity_filtered
        elif data_type == 'acceleration':
            z, e, n = station.acceleration_filtered
        elif data_type == 'displacement':
            z, e, n = station.displacement_filtered
        
        for ax, data, comp in zip(axes, [z, e, n], ['Z', 'E', 'N']):
            ax.plot(station.t, data/factor, label=f"{comp} ({label})")
    
    for ax, comp in zip(axes, ['Z', 'E', 'N']):
        ax.set_ylabel(f"{ylabel}$_{comp.lower()}$")
        ax.grid(True)
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)
    
    axes[0].set_title(title, fontweight='bold')
    axes[-1].set_xlabel("Time, $t$ (s)")
    plt.tight_layout()
    plt.show()

def compare_fourier_filtered(stations, component='acceleration', xlim=None, factor=9.81):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for station in stations:
        label = station.name if station.name else "Station"
        freqs, z_amp, e_amp, n_amp = station.get_fourier(component, filtered=True)
        
        for ax, amp, comp in zip(axes, [z_amp, e_amp, n_amp], ['Z', 'E', 'N']):
            ax.plot(freqs, amp/factor, label=label)
    
    for ax, comp in zip(axes, ['Z', 'E', 'N']):
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_xscale('log')
        ax.set_title(f'{comp} Spectrum')
        ax.legend()
        ax.grid(True)
        if xlim:
            ax.set_xlim(xlim)
    
    fig.suptitle(f'Fourier Comparison - {component.capitalize()} (Filtered)', fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_newmark_filtered(stations, xlim=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for station in stations:
        label = station.name if station.name else "Station"
        spec = station.get_newmark(filtered=True)
        
        for ax, psa, comp in zip(axes, [spec['PSa_z'], spec['PSa_e'], spec['PSa_n']], ['Z', 'E', 'N']):
            ax.plot(spec['T'], psa, label=label, linewidth=2)
    
    for ax, comp in zip(axes, ['Z', 'E', 'N']):
        ax.set_xlabel('T (s)')
        ax.set_ylabel('Sa (g)')
        ax.set_title(f'{comp} Component')
        ax.legend()
        ax.grid(True)
        if xlim:
            ax.set_xlim(xlim)
    
    fig.suptitle('Newmark Spectrum Comparison (Filtered)', fontweight='bold')
    plt.tight_layout()
    plt.show()