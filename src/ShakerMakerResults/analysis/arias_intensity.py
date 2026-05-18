"""
arias_intensity.py
==================
Static analyzer for Arias intensity, significant duration and Araya-Saragoni
destructiveness potential.

Originally drafted 2025-05-01 alongside the response-spectrum work.
"""

__author__ = "Ing. Patricio Palacios B., M.Sc."
__version__ = "1.0.0"

import numpy as np

class AriasIntensityAnalyzer:
    """Static helper computing Arias intensity quantities from one trace."""

    @staticmethod
    def compute(signal, dt):
        """Compute Arias intensity and significant duration (5%-95%).

        Parameters
        ----------
        signal : np.ndarray
            Acceleration time series in **m/s^2**.
        dt : float
            Time step in seconds.

        Returns
        -------
        IA_percent : np.ndarray
            Normalised Arias intensity (0..100 %) as a function of time.
        t_start : float
            Time at 5 % of the total Arias intensity (start of significant
            shaking) [s].
        t_end : float
            Time at 95 % [s]. The 5..95 window is the significant duration.
        ia_total : float
            Total Arias intensity (the unnormalised value) [m/s].
        pot_dest : float
            Araya-Saragoni destructiveness potential ``Ia / nu_zero^2``,
            where ``nu_zero`` is the zero-crossing frequency [m * s].

        Notes
        -----
        The historical convention used ``g = 1`` (i.e. we assume the input
        is already in m/s^2 -- no division by gravity here). The
        ``signal = signal * 1`` line is kept as a defensive copy so that
        callers can safely pass views without us mutating them.
        """
        g = 1
        signal=signal*1  # defensive copy: avoid mutating the caller's array
        t = np.arange(len(signal)) * dt

        # Arias intensity curve (non-normalized)
        IA = (np.pi / (2 * g)) * np.cumsum(signal**2) * dt
        ia_total = IA[-1]

        # Normalized to 100%
        IA_percent = 100 * IA / ia_total

        # Significant duration (5%–95%)
        idx5 = np.argmax(IA_percent >= 5)
        idx95 = np.argmax(IA_percent >= 95)
        t_start = t[idx5]
        t_end = t[idx95]

        # Count zero crossings
        cont_pd = np.count_nonzero(np.diff(np.sign(signal)))
        freq_cross = cont_pd / t[-1] if t[-1] > 0 else 0

        # Destructiveness potential
        pot_dest = ia_total / (freq_cross**2) if freq_cross > 0 else 0

        return IA_percent, t_start, t_end, ia_total, pot_dest
