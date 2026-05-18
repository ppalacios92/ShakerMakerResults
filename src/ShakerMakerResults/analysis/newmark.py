"""
newmark.py
==========
SDOF Newmark beta solver and response-spectrum analyzer.

The solver (:func:`solve_newmark`) uses the classical linear-acceleration
formulation (``gamma = 1/2``, ``beta = 1/4``) -- unconditionally stable
for the periods we care about.

The analyzer (:class:`NewmarkSpectrumAnalyzer`) wraps the solver in a
period sweep and returns a dict with all spectral quantities plus the
time histories at ``T = 1.0 s`` (handy for sanity-checks against the
literature).

History note: we used to import ``cumtrapz`` from ``scipy.integrate``;
SciPy renamed it to ``cumulative_trapezoid`` in 1.6+. The historical
import is left commented in the import block for traceability.
"""

import numpy as np
# from scipy.integrate import cumtrapz  # kept as history -- renamed in SciPy 1.6+
from scipy.integrate import cumulative_trapezoid  # noqa: F401  (kept for downstream re-imports)

from numba import njit


@njit
def solve_newmark(ag, dt, zeta, Tj):
    """Run one Newmark beta SDOF response computation.

    Parameters
    ----------
    ag : np.ndarray
        Ground acceleration time series in **m/s^2**.
    dt : float
        Time step in seconds.
    zeta : float
        Damping ratio (0.05 = 5%).
    Tj : float
        SDOF period in seconds.

    Returns
    -------
    tuple
        ``(Sd, Sv, Sa, PSv, PSa, u, v, a, at)`` where the first five are
        scalars (the spectral peak values) and the last four are the
        relative-displacement, relative-velocity, relative-acceleration
        and total-acceleration time histories.

    Notes
    -----
    Compiled with ``numba.njit``. The function is called inside a Python
    loop in :meth:`NewmarkSpectrumAnalyzer.compute`, so all numerics live
    here.
    """
    gama = 1/2
    beta = 1/4
    w = 2 * np.pi / Tj
    m = 1.0
    k = m * w**2
    c = 2 * m * w * zeta
    a1 = m / (beta * dt ** 2) + c * gama / (beta * dt)
    a2 = m / (beta * dt) + c * (gama / beta - 1)
    a3 = m * (1 / (2 * beta) - 1) + c * dt * (gama / (2 * beta) - 1)
    kp = k + a1

    u = np.zeros(len(ag))
    v = np.zeros(len(ag))
    a = np.zeros(len(ag))
    at = np.zeros(len(ag))

    for i in range(len(ag) - 1):
        p_eff = -m * ag[i] + a1 * u[i] + a2 * v[i] + a3 * a[i]
        u[i + 1] = p_eff / kp
        a[i + 1] = (u[i + 1] - u[i]) / (beta * dt ** 2) - v[i] / (beta * dt) - a[i] * (1 / (2 * beta) - 1)
        at[i + 1] = a[i + 1] + ag[i]
        v[i + 1] = v[i] + dt * ((1 - gama) * a[i] + gama * a[i + 1])

    Sd = np.max(np.abs(u))
    Sv = np.max(np.abs(v))
    Sa = np.max(np.abs(at))
    PSv = w * Sd
    PSa = w ** 2 * Sd

    return Sd, Sv, Sa, PSv, PSa, u, v, a, at


class NewmarkSpectrumAnalyzer:
    """SDOF Newmark beta response-spectrum analyzer (static API).

    The class has no state -- it groups :meth:`compute` and any future
    spectrum helpers under a single namespace so callers can write
    ``NewmarkSpectrumAnalyzer.compute(ag, dt)`` without instantiating.
    """

    @staticmethod
    def compute(ag, dt, zeta=0.05 , max_period=5.01 , intervals=0.01):
        """Compute the full SDOF response spectrum for a ground motion.

        Parameters
        ----------
        ag : np.ndarray
            Ground acceleration time series in **g** (the function
            converts to m/s^2 internally).
        dt : float
            Time step in seconds.
        zeta : float, default ``0.05``
            Damping ratio (0.05 = 5%).
        max_period : float, default ``5.01``
            Upper bound of the period sweep in seconds.
        intervals : float, default ``0.01``
            Period spacing in seconds.

        Returns
        -------
        dict
            ``'T'``   : np.ndarray, period sweep [s].
            ``'PSa'`` : np.ndarray, pseudo-acceleration spectrum [g].
            ``'PSv'`` : np.ndarray, pseudo-velocity spectrum [m/s].
            ``'Sd'``  : np.ndarray, displacement spectrum [m].
            ``'Sv'``  : np.ndarray, velocity spectrum [m/s].
            ``'Sa'``  : np.ndarray, acceleration spectrum [g].
            ``'u'``   : np.ndarray, displacement history at T~=1.0 s [m].
            ``'v'``   : np.ndarray, velocity history at T~=1.0 s [m/s].
            ``'a'``   : np.ndarray, relative-accel history at T~=1.0 s [g].
            ``'at'``  : np.ndarray, total-accel history at T~=1.0 s [g].

        Notes
        -----
        Periods that fail the stability check ``Tj > dt * pi * sqrt(2)``
        get PGA-floored values (``Sa = PSa = PGA``, the rest set to 0).
        Time histories are only populated for the period closest to 1.0 s.
        """
        T = np.arange(0.0, max_period, intervals)
        ag = np.asarray(ag) * 9.81  # convert from g to m/s²

        Sd, Sv, Sa, PSv, PSa = [], [], [], [], []
        u_hist, v_hist, a_hist, at_hist = [], [], [], []

        # Minimum stability check for the Newmark beta scheme
        gama = 1/2
        beta = 1/4
        q = dt * np.pi * np.sqrt(2) * np.sqrt(gama - 2 * beta)

        for Tj in T:
            if Tj > q:
                Sd_, Sv_, Sa_, PSv_, PSa_, u, v, a, at = solve_newmark(ag, dt, zeta, Tj)
                Sd.append(Sd_)
                Sv.append(Sv_)
                Sa.append(Sa_)
                PSv.append(PSv_)
                PSa.append(PSa_)

                if np.isclose(Tj, 1.0, atol=0.01):
                    u_hist = u
                    v_hist = v
                    a_hist = a
                    at_hist = at
            else:
                PGA = np.max(np.abs(ag))
                Sd.append(0)
                Sv.append(0)
                Sa.append(PGA)
                PSv.append(0)
                PSa.append(PGA)

        # Convert the per-period accumulators to NumPy arrays
        Sd = np.array(Sd)
        Sv = np.array(Sv)
        Sa = np.array(Sa) / 9.81
        PSv = np.array(PSv)
        PSa = np.array(PSa) / 9.81
        a_hist = np.array(a_hist) / 9.81
        at_hist = np.array(at_hist) / 9.81

        return {
            'T': T,
            'PSa': PSa,
            'PSv': PSv,
            'Sd': Sd,
            'Sv': Sv,
            'Sa': Sa,
            'u': np.array(u_hist),
            'v': np.array(v_hist),
            'a': a_hist,
            'at': at_hist
        }
