"""HDF5 reader and time-evolution helpers for FFSP rupture sources.

The viewer loads the ``best_realization`` group from an FFSPSource HDF5
file (produced by :class:`shakermaker.ffspsource.FFSPSource.write_hdf5`)
and exposes per-subfault arrays plus an ``instantaneous_slip(t)`` helper
so the fault can be animated in sync with the seismic propagation time.

Kept separate from :mod:`adapter` because rupture data has a completely
different ownership model: no HDF5 time-series cache, no per-pane
overrides, no shared state with the seismic ``ViewerDataAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


#: Static per-subfault scalar fields exposed to the UI.  ``slip_inst`` is
#: synthesised on the fly from ``slip`` + ``rupture_time`` + ``rise_time``
#: and is not in this list.
FAULT_FIELDS: tuple[str, ...] = (
    "slip",
    "rupture_time",
    "rise_time",
    "peak_time",
    "strike",
    "dip",
    "rake",
)


@dataclass
class RuptureSource:
    """In-memory FFSP fault realization.

    All arrays are flat ``(n_subfaults,)``; ``points`` is ``(n_subfaults, 3)``
    in **metres**, fault-local frame with the hypocentre at the origin.
    ``z`` is kept as the raw depth from the HDF5 (positive going down), so
    overlays can pipe the points through the seismic Display Transform —
    whose ``[0,0,-1]`` row already takes care of flipping depth-positive
    into the viewer's "negative-z = below surface" convention.  Standalone
    scenes (e.g. the dedicated FFSP tab) apply that flip themselves.
    """

    points: np.ndarray
    slip: np.ndarray
    rupture_time: np.ndarray
    rise_time: np.ndarray
    peak_time: np.ndarray
    strike: np.ndarray
    dip: np.ndarray
    rake: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_h5(cls, path: str | Path) -> "RuptureSource":
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "h5py is required to load FFSP rupture files. "
                "Install with `pip install h5py`."
            ) from exc

        path = Path(path)
        with h5py.File(str(path), "r") as f:
            if "best_realization" not in f:
                raise ValueError(
                    f"{path.name}: missing /best_realization group — "
                    "is this an FFSPSource HDF5 export?"
                )
            br = f["best_realization"]
            for name in ("slip", "rupture_time", "rise_time", "x", "y", "z"):
                if name not in br:
                    raise ValueError(
                        f"{path.name}: /best_realization is missing required "
                        f"dataset '{name}'."
                    )

            slip         = np.asarray(br["slip"][:], dtype=np.float32)
            rupture_time = np.asarray(br["rupture_time"][:], dtype=np.float32)
            # A zero rise_time would NaN the slip ramp; clamp to a tiny
            # positive epsilon so the worst-case subfault still renders.
            rise_raw     = np.asarray(br["rise_time"][:], dtype=np.float32)
            rise_time    = np.where(rise_raw > 1e-6, rise_raw, 1e-6).astype(np.float32)
            peak_time    = np.asarray(br["peak_time"][:], dtype=np.float32) \
                           if "peak_time" in br else np.zeros_like(slip)
            strike       = np.asarray(br["strike"][:], dtype=np.float32) \
                           if "strike" in br else np.zeros_like(slip)
            dip          = np.asarray(br["dip"][:], dtype=np.float32) \
                           if "dip" in br else np.zeros_like(slip)
            rake         = np.asarray(br["rake"][:], dtype=np.float32) \
                           if "rake" in br else np.zeros_like(slip)

            # Points: stored as (x, y, z) in metres, hypocentre-origin.
            # Keep z positive (= depth below surface) so the seismic
            # Display Transform (``[[0,1,0],[1,0,0],[0,0,-1]]``) applies
            # consistently when the overlay is projected onto the main
            # scene — the matrix's ``z=-1`` row is what flips depth into
            # the viewer's downward direction.
            x = np.asarray(br["x"][:], dtype=np.float32)
            y = np.asarray(br["y"][:], dtype=np.float32)
            z = np.asarray(br["z"][:], dtype=np.float32)
            points = np.column_stack([x, y, z]).astype(np.float32)

            params: dict[str, Any] = {}
            if "parameters" in f:
                for key, val in f["parameters"].attrs.items():
                    if hasattr(val, "item") and getattr(val, "shape", ()) == ():
                        params[str(key)] = val.item()
                    else:
                        params[str(key)] = val

        return cls(
            points=points,
            slip=slip,
            rupture_time=rupture_time,
            rise_time=rise_time,
            peak_time=peak_time,
            strike=strike,
            dip=dip,
            rake=rake,
            params=params,
            source_path=path,
        )

    # ── Geometry ────────────────────────────────────────────────────────────

    @property
    def n_subfaults(self) -> int:
        return int(self.slip.shape[0])

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the point cloud."""
        p = self.points
        return (float(p[:, 0].min()), float(p[:, 0].max()),
                float(p[:, 1].min()), float(p[:, 1].max()),
                float(p[:, 2].min()), float(p[:, 2].max()))

    def hypocenter_xyz(self) -> tuple[float, float, float]:
        """Position of the subfault with the earliest rupture time."""
        idx = int(np.argmin(self.rupture_time))
        x, y, z = self.points[idx]
        return float(x), float(y), float(z)

    # ── Static fields ───────────────────────────────────────────────────────

    def field(self, name: str) -> np.ndarray:
        if name == "slip":
            return self.slip
        if name == "rupture_time":
            return self.rupture_time
        if name == "rise_time":
            return self.rise_time
        if name == "peak_time":
            return self.peak_time
        if name == "strike":
            return self.strike
        if name == "dip":
            return self.dip
        if name == "rake":
            return self.rake
        raise KeyError(f"Unknown rupture field: {name!r}")

    def field_limits(self, name: str) -> tuple[float, float]:
        arr = self.field(name)
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    # ── Time evolution ──────────────────────────────────────────────────────

    def instantaneous_slip(self, t: float) -> np.ndarray:
        """Per-subfault slip accumulated up to propagation time *t*.

        Linear ramp from ``rupture_time`` to ``rupture_time + rise_time``.
        Cheap vectorised NumPy; ~0.3 ms for 32k subfaults.
        """
        progress = np.clip(
            (float(t) - self.rupture_time) / self.rise_time,
            0.0, 1.0,
        )
        return (self.slip * progress).astype(np.float32)

    def rupture_mask(self, t: float) -> np.ndarray:
        """Boolean mask: ``True`` where the subfault has started rupturing."""
        return self.rupture_time <= float(t)

    def time_limits(self) -> tuple[float, float]:
        """Internal time range of the rupture process (s)."""
        return 0.0, float((self.rupture_time + self.rise_time).max())


__all__ = ["RuptureSource", "FAULT_FIELDS"]
