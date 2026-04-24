"""Color and scalar helpers for the interactive viewer."""

from __future__ import annotations

import numpy as np

BACKGROUND_PRESETS = {
    "White": "#ffffff",
    "Gray": "#d7d9dd",
    "Dark": "#1f2430",
}

COLORMAP_OPTIONS = ("RdBu_r", "viridis", "hot_r", "seismic")


def colormap_for_component(component: str) -> str:
    """Return a sensible colormap for the selected component."""
    return "viridis" if component == "resultant" else "RdBu_r"


def scalar_limits(values: np.ndarray, component: str) -> tuple[float, float]:
    """Return stable color limits for one scalar field."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0

    if component == "resultant":
        vmax = float(np.max(finite))
        if vmax <= 0.0:
            vmax = 1.0
        return 0.0, vmax

    vmax = float(np.max(np.abs(finite)))
    if vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax
