"""Color and scalar helpers for the interactive viewer."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

BACKGROUND_PRESETS = {
    "White": "#ffffff",
    "Gray": "#d7d9dd",
    # Warm charcoal — matches the "hueso" dark theme so the renderer
    # background reads consistently with the rest of the UI in dark mode.
    "Dark": "#2a2724",
}

COLORMAP_OPTIONS = (
    "RdBu_r",
    "viridis",
    "hot_r",
    "seismic",
    "terrain",
    "plasma",
    "magma",
    "cividis",
    "coolwarm",
)


def effective_colormap_name(name: str, *, inverted: bool = False) -> str:
    """Return a Matplotlib colormap name honoring the viewer inversion flag."""
    cmap = str(name or "viridis")
    if inverted and not cmap.endswith("_r"):
        return f"{cmap}_r"
    if not inverted and cmap.endswith("_r"):
        return cmap[:-2]
    return cmap


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


@lru_cache(maxsize=16)
def _matplotlib_cmap(name: str):
    import matplotlib.pyplot as plt

    return plt.get_cmap(name)


def scalars_to_rgb(values: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> np.ndarray:
    """Map scalar values to RGB uint8 colors using a Matplotlib colormap."""
    values = np.asarray(values, dtype=np.float32)
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax > vmin:
        norm = (values - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(values, dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = _matplotlib_cmap(str(cmap_name))(norm)
    return (rgba[:, :3] * 255.0).astype(np.uint8, copy=False)


@lru_cache(maxsize=64)
def colormap_strip_bytes(name: str, width: int = 160, height: int = 14) -> bytes:
    """Return raw RGBA bytes for a horizontal 0→1 strip of *name*.

    The result is cached so the colormap preview widget can rebuild its
    pixmap quickly on every paint.
    """
    cmap = _matplotlib_cmap(str(name))
    gradient = np.linspace(0.0, 1.0, max(int(width), 2), dtype=np.float32)
    row = (cmap(gradient)[:, :4] * 255.0).astype(np.uint8, copy=False)
    img = np.broadcast_to(row[None, :, :], (max(int(height), 1), row.shape[0], 4))
    return np.ascontiguousarray(img).tobytes()
