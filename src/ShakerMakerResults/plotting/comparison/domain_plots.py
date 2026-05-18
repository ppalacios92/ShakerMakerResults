"""Comparison plots for model domains."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ...utils import _rotate

__all__ = ["plot_models_domain"]


def plot_models_domain(
    models,
    xlim=None,
    ylim=None,
    zlim=None,
    label_nodes=False,
    show="all",
    show_nodes=True,
    show_cubes=True,
    axis_equal=True,
    figsize=(10, 8),
):
    """Overlay several ShakerMakerData domains in a single 3-D figure.

    Parameters
    ----------
    models : list of ShakerMakerData
    xlim, ylim, zlim : tuple of float, optional
        Axis bounds in metres (display frame).
    label_nodes : bool, default ``False``
        Currently unused but kept for API symmetry with
        :func:`plotting.single_model.domain_plots.plot_domain`.
    show : {'all', 'internal', 'boundary'}, default ``'all'``
        Which DRM node category to render. ``'all'`` falls back to the
        union when a model has no internal nodes (SurfaceGrid layouts).
    show_nodes : bool, default ``True``
        Whether the per-node scatter is drawn. Disable to keep only the
        bounding cubes.
    show_cubes : bool, default ``True``
        Whether the bounding cube is drawn for each model.
    axis_equal : bool, default ``True``
    figsize : tuple, default ``(10, 8)``

    Returns
    -------
    None
        Calls ``plt.show()`` directly.
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for obj, color in zip(models, colors):
        xyz_t = _rotate(obj.xyz)
        xyz_qa_t = _rotate(obj.xyz_qa) if obj.xyz_qa is not None else None
        xyz_int = xyz_t[obj.internal]
        xyz_ext = xyz_t[~obj.internal]

        if show_nodes:
            if show in ("all", "boundary") and len(xyz_ext) > 0:
                ax.scatter(xyz_ext[:, 0], xyz_ext[:, 1], xyz_ext[:, 2], c=[color], marker="o", s=50, alpha=0.3)
            if show in ("all", "internal") and len(xyz_int) > 0:
                ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2], c=[color], marker="s", s=30, alpha=0.6)
            if show == "all" and len(xyz_int) == 0:
                ax.scatter(xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2], c=[color], marker="o", s=30, alpha=0.4)

        if xyz_qa_t is not None:
            ax.scatter(
                xyz_qa_t[:, 0],
                xyz_qa_t[:, 1],
                xyz_qa_t[:, 2],
                c=[color],
                marker="*",
                s=300,
                edgecolors="black",
                linewidths=2,
                label=obj.name,
            )

        if show_cubes:
            bbox = xyz_int if len(xyz_int) > 0 else xyz_t
            x0, x1 = bbox[:, 0].min(), bbox[:, 0].max()
            y0, y1 = bbox[:, 1].min(), bbox[:, 1].max()
            z0, z1 = bbox[:, 2].min(), bbox[:, 2].max()
            c = np.array(
                [
                    [x0, y0, z0],
                    [x1, y0, z0],
                    [x1, y1, z0],
                    [x0, y1, z0],
                    [x0, y0, z1],
                    [x1, y0, z1],
                    [x1, y1, z1],
                    [x0, y1, z1],
                ]
            )
            faces = [
                [c[4], c[5], c[6], c[7]],
                [c[0], c[1], c[5], c[4]],
                [c[2], c[3], c[7], c[6]],
                [c[0], c[3], c[7], c[4]],
                [c[1], c[2], c[6], c[5]],
            ]
            ax.add_collection3d(
                Poly3DCollection(faces, alpha=0.15, facecolor=color, edgecolor=color, linewidths=2)
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)
    ax.legend()
    ax.grid(False)
    if axis_equal is True:
        ax.axis("equal")
    plt.tight_layout()
    plt.show()
