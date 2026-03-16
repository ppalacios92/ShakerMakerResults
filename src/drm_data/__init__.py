"""
DRM_DATA: Domain Reduction Method data reader and visualization toolkit.
"""

__version__ = "0.1.0"
__author__ = "Patricio Palacios B."

from .drm import DRM
from .station import StationRead
from .newmark import NewmarkSpectrumAnalyzer
from .plotting import (
    plot_models_response,
    plot_models_gf,
    plot_models_f_spectrum,
    plot_models_newmark_spectra,
    plot_models_DRM,
    plot_models_tensor_gf,
    compare_models_node_response,
)

__all__ = [
    "DRM",
    "StationRead",
    "NewmarkSpectrumAnalyzer",
    "plot_models_response",
    "plot_models_gf",
    "plot_models_f_spectrum",
    "plot_models_newmark_spectra",
    "plot_models_DRM",
    "plot_models_tensor_gf",
    "compare_models_node_response",
]