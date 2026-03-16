"""
ShakerMakerResults
==================
Reader and visualisation toolkit for ShakerMaker HDF5 output files.

Supports both DRM (Domain Reduction Method) outputs (written by
``DRMHDF5StationListWriter``) and plain station outputs (written by
``HDF5StationListWriter``). The file format is detected automatically.

Modules
-------
shakermaker_data
    ``ShakerMakerData`` — unified reader class.
plotting
    Multi-model plotting functions.
comparison
    Quantitative signal and spectral comparison functions.
newmark
    ``NewmarkSpectrumAnalyzer`` — β-Newmark response spectrum calculator.

Typical usage
-------------
>>> from shakermaker_results import ShakerMakerData
>>> result = ShakerMakerData("DRM_5m_H1_s0.h5drm")
>>> result.plot_node_response(node_id='QA')

>>> from shakermaker_results import plot_models_response
>>> plot_models_response([result_5m, result_10m], node_id='QA')

>>> from shakermaker_results import compare_node_response
>>> compare_node_response([result_5m, result_10m], node_id='QA')
"""

__version__ = "1.0.0"
__author__  = "Patricio Palacios B."

# Core reader
from .shakermaker_data import ShakerMakerData

# Newmark spectrum analyser
from .newmark import NewmarkSpectrumAnalyzer

# Multi-model plotting
from .plotting import (
    plot_models_response,
    plot_models_gf,
    plot_models_f_spectrum,
    plot_models_newmark_spectra,
    plot_models_DRM,
    plot_models_tensor_gf,
    plot_combined_response,
    compare_newmark,
    compare_fourier,
    plot_arias,
)

# Quantitative comparison
from .comparison import (
    compare_node_response,
    compare_spectra,
)

__all__ = [
    # Core
    "ShakerMakerData",
    "NewmarkSpectrumAnalyzer",
    # Plotting
    "plot_models_response",
    "plot_models_gf",
    "plot_models_f_spectrum",
    "plot_models_newmark_spectra",
    "plot_models_DRM",
    "plot_models_tensor_gf",
    "plot_combined_response",
    "compare_newmark",
    "compare_fourier",
    "plot_arias",
    # Comparison
    "compare_node_response",
    "compare_spectra",
]
