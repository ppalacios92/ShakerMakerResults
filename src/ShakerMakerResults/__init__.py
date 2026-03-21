"""
ShakerMakerResults
==================
Reader and visualisation toolkit for ShakerMaker HDF5 output files.

Supports DRM outputs, SurfaceGrid outputs, and real seismic station
recordings. File format is detected automatically for HDF5 outputs.

Modules
-------
shakermaker_data
    ``ShakerMakerData`` — unified HDF5 reader.
    ``DRMData``         — alias for DRM box outputs.
    ``SurfaceData``     — alias for SurfaceGrid outputs.
station_data
    ``StationData``     — reader for real station recordings (NPZ / HDF5).
plotting
    Multi-model plotting functions.
comparison
    Quantitative signal and spectral comparison functions.
newmark
    ``NewmarkSpectrumAnalyzer`` — β-Newmark response spectrum calculator.

Typical usage
-------------
>>> from shakermaker_results import DRMData, SurfaceData, StationData
>>> drm     = DRMData("DRM_5m_H1_s0.h5drm")
>>> surface = SurfaceData("Surface_10m_H1_s0.h5drm")
>>> station = StationData("station_H1.npz", name="H1 field")

>>> from shakermaker_results import compare_node_response
>>> compare_node_response([drm, surface, station], node_id='QA')
"""

from .shakermaker_data import ShakerMakerData
from .station_data     import StationData
from .newmark          import NewmarkSpectrumAnalyzer

from .plotting import (
    plot_models_response,
    plot_models_gf,
    plot_models_newmark,
    plot_models_tensor_gf,
    plot_models_DRM,
)

from .comparison import (
    compare_node_response,
    compare_spectra,
)

__all__ = [
    "ShakerMakerData", "DRMData", "SurfaceData", "StationData",
    "NewmarkSpectrumAnalyzer",
    "plot_models_response", "plot_models_gf",
    "plot_models_newmark_spectra", "plot_models_DRM", "plot_models_tensor_gf",
    "plot_combined_response", "compare_newmark", "compare_fourier", "plot_arias",
    "compare_node_response", "compare_spectra",
]