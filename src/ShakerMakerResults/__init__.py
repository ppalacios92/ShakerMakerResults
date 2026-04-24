"""
ShakerMakerResults
==================
Reader and visualisation toolkit for ShakerMaker HDF5 output files.
Supports DRM outputs, SurfaceGrid outputs, and real seismic station
recordings. File format is detected automatically for HDF5 outputs.

The public API is loaded lazily so the package can be imported even when
optional viewer dependencies are not installed yet.
"""

from importlib import import_module

__all__ = [
    "ShakerMakerData",
    "DRMData",
    "SurfaceData",
    "StationData",
    "NewmarkSpectrumAnalyzer",
    "plot_models_response",
    "plot_models_newmark",
    "plot_models_gf",
    "plot_models_tensor_gf",
    "plot_models_domain",
    "plot_models_arias",
    "compare_node_response",
    "compare_spectra",
    "ViewerDataAdapter",
    "ViewerState",
    "ViewerSession",
]

_EXPORTS = {
    "ShakerMakerData": (".shakermaker_data", "ShakerMakerData"),
    "DRMData": (".shakermaker_data", "ShakerMakerData"),
    "SurfaceData": (".shakermaker_data", "ShakerMakerData"),
    "StationData": (".station_data", "StationData"),
    "NewmarkSpectrumAnalyzer": (".newmark", "NewmarkSpectrumAnalyzer"),
    "plot_models_response": (".plotting", "plot_models_response"),
    "plot_models_newmark": (".plotting", "plot_models_newmark"),
    "plot_models_gf": (".plotting", "plot_models_gf"),
    "plot_models_tensor_gf": (".plotting", "plot_models_tensor_gf"),
    "plot_models_domain": (".plotting", "plot_models_domain"),
    "plot_models_arias": (".plotting", "plot_models_arias"),
    "compare_node_response": (".comparison", "compare_node_response"),
    "compare_spectra": (".comparison", "compare_spectra"),
    "ViewerDataAdapter": (".viewer", "ViewerDataAdapter"),
    "ViewerState": (".viewer", "ViewerState"),
    "ViewerSession": (".viewer", "ViewerSession"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
