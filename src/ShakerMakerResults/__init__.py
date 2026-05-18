"""
ShakerMakerResults
==================
Reader and visualisation toolkit for ShakerMaker HDF5 output files.
Supports DRM outputs, SurfaceGrid outputs, and real seismic station
recordings. File format is detected automatically for HDF5 outputs.

The public API is loaded lazily so the package can be imported even when
optional viewer dependencies are not installed yet.

Public surface (lazy):

* Readers   -- ``ShakerMakerData``, ``DRMData``, ``SurfaceData``, ``StationData``
* Analysis  -- ``NewmarkSpectrumAnalyzer``, ``compute_vmax``
* Plotting  -- ``plot_node_*``, ``plot_domain*``, ``plot_surface*``,
  ``create_animation*``, ``plot_models_*``, ``compare_*``
* Viewer    -- ``ViewerSession``, ``ViewerState``, ``ViewerDataAdapter``

Every name in :data:`__all__` is registered in :data:`_EXPORTS` and only
imported on first access via the module-level :func:`__getattr__` hook.
That way ``import ShakerMakerResults`` stays cheap even if the user has not
installed PyVista/Qt -- the viewer dependencies are only pulled in when
``ViewerSession`` (or another viewer name) is touched.
"""

from importlib import import_module

__all__ = [
    "ShakerMakerData",
    "DRMData",
    "SurfaceData",
    "StationData",
    "NewmarkSpectrumAnalyzer",
    "compute_vmax",
    "plot_node_response",
    "plot_node_gf",
    "plot_node_tensor_gf",
    "plot_node_newmark",
    "plot_node_arias",
    "plot_domain",
    "plot_domain_calculated_t0",
    "plot_gf_connections",
    "plot_calculated_vs_reused",
    "plot_surface",
    "plot_surface_newmark",
    "plot_surface_arias",
    "plot_surface_on_map",
    "create_animation",
    "create_animation_plane",
    "create_animation_map",
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
    "ShakerMakerData": (".core", "ShakerMakerData"),
    "DRMData": (".core", "DRMData"),
    "SurfaceData": (".core", "SurfaceData"),
    "StationData": (".core", "StationData"),
    "NewmarkSpectrumAnalyzer": (".analysis", "NewmarkSpectrumAnalyzer"),
    "compute_vmax": (".analysis", "compute_vmax"),
    "plot_node_response": (".plotting", "plot_node_response"),
    "plot_node_gf": (".plotting", "plot_node_gf"),
    "plot_node_tensor_gf": (".plotting", "plot_node_tensor_gf"),
    "plot_node_newmark": (".plotting", "plot_node_newmark"),
    "plot_node_arias": (".plotting", "plot_node_arias"),
    "plot_domain": (".plotting", "plot_domain"),
    "plot_domain_calculated_t0": (".plotting", "plot_domain_calculated_t0"),
    "plot_gf_connections": (".plotting", "plot_gf_connections"),
    "plot_calculated_vs_reused": (".plotting", "plot_calculated_vs_reused"),
    "plot_surface": (".plotting", "plot_surface"),
    "plot_surface_newmark": (".plotting", "plot_surface_newmark"),
    "plot_surface_arias": (".plotting", "plot_surface_arias"),
    "plot_surface_on_map": (".plotting", "plot_surface_on_map"),
    "create_animation": (".plotting", "create_animation"),
    "create_animation_plane": (".plotting", "create_animation_plane"),
    "create_animation_map": (".plotting", "create_animation_map"),
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
    """Lazy attribute hook -- imports the right submodule on first access.

    Parameters
    ----------
    name : str
        Attribute requested by the caller (e.g. ``"ShakerMakerData"``).

    Returns
    -------
    object
        The resolved class or function from the underlying submodule.

    Raises
    ------
    AttributeError
        If ``name`` is not registered in :data:`_EXPORTS`.
    """
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    """Expose lazy names to ``dir(ShakerMakerResults)`` and tab-completion."""
    return sorted(set(globals()) | set(__all__))


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
# We used to print the banner unconditionally, which meant every MPI rank
# spammed it -- a 64-process run gave 64 copies of the same block. The hook
# below keeps the banner intact for interactive work and notebooks but
# silences it on non-zero ranks, on plain CI (``CI=1``) and when the user
# sets ``SHAKERMAKER_NO_BANNER=1``. Detection covers mpi4py and the common
# launcher env vars (OpenMPI / MPICH / Slurm) without forcing mpi4py as a
# hard dependency.

def _process_rank() -> int:
    """Best-effort rank lookup. Returns 0 when not running under MPI.

    Returns
    -------
    int
        MPI rank if detectable, else ``0``.
    """
    import os as _os

    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"):
        val = _os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass

    # mpi4py only as a last resort: importing it has side effects.
    try:
        from mpi4py import MPI  # type: ignore
        return int(MPI.COMM_WORLD.Get_rank())
    except Exception:
        return 0


def _should_show_banner() -> bool:
    """Return ``True`` if this process is the one allowed to print the banner."""
    import os as _os

    if _os.environ.get("SHAKERMAKER_NO_BANNER"):
        return False
    if _os.environ.get("CI"):
        return False
    return _process_rank() == 0


def _print_banner() -> None:
    """Print the welcome block once, on the leader process only."""
    print(
        """
  ShakerMakerResults -- Visualization and Analysis Toolkit
  Built on top of Shakermaker Tool

  Version 1.0.0                        (c) 2026 All Rights Reserved

  Repository  :  https://github.com/ppalacios92/ShakerMakerResults
  ShakerMaker :  https://github.com/ppalacios92/ShakerMaker

  Patricio Palacios B. | Nicolas Mora Bowen | Jose Abell | Ladruno Team

  ********* (>'-')> Ladruno4ever  *********
"""
    )


if _should_show_banner():
    _print_banner()
