"""Helpers for importing optional viewer dependencies."""


def require_viewer_dependencies():
    """Return the GUI stack required by the interactive viewer.

    Raises
    ------
    ImportError
        If one or more viewer dependencies are missing.
    """
    missing = []

    try:
        import pyvista as pv
    except ImportError:
        pv = None
        missing.append("pyvista")

    try:
        from pyvistaqt import QtInteractor
    except ImportError:
        QtInteractor = None
        missing.append("pyvistaqt")

    try:
        import vtk
    except ImportError:
        vtk = None
        missing.append("vtk")

    try:
        from qtpy import QtCore, QtGui, QtWidgets
    except ImportError:
        QtCore = None
        QtGui = None
        QtWidgets = None
        missing.append("qtpy")

    if missing:
        deps = ", ".join(sorted(set(missing)))
        raise ImportError(
            "The interactive viewer requires optional dependencies not "
            f"currently installed: {deps}. "
            "Install the viewer extras with "
            "`pip install 'ShakerMakerResults[viewer]'`."
        )

    return pv, QtInteractor, vtk, QtCore, QtGui, QtWidgets
