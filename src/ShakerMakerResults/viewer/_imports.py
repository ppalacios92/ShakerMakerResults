"""
_imports.py
===========
Single source of truth for "do I have the GUI stack installed?".

Every viewer module that needs PyVista / Qt / VTK calls
:func:`require_viewer_dependencies` first. That way we get one clean
``ImportError`` mentioning *which* package is missing and how to install
the extras, instead of a stack trace from deep inside a Qt event loop.
"""


def require_viewer_dependencies():
    """Return the GUI stack required by the interactive viewer.

    Returns
    -------
    tuple
        ``(pyvista, QtInteractor, vtk, QtCore, QtGui, QtWidgets)`` -- the
        modules / classes the rest of the viewer needs at import time.

    Raises
    ------
    ImportError
        If one or more of the optional dependencies (pyvista, pyvistaqt,
        vtk, qtpy) are missing. The error message lists which packages
        need to be installed and points to the ``[viewer]`` extras.
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
