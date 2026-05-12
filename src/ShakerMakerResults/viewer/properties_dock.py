"""Tabbed *Properties* dock that hosts the data-side editors.

Mirrors ParaView's *Properties* pane: a tabbed inspector that gathers the
editors operating on the active object's *data* properties (selection
node, visibility filters, warp settings, geographic transform).  Visual /
rendering properties (color map, vector field, static color, appearance)
live in their own docks on the right — keeping the "data on the left,
appearance on the right" rhythm the user asked for.

Every tab is one of the existing :mod:`.side_panel` sections (re-parented
into the dock, not re-implemented), so all behaviour, Apply buttons,
busy dialogs and refresh routing stay identical to the legacy side panel.
"""

from __future__ import annotations

from typing import Iterable

from ._imports import require_viewer_dependencies
from .side_panel import (
    GeographicSection,
    NodeSearchSection,
    VisualizationSection,
    _PageScrollArea,
)

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


class PropertiesPanel(QtWidgets.QTabWidget):
    """Tabbed inspector that exposes data-side editors for the active model.

    Tab order intentionally follows the typical workflow:

    1. **Node**       — pick / search a node, edit station tags
    2. **Visibility** — toggle internal / external / QA point classes
    3. **Geographic** — UTM reference + display transform matrix

    Warp used to live here too; it now has its own dock on the right
    side next to Display so the user does not have to dig into the data
    inspector to deform the scene.

    The active tab is preserved across :meth:`refresh` calls.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setObjectName("PropertiesPanel")
        self.setDocumentMode(True)
        self.setTabPosition(QtWidgets.QTabWidget.North)
        self.setMovable(False)

        # Build each section inside its own scroll area so long pages
        # (Stations table on Node, transform matrix on Geographic) don't
        # blow the dock's preferred size out of proportion.
        self.node_section       = NodeSearchSection(session)
        self.visibility_section = VisualizationSection(session)
        self.geographic_section = GeographicSection(session)

        self._tabs: list[tuple[str, QtWidgets.QWidget]] = [
            ("Node",       self.node_section),
            ("Visibility", self.visibility_section),
            ("Geographic", self.geographic_section),
        ]
        for title, widget in self._tabs:
            self.addTab(_PageScrollArea(widget), title)

    # ── Refresh routing ─────────────────────────────────────────────────────

    def refresh(self, reason: str = "full") -> None:
        for _, widget in self._tabs:
            try:
                widget.refresh(reason)
            except Exception:
                # Section-level failures must not break the broadcast loop.
                pass

    def all_sections(self) -> Iterable[QtWidgets.QWidget]:
        return (widget for _, widget in self._tabs)


# ── Dock wrapper ─────────────────────────────────────────────────────────────

class PropertiesDock(QtWidgets.QDockWidget):
    """:class:`QDockWidget` shell around :class:`PropertiesPanel`."""

    def __init__(self, session, parent=None):
        super().__init__("Properties", parent)
        self.setObjectName("Dock_properties")
        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.panel = PropertiesPanel(session, self)
        self.setWidget(self.panel)

    def refresh(self, reason: str = "full") -> None:
        self.panel.refresh(reason)

    # Convenience used by the pipeline-browser → active-node bridge.
    def focus_tab(self, tab_name: str) -> None:
        for index in range(self.panel.count()):
            if self.panel.tabText(index).lower() == tab_name.lower():
                self.panel.setCurrentIndex(index)
                return


__all__ = ["PropertiesPanel", "PropertiesDock"]
