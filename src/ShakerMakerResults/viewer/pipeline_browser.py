"""ParaView-style pipeline browser for the viewer.

Today the viewer manages a single :class:`~ShakerMakerResults.ShakerMakerData`
model per window.  The pipeline browser surfaces the *structure* of that
model the same way ParaView surfaces sources / filters in its Pipeline
Browser dock:

    builtin:
      ◉ <project name>                   (root source — the loaded .h5drm)
         ├─ 📍 Stations  (n)             ← editable in Properties → Stations
         ├─ ◎ Selected Nodes (n)         ← multi-selection set
         ├─ ⚡ GF database  (✓ / —)      ← informational, drives demand=gf
         └─ 🧭 Geographic transform      ← UTM + transform matrix

Selecting a tree item activates the matching tab in the right-hand
*Properties* dock so the user can edit the relevant settings without
hunting around.  No data-loading code is touched — this widget is purely
a navigation surface over the existing :class:`ViewerSession` state.

The pipeline node identifiers are stable strings (``"root"``, ``"stations"``,
``"selection"``, ``"gf"``, ``"geographic"``) and travel through Qt's
``Qt::UserRole`` data slot.  They're forwarded to subscribers (the
properties dock, etc.) via the ``activeNodeChanged`` signal.
"""

from __future__ import annotations

from typing import Callable

from ._imports import require_viewer_dependencies
from .theme import active_palette

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


# ── Node identifiers used as Qt::UserRole payload ────────────────────────────

#: Strings that travel as the tree-item ``UserRole`` data.  These are the
#: stable contract between :class:`PipelineBrowserDock` and any subscriber.
NODE_ROOT       = "root"
NODE_STATIONS   = "stations"
NODE_SELECTION  = "selection"
NODE_GF         = "gf"
NODE_GEOGRAPHIC = "geographic"

#: Friendly default title shown for each node.  Counts are appended at runtime.
_NODE_TITLES = {
    NODE_ROOT:       "{name}",
    NODE_STATIONS:   "Stations ({n})",
    NODE_SELECTION:  "Selected Nodes ({n})",
    NODE_GF:         "Green Functions",
    NODE_GEOGRAPHIC: "Geographic Transform",
}


def _icon_for(node_key: str, palette) -> QtGui.QIcon:
    """Return a small monochrome SVG icon for *node_key*.

    Drawing the icons inline (instead of importing :mod:`.icons`) keeps the
    pipeline browser visually independent from the toolbar / dock chrome and
    lets us re-tint them on theme changes without invalidating other caches.
    """
    palette_color = palette.accent if node_key == NODE_ROOT else palette.text_2
    svg_paths = {
        NODE_ROOT:       "M5 5h14v14H5V5zm2 2v10h10V7H7z",                     # filled square
        NODE_STATIONS:   "M12 2C8 2 5 5 5 9c0 5 7 13 7 13s7-8 7-13c0-4-3-7-7-7zm0 9.5a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5z",
        NODE_SELECTION:  "M12 2 4 7v10l8 5 8-5V7l-8-5zm0 3 5 3v8l-5 3-5-3V8l5-3z",
        NODE_GF:          "M4 5h16v2H4V5zm0 6h16v2H4v-2zm0 6h16v2H4v-2z",        # equalizer-ish
        NODE_GEOGRAPHIC: "M12 2a10 10 0 1 0 .001 20.001A10 10 0 0 0 12 2zm-1 18a8 8 0 0 1 0-16v16zm2 0V4a8 8 0 0 1 0 16z",
    }
    path = svg_paths.get(node_key)
    if path is None:
        return QtGui.QIcon()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
        f'viewBox="0 0 24 24"><path fill="{palette_color}" d="{path}"/></svg>'
    ).encode("utf-8")
    pixmap = QtGui.QPixmap(16, 16)
    pixmap.fill(QtCore.Qt.transparent)
    pixmap.loadFromData(svg, "SVG")
    return QtGui.QIcon(pixmap)


# ── Widget ───────────────────────────────────────────────────────────────────

class PipelineBrowser(QtWidgets.QTreeWidget):
    """Tree widget that lists the active model and its auxiliary nodes.

    The widget owns no data of its own; it reads the live state of the
    :class:`ViewerSession` every time :meth:`refresh` is called and rebuilds
    the tree in-place (the tree is tiny — a handful of nodes — so a full
    rebuild is cheaper than a diff).
    """

    activeNodeChanged = QtCore.Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setObjectName("PipelineBrowser")
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setIndentation(14)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setRootIsDecorated(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # Click directly emits the change so subscribers don't wait for a
        # selectionChanged round-trip.
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.refresh()

    # ── Public API ──────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the tree from the current :class:`ViewerSession` state."""
        # Preserve the active node across the rebuild so it doesn't bounce
        # back to the root every time the model's counts change.
        previous = self.active_node_key()
        self.blockSignals(True)
        self.clear()
        palette = active_palette()

        summary = self.session.adapter.summary()
        project_name = getattr(summary, "name", "model") or "model"

        root = QtWidgets.QTreeWidgetItem([_NODE_TITLES[NODE_ROOT].format(name=project_name)])
        root.setData(0, QtCore.Qt.UserRole, NODE_ROOT)
        root.setIcon(0, _icon_for(NODE_ROOT, palette))
        root.setToolTip(0, f"Source: {project_name}")
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)
        self.addTopLevelItem(root)

        stations = QtWidgets.QTreeWidgetItem(
            [_NODE_TITLES[NODE_STATIONS].format(n=self._station_count())]
        )
        stations.setData(0, QtCore.Qt.UserRole, NODE_STATIONS)
        stations.setIcon(0, _icon_for(NODE_STATIONS, palette))
        root.addChild(stations)

        selection = QtWidgets.QTreeWidgetItem(
            [_NODE_TITLES[NODE_SELECTION].format(n=self._selection_count())]
        )
        selection.setData(0, QtCore.Qt.UserRole, NODE_SELECTION)
        selection.setIcon(0, _icon_for(NODE_SELECTION, palette))
        root.addChild(selection)

        gf_status = "loaded" if self._has_gf() else "—"
        gf_title = f"{_NODE_TITLES[NODE_GF]}  ({gf_status})"
        gf = QtWidgets.QTreeWidgetItem([gf_title])
        gf.setData(0, QtCore.Qt.UserRole, NODE_GF)
        gf.setIcon(0, _icon_for(NODE_GF, palette))
        if not self._has_gf():
            gf.setForeground(0, QtGui.QBrush(QtGui.QColor(palette.text_muted)))
        root.addChild(gf)

        geo = QtWidgets.QTreeWidgetItem([_NODE_TITLES[NODE_GEOGRAPHIC]])
        geo.setData(0, QtCore.Qt.UserRole, NODE_GEOGRAPHIC)
        geo.setIcon(0, _icon_for(NODE_GEOGRAPHIC, palette))
        root.addChild(geo)

        self.expandAll()

        # Restore selection (or default to root on first build).
        target = previous or NODE_ROOT
        self._select_node_key(target)
        self.blockSignals(False)
        # Always fire so listeners can sync on initial build too.
        self._on_selection_changed()

    def active_node_key(self) -> str | None:
        items = self.selectedItems()
        if not items:
            return None
        return items[0].data(0, QtCore.Qt.UserRole)

    def select_node(self, key: str) -> None:
        self._select_node_key(key)

    # ── Internals ───────────────────────────────────────────────────────────

    def _station_count(self) -> int:
        try:
            tags = self.session.current_station_tags()
            return int(len(tags))
        except Exception:
            return 0

    def _selection_count(self) -> int:
        try:
            multi = getattr(self.session.state, "multi_selection", None)
            return int(len(multi or ()))
        except Exception:
            return 0

    def _has_gf(self) -> bool:
        try:
            adapter = self.session.adapter
            return bool(getattr(adapter, "has_gf", False) and getattr(adapter, "has_map", False))
        except Exception:
            return False

    def _select_node_key(self, key: str) -> None:
        for index in range(self.topLevelItemCount()):
            root = self.topLevelItem(index)
            if self._select_in_subtree(root, key):
                return
        # Fall back to root if the requested node is not found.
        if self.topLevelItemCount():
            self.topLevelItem(0).setSelected(True)

    def _select_in_subtree(self, item: QtWidgets.QTreeWidgetItem, key: str) -> bool:
        if item.data(0, QtCore.Qt.UserRole) == key:
            self.clearSelection()
            item.setSelected(True)
            return True
        for child_index in range(item.childCount()):
            if self._select_in_subtree(item.child(child_index), key):
                return True
        return False

    def _on_selection_changed(self) -> None:
        key = self.active_node_key()
        if key is not None:
            self.activeNodeChanged.emit(key)


# ── Dock wrapper ─────────────────────────────────────────────────────────────

class PipelineBrowserDock(QtWidgets.QDockWidget):
    """:class:`QDockWidget` shell around :class:`PipelineBrowser`.

    Titled "Scene Browser" externally — the old "Pipeline Browser" name
    implied ParaView-style source/filter pipelines we do not actually
    implement.  The new name better describes what the tree shows: the
    objects currently composing the rendered scene.
    """

    def __init__(self, session, parent=None):
        super().__init__("Scene Browser", parent)
        self.setObjectName("Dock_pipeline")
        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.browser = PipelineBrowser(session, self)
        self.setWidget(self.browser)

    def refresh(self, reason: str = "full") -> None:
        """Match the refresh signature used by every other dock page."""
        # Anything that could change a count or status (selection, stations,
        # gf loaded) refreshes the tree.  ``time`` is the high-frequency
        # update path during playback — skip it.
        if reason == "time":
            return
        self.browser.refresh()


__all__ = [
    "PipelineBrowser",
    "PipelineBrowserDock",
    "NODE_ROOT",
    "NODE_STATIONS",
    "NODE_SELECTION",
    "NODE_GF",
    "NODE_GEOGRAPHIC",
]
