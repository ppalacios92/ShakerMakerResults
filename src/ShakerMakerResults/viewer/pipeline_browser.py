"""Live "Scene Browser" tree — tabs, panes and auxiliary scene nodes.

The browser is a thin reflection of what the central
:class:`~.multi_view.TabbedMultiViewArea` currently holds:

    ▾ Active Views
       ▾ Main view             ← tab
          • 3D NE   accel/resultant · viridis · range
          • Top     vel/z · seismic · range
       ▸ View 2                 ← tab
    Stations (n)
    Selected Nodes (n)
    Green Functions
    Geographic

Selecting any node emits :pyattr:`activeNodeChanged` with a string
identifier the window uses to flip the Display dock's "Apply to" combo
between **Active pane**, **Current tab** and **All panes**.  That way
the user can right-click → ``Apply``  on a single pane without leaving
the tree.

The widget never holds references to live ViewPanes — it stores their
``id(pane)`` only and asks the multi-view to resolve them at click time.
This keeps the browser cheap to rebuild and immune to dangling pointer
issues when the user closes a tab or pane.
"""

from __future__ import annotations

from typing import Callable, Optional

from ._imports import require_viewer_dependencies
from .theme import active_palette

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


# ── Stable node identifiers ──────────────────────────────────────────────────
#: Sent through ``activeNodeChanged`` so the window can route the selection.
NODE_ROOT       = "root"
NODE_TAB        = "tab"            # qualified at runtime: "tab:<index>"
NODE_PANE       = "pane"           # qualified at runtime: "pane:<tab>/<index>"
NODE_STATIONS   = "stations"
NODE_SELECTION  = "selection"
NODE_GF         = "gf"
NODE_GEOGRAPHIC = "geographic"


_ROOT_LABEL  = "Active Views"


# ── SVG marker for tab / pane / auxiliary nodes ──────────────────────────────

_NODE_SVG_PATHS = {
    NODE_ROOT:       "M5 5h14v14H5V5zm2 2v10h10V7H7z",
    NODE_TAB:        "M3 4h18v4H3V4zm0 6h18v10H3V10z",
    NODE_PANE:       "M4 4h7v7H4V4zm9 0h7v7h-7V4zM4 13h7v7H4v-7zm9 0h7v7h-7v-7z",
    NODE_STATIONS:   "M12 2C8 2 5 5 5 9c0 5 7 13 7 13s7-8 7-13c0-4-3-7-7-7zm0 9.5a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5z",
    NODE_SELECTION:  "M12 2 4 7v10l8 5 8-5V7l-8-5zm0 3 5 3v8l-5 3-5-3V8l5-3z",
    NODE_GF:         "M4 5h16v2H4V5zm0 6h16v2H4v-2zm0 6h16v2H4v-2z",
    NODE_GEOGRAPHIC: "M12 2a10 10 0 1 0 .001 20.001A10 10 0 0 0 12 2zm-1 18a8 8 0 0 1 0-16v16zm2 0V4a8 8 0 0 1 0 16z",
}


def _icon_for(node_key: str, palette) -> QtGui.QIcon:
    color = palette.accent if node_key in (NODE_ROOT, NODE_TAB) else palette.text_2
    path = _NODE_SVG_PATHS.get(node_key)
    if path is None:
        return QtGui.QIcon()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
        f'viewBox="0 0 24 24"><path fill="{color}" d="{path}"/></svg>'
    ).encode("utf-8")
    pixmap = QtGui.QPixmap(16, 16)
    pixmap.fill(QtCore.Qt.transparent)
    pixmap.loadFromData(svg, "SVG")
    return QtGui.QIcon(pixmap)


# ── Tree widget ──────────────────────────────────────────────────────────────


class PipelineBrowser(QtWidgets.QTreeWidget):
    """Live tree mirroring TabbedMultiViewArea + a few auxiliary nodes.

    The window injects the multi-view via ``set_multi_view`` (so the
    constructor only needs the session); we keep the multi-view in a
    weak attribute and rebuild the tree on demand.
    """

    activeNodeChanged = QtCore.Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._multi_view = None
        self.setObjectName("PipelineBrowser")
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setIndentation(14)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setRootIsDecorated(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.refresh()

    # ── External wiring ─────────────────────────────────────────────────────

    def set_multi_view(self, multi_view) -> None:
        """Bind to a ``TabbedMultiViewArea`` instance and rebuild."""
        self._multi_view = multi_view
        self.refresh()

    # ── Tree (re)build ──────────────────────────────────────────────────────

    def refresh(self, reason: str | None = None) -> None:
        """Rebuild the tree from the current scene structure.

        ``reason`` is accepted to match the dock-page refresh signature
        used everywhere else; the implementation is reason-agnostic — a
        full rebuild is cheap because the tree only has a handful of
        nodes — except for ``"time"`` which is skipped to keep playback
        free of widget churn.
        """
        if reason == "time":
            return
        previous_key = self.active_node_key() or NODE_ROOT
        self.blockSignals(True)
        self.clear()
        palette = active_palette()

        root = QtWidgets.QTreeWidgetItem([_ROOT_LABEL])
        root.setData(0, QtCore.Qt.UserRole, NODE_ROOT)
        root.setIcon(0, _icon_for(NODE_ROOT, palette))
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)
        self.addTopLevelItem(root)

        # Tabs + panes — live structure from the multi-view.
        self._build_tabs(root, palette)

        # Auxiliary nodes (Stations, Selection set, GF, Geographic).
        self._build_auxiliary(palette)

        self.expandAll()

        # Restore selection if the previously selected node still exists.
        self._select_node_key(previous_key)

        self.blockSignals(False)
        self._on_selection_changed()

    def _build_tabs(self, root: QtWidgets.QTreeWidgetItem, palette) -> None:
        tabs = self._tabs_widget()
        if tabs is None:
            return
        for tab_index in range(tabs.count()):
            area = tabs.widget(tab_index)
            tab_label = tabs.tabText(tab_index) or f"View {tab_index + 1}"
            tab_item = QtWidgets.QTreeWidgetItem([tab_label])
            tab_item.setData(0, QtCore.Qt.UserRole, f"{NODE_TAB}:{tab_index}")
            tab_item.setIcon(0, _icon_for(NODE_TAB, palette))
            root.addChild(tab_item)

            panes = list(getattr(area, "_panes", []) or [])
            for pane_index, pane in enumerate(panes):
                pane_label = getattr(pane, "label", None) or f"Pane {pane_index + 1}"
                pane_item = QtWidgets.QTreeWidgetItem([str(pane_label)])
                pane_item.setData(
                    0, QtCore.Qt.UserRole,
                    f"{NODE_PANE}:{tab_index}/{pane_index}",
                )
                pane_item.setIcon(0, _icon_for(NODE_PANE, palette))
                # Bold the pane label so it stands out from the property
                # rows that hang below it.
                font = pane_item.font(0)
                font.setBold(True)
                pane_item.setFont(0, font)
                # Tint pane rows that carry one or more per-pane overrides
                # so the user can scan the tree to spot "non-default" panes.
                pane_state = getattr(pane, "pane_state", None)
                if pane_state is not None and pane_state.overridden_names():
                    pane_item.setForeground(0, QtGui.QBrush(QtGui.QColor(palette.accent_dark)))
                tab_item.addChild(pane_item)

                # Add a property row per displayable attribute so the user
                # can read the pane's current visual state at a glance
                # (vertical list instead of a cramped horizontal sentence).
                self._append_pane_property_rows(pane_item, pane, palette)

    def _build_auxiliary(self, palette) -> None:
        # Auxiliary nodes live at the same top level as Active Views.
        stations = QtWidgets.QTreeWidgetItem([f"Stations ({self._station_count()})"])
        stations.setData(0, QtCore.Qt.UserRole, NODE_STATIONS)
        stations.setIcon(0, _icon_for(NODE_STATIONS, palette))
        self.addTopLevelItem(stations)

        selection = QtWidgets.QTreeWidgetItem(
            [f"Selected Nodes ({self._selection_count()})"]
        )
        selection.setData(0, QtCore.Qt.UserRole, NODE_SELECTION)
        selection.setIcon(0, _icon_for(NODE_SELECTION, palette))
        self.addTopLevelItem(selection)

        gf_status = "loaded" if self._has_gf() else "—"
        gf_item = QtWidgets.QTreeWidgetItem([f"Green Functions  ({gf_status})"])
        gf_item.setData(0, QtCore.Qt.UserRole, NODE_GF)
        gf_item.setIcon(0, _icon_for(NODE_GF, palette))
        if not self._has_gf():
            gf_item.setForeground(0, QtGui.QBrush(QtGui.QColor(palette.text_muted)))
        self.addTopLevelItem(gf_item)

        geo_item = QtWidgets.QTreeWidgetItem(["Geographic Transform"])
        geo_item.setData(0, QtCore.Qt.UserRole, NODE_GEOGRAPHIC)
        geo_item.setIcon(0, _icon_for(NODE_GEOGRAPHIC, palette))
        self.addTopLevelItem(geo_item)

    # ── Labels ──────────────────────────────────────────────────────────────

    def _append_pane_property_rows(self, pane_item, pane, palette) -> None:
        """Add child rows under *pane_item* — one row per visual property.

        The rows are read-only "leaf" nodes the user can scan to know
        exactly what the pane is rendering.  Each row uses a muted
        colour so the tree's primary structure (Active Views → tabs →
        panes) keeps the visual hierarchy.
        """
        state = getattr(pane, "pane_state", None)
        if state is None:
            return
        try:
            # Format displayable values.  Anything that fails defaults
            # to "—" so a missing attribute never breaks the tree.
            def _safe(name, default="—"):
                try:
                    value = getattr(state, name)
                    if value is None or value == "":
                        return default
                    return value
                except Exception:
                    return default

            try:
                vmin = float(_safe("user_vmin", float("nan")))
                vmax = float(_safe("user_vmax", float("nan")))
                range_text = f"[{vmin:.3g}, {vmax:.3g}]"
            except Exception:
                range_text = "—"

            warp = "on" if _safe("disp_warp_enabled", False) else "off"
            warp_scale = _safe("warp_scale")
            if warp == "on" and warp_scale not in ("—", None):
                try:
                    warp = f"on ×{float(warp_scale):g}"
                except Exception:
                    pass

            rows = [
                ("demand",     str(_safe("demand"))),
                ("component",  str(_safe("component"))),
                ("colormap",   str(_safe("colormap"))),
                ("range",      range_text),
                ("clamp",      "on" if _safe("clamp_enabled", False) else "off"),
                ("warp",       warp),
            ]

            muted = QtGui.QBrush(QtGui.QColor(palette.text_2))
            accent_muted = QtGui.QBrush(QtGui.QColor(palette.accent_dark))
            override_names = set(getattr(state, "overridden_names", lambda: ())())

            for key, value in rows:
                row = QtWidgets.QTreeWidgetItem([f"{key}:  {value}"])
                # Property rows share the pane's identifier so clicking
                # one still targets the pane.  Using the same identifier
                # also means the cached selection restore picks up the
                # parent (which is what we actually want to highlight).
                row.setData(
                    0, QtCore.Qt.UserRole,
                    pane_item.data(0, QtCore.Qt.UserRole),
                )
                # Tint rows that correspond to a per-pane override so the
                # user can spot non-default attributes at a glance.
                override_attr = {
                    "demand":    "demand",
                    "component": "component",
                    "colormap":  "colormap",
                    "range":     "user_vmin",     # user_vmax follows
                    "clamp":     "clamp_enabled",
                    "warp":      "disp_warp_enabled",
                }.get(key)
                if override_attr in override_names:
                    row.setForeground(0, accent_muted)
                else:
                    row.setForeground(0, muted)
                pane_item.addChild(row)
        except Exception:
            # Property rows are a UX nicety; if anything fails we just
            # leave the pane node as a leaf — the user still gets a
            # working tree.
            pass

    # ── Accessors ───────────────────────────────────────────────────────────

    def active_node_key(self) -> Optional[str]:
        items = self.selectedItems()
        if not items:
            return None
        return items[0].data(0, QtCore.Qt.UserRole)

    def _select_node_key(self, key: str) -> None:
        for index in range(self.topLevelItemCount()):
            root = self.topLevelItem(index)
            if self._select_in_subtree(root, key):
                return
        # Fall back to top-most item (Active Views).
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

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _tabs_widget(self):
        multi = self._multi_view
        if multi is None:
            return None
        return getattr(multi, "_tabs", None)

    def _station_count(self) -> int:
        try:
            return int(len(self.session.current_station_tags()))
        except Exception:
            return 0

    def _selection_count(self) -> int:
        try:
            return int(len(getattr(self.session.state, "multi_selection", ()) or ()))
        except Exception:
            return 0

    def _has_gf(self) -> bool:
        adapter = getattr(self.session, "adapter", None)
        return bool(
            getattr(adapter, "has_gf", False) and getattr(adapter, "has_map", False)
        )

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
        self.browser.refresh(reason)


__all__ = [
    "PipelineBrowser",
    "PipelineBrowserDock",
    "NODE_ROOT",
    "NODE_TAB",
    "NODE_PANE",
    "NODE_STATIONS",
    "NODE_SELECTION",
    "NODE_GF",
    "NODE_GEOGRAPHIC",
]
