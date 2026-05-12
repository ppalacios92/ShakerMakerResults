"""Read-only summary of the active pipeline object.

Mirrors ParaView's *Information* tab: factual data about whatever is
selected in the pipeline browser — time range, node counts, GF / map
presence, cache occupancy, current selection.  The panel never edits state;
it only reflects it.

Refreshed on every session update through :meth:`refresh`.
"""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .theme import active_palette

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


# Field key → (label shown in the table, resolver(session) -> str).
# Keeping the resolvers in one ordered tuple makes it trivial to add or
# reorder rows without touching the widget construction code.
_FIELD_KEYS: tuple[str, ...] = (
    "type", "source", "time", "nodes", "active_field", "gf", "map",
    "selection", "stations", "cache",
)


def _bytes_to_gib(value: float) -> float:
    return float(value) / 1_073_741_824 if value else 0.0


class InformationPanel(QtWidgets.QWidget):
    """Compact two-column key / value list."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setObjectName("InformationPanel")

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(3)
        layout.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self._labels: dict[str, QtWidgets.QLabel] = {}
        self._name_labels: list[QtWidgets.QLabel] = []

        rows = (
            ("type",         "Type"),
            ("source",       "Source"),
            ("time",         "Time"),
            ("nodes",        "Nodes"),
            ("active_field", "Active field"),
            ("gf",           "GF"),
            ("map",          "Map"),
            ("selection",    "Selection"),
            ("stations",     "Stations"),
            ("cache",        "Cache"),
        )
        for key, label in rows:
            name_lbl = QtWidgets.QLabel(label)
            value_lbl = QtWidgets.QLabel("—")
            value_lbl.setWordWrap(True)
            value_lbl.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
            )
            self._labels[key] = value_lbl
            self._name_labels.append(name_lbl)
            layout.addRow(name_lbl, value_lbl)
        # Apply colours after the labels exist so a theme switch can
        # re-tint them in one place.
        self._apply_label_styles()

        layout.addItem(QtWidgets.QSpacerItem(
            10, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        ))
        self.refresh("init")

    # ── Refresh ─────────────────────────────────────────────────────────────

    def _apply_label_styles(self) -> None:
        """(Re)apply label colours from the active palette.

        Called from :meth:`__init__` and again from :meth:`refresh` on a
        ``"full"`` reason so a runtime theme switch retints every row.
        """
        palette = active_palette()
        label_style = (
            f"color: {palette.text_muted}; font-size: 11px; font-weight: 600;"
        )
        value_style = (
            f"color: {palette.text}; font-size: 11px; "
            f'font-family: Consolas, "SF Mono", "Cascadia Mono", monospace;'
        )
        for lbl in self._name_labels:
            lbl.setStyleSheet(label_style)
        for lbl in self._labels.values():
            lbl.setStyleSheet(value_style)

    def refresh(self, reason: str = "full") -> None:
        if reason == "time":
            # Just update the time field — cheap, runs every playback tick.
            self._labels["time"].setText(self._time_text())
            return
        if reason == "full":
            # Theme might have changed — retint the rows.
            self._apply_label_styles()
        try:
            self._labels["type"].setText(self._type_text())
            self._labels["source"].setText(self._source_text())
            self._labels["time"].setText(self._time_text())
            self._labels["nodes"].setText(self._nodes_text())
            self._labels["active_field"].setText(self._active_field_text())
            self._labels["gf"].setText(self._gf_text())
            self._labels["map"].setText(self._map_text())
            self._labels["selection"].setText(self._selection_text())
            self._labels["stations"].setText(self._stations_text())
            self._labels["cache"].setText(self._cache_text())
        except Exception:
            # Never let an information-panel failure break the viewer.
            pass

    # ── Resolvers ───────────────────────────────────────────────────────────

    def _type_text(self) -> str:
        return "ShakerMakerData (root)"

    def _source_text(self) -> str:
        summary = self.session.adapter.summary()
        name = getattr(summary, "name", None) or "unknown"
        return str(name)

    def _time_text(self) -> str:
        adapter = self.session.adapter
        time = getattr(adapter, "time", None)
        if time is None or len(time) == 0:
            return "—"
        try:
            t0 = float(time[0])
            t1 = float(time[-1])
            dt = float(time[1] - time[0]) if len(time) > 1 else 0.0
            nt = int(len(time))
            cur = float(self.session.current_time())
            return f"{cur:.3f} s   ·   range {t0:.3f} – {t1:.3f} s   ·   NT {nt}   ·   dt {dt:.4g} s"
        except Exception:
            return "—"

    def _nodes_text(self) -> str:
        adapter = self.session.adapter
        try:
            total = int(len(adapter.node_ids))
        except Exception:
            return "—"
        parts = [f"{total} total"]
        try:
            int_ids = getattr(adapter, "internal_node_ids", None)
            ext_ids = getattr(adapter, "external_node_ids", None)
            qa_ids  = getattr(adapter, "qa_node_ids", None)
            if int_ids is not None:
                parts.append(f"int {len(int_ids)}")
            if ext_ids is not None:
                parts.append(f"ext {len(ext_ids)}")
            if qa_ids is not None:
                parts.append(f"qa {len(qa_ids)}")
        except Exception:
            pass
        return "   ·   ".join(parts)

    def _active_field_text(self) -> str:
        state = self.session.state
        demand = getattr(state, "demand", "—")
        component = getattr(state, "component", "—")
        vmin, vmax = self.session.current_color_limits()
        clamp = "clamp" if getattr(state, "clamp_enabled", False) else "auto"
        return f"{demand} / {component}   ·   {clamp} [{vmin:.3g}, {vmax:.3g}]"

    def _gf_text(self) -> str:
        adapter = self.session.adapter
        if not getattr(adapter, "has_gf", False):
            return "not loaded"
        try:
            n_sf = int(self.session.gf_subfault_count())
            return f"loaded   ·   {n_sf} subfaults"
        except Exception:
            return "loaded"

    def _map_text(self) -> str:
        adapter = self.session.adapter
        return "loaded" if getattr(adapter, "has_map", False) else "not loaded"

    def _selection_text(self) -> str:
        state = self.session.state
        sel = getattr(state, "selected_node", None)
        multi = getattr(state, "multi_selection", None) or ()
        if sel is None and not multi:
            return "—"
        parts = []
        if sel is not None:
            parts.append(f"Node {sel}")
        if multi:
            parts.append(f"multi {len(multi)}")
        return "   ·   ".join(parts)

    def _stations_text(self) -> str:
        try:
            tags = self.session.current_station_tags() or []
            return f"{len(tags)} entries"
        except Exception:
            return "—"

    def _cache_text(self) -> str:
        info = self.session.adapter.cache_info
        used = _bytes_to_gib(info["bytes"])
        budget = _bytes_to_gib(info["budget"])
        # float16 is the standard adapter precision; surface it so the user
        # knows the bytes/sample factor without digging into the adapter.
        return f"{used:.2f} / {budget:.2f} GiB   ·   precision float16"


# ── Dock wrapper ─────────────────────────────────────────────────────────────

class InformationDock(QtWidgets.QDockWidget):
    """:class:`QDockWidget` shell around :class:`InformationPanel`."""

    def __init__(self, session, parent=None):
        super().__init__("Information", parent)
        self.setObjectName("Dock_information")
        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.panel = InformationPanel(session, self)
        self.setWidget(self.panel)

    def refresh(self, reason: str = "full") -> None:
        self.panel.refresh(reason)


__all__ = ["InformationPanel", "InformationDock"]
