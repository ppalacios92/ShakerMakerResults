"""Reusable colormap preview strip with interactive arrow markers.

A small horizontal widget that paints the active matplotlib colormap as
a continuous gradient with small triangle markers underneath — visually
similar to the SNAP colour-legend editor.  No numeric labels are drawn
on top of the strip; the data range is communicated by the User min /
User max spinboxes in the surrounding section.

Mouse interactions
------------------

* **Left-click on the strip** (anywhere except on a marker) emits
  :pyattr:`clicked`.  The side-panel sections wire this to the full
  :class:`TransferFunctionDialog` so the user gets the comprehensive
  editor on demand.
* **Left-click on a marker** opens a :class:`QColorDialog` so the user
  can recolour that individual stop without leaving the side panel.
  The chosen colour is stored as a per-marker *override* and the marker
  is repainted in that colour.  :pyattr:`markerColorChanged` carries
  ``(index, hex)`` to any listener.
* **Right-click on a marker** clears the override and emits
  ``markerColorChanged(index, "")`` so listeners can drop the entry.

The widget keeps the override map **locally** — it never reaches into
:class:`ViewerSession` or :class:`ViewerState`.  Backend wiring (taking
those overrides and rebuilding the actual matplotlib colormap that the
3-D scene uses) is intentionally left for a follow-up task; this commit
delivers the UI plumbing so the integration point is a single
``markerColorChanged`` slot away.
"""

from __future__ import annotations

from typing import Dict, Optional

from ._imports import require_viewer_dependencies
from .colors import colormap_strip_bytes

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


# Geometry constants — the *default* compact size used by the form-row
# previews on the side-panel sections.  Larger consumers (e.g. the
# TransferFunctionDialog) can bump ``setMinimumHeight`` on the instance
# and the strip auto-uses the extra vertical space.
_DEFAULT_STRIP_HEIGHT = 16
_MARKER_HEIGHT = 7
_PADDING_X = 2
_DEFAULT_TOTAL_HEIGHT = _DEFAULT_STRIP_HEIGHT + _MARKER_HEIGHT + 4


class ColormapPreview(QtWidgets.QWidget):
    """Gradient strip + interactive arrow markers for a matplotlib cmap."""

    clicked = QtCore.Signal()
    markerColorChanged = QtCore.Signal(int, str)  # (index, hex)  "" = reset

    def __init__(self, name: str = "viridis", parent=None):
        super().__init__(parent)
        self.setObjectName("CmapPreview")
        self.setMinimumHeight(_DEFAULT_TOTAL_HEIGHT)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)  # custom right-click

        self._cmap = str(name)
        self._marker_color = QtGui.QColor("#666666")
        self._marker_count = 5
        # User-picked overrides for individual markers (index → QColor).
        # When a marker is in the dict it paints in that colour; otherwise
        # it uses ``self._marker_color`` (the theme-driven outline tone).
        self._marker_overrides: Dict[int, QtGui.QColor] = {}
        self.setToolTip(self._tooltip())

    # ── Public API ─────────────────────────────────────────────────────────

    def setColormap(self, name: str) -> None:  # noqa: N802 (Qt-style)
        name = str(name).strip() or self._cmap
        if name == self._cmap:
            return
        self._cmap = name
        self.setToolTip(self._tooltip())
        self.update()

    def setTickColor(self, color: str) -> None:  # noqa: N802
        self._marker_color = QtGui.QColor(color)
        self.update()

    def setMarkerCount(self, count: int) -> None:  # noqa: N802
        count = max(2, int(count))
        if count == self._marker_count:
            return
        self._marker_count = count
        # Drop overrides that no longer have a corresponding marker.
        self._marker_overrides = {
            idx: clr
            for idx, clr in self._marker_overrides.items()
            if 0 <= idx < self._marker_count
        }
        self.update()

    def markerOverrides(self) -> Dict[int, str]:  # noqa: N802
        """Snapshot of the current per-marker colour overrides (hex)."""
        return {idx: clr.name() for idx, clr in self._marker_overrides.items()}

    def clearMarkerOverrides(self) -> None:  # noqa: N802
        """Drop every per-marker colour override."""
        if not self._marker_overrides:
            return
        self._marker_overrides.clear()
        self.update()

    # ── Events ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):  # noqa: N802
        idx = self._marker_at(event.pos())
        button = event.button()
        if button == QtCore.Qt.LeftButton:
            if idx is not None:
                self._pick_color_for_marker(idx)
            else:
                # Left-click on the strip body (not on a marker) → defer
                # the full editor open to mouseReleaseEvent so a quick
                # press-without-release does not trigger the dialog.
                self._pending_strip_click = True
            event.accept()
            return
        if button == QtCore.Qt.RightButton and idx is not None:
            # Right-click on a marker → clear that override.
            self._reset_marker(idx)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if (
            event.button() == QtCore.Qt.LeftButton
            and getattr(self, "_pending_strip_click", False)
            and self.rect().contains(event.pos())
            and self._marker_at(event.pos()) is None
        ):
            self._pending_strip_click = False
            self.clicked.emit()
            event.accept()
            return
        self._pending_strip_click = False
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            self._paint_strip(painter)
            self._paint_markers(painter)
        finally:
            painter.end()

    # ── Internals ──────────────────────────────────────────────────────────

    def _tooltip(self) -> str:
        return (
            f"Colormap: {self._cmap}\n"
            "Left-click a triangle to pick that stop's colour.\n"
            "Right-click a triangle to reset its override.\n"
            "Left-click the strip body for the full colour-mapping dialog."
        )

    def _strip_rect(self) -> QtCore.QRect:
        # Reserve the bottom band for marker triangles; the gradient strip
        # fills everything above.  This keeps the markers anchored at the
        # bottom edge regardless of how tall the widget is laid out.
        reserved_below = _MARKER_HEIGHT + 4
        strip_h = max(self.height() - reserved_below - 2, 12)
        return QtCore.QRect(
            _PADDING_X,
            1,
            max(self.width() - 2 * _PADDING_X, 1),
            strip_h,
        )

    def _marker_positions(self) -> list[tuple[int, int]]:
        """Return ``[(index, x_center), …]`` for every painted marker."""
        rect = self._strip_rect()
        if self._marker_count < 2 or rect.width() < 4:
            return []
        usable_w = rect.width() - 1
        return [
            (i, rect.left() + int(round(usable_w * i / (self._marker_count - 1))))
            for i in range(self._marker_count)
        ]

    def _marker_at(self, pos: QtCore.QPoint) -> Optional[int]:
        """Index of the marker whose triangle contains *pos*, or ``None``."""
        rect = self._strip_rect()
        marker_top_y = rect.bottom() + 2
        marker_bottom_y = marker_top_y + _MARKER_HEIGHT
        # Allow a tiny vertical pad so clicking just on the strip's edge
        # still selects an arrow.
        if pos.y() < marker_top_y - 1 or pos.y() > marker_bottom_y + 2:
            return None
        half = max(_MARKER_HEIGHT // 2 + 1, 4)
        for idx, x in self._marker_positions():
            if abs(pos.x() - x) <= half + 1:
                return idx
        return None

    def _paint_strip(self, painter: QtGui.QPainter) -> None:
        rect = self._strip_rect()
        strip_w = max(rect.width(), 2)
        strip_h = max(rect.height(), 2)
        try:
            raw = colormap_strip_bytes(self._cmap, strip_w, strip_h)
        except Exception:
            # Fallback: neutral gray when matplotlib cannot resolve the cmap.
            painter.fillRect(rect, QtGui.QColor("#808080"))
            return
        image = QtGui.QImage(raw, strip_w, strip_h, QtGui.QImage.Format_RGBA8888)
        painter.drawImage(rect, image)
        # Thin border so the strip reads as a single tile against dark
        # and light themes alike.
        pen = QtGui.QPen(self._marker_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

    def _paint_markers(self, painter: QtGui.QPainter) -> None:
        rect = self._strip_rect()
        positions = self._marker_positions()
        if not positions:
            return

        marker_top_y = rect.bottom() + 2
        half = max(_MARKER_HEIGHT // 2 + 1, 4)

        # Override markers paint with their picked colour + a thin outline
        # so they read clearly against light AND dark themes.
        outline_pen = QtGui.QPen(self._marker_color)
        outline_pen.setWidth(1)

        for idx, x in positions:
            triangle = QtGui.QPolygon(
                [
                    QtCore.QPoint(x - half, marker_top_y + _MARKER_HEIGHT),
                    QtCore.QPoint(x + half, marker_top_y + _MARKER_HEIGHT),
                    QtCore.QPoint(x, marker_top_y),
                ]
            )
            override = self._marker_overrides.get(idx)
            if override is not None:
                painter.setPen(outline_pen)
                painter.setBrush(QtGui.QBrush(override))
            else:
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(self._marker_color))
            painter.drawPolygon(triangle)

    # ── Color picking ──────────────────────────────────────────────────────

    def _pick_color_for_marker(self, index: int) -> None:
        initial = self._marker_overrides.get(index) or self._marker_color
        color = QtWidgets.QColorDialog.getColor(
            initial,
            self,
            f"Marker {index + 1} — pick colour",
            QtWidgets.QColorDialog.ShowAlphaChannel,
        )
        if not color.isValid():
            return
        self._marker_overrides[index] = color
        self.update()
        self.markerColorChanged.emit(index, color.name())

    def _reset_marker(self, index: int) -> None:
        if index not in self._marker_overrides:
            return
        del self._marker_overrides[index]
        self.update()
        # Signal listeners with an empty hex so they know to drop the slot.
        self.markerColorChanged.emit(index, "")


__all__ = ["ColormapPreview"]
