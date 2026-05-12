"""Reusable colormap preview strip with arrow markers.

A small horizontal widget that paints the active matplotlib colormap as
a continuous gradient with small triangle markers underneath — visually
similar to the SNAP colour-legend editor.  No numeric labels are drawn
on top of the strip; the data range is communicated by the User min /
User max spinboxes in the surrounding section.

The widget is **clickable** — clicking anywhere on the strip emits the
``clicked`` signal so the parent section can hand control off to a
richer editor (typically :class:`TransferFunctionDialog`).  Editing the
ramp directly inside the strip (drag-to-move a stop, double-click to
recolour, right-click to add / remove) is left for a follow-up — the
current version focuses on the requested visual: gradient + arrows,
no numbers.
"""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .colors import colormap_strip_bytes

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


# Geometry constants — small enough that several previews stack nicely
# inside a form layout row.
_STRIP_HEIGHT = 16
_MARKER_HEIGHT = 6
_PADDING_X = 2
_TOTAL_HEIGHT = _STRIP_HEIGHT + _MARKER_HEIGHT + 4


class ColormapPreview(QtWidgets.QWidget):
    """Gradient strip + arrow markers for a named matplotlib colormap.

    Inherits :class:`QWidget` (not :class:`QLabel`) so future iterations
    can hook mouse events directly onto the markers — drag-to-reposition,
    double-click to recolour, right-click to add / remove a stop, etc.
    """

    clicked = QtCore.Signal()

    def __init__(self, name: str = "viridis", parent=None):
        super().__init__(parent)
        self.setObjectName("CmapPreview")
        self.setMinimumHeight(_TOTAL_HEIGHT)
        self.setMaximumHeight(_TOTAL_HEIGHT + 2)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.setCursor(QtCore.Qt.PointingHandCursor)

        self._cmap = str(name)
        self._marker_color = QtGui.QColor("#666666")
        self._marker_count = 5
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
        self.update()

    # ── Events ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
            event.accept()
            return
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
            "Click to open the advanced colour editor."
        )

    def _strip_rect(self) -> QtCore.QRect:
        return QtCore.QRect(
            _PADDING_X,
            1,
            max(self.width() - 2 * _PADDING_X, 1),
            _STRIP_HEIGHT,
        )

    def _paint_strip(self, painter: QtGui.QPainter) -> None:
        rect = self._strip_rect()
        try:
            raw = colormap_strip_bytes(self._cmap, max(rect.width(), 2), _STRIP_HEIGHT)
        except Exception:
            # Fallback: neutral gray when matplotlib cannot resolve the cmap.
            painter.fillRect(rect, QtGui.QColor("#808080"))
            return
        image = QtGui.QImage(
            raw, max(rect.width(), 2), _STRIP_HEIGHT, QtGui.QImage.Format_RGBA8888
        )
        painter.drawImage(rect, image)
        # Thin border so the strip reads as a single tile against dark
        # and light themes alike.
        pen = QtGui.QPen(self._marker_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

    def _paint_markers(self, painter: QtGui.QPainter) -> None:
        rect = self._strip_rect()
        if self._marker_count < 2 or rect.width() < 4:
            return

        marker_top_y = rect.bottom() + 2
        half = max(_MARKER_HEIGHT // 2 + 1, 4)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(self._marker_color))

        usable_w = rect.width() - 1
        for i in range(self._marker_count):
            x = rect.left() + int(round(usable_w * i / (self._marker_count - 1)))
            # Triangle pointing up — apex touches the bottom edge of the strip.
            triangle = QtGui.QPolygon(
                [
                    QtCore.QPoint(x - half, marker_top_y + _MARKER_HEIGHT),
                    QtCore.QPoint(x + half, marker_top_y + _MARKER_HEIGHT),
                    QtCore.QPoint(x, marker_top_y),
                ]
            )
            painter.drawPolygon(triangle)


__all__ = ["ColormapPreview"]
