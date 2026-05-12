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


# Geometry constants — the *default* compact size used by the form-row
# previews on the side-panel sections.  Larger consumers (e.g. the
# TransferFunctionDialog) can bump ``setMinimumHeight`` on the instance
# and the strip automatically uses the extra vertical space; the marker
# row stays fixed at the bottom of the widget.
_DEFAULT_STRIP_HEIGHT = 16
_MARKER_HEIGHT = 6
_PADDING_X = 2
_DEFAULT_TOTAL_HEIGHT = _DEFAULT_STRIP_HEIGHT + _MARKER_HEIGHT + 4


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
        # Default to the compact size used by the side-panel form rows.
        # Consumers that want a taller preview (e.g. the color-mapping
        # dialog) just call ``setMinimumHeight(...)`` on the instance and
        # the strip auto-grows; the marker row stays anchored at the
        # bottom.  Maximum height is intentionally unbounded so a
        # ``QVBoxLayout.addWidget(..., 1)`` can stretch the widget too.
        self.setMinimumHeight(_DEFAULT_TOTAL_HEIGHT)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Preferred,
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
