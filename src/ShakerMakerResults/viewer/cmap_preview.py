"""Reusable 0→1 colormap preview strip used by the side-panel sections.

Each instance keeps its current colormap name and rebuilds its pixmap on
``setColormap()`` and on resize.  The 0 / 1 bracket text is painted at the
ends so the user immediately sees the normalised mapping the picker uses.
"""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .colors import colormap_strip_bytes

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


class ColormapPreview(QtWidgets.QLabel):
    """Horizontal strip showing the active colormap from 0 to 1."""

    def __init__(self, name: str = "viridis", parent=None):
        super().__init__(parent)
        self.setObjectName("CmapPreview")
        self.setMinimumHeight(20)
        self.setMaximumHeight(22)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self._cmap = str(name)
        self._tick_color = QtGui.QColor("#666666")
        self.setToolTip(f"Colormap preview — {self._cmap}  (0 → 1)")
        self._rebuild_pixmap()

    def setColormap(self, name: str) -> None:  # noqa: N802 (Qt-style)
        name = str(name).strip() or self._cmap
        if name == self._cmap:
            return
        self._cmap = name
        self.setToolTip(f"Colormap preview — {self._cmap}  (0 → 1)")
        self._rebuild_pixmap()

    def setTickColor(self, color: str) -> None:  # noqa: N802
        self._tick_color = QtGui.QColor(color)
        self._rebuild_pixmap()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._rebuild_pixmap()

    def _rebuild_pixmap(self) -> None:
        width = max(self.width(), 80)
        strip_h = max(self.height() - 2, 12)
        try:
            raw = colormap_strip_bytes(self._cmap, width, strip_h)
        except Exception:
            self.setText(self._cmap)
            return
        image = QtGui.QImage(raw, width, strip_h, QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(image)
        # Draw 0 / 1 ticks on top so the strip reads as a normalised colormap.
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        font = painter.font()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(self._tick_color, 1))
        painter.drawText(QtCore.QRect(2, 0, 24, strip_h), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, "0")
        painter.drawText(
            QtCore.QRect(width - 26, 0, 24, strip_h),
            QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
            "1",
        )
        painter.end()
        self.setPixmap(pixmap)


__all__ = ["ColormapPreview"]
