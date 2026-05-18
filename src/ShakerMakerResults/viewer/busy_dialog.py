"""
busy_dialog.py
==============
Lightweight modal "Working..." dialog used by long-running viewer
operations (Newmark / Arias surface scans, pre-warm of the playback
triplet, GF subfault swaps, etc.).

The dialog supports two modes:

* **Deterministic** (default: ``max=0``) -- shows an indeterminate
  marquee progress bar.
* **Numeric** (``max=N``) -- shows a regular 0..N progress bar that
  callers update through :meth:`set_step`.

The close button is intentionally removed so the user cannot dismiss
the dialog mid-operation and leave caches half-built.
"""

from __future__ import annotations

from ._imports import require_viewer_dependencies

_, _, _, QtCore, _, QtWidgets = require_viewer_dependencies()


class BusyDialog(QtWidgets.QDialog):
    """Modal dialog with unified "Ladruno working..." look and messaging."""

    def __init__(self, message: str, parent=None, *, total_steps: int | None = None):
        super().__init__(parent)
        self.setWindowTitle("Ladruno")
        # Remove close/max/min buttons so users cannot dismiss mid-operation.
        self.setWindowFlags(
            QtCore.Qt.Dialog
            | QtCore.Qt.CustomizeWindowHint
            | QtCore.Qt.WindowTitleHint
        )
        self.setModal(True)
        self.setFixedWidth(320)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(10)

        header = QtWidgets.QLabel("Ladruno working...")
        header.setStyleSheet("font-weight: bold; font-size: 13px; color: #1565C0;")
        lay.addWidget(header)

        self._message_label = QtWidgets.QLabel(message)
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet("color: #404040; font-size: 11px;")
        lay.addWidget(self._message_label)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(
            "QProgressBar { border: none; background: #e0e0e0; border-radius: 3px; }"
            "QProgressBar::chunk { background: #1565C0; border-radius: 3px; }"
        )
        lay.addWidget(self._bar)

        if total_steps is None:
            self._bar.setRange(0, 0)  # indeterminate mode
        else:
            self._bar.setRange(0, max(int(total_steps), 0))
            self._bar.setValue(0)

    def set_message(self, message: str) -> None:
        self._message_label.setText(str(message))

    def set_step(self, step: int) -> None:
        if self._bar.maximum() != 0:
            self._bar.setValue(int(step))
