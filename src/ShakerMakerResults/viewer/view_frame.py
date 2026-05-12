"""ParaView-style title-bar frame that wraps a single :class:`ViewPane`.

Each frame paints a thin border (changes color when the pane is active) and
hosts a compact title-bar at the top with:

    [ Title text ]   [ split-H ]  [ split-V ]  [ maximize ]  [ pop-out ]  [ close ]

The frame is layout-agnostic: it does not own the placement of the pane in
the grid — that is still :class:`MultiViewArea`'s job.  The frame only emits
high-level requests (``split_h``, ``split_v``, ``maximize_toggle``,
``popout_toggle``, ``close``) that the multi-view area translates into
splitter operations.

Drawing the title-bar in Qt (instead of in the VTK render window) keeps the
buttons crisp under any DPI and lets them participate in the global Qt theme.
"""

from __future__ import annotations

from typing import Callable, Optional

from ._imports import require_viewer_dependencies
from .icons import icon as viewer_icon
from .theme import active_palette

_, _, _, QtCore, QtGui, QtWidgets = require_viewer_dependencies()


# ── Inline SVG icons local to the title-bar ──────────────────────────────────
# Kept here (instead of in icons.py) so each frame button can be re-tinted on
# the fly when the theme is toggled without invalidating the icon cache used
# by the rest of the viewer.

_BUTTON_PATHS = {
    "split_h": "M3 4h18v16H3V4zm2 2v12h7V6H5zm9 0v12h5V6h-5z",
    "split_v": "M3 4h18v16H3V4zm2 2v5h14V6H5zm0 7v5h14v-5H5z",
    "maximize": "M5 5h14v14H5V5zm2 2v10h10V7H7z",
    "restore": (
        "M8 4h12v12h-4v2H4V8h2V4zm2 2v2h6v8h2V6H10zM6 10v8h8v-8H6z"
    ),
    "popout": (
        "M14 4h6v6h-2V7.4l-7.3 7.3-1.4-1.4L16.6 6H14V4z"
        "M5 6h6v2H7v9h9v-4h2v6H5V6z"
    ),
    "close": (
        "M6.2 4.8l5.8 5.8 5.8-5.8 1.4 1.4L13.4 12l5.8 5.8-1.4 1.4-5.8-5.8-5.8 5.8"
        "-1.4-1.4 5.8-5.8L4.8 6.2z"
    ),
}


def _svg_icon(name: str, color: str, size: int = 14) -> QtGui.QIcon:
    path = _BUTTON_PATHS.get(name)
    if path is None:
        return QtGui.QIcon()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24"><path fill="{color}" d="{path}"/></svg>'
    ).encode("utf-8")
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    pixmap.loadFromData(svg, "SVG")
    return QtGui.QIcon(pixmap)


class ViewFrame(QtWidgets.QFrame):
    """Title-bar + bordered container hosting one :class:`ViewPane`.

    Parameters
    ----------
    pane:
        The viewport widget to host.  Reparented into the frame's central area.
    title:
        Text shown on the left of the title-bar (e.g. ``"3D NE"``).
    on_split_h, on_split_v:
        Callbacks invoked when the user clicks the split-horizontal / split-
        vertical buttons.  They receive the frame instance so the multi-view
        area can locate the originating pane and rebuild the splitter tree.
    on_maximize_toggle:
        Called when the user clicks the maximize button.  Receives ``(frame,
        wants_maximized)``.  The multi-view area is responsible for swapping
        the visible widget between the grid and the maximised pane.
    on_popout_toggle:
        Called when the user clicks the pop-out button.  Receives ``(frame,
        wants_popout)``.  The multi-view area detaches / re-docks the frame
        from a floating top-level window.
    on_close:
        Optional callback for the close button.  When ``None`` the close
        button is hidden (1×1 layouts cannot close their only frame).
    """

    def __init__(
        self,
        pane: QtWidgets.QWidget,
        title: str,
        *,
        on_split_h: Optional[Callable[["ViewFrame"], None]] = None,
        on_split_v: Optional[Callable[["ViewFrame"], None]] = None,
        on_maximize_toggle: Optional[Callable[["ViewFrame", bool], None]] = None,
        on_popout_toggle: Optional[Callable[["ViewFrame", bool], None]] = None,
        on_close: Optional[Callable[["ViewFrame"], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("ViewFrame")
        self.setProperty("active", False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self._pane = pane
        self._title = str(title)
        self._is_active = False
        self._is_maximized = False
        self._is_popped_out = False

        self._on_split_h = on_split_h
        self._on_split_v = on_split_v
        self._on_maximize_toggle = on_maximize_toggle
        self._on_popout_toggle = on_popout_toggle
        self._on_close = on_close

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(0)

        # ── Title-bar ────────────────────────────────────────────────────────
        self._title_bar = QtWidgets.QWidget()
        self._title_bar.setObjectName("ViewFrameTitleBar")
        self._title_bar.setProperty("active", False)
        self._title_bar.setFixedHeight(22)
        bar_layout = QtWidgets.QHBoxLayout(self._title_bar)
        bar_layout.setContentsMargins(4, 0, 2, 0)
        bar_layout.setSpacing(2)

        self._title_label = QtWidgets.QLabel(self._title)
        self._title_label.setObjectName("ViewFrameTitle")
        bar_layout.addWidget(self._title_label)
        bar_layout.addStretch(1)

        self._split_h_btn = self._make_button(
            "split_h", "Split horizontally  (new pane to the right)",
        )
        self._split_v_btn = self._make_button(
            "split_v", "Split vertically  (new pane below)",
        )
        self._max_btn = self._make_button(
            "maximize", "Maximise this view inside the tab",
            checkable=True,
        )
        self._popout_btn = self._make_button(
            "popout", "Detach view to a floating window",
            checkable=True,
        )
        self._close_btn = self._make_button("close", "Close this view")

        self._split_h_btn.clicked.connect(self._fire_split_h)
        self._split_v_btn.clicked.connect(self._fire_split_v)
        self._max_btn.toggled.connect(self._fire_maximize)
        self._popout_btn.toggled.connect(self._fire_popout)
        self._close_btn.clicked.connect(self._fire_close)

        for btn in (
            self._split_h_btn,
            self._split_v_btn,
            self._max_btn,
            self._popout_btn,
            self._close_btn,
        ):
            bar_layout.addWidget(btn)

        # Close is hidden when no callback is wired (e.g. 1×1 layouts).
        self._close_btn.setVisible(self._on_close is not None)

        outer.addWidget(self._title_bar)

        # ── Central area (host for the pane) ─────────────────────────────────
        self._central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(self._central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._pane)
        outer.addWidget(self._central, 1)

        # Make a click anywhere on the title-bar or pane mark this frame active.
        # The pane already installs its own event filter for clicks inside the
        # 3-D viewport; clicking the title-bar should also activate.
        self._title_bar.mousePressEvent = self._title_press  # type: ignore[assignment]

        self._refresh_button_icons()

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def pane(self) -> QtWidgets.QWidget:
        return self._pane

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def is_maximized(self) -> bool:
        return self._is_maximized

    @property
    def is_popped_out(self) -> bool:
        return self._is_popped_out

    def title(self) -> str:
        return self._title

    def set_title(self, value: str) -> None:
        self._title = str(value)
        self._title_label.setText(self._title)

    def set_active(self, active: bool) -> None:
        active = bool(active)
        if self._is_active == active:
            return
        self._is_active = active
        for w in (self, self._title_bar):
            w.setProperty("active", active)
            style = w.style()
            style.unpolish(w)
            style.polish(w)
            w.update()

    def set_maximized_state(self, value: bool) -> None:
        """Sync the maximize button visual state without re-emitting a signal."""
        value = bool(value)
        if self._is_maximized == value:
            return
        self._is_maximized = value
        block = self._max_btn.blockSignals(True)
        self._max_btn.setChecked(value)
        self._max_btn.blockSignals(block)
        self._max_btn.setIcon(
            _svg_icon("restore" if value else "maximize", active_palette().text)
        )
        self._max_btn.setToolTip(
            "Restore this view to the grid" if value
            else "Maximise this view inside the tab"
        )

    def set_popout_state(self, value: bool) -> None:
        """Sync the popout button visual state without re-emitting a signal."""
        value = bool(value)
        if self._is_popped_out == value:
            return
        self._is_popped_out = value
        block = self._popout_btn.blockSignals(True)
        self._popout_btn.setChecked(value)
        self._popout_btn.blockSignals(block)
        self._popout_btn.setToolTip(
            "Re-dock this view inside the main window" if value
            else "Detach view to a floating window"
        )

    def refresh_theme(self) -> None:
        """Rebuild the title-bar icons against the current palette."""
        self._refresh_button_icons()

    def set_close_button_visible(self, visible: bool) -> None:
        """Show / hide the close button (hidden on the only pane of 1×1)."""
        self._close_btn.setVisible(bool(visible))

    # ── Internals ────────────────────────────────────────────────────────────

    def _title_press(self, _event):
        # Activating the underlying pane keeps the toolbar and active-pane
        # logic in MultiViewArea consistent with what the user clicked.
        scene_fire = getattr(self._pane, "_fire_activated", None)
        if callable(scene_fire):
            scene_fire()

    def _make_button(self, icon_name: str, tooltip: str, *, checkable: bool = False):
        btn = QtWidgets.QToolButton(self)
        btn.setObjectName("ViewFrameButton")
        btn.setIcon(_svg_icon(icon_name, active_palette().text))
        btn.setIconSize(QtCore.QSize(14, 14))
        btn.setToolTip(tooltip)
        btn.setCheckable(checkable)
        btn.setAutoRaise(True)
        btn.setProperty("_icon_name", icon_name)
        return btn

    def _refresh_button_icons(self):
        color = active_palette().text
        for btn in (
            self._split_h_btn,
            self._split_v_btn,
            self._popout_btn,
            self._close_btn,
        ):
            name = btn.property("_icon_name") or ""
            btn.setIcon(_svg_icon(name, color))
        self._max_btn.setIcon(
            _svg_icon("restore" if self._is_maximized else "maximize", color)
        )

    def _fire_split_h(self):
        if callable(self._on_split_h):
            self._on_split_h(self)

    def _fire_split_v(self):
        if callable(self._on_split_v):
            self._on_split_v(self)

    def _fire_maximize(self, checked: bool):
        self._is_maximized = bool(checked)
        self._max_btn.setIcon(
            _svg_icon(
                "restore" if self._is_maximized else "maximize",
                active_palette().text,
            )
        )
        self._max_btn.setToolTip(
            "Restore this view to the grid" if self._is_maximized
            else "Maximise this view inside the tab"
        )
        if callable(self._on_maximize_toggle):
            self._on_maximize_toggle(self, self._is_maximized)

    def _fire_popout(self, checked: bool):
        self._is_popped_out = bool(checked)
        self._popout_btn.setToolTip(
            "Re-dock this view inside the main window" if self._is_popped_out
            else "Detach view to a floating window"
        )
        if callable(self._on_popout_toggle):
            self._on_popout_toggle(self, self._is_popped_out)

    def _fire_close(self):
        if callable(self._on_close):
            self._on_close(self)


__all__ = ["ViewFrame"]
