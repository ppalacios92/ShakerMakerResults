"""
icons.py
========
Inline SVG icon library for the Qt viewer.

We embed every icon as a path string here rather than shipping a
``resources/`` directory so:

* the package is one ``pip install`` away with no extra data files,
* themed re-tinting is trivial (we just inject the colour into the SVG
  body before rendering),
* the icons render at any DPI without a separate ``@2x`` asset.

The single public entry point is :func:`icon`, which is LRU-cached on
``(name, color, size)`` so toolbars / docks that rebuild themselves do
not hit the SVG renderer twice for the same glyph.
"""

from __future__ import annotations

from functools import lru_cache

from ._imports import require_viewer_dependencies

_, _, _, QtCore, QtGui, _ = require_viewer_dependencies()


_PATHS = {
    "play": "M8 5v14l11-7z",
    "pause": "M7 5h4v14H7zM13 5h4v14h-4z",
    "step_back": "M18 6v12l-8-6zM7 6h2v12H7z",
    "step_forward": "M6 6v12l8-6zM15 6h2v12h-2z",
    "skip_back": "M19 5v14l-9-7zM8 5h2v14H8zM5 5h2v14H5z",
    "skip_forward": "M5 5v14l9-7zM14 5h2v14h-2zM17 5h2v14h-2z",
    "gear": (
        "M19.4 13.5c.1-.5.1-1 .1-1.5s0-1-.1-1.5l2-1.5-2-3.4-2.4 1"
        "c-.8-.6-1.6-1-2.6-1.3L14 2.8h-4l-.4 2.5c-.9.3-1.8.7-2.6 1.3l-2.4-1-2 3.4"
        " 2 1.5c-.1.5-.1 1-.1 1.5s0 1 .1 1.5l-2 1.5 2 3.4 2.4-1"
        "c.8.6 1.6 1 2.6 1.3l.4 2.5h4l.4-2.5c.9-.3 1.8-.7 2.6-1.3l2.4 1 2-3.4z"
        "M12 15.5A3.5 3.5 0 1 1 12 8a3.5 3.5 0 0 1 0 7.5z"
    ),
    "view_iso_ne": (
        "M12 3 20 7.5v9L12 21 4 16.5v-9L12 3zm0 2.4L7 8.2l5 2.8 5-2.8-5-2.8z"
        "M6 10v5.2l5 2.8v-5.2L6 10zm7 2.8V18l5-2.8V10l-5 2.8z"
        "M15 4h5v5h-2V7.4l-3.2 3.2-1.4-1.4L16.6 6H15V4z"
    ),
    "view_iso_nw": (
        "M12 3 20 7.5v9L12 21 4 16.5v-9L12 3zm0 2.4L7 8.2l5 2.8 5-2.8-5-2.8z"
        "M6 10v5.2l5 2.8v-5.2L6 10zm7 2.8V18l5-2.8V10l-5 2.8z"
        "M9 4H4v5h2V7.4l3.2 3.2 1.4-1.4L7.4 6H9V4z"
    ),
    "view_iso_sw": (
        "M12 3 20 7.5v9L12 21 4 16.5v-9L12 3zm0 2.4L7 8.2l5 2.8 5-2.8-5-2.8z"
        "M6 10v5.2l5 2.8v-5.2L6 10zm7 2.8V18l5-2.8V10l-5 2.8z"
        "M9 20H4v-5h2v1.6l3.2-3.2 1.4 1.4L7.4 18H9v2z"
    ),
    "view_iso_se": (
        "M12 3 20 7.5v9L12 21 4 16.5v-9L12 3zm0 2.4L7 8.2l5 2.8 5-2.8-5-2.8z"
        "M6 10v5.2l5 2.8v-5.2L6 10zm7 2.8V18l5-2.8V10l-5 2.8z"
        "M15 20h5v-5h-2v1.6l-3.2-3.2-1.4 1.4 3.2 3.2H15v2z"
    ),
    "view_top": "M4 5h16v14H4V5zm2 2v10h12V7H6zm3 3h6v4H9v-4z",
    "view_bottom": "M4 5h16v14H4V5zm2 2v10h12V7H6zm2 8h8v2H8v-2z",
    "view_front": "M5 4h14v16H5V4zm2 2v12h10V6H7zm2 3h6v6H9V9z",
    "view_back": "M5 4h14v16H5V4zm2 2v12h10V6H7zm2 2h6v2H9V8zm0 4h6v2H9v-2z",
    "view_left": "M5 5h14v14H5V5zm2 2v10h3V7H7zm5 0v10h5V7h-5z",
    "view_right": "M5 5h14v14H5V5zm2 2v10h5V7H7zm7 0v10h3V7h-3z",
    "fit": (
        "M5 10H3V3h7v2H6.4l3.3 3.3-1.4 1.4L5 6.4V10z"
        "M19 10h2V3h-7v2h3.6l-3.3 3.3 1.4 1.4L19 6.4V10z"
        "M5 14H3v7h7v-2H6.4l3.3-3.3-1.4-1.4L5 17.6V14z"
        "M19 14h2v7h-7v-2h3.6l-3.3-3.3 1.4-1.4 3.3 3.3V14z"
    ),
    "ortho": "M4 4h16v16H4V4zm2 2v5h5V6H6zm7 0v5h5V6h-5zM6 13v5h5v-5H6zm7 0v5h5v-5h-5z",
    "rotate_left_90": (
        "M8 7h7a5 5 0 1 1-4.5 7.2l1.8-.9A3 3 0 1 0 15 9H8v3L4 8l4-4v3z"
        "M5 18h4v2H5v-2zm1-5h2v4H6v-4z"
    ),
    "rotate_right_90": (
        "M16 7H9a5 5 0 1 0 4.5 7.2l-1.8-.9A3 3 0 1 1 9 9h7v3l4-4-4-4v3z"
        "M15 18h4v2h-4v-2zm1-5h2v4h-2v-4z"
    ),
    "capture_screen": (
        "M4 5h16c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V7"
        "c0-1.1.9-2 2-2zm0 2v10h16V7H4zm4 2h2l.8-1h2.4l.8 1h2"
        "c.55 0 1 .45 1 1v5c0 .55-.45 1-1 1H8c-.55 0-1-.45-1-1v-5"
        "c0-.55.45-1 1-1zm4 1.5A2.5 2.5 0 1 0 12 15.5a2.5 2.5 0 0 0 0-5z"
    ),
    "record_screen": (
        "M4 5h16c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V7"
        "c0-1.1.9-2 2-2zm0 2v10h16V7H4zm8 2.5a3.5 3.5 0 1 1 0 7"
        " 3.5 3.5 0 0 1 0-7z"
    ),
    "stop_screen": (
        "M4 5h16c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V7"
        "c0-1.1.9-2 2-2zm0 2v10h16V7H4zm6 3h4v4h-4v-4z"
    ),
    "colormap_editor": (
        "M4 18h16v2H4v-2zm1-4h3v2H5v-2zm4-4h3v6H9v-6zm4-4h3v10h-3V6zm4-2h3v12h-3V4z"
    ),
    "legend_edit": (
        "M5 4h10a2 2 0 0 1 2 2v12H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm1 3v2h8V7H6zm0 4v2h8v-2H6z"
        "M19.7 11.3 22 13.6 17.6 18H15v-2.6l4.7-4.1z"
    ),
    "axes_grid": (
        "M4 4h16v16H4V4zm2 2v4h4V6H6zm6 0v4h6V6h-6zM6 12v6h4v-6H6zm6 0v6h6v-6h-6z"
    ),
    "selection_label": (
        "M4 4h9v6H8l-4 4V4zm11 2h5v12h-5V6zm2 2v2h1V8h-1zm0 4v2h1v-2h-1z"
    ),
    "ghost_warp": (
        "M5 6c0-1.1.9-2 2-2h5c1.1 0 2 .9 2 2v2h3c1.1 0 2 .9 2 2v8H8c-1.1 0-2-.9-2-2v-2H5V6zm3 0v8h6V6H8zm6 4v6h3v-6h-3z"
    ),
    "chart_settings": (
        "M4 18h16v2H4v-2zm2-3h3v2H6v-2zm5-5h3v7h-3v-7zm5-4h3v11h-3V6z"
    ),
    # ── Side-panel nav icons ───────────────────────────────────────────────────
    "nav_node": (
        "M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"
        "M12 11.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"
    ),
    "nav_display": (
        "M20 3H4c-1.1 0-2 .9-2 2v11c0 1.1.9 2 2 2h3l-1 1v1h12v-1l-1-1h3c1.1 0 2-.9 2-2V5"
        "c0-1.1-.9-2-2-2zM20 15H4V5h16v10z"
    ),
    "nav_visibility": (
        "M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5"
        "C21.27 7.61 17 4.5 12 4.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"
        "M12 10c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"
    ),
    "nav_warp": "M2 13c2-5 4-5 6 0s4 5 6 0 4-5 6 0M5 8l1 3M19 8l1 3M5 17l1-3M19 17l1-3",
    "nav_responses": (
        '<path d="M4 13h3l2-6 3 11 3-8 2 3h3" '
        'fill="none" stroke="{color}" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "nav_gf": (
        '<path d="M5 5v14h14" fill="none" stroke="{color}" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M8 15l3-5 3 3 4-7" fill="none" stroke="{color}" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    # ── Selection toolbar icons ────────────────────────────────────────────────
    "sel_cursor": (
        "M4 4l5 15 3-6 6-3z"
        "M14.5 14.5l4 4-1.4 1.4-4-4 1.4-1.4z"
    ),
    "sel_all": "M3 3h7v7H3V3zm0 11h7v7H3v-7zm11-11h7v7h-7V3zm0 11h7v7h-7v-7z",
    "sel_show": "M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z",
    "sel_hide": (
        "M2.81 3.22L1.39 4.63 8 11.24V11c0 2.76 2.24 5 5 5 .34 0 .67-.04 1-.1l.92.92"
        "C14.03 17.26 13.05 17.5 12 17.5c-5 0-9.27-3.11-11-7.5.69-1.76 1.79-3.3 3.14-4.54z"
        "M12 6.5c2.76 0 5 2.24 5 5 0 .51-.08 1-.22 1.47l3.06 3.06A11.6 11.6 0 0 0 23 10"
        "c-1.73-4.39-6-7.5-11-7.5-1.51 0-2.95.31-4.27.83l2.3 2.3c.6-.2 1.28-.13 1.97-.13z"
    ),
    # ── Geographic panel icon ──────────────────────────────────────────────────
    "nav_geographic": (
        "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"
        "m-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1"
        "c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3"
        "c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41"
        "c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"
    ),
}


def _svg(icon_name: str, color: str, size: int) -> bytes:
    path = _PATHS.get(icon_name)
    if path is None:
        raise KeyError(f"Unknown viewer icon '{icon_name}'.")
    if path.lstrip().startswith("<"):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24">{path.format(color=color)}</svg>'
        ).encode("utf-8")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24"><path fill="{color}" d="{path}"/></svg>'
    ).encode("utf-8")


@lru_cache(maxsize=256)
def icon(icon_name: str, color: str = "#172033", size: int = 18) -> QtGui.QIcon:
    """Build a ``QIcon`` from the local SVG registry."""

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    pixmap.loadFromData(_svg(icon_name, color, size), "SVG")
    return QtGui.QIcon(pixmap)


__all__ = ["icon"]
