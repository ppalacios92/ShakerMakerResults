"""Visual theme tokens and Qt stylesheet for the interactive viewer.

Two palettes are bundled — :data:`LIGHT_PALETTE` and :data:`DARK_PALETTE` —
and the active one is chosen at runtime by :func:`active_palette`.  The user
selection is persisted in :class:`QtCore.QSettings` so a toggle from the View
menu survives across sessions.

The stylesheet for the dock widget title-bar, the view-frame title-bar and the
new tabbed multi-view widget is rebuilt every time the palette changes so the
whole UI keeps a coherent feel between light and dark.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._imports import require_viewer_dependencies

_, _, _, QtCore, _, _ = require_viewer_dependencies()


@dataclass(frozen=True)
class ViewerPalette:
    """Color tokens consumed by every Qt widget in the viewer."""

    name: str = "light"
    accent: str = "#2a6bc2"
    accent_dark: str = "#1e5299"
    navy: str = "#1e3558"
    surface: str = "#fbfaf7"
    surface_2: str = "#f2f5f7"
    surface_3: str = "#e6ebef"
    text: str = "#172033"
    text_2: str = "#4b5d74"
    text_muted: str = "#7d8a9a"
    border: str = "#d7dee6"
    border_strong: str = "#aeb9c7"
    danger: str = "#b24a3f"
    plot_white: str = "#ffffff"
    # Used by the 3-D plotter background and the in-scene branding text.
    render_background: str = "#ffffff"
    render_text: str = "#555555"
    # Active-view highlight (border + title-bar) — matches the accent color.
    active_border: str = "#2a6bc2"


LIGHT_PALETTE = ViewerPalette()

DARK_PALETTE = ViewerPalette(
    name="dark",
    accent="#5aa7ff",
    accent_dark="#3f8de0",
    navy="#cfd8e3",
    surface="#262a33",
    surface_2="#1d2027",
    surface_3="#323845",
    text="#e7ebf2",
    text_2="#bfc7d1",
    text_muted="#8e97a4",
    border="#3a404c",
    border_strong="#4c5363",
    danger="#e07060",
    plot_white="#1e1e1e",
    render_background="#1e1e1e",
    render_text="#bfc7d1",
    active_border="#5aa7ff",
)


_SETTINGS_KEY = "viewer/theme"
_AVAILABLE = {"light": LIGHT_PALETTE, "dark": DARK_PALETTE}


def _settings() -> "QtCore.QSettings":
    return QtCore.QSettings("ShakerMakerResults", "Viewer")


def saved_theme_name() -> str:
    """Return the persisted theme name, defaulting to ``"light"``."""
    value = _settings().value(_SETTINGS_KEY, "light")
    name = str(value).lower()
    return name if name in _AVAILABLE else "light"


def save_theme_name(name: str) -> None:
    """Persist *name* (``"light"`` or ``"dark"``) to QSettings."""
    if name in _AVAILABLE:
        _settings().setValue(_SETTINGS_KEY, name)


def active_palette() -> ViewerPalette:
    """Return the palette selected by the user (persisted in QSettings)."""
    return _AVAILABLE[saved_theme_name()]


def palette_by_name(name: str) -> ViewerPalette:
    return _AVAILABLE.get(str(name).lower(), LIGHT_PALETTE)


def build_stylesheet(palette: ViewerPalette = LIGHT_PALETTE) -> str:
    """Return the global Qt stylesheet for *palette*."""

    return f"""
    QMainWindow,
    QDialog {{
        background: {palette.surface_2};
        color: {palette.text};
    }}
    QWidget#ViewerCentral {{
        background: {palette.surface_2};
        color: {palette.text};
    }}
    QWidget#ViewerHeader {{
        background: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 6px;
    }}
    QWidget#ViewerHeader QLabel {{
        color: {palette.navy};
    }}
    QWidget#TimeControls {{
        background: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 6px;
    }}
    QLabel {{
        color: {palette.text};
    }}
    QLabel#TimeLabel {{
        color: {palette.accent_dark};
        font-family: Consolas, "SF Mono", "Cascadia Mono", monospace;
        font-weight: 600;
        padding: 0 8px;
    }}
    QLabel#SpeedLabel {{
        color: {palette.text_muted};
        font-size: 11px;
    }}
    QLabel#StatusChip {{
        padding: 2px 9px;
        border-radius: 9px;
        background: {palette.surface};
        border: 1px solid {palette.border};
        color: {palette.text_2};
        font-family: Consolas, "SF Mono", "Cascadia Mono", monospace;
        font-size: 11px;
    }}
    QLabel#CmapPreview {{
        border: 1px solid {palette.border};
        border-radius: 3px;
        background: {palette.surface};
    }}
    QToolButton#IconButton,
    QToolButton#TextIconButton,
    QPushButton {{
        min-height: 24px;
        border: 1px solid {palette.border_strong};
        border-radius: 5px;
        background: {palette.surface};
        color: {palette.text};
        padding: 2px 8px;
    }}
    QToolButton#IconButton {{
        min-width: 26px;
        max-width: 26px;
        padding: 2px;
    }}
    QToolButton#PlayButton {{
        min-height: 28px;
        border: 1px solid {palette.accent};
        border-radius: 5px;
        background: {palette.surface_3};
        color: {palette.navy};
        font-weight: 600;
        padding: 2px 12px;
    }}
    QToolButton#IconButton:hover,
    QToolButton#TextIconButton:hover,
    QPushButton:hover {{
        background: {palette.surface_3};
        border-color: {palette.accent};
    }}
    QToolButton#PlayButton:hover {{
        background: {palette.surface_3};
        border-color: {palette.accent_dark};
    }}
    QToolButton#IconButton:checked,
    QToolButton#TextIconButton:checked {{
        background: {palette.surface_3};
        border-color: {palette.accent};
        color: {palette.navy};
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        border-radius: 3px;
        background: {palette.surface_3};
    }}
    QSlider::sub-page:horizontal {{
        border-radius: 3px;
        background: {palette.accent};
    }}
    QSlider::handle:horizontal {{
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
        background: {palette.surface};
        border: 2px solid {palette.accent};
    }}
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QLineEdit {{
        min-height: 24px;
        border: 1px solid {palette.border};
        border-radius: 5px;
        background: {palette.surface};
        color: {palette.text};
        padding: 1px 6px;
        selection-background-color: {palette.accent};
        selection-color: {palette.surface};
    }}
    QComboBox QAbstractItemView {{
        background: {palette.surface};
        color: {palette.text};
        selection-background-color: {palette.accent};
        selection-color: {palette.surface};
        border: 1px solid {palette.border};
    }}
    QDoubleSpinBox:focus,
    QSpinBox:focus,
    QComboBox:focus,
    QLineEdit:focus {{
        border-color: {palette.accent};
    }}
    QMenu {{
        background: {palette.surface};
        border: 1px solid {palette.border};
        color: {palette.text};
    }}
    QMenu::item:selected {{
        background: {palette.surface_3};
        color: {palette.navy};
    }}
    QMenuBar {{
        background: {palette.surface};
        color: {palette.text};
        border-bottom: 1px solid {palette.border};
    }}
    QMenuBar::item:selected {{
        background: {palette.surface_3};
    }}
    QTabWidget::pane {{
        border: 1px solid {palette.border};
        border-radius: 5px;
        background: {palette.surface};
        top: -1px;
    }}
    QTabBar::tab {{
        min-height: 24px;
        padding: 4px 13px;
        border: 1px solid {palette.border};
        border-bottom-color: {palette.border};
        background: {palette.surface_2};
        color: {palette.text_2};
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background: {palette.surface};
        color: {palette.navy};
        border-color: {palette.accent};
        border-bottom-color: {palette.surface};
        font-weight: 600;
    }}
    QTabBar::tab:hover:!selected {{
        background: {palette.surface_3};
    }}
    QTabBar::close-button {{
        subcontrol-position: right;
    }}
    QGroupBox {{
        border: 1px solid {palette.border};
        border-radius: 6px;
        margin-top: 6px;
        padding: 8px 6px 6px 6px;
        background: {palette.surface};
        color: {palette.navy};
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {palette.navy};
        background: {palette.surface};
    }}
    QWidget#CameraViewRail {{
        background: {palette.surface};
        border-right: 1px solid {palette.border};
    }}
    QLabel#ViewRailSection {{
        color: {palette.text_muted};
        font-size: 10px;
        font-weight: 600;
        padding: 4px 0 2px 0;
    }}
    QToolButton#ViewRailButton {{
        min-width: 44px;
        max-width: 44px;
        min-height: 42px;
        border: 1px solid transparent;
        border-radius: 5px;
        background: transparent;
        color: {palette.navy};
        padding: 2px;
        font-size: 10px;
    }}
    QToolButton#ViewRailButton:hover {{
        background: {palette.surface_3};
        border-color: {palette.border};
    }}
    QToolButton#ViewRailButton:checked {{
        background: {palette.surface_3};
        border-color: {palette.accent};
        color: {palette.navy};
        font-weight: 600;
    }}
    QStatusBar {{
        background: {palette.surface_2};
        color: {palette.text};
        border-top: 1px solid {palette.border};
    }}
    /* ── View-frame title-bar ─────────────────────────────────────────────── */
    QFrame#ViewFrame {{
        background: {palette.surface_2};
        border: 1px solid {palette.border};
        border-radius: 4px;
    }}
    QFrame#ViewFrame[active="true"] {{
        border: 2px solid {palette.active_border};
    }}
    QWidget#ViewFrameTitleBar {{
        background: {palette.surface};
        border-bottom: 1px solid {palette.border};
    }}
    QWidget#ViewFrameTitleBar[active="true"] {{
        background: {palette.surface_3};
    }}
    QLabel#ViewFrameTitle {{
        color: {palette.navy};
        font-size: 11px;
        font-weight: 600;
        padding: 0 6px;
    }}
    QToolButton#ViewFrameButton {{
        min-width: 20px; max-width: 20px;
        min-height: 20px; max-height: 20px;
        border: none;
        border-radius: 3px;
        background: transparent;
        padding: 1px;
    }}
    QToolButton#ViewFrameButton:hover {{
        background: {palette.surface_3};
    }}
    /* ── Dock widgets (sidebar panels) ────────────────────────────────────── */
    QDockWidget {{
        color: {palette.navy};
        font-size: 11px;
        font-weight: 600;
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        text-align: left;
        padding: 4px 8px;
        background: {palette.surface};
        border: 1px solid {palette.border};
        border-bottom: 1px solid {palette.accent};
    }}
    QDockWidget > QWidget {{
        background: {palette.surface};
        color: {palette.text};
    }}
    /* ── Multi-view layout tab strip ──────────────────────────────────────── */
    QTabBar#MultiViewTabs::tab {{
        min-height: 22px;
        padding: 3px 12px;
        font-size: 11px;
    }}
    QTabBar#MultiViewTabs::tab:selected {{
        background: {palette.surface};
        color: {palette.navy};
        border-color: {palette.accent};
    }}
    /* ── Themed toolbars ──────────────────────────────────────────────────── */
    QToolBar {{
        background: {palette.surface};
        border: none;
        spacing: 2px;
        padding: 2px;
    }}
    QToolBar::separator {{
        width: 1px;
        background: {palette.border};
        margin: 4px 4px;
    }}
    """


__all__ = [
    "LIGHT_PALETTE",
    "DARK_PALETTE",
    "ViewerPalette",
    "build_stylesheet",
    "active_palette",
    "palette_by_name",
    "saved_theme_name",
    "save_theme_name",
]
