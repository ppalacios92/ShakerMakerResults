"""Per-pane visual override layer over :class:`ViewerState`.

The viewer originally drives every pane from a single global
:class:`ViewerSession` ``state``.  That works for "show everything the
same way" but breaks the moment the user wants two panes inside the same
window to render different visual choices — e.g. one pane with the
``viridis`` ramp clamped to a tight band while a neighbouring pane uses
``RdBu_r`` over the full data range.

:class:`PaneDisplayState` is a thin override layer:

* Reads fall through to the session's global state when no override has
  been set for the requested attribute.
* Writes always land in the local ``_overrides`` map; they never touch
  ``session.state``.
* The set of "overridable" attributes is everything in
  :class:`ViewerState` *except* the few that intrinsically belong to the
  application as a whole (the playback time index, the active
  selection set, playback speed / is_playing, station tags, etc.).
  Trying to override those drops through to a no-op so callers can
  safely "Apply to active pane" without worrying about accidentally
  desynchronising the global animation timeline.

Data-loading attributes (``demand``, ``component``, GF subfault index)
are explicitly *not* part of this protocol; they still live on the
session because the underlying HDF5 cache is shared and a per-pane
demand would cost an extra triplet to be resident in memory.  Visual
attributes such as ``colormap`` / ``user_vmin`` / ``warp_scale`` are
free of any I/O and can diverge freely.

The class is intentionally small (no signals, no dataclass): the
``ViewerScene`` reads from it on every rebuild, and changes propagate
through the existing ``on_session_updated`` broadcast, just keyed off
the pane that owns the state.
"""

from __future__ import annotations

from typing import Any, Iterable


#: Attributes that *cannot* be overridden per pane.
#:
#: Only application-wide axes that would desynchronise multiple panes
#: are listed here — the animation clock, playback state, and the
#: shared selection set.  Everything else (including ``demand`` /
#: ``component``) is per-pane, even though that means two panes can
#: hold different cached series at the same time; the adapter cache
#: already supports the extra triplet and the user explicitly asked
#: for "all visual attributes per pane".
_GLOBAL_ONLY: frozenset[str] = frozenset(
    {
        "time_index",
        "is_playing",
        "playback_speed",
        "selected_node",
        "multi_selection",
        "selection_visibility",
        "show_internal",
        "show_external",
        "show_qa",
    }
)


class PaneDisplayState:
    """Per-pane overrides backed by ``session.state``.

    Use it exactly as you would the global :class:`ViewerState`:

        >>> pane.state.colormap          # falls through if no override
        'viridis'
        >>> pane.state.colormap = 'hot_r'   # writes to override map
        >>> pane.state.colormap
        'hot_r'
        >>> pane.state.clear_override('colormap')   # back to global
        >>> pane.state.colormap
        'viridis'

    Reading any attribute marked in :data:`_GLOBAL_ONLY` always returns
    the global value (writes go through too, but they are stored without
    effect — :meth:`has_override` will not report them).  This keeps
    ``time_index`` / ``selected_node`` / ``is_playing`` consistent
    everywhere even when a careless caller tries to override them.
    """

    __slots__ = ("_session", "_overrides")

    def __init__(self, session):
        # ``object.__setattr__`` because we use a custom ``__setattr__``.
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_overrides", {})

    # ── Attribute protocol ──────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for attributes Python could not resolve
        # the usual way, so this catches every "visual" read.
        if name.startswith("_"):
            raise AttributeError(name)
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        session = object.__getattribute__(self, "_session")
        return getattr(session.state, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_session", "_overrides"):
            object.__setattr__(self, name, value)
            return
        if name in _GLOBAL_ONLY:
            # Silently drop — the global keeps the value.  Callers can
            # still use session.state.X = ... if they really want.
            return
        object.__getattribute__(self, "_overrides")[name] = value

    # ── Inspection ──────────────────────────────────────────────────────────

    def has_override(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_overrides")

    def overridden_names(self) -> tuple[str, ...]:
        return tuple(sorted(object.__getattribute__(self, "_overrides").keys()))

    def snapshot_overrides(self) -> dict[str, Any]:
        """Return a shallow copy of the active overrides (for debugging /
        Scene Browser tooltips)."""
        return dict(object.__getattribute__(self, "_overrides"))

    # ── Mutation ────────────────────────────────────────────────────────────

    def set_override(self, name: str, value: Any) -> None:
        """Programmatic override setter.

        Equivalent to ``pane.state.<name> = value`` but lets callers pass
        the name as a string when iterating over a settings dict.
        """
        setattr(self, name, value)

    def set_many(self, overrides: dict[str, Any]) -> None:
        """Apply a batch of overrides in one call."""
        for name, value in overrides.items():
            setattr(self, name, value)

    def clear_override(self, name: str | None = None) -> None:
        """Drop a single override, or all overrides when ``name is None``."""
        target = object.__getattribute__(self, "_overrides")
        if name is None:
            target.clear()
        else:
            target.pop(name, None)

    def clear_overrides(self, names: Iterable[str]) -> None:
        target = object.__getattribute__(self, "_overrides")
        for name in names:
            target.pop(name, None)


__all__ = ["PaneDisplayState"]
