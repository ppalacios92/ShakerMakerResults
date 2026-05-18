# `ShakerMakerResults.viewer` — architecture notes

The viewer is the **optional** GUI front-end of `ShakerMakerResults`.
Everything below assumes you came in through `model.viewer()` (the
factory hung off `ShakerMakerData`). The whole package is designed so
that the rest of the toolkit works without it — if PyVista / PyQt / VTK
are not installed, only the import of this subpackage fails, never the
core readers.

> **TL;DR** — the viewer is the cherry on top, not the cake. The data
> goes through `ShakerMakerData` and friends; the viewer just reads it
> back in a slick way.

---

## 1. Layered design

```
┌────────────────────────────────────────────────────────────────────┐
│  ViewerSession             session.py    -- orchestration / API    │
│    │                                                               │
│    ├── ViewerDataAdapter   adapter.py    -- HDF5 + LRU + KDTree    │
│    ├── ViewerState         state.py      -- knobs (validated)      │
│    │                                                               │
│    └── ViewerMainWindow    window.py     -- Qt main window         │
│           │                                                        │
│           ├── TabbedMultiViewArea          multi_view.py           │
│           │     │                                                  │
│           │     └── ViewPane               multi_view.py           │
│           │            │                                           │
│           │            ├── ViewerScene     scene.py  (PyVista)     │
│           │            ├── PaneDisplayState  pane_state.py         │
│           │            └── RevitInteractor   interaction.py        │
│           │                                                        │
│           ├── ViewerSidePanel              side_panel.py           │
│           ├── toolbars                     toolbar.py              │
│           ├── docks                        properties_dock.py,     │
│           │                                information_panel.py,   │
│           │                                pipeline_browser.py     │
│           ├── trace panels                 trace_panel.py          │
│           └── RuptureTab (optional)        rupture_pane.py +       │
│                                            rupture_adapter.py      │
└────────────────────────────────────────────────────────────────────┘
```

Three things to remember when reading the source:

1. **No data lives in the viewer.** `ViewerDataAdapter` is the *only*
   piece that touches HDF5 -- everything else above it asks the adapter.
   That's the part we explicitly do not refactor lightly, because all
   the performance work (LRU cache, persistent handle, contiguous
   reads, GF slot cache) lives there.

2. **State is split in two layers.** `ViewerState` is the global per
   session knobs; `PaneDisplayState` is the per-pane overlay on top of
   it. Reads on the overlay fall through to the session state when the
   attribute is not overridden, so old call sites that ignore panes keep
   working.

3. **Updates flow one way.** Anything that mutates state goes through
   `ViewerSession.set_*` / `apply_*`. Those methods finish by calling
   `_notify_window(reason)`; the main window then dispatches the
   `reason` string to every widget that subscribes (`on_session_updated`).
   Widgets never poke each other.

---

## 2. The session is the API

A notebook user only ever sees these:

```python
session = model.viewer(show=True)            # build + show Qt window
session.set_demand("vel")                    # change demand
session.set_component("resultant")
session.select_node(42)                      # picks node 42
session.set_warp_enabled(True)               # turn on real-motion warp
session.set_playing(True)                    # start the animation
session.close()                              # tear down cleanly
```

If you find yourself reaching into `session.state.something = X`,
prefer `session.set_X(...)` -- the setter takes care of validation +
window notification + cache invalidation.

---

## 3. The adapter is the I/O contract

`ViewerDataAdapter` wraps a `ShakerMakerData` instance and gives the
rest of the viewer a small, fast surface:

| Method                              | Returns                                |
| :---------------------------------- | :------------------------------------- |
| `scalar_snapshot(t, demand, comp)`  | `(N,)` scalar field at one time step   |
| `scalar_series(demand, comp)`       | `(N, Nt)` whole series, cached in LRU  |
| `prewarm_component_triplet(demand)` | loads (E, N, Z) in one HDF5 pass       |
| `trace(node_id, demand)`            | `dict` with `t / z / e / n / labels`   |
| `gf_tensor(node_id, subfault_id)`   | dict with `tdata (Nt, 9)`, `t0`, ...   |
| `displacement_snapshot(t)`          | `(N, 3)` per-node displacement         |
| `suggested_warp_scale()`            | auto warp factor for the dataset       |
| `nearest_node_id(point)`            | KDTree query in display coords         |
| `node_info(node_id)`                | metadata dict for the info panel       |

The cache (`_series_cache`) is byte-budgeted. When the budget is too
tight to hold the full triplet, the adapter opens a persistent HDF5
handle (`open_playback_handle`) so per-frame reads avoid the file-open
cost.

`get_window(t0, t1)` and `resample(dt)` on `ShakerMakerData` build a
*new* reader; the adapter takes the new instance as input and rebuilds
its caches accordingly. There's no in-place windowing.

---

## 4. The state is the truth

`ViewerState` is a plain dataclass. Every field has a typed setter and
`__post_init__` re-validates the whole struct, so the same class can be
instantiated from a YAML / QSettings blob without crashing on garbage.

Grouped fields (see the module docstring of `state.py` for the full
list):

* time / playback
* demand / component / selection
* appearance basics, advanced colormap
* color-range overrides
* legend / scalar bar
* axes / grid
* selection labels and warp ghost
* geometry visibility
* displacement warp
* multi-selection
* vector field overlay

`PaneDisplayState` (in `pane_state.py`) is a thin override layer with a
custom `__getattr__` / `__setattr__`. Anything in
`pane_state._GLOBAL_ONLY` is forced to live on the session state -- the
animation clock, playback flag and selection set are shared across
panes by design.

---

## 5. The window is dumb glue

`ViewerMainWindow` mostly:

* builds dock widgets the first time they become visible (lazy);
* runs the playback timer and forwards `_advance_playback_once` to
  `session.step_time(1)`;
* receives `on_session_updated(reason)` from the session and forwards
  the reason to every visible widget;
* persists / restores Qt geometry through `QSettings`.

No business logic lives here. If you find a `model.X = Y` style
mutation inside `window.py`, that's a bug.

---

## 6. Multi-view + tabs

`TabbedMultiViewArea` wraps several `MultiViewArea` instances (one tab
each). Each `MultiViewArea` is a recursive Qt splitter that hosts
`ViewFrame` wrappers around `ViewPane` instances; every `ViewPane` is
a PyVista `QtInteractor` + a `ViewerScene` + a `PaneDisplayState`.

That sounds heavy but the practical consequences are simple:

* Splitting / popping out / maximising a pane never touches the data
  layer -- the adapter is shared.
* Two panes in the same window can render the same data with
  different colormaps, vmin/vmax, warp axes or scale.
* The active pane is whoever was clicked last; the side panel and the
  toolbar's "Apply to" combo route their changes to the active pane,
  to a specific pane, or to all panes depending on the combo state.

GF mode (`MultiViewArea.toggle_gf_mode`) tears the active layout down
and lays out a 3x3 grid, one pane per GF tensor cell.

---

## 7. Side panel sections

`ViewerSidePanel` (in `side_panel.py`) is a stacked widget driven by a
vertical nav menu. Sections (subclasses of `_SectionBase`):

* `NodeSearchSection`   -- node id / nearest-coordinate + station table
* `DisplaySection`      -- demand / component / colormap / range
* `VisualizationSection`-- visibility toggles
* `WarpSection`         -- warp toggle / axes / scale
* `VectorFieldSection`  -- arrow overlay
* `StaticColorSection`  -- elevation / Newmark Sa / Arias overlay
* `GeographicSection`   -- 3x3 display transform editor

Each section follows the same dirty-flag pattern: edits flag the
section dirty, the `Apply` button reads "Apply to" from the toolbar
and routes the change to:

* the global session (`Apply to all`),
* the active pane only (`Apply to active`),
* or every pane in the current tab (`Apply to all in tab`).

Heavy analysis pages (`Responses`, `Green functions`) are lazy and
behind `_LazyPage` / `_ResponsesAnalysisTabs`; they only build their
widgets when the user navigates to them the first time.

---

## 8. Optional rupture tab

`RuptureTab` (`rupture_pane.py`) is a self-contained tab that shows an
FFSP rupture realisation on its own PyVista plotter, with its own time
animation and field selector. It can also project the rupture surface
into any seismic tab as a translucent overlay (`RuptureOverlay`).

`RuptureSource` (`rupture_adapter.py`) is the HDF5 reader feeding the
tab; it lives next to the GUI code rather than in `core/` because it
deals with FFSP-format files, which are an input to ShakerMaker, not
an output.

---

## 9. Files you almost never touch

* `icons.py`         -- the SVG icon strings.
* `colors.py`        -- background presets, default colormaps, RGB blend.
* `theme.py`         -- dark / light QPalette tokens, Qt stylesheet
  generator and `QSettings` persistence for the user's pick.
* `busy_dialog.py`   -- modal "Working..." progress dialog.
* `cmap_preview.py`  -- horizontal colormap preview strip with the
  draggable per-stop arrows; used by the transfer-function dialog.
* `_imports.py`      -- single source of truth for PyVista / Qt / VTK
  optional imports; raises a friendly error if the user has not
  installed the `[viewer]` extras.

---

## 10. How to add a new "knob"

Concrete checklist when, say, you want to add a "show subfault edges"
toggle:

1. Add the field + validator to `ViewerState` (`state.py`).
2. Add a `set_subfault_edges_visible(...)` method on `ViewerSession`
   that mutates state and calls `_notify_window("subfault_edges")`.
3. Add a UI control in the side panel (or a toolbar button) that calls
   the session setter.
4. In `ViewerScene.on_session_updated` (or wherever you handle the
   reason), rebuild / hide the relevant actor.
5. Run `python scripts/smoke_test.py` and visually exercise both
   themes + at least one split layout.

Anything that touches the data path (`adapter.py`) needs a real HDF5
file in front of it -- the smoke test only covers imports.
