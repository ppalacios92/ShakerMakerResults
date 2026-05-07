"""Adapter between :class:`ShakerMakerData` and the interactive viewer."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

import numpy as np

from ..core.gf_service import get_gf_tensor
from .colors import scalar_limits

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - fallback kept for optional envs
    cKDTree = None


REGULAR_DEMANDS = ("accel", "vel", "disp")
GF_DEMAND = "gf"
VALID_DEMANDS = REGULAR_DEMANDS + (GF_DEMAND,)

REGULAR_COMPONENTS = ("z", "e", "n", "resultant")
GF_COMPONENTS = tuple(f"g{i}{j}" for i in range(1, 4) for j in range(1, 4))
VALID_COMPONENTS = REGULAR_COMPONENTS + GF_COMPONENTS
TRACE_COMPONENTS = ("z", "e", "n")

DEFAULT_DISPLAY_TRANSFORM = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
    ],
    dtype=float,
)

DISPLAY_DEMAND_LABELS = {
    "accel": "accel",
    "vel": "vel",
    "disp": "disp",
    GF_DEMAND: "Green Functions",
}
DISPLAY_COMPONENT_LABELS = {
    "z": "z",
    "e": "e",
    "n": "n",
    "resultant": "resultant",
    **{comp: comp.upper() for comp in GF_COMPONENTS},
}


@dataclass(frozen=True)
class DatasetSummary:
    """Human-readable summary of the current model for the viewer."""

    name: str
    dataset_type: str
    node_count: int
    display_node_count: int
    time_steps: int
    has_qa: bool
    has_gf: bool
    has_map: bool


class ViewerDataAdapter:
    """Expose the subset of ``ShakerMakerData`` needed by the viewer.

    The 3-D scene uses snapshot-level lazy loading: each animation frame reads
    only the active demand/component/time column. Full time-series are loaded
    only for selected-node panels such as traces, spectra, and Arias intensity.
    """

    def __init__(
        self,
        model,
        *,
        cache_time_series: bool = True,
        max_cache_bytes: int | None = None,
        max_cache_entries: int = 8,  # kept for API compat; byte-budget LRU is used instead
    ):
        self.model = model
        self.cache_time_series = bool(cache_time_series)

        # Auto-detect cache budget: 50 % of available RAM, at least 1 GB.
        # No hard upper cap — large-RAM workstations benefit from caching
        # multiple full demand triplets simultaneously.
        if max_cache_bytes is None:
            try:
                import psutil as _psutil
                _available = int(_psutil.virtual_memory().available)
            except Exception:
                _available = 4 * 1024 * 1024 * 1024
            max_cache_bytes = max(1 * 1024 * 1024 * 1024, int(_available * 0.5))
        self.max_cache_bytes = int(max_cache_bytes)
        # Running total of bytes currently held in _series_cache.
        # Maintained by _cache_insert / _cache_touch / clear_runtime_caches.
        self._total_cache_bytes: int = 0
        self._display_transform = DEFAULT_DISPLAY_TRANSFORM.copy()

        self._points = np.empty((0, 3), dtype=float)
        self._qa_point = None
        self._display_points = np.empty((0, 3), dtype=float)
        self._display_node_ids = list(range(len(model.xyz)))
        if getattr(model, "xyz_qa", None) is not None:
            self._display_node_ids.append("QA")
        self._node_to_index = {
            node_id: idx for idx, node_id in enumerate(self._display_node_ids)
        }
        self._kdtree = None

        self._display_internal = np.zeros(len(self._display_node_ids), dtype=bool)
        self._display_is_qa = np.zeros(len(self._display_node_ids), dtype=bool)

        self._series_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._series_cache_hits = 0
        self._spectrum_cache: dict[tuple[int | str, float, float, float], dict[str, np.ndarray]] = {}
        self._arias_cache: dict[int | str, dict[str, object]] = {}
        self._gf_slot_cache: dict[int, np.ndarray] = {}
        self._gf_t0_cache: dict[int, np.ndarray] = {}
        self._gf_limit_cache: dict[tuple[int, str], tuple[float, float]] = {}
        # Pre-warmed GF field: (subfault_id, component_idx) → (n_nodes, nt_gf) float32
        # One HDF5 read fills this; subsequent playback frames are pure NumPy.
        self._gf_series_cache: dict[tuple[int, int], np.ndarray] = {}
        # Persistent HDF5 handle kept open during playback for large regular-demand
        # models (series too big to cache).  Eliminates ~1-5 ms file-open overhead
        # per frame when _try_direct_component_snapshot is the active path.
        self._playback_h5_handle = None
        # Cached visibility mask — recomputed only when show_* flags change.
        # Avoids a 497k-element boolean array allocation on every animation frame.
        self._visibility_mask_cache: np.ndarray | None = None
        self._visibility_mask_key: tuple[bool, bool, bool] | None = None

        self._visible_node_ids = list(self._display_node_ids)
        self._visible_to_display_index = np.arange(len(self._display_node_ids), dtype=int)
        self._rebuild_display_geometry()

    @property
    def points(self) -> np.ndarray:
        """All display coordinates in metres, rotated for plotting."""
        return self._display_points

    @property
    def display_transform(self) -> np.ndarray:
        """Global model-to-display transform used by every viewer pane."""
        return self._display_transform.copy()

    @property
    def time(self) -> np.ndarray:
        """Return the simulation time vector."""
        return self.model.time

    @property
    def has_qa(self) -> bool:
        return self._qa_point is not None

    @property
    def has_gf(self) -> bool:
        return bool(getattr(self.model, "_has_gf", False))

    @property
    def has_map(self) -> bool:
        return bool(getattr(self.model, "_has_map", False))

    @property
    def available_demands(self) -> tuple[str, ...]:
        if self.has_gf and self.has_map:
            return VALID_DEMANDS
        return REGULAR_DEMANDS

    @property
    def available_components(self) -> tuple[str, ...]:
        return REGULAR_COMPONENTS

    def available_components_for_demand(self, demand: str) -> tuple[str, ...]:
        demand = self._validate_demand(demand)
        if demand == GF_DEMAND:
            return GF_COMPONENTS
        return REGULAR_COMPONENTS

    def display_demand_options(self) -> list[tuple[str, str]]:
        return [
            (demand, DISPLAY_DEMAND_LABELS.get(demand, demand))
            for demand in self.available_demands
        ]

    def display_component_options(self, demand: str) -> list[tuple[str, str]]:
        return [
            (component, DISPLAY_COMPONENT_LABELS.get(component, component))
            for component in self.available_components_for_demand(demand)
        ]

    def elevation_snapshot(self) -> np.ndarray:
        """Return one static elevation value per displayed node in viewer-space metres."""
        return np.asarray(self._display_points[:, 2], dtype=float)

    def elevation_limits(self) -> tuple[float, float]:
        """Return stable min/max limits for viewer-space node elevations."""
        values = self.elevation_snapshot()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return -1.0, 1.0
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            return vmin, vmin + 1.0
        return vmin, vmax

    @property
    def trace_components(self) -> tuple[str, ...]:
        return TRACE_COMPONENTS

    @property
    def node_ids(self) -> list[int | str]:
        return list(self._display_node_ids)

    @property
    def visible_node_ids(self) -> list[int | str]:
        return list(self._visible_node_ids)

    @property
    def dataset_type(self) -> str:
        if getattr(self.model, "is_drm", False):
            if np.any(getattr(self.model, "internal", np.array([], dtype=bool))):
                return "DRM"
            return "SurfaceGrid"
        return "Station"

    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            name=str(getattr(self.model, "name", getattr(self.model, "filename", "Model"))),
            dataset_type=self.dataset_type,
            node_count=len(self._points),
            display_node_count=len(self._display_points),
            time_steps=len(self.model.time),
            has_qa=self.has_qa,
            has_gf=self.has_gf,
            has_map=self.has_map,
        )

    def open_playback_handle(self) -> None:
        """Open a persistent HDF5 handle for the DRM file during playback.

        When the scalar series is too large to cache (large models), each frame
        would otherwise open, read, and close the HDF5 file — that is
        ~1-5 ms of filesystem overhead per frame.  Keeping the handle open
        amortises that cost to zero for the duration of playback.

        Safe to call multiple times; opens only once.  Must be paired with
        :meth:`close_playback_handle`.
        """
        if self._playback_h5_handle is not None:
            return
        if not self._supports_direct_snapshot():
            return
        try:
            import h5py
            self._playback_h5_handle = h5py.File(self.model.filename, "r")
        except Exception:
            self._playback_h5_handle = None

    def close_playback_handle(self) -> None:
        """Close the persistent playback HDF5 handle (if open)."""
        if self._playback_h5_handle is not None:
            try:
                self._playback_h5_handle.close()
            except Exception:
                pass
            self._playback_h5_handle = None

    def clear_runtime_caches(self) -> None:
        """Release viewer-only cached arrays and derived analysis results."""
        self.close_playback_handle()
        self._series_cache.clear()
        self._total_cache_bytes = 0
        self._series_cache_hits = 0
        self._spectrum_cache.clear()
        self._arias_cache.clear()
        self._gf_limit_cache.clear()
        self._gf_series_cache.clear()

    @property
    def cache_info(self) -> dict[str, int]:
        """Return scalar-series cache statistics (regular + GF combined)."""
        return {
            "entries": len(self._series_cache) + len(self._gf_series_cache),
            "bytes": int(self._total_cache_bytes),
            "budget": int(self.max_cache_bytes),
            "hits": int(self._series_cache_hits),
        }

    def clamp_time_index(self, time_index: int) -> int:
        if len(self.model.time) == 0:
            return 0
        return max(0, min(int(time_index), len(self.model.time) - 1))

    def node_id_from_index(self, point_index: int) -> int | str:
        return self._display_node_ids[int(point_index)]

    def node_id_from_visible_index(self, point_index: int) -> int | str:
        return self._visible_node_ids[int(point_index)]

    def point_index_for_node(self, node_id: int | str) -> int:
        return self._node_to_index[node_id]

    def point_for_node(self, node_id: int | str) -> np.ndarray:
        return self._display_points[self.point_index_for_node(node_id)]

    def display_points_from_model_xyz_m(self, xyz_m: np.ndarray) -> np.ndarray:
        """Transform arbitrary model coordinates expressed in metres into viewer space."""
        xyz_m = np.asarray(xyz_m, dtype=float)
        return self._apply_display_transform_m(xyz_m)

    def set_display_transform(self, matrix) -> np.ndarray:
        """Update the global display transform and rebuild viewer geometry."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("Display transform must be a 3x3 matrix.")
        # TODO: Keep this as the single viewer geometry rebuild path for future transform debugging.
        self._display_transform = matrix.copy()
        self._rebuild_display_geometry()
        return self.display_transform

    def nearest_node_id(self, point: np.ndarray) -> int | str:
        """Return nearest node in viewer coordinates [m]."""
        point = np.asarray(point, dtype=float)
        if self._kdtree is not None:
            _, idx = self._kdtree.query(point)
            return self._display_node_ids[int(idx)]

        distances = np.linalg.norm(self._display_points - point, axis=1)
        return self._display_node_ids[int(np.argmin(distances))]

    def nearest_node_from_model_xyz_m(self, xyz_m: np.ndarray) -> tuple[int | str, float]:
        """Resolve nearest node from original model coordinates in metres.

        This intentionally delegates to ShakerMakerData._collect_node_ids when
        available so the viewer uses the same node-resolution semantics as the
        plotting API.
        """
        xyz_km = np.asarray(xyz_m, dtype=float) / 1000.0
        if hasattr(self.model, "_collect_node_ids"):
            nids = self.model._collect_node_ids(target_pos=xyz_km, print_info=False)
            node_id = nids[0]
        else:
            xyz_all = getattr(self.model, "xyz_all", None)
            if xyz_all is None:
                xyz_all = self.model.xyz
            dist = np.linalg.norm(np.asarray(xyz_all) - xyz_km, axis=1)
            idx = int(np.argmin(dist))
            node_id = "QA" if self.has_qa and idx == len(self.model.xyz) else idx

        if node_id in ("QA", "qa"):
            pos_km = self.model.xyz_qa[0]
        else:
            pos_km = self.model.xyz[int(node_id)]
        distance_m = float(np.linalg.norm(np.asarray(pos_km, dtype=float) - xyz_km) * 1000.0)
        return node_id, distance_m

    def visibility_mask(
        self,
        *,
        show_internal: bool = True,
        show_external: bool = True,
        show_qa: bool = True,
    ) -> np.ndarray:
        """Return a boolean mask of visible nodes.

        The result is cached keyed on (show_internal, show_external, show_qa).
        During animation the flags never change, so every frame after the first
        is a zero-cost dict lookup instead of a 497k-element NumPy computation.
        The cache is invalidated by ``_rebuild_display_geometry``.
        """
        key = (bool(show_internal), bool(show_external), bool(show_qa))
        if self._visibility_mask_key == key and self._visibility_mask_cache is not None:
            return self._visibility_mask_cache

        internal = self._display_internal
        is_qa = self._display_is_qa
        external = (~internal) & (~is_qa)
        mask = np.zeros(len(self._display_points), dtype=bool)
        if show_internal:
            mask |= internal
        if show_external:
            mask |= external
        if show_qa:
            mask |= is_qa

        self._visibility_mask_key = key
        self._visibility_mask_cache = mask
        return mask

    def visible_points(
        self,
        *,
        show_internal: bool = True,
        show_external: bool = True,
        show_qa: bool = True,
    ) -> np.ndarray:
        mask = self.visibility_mask(
            show_internal=show_internal,
            show_external=show_external,
            show_qa=show_qa,
        )
        self._visible_to_display_index = np.flatnonzero(mask)
        self._visible_node_ids = [self._display_node_ids[i] for i in self._visible_to_display_index]
        return self._display_points[mask]

    def visible_scalars(
        self,
        values: np.ndarray,
        *,
        show_internal: bool = True,
        show_external: bool = True,
        show_qa: bool = True,
    ) -> np.ndarray:
        """Return scalars filtered to visible nodes.

        Short-circuits the mask application when all nodes are visible
        (the common case for DRM models) — avoids a full 497k-element
        fancy-index copy and returns a view of the original array instead.
        """
        mask = self.visibility_mask(
            show_internal=show_internal,
            show_external=show_external,
            show_qa=show_qa,
        )
        values = np.asarray(values)
        if mask.all():
            return values
        return values[mask]

    def scalar_snapshot(
        self,
        time_index: int,
        demand: str = "accel",
        component: str = "resultant",
        *,
        subfault_id: int = 0,
    ) -> np.ndarray:
        """Return one scalar value per displayed point.

        Fast path: if the full series is already cached (e.g. pre-warmed by
        playback), returns the requested column with zero I/O overhead.
        Falls back to a direct single-column HDF5 read, then to the
        ``get_surface_snapshot`` API.
        """
        demand = self._validate_demand(demand)
        component = self._validate_component(component)
        time_index = self.clamp_time_index(time_index)

        if demand == GF_DEMAND:
            return self._gf_scalar_snapshot(time_index, component, int(subfault_id))

        # ── Fast path: series already in memory ──────────────────────────────
        key = (demand, component)
        if key in self._series_cache:
            self._series_cache_hits += 1
            cached = self._series_cache[key]
            col = min(time_index, cached.shape[1] - 1)
            return np.asarray(cached[:, col], dtype=float)

        # ── Resultant fast path: derive from cached E/N/Z components ─────────
        # set_playing pre-warms E, N, Z individually (cheaper HDF5 pattern).
        # Once they are in memory, resultant is free NumPy arithmetic.
        if component == "resultant":
            e_key, n_key, z_key = (demand, "e"), (demand, "n"), (demand, "z")
            if all(k in self._series_cache for k in (e_key, n_key, z_key)):
                self._series_cache_hits += 1
                col = min(time_index, self._series_cache[e_key].shape[1] - 1)
                e = self._series_cache[e_key][:, col]
                n = self._series_cache[n_key][:, col]
                z = self._series_cache[z_key][:, col]
                return np.sqrt(e ** 2 + n ** 2 + z ** 2).astype(float)

        direct = self._try_direct_component_snapshot(demand, component, time_index)
        if direct is not None:
            return np.asarray(direct, dtype=float)

        if component == "resultant":
            ex = self.model.get_surface_snapshot(time_index, "e", demand)
            ny = self.model.get_surface_snapshot(time_index, "n", demand)
            zz = self.model.get_surface_snapshot(time_index, "z", demand)
            values = np.sqrt(ex ** 2 + ny ** 2 + zz ** 2)
        else:
            values = self.model.get_surface_snapshot(time_index, component, demand)

        if self.has_qa:
            qa_values = self.trace("QA", demand)
            if component == "resultant":
                qa_scalar = float(np.linalg.norm(qa_values[:, time_index]))
            else:
                qa_scalar = float(qa_values[self._component_to_trace_index(component), time_index])
            values = np.concatenate([np.asarray(values), np.array([qa_scalar])])
        return np.asarray(values, dtype=float)

    def scalar_series(self, demand: str = "accel", component: str = "resultant") -> np.ndarray:
        """Return the full time history matrix for one displayed scalar field."""
        demand = self._validate_demand(demand)
        component = self._validate_component(component)
        if demand == GF_DEMAND:
            raise NotImplementedError(
                "GF field playback uses per-frame snapshots only; full GF series "
                "caching is not enabled in the viewer."
            )
        key = (demand, component)

        if key in self._series_cache:
            self._series_cache_hits += 1
            self._cache_touch(key)
            return self._series_cache[key]

        # ── Resultant fast path: derive from already-cached E/N/Z ─────────────
        # When _prewarm_on_show (or a prior warm cycle) has already loaded the
        # three component series, resultant is free NumPy arithmetic — no HDF5
        # I/O.  This eliminates the biggest single-threaded delay on first Play.
        if component == "resultant":
            e_k, n_k, z_k = (demand, "e"), (demand, "n"), (demand, "z")
            if all(k in self._series_cache for k in (e_k, n_k, z_k)):
                self._series_cache_hits += 1
                for _k in (e_k, n_k, z_k):
                    self._cache_touch(_k)
                values = self._resultant_from_cached_components(demand)
                self._cache_insert(key, values)
                return values

        values = self._build_scalar_series(demand, component)
        self._cache_insert(key, values)
        return values

    def prewarm_component_triplet(
        self, demand: str, *, progress_cb=None
    ) -> tuple[tuple[str, str], ...]:
        """Load E/N/Z full series for *demand* into cache with one HDF5 pass.

        A full triplet (E + N + Z) must fit in the byte budget simultaneously
        for the animation fast-path to work (resultant derived from cached trio).
        If the triplet does not fit the budget the method returns an empty tuple
        and lets per-frame single-column HDF5 reads handle animation instead.

        Parameters
        ----------
        progress_cb:
            Optional ``(bytes_done: int, bytes_total: int) -> None`` callable
            invoked after each node-chunk is read.  Use it to update a progress
            dialog.  Called from the main thread — do not block.
        """
        demand = self._validate_demand(demand)
        if demand == GF_DEMAND:
            return tuple()
        keys = tuple((demand, comp) for comp in ("e", "n", "z"))
        missing = [key for key in keys if key not in self._series_cache]
        if not missing:
            self._series_cache_hits += len(keys)
            for key in keys:
                self._cache_touch(key)
            return keys
        if not self.cache_time_series:
            return tuple()

        # Guard: the FULL triplet must fit in the budget at the same time.
        # If even one component doesn't fit, the animation resultant fast-path
        # (which requires all three) cannot work — skip caching entirely and
        # let the persistent HDF5 handle deliver per-frame snapshots.
        one_series = self._estimated_series_bytes_for_demand(demand)
        if one_series * 3 > self.max_cache_bytes:
            return tuple()

        if self._supports_direct_series():
            try:
                import h5py
                with h5py.File(self.model.filename, "r") as handle:
                    data_handle = handle[self._data_path_for_demand(demand)]
                    column_selector, n_cols = self._column_selector(data_handle.shape[1])
                    series_by_component = self._read_component_triplet_series(
                        data_handle, column_selector, n_cols, progress_cb=progress_cb
                    )
                    for comp, values in series_by_component.items():
                        qa = self._read_qa_component_series(
                            handle, demand, comp, column_selector
                        )
                        if qa is not None:
                            values = np.vstack([values, qa[None, :]])
                        self._cache_insert((demand, comp), values)
                return keys
            except Exception:
                pass

        loaded: list[tuple[str, str]] = []
        for comp in ("e", "n", "z"):
            try:
                self.scalar_series(demand, comp)
                loaded.append((demand, comp))
            except Exception:
                pass
        return tuple(loaded)

    def trace(self, node_id: int | str, demand: str = "accel") -> np.ndarray:
        """Return the selected node trace as ``[z, e, n]``.

        Fast path: when all three components for *demand* are already in the
        series cache (e.g. pre-warmed by ``_prewarm_on_show`` or ``set_playing``),
        extracts the node row with a pure NumPy slice — zero HDF5 I/O, instant
        response even on 50+ GB models.

        Falls back to ``get_node_data`` (HDF5 read) when the cache is cold or
        the demand was never pre-warmed (e.g. requesting accel when only vel was
        loaded via ``field='vel'``).
        """
        demand = self._validate_demand(demand)
        if node_id in ("QA", "qa"):
            if not self.has_qa:
                raise KeyError("QA station is not available for this model.")
            data = self.model.get_qa_data(demand)
            return np.asarray(data, dtype=float)

        # ── Fast path: all three components already in RAM ────────────────────
        z_key, e_key, n_key = (demand, "z"), (demand, "e"), (demand, "n")
        if all(k in self._series_cache for k in (z_key, e_key, n_key)):
            idx = int(node_id)
            z = self._series_cache[z_key][idx, :]
            e = self._series_cache[e_key][idx, :]
            n = self._series_cache[n_key][idx, :]
            # Row order matches get_node_data convention: [z, e, n]
            return np.stack([z, e, n], axis=0).astype(float)

        # ── Cold path: read from HDF5 ─────────────────────────────────────────
        data = self.model.get_node_data(int(node_id), demand)
        return np.asarray(data, dtype=float)

    def gf_trace(self, node_id: int | str, subfault_id: int, component: str = "z") -> np.ndarray:
        """Return one Green function trace when GF data is available."""
        if not self.has_gf or not self.has_map:
            raise RuntimeError("GF/map data is not available in this viewer session.")
        component = component.lower()
        if component not in ("z", "e", "n", "tdata"):
            raise KeyError("GF component must be one of 'z', 'e', 'n', 'tdata'.")
        return np.asarray(self.model.get_gf(node_id, subfault_id, component))

    def gf_subfault_count(self) -> int:
        """Return the number of subfaults (sources) in the GF dataset.

        ``_nsources`` is written by :func:`load_map` from the map HDF5 file
        and is the most reliable source.  Falls back to other attribute names
        if the model was built by an older version of the code.
        """
        if not self.has_gf or not self.has_map:
            return 0
        for attr in ("_nsources", "nsources", "_nsources_db", "n_sources", "num_sources"):
            val = getattr(self.model, attr, None)
            if val is not None:
                try:
                    return max(1, int(val))
                except (TypeError, ValueError):
                    pass
        return 1

    def gf_tensor(self, node_id, subfault_id: int = 0) -> dict:
        """Return the full 9-component GF tensor for *(node_id, subfault_id)*.

        The GF HDF5 file stores ``tdata`` with shape ``(n_slots, nt, 9)``.
        Column ordering matches ``plot_node_tensor_gf`` conventions::

            col 0  G_11   col 1  G_12   col 2  G_13
            col 3  G_21   col 4  G_22   col 5  G_23
            col 6  G_31   col 7  G_32   col 8  G_33

        Diagonal (i == j): columns 0, 4, 8  →  G_11, G_22, G_33.
        Time axis: ``np.arange(nt) * model._dt_orig + t0`` where ``t0`` is
        read from ``f['t0'][slot]`` (per-slot offset stored in the HDF5).

        Returns
        -------
        dict
            ``"time"``  – 1-D float array, length *nt*.
            ``"rows"``  – list of ``(label, is_diagonal, data_1d)`` tuples,
                          length 9.
        """
        gf_data = get_gf_tensor(self.model, node_id, int(subfault_id))
        time_arr = np.asarray(gf_data["time"], dtype=float)
        tdata = np.asarray(gf_data["tdata"], dtype=float)

        # ── Build 9 rows, G_11 … G_33 in row-major order ─────────────────────
        # Diagonal cells sit at flat indices where (i == j): 0, 4, 8.
        _DIAG = {0, 4, 8}
        labels = [f"G_{i + 1}{j + 1}" for i in range(3) for j in range(3)]
        rows = [
            (labels[k], k in _DIAG, tdata[:, k])
            for k in range(min(9, tdata.shape[1]))
        ]

        return {"time": time_arr, "rows": rows}

    def spectrum(
        self,
        node_id: int | str,
        *,
        zeta: float = 0.05,
        max_period: float = 5.01,
        intervals: float = 0.02,
    ) -> dict[str, np.ndarray]:
        """Return cached Newmark spectra for the selected node."""
        cache_key = (node_id, float(zeta), float(max_period), float(intervals))
        if cache_key in self._spectrum_cache:
            return self._spectrum_cache[cache_key]

        try:
            from ..newmark import NewmarkSpectrumAnalyzer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError(
                "Spectrum plotting requires the analysis dependencies "
                "used by ShakerMakerResults.newmark."
            ) from exc

        trace = self.trace(node_id, "accel")
        dt = float(self.model.time[1] - self.model.time[0]) if len(self.model.time) > 1 else 0.0
        if dt <= 0.0:
            raise ValueError("Cannot compute a spectrum without at least two time samples.")

        labels = ("z", "e", "n")
        result = {}
        for label, series in zip(labels, trace):
            spectrum = NewmarkSpectrumAnalyzer.compute(
                series / 9.81,
                dt,
                zeta=zeta,
                max_period=max_period,
                intervals=intervals,
            )
            result[f"PSa_{label}"] = spectrum["PSa"]
            result[f"Sa_{label}"] = spectrum["Sa"]
            result[f"Sv_{label}"] = spectrum["Sv"]
            result[f"Sd_{label}"] = spectrum["Sd"]
            result["T"] = spectrum["T"]

        self._spectrum_cache[cache_key] = result
        return result

    def arias(self, node_id: int | str) -> dict[str, object]:
        """Return cached Arias-intensity curves for the selected node.

        Uses EarthquakeSignal.core.arias_intensity.AriasIntensityAnalyzer, the
        same analyzer used by ShakerMakerData.plot_node_arias().
        """
        if node_id in self._arias_cache:
            return self._arias_cache[node_id]

        try:
            from ..analysis.arias_intensity import AriasIntensityAnalyzer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError(
                "Arias intensity requires ShakerMakerResults.analysis.arias_intensity."
            ) from exc

        acc = self.trace(node_id, "accel")
        dt = float(self.model.time[1] - self.model.time[0]) if len(self.model.time) > 1 else 0.0
        if dt <= 0.0:
            raise ValueError("Cannot compute Arias intensity without at least two time samples.")

        result: dict[str, object] = {"time": None, "components": {}}
        for label, series in zip(("z", "e", "n"), acc):
            ia_pct, t_start, t_end, ia_total, extra = AriasIntensityAnalyzer.compute(series / 9.81, dt)
            time = np.arange(len(ia_pct), dtype=float) * dt
            result["time"] = time
            result["components"][label] = {
                "IA_pct": np.asarray(ia_pct, dtype=float),
                "t_start": float(t_start),
                "t_end": float(t_end),
                "ia_total": float(ia_total),
                "extra": extra,
            }
        self._arias_cache[node_id] = result
        return result

    def displacement_snapshot(self, time_index: int) -> np.ndarray:
        """Return per-node displacement as an ``(N_display, 3)`` array [E, N, Z] in metres.

        Column mapping: 0 → East/X, 1 → North/Y, 2 → vertical/Z.
        Uses ``scalar_snapshot`` internally so it benefits from the series cache
        when pre-warmed by ``set_playing`` or ``set_warp_enabled``.
        Returns zeros on any error so the scene degrades gracefully when the
        model has no displacement data.
        """
        n = len(self._display_points)
        try:
            time_index = self.clamp_time_index(time_index)
            disp_e = self.scalar_snapshot(time_index, "disp", "e")
            disp_n = self.scalar_snapshot(time_index, "disp", "n")
            disp_z = self.scalar_snapshot(time_index, "disp", "z")
            disp_model = np.column_stack([disp_e, disp_n, disp_z])
            return self._apply_display_transform_m(disp_model)
        except Exception:
            return np.zeros((n, 3), dtype=float)

    def suggested_warp_scale(self) -> float:
        """Estimate a display-scale factor so peak displacements fill ~5 % of the domain.

        Strategy (cheapest first):
        1. Read ``model._vmax["disp"]`` — O(1), no I/O.
        2. Fall back to a single ``scalar_snapshot(0, "disp", "resultant")`` call.
        Returns a rounded power-of-10 value ≥ 1.
        """
        import math

        pts = self._display_points
        if len(pts) == 0:
            return 1.0
        domain = float(np.ptp(pts, axis=0).max())
        if domain <= 0.0:
            return 1.0

        # Try cached _vmax from the model (written by ShakerMakerData._compute_vmax).
        max_d = 0.0
        vmax_by_type = getattr(self.model, "_vmax", None)
        if isinstance(vmax_by_type, dict) and "disp" in vmax_by_type:
            for v in vmax_by_type["disp"].values():
                try:
                    max_d = max(max_d, abs(float(v)))
                except (TypeError, ValueError):
                    pass

        # Fall back to a single snapshot — still cheap (single HDF5 column or cache hit).
        if max_d <= 0.0:
            try:
                snap = self.scalar_snapshot(0, "disp", "resultant")
                max_d = float(np.max(np.abs(snap))) if len(snap) > 0 else 0.0
            except Exception:
                pass

        if max_d <= 0.0:
            return 1.0

        raw = 0.05 * domain / max_d
        # Round to nearest "nice" power-of-10 step.
        magnitude = 10 ** math.floor(math.log10(max(raw, 1e-9)))
        nice = max(1.0, round(raw / magnitude) * magnitude)
        return float(nice)

    def default_scalar_limits(
        self,
        demand: str = "accel",
        component: str = "resultant",
        *,
        subfault_id: int = 0,
    ) -> tuple[float, float]:
        """Return cheap default color limits without scanning the full HDF5 file."""
        demand = self._validate_demand(demand)
        component = self._validate_component(component)
        if demand == GF_DEMAND:
            return self._gf_default_scalar_limits(component, int(subfault_id))
        try:
            vmax_by_type = getattr(self.model, "_vmax", None)
            if vmax_by_type is not None and demand in vmax_by_type and component in vmax_by_type[demand]:
                vmax = float(vmax_by_type[demand][component])
                if component == "resultant":
                    return 0.0, max(vmax, 1.0 if vmax <= 0.0 else vmax)
                return -max(abs(vmax), 1.0 if vmax == 0.0 else abs(vmax)), max(abs(vmax), 1.0 if vmax == 0.0 else abs(vmax))
        except Exception:
            pass

        snapshot = self.scalar_snapshot(len(self.model.time) // 2, demand, component)
        return scalar_limits(snapshot, component)

    def node_info(self, node_id: int | str) -> dict[str, object]:
        """Return metadata for the selected node."""
        if node_id in ("QA", "qa"):
            coords = self.model.xyz_qa[0] if self.has_qa else None
            display = self._qa_point if self.has_qa else None
            node_type = "QA"
            internal = False
        else:
            idx = int(node_id)
            coords = self.model.xyz[idx]
            display = self._points[idx]
            internal = bool(self.model.internal[idx]) if hasattr(self.model, "internal") else False
            node_type = "internal" if internal else "external"

        info = {
            "node_id": node_id,
            "type": node_type,
            "internal": internal,
            "xyz_km": None if coords is None else np.asarray(coords, dtype=float),
            "xyz_model_m": None if coords is None else np.asarray(coords, dtype=float) * 1000.0,
            "xyz_m": None if display is None else np.asarray(display, dtype=float),
            "has_gf": self.has_gf and self.has_map,
            "gf_slot_s0": None,
        }
        if info["has_gf"]:
            try:
                info["gf_slot_s0"] = int(self.model._get_slot(node_id, 0))
            except Exception:
                info["gf_slot_s0"] = None
        return info

    def _apply_display_transform_m(self, xyz_m: np.ndarray) -> np.ndarray:
        xyz_m = np.asarray(xyz_m, dtype=float)
        if xyz_m.ndim == 1:
            return xyz_m @ self._display_transform
        return xyz_m @ self._display_transform

    def _rebuild_display_geometry(self) -> None:
        model_xyz_m = np.asarray(self.model.xyz, dtype=float) * 1000.0
        self._points = self._apply_display_transform_m(model_xyz_m)
        qa_xyz = getattr(self.model, "xyz_qa", None)
        self._qa_point = (
            self._apply_display_transform_m(np.asarray(qa_xyz, dtype=float) * 1000.0)[0]
            if qa_xyz is not None
            else None
        )
        self._display_points = (
            np.vstack([self._points, self._qa_point[None, :]])
            if self._qa_point is not None
            else self._points.copy()
        )
        self._node_to_index = {
            node_id: idx for idx, node_id in enumerate(self._display_node_ids)
        }
        internal = np.asarray(
            getattr(self.model, "internal", np.zeros(len(self._points), dtype=bool)),
            dtype=bool,
        )
        if len(internal) != len(self._points):
            internal = np.zeros(len(self._points), dtype=bool)
        self._display_internal = (
            np.concatenate([internal, np.array([False])])
            if self._qa_point is not None
            else internal.copy()
        )
        self._display_is_qa = np.zeros(len(self._display_points), dtype=bool)
        if self._qa_point is not None:
            self._display_is_qa[-1] = True
        self._kdtree = (
            cKDTree(self._display_points) if cKDTree is not None else None
        )
        self._visible_node_ids = list(self._display_node_ids)
        self._visible_to_display_index = np.arange(len(self._display_node_ids), dtype=int)
        # Invalidate visibility mask cache — geometry or internal flags changed.
        self._visibility_mask_cache = None
        self._visibility_mask_key = None

    def _validate_demand(self, demand: str) -> str:
        demand = demand.lower()
        if demand not in VALID_DEMANDS:
            raise KeyError(
                f"Unknown demand '{demand}'. Use one of {', '.join(VALID_DEMANDS)}."
            )
        return demand

    def _validate_component(self, component: str) -> str:
        component = component.lower()
        if component not in VALID_COMPONENTS:
            raise KeyError(
                "Unknown component "
                f"'{component}'. Use one of {', '.join(VALID_COMPONENTS)}."
            )
        return component

    @staticmethod
    def _component_to_trace_index(component: str) -> int:
        return {"z": 0, "e": 1, "n": 2}[component]

    def _try_direct_component_snapshot(self, demand: str, component: str, time_index: int) -> np.ndarray | None:
        """Read one scalar-field column from the HDF5 file.

        Uses the persistent playback handle (opened by
        :meth:`open_playback_handle`) when available — zero file-open overhead
        during playback.  Falls back to a fresh ``h5py.File`` context otherwise.
        """
        if not self._supports_direct_snapshot():
            return None

        # ── Fast path: persistent handle already open (during playback) ───────
        if self._playback_h5_handle is not None:
            try:
                return self._read_snapshot_from_handle(
                    self._playback_h5_handle, demand, component, time_index
                )
            except Exception:
                # Handle may have been closed externally — fall through.
                self._playback_h5_handle = None

        # ── Normal path: open fresh handle per call ───────────────────────────
        try:
            import h5py
        except ImportError:  # pragma: no cover
            return None
        try:
            with h5py.File(self.model.filename, "r") as handle:
                return self._read_snapshot_from_handle(handle, demand, component, time_index)
        except Exception:
            return None

    def _read_snapshot_from_handle(
        self, handle, demand: str, component: str, time_index: int
    ) -> np.ndarray:
        """Extract one scalar-field column from an already-open HDF5 handle."""
        data_handle = handle[self._data_path_for_demand(demand)]
        source_col  = self._source_column_index(time_index, data_handle.shape[1])

        if component == "resultant":
            # One contiguous HDF5 read → NumPy strided slice (3× faster than
            # three separate strided reads against HDF5 chunks).
            all_data = np.asarray(data_handle[:, source_col], dtype=np.float32)
            e = all_data[0::3]
            n = all_data[1::3]
            z = all_data[2::3]
            values    = np.sqrt(e ** 2 + n ** 2 + z ** 2).astype(np.float32, copy=False)
            qa_scalar = self._read_qa_resultant_snapshot(handle, demand, source_col)
        else:
            row       = {"e": 0, "n": 1, "z": 2}[component]
            values    = np.asarray(data_handle[row::3, source_col], dtype=np.float32)
            qa_scalar = self._read_qa_component_snapshot(handle, demand, component, source_col)

        if qa_scalar is not None:
            values = np.concatenate([values, np.array([qa_scalar], dtype=np.float32)])
        return values

    def _supports_direct_snapshot(self) -> bool:
        return (
            hasattr(self.model, "filename")
            and hasattr(self.model, "_data_grp")
            and not hasattr(self.model, "_resample_cache")
        )

    def _source_column_index(self, time_index: int, total_cols: int) -> int:
        window_mask = getattr(self.model, "_window_mask", None)
        if window_mask is None:
            return max(0, min(int(time_index), total_cols - 1))
        col_idx = np.flatnonzero(np.asarray(window_mask, dtype=bool))
        if len(col_idx) == 0:
            return 0
        return int(col_idx[max(0, min(int(time_index), len(col_idx) - 1))])

    def _read_qa_component_snapshot(self, handle, demand: str, component: str, column_index: int) -> float | None:
        if not self.has_qa:
            return None
        path = self._qa_path_for_demand(demand)
        row = {"e": 0, "n": 1, "z": 2}[component]
        return float(handle[path][row, column_index])

    def _read_qa_resultant_snapshot(self, handle, demand: str, column_index: int) -> float | None:
        if not self.has_qa:
            return None
        path = self._qa_path_for_demand(demand)
        qa = np.asarray(handle[path][:, column_index], dtype=np.float32)
        return float(np.linalg.norm(qa[[2, 0, 1]]))

    def _build_scalar_series(self, demand: str, component: str) -> np.ndarray:
        direct = self._try_direct_component_series(demand, component)
        if direct is not None:
            return direct

        n_times = len(self.model.time)
        series = []
        qa_values = self.trace("QA", demand) if self.has_qa else None
        for time_index in range(n_times):
            if component == "resultant":
                ex = self.model.get_surface_snapshot(time_index, "e", demand)
                ny = self.model.get_surface_snapshot(time_index, "n", demand)
                zz = self.model.get_surface_snapshot(time_index, "z", demand)
                values = np.sqrt(ex ** 2 + ny ** 2 + zz ** 2)
            else:
                values = self.model.get_surface_snapshot(time_index, component, demand)
            if qa_values is not None:
                if component == "resultant":
                    qa_scalar = float(np.linalg.norm(qa_values[:, time_index]))
                else:
                    qa_scalar = float(qa_values[self._component_to_trace_index(component), time_index])
                values = np.concatenate([values, np.array([qa_scalar])])
            series.append(np.asarray(values, dtype=np.float32))
        return np.column_stack(series)

    def _try_direct_component_series(self, demand: str, component: str) -> np.ndarray | None:
        if not self._supports_direct_series():
            return None
        if not self.cache_time_series:
            return None

        try:
            import h5py
        except ImportError:  # pragma: no cover - package dependency in real runtime
            return None

        # Gate the cache on the size of the final cached array, not on the
        # temporary operands needed to compute it. Resultant series are built
        # chunk-by-chunk into a preallocated output matrix, so they should
        # still use the direct fast path whenever the final matrix fits.
        series_bytes = self._estimated_series_bytes()
        if series_bytes > self.max_cache_bytes:
            return None

        with h5py.File(self.model.filename, "r") as handle:
            data_handle = handle[self._data_path_for_demand(demand)]
            column_selector, n_cols = self._column_selector(data_handle.shape[1])
            if component == "resultant":
                values = self._read_resultant_series(data_handle, column_selector, n_cols)
                qa_scalar = self._read_qa_resultant_series(handle, demand, column_selector)
            else:
                values = self._read_component_series(data_handle, component, column_selector)
                qa_scalar = self._read_qa_component_series(handle, demand, component, column_selector)

        if qa_scalar is not None:
            values = np.vstack([values, qa_scalar[None, :]])
        return np.asarray(values, dtype=np.float32)

    def _supports_direct_series(self) -> bool:
        return (
            hasattr(self.model, "filename")
            and hasattr(self.model, "_data_grp")
            and self.cache_time_series
            and not hasattr(self.model, "_resample_cache")
        )

    def _estimated_series_bytes(self) -> int:
        rows = len(self._points) + (1 if self.has_qa else 0)
        cols = len(self.model.time)
        return rows * cols * np.dtype(np.float32).itemsize

    def _estimated_series_bytes_for_demand(self, demand: str) -> int:
        demand = self._validate_demand(demand)
        rows = len(self._points) + (1 if self.has_qa else 0)
        if demand == GF_DEMAND:
            cols = len(getattr(self.model, "gf_time", getattr(self.model, "time", [])))
        else:
            cols = len(self.model.time)
        return rows * cols * np.dtype(np.float32).itemsize

    def _cache_insert(self, key: tuple[str, str], values: np.ndarray) -> bool:
        """Insert *values* into the series cache using a global byte-budget LRU.

        The total in-memory cost of all cached series is tracked in
        ``_total_cache_bytes``.  LRU eviction (oldest entry first) runs until
        the new entry fits within ``max_cache_bytes``.

        Returns ``True`` when stored, ``False`` when the entry alone exceeds
        the entire budget (will never fit regardless of evictions).
        """
        if not self.cache_time_series:
            return False
        incoming = int(np.asarray(values, dtype=np.float32).nbytes)
        if incoming > self.max_cache_bytes:
            return False   # Single series larger than the whole budget

        # Re-insertion: remove old version to refresh LRU position.
        if key in self._series_cache:
            old = self._series_cache.pop(key)
            self._total_cache_bytes -= int(old.nbytes)

        # Evict globally least-recently-used entries until there is room.
        while self._total_cache_bytes + incoming > self.max_cache_bytes:
            if not self._series_cache:
                return False
            _, evicted = self._series_cache.popitem(last=False)
            self._total_cache_bytes -= int(evicted.nbytes)

        self._series_cache[key] = np.asarray(values, dtype=np.float32)
        self._total_cache_bytes += incoming
        return True

    def _cache_touch(self, key: tuple[str, str]) -> None:
        """Move *key* to the MRU (most-recently-used) end of the LRU queue."""
        if key in self._series_cache:
            val = self._series_cache.pop(key)
            self._series_cache[key] = val

    def _resultant_from_cached_components(self, demand: str) -> np.ndarray:
        e = self._series_cache[(demand, "e")]
        n = self._series_cache[(demand, "n")]
        z = self._series_cache[(demand, "z")]
        values = np.empty_like(e, dtype=np.float32)
        chunk_rows = self._series_chunk_rows(e.shape[1], component_count=4)
        for start in range(0, e.shape[0], chunk_rows):
            stop = min(start + chunk_rows, e.shape[0])
            out = values[start:stop, :]
            np.square(e[start:stop, :], out=out)
            out += n[start:stop, :] * n[start:stop, :]
            out += z[start:stop, :] * z[start:stop, :]
            np.sqrt(out, out=out)
        return values

    def _read_component_series(self, data_handle, component: str, column_selector) -> np.ndarray:
        row = {"e": 0, "n": 1, "z": 2}[component]
        n_nodes = data_handle.shape[0] // 3
        if isinstance(column_selector, slice):
            start = int(column_selector.start or 0)
            stop = int(column_selector.stop if column_selector.stop is not None else data_handle.shape[1])
            n_cols = max(0, stop - start)
        else:
            n_cols = len(column_selector)
        values = np.empty((n_nodes, n_cols), dtype=np.float32)
        chunk_nodes = self._series_chunk_rows(n_cols, component_count=3)
        for node_start in range(0, n_nodes, chunk_nodes):
            node_stop = min(node_start + chunk_nodes, n_nodes)
            row_start = node_start * 3
            row_stop = node_stop * 3
            raw = np.asarray(data_handle[row_start:row_stop, column_selector], dtype=np.float32)
            values[node_start:node_stop, :] = raw[row::3, :]
        return values

    def _read_component_triplet_series(
        self, data_handle, column_selector, n_cols: int, *, progress_cb=None
    ) -> dict[str, np.ndarray]:
        """Read E/N/Z for all nodes in node-row chunks with optional progress.

        Reads contiguous row blocks ``data_handle[row_start:row_stop, cols]``
        (HDF5-friendly rectangular access) then slices E/N/Z in NumPy.  This
        avoids strided HDF5 reads which would be ~10× slower.

        Each chunk reports ``(bytes_done, bytes_total)`` to *progress_cb* after
        the HDF5 read so the calling dialog can update in real time.
        """
        n_nodes = data_handle.shape[0] // 3
        values = {
            "e": np.empty((n_nodes, n_cols), dtype=np.float32),
            "n": np.empty((n_nodes, n_cols), dtype=np.float32),
            "z": np.empty((n_nodes, n_cols), dtype=np.float32),
        }
        chunk_nodes = self._series_chunk_rows(n_cols, component_count=3)
        total_bytes = n_nodes * n_cols * 4 * 3   # 3 components × float32
        bytes_done = 0
        for node_start in range(0, n_nodes, chunk_nodes):
            node_stop = min(node_start + chunk_nodes, n_nodes)
            row_start = node_start * 3
            row_stop = node_stop * 3
            raw = np.asarray(data_handle[row_start:row_stop, column_selector], dtype=np.float32)
            values["e"][node_start:node_stop, :] = raw[0::3, :]
            values["n"][node_start:node_stop, :] = raw[1::3, :]
            values["z"][node_start:node_stop, :] = raw[2::3, :]
            bytes_done += (node_stop - node_start) * n_cols * 4 * 3
            if progress_cb is not None:
                progress_cb(bytes_done, total_bytes)
        return values

    def _read_qa_component_series(self, handle, demand: str, component: str, column_selector) -> np.ndarray | None:
        if not self.has_qa:
            return None
        path = self._qa_path_for_demand(demand)
        row = {"e": 0, "n": 1, "z": 2}[component]
        return np.asarray(handle[path][row, column_selector], dtype=np.float32)

    def _read_qa_resultant_series(self, handle, demand: str, column_selector) -> np.ndarray | None:
        if not self.has_qa:
            return None
        path = self._qa_path_for_demand(demand)
        qa_data = np.asarray(handle[path][:, column_selector], dtype=np.float32)
        qa_reordered = qa_data[[2, 0, 1], :]
        return np.sqrt(np.sum(qa_reordered ** 2, axis=0)).astype(np.float32, copy=False)

    def _read_resultant_series(self, data_handle, column_selector, n_cols: int) -> np.ndarray:
        n_nodes = data_handle.shape[0] // 3
        values = np.empty((n_nodes, n_cols), dtype=np.float32)
        chunk_nodes = self._series_chunk_rows(n_cols, component_count=4)
        for node_start in range(0, n_nodes, chunk_nodes):
            node_stop = min(node_start + chunk_nodes, n_nodes)
            row_start = node_start * 3
            row_stop = node_stop * 3
            raw = np.asarray(data_handle[row_start:row_stop, column_selector], dtype=np.float32)
            out = values[node_start:node_stop, :]
            np.square(raw[0::3, :], out=out)
            out += raw[1::3, :] * raw[1::3, :]
            out += raw[2::3, :] * raw[2::3, :]
            np.sqrt(out, out=out)
        return values

    def _column_selector(self, total_cols: int):
        window_mask = getattr(self.model, "_window_mask", None)
        if window_mask is None:
            return slice(0, total_cols), total_cols

        col_idx = np.flatnonzero(np.asarray(window_mask, dtype=bool))
        if len(col_idx) == 0:
            return slice(0, 0), 0

        if len(col_idx) == int(col_idx[-1] - col_idx[0] + 1):
            start = int(col_idx[0])
            stop = int(col_idx[-1]) + 1
            return slice(start, stop), stop - start

        return col_idx.astype(np.int64), len(col_idx)

    @staticmethod
    def _resultant_chunk_cols(n_nodes: int) -> int:
        target_bytes = 64 * 1024 * 1024
        bytes_per_col = max(n_nodes * np.dtype(np.float32).itemsize * 3, 1)
        return max(16, target_bytes // bytes_per_col)

    @staticmethod
    def _series_chunk_rows(n_cols: int, *, component_count: int) -> int:
        target_bytes = 256 * 1024 * 1024
        bytes_per_node = max(int(n_cols) * np.dtype(np.float32).itemsize * int(component_count), 1)
        return max(1024, target_bytes // bytes_per_node)

    def _data_path_for_demand(self, demand: str) -> str:
        return {
            "accel": f"{self.model._data_grp}/acceleration",
            "vel": f"{self.model._data_grp}/velocity",
            "disp": f"{self.model._data_grp}/displacement",
        }[demand]

    def _qa_path_for_demand(self, demand: str) -> str:
        if not self.has_qa or self.model._qa_grp is None:
            raise KeyError("QA station is not available for this model.")
        return {
            "accel": f"{self.model._qa_grp}/acceleration",
            "vel": f"{self.model._qa_grp}/velocity",
            "disp": f"{self.model._qa_grp}/displacement",
        }[demand]

    def _gf_component_index(self, component: str) -> int:
        component = self._validate_component(component)
        if component not in GF_COMPONENTS:
            raise KeyError(
                f"GF component '{component}' is invalid. Use one of {', '.join(GF_COMPONENTS)}."
            )
        return GF_COMPONENTS.index(component)

    def _gf_slots_for_subfault(self, subfault_id: int = 0) -> np.ndarray:
        subfault_id = int(subfault_id)
        cached = self._gf_slot_cache.get(subfault_id)
        if cached is not None:
            return cached

        slots = []
        for node_id in self._display_node_ids:
            slots.append(int(self.model._get_slot(node_id, subfault_id)))
        values = np.asarray(slots, dtype=np.int64)
        self._gf_slot_cache[subfault_id] = values
        return values

    def _gf_t0_for_subfault(self, subfault_id: int = 0) -> np.ndarray:
        subfault_id = int(subfault_id)
        cached = self._gf_t0_cache.get(subfault_id)
        if cached is not None:
            return cached

        slots = self._gf_slots_for_subfault(subfault_id)
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise ImportError("GF field display requires h5py.") from exc

        with h5py.File(self.model._gf_h5_path, "r") as handle:
            if "t0" in handle:
                unique_slots, inverse = np.unique(slots, return_inverse=True)
                unique_t0 = np.asarray(handle["t0"][unique_slots.tolist()], dtype=float)
                t0_values = unique_t0[inverse]
            else:
                t0_values = np.zeros(len(slots), dtype=float)
        self._gf_t0_cache[subfault_id] = t0_values
        return t0_values

    def warm_gf_series(self, subfault_id: int = 0, component_index: int = 0) -> bool:
        """Pre-load the full GF field series into RAM with a single HDF5 read.

        Reads ``tdata[unique_slots, :, component_index]`` — shape
        ``(n_unique_slots, nt_gf)`` — then expands to ``(n_display_nodes, nt_gf)``
        via NumPy advanced indexing.  Subsequent calls to ``_gf_scalar_snapshot``
        use this in-memory array (pure NumPy, zero I/O) instead of re-opening
        the HDF5 file on every animation frame.

        Returns ``True`` when the series is now in cache (either already was,
        or was just loaded), ``False`` on any failure.
        """
        cache_key = (int(subfault_id), int(component_index))
        if cache_key in self._gf_series_cache:
            return True

        if not self.has_gf or not self.has_map:
            return False

        nt = int(getattr(self.model, "_nt_gf", 0))
        if nt <= 0:
            return False

        try:
            import h5py
        except ImportError:  # pragma: no cover
            return False

        try:
            slots = self._gf_slots_for_subfault(int(subfault_id))
            unique_slots, inverse = np.unique(slots, return_inverse=True)

            with h5py.File(self.model._gf_h5_path, "r") as f:
                # One contiguous(ish) read: (n_unique_slots, nt_gf)
                raw = np.asarray(
                    f["tdata"][unique_slots.tolist(), :, int(component_index)],
                    dtype=np.float32,
                )

            # Expand to all display nodes: (n_nodes, nt_gf)
            self._gf_series_cache[cache_key] = raw[inverse]
            return True
        except Exception:
            return False

    def _gf_scalar_snapshot(self, time_index: int, component: str, subfault_id: int = 0) -> np.ndarray:
        """Return per-node GF scalar at *time_index* for *component*.

        Hot path (warm cache)
        ---------------------
        Pure NumPy — maps simulation time → GF column index per node, then
        does one advanced-index lookup into the pre-warmed ``(n_nodes, nt_gf)``
        array.  Zero HDF5 I/O, < 0.1 ms per frame.

        Cold path (first call / cache miss)
        ------------------------------------
        Calls :meth:`warm_gf_series` which does **one** HDF5 read for the entire
        series, stores it, then serves the frame from RAM.  Subsequent frames are
        on the hot path.

        Legacy fallback
        ---------------
        If warming fails (e.g. h5py absent), the original per-frame scatter-read
        runs as before.
        """
        if not self.has_gf or not self.has_map:
            raise RuntimeError("GF/map data is not available in this viewer session.")

        component_index = self._gf_component_index(component)
        n = len(self._display_node_ids)

        if len(self.model.time) == 0:
            return np.zeros(n, dtype=np.float32)

        current_time = float(self.model.time[self.clamp_time_index(time_index)])
        dt = float(getattr(self.model, "_dt_orig", getattr(self.model, "dt", 0.0)))
        nt = int(getattr(self.model, "_nt_gf", 0))
        if dt <= 0.0 or nt <= 0:
            return np.zeros(n, dtype=np.float32)

        slots     = self._gf_slots_for_subfault(subfault_id)
        t0_values = self._gf_t0_for_subfault(subfault_id)

        # ── Hot / warm path: serve entirely from pre-warmed series ───────────
        cache_key = (int(subfault_id), int(component_index))
        series = self._gf_series_cache.get(cache_key)

        if series is None:
            # Lazy warm on first access — one HDF5 read, then cache forever.
            self.warm_gf_series(int(subfault_id), int(component_index))
            series = self._gf_series_cache.get(cache_key)

        if series is not None:
            # Pure NumPy: O(n_nodes) arithmetic + vectorised fancy index
            raw_pos   = (current_time - t0_values) / dt
            col_idx   = np.rint(raw_pos).astype(np.int64)
            valid     = (col_idx >= 0) & (col_idx < nt)
            values    = np.zeros(n, dtype=np.float32)
            if np.any(valid):
                vrows         = np.flatnonzero(valid)
                clipped_cols  = np.clip(col_idx[valid], 0, nt - 1)
                values[vrows] = series[vrows, clipped_cols]
            return values

        # ── Legacy fallback: per-frame scatter HDF5 read ─────────────────────
        # Reached only when warm_gf_series() failed (e.g. h5py unavailable).
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise ImportError("GF field display requires h5py.") from exc

        raw_positions = (current_time - t0_values) / dt
        column_indices = np.rint(raw_positions).astype(np.int64)
        valid = (column_indices >= 0) & (column_indices < nt)
        values = np.zeros(n, dtype=np.float32)
        if not np.any(valid):
            return values

        valid_rows  = np.flatnonzero(valid)
        valid_slots = slots[valid]
        valid_cols  = column_indices[valid]

        with h5py.File(self.model._gf_h5_path, "r") as handle:
            tdata = handle["tdata"]
            for source_col in np.unique(valid_cols):
                local_mask  = valid_cols == int(source_col)
                row_subset  = valid_rows[local_mask]
                slot_subset = valid_slots[local_mask]
                u_slots, inv = np.unique(slot_subset, return_inverse=True)
                read_vals = np.asarray(
                    tdata[u_slots.tolist(), int(source_col), component_index],
                    dtype=np.float32,
                )
                values[row_subset] = read_vals[inv]
        return values

    def _gf_default_scalar_limits(self, component: str, subfault_id: int = 0) -> tuple[float, float]:
        key = (int(subfault_id), self._validate_component(component))
        if key in self._gf_limit_cache:
            return self._gf_limit_cache[key]

        component_index = self._gf_component_index(component)

        # ── Fast path: series already in RAM (pre-warmed by playback) ────────
        series = self._gf_series_cache.get((int(subfault_id), int(component_index)))
        if series is not None:
            vmax = float(np.max(np.abs(series))) if series.size else 0.0
            if vmax <= 0.0:
                vmax = 1.0
            limits = (-vmax, vmax)
            self._gf_limit_cache[key] = limits
            return limits

        # ── Normal path: one HDF5 read of all unique slots for this component ─
        try:
            import h5py
        except ImportError:  # pragma: no cover
            snapshot = self._gf_scalar_snapshot(0, component, subfault_id)
            limits = scalar_limits(snapshot, component)
            self._gf_limit_cache[key] = limits
            return limits

        slots = self._gf_slots_for_subfault(subfault_id)
        if len(slots) == 0:
            return -1.0, 1.0

        with h5py.File(self.model._gf_h5_path, "r") as handle:
            unique_slots = np.unique(slots)
            data = np.asarray(handle["tdata"][unique_slots.tolist(), :, component_index], dtype=np.float32)
        vmax = float(np.max(np.abs(data))) if data.size else 0.0
        if vmax <= 0.0:
            vmax = 1.0
        limits = (-vmax, vmax)
        self._gf_limit_cache[key] = limits
        return limits
