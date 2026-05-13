"""
smoke_test.py
=============
Internal sanity check for ShakerMakerResults during the documentation pass.

The goal is *not* to test correctness of computations (we have notebooks for
that) but to make sure that:

  1) Every public entry point still imports.
  2) The three legacy "shim" import paths still resolve.
  3) Identity invariants hold (e.g. DRMData is ShakerMakerData).
  4) Lazy-facade `__getattr__` still wires every name in `__all__`.
  5) The viewer module imports even when Qt/PyVista are not exercised.

Run with the worktree on PYTHONPATH so we test the working copy, not the
installed editable package:

    set PYTHONPATH=<worktree>/src
    python scripts/smoke_test.py

Exits non-zero on the first failure so the harness can stop early.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback


# --- small helpers ----------------------------------------------------------

_FAIL: list[str] = []


def _ok(label: str) -> None:
    print(f"  [ok]   {label}")


def _fail(label: str, exc: BaseException) -> None:
    print(f"  [FAIL] {label}")
    print("         " + repr(exc))
    _FAIL.append(label)


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


# --- 1) raw package import --------------------------------------------------

_section("package import")
try:
    import ShakerMakerResults as smr  # noqa: E402
    _ok(f"import ShakerMakerResults  ->  {smr.__file__}")
except Exception as exc:  # pragma: no cover - smoke
    _fail("import ShakerMakerResults", exc)
    traceback.print_exc()
    sys.exit(1)


# --- 2) public API (`__all__`) is wired -------------------------------------

_section("public API (__all__) lazy resolution")
missing: list[str] = []
for name in smr.__all__:
    try:
        getattr(smr, name)
    except Exception as exc:
        missing.append(name)
        _fail(f"smr.{name}", exc)
if not missing:
    _ok(f"all {len(smr.__all__)} names resolved")


# --- 3) legacy shim modules at the package root -----------------------------
# We have three tiny re-exporters at the root that older notebooks may import
# directly. They must keep working even if we never use them internally.

_section("legacy root shims")
shim_checks = [
    ("ShakerMakerResults.newmark",          "NewmarkSpectrumAnalyzer"),
    ("ShakerMakerResults.shakermaker_data", "ShakerMakerData"),
    ("ShakerMakerResults.shakermaker_data", "DRMData"),
    ("ShakerMakerResults.shakermaker_data", "SurfaceData"),
    ("ShakerMakerResults.station_data",     "StationData"),
]
for mod_name, attr_name in shim_checks:
    try:
        mod = importlib.import_module(mod_name)
        getattr(mod, attr_name)
        _ok(f"from {mod_name} import {attr_name}")
    except Exception as exc:
        _fail(f"from {mod_name} import {attr_name}", exc)


# --- 4) identity invariants -------------------------------------------------
# The shim and the canonical location must hand us the same class object,
# otherwise `isinstance(...)` checks break for users who mix the two paths.

_section("identity invariants")
try:
    from ShakerMakerResults.core.shakermaker_data import ShakerMakerData as _Canonical
    from ShakerMakerResults.shakermaker_data import ShakerMakerData as _ViaShim
    from ShakerMakerResults import ShakerMakerData as _ViaFacade

    assert _Canonical is _ViaShim,   "shim diverged from canonical"
    assert _Canonical is _ViaFacade, "facade diverged from canonical"
    _ok("ShakerMakerData identity: canonical is shim is facade")

    from ShakerMakerResults.core.shakermaker_data import DRMData, SurfaceData
    assert DRMData is _Canonical
    assert SurfaceData is _Canonical
    _ok("DRMData is SurfaceData is ShakerMakerData")
except Exception as exc:
    _fail("identity invariants", exc)


# --- 5) plotting facade ----------------------------------------------------

_section("plotting facade")
try:
    from ShakerMakerResults import plotting as _plot
    for name in _plot.__all__:
        getattr(_plot, name)
    _ok(f"plotting facade resolved {len(_plot.__all__)} names")
except Exception as exc:
    _fail("plotting facade", exc)


# --- 6) analysis ------------------------------------------------------------

_section("analysis package")
try:
    from ShakerMakerResults.analysis.newmark import NewmarkSpectrumAnalyzer
    from ShakerMakerResults.analysis.arias_intensity import AriasIntensityAnalyzer
    from ShakerMakerResults.analysis.vmax_service import compute_vmax  # noqa: F401
    # Run a tiny Newmark to make sure numba JIT is happy.
    import numpy as _np
    dummy = _np.zeros(64, dtype=float)
    res = NewmarkSpectrumAnalyzer.compute(dummy, dt=0.01,
                                          max_period=0.05, intervals=0.01)
    assert "T" in res and "PSa" in res
    _ok("NewmarkSpectrumAnalyzer.compute(zeros) returns dict with T/PSa")

    # Arias should produce a 0..100 curve.
    ia, t0, t1, ia_total, _ = AriasIntensityAnalyzer.compute(
        _np.ones(64) * 1.0, dt=0.01)
    assert 0.0 <= ia[-1] <= 100.0 + 1e-6
    _ok("AriasIntensityAnalyzer.compute(ones) produces normalised curve")
except Exception as exc:
    _fail("analysis package", exc)


# --- 7) viewer import (no Qt loop) -----------------------------------------

_section("viewer package (import only, no GUI)")
try:
    from ShakerMakerResults import viewer as _viewer
    # The names below are part of the public viewer surface.
    for name in ("ViewerSession", "ViewerState", "ViewerDataAdapter"):
        getattr(_viewer, name)
    _ok("ViewerSession / ViewerState / ViewerDataAdapter importable")
except Exception as exc:
    # Viewer deps are optional at import time. We do not fail the smoke
    # for missing PyQt/PyVista, but we do log them so we know what is going
    # on in CI.
    msg = repr(exc)
    if "PyQt" in msg or "PySide" in msg or "pyvista" in msg.lower():
        print(f"  [skip] viewer optional deps missing: {msg}")
    else:
        _fail("viewer package import", exc)


# --- final summary ----------------------------------------------------------

_section("summary")
if _FAIL:
    print(f"  {len(_FAIL)} failure(s):")
    for label in _FAIL:
        print(f"    - {label}")
    sys.exit(1)

print("  ALL GREEN.")
sys.exit(0)
