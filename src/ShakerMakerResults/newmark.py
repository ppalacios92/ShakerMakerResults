"""
newmark.py  (root-level shim)
=============================
Backwards-compatible re-export so notebooks that pre-date the package
re-organisation keep working unchanged::

    from ShakerMakerResults.newmark import NewmarkSpectrumAnalyzer

The canonical home for this class is :mod:`ShakerMakerResults.analysis.newmark`
and new code should import it from there (or, even better, lazily from the
top-level package: ``from ShakerMakerResults import NewmarkSpectrumAnalyzer``).

This file is intentionally tiny -- if it grows, we are doing something wrong.
The smoke test in ``scripts/smoke_test.py`` checks that the shim and the
canonical class remain the *same* object (identity, not just equality), so
old ``isinstance(...)`` calls keep behaving.
"""

from .analysis.newmark import NewmarkSpectrumAnalyzer

__all__ = ["NewmarkSpectrumAnalyzer"]
