"""Alken meta-labelling metamodel (T3.03 coursework).

A secondary *act/skip* classifier over the provided primary trading signal for
11 futures instruments across three asset-class metamodels (Equity/Energy/Metals).

The design, citations, and build sequence live in
``docs/plans/2026-05-30-metamodel-build.md``. The methodological backbone is the
literature review at ``../reports/apb/nlr-cw-v1.md`` (8 commitments, 60 refs).

Leakage discipline: every feature is recomputed *inside* each CV fold on the
fold-train slice (we reuse stml's causal feature *functions*, never its frozen
``feature_matrix.parquet``); validation is purged k-fold + embargo -> CPCV ->
nested CPCV -> a single Jan-Jun 2022 hold-out. See ``cross_validation`` and
``pipeline``.
"""

# Importing the package pins OpenMP/BLAS threads BEFORE any heavy submodule (xgboost/
# lightgbm/torch) loads its native runtime — Python imports this parent first, so every
# entry point inherits the single-threaded determinism + macOS libomp stability guard.
from . import _env  # noqa: F401

__all__: list[str] = []
