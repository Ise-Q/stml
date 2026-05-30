"""Session setup: pin native-library threading + OpenMP before any heavy import.

XGBoost, LightGBM, torch, and the BLAS behind numpy each bundle an OpenMP runtime; loading
several of them in one process triggers a libomp duplicate-runtime crash (segfault) on macOS.
Forcing a single OpenMP/BLAS thread and allowing the duplicate runtime both stabilises the
process AND removes thread-scheduling nondeterminism from the native kernels (XGBoost /
LightGBM / BLAS) — which serves the project's determinism contract directly.

These environment variables only take effect if set *before* the native libraries
initialise, so they live in ``conftest.py`` (imported by pytest before any test module).
Runtime entry points (``pipeline``/``emit``) set the same vars at the top of ``__main__``.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
