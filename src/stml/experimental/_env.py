"""Pin every native math kernel to a single thread for byte-stable re-runs.

Plan R3 (byte-identical emit) and R5 (determinism). Lifted from
``metamodel-apb/src/alken_metamodel/_env.py`` (alken parity) with attribution.

The side effects must fire *before* any of {numpy, scipy, scikit-learn, xgboost,
lightgbm, torch, tensorflow} imports its native backend. The experimental package
imports this module at top level so a downstream
``from stml.experimental import labels`` chain runs the env setup first.

Why every kernel is single-threaded:
    * BLAS / OpenMP / MKL thread pools schedule reductions non-deterministically;
      a re-run on the same machine can produce different bytes.
    * macOS ships its own libomp under ``/opt/homebrew/lib/libomp.dylib``. When
      both xgboost (bundled libomp) and torch (bundled libomp) load, two libomps
      collide and the process segfaults. Setting ``KMP_DUPLICATE_LIB_OK=TRUE``
      tells Intel OMP to accept the duplicate.
"""

from __future__ import annotations

import os

_ENV_VARS = {
    # BLAS thread pools
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    # OpenMP family
    "OMP_NUM_THREADS": "1",
    "OMP_PROC_BIND": "false",
    # macOS libomp duplicate-load fix — alken parity
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    # Disable hash randomisation so dict iteration order is byte-stable
    "PYTHONHASHSEED": "42",
}

for _key, _value in _ENV_VARS.items():
    # Respect a user override (set in the shell before launching python).
    os.environ.setdefault(_key, _value)


def applied_env() -> dict[str, str]:
    """Return the env vars this module pinned; used in tests + experiment logs."""
    return {k: os.environ.get(k, "") for k in _ENV_VARS}
