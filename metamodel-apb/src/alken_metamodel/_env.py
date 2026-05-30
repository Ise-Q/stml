"""Pin OpenMP/BLAS threading before native libraries initialise.

XGBoost, LightGBM, torch and the BLAS behind numpy each bundle an OpenMP runtime; co-loading
them crashes on macOS (libomp duplicate runtime) and introduces thread-scheduling
nondeterminism. Importing this module (which must happen before those libraries load) forces a
single OpenMP/BLAS thread and tolerates the duplicate runtime — stabilising the process AND
serving the determinism contract. ``conftest.py`` and the ``emit`` entry point import this first.
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
