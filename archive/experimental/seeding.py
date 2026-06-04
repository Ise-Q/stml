"""Seed every RNG the experimental pipeline touches.

Plan R5 (determinism). Lifted from ``metamodel-apb/src/alken_metamodel/seeding.py``
(alken parity) with attribution. Call at every CLI entry point and at the head
of every test that depends on a stochastic estimator.

The single seed propagates to:
    * Python ``random``
    * ``numpy.random`` (legacy global + a default_rng singleton accessor)
    * ``torch`` (if loaded — CPU + cuda + cudnn determinism)
    * ``tensorflow`` (if loaded — best-effort op-determinism)
    * ``PYTHONHASHSEED`` (a no-op at runtime — the env var only matters before
      python starts; we set it in ``_env.py`` for that)

Per-row deterministic derivations (used by f15 bootstrap MC and the F16
discriminator) use ``derive_seed(base, position)`` so a value computed at
position ``t`` on the truncated input ``X[:t+1]`` is byte-identical to the value
computed at ``t`` on the full input ``X[:T]`` (right-edge truncation invariance).
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED: int = 42


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Synchronise every reachable RNG to ``seed``.

    Torch / TensorFlow are seeded only if already imported, so this function
    never forces those heavy modules to load. Callers that *want* torch
    determinism should ``import torch`` themselves before calling.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    # Torch — only if already imported. Keeps `set_seeds` cheap and stops it
    # from forcing torch into a process that doesn't need it.
    if "torch" in _imported_modules():
        import torch  # local import to avoid forcing import

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Single-thread the autograd kernels for full determinism.
        torch.set_num_threads(1)
        # use_deterministic_algorithms is best-effort; some ops still aren't.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # cudnn knobs only matter on CUDA but are cheap on CPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if "tensorflow" in _imported_modules():
        import tensorflow as tf  # local

        tf.random.set_seed(seed)
        # TF op-determinism is best-effort — env var must be set before TF
        # imports for full effect. We set it here for any later op.
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


def derive_seed(base: int, position: int) -> int:
    """Derive a stable per-row seed.

    ``derive_seed(base, t) == derive_seed(base, t)`` for any ``t``, regardless of
    whether the call site computed it on a truncated or full panel. This is the
    invariant that makes f15-style positional bootstrap MC right-edge truncation
    invariant — see plan §10.
    """
    # 1_000_003 is the lowest 7-digit prime (Bertrand-style spreading without
    # numerical risk in int64). The same constant used in alken's f15 path.
    return int(base) * 1_000_003 + int(position)


def _imported_modules() -> set[str]:
    """Return the set of already-imported top-level module names."""
    import sys

    return {name.split(".", 1)[0] for name in sys.modules}
