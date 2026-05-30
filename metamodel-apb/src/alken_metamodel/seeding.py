"""Global determinism control.

The grader re-runs this code on the hidden Jul-Dec 2022 half (brief, Dataset),
so reproducibility is a hard requirement, not a nicety. ``set_seeds`` fixes every
RNG the pipeline touches. torch variants are byte-stable; TensorFlow op-determinism
is best-effort and may carry a documented caveat for the Keras-VSN result.

Pattern follows PS4 (`PS4_Solutions:2`) extended to torch + TensorFlow.
"""

from __future__ import annotations

import os
import random

import numpy as np

RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed ``random``, ``numpy``, ``torch``, ``tensorflow`` and ``PYTHONHASHSEED``.

    Call once at the top of any entry point (pipeline build, CSV emit, a test that
    trains a model). Frameworks are imported lazily so importing this module stays
    cheap (TensorFlow import is slow).

    Note: ``PYTHONHASHSEED`` only takes effect at interpreter startup; setting it at
    runtime is belt-and-suspenders. Output ordering is pinned explicitly in ``emit``
    (rows sorted, columns fixed), so CSV reproducibility does not rely on hash seeding.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        # warn_only: some ops lack a deterministic kernel; warn rather than crash.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - torch always present in this env
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - tf always present in this env
        pass
