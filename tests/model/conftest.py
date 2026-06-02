"""Pytest fixtures for the meta-model (``stml.model``) test package.

Provides:

* ``src/`` on ``sys.path`` (editable-install-free imports, like ``tests/harry/conftest.py``).
* A deterministic **synthetic panel** (3 instruments, contiguous ``bar_pos``, a learnable binary
  target) plus its matching wide **close panel** and **labels** frame -- enough to exercise the
  CPCV purge/embargo, per-fold uniqueness weights, calibration and economic-eval logic without
  touching the real (slow) matrix.
* A small **real-matrix slice** (2 instruments, dev rows, ``bar_pos`` attached) for the end-to-end
  smoke test and anything that needs genuine feature distributions.

All randomness is seeded with 42 (the project-mandated seed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

SEED = 42
INSTRUMENTS = ("AAA", "BBB", "CCC")
N_BARS = 72  # divisible by 6 -> clean CPCV blocks of 12


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """Deterministic ``date x instrument`` event panel with contiguous per-instrument ``bar_pos``.

    Columns: ``date, instrument, bar_pos, partition, f1_a, f1_b, f2_vol, side, sigma, bin``.
    All three instruments share the same ``N_BARS`` business-day calendar, so ``bar_pos`` equals
    the positional index 0..N_BARS-1 on each. The target ``bin`` is a noisy function of ``f1_a``
    so a classifier reaches AUC > 0.5 (both classes always present panel-wide).
    """
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2020-01-02", periods=N_BARS)
    frames = []
    for inst in INSTRUMENTS:
        f1_a = rng.normal(size=N_BARS)
        f1_b = rng.normal(size=N_BARS)
        f2_vol = np.abs(rng.normal(0.012, 0.003, size=N_BARS)) + 1e-3
        logit = 1.3 * f1_a - 0.5 * f1_b + rng.normal(scale=0.6, size=N_BARS)
        bin_ = (logit > np.median(logit)).astype(int)
        side = np.where(rng.uniform(size=N_BARS) > 0.4, 1, -1)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "instrument": inst,
                    "bar_pos": np.arange(N_BARS, dtype=int),
                    "partition": "train",
                    "f1_a": f1_a,
                    "f1_b": f1_b,
                    "f2_vol": f2_vol,
                    "side": side.astype(float),
                    "sigma": f2_vol,
                    "bin": bin_,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["instrument", "date"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_close() -> pd.DataFrame:
    """Wide ``date x instrument`` close panel matching :func:`synthetic_panel`'s calendar."""
    rng = np.random.default_rng(SEED + 1)
    dates = pd.bdate_range("2020-01-02", periods=N_BARS)
    data = {}
    for inst in INSTRUMENTS:
        rets = rng.normal(0.0, 0.012, size=N_BARS)
        data[inst] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates).sort_index()


@pytest.fixture(scope="session")
def synthetic_labels(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """Triple-barrier-style labels frame (``date, instrument, t1, ret, bin``) with overlap.

    ``t1`` is set 4 bars after each event on the shared calendar, so consecutive events overlap --
    the structure :func:`stml.model.labels.sample_uniqueness` down-weights and the per-fold weight
    recompute tests rely on.
    """
    dates = pd.bdate_range("2020-01-02", periods=N_BARS)
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for inst in INSTRUMENTS:
        g = synthetic_panel[synthetic_panel["instrument"] == inst]
        for _, r in g.iterrows():
            p = int(r["bar_pos"])
            t1 = dates[min(p + 4, N_BARS - 1)]
            rows.append((r["date"], inst, t1, float(rng.normal(0.0, 0.01)), int(r["bin"])))
    return pd.DataFrame(rows, columns=["date", "instrument", "t1", "ret", "bin"])


@pytest.fixture(scope="session")
def real_slice():
    """Small real-matrix slice (2 instruments, dev rows) with ``bar_pos`` + a wide close panel.

    Returns ``(matrix_slice, close_wide)``. Loads the genuine feature matrix once per session; used
    by the end-to-end smoke test and any check that needs real feature distributions.
    """
    from stml.model.dataset import DEV_PARTITIONS, attach_bar_pos, close_panel, load_matrix

    m = load_matrix()
    cp = close_panel()
    m = attach_bar_pos(m, cp)
    keep = m["instrument"].isin(["es1s", "si1s"]) & m["partition"].isin(DEV_PARTITIONS)
    sl = m[keep].sort_values(["instrument", "date"]).reset_index(drop=True)
    return sl, cp
