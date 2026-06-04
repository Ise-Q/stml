"""Shared pytest fixtures for the ``stml.experimental`` test suite.

Plan §9 (RED-first TDD) + plan §10 (determinism). The conftest:

* Seeds every RNG before each test runs, so a test that uses random data is
  byte-stable across pytest invocations.
* Re-applies the single-thread native-kernel env (in case a child process
  inherited a different default).
* Exposes ``synth_ohlc(n=300, seed=42)`` and friends — synthetic panels used by
  unit tests so they don't depend on the released data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stml.experimental import seeding


@pytest.fixture(autouse=True)
def _set_seeds_per_test() -> None:
    """Reseed every RNG before each test. ``autouse`` so no test forgets."""
    seeding.set_seeds(seed=42)


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repo root — used by tests that need to load real data."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


@pytest.fixture
def synth_close() -> pd.Series:
    """A deterministic 300-bar close series with a regime break at t=150."""
    seeding.set_seeds(seed=42)
    rng = np.random.default_rng(42)
    log_rets = rng.normal(0, 0.01, 300)
    log_rets[150:] += 0.005  # post-150 drift
    close = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.Series(
        close,
        index=pd.bdate_range("2020-01-02", periods=300),
        name="close",
    )


@pytest.fixture
def synth_ohlc(synth_close: pd.Series) -> pd.DataFrame:
    """A deterministic 300-bar OHLC + volume panel derived from ``synth_close``."""
    seeding.set_seeds(seed=42)
    rng = np.random.default_rng(123)
    close = synth_close.values
    # Generate intra-bar ranges as 30 % of stdev — plausible OHLC structure.
    log_close = np.log(close)
    high = np.exp(log_close + np.abs(rng.normal(0, 0.005, len(close))))
    low = np.exp(log_close - np.abs(rng.normal(0, 0.005, len(close))))
    open_ = np.exp(log_close + rng.normal(0, 0.003, len(close)))
    volume = np.maximum(rng.poisson(10000, len(close)), 1).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, high, close]),
            "low": np.minimum.reduce([open_, low, close]),
            "close": close,
            "volume": volume,
        },
        index=synth_close.index,
    )


@pytest.fixture
def synth_signal(synth_close: pd.Series) -> pd.Series:
    """A deterministic +1/0/-1 signal aligned to ``synth_close``."""
    seeding.set_seeds(seed=42)
    rng = np.random.default_rng(7)
    # 40 % +1, 20 % -1, 40 % zero — close to the equity profile.
    raw = rng.choice([+1, 0, -1], size=len(synth_close), p=[0.4, 0.4, 0.2])
    return pd.Series(raw, index=synth_close.index, name="signal").astype(int)
