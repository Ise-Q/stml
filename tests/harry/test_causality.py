"""Universal causality harness for Harry's feature functions.

Every feature function in ``src/stml/harry/features/`` registers itself
here by adding one entry to :data:`REGISTRATIONS`. The harness then
parametrises three universal tests over the registry:

1. **Truncation-invariance** — the value at row ``t`` computed on
   ``panel.iloc[: t + 1]`` must equal the value at row ``t`` computed on
   the full panel. This is the canonical no-leakage check.

2. **Shape preservation** — output length matches input length.

3. **No NaN / Inf past warmup** — after the documented ``warmup_window``,
   every output value must be finite. NaN inside the warmup is allowed.

Initially :data:`REGISTRATIONS` is empty; each feature module commit
appends its entries. Empty parametrize collections are skipped cleanly.

Registration schema::

    {
      "name":   str,        # short id for pytest output
      "module": str,        # importable module path
      "func":   str,        # callable name in the module
      "adapter": str,       # named adapter (see :data:`ADAPTERS` below)
      "kwargs": dict,       # extra kwargs (e.g. window sizes, halflives)
      "warmup": int,        # leading rows allowed to be NaN
      "data_kind": str,     # synthetic panel kind (see :func:`_synth_panel`)
    }

The ``adapter`` field maps the synthetic panel to the positional
arguments the feature function actually wants. New adapters live in
:data:`ADAPTERS`; they are intentionally explicit so a teammate reading
the registry can see what each feature consumes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Synthetic input panels                                                       #
# --------------------------------------------------------------------------- #
def _synth_single_instrument(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """One-instrument OHLCV + signal + return panel.

    Used by every single-instrument feature. Returns are i.i.d. Gaussian
    so EWMA / rolling estimators are well-behaved; the signal is a
    mostly-stationary {-1, 0, +1} draw with persistence.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    r = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(r))
    intra_up = rng.uniform(0.0005, 0.008, n)
    intra_dn = rng.uniform(0.0005, 0.008, n)
    high = close * np.exp(intra_up)
    low = close * np.exp(-intra_dn)
    open_ = close * np.exp(rng.normal(0.0, 0.002, n))
    volume = rng.integers(1_000, 100_000, n).astype(float)
    open_interest = rng.integers(5_000, 100_000, n).astype(float)
    # Persistent signal: every 10 bars we redraw; otherwise we carry forward.
    base = rng.choice([-1, 0, 1], size=(n + 9) // 10, p=[0.4, 0.2, 0.4])
    s = np.repeat(base, 10)[:n].astype(float)
    # Rolling-60 realised vol — features in conditional_risk need this as
    # the barrier scale. NaN for the first 59 rows.
    vol = pd.Series(r).rolling(60, min_periods=60).std().to_numpy()
    panel = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "open_interest": open_interest,
            "signal": s,
            "return": r,
            "vol": vol,
        },
        index=dates,
    )
    panel.index.name = "date"
    return panel


def _synth_returns_panel(
    instruments: tuple[str, ...] = (
        "es1s", "nq1s", "fesx1s",
        "cl1s", "ho1s", "rb1s", "ng1s",
        "gc1s", "si1s", "hg1s", "pl1s",
    ),
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Wide returns panel ``date × instrument`` for cross-asset features."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    data = {
        inst: rng.normal(0.0, 0.01, n)
        for inst in instruments
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


def _synth_panel(kind: str, n: int = 500, seed: int = 42) -> pd.DataFrame:
    if kind == "single_instrument":
        return _synth_single_instrument(n=n, seed=seed)
    if kind == "returns_panel":
        return _synth_returns_panel(n=n, seed=seed)
    raise ValueError(f"unknown data_kind {kind!r}")


# --------------------------------------------------------------------------- #
# Adapters — extract positional args from a synthetic panel                    #
# --------------------------------------------------------------------------- #
ADAPTERS: dict[str, Callable[[pd.DataFrame], tuple]] = {
    # Adapters take the synthetic panel and return a tuple of positional
    # arguments to pass to the registered feature function. Each adapter
    # is a tiny, named function so the registry stays readable.
    "signal":          lambda df: (df["signal"],),
    "signal_returns":  lambda df: (df["signal"], df["return"]),
    "close":           lambda df: (df["close"],),
    "returns":         lambda df: (df["return"],),
    "returns_vol":     lambda df: (df["return"], df["vol"]),
    "vol_signal_return": lambda df: (df["vol"], df["signal"], df["return"]),
    "returns_volume":  lambda df: (df["return"], df["volume"]),
    "close_only":      lambda df: (df["close"],),
    "open_close_lagged": lambda df: (df["open"], df["close"].shift(1)),
    "panel":           lambda df: (df,),
}


def _adapt(panel: pd.DataFrame, adapter: str) -> tuple:
    if adapter not in ADAPTERS:
        raise KeyError(f"unknown adapter {adapter!r}; register it in test_causality.py")
    return ADAPTERS[adapter](panel)


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
# Each feature module exposes a module-level constant
# ``CAUSALITY_REGISTRATIONS`` (a list of dicts following the schema above)
# and the harness picks them up automatically. Adding a new feature module
# requires no edits here — just create the file and list it in
# ``_FEATURE_MODULES``.
_FEATURE_MODULES: tuple[str, ...] = (
    "stml.harry.features.signal_trajectory",
    "stml.harry.features.conditional_risk",
    "stml.harry.features.information_theoretic",
    "stml.harry.features.microstructure_fixed",
    "stml.harry.features.cross_asset",
    "stml.harry.features.wavelet",
    "stml.harry.features.concept_drift",
)

REGISTRATIONS: list[dict] = []
for _module_name in _FEATURE_MODULES:
    try:
        _mod = importlib.import_module(_module_name)
    except ImportError:
        # Acceptable for optional extras (e.g. ``wavelet`` if pywavelets
        # is not installed). The harness silently drops the module's
        # entries; the unit tests in tests/harry/test_<name>.py will
        # raise explicit ImportErrors if the user tries to use them.
        continue
    REGISTRATIONS.extend(getattr(_mod, "CAUSALITY_REGISTRATIONS", ()))


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _call(reg: dict, panel: pd.DataFrame):
    func = getattr(importlib.import_module(reg["module"]), reg["func"])
    args = _adapt(panel, reg["adapter"])
    return func(*args, **reg.get("kwargs", {}))


def _assert_equal_or_both_nan(a, b, label: str) -> None:
    if pd.isna(a) and pd.isna(b):
        return
    if pd.isna(a) or pd.isna(b):
        pytest.fail(f"{label}: one is NaN ({a!r} vs {b!r})")
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        assert int(a) == int(b), f"{label}: {a!r} != {b!r}"
    else:
        assert float(a) == pytest.approx(float(b), rel=1e-9, abs=1e-12), (
            f"{label}: {a!r} != {b!r}"
        )


# --------------------------------------------------------------------------- #
# Universal tests                                                              #
# --------------------------------------------------------------------------- #
if REGISTRATIONS:
    parametrize = pytest.mark.parametrize(
        "reg", REGISTRATIONS, ids=lambda r: r["name"],
    )
else:
    # Empty registry: emit a single skip so the file has at least one item to
    # collect (avoids pytest's "no tests ran" warning) while making the empty
    # state explicit. Each subsequent feature module commit will populate
    # REGISTRATIONS and these tests will run for real.
    parametrize = pytest.mark.skip(reason="no Harry features registered yet")


@parametrize
def test_truncation_invariance(reg: dict):
    """``feature(panel.iloc[:t+1]).iloc[t]`` must equal ``feature(panel).iloc[t]``.

    Checked at three values of ``t`` (100, 200, 400) on a 500-row panel.
    Tolerates both NaN-on-NaN (warmup) and a small numerical tolerance.
    """
    panel = _synth_panel(reg["data_kind"], n=500, seed=42)
    full_out = _call(reg, panel)
    assert len(full_out) == len(panel), f"{reg['name']}: full length mismatch"

    for t in (100, 200, 400):
        trunc = panel.iloc[: t + 1]
        trunc_out = _call(reg, trunc)
        assert len(trunc_out) == t + 1, (
            f"{reg['name']}: truncated length mismatch at t={t}"
        )
        if isinstance(full_out, pd.DataFrame):
            assert isinstance(trunc_out, pd.DataFrame)
            assert list(full_out.columns) == list(trunc_out.columns), (
                f"{reg['name']}: columns differ between full and truncated"
            )
            for col in full_out.columns:
                _assert_equal_or_both_nan(
                    full_out[col].iloc[t],
                    trunc_out[col].iloc[t],
                    label=f"{reg['name']}.{col} @ t={t}",
                )
        else:
            _assert_equal_or_both_nan(
                full_out.iloc[t],
                trunc_out.iloc[t],
                label=f"{reg['name']} @ t={t}",
            )


@parametrize
def test_shape_preservation(reg: dict):
    """``len(feature(panel)) == len(panel)``."""
    panel = _synth_panel(reg["data_kind"], n=300, seed=42)
    out = _call(reg, panel)
    assert len(out) == len(panel), f"{reg['name']}: length {len(out)} != {len(panel)}"


@parametrize
def test_no_nan_or_inf_after_warmup(reg: dict):
    """Past the warmup window every output value is finite."""
    panel = _synth_panel(reg["data_kind"], n=400, seed=42)
    out = _call(reg, panel)
    warmup = int(reg["warmup"])
    if isinstance(out, pd.DataFrame):
        for col in out.columns:
            tail = out[col].iloc[warmup:]
            assert tail.notna().all(), (
                f"{reg['name']}.{col} has NaN past warmup={warmup}"
            )
            assert np.isfinite(tail).all(), (
                f"{reg['name']}.{col} has Inf past warmup={warmup}"
            )
    else:
        tail = out.iloc[warmup:]
        assert tail.notna().all(), (
            f"{reg['name']} has NaN past warmup={warmup}"
        )
        assert np.isfinite(tail).all(), (
            f"{reg['name']} has Inf past warmup={warmup}"
        )
