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


def _synth_macro_panel(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic macro panel for M1–M6 feature tests.

    All columns are fully non-NaN (no structural missing data) so the
    no-NaN-past-warmup harness test passes cleanly on the synthetic panel.

    * EIA columns change every 5 rows (weekly release cadence).
    * PMI columns change every 21 rows (monthly release cadence).
    * All level series are constrained to physically meaningful ranges.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n, freq="B")

    # M1 — volatility
    vix  = np.clip(15.0 + np.cumsum(rng.normal(0, 0.3, n)), 8.0, 80.0)
    vix3m = np.clip(vix + rng.normal(0, 0.5, n), 8.0, 85.0)
    move  = np.clip(80.0 + np.cumsum(rng.normal(0, 1.0, n)), 40.0, 200.0)
    skew  = np.clip(115.0 + np.cumsum(rng.normal(0, 0.5, n)), 100.0, 150.0)

    # M2 — rates
    ust10 = np.clip(3.0 + np.cumsum(rng.normal(0, 0.02, n)), 0.1, 10.0)
    ust2  = np.clip(ust10 - 1.0 + rng.normal(0, 0.1, n), 0.05, 9.0)
    bund  = np.clip(1.0 + np.cumsum(rng.normal(0, 0.02, n)), -0.9, 5.0)
    tips  = np.clip(ust10 - 2.0 + rng.normal(0, 0.05, n), -2.0, 5.0)
    be    = np.clip(2.0 + rng.normal(0, 0.05, n), 0.5, 4.0)

    # M3 — credit
    hy = np.clip(400.0 + np.cumsum(rng.normal(0, 2.0, n)), 100.0, 2000.0)
    ig = np.clip(100.0 + np.cumsum(rng.normal(0, 1.0, n)),  20.0,  500.0)

    # M4 — FX
    dxy    = np.clip(95.0 + np.cumsum(rng.normal(0, 0.1, n)), 70.0, 120.0)
    eurusd = np.clip(1.1  + np.cumsum(rng.normal(0, 0.002, n)), 0.8, 1.6)

    # M5 — EIA weekly (change every 5 rows) + copper + BDI (daily)
    def _eia_walk(start: float, step_std: float) -> np.ndarray:
        vals = np.empty(n, dtype=float)
        level = start
        for i in range(n):
            if i % 5 == 0:
                level += rng.normal(0, step_std)
            vals[i] = level
        return vals

    crude    = _eia_walk(450_000.0, 2_000.0)
    dist_    = _eia_walk(120_000.0, 1_000.0)
    gasoline = _eia_walk(220_000.0, 1_500.0)
    ng       = _eia_walk(3_000.0, 100.0)
    copper   = np.clip(200_000.0 + np.cumsum(rng.normal(0, 500.0, n)), 10_000.0, 600_000.0)
    bdi      = np.clip(1_500.0   + np.cumsum(rng.normal(0, 10.0, n)),  200.0,    10_000.0)

    # M6 — PMI monthly (change every 21 rows)
    def _pmi_walk(center: float, std: float) -> np.ndarray:
        vals = np.empty(n, dtype=float)
        level = center + rng.normal(0, std)
        for i in range(n):
            if i % 21 == 0:
                level = center + rng.normal(0, std)
            vals[i] = level
        return vals

    ism   = _pmi_walk(52.0, 4.0)
    china = _pmi_walk(51.0, 3.0)

    df = pd.DataFrame(
        {
            "VIX": vix, "VIX3M": vix3m, "MOVE": move, "CBOE_SKEW": skew,
            "10Y_UST": ust10, "2Y_UST": ust2, "10Y_BUND": bund,
            "TIPS10Y": tips, "BE10Y": be,
            "HY_OAS": hy, "IG_OAS": ig,
            "DXY": dxy, "EURUSD": eurusd,
            "EIA_CRUDE_STOCK": crude, "EIA_DIST_STOCK": dist_,
            "EIA_GASOLINE_STOCK": gasoline, "EIA_NG_STOCK": ng,
            "LME_COPPER_STOCK": copper, "BAL_DRY_INDEX": bdi,
            "US_ISM_MFG_PMI": ism, "CHINA_PMI_MFG": china,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _synth_panel(kind: str, n: int = 500, seed: int = 42) -> pd.DataFrame:
    if kind == "single_instrument":
        return _synth_single_instrument(n=n, seed=seed)
    if kind == "returns_panel":
        return _synth_returns_panel(n=n, seed=seed)
    if kind == "macro_panel":
        return _synth_macro_panel(n=n, seed=seed)
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
    "drift_features":  lambda df: (df[["return", "vol", "signal"]],),
    # Macro panel passes the entire DataFrame to macro group functions.
    "macro_panel":     lambda df: (df,),
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
    "stml.harry.features.macro_features",
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
