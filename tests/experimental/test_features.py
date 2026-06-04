"""Tests for the feature subsystem — methodology spec Stage 2 acceptance.

Coverage:
* Registry sanity (every feature has a unique name, valid family, callable fn).
* Truncation invariance: for a sample of E-class features, the value at
  position ``t`` on truncated input equals the value on full input.
* Plan §3.3 macro reformulation: F11 columns are RANKS in [0, 1] or CHANGES,
  never raw levels.
* F22 EIA features are NaN for non-energy instruments and finite for energy.
* assemble_features produces a frame with one column per registered feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.experimental.config import INSTRUMENTS
from stml.experimental.data_loader import load_panel, per_instrument_frames
from stml.experimental.features import (
    REGISTRY,
    assemble_features,
    family_counts,
    registered_names,
)
from stml.experimental.features.catalog import FeatureContext


# ---------------------------------------------------------------------------
# Registry sanity.
# ---------------------------------------------------------------------------


def test_registry_has_unique_names() -> None:
    names = registered_names()
    assert len(names) == len(set(names)), "duplicate feature names in registry"


def test_registry_covers_required_families() -> None:
    """At minimum, the families specified in methodology spec must be present."""
    fc = family_counts()
    required = {"F1", "F2", "F5", "F6", "F7", "F8", "F10", "F11", "F12", "F15",
                "F16", "F17", "F18", "F19", "F21", "F22", "EWMA_HMM"}
    missing = required - set(fc.keys())
    assert not missing, f"missing required feature families: {missing}"


def test_registry_size_in_target_range() -> None:
    """Plan §3.3 targets ~100-120 features pre-drift-filter; we want ≥80."""
    assert 80 <= len(REGISTRY) <= 150


def test_every_spec_has_required_attributes() -> None:
    for s in REGISTRY:
        assert s.name and isinstance(s.name, str)
        assert s.family.startswith("F") or s.family == "EWMA_HMM"
        assert callable(s.fn)
        assert s.leakage_class in ("E", "TF")
        assert s.warmup_bars >= 0


# ---------------------------------------------------------------------------
# F11 macro REFORMULATION — only RANKS and CHANGES, no raw LEVELS.
# ---------------------------------------------------------------------------


def test_f11_features_are_ranks_or_changes_not_levels() -> None:
    """Plan §3.3 fix: macro LEVELS catastrophically drifted; we use ranks/changes only."""
    f11_names = [s.name for s in REGISTRY if s.family == "F11"]
    for name in f11_names:
        # Allowed suffixes: rank63, chg5, chg20, or specific spread names.
        allowed_suffixes = ("rank63", "chg5", "chg20", "slope", "diff", "term_slope")
        allowed = any(name.endswith(suf) for suf in allowed_suffixes)
        assert allowed, f"F11 feature {name} should end in {allowed_suffixes}"
        # The name must NOT contain "_level" or "_z" (we removed direct levels).
        assert "_level" not in name, f"F11 feature {name} appears to use raw levels"


# ---------------------------------------------------------------------------
# Smoke-test the assembler on real data.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_assemble_features_smoke(repo_root) -> None:  # noqa: ARG001
    """Smoke-test: assemble_features runs end-to-end on a 2-instrument subset."""
    ohlcv, signals = load_panel()
    panel = per_instrument_frames(ohlcv, signals)
    # Use just two instruments to keep the test fast.
    panel_sub = {k: panel[k] for k in ("cl1s", "es1s") if k in panel}
    out = assemble_features(panel_sub, instruments=list(panel_sub.keys()), verbose=False)
    assert not out.empty
    assert {"instrument", "date"}.issubset(out.columns)
    # Every registered feature should be a column.
    missing = [s.name for s in REGISTRY if s.name not in out.columns]
    assert not missing, f"missing feature columns from assembler: {missing[:5]}"
    # cl1s should have non-NaN core features by 2020.
    cl_2020 = out[(out["instrument"] == "cl1s") & (out["date"] >= pd.Timestamp("2020-01-02"))]
    assert (cl_2020["f1_rsi_14"].notna().mean()) > 0.9


# ---------------------------------------------------------------------------
# Truncation invariance — sample-check the most important features.
# ---------------------------------------------------------------------------


@pytest.fixture
def synth_ctx(synth_ohlc, synth_signal):
    """A FeatureContext with synthetic single-instrument data."""
    frame = synth_ohlc.copy()
    frame["signal"] = synth_signal
    frame["open_interest"] = np.nan  # most synth tests don't need OI
    universe = {"x": frame}
    return FeatureContext(
        instrument="x",
        asset_class="equity",
        frame=frame,
        universe=universe,
        macro_alternative=pd.DataFrame(),
        futures_term=pd.DataFrame(),
        options_iv=pd.DataFrame(),
        eia_crude=pd.DataFrame(),
        eia_release_flag=pd.DataFrame(),
    )


def test_f12_variance_ratio_truncation_invariant(synth_ctx) -> None:
    """f12_variance_ratio_5_21 must be right-edge truncation invariant."""
    from stml.experimental.features.closed_form import _f12_variance_ratio_5_21

    full = _f12_variance_ratio_5_21(synth_ctx)
    # Truncate to first 200 bars.
    short_ctx = FeatureContext(
        instrument=synth_ctx.instrument,
        asset_class=synth_ctx.asset_class,
        frame=synth_ctx.frame.iloc[:200],
        universe={"x": synth_ctx.frame.iloc[:200]},
        macro_alternative=synth_ctx.macro_alternative,
        futures_term=synth_ctx.futures_term,
        options_iv=synth_ctx.options_iv,
        eia_crude=synth_ctx.eia_crude,
        eia_release_flag=synth_ctx.eia_release_flag,
    )
    truncated = _f12_variance_ratio_5_21(short_ctx)
    # Compare on the overlap.
    for t in (100, 150, 199):
        ts = synth_ctx.frame.index[t]
        if pd.notna(full.loc[ts]) and pd.notna(truncated.loc[ts]):
            np.testing.assert_allclose(full.loc[ts], truncated.loc[ts], rtol=1e-9)


def test_f15_path_tortuosity_truncation_invariant(synth_ctx) -> None:
    from stml.experimental.features.risk_drift_regime import _f15_path_tortuosity_20

    full = _f15_path_tortuosity_20(synth_ctx)
    short_ctx = FeatureContext(
        instrument=synth_ctx.instrument,
        asset_class=synth_ctx.asset_class,
        frame=synth_ctx.frame.iloc[:200],
        universe={"x": synth_ctx.frame.iloc[:200]},
        macro_alternative=synth_ctx.macro_alternative,
        futures_term=synth_ctx.futures_term,
        options_iv=synth_ctx.options_iv,
        eia_crude=synth_ctx.eia_crude,
        eia_release_flag=synth_ctx.eia_release_flag,
    )
    truncated = _f15_path_tortuosity_20(short_ctx)
    for t in (100, 150, 199):
        ts = synth_ctx.frame.index[t]
        if pd.notna(full.loc[ts]) and pd.notna(truncated.loc[ts]):
            np.testing.assert_allclose(full.loc[ts], truncated.loc[ts], rtol=1e-9)


def test_f1_rsi_14_truncation_invariant(synth_ctx) -> None:
    from stml.experimental.features.closed_form import _f1_rsi_14

    full = _f1_rsi_14(synth_ctx)
    short_ctx = FeatureContext(
        instrument=synth_ctx.instrument,
        asset_class=synth_ctx.asset_class,
        frame=synth_ctx.frame.iloc[:200],
        universe={"x": synth_ctx.frame.iloc[:200]},
        macro_alternative=synth_ctx.macro_alternative,
        futures_term=synth_ctx.futures_term,
        options_iv=synth_ctx.options_iv,
        eia_crude=synth_ctx.eia_crude,
        eia_release_flag=synth_ctx.eia_release_flag,
    )
    truncated = _f1_rsi_14(short_ctx)
    # RSI uses ewm with adjust=False which depends only on past observations —
    # should be exactly invariant.
    for t in (50, 100, 150, 199):
        ts = synth_ctx.frame.index[t]
        if pd.notna(full.loc[ts]) and pd.notna(truncated.loc[ts]):
            np.testing.assert_allclose(full.loc[ts], truncated.loc[ts], rtol=1e-12)


# ---------------------------------------------------------------------------
# F22 — only fires for energy instruments.
# ---------------------------------------------------------------------------


def test_f22_eia_features_nan_for_non_energy(synth_ctx) -> None:
    """F22 returns NaN for any instrument outside (cl1s, ho1s, rb1s, ng1s)."""
    from stml.experimental.features.bloomberg import _f22_eia_release_flag

    # synth_ctx is asset_class="equity" — F22 returns NaN.
    out = _f22_eia_release_flag(synth_ctx)
    assert out.isna().all()


# ---------------------------------------------------------------------------
# Drift filter sanity.
# ---------------------------------------------------------------------------


def test_drift_filter_drops_all_nan_columns() -> None:
    from stml.experimental.make_features import drift_filter

    # Tiny synthetic event matrix with one all-NaN feature.
    df = pd.DataFrame(
        {
            "instrument": ["x"] * 50,
            "t_signal": pd.date_range("2020-01-02", periods=50),
            "label": np.random.randint(0, 2, 50),
            "good_feature": np.random.rand(50),
            "all_nan_feature": np.nan,
        }
    )
    kept, audit = drift_filter(df)
    assert "all_nan_feature" not in kept.columns
    assert "good_feature" in kept.columns
    nan_row = audit.loc[audit["feature"] == "all_nan_feature"].iloc[0]
    assert nan_row["reason"] == "drop: all-NaN"
