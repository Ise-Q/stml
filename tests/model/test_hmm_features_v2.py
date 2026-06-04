"""Causality + contract tests for the f17_v2 label-aware regime HMM
(:mod:`stml.model.hmm_features_v2`).

Two decisive leakage proofs:

* ``test_filtered_posteriors_are_causal`` — a FIXED frozen bundle transforming the full event
  sequence vs a future-truncated one yields IDENTICAL regime posteriors on the shared rows (a
  smoothed posterior would fail). Mirrors ``tests/test_regimes_hmm.py``.
* ``test_rhr_resolved_before_entry`` — the rolling performance features at event ``i`` use only
  prior events whose triple barrier has RESOLVED by entry (``t1[j] <= date[i]``); flipping an
  unresolved/future outcome (including the event's own) does not change them. Pure pandas, no
  ``hmmlearn`` needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.model.hmm_features_v2 import (
    HAS_HMMLEARN,
    OBS_COLS,
    fit_hmm_v2,
    perf_columns,
    regime_columns,
    rolling_performance_features,
    transform_hmm_v2,
)

requires_hmm = pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn not installed")


def _synth_obs(n: int = 400, seed: int = 0, inst: str = "AAA") -> pd.DataFrame:
    """Synthetic two-regime observation frame: ``[date, instrument, *OBS_COLS, bin]``.

    Alternating ~50-event regimes shift every observation dimension, and ``bin`` is drawn with a
    regime-dependent hit-rate (0.7 in the 'good' regime, 0.3 otherwise) so the fitted states have a
    real hit-rate ordering to recover.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    good = (np.arange(n) // 50) % 2 == 0
    df = pd.DataFrame({"date": idx, "instrument": inst})
    for j, c in enumerate(OBS_COLS):
        shift = np.where(good, 1.0, -1.0) * (0.7 + 0.05 * j)
        df[c] = rng.standard_normal(n) * 0.3 + shift
    df["bin"] = (rng.uniform(size=n) < np.where(good, 0.7, 0.3)).astype(int)
    return df


@pytest.fixture(scope="module")
def bundle_and_obs():
    obs = _synth_obs(n=400, seed=0)
    boundary = obs["date"].iloc[250]
    train = obs[obs["date"] <= boundary]
    bundle = fit_hmm_v2(train, n_states=3, scope="AAA", min_train=150, seed=0)
    return bundle, obs, boundary


@requires_hmm
def test_fit_succeeds_and_freezes_train(bundle_and_obs) -> None:
    bundle, _, boundary = bundle_and_obs
    assert bundle.ok
    assert bundle.n_states == 3
    assert bundle.train_index.max() <= boundary  # frozen on FE-train only
    assert bundle.obs_cols == tuple(OBS_COLS)


@requires_hmm
def test_transform_columns_and_simplex(bundle_and_obs) -> None:
    bundle, obs, _ = bundle_and_obs
    out = transform_hmm_v2(bundle, obs)
    assert list(out.columns) == regime_columns(3)
    post = out[["f17_v2_regime_good", "f17_v2_regime_neutral", "f17_v2_regime_bad"]].dropna()
    assert len(post) > 100
    assert np.allclose(post.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    pred = out[
        ["f17_v2_regime_pred_good", "f17_v2_regime_pred_neutral", "f17_v2_regime_pred_bad"]
    ].dropna()
    assert np.allclose(pred.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert set(out["f17_v2_regime_argmax"].dropna().unique()).issubset({0.0, 1.0, 2.0})
    assert (out["f17_v2_regime_entropy"].dropna() >= -1e-9).all()


@requires_hmm
def test_filtered_posteriors_are_causal(bundle_and_obs) -> None:
    """A FIXED bundle's filtered posterior at event t must not change when future events are
    removed (forward-only filter; a smoothed posterior would leak the future)."""
    bundle, obs, _ = bundle_and_obs
    cut = 350
    full = transform_hmm_v2(bundle, obs)
    trunc = transform_hmm_v2(bundle, obs.iloc[:cut])
    common = obs.index[obs.index < cut - 1]  # drop the last truncated row (its filter is the tip)
    a = full.reindex(common).to_numpy(dtype=float)
    b = trunc.reindex(common).to_numpy(dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    diff = np.abs(a - b)
    fin = diff[np.isfinite(diff)]
    assert fin.size and fin.max() <= 1e-9, "f17_v2 filtered posterior leaked the future"


@requires_hmm
def test_state_order_frozen_on_train_is_descending_hit_rate(bundle_and_obs) -> None:
    bundle, _, _ = bundle_and_obs
    hr = bundle.state_hit_rate
    assert np.all(np.diff(hr) <= 1e-9)  # GOOD (col 0) has the highest FE-train hit-rate
    assert hr[0] > hr[-1]  # the synthetic good/bad gap is real, so the order is non-degenerate


@requires_hmm
def test_failed_fit_yields_structural_nan() -> None:
    obs = _synth_obs(n=80, seed=1)
    bundle = fit_hmm_v2(obs, n_states=3, scope="tiny", min_train=150, seed=0)
    assert not bundle.ok
    out = transform_hmm_v2(bundle, obs)
    assert list(out.columns) == regime_columns(3)
    assert out.isna().all().all()


# --------------------------------------------------------------------------------------------- #
# Rolling-performance causality (no hmmlearn dependency).
# --------------------------------------------------------------------------------------------- #


def _toy_labels(bins: list[int]) -> pd.DataFrame:
    """One instrument, daily events, each resolving 2 bars later (overlap-2)."""
    n = len(bins)
    days = pd.bdate_range("2020-01-01", periods=n + 3)
    return pd.DataFrame({
        "date": days[:n],
        "instrument": "AAA",
        "t1": days[2 : n + 2],  # entry day i resolves on day i+2
        "ret": [0.01 if b else -0.01 for b in bins],
        "bin": bins,
    })


def test_rhr_resolved_before_entry() -> None:
    """RHR/RER use only events resolved by entry (``t1[j] <= date[i]``); no self/future inclusion."""
    bins = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    lab = _toy_labels(bins)
    out = rolling_performance_features(lab, window=5, min_count=2)
    rhr_col = perf_columns(5)[0]

    # Event 5 (entry day 5): resolved priors are j with day j+2 <= day 5  ->  j <= 3 -> {0,1,2,3}.
    expected = np.mean(bins[:4])
    assert out[rhr_col].iloc[5] == pytest.approx(expected)

    # Flipping an outcome NOT yet resolved at entry-5 (event 5 itself, or event 6) must not change it.
    for flip in (5, 6):
        b2 = bins.copy()
        b2[flip] ^= 1
        out2 = rolling_performance_features(_toy_labels(b2), window=5, min_count=2)
        assert out2[rhr_col].iloc[5] == pytest.approx(expected), "future/self outcome leaked"

    # Flipping a RESOLVED prior (event 2) DOES change it — the feature genuinely uses past outcomes.
    b3 = bins.copy()
    b3[2] ^= 1
    out3 = rolling_performance_features(_toy_labels(b3), window=5, min_count=2)
    assert out3[rhr_col].iloc[5] != pytest.approx(expected)


def test_rolling_min_count_is_structural_nan() -> None:
    lab = _toy_labels([1, 0, 1, 1, 0])
    out = rolling_performance_features(lab, window=5, min_count=3)
    rhr_col = perf_columns(5)[0]
    # Events 0,1,2 have <3 resolved priors -> NaN; later events fill in.
    assert out[rhr_col].iloc[0:3].isna().all()
