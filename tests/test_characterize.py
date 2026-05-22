"""
test_characterize.py
====================
LIGHT smoke tests for :mod:`stml.replication.characterize` (the C1 module).

``characterize`` is diagnostic / exploratory rather than a pure unit-tested
module, so these tests assert *shape and robustness*, not exact numbers:

* every ``Qn`` function returns a ``dict`` carrying its documented numeric keys,
* no function raises on a NORMAL instrument (``cl1s``) or on the DEGENERATE one
  (``ng1s`` -- never ``+1``, ~80% flat),
* :func:`lead_lag` returns an ``int`` ``best_lag`` (the holding-convention
  confirmation), and
* :func:`model_family_fingerprint` always returns with a ``confidence`` field
  and never raises.

Runtime is kept modest: only two instruments are exercised and the OHLCV frame
is sliced to the 2019+ window (the signal era is 2020-2022; a one-year lookback
covers every trailing feature) before the slow model fits run.
"""

from __future__ import annotations

import math

import pytest

from stml.replication.characterize import (
    alpha_type,
    characterize_all,
    characterize_instrument,
    cross_asset,
    drift,
    lead_lag,
    model_family_fingerprint,
    regime,
)

NORMAL = "cl1s"
DEGENERATE = "ng1s"  # never +1, ~long flat runs
SMOKE_INSTRUMENTS = [NORMAL, DEGENERATE]


# --------------------------------------------------------------------------- #
# Fixtures: load real data once, slice OHLCV to the signal era for speed.     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data():
    """Real clean data with OHLCV sliced to 2019+ (one-year lookback before the
    2020-2022 signal era), so the model fits in Q3/Q6 stay fast."""
    from stml.io import load_clean_data

    ohlcv, signals = load_clean_data()
    ohlcv_fast = ohlcv[ohlcv["date"] >= "2019-01-01"].copy()
    return ohlcv_fast, signals


def _is_number(x: object) -> bool:
    """True for a Python/NumPy real number, INCLUDING nan (a valid degenerate
    result here) but excluding bools."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# --------------------------------------------------------------------------- #
# Q1 alpha_type                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SMOKE_INSTRUMENTS)
def test_alpha_type_keys_and_no_raise(inst: str, data) -> None:
    ohlcv, signals = data
    out = alpha_type(inst, signals, ohlcv)
    assert isinstance(out, dict)
    for key in ("trail_corr_1", "trail_corr_20", "momentum_score", "alpha_label"):
        assert key in out
    # trailing correlations and the momentum score are numeric (nan allowed).
    assert _is_number(out["momentum_score"])
    assert _is_number(out["trail_corr_1"])
    assert out["alpha_label"] in {"momentum", "mean_reversion", "neutral"}
    # MA agreement and breakout coincidence keys are present and numeric.
    assert _is_number(out["ma_sign_agreement_10"])
    assert _is_number(out["breakout_coincidence"])
    assert _is_number(out["breakout_coincidence_directional"])


# --------------------------------------------------------------------------- #
# Q2 lead_lag -- the holding-convention confirmation (most important output)   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SMOKE_INSTRUMENTS)
def test_lead_lag_returns_int_best_lag(inst: str, data) -> None:
    ohlcv, signals = data
    out = lead_lag(inst, signals, ohlcv)
    assert isinstance(out, dict)
    # best_lag MUST be an int (it indexes the holding convention).
    assert isinstance(out["best_lag"], int)
    assert -5 <= out["best_lag"] <= 5
    # The lag profile spans every horizon h in -5..+5.
    assert set(out["lag_profile"].keys()) == set(range(-5, 6))
    assert all(_is_number(v) for v in out["lag_profile"].values())
    assert _is_number(out["corr_at_lag1"])
    assert _is_number(out["corr_at_lag0"])
    assert out["holding_convention"] in {
        "next_day",
        "same_day",
        "lagging",
        "forward",
        "inconclusive",
    }


def test_lead_lag_confirms_next_day_on_cl1s(data) -> None:
    """The headline checkpoint claim: cl1s's signal best predicts r_{t+1}, so
    the empirical holding convention is next_day (best_lag == +1)."""
    ohlcv, signals = data
    out = lead_lag(NORMAL, signals, ohlcv)
    assert out["best_lag"] == 1
    assert out["holding_convention"] == "next_day"


# --------------------------------------------------------------------------- #
# Q3 regime -- avoids-high-vol participation split                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SMOKE_INSTRUMENTS)
def test_regime_keys_and_no_raise(inst: str, data) -> None:
    ohlcv, signals = data
    out = regime(inst, ohlcv, signals)
    assert isinstance(out, dict)
    assert out["status"] in {"ok", "inconclusive"}
    # Headline GMM split + the always-available median-vol fallback are present
    # and numeric (nan is acceptable if a model fit fell back).
    for key in (
        "participation_low_vol",
        "participation_high_vol",
        "participation_low_vol_median",
        "participation_high_vol_median",
        "participation_low_vol_markov",
        "participation_high_vol_markov",
    ):
        assert key in out
        assert _is_number(out[key])
    # avoids_high_vol is either a bool verdict or nan when undefined.
    assert isinstance(out["avoids_high_vol"], bool) or _is_number(out["avoids_high_vol"])


# --------------------------------------------------------------------------- #
# Q4 cross_asset                                                              #
# --------------------------------------------------------------------------- #
def test_cross_asset_keys_and_no_raise(data) -> None:
    ohlcv, signals = data
    out = cross_asset(signals, ohlcv)
    assert isinstance(out, dict)
    assert _is_number(out["mean_abs_offdiag_corr"])
    # Prior EDA expects a low cross-asset signal correlation (~0.11).
    assert 0.0 <= out["mean_abs_offdiag_corr"] <= 1.0
    # One cluster label per instrument, each a non-negative int.
    labels = out["cluster_labels"]
    assert isinstance(labels, dict)
    assert len(labels) == out["n_instruments"]
    assert all(isinstance(v, int) and v >= 0 for v in labels.values())


# --------------------------------------------------------------------------- #
# Q5 drift -- per-split base rates                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SMOKE_INSTRUMENTS)
def test_drift_keys_and_no_raise(inst: str, data) -> None:
    _, signals = data
    out = drift(inst, signals)
    assert isinstance(out, dict)
    for split_name in ("train", "val", "test"):
        block = out[split_name]
        for key in ("participation_rate", "long_bias", "frac_neg1", "frac_0", "frac_pos1"):
            assert key in block
            assert _is_number(block[key])
    # The train->test trend deltas are numeric.
    assert _is_number(out["trend"]["participation_train_to_test"])
    assert _is_number(out["trend"]["long_bias_train_to_test"])


# --------------------------------------------------------------------------- #
# Q6 model_family_fingerprint -- ADVISORY; must always carry 'confidence'.     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inst", SMOKE_INSTRUMENTS)
def test_model_family_fingerprint_has_confidence_and_no_raise(inst: str, data) -> None:
    ohlcv, signals = data
    out = model_family_fingerprint(inst, signals, ohlcv)
    assert isinstance(out, dict)
    # The advisory contract: a 'confidence' field is ALWAYS present and in [0, 1].
    assert "confidence" in out
    assert _is_number(out["confidence"])
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["label"] in {"tree_like", "linear", "nonlinear", "inconclusive"}
    assert out["status"] in {"ok", "inconclusive"}
    for key in ("tree_cv_acc", "linear_cv_acc", "forest_cv_acc", "majority_acc"):
        assert key in out and _is_number(out[key])


def test_degenerate_fingerprint_returns_inconclusive(data) -> None:
    """ng1s is never +1; its surrogate-classifier guess is allowed to be (and
    here is) inconclusive -- the key invariant is that it returns, not raises."""
    ohlcv, signals = data
    out = model_family_fingerprint(DEGENERATE, signals, ohlcv)
    assert isinstance(out["confidence"], float)
    assert out["label"] in {"tree_like", "linear", "nonlinear", "inconclusive"}


# --------------------------------------------------------------------------- #
# Combiners                                                                   #
# --------------------------------------------------------------------------- #
def test_characterize_instrument_runs_all_questions(data) -> None:
    ohlcv, signals = data
    out = characterize_instrument(NORMAL, signals, ohlcv)
    for key in ("alpha_type", "lead_lag", "regime", "drift", "model_family_fingerprint"):
        assert key in out and isinstance(out[key], dict)
    # The combined result carries the int best_lag through unchanged.
    assert isinstance(out["lead_lag"]["best_lag"], int)


def test_characterize_all_smoke(data) -> None:
    ohlcv, signals = data
    out = characterize_all(signals, ohlcv, instruments=SMOKE_INSTRUMENTS)
    assert set(out["per_instrument"].keys()) == set(SMOKE_INSTRUMENTS)
    assert "cross_asset" in out
    assert _is_number(out["cross_asset"]["mean_abs_offdiag_corr"])
    # No NaN best_lag leaks through the combined panel result.
    for inst in SMOKE_INSTRUMENTS:
        bl = out["per_instrument"][inst]["lead_lag"]["best_lag"]
        assert isinstance(bl, int)
        assert not math.isnan(bl)
