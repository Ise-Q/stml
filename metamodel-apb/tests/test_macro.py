"""S1.7 PIT-lagged macro block (RED-first).

The macro features (theory-of-storage + cross-asset drivers, §1/§3) are derived from
``additional_data.xlsx``, which carries observation dates only. The non-negotiable property is
**point-in-time correctness**: a macro feature at trade date ``t`` may use only observations
that were *released* on or before ``t``. We test that directly on the low-level ``pit_align``
helper (a multi-day publication lag must defer a release) and on the assembled block via
truncation-invariance (adding future raw data must not change any earlier feature row).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alken_metamodel.macro import load_macro_series, macro_features, pit_align


def test_pit_align_defers_release_by_lag():
    obs = pd.to_datetime(["2021-03-01", "2021-03-08", "2021-03-15"])
    s = pd.Series([10.0, 20.0, 30.0], index=obs)
    trade = pd.bdate_range("2021-03-01", "2021-03-19")
    aligned = pit_align(s, trade, lag_days=5)
    # before the first release (2021-03-01 + 5 = 2021-03-06) the value is unknown
    assert pd.isna(aligned.loc["2021-03-01"])
    # on 2021-03-08 the 03-08 observation is NOT yet released (release 03-13); still the 03-01 obs
    assert aligned.loc["2021-03-08"] == 10.0
    # on 2021-03-15 the 03-08 obs IS released (03-13) but the 03-15 obs (release 03-20) is not
    assert aligned.loc["2021-03-15"] == 20.0


def test_load_macro_series_parses_paired_columns():
    series = load_macro_series()
    for name in ("VIX", "VIX3M", "EIA_CRUDE_STOCK", "CHINA_PMI_MFG", "HY_OAS", "TIPS10Y"):
        assert name in series, name
    vix = series["VIX"]
    assert isinstance(vix.index, pd.DatetimeIndex)
    assert vix.index.is_monotonic_increasing
    assert vix.notna().all()  # dropna'd at load
    assert not vix.index.has_duplicates


def test_macro_features_schema_and_known_signs():
    dates = pd.bdate_range("2021-01-04", "2021-06-30")
    feats = macro_features(dates)
    assert list(feats.index) == list(dates)
    for col in (
        "macro_vix_term_slope",
        "macro_credit_slope",
        "macro_eia_crude_chg",
        "macro_china_pmi",
        "macro_real_rate",
    ):
        assert col in feats.columns, col
    # term slope = VIX3M - VIX and credit slope = HY - IG are exact differences of aligned levels
    s = load_macro_series()
    vix = pit_align(s["VIX"], dates, 1)
    vix3m = pit_align(s["VIX3M"], dates, 1)
    expected_slope = (vix3m - vix).dropna()
    got = feats["macro_vix_term_slope"].reindex(expected_slope.index)
    pd.testing.assert_series_equal(got, expected_slope, check_names=False)


def test_macro_features_are_truncation_invariant():
    # PIT == no-look-ahead: truncating the trade calendar must not change any earlier feature row.
    dates = pd.bdate_range("2021-01-04", "2021-12-31")
    full = macro_features(dates)
    cut = dates[:120]
    partial = macro_features(cut)
    pd.testing.assert_frame_equal(full.loc[cut], partial)


def test_macro_features_inject_custom_series():
    # macro_features accepts an injected series dict (testability) and never peeks past availability
    idx = pd.to_datetime(["2021-02-01", "2021-02-08", "2021-02-15", "2021-02-22"])
    series = {
        "VIX": pd.Series([20.0, 21.0, 22.0, 23.0], index=idx),
        "VIX3M": pd.Series([22.0, 22.0, 22.0, 22.0], index=idx),
        "HY_OAS": pd.Series([4.0, 4.1, 4.2, 4.3], index=idx),
        "IG_OAS": pd.Series([1.0, 1.0, 1.0, 1.0], index=idx),
        "TIPS10Y": pd.Series([0.5, 0.5, 0.5, 0.5], index=idx),
        "EIA_CRUDE_STOCK": pd.Series([100.0, 101.0, 102.0, 103.0], index=idx),
    }
    dates = pd.bdate_range("2021-02-01", "2021-02-26")
    feats = macro_features(dates, series=series)
    assert np.isfinite(feats["macro_credit_slope"].dropna()).all()
    # credit slope on a released date = HY - IG (both lag 1) = 4.x - 1.0 > 0
    assert (feats["macro_credit_slope"].dropna() > 0).all()
