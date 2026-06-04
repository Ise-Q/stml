"""Smoke test for ``stml.experimental.pipeline.run_asset_class``.

Doesn't run a full asset-class horse-race (too slow); instead verifies the
plumbing on a small synthetic class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.experimental.config import PipelineConfig
from stml.experimental.pipeline import (
    _SCHEMA_COLS,
    _is_bbg_feature,
    _select_feature_cols,
    _slice_class,
)


def test_is_bbg_feature_detects_prefixes() -> None:
    assert _is_bbg_feature("f18_term_spread")
    assert _is_bbg_feature("f19_atm_iv_1m")
    assert _is_bbg_feature("f22_eia_release_flag")
    assert not _is_bbg_feature("f1_rsi_14")
    assert not _is_bbg_feature("f21_gold_silver_ratio")  # F21 is OHLCV-derived


def test_select_feature_cols_drops_bbg_for_without_bbg() -> None:
    features = pd.DataFrame(
        columns=[
            *_SCHEMA_COLS,
            "f1_rsi_14", "f2_vol_20", "f18_term_spread", "f19_atm_iv_1m",
            "f22_eia_release_flag", "f21_gold_silver_ratio",
        ]
    )
    with_bbg = _select_feature_cols(features, "with_bbg")
    without_bbg = _select_feature_cols(features, "without_bbg")
    sim_miss = _select_feature_cols(features, "simulated_missingness")
    # with_bbg has all feature cols.
    assert "f18_term_spread" in with_bbg
    assert "f19_atm_iv_1m" in with_bbg
    # without_bbg drops F18 / F19 / F22.
    assert "f18_term_spread" not in without_bbg
    assert "f19_atm_iv_1m" not in without_bbg
    assert "f22_eia_release_flag" not in without_bbg
    # But keeps F1 / F2 / F21 (F21 is OHLCV-derived).
    assert "f1_rsi_14" in without_bbg
    assert "f21_gold_silver_ratio" in without_bbg
    # sim_miss matches with_bbg (NaN forcing happens at predict, not column selection).
    assert sim_miss == with_bbg


def test_slice_class_keeps_only_class_members() -> None:
    df = pd.DataFrame({
        "instrument": ["es1s", "nq1s", "fesx1s", "cl1s", "gc1s"],
        "label": [1, 0, 1, 0, 1],
    })
    eq = _slice_class(df, "equity")
    assert sorted(eq["instrument"].unique()) == ["es1s", "fesx1s", "nq1s"]
    en = _slice_class(df, "energy")
    assert list(en["instrument"]) == ["cl1s"]
    me = _slice_class(df, "metals")
    assert list(me["instrument"]) == ["gc1s"]
