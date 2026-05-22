"""
test_run_characterize.py
========================
LIGHT smoke test for the C1 checkpoint orchestrator
:mod:`stml.replication.run_characterize` (US-006).

Kept fast by running ``main`` for ONE instrument (``cl1s``). The test asserts
that both artifacts are produced and that ``thresholds.json`` is valid JSON
carrying the provenance block with the train-only marker. It does NOT assert
exact numbers -- those come from the already-tested upstream modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stml.io import load_clean_data
from stml.replication.run_characterize import (
    _INSTRUMENTS,
    _calibrate_thresholds,
    _convention_split,
    _pooling_map,
    main,
)

SMOKE_INSTRUMENT = "cl1s"

# Chronological 60/20/20 split boundaries (frozen GROUND TRUTH from CONTRACT.md):
# train[0:387], val[387:516], test[516:645].
_TRAIN_SLICE = slice(0, 387)
_VAL_TEST_SLICE = slice(387, 645)


def test_main_writes_both_artifacts_with_train_only_provenance(tmp_path) -> None:
    # Write under a temp dir so this single-instrument smoke run never clobbers
    # the real, full-universe deliverables in reports/ and results/jj/.
    paths = main(["--instruments", SMOKE_INSTRUMENT, "--out-dir", str(tmp_path)])

    report_path = paths["report"]
    thresholds_path = paths["thresholds"]
    assert isinstance(report_path, Path)
    assert isinstance(thresholds_path, Path)
    # Isolation guarantee: artifacts live under the temp dir, not the repo.
    assert tmp_path in report_path.parents
    assert tmp_path in thresholds_path.parents

    # Both artifacts exist and are non-empty.
    assert report_path.exists()
    assert report_path.stat().st_size > 0
    assert report_path.name == "signal-characterization.md"

    assert thresholds_path.exists()
    assert thresholds_path.stat().st_size > 0
    assert thresholds_path.name == "thresholds.json"

    # The report mentions the instrument analysed.
    report_text = report_path.read_text(encoding="utf-8")
    assert SMOKE_INSTRUMENT in report_text

    # thresholds.json is valid JSON with the train-only provenance block.
    payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    assert "provenance" in payload
    prov = payload["provenance"]
    assert prov["calibration_window"] == "train"
    assert prov["convention"] == "next_day"
    assert "floor" in prov
    assert "train_date_range" in prov and len(prov["train_date_range"]) == 2
    assert "train" in prov["note"].lower()
    assert "no val/test data used" in prov["note"]

    # The calibrated instrument is present with suggested cutoffs.
    assert SMOKE_INSTRUMENT in payload["per_instrument"]
    inst_entry = payload["per_instrument"][SMOKE_INSTRUMENT]
    assert inst_entry["n_train"] > 0
    for key in ("kappa", "mcc", "macro_f1", "ordinal_skill"):
        assert key in inst_entry["suggested_cutoffs"]


# --------------------------------------------------------------------------- #
# M1 invariant tests (pure, deterministic — no artifact I/O needed)           #
# --------------------------------------------------------------------------- #
def test_convention_split_known_forward_peak() -> None:
    """`_convention_split` reads the convention off the FORWARD side.

    A synthetic lead/lag profile with the largest positive correlation at
    ``h=+1`` (next-day) and a negative construction side (``h<0``) must yield
    ``best_forward_lag == 1``, ``next_day_confirmed`` True, and a construction
    reading that reflects the negative trailing relationship.
    """
    lag_profile = {
        -5: -0.10,
        -4: -0.12,
        -3: -0.20,
        -2: -0.30,
        -1: -0.45,
        0: 0.05,
        1: 0.60,  # the forward peak (largest positive |corr|)
        2: 0.15,
        3: 0.08,
        4: 0.02,
        5: 0.01,
    }
    out = _convention_split(lag_profile)

    # Forward (PnL) convention: next-day peak.
    assert out["best_forward_lag"] == 1
    assert out["next_day_confirmed"] is True
    assert out["corr_at_lag1"] == 0.60
    assert out["best_forward_corr"] == 0.60

    # Construction side: the negative trailing relationship (mean-reversion).
    assert out["best_construction_lag"] == -1
    assert out["best_construction_corr"] < 0
    assert out["best_construction_corr"] == -0.45


def test_calibration_is_train_only_isolated() -> None:
    """KEY ANTI-LEAK TEST: val/test data must NOT enter threshold calibration.

    Calibrate on the real signals; then build a perturbed copy where every
    VAL+TEST row (index 387..645) is sign-flipped and shuffled while every TRAIN
    row (0..387) is left byte-for-byte identical. Recalibrate. The resulting
    ``suggested_cutoffs`` and ``persistence_reference`` must be IDENTICAL for
    every instrument and asset class — proving the calibration touches train
    rows only.
    """
    _ohlcv, signals = load_clean_data()
    base = _calibrate_thresholds(signals, _INSTRUMENTS)

    perturbed = signals.copy()
    rng = np.random.default_rng(123)
    signal_cols = [c for c in signals.columns if c != "date"]
    for col in signal_cols:
        values = perturbed[col].to_numpy().copy()
        block = -values[_VAL_TEST_SLICE]  # flip sign on val+test
        rng.shuffle(block)  # and shuffle the block
        values[_VAL_TEST_SLICE] = block
        perturbed[col] = values

    # Guard the construction of the fixture itself: train untouched, val/test moved.
    assert signals.iloc[_TRAIN_SLICE][signal_cols].equals(
        perturbed.iloc[_TRAIN_SLICE][signal_cols]
    )
    assert not signals.iloc[_VAL_TEST_SLICE][signal_cols].equals(
        perturbed.iloc[_VAL_TEST_SLICE][signal_cols]
    )

    after = _calibrate_thresholds(perturbed, _INSTRUMENTS)

    for inst in _INSTRUMENTS:
        base_e = base["per_instrument"][inst]
        after_e = after["per_instrument"][inst]
        assert base_e["suggested_cutoffs"] == after_e["suggested_cutoffs"]
        assert base_e["persistence_reference"] == after_e["persistence_reference"]
    for cls, base_e in base["per_asset_class"].items():
        after_e = after["per_asset_class"][cls]
        assert base_e["suggested_cutoffs"] == after_e["suggested_cutoffs"]
        assert base_e["persistence_reference"] == after_e["persistence_reference"]


def test_pooling_map_floor_decision() -> None:
    """`_pooling_map` floors sub-FLOOR instruments and keeps the rest standalone.

    On the real data ng1s has a post-embargo val ``n_eff`` of ~2 (below FLOOR)
    so its decision must start with ``pool:``; es1s has ``n_eff`` ~35 (well
    above FLOOR) so it must be ``standalone``.
    """
    _ohlcv, signals = load_clean_data()
    pooling = _pooling_map(signals, ["es1s", "ng1s"])

    ng1s = pooling["per_instrument"]["ng1s"]
    assert ng1s["n_eff_post_embargo"] < 10  # FLOOR
    assert ng1s["decision"].startswith("pool:")

    es1s = pooling["per_instrument"]["es1s"]
    assert es1s["n_eff_post_embargo"] >= 10  # FLOOR
    assert es1s["decision"] == "standalone"
