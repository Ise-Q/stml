"""
test_features_provenance.py
===========================
Leakage / provenance tests for :class:`stml.metamodel.pipeline.FeaturePipeline`
(US-FE-007, AC-6b / AC-6c) on REAL data via :func:`stml.io.load_clean_data`.

These prove the integration keystone honours the CONTRACT_FE Section 0 / 3
leakage rules end-to-end:

* **Fit-on-train.** Every per-instrument regime bundle's ``train_index`` and
  every per-class latent bundle's ``train_index`` lie within the FE-train
  partition (dates ``<= 2021-07-01`` for the regime, the FE-train nonzero-signal
  dates for the latent).
* **Frozen-from-train scaler stats (the load-bearing one).** A regime bundle's
  ``gmm_feat_mean`` / ``gmm_feat_std`` equal the TRAIN ``(ret, vol)`` mean/std --
  recomputed directly from the data here -- and are shown to DIFFER from the
  full-series stats, proving the standardization is frozen from train, not
  refit on the whole series.
* **Provenance columns.** The transform carries ``partition`` and a constant
  ``fe_train_end_date == "2021-07-01"`` column (and ``df.attrs``); partition
  labels are consistent with the chronological split (train rows ``<= 2021-07-01``,
  test rows latest).
* **Nonzero-signal restriction.** The matrix holds exactly the nonzero-signal
  trade-days (``f5_signal != 0`` everywhere, and per-instrument row counts match
  the raw nonzero-signal counts).
* **No structural-NaN ffill.** A column with a known warm-up / structural NaN
  still contains NaN in the persisted matrix (it is not forward-filled).

A representative subset (one instrument per asset class: ``es1s`` EQ, ``cl1s``
EN, ``gc1s`` ME) keeps the real-data fit fast while exercising every code path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from stml.io import load_clean_data
from stml.metamodel.pipeline import FeaturePipeline
from stml.na_checks import native_returns, rolling_vol
from stml.replication.splits import chronological_split

FE_TRAIN_END = pd.Timestamp("2021-07-01")
VOL_WINDOW = 20

# One instrument per asset class (EQ / EN / ME) for a fast representative fit.
SUBSET = ["es1s", "cl1s", "gc1s"]


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data():
    """Real clean OHLCV + signals, restricted to the representative subset."""
    ohlcv, signals = load_clean_data()
    ohlcv_sub = ohlcv[ohlcv["instrument"].isin(SUBSET)].copy()
    signals_sub = signals[["date", *SUBSET]].copy()
    return ohlcv_sub, signals_sub


@pytest.fixture(scope="module")
def fitted(data):
    """A FeaturePipeline fit on the subset (warnings silenced for the EM fits)."""
    ohlcv_sub, signals_sub = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = FeaturePipeline().fit(ohlcv_sub, signals_sub)
    return pipe


@pytest.fixture(scope="module")
def matrix(fitted, data):
    """The transformed tidy-long feature matrix on the subset."""
    ohlcv_sub, signals_sub = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitted.transform(ohlcv_sub, signals_sub)


def _ret_vol(ohlcv: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Recompute date-indexed ``(ret, vol)`` directly (the provenance oracle)."""
    inst = ohlcv[ohlcv["instrument"] == instrument]
    rets = native_returns(inst, kind="log")
    ret = rets.set_index("date")["ret"].sort_index()
    vol = rolling_vol(rets, instrument, window=VOL_WINDOW)
    return pd.DataFrame({"ret": ret, "vol": vol}).dropna().sort_index()


# --------------------------------------------------------------------------- #
# AC-6b: fit-on-train provenance (regime + latent train_index ⊆ FE-train)     #
# --------------------------------------------------------------------------- #
def test_every_regime_train_index_within_fe_train(fitted) -> None:
    """Every per-instrument regime bundle was fit on dates <= 2021-07-01."""
    assert set(fitted._regime) == set(SUBSET)
    for inst, bundle in fitted._regime.items():
        assert len(bundle.train_index) > 0, f"{inst}: empty train_index"
        assert bundle.train_index.max() <= FE_TRAIN_END, (
            f"{inst}: regime train_index extends past FE-train"
        )


def test_every_latent_train_index_within_fe_train(fitted, data) -> None:
    """Every per-class latent bundle was fit on FE-train nonzero-signal dates."""
    _, signals_sub = data
    split = chronological_split(signals_sub["date"])
    train_dates = set(pd.DatetimeIndex(split.train_dates))

    assert set(fitted._latent) == {"EQ", "EN", "ME"}
    for asset_class, bundle in fitted._latent.items():
        idx = pd.DatetimeIndex(bundle.train_index)
        assert len(idx) > 0, f"{asset_class}: empty latent train_index"
        # Pooled FE-train rows: every date is a train-partition date.
        assert set(idx) <= train_dates, (
            f"{asset_class}: latent train_index has non-FE-train dates"
        )
        assert idx.max() <= FE_TRAIN_END


# --------------------------------------------------------------------------- #
# AC-6b: scaler-stat provenance -- frozen FROM TRAIN, not the full series      #
# --------------------------------------------------------------------------- #
def test_gmm_scaler_stats_frozen_from_train_not_full(fitted, data) -> None:
    """A regime bundle's ``(ret, vol)`` mean/std equal the TRAIN stats (recomputed
    directly), and DIFFER from the full-series stats -- proving frozen-from-train."""
    ohlcv_sub, _ = data
    inst = "es1s"
    bundle = fitted._regime[inst]
    assert bundle.ok

    rv = _ret_vol(ohlcv_sub, inst)
    train = rv[rv.index <= FE_TRAIN_END]
    assert len(train) < len(rv), "train must be a strict subset of the full series"

    train_mean = train[["ret", "vol"]].to_numpy(dtype=float).mean(axis=0)
    train_std_raw = train[["ret", "vol"]].to_numpy(dtype=float).std(axis=0)
    train_std = np.where(train_std_raw == 0.0, 1.0, train_std_raw)
    full_mean = rv[["ret", "vol"]].to_numpy(dtype=float).mean(axis=0)
    full_std = rv[["ret", "vol"]].to_numpy(dtype=float).std(axis=0)

    # Frozen stats == TRAIN stats (to numerical exactness).
    np.testing.assert_allclose(bundle.gmm_feat_mean, train_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(bundle.gmm_feat_std, train_std, rtol=0, atol=1e-12)

    # ... and NOT the full-series stats (train is a strict subset of the full
    # real series, so both the mean and the std must differ).
    assert not np.allclose(bundle.gmm_feat_mean, full_mean)
    assert not np.allclose(bundle.gmm_feat_std, full_std)


# --------------------------------------------------------------------------- #
# AC-6c: provenance columns + partition consistency                            #
# --------------------------------------------------------------------------- #
def test_matrix_has_partition_and_fe_train_end_columns(matrix) -> None:
    """The tidy-long matrix carries the meta columns in canonical leading order."""
    assert list(matrix.columns[:4]) == [
        "date",
        "instrument",
        "partition",
        "fe_train_end_date",
    ]


def test_fe_train_end_date_constant_and_in_attrs(matrix) -> None:
    """``fe_train_end_date`` is the constant ISO string and is set in attrs."""
    assert (matrix["fe_train_end_date"] == "2021-07-01").all()
    assert matrix.attrs["fe_train_end_date"] == "2021-07-01"


def test_partition_labels_consistent_with_split(matrix) -> None:
    """Partition labels match the chronological split: train rows are dated
    <= 2021-07-01 and the test rows are the latest dates in the matrix."""
    assert set(matrix["partition"].unique()) <= {"train", "val", "test"}

    train_dates = matrix.loc[matrix["partition"] == "train", "date"]
    val_dates = matrix.loc[matrix["partition"] == "val", "date"]
    test_dates = matrix.loc[matrix["partition"] == "test", "date"]

    assert (train_dates <= FE_TRAIN_END).all(), "train rows must be <= FE-train end"
    assert (val_dates > FE_TRAIN_END).all(), "val rows must be after FE-train end"
    # Test is the latest block: every test date is after every train/val date.
    assert test_dates.min() > train_dates.max()
    assert test_dates.min() > val_dates.max()


# --------------------------------------------------------------------------- #
# AC-5: nonzero-signal restriction                                            #
# --------------------------------------------------------------------------- #
def test_matrix_restricted_to_nonzero_signal_days(matrix) -> None:
    """Every row is a participating (nonzero-signal) trade day: f5_signal != 0."""
    sig = matrix["f5_signal"].to_numpy(dtype=float)
    assert np.isfinite(sig).all(), "f5_signal must be finite on every retained row"
    assert (sig != 0).all(), "matrix must hold only nonzero-signal trade-days"


def test_row_counts_match_raw_nonzero_signal_counts(matrix, data) -> None:
    """Per-instrument row counts equal the raw nonzero-signal counts (and the
    pinned CONTRACT_FE §2 ground-truth subset totals)."""
    _, signals_sub = data
    sig_idx = signals_sub.set_index("date")
    expected = {inst: int((sig_idx[inst] != 0).sum()) for inst in SUBSET}

    actual = matrix["instrument"].value_counts().to_dict()
    assert actual == expected, f"row counts {actual} != raw nonzero {expected}"

    # Pinned ground-truth from CONTRACT_FE §2 for this subset.
    assert expected == {"es1s": 575, "cl1s": 422, "gc1s": 168}
    assert len(matrix) == sum(expected.values())


# --------------------------------------------------------------------------- #
# CONTRACT §0.4: no structural-NaN ffill                                       #
# --------------------------------------------------------------------------- #
def test_structural_nan_not_forward_filled(matrix) -> None:
    """A feature with a warm-up / structural NaN still contains NaN in the
    persisted matrix -- it is NOT forward-filled or fillna(0)-ed.

    ``f5_participation_60`` is a trailing 60-day rolling mean over the released
    signal series, so the earliest nonzero-signal rows of each instrument fall
    inside its warm-up and are structurally NaN. A forward-fill would have
    eliminated those leading NaNs, so their survival proves no ffill occurred.
    """
    col = "f5_participation_60"
    assert col in matrix.columns
    assert matrix[col].isna().any(), (
        f"{col} should retain warm-up structural NaNs (not be forward-filled)"
    )

    # Stronger guard: within EVERY instrument the column starts with a run of
    # NaN that later turns finite -- a forward-fill would have removed it.
    n_checked = 0
    for inst in SUBSET:
        sub = (
            matrix.loc[matrix["instrument"] == inst]
            .sort_values("date")[col]
            .to_numpy(dtype=float)
        )
        if np.isnan(sub).any() and np.isfinite(sub).any():
            first_finite = int(np.argmax(np.isfinite(sub)))
            assert first_finite > 0, f"{inst}: expected a leading NaN warm-up run"
            assert np.isnan(sub[:first_finite]).all(), (
                f"{inst}: leading warm-up rows were filled (ffill detected)"
            )
            n_checked += 1
    assert n_checked > 0, f"no instrument had a mix of NaN/finite {col} values"
