"""
v4.py — Stacked Conditional Ensemble (the ambitious build)
==========================================================

Architecture::

    LEVEL 0 BASE MODELS (all trained on commodity events only):
      M1 : XGBoost, h=10 labels, full features (G1-G8)
      M2 : XGBoost, h=10 labels, recency-weighted (decay 0.3/yr)
      M3 : XGBoost, h=5 labels (shorter horizon, more events)
      M4 : XGBoost, h=15 labels (longer horizon, fewer events)
      M5 : ElasticNet LogReg, h=10
      M6 : Random Forest, h=10 (depth-limited, bagging-only — different bias)
      M7 : XGBoost trained ONLY on long (side=+1) bets, h=10
      M8 : XGBoost trained ONLY on short (side=-1) bets, h=10

    LEVEL 1 STACK META-LEARNER:
      LogReg trained on OUT-OF-FOLD predictions from the level-0 models
      paired with true labels. Stack predictions are then per-instrument
      isotonic-calibrated for the final output.

Economic rationale per base model:
  - M1: the workhorse — gradient-boosted trees on the full feature set.
  - M2: recency tilt — recent training events better represent the test regime.
  - M3-M4: multi-horizon — different h values capture different bet styles
    (short = scalping, long = position-trading); ensemble benefits from
    diversification across label semantics.
  - M5: linear baseline — interpretable, regularises against tree overfit.
  - M6: bagging-only ensemble (RF) — different bias profile than boosting.
  - M7-M8: side-specialised — long and short bets have different
    micro-structure (drift, asymmetric tails, vol-of-vol differs).

Stacking trains the meta-learner on OOF base predictions, so no leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from stml.cv import PurgedKFold, split_by_boundary
from stml.experiments import ASSET_CLASSES
from stml.features import compute_features
from stml.io import load_clean_data
from stml.labeling import (
    extract_signal_events,
    get_meta_labels,
    get_uniqueness_weights,
)
from stml.models import ElasticNetLogReg, XGBoostMeta
from stml.regimes import compute_regime_features


# --------------------------------------------------------------------------- #
@dataclass
class _Data:
    ohlcv: pd.DataFrame
    signals: pd.DataFrame
    events_all: pd.DataFrame
    X_all: pd.DataFrame
    # Per-h dictionaries:
    events_lab_by_h: dict[int, pd.DataFrame]   # h -> events_lab (labelable subset)
    X_lab_by_h: dict[int, pd.DataFrame]
    y_lab_by_h: dict[int, pd.Series]
    t_lab_by_h: dict[int, pd.Series]
    t1_lab_by_h: dict[int, pd.Series]
    w_lab_by_h: dict[int, pd.Series]
    side_lab_by_h: dict[int, pd.Series]
    tr_pos_by_h: dict[int, np.ndarray]
    boundary: pd.Timestamp
    predict_end: pd.Timestamp
    embargo: pd.Timedelta


def _build_v4_data(
    boundary: pd.Timestamp = pd.Timestamp("2022-01-01"),
    predict_end: pd.Timestamp = pd.Timestamp("2022-07-01"),
    horizons: tuple[int, ...] = (5, 10, 15),
    embargo_days: int = 10,
    feature_groups: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G7", "G8"),
    verbose: bool = True,
) -> _Data:
    if verbose:
        print(f"[v4] loading data, boundary={boundary.date()}")
    ohlcv, signals = load_clean_data()
    events_all = extract_signal_events(signals).reset_index(drop=True)
    if verbose:
        print(f"[v4] computing features (incl. G8 cross-sectional)...")
    feats = compute_features(ohlcv, events_all, signals, include_groups=feature_groups)
    regs = compute_regime_features(ohlcv, events_all, boundary=boundary)
    X_all = feats.join(regs, how="left").fillna(0.0)
    if verbose:
        print(f"[v4] feature matrix: {X_all.shape}")

    key = ["t", "instrument"]
    embargo = pd.Timedelta(days=embargo_days)

    events_lab_by_h: dict[int, pd.DataFrame] = {}
    X_lab_by_h: dict[int, pd.DataFrame] = {}
    y_lab_by_h: dict[int, pd.Series] = {}
    t_lab_by_h: dict[int, pd.Series] = {}
    t1_lab_by_h: dict[int, pd.Series] = {}
    w_lab_by_h: dict[int, pd.Series] = {}
    side_lab_by_h: dict[int, pd.Series] = {}
    tr_pos_by_h: dict[int, np.ndarray] = {}

    for h in horizons:
        if verbose:
            print(f"[v4] labeling at h={h}...")
        labels = get_meta_labels(ohlcv, signals, h=h, pt_mult=1.0, sl_mult=1.0, verbose=False)
        events_h = events_all.copy()
        events_h["label"] = labels.set_index(key)["label"].reindex(events_h.set_index(key).index).reset_index(drop=True).values
        events_h["t1_orig"] = labels.set_index(key)["t1"].reindex(events_h.set_index(key).index).reset_index(drop=True).values
        events_h["ret"] = labels.set_index(key)["ret"].reindex(events_h.set_index(key).index).reset_index(drop=True).values
        weights = get_uniqueness_weights(labels)
        w_aligned = pd.Series(weights.values, index=labels.set_index(key).index).reindex(events_h.set_index(key).index).reset_index(drop=True)

        mask = ~events_h["label"].isna()
        events_lab = events_h.loc[mask].reset_index(drop=True)
        X_lab = X_all.loc[mask].reset_index(drop=True)

        events_lab_by_h[h] = events_lab
        X_lab_by_h[h] = X_lab
        y_lab_by_h[h] = events_lab["label"].astype(int)
        t_lab_by_h[h] = events_lab["t"]
        t1_lab_by_h[h] = events_lab["t1_orig"]
        side_lab_by_h[h] = events_lab["side"]
        w_lab_by_h[h] = w_aligned.loc[mask].reset_index(drop=True)

        tr_pos, _ = split_by_boundary(events_lab["t"], boundary, embargo_td=embargo)
        tr_pos_by_h[h] = tr_pos

    return _Data(
        ohlcv=ohlcv, signals=signals, events_all=events_all, X_all=X_all,
        events_lab_by_h=events_lab_by_h, X_lab_by_h=X_lab_by_h,
        y_lab_by_h=y_lab_by_h, t_lab_by_h=t_lab_by_h,
        t1_lab_by_h=t1_lab_by_h, w_lab_by_h=w_lab_by_h,
        side_lab_by_h=side_lab_by_h, tr_pos_by_h=tr_pos_by_h,
        boundary=boundary, predict_end=predict_end, embargo=embargo,
    )


def _commodity_filter(events_lab: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    """Return positions whose instrument is non-equity."""
    return np.array([
        i for i in positions
        if ASSET_CLASSES.get(events_lab.iloc[i]["instrument"]) != "equity"
    ])


def _recency_weights(t: pd.Series, boundary: pd.Timestamp, base_w: pd.Series,
                      decay: float) -> pd.Series:
    boundary_ts = pd.to_datetime(boundary)
    days = np.array([(boundary_ts - pd.to_datetime(d)).days for d in t.values], dtype=float)
    years = days / 365.25
    rec = (decay ** years) * base_w.values
    rec = rec / rec.mean()
    return pd.Series(rec, index=t.index)


# --------------------------------------------------------------------------- #
# OOF base-model predictions for stacking                                     #
# --------------------------------------------------------------------------- #
def _oof_predict(
    base_factory,  # callable() -> fresh untrained model
    X: pd.DataFrame, y: pd.Series, t: pd.Series, t1: pd.Series, w: pd.Series,
    embargo: pd.Timedelta, n_splits: int = 5,
    fit_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Out-of-fold predictions via purged K-fold. No leakage.

    Trains a fresh model on each fold's training portion, predicts on the
    held-out fold, accumulates predictions across folds.
    """
    if fit_kwargs is None:
        fit_kwargs = {}
    cv = PurgedKFold(n_splits=n_splits, t=t, t1=t1, embargo_td=embargo)
    oof = np.full(len(X), np.nan)
    for tr_idx, te_idx in cv.split(X):
        m = base_factory()
        # Sub-split the training fold further inside the model's internal tuning;
        # to avoid that, set internal n_splits low (≥2). We pass through fit_kwargs.
        m.fit(
            X.iloc[tr_idx],
            pd.Series(y.values[tr_idx], index=X.iloc[tr_idx].index),
            t=pd.Series(t.values[tr_idx], index=X.iloc[tr_idx].index),
            t1=pd.Series(t1.values[tr_idx], index=X.iloc[tr_idx].index),
            sample_weight=pd.Series(w.values[tr_idx], index=X.iloc[tr_idx].index),
            **fit_kwargs,
        )
        oof[te_idx] = m.predict_proba(X.iloc[te_idx])
    return oof


# --------------------------------------------------------------------------- #
# Sklearn-compatible Random Forest wrapper to match our model interface       #
# --------------------------------------------------------------------------- #
class _RFMeta:
    """Lightweight RF wrapper with the same fit/predict_proba interface."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 5,
                 random_state: int = 42, **kwargs):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           min_samples_split=10, min_samples_leaf=5,
                           max_features="sqrt", class_weight="balanced",
                           random_state=random_state, n_jobs=-1)
        self.best_params_ = self.params
        self.model_ = None

    def fit(self, X, y, t=None, t1=None, sample_weight=None):  # noqa: ARG002
        from sklearn.ensemble import RandomForestClassifier as _RF
        self.model_ = _RF(**self.params)
        sw = sample_weight.values if sample_weight is not None else None
        self.model_.fit(X.values, y.values, sample_weight=sw)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X.values)[:, 1]


# --------------------------------------------------------------------------- #
# Train all level-0 base models on commodity events                           #
# --------------------------------------------------------------------------- #
def _train_base_models(
    data: _Data,
    verbose: bool = True,
) -> dict:
    """Returns a dict ``{model_name: (model, train_positions, h)}``.

    All models are trained on commodity events only. Side-specialised ones
    additionally filter on side.
    """
    out = {}
    embargo = data.embargo
    boundary = data.boundary

    # M1: XGB h=10 full features
    h = 10
    tr_pos = _commodity_filter(data.events_lab_by_h[h], data.tr_pos_by_h[h])
    X = data.X_lab_by_h[h].iloc[tr_pos]
    y = pd.Series(data.y_lab_by_h[h].values[tr_pos], index=X.index)
    t = pd.Series(data.t_lab_by_h[h].values[tr_pos], index=X.index)
    t1 = pd.Series(data.t1_lab_by_h[h].values[tr_pos], index=X.index)
    w = pd.Series(data.w_lab_by_h[h].values[tr_pos], index=X.index)
    if verbose: print(f"  M1: XGB h=10  n={len(X)}")
    m1 = XGBoostMeta(n_iter=15, embargo_td=embargo, random_state=42)
    m1.fit(X, y, t=t, t1=t1, sample_weight=w)
    out["M1_xgb_h10"] = {"model": m1, "tr_pos": tr_pos, "h": h, "X": X, "y": y, "t": t, "t1": t1, "w": w}

    # M2: XGB h=10 recency-weighted (decay 0.3)
    w_rec = _recency_weights(t, boundary, w, decay=0.3)
    if verbose: print(f"  M2: XGB h=10 recency-weighted")
    m2 = XGBoostMeta(n_iter=15, embargo_td=embargo, random_state=43)
    m2.fit(X, y, t=t, t1=t1, sample_weight=w_rec)
    out["M2_xgb_recency"] = {"model": m2, "tr_pos": tr_pos, "h": h, "X": X, "y": y, "t": t, "t1": t1, "w": w_rec}

    # M3: XGB h=5
    h = 5
    tr_pos = _commodity_filter(data.events_lab_by_h[h], data.tr_pos_by_h[h])
    X3 = data.X_lab_by_h[h].iloc[tr_pos]
    y3 = pd.Series(data.y_lab_by_h[h].values[tr_pos], index=X3.index)
    t3 = pd.Series(data.t_lab_by_h[h].values[tr_pos], index=X3.index)
    t13 = pd.Series(data.t1_lab_by_h[h].values[tr_pos], index=X3.index)
    w3 = pd.Series(data.w_lab_by_h[h].values[tr_pos], index=X3.index)
    if verbose: print(f"  M3: XGB h=5   n={len(X3)}")
    m3 = XGBoostMeta(n_iter=15, embargo_td=embargo, random_state=44)
    m3.fit(X3, y3, t=t3, t1=t13, sample_weight=w3)
    out["M3_xgb_h5"] = {"model": m3, "tr_pos": tr_pos, "h": h, "X": X3, "y": y3, "t": t3, "t1": t13, "w": w3}

    # M4: XGB h=15
    h = 15
    tr_pos = _commodity_filter(data.events_lab_by_h[h], data.tr_pos_by_h[h])
    X4 = data.X_lab_by_h[h].iloc[tr_pos]
    y4 = pd.Series(data.y_lab_by_h[h].values[tr_pos], index=X4.index)
    t4 = pd.Series(data.t_lab_by_h[h].values[tr_pos], index=X4.index)
    t14 = pd.Series(data.t1_lab_by_h[h].values[tr_pos], index=X4.index)
    w4 = pd.Series(data.w_lab_by_h[h].values[tr_pos], index=X4.index)
    if verbose: print(f"  M4: XGB h=15  n={len(X4)}")
    m4 = XGBoostMeta(n_iter=15, embargo_td=embargo, random_state=45)
    m4.fit(X4, y4, t=t4, t1=t14, sample_weight=w4)
    out["M4_xgb_h15"] = {"model": m4, "tr_pos": tr_pos, "h": h, "X": X4, "y": y4, "t": t4, "t1": t14, "w": w4}

    # M5: ElasticNet LogReg h=10
    h = 10
    tr_pos = _commodity_filter(data.events_lab_by_h[h], data.tr_pos_by_h[h])
    X5 = data.X_lab_by_h[h].iloc[tr_pos]
    y5 = pd.Series(data.y_lab_by_h[h].values[tr_pos], index=X5.index)
    t5 = pd.Series(data.t_lab_by_h[h].values[tr_pos], index=X5.index)
    t15 = pd.Series(data.t1_lab_by_h[h].values[tr_pos], index=X5.index)
    w5 = pd.Series(data.w_lab_by_h[h].values[tr_pos], index=X5.index)
    if verbose: print(f"  M5: LogReg h=10")
    m5 = ElasticNetLogReg(n_iter=15, embargo_td=embargo)
    m5.fit(X5, y5, t=t5, t1=t15, sample_weight=w5)
    out["M5_lr_h10"] = {"model": m5, "tr_pos": tr_pos, "h": h, "X": X5, "y": y5, "t": t5, "t1": t15, "w": w5}

    # M6: Random Forest h=10 (different bias profile)
    if verbose: print(f"  M6: Random Forest h=10")
    m6 = _RFMeta(n_estimators=300, max_depth=5, random_state=42)
    m6.fit(X5, y5, t=t5, t1=t15, sample_weight=w5)
    out["M6_rf_h10"] = {"model": m6, "tr_pos": tr_pos, "h": h, "X": X5, "y": y5, "t": t5, "t1": t15, "w": w5}

    # M7: XGB on long-only commodity bets
    long_mask = (data.side_lab_by_h[10].values == 1)
    tr_long = np.array([i for i in tr_pos if long_mask[i]])
    if len(tr_long) >= 200:
        X7 = data.X_lab_by_h[10].iloc[tr_long]
        y7 = pd.Series(data.y_lab_by_h[10].values[tr_long], index=X7.index)
        t7 = pd.Series(data.t_lab_by_h[10].values[tr_long], index=X7.index)
        t17 = pd.Series(data.t1_lab_by_h[10].values[tr_long], index=X7.index)
        w7 = pd.Series(data.w_lab_by_h[10].values[tr_long], index=X7.index)
        if verbose: print(f"  M7: XGB long-only n={len(X7)}")
        m7 = XGBoostMeta(n_iter=10, embargo_td=embargo, random_state=46,
                         n_splits_inner=3)
        m7.fit(X7, y7, t=t7, t1=t17, sample_weight=w7)
        out["M7_xgb_long"] = {"model": m7, "tr_pos": tr_long, "h": 10,
                              "X": X7, "y": y7, "t": t7, "t1": t17, "w": w7}

    # M8: XGB on short-only commodity bets
    short_mask = (data.side_lab_by_h[10].values == -1)
    tr_short = np.array([i for i in tr_pos if short_mask[i]])
    if len(tr_short) >= 200:
        X8 = data.X_lab_by_h[10].iloc[tr_short]
        y8 = pd.Series(data.y_lab_by_h[10].values[tr_short], index=X8.index)
        t8 = pd.Series(data.t_lab_by_h[10].values[tr_short], index=X8.index)
        t18 = pd.Series(data.t1_lab_by_h[10].values[tr_short], index=X8.index)
        w8 = pd.Series(data.w_lab_by_h[10].values[tr_short], index=X8.index)
        if verbose: print(f"  M8: XGB short-only n={len(X8)}")
        m8 = XGBoostMeta(n_iter=10, embargo_td=embargo, random_state=47,
                         n_splits_inner=3)
        m8.fit(X8, y8, t=t8, t1=t18, sample_weight=w8)
        out["M8_xgb_short"] = {"model": m8, "tr_pos": tr_short, "h": 10,
                                "X": X8, "y": y8, "t": t8, "t1": t18, "w": w8}

    return out


def _generate_oof_predictions(
    data: _Data, base_results: dict, verbose: bool = True,
) -> pd.DataFrame:
    """For each base model, generate purged-CV OOF predictions on its
    own commodity-training set. Returns a DataFrame with one column per model
    aligned to a *unified* index (h=10 commodity training events)."""
    embargo = data.embargo
    # Use h=10 commodity training events as the canonical index for stacking.
    h = 10
    tr_pos_main = _commodity_filter(data.events_lab_by_h[h], data.tr_pos_by_h[h])
    canonical_events = data.events_lab_by_h[h].iloc[tr_pos_main].reset_index(drop=True)
    # Map (t, instrument) to canonical position
    canonical_key = pd.MultiIndex.from_arrays(
        [canonical_events["t"], canonical_events["instrument"]],
        names=["t", "instrument"],
    )
    oof_df = pd.DataFrame(index=range(len(canonical_events)))

    for name, info in base_results.items():
        if verbose: print(f"  OOF for {name}...")
        # Run OOF with the same factory used to train.
        # For models trained on subsets (h=5/15 or side-only), we generate OOF
        # on that subset, then map back to canonical index via (t, instrument);
        # rows for canonical events NOT in the subset get NaN.
        X_b = info["X"]; y_b = info["y"]
        t_b = info["t"]; t1_b = info["t1"]; w_b = info["w"]

        # Factory for OOF (recreate the same kind of model).
        if name.startswith("M1") or name.startswith("M2"):
            factory = lambda rs=42: XGBoostMeta(n_iter=10, embargo_td=embargo,
                                                  random_state=rs, n_splits_inner=3)
        elif name.startswith("M3") or name.startswith("M4"):
            factory = lambda rs=44: XGBoostMeta(n_iter=10, embargo_td=embargo,
                                                  random_state=rs, n_splits_inner=3)
        elif name.startswith("M5"):
            factory = lambda: ElasticNetLogReg(n_iter=10, embargo_td=embargo)
        elif name.startswith("M6"):
            factory = lambda: _RFMeta(n_estimators=300, max_depth=5)
        elif name.startswith("M7") or name.startswith("M8"):
            factory = lambda rs=46: XGBoostMeta(n_iter=8, embargo_td=embargo,
                                                  random_state=rs, n_splits_inner=3)
        else:
            continue

        try:
            oof = _oof_predict(factory, X_b, y_b, t_b, t1_b, w_b, embargo=embargo,
                              n_splits=5)
        except Exception as e:
            if verbose: print(f"    {name} OOF failed: {e}")
            continue

        # Map OOF predictions to the canonical index via (t, instrument).
        base_events = data.events_lab_by_h[info["h"]].iloc[info["tr_pos"]].reset_index(drop=True)
        base_key = pd.MultiIndex.from_arrays(
            [base_events["t"], base_events["instrument"]], names=["t", "instrument"],
        )
        oof_series = pd.Series(oof, index=base_key)
        # Reindex to canonical
        canonical_series = oof_series.reindex(canonical_key)
        oof_df[name] = canonical_series.values

    # Fill NaN (events not labelable at this h, or not in subset) with 0.5 (neutral).
    oof_df = oof_df.fillna(0.5)
    # Also store y_canonical alongside.
    oof_df["__y__"] = canonical_events["label"].astype(int).values
    return oof_df


def _train_stack_meta(oof_df: pd.DataFrame) -> LogisticRegression:
    """Train a LogReg meta-learner on the OOF base predictions."""
    feat_cols = [c for c in oof_df.columns if c != "__y__"]
    X_meta = oof_df[feat_cols].values
    y_meta = oof_df["__y__"].values
    # Light regularization; non-negative coefs would be ideal but skip for simplicity.
    meta = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, class_weight="balanced",
        random_state=42,
    )
    meta.fit(X_meta, y_meta)
    return meta


# --------------------------------------------------------------------------- #
def _predict_all_base(
    data: _Data, base_results: dict, X_target: pd.DataFrame,
    target_events: pd.DataFrame,
) -> pd.DataFrame:
    """Generate base model predictions on a target set (e.g. OOS events).

    All base models predict on the same target X. For side-specialised models
    (M7 long, M8 short), only predict on matching-side events; the other rows
    get 0.5 (neutral / no opinion)."""
    out = pd.DataFrame(index=X_target.index)
    target_sides = target_events["side"].values if "side" in target_events.columns else None
    for name, info in base_results.items():
        p = info["model"].predict_proba(X_target)
        if name.startswith("M7") and target_sides is not None:  # long only
            p_out = np.where(target_sides == 1, p, 0.5)
        elif name.startswith("M8") and target_sides is not None:  # short only
            p_out = np.where(target_sides == -1, p, 0.5)
        else:
            p_out = p
        out[name] = p_out
    return out


def _per_instrument_isotonic_calibrate(
    proba: np.ndarray, y: np.ndarray, instruments: np.ndarray,
    fit_proba: np.ndarray, fit_y: np.ndarray, fit_instruments: np.ndarray,
) -> np.ndarray:
    """Per-instrument isotonic calibration. Fits on (fit_*, fit_y) per
    instrument, applies to (proba) per instrument."""
    from sklearn.isotonic import IsotonicRegression
    out = proba.copy()
    for inst in np.unique(instruments):
        fit_mask = fit_instruments == inst
        tgt_mask = instruments == inst
        if fit_mask.sum() < 20 or len(set(fit_y[fit_mask])) < 2:
            continue
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(fit_proba[fit_mask], fit_y[fit_mask])
            out[tgt_mask] = iso.predict(proba[tgt_mask])
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# Master entry                                                                #
# --------------------------------------------------------------------------- #
def run_v4(
    boundary: pd.Timestamp = pd.Timestamp("2022-01-01"),
    predict_end: pd.Timestamp = pd.Timestamp("2022-07-01"),
    horizons: tuple[int, ...] = (5, 10, 15),
    embargo_days: int = 10,
    output_dir: Path = Path("results/sreeram"),
    predictions_filename: str = "predictions_v4.csv",
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"=== V4 STACKED ENSEMBLE :: boundary={boundary.date()} ===\n")

    data = _build_v4_data(
        boundary=boundary, predict_end=predict_end, horizons=horizons,
        embargo_days=embargo_days, verbose=verbose,
    )

    # 1. Train base models on commodity events
    if verbose:
        print("\n[v4] Training base models...")
    base = _train_base_models(data, verbose=verbose)

    # 2. Generate OOF predictions for stacking
    if verbose:
        print("\n[v4] Generating OOF predictions for stacking...")
    oof_df = _generate_oof_predictions(data, base, verbose=verbose)

    # 3. Train stack meta-learner
    if verbose:
        print("\n[v4] Training stack meta-learner (LogReg over base preds)...")
    meta = _train_stack_meta(oof_df)
    if verbose:
        feat_cols = [c for c in oof_df.columns if c != "__y__"]
        print(f"  Meta coefs: " + ", ".join(f"{n}={c:.2f}" for n,c in zip(feat_cols, meta.coef_[0])))

    # 4. OOS prediction
    h = 10  # canonical
    predict_mask = (data.t_lab_by_h[h].values >= boundary) & (data.t_lab_by_h[h].values < predict_end)
    oos_pos = np.where(predict_mask)[0]
    X_oos = data.X_lab_by_h[h].iloc[oos_pos]
    y_oos = data.y_lab_by_h[h].iloc[oos_pos].reset_index(drop=True)
    events_oos = data.events_lab_by_h[h].iloc[oos_pos].reset_index(drop=True)

    if verbose:
        print(f"\n[v4] OOS prediction: {len(X_oos)} events")
    base_oos_preds = _predict_all_base(data, base, X_oos, events_oos)
    stack_oos_logits = meta.decision_function(base_oos_preds.values)
    stack_oos_proba_raw = 1.0 / (1.0 + np.exp(-stack_oos_logits))

    # Per-instrument calibration: use OOF preds + true labels (in training)
    # as the calibration set
    canonical_h10_commod_tr = _commodity_filter(data.events_lab_by_h[h], data.tr_pos_by_h[h])
    canonical_events_tr = data.events_lab_by_h[h].iloc[canonical_h10_commod_tr].reset_index(drop=True)
    # Stack-predict on TRAINING events using the OOF base preds (this avoids leakage).
    stack_tr_logits = meta.decision_function(oof_df[[c for c in oof_df.columns if c != "__y__"]].values)
    stack_tr_proba = 1.0 / (1.0 + np.exp(-stack_tr_logits))

    stack_oos_proba = _per_instrument_isotonic_calibrate(
        stack_oos_proba_raw, y_oos.values, events_oos["instrument"].values,
        stack_tr_proba, oof_df["__y__"].values, canonical_events_tr["instrument"].values,
    )

    # 5. Evaluation
    from stml.evaluation import classification_report, per_instrument_breakdown
    rep_raw = classification_report(y_oos, stack_oos_proba_raw)
    rep_cal = classification_report(y_oos, stack_oos_proba)
    per_inst_cal = per_instrument_breakdown(events_oos, y_oos, stack_oos_proba)
    if verbose:
        print(f"\n[v4] STACK OOS (uncalibrated): AUC={rep_raw['auc']:.3f}  F1={rep_raw['f1']:.3f}  Brier={rep_raw['brier']:.3f}")
        print(f"[v4] STACK OOS (calibrated):  AUC={rep_cal['auc']:.3f}  F1={rep_cal['f1']:.3f}  Brier={rep_cal['brier']:.3f}")

    # 6. Per-base-model OOS AUCs for diagnostic
    from sklearn.metrics import roc_auc_score
    base_aucs = {}
    for name in base_oos_preds.columns:
        try:
            base_aucs[name] = roc_auc_score(y_oos, base_oos_preds[name].values)
        except Exception:
            base_aucs[name] = None
    if verbose:
        print("\n[v4] Per-base-model OOS AUC:")
        for n, a in base_aucs.items():
            print(f"  {n:20s}  {a:.3f}" if a is not None else f"  {n:20s}  N/A")

    # 7. Write predictions_v4.csv (predict on ALL ±1 events in window)
    predict_mask_all = (data.events_all["t"].values >= boundary) & (data.events_all["t"].values < predict_end)
    predict_pos_all = np.where(predict_mask_all)[0]
    X_pred_all = data.X_all.iloc[predict_pos_all]
    events_pred = data.events_all.iloc[predict_pos_all].reset_index(drop=True)
    base_pred_all = _predict_all_base(data, base, X_pred_all, events_pred)
    stack_logits_all = meta.decision_function(base_pred_all.values)
    stack_proba_raw_all = 1.0 / (1.0 + np.exp(-stack_logits_all))
    stack_proba_all = _per_instrument_isotonic_calibrate(
        stack_proba_raw_all, np.zeros(len(stack_proba_raw_all), dtype=int),  # dummy y (not used)
        events_pred["instrument"].values,
        stack_tr_proba, oof_df["__y__"].values, canonical_events_tr["instrument"].values,
    )

    # Build the CSV
    if "date" in data.signals.columns:
        sig_indexed = data.signals.set_index("date")
    else:
        sig_indexed = data.signals
    instruments = list(sig_indexed.columns)
    pred_window = sig_indexed.loc[
        (sig_indexed.index >= boundary) & (sig_indexed.index < predict_end),
        instruments,
    ]
    key_to_proba = pd.Series(
        stack_proba_all,
        index=pd.MultiIndex.from_arrays(
            [events_pred["t"], events_pred["instrument"]], names=["date", "instrument"],
        ),
    )
    rows = []
    for d, row in pred_window.iterrows():
        for inst in instruments:
            s = int(row[inst]) if not pd.isna(row[inst]) else 0
            if s == 0:
                rows.append({"date": d, "instrument": inst, "prediction": 0.0})
            else:
                p = float(key_to_proba.get((d, inst), 0.5))
                rows.append({"date": d, "instrument": inst, "prediction": p})
    df = pd.DataFrame(rows)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / predictions_filename
    df.to_csv(out_path, index=False, float_format="%.4f")
    if verbose:
        print(f"\n[v4] Wrote {len(df)} rows → {out_path}")

    return {
        "data": data,
        "base_models": base,
        "oof_df": oof_df,
        "meta_model": meta,
        "report_raw": rep_raw,
        "report_calibrated": rep_cal,
        "per_instrument": per_inst_cal,
        "base_oos_aucs": base_aucs,
        "predictions_df": df,
        "output_path": str(out_path),
        "stack_proba_oos_raw": stack_oos_proba_raw,
        "stack_proba_oos_cal": stack_oos_proba,
        "y_oos": y_oos,
        "events_oos": events_oos,
    }
