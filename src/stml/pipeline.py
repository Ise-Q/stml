"""
pipeline.py
===========
End-to-end master pipeline: data → labels → features → train → predict → CSV.

The pipeline is callable as :func:`run_pipeline` with the train/predict boundary
as a parameter. For our submission the boundary is ``2022-01-01``; on rerun the
grader sets it to ``2022-07-01`` and reproducibility *is* the deliverable.

This module wires together the building blocks in:
    stml.io         — data loading
    stml.labeling   — triple-barrier meta-labels + sample weights
    stml.features   — causal feature engineering
    stml.cv         — purged-CV / walk-forward / boundary splits
    stml.models     — model wrappers (ElasticNetLogReg here; XGBoost / VSN later)
    stml.evaluation — classification + per-instrument breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stml.cv import PurgedKFold, split_by_boundary, assert_no_leakage
from stml.features import compute_features
from stml.io import load_clean_data
from stml.labeling import (
    extract_signal_events,
    get_meta_labels,
    get_uniqueness_weights,
)
from stml.models import ElasticNetLogReg
from stml.regimes import compute_regime_features
from stml.evaluation import (
    classification_report,
    confusion_matrix_df,
    per_instrument_breakdown,
    baseline_compare,
    threshold_sweep,
)


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """Master configuration. All defaults are the *base* settings; ablations
    later vary these and the rerun mechanic preserves them as code parameters."""

    # Labeling
    h: int = 10
    pt_mult: float = 1.0
    sl_mult: float = 1.0
    vol_span: int = 100

    # Boundary / window
    train_predict_boundary: pd.Timestamp = pd.Timestamp("2022-01-01")
    predict_end: pd.Timestamp = pd.Timestamp("2022-07-01")
    embargo_days: int = 10

    # Model tuning
    inner_n_splits: int = 5
    n_iter: int = 20
    random_state: int = 42

    # Feature config
    feature_groups: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
    zscore_min_periods: int = 60
    include_regimes: bool = True
    hmm_n_states: int = 3
    gmm_n_components: int = 3

    # Output
    output_dir: Path = Path("results/sreeram")
    predictions_filename: str = "predictions_v0.csv"

    # Universe (None = all 11 instruments)
    instruments: Optional[list[str]] = None


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    config: PipelineConfig
    labels: pd.DataFrame
    features: pd.DataFrame
    weights: pd.Series
    train_idx: np.ndarray
    predict_idx: np.ndarray
    model: ElasticNetLogReg
    predictions: pd.DataFrame                # deliverable CSV content
    eval_report: dict[str, float] = field(default_factory=dict)
    per_instrument: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline: pd.DataFrame = field(default_factory=pd.DataFrame)
    threshold_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    in_sample_report: dict[str, float] = field(default_factory=dict)


def run_pipeline(
    config: PipelineConfig | None = None,
    verbose: bool = True,
) -> PipelineResult:
    """Run the end-to-end pipeline once.

    Returns a :class:`PipelineResult` with everything needed for diagnostics
    and writing the deliverable CSV.
    """
    cfg = config or PipelineConfig()
    if verbose:
        print(f"[run_pipeline] boundary={cfg.train_predict_boundary.date()}, "
              f"predict_end={cfg.predict_end.date()}, h={cfg.h}")

    # 1. Load data --------------------------------------------------------- #
    ohlcv, signals = load_clean_data()
    if verbose:
        print(f"[1/7] loaded ohlcv {ohlcv.shape}, signals {signals.shape}")

    # 2. Labeling (triple-barrier) ---------------------------------------- #
    labels = get_meta_labels(
        ohlcv, signals,
        h=cfg.h, pt_mult=cfg.pt_mult, sl_mult=cfg.sl_mult,
        vol_span=cfg.vol_span,
        instruments=cfg.instruments,
        verbose=False,
    )
    weights = get_uniqueness_weights(labels)
    if verbose:
        print(f"[2/7] labeled events: {len(labels):,} | label_1 share = {labels.label.mean():.3f}")

    # 3. Build the full event universe (incl. unlabelable end-of-data ±1   #
    #    events that we still must predict on). Then compute features.    #
    events_all = extract_signal_events(signals, instruments=cfg.instruments)
    # Use labels.index for labelable events; events_all has its own index.
    # Re-key everything for clarity.
    events_all = events_all.reset_index(drop=True)
    features_all = compute_features(
        ohlcv, events_all, signals,
        include_groups=tuple(g for g in cfg.feature_groups if g != "G6"),
        zscore_min_periods=cfg.zscore_min_periods,
    )
    if cfg.include_regimes and "G6" in cfg.feature_groups:
        regime_feats = compute_regime_features(
            ohlcv, events_all,
            boundary=cfg.train_predict_boundary,
            n_states=cfg.hmm_n_states,
            n_components=cfg.gmm_n_components,
        )
        features_all = features_all.join(regime_feats, how="left")
        if verbose:
            print(f"[3/7] features computed: {features_all.shape} (incl. G6 regimes)")
    else:
        if verbose:
            print(f"[3/7] features computed: {features_all.shape}")

    # Align labels back to events_all by (t, instrument).
    key_cols = ["t", "instrument"]
    events_all_keyed = events_all.set_index(key_cols)
    labels_keyed = labels.set_index(key_cols)
    # Build a label series aligned to events_all (NaN where label unavailable).
    label_aligned = labels_keyed["label"].reindex(events_all_keyed.index)
    side_aligned = events_all_keyed["side"]
    t_series = events_all["t"]
    t1_series = labels_keyed["t1"].reindex(events_all_keyed.index)
    # For events without a label, t1 is also missing; use t + h business days
    # as a *bookkeeping* t1 (only ever used in CV for these unlabelled events
    # if they were in training, which they won't be — training requires a label).
    fallback_t1 = (events_all["t"] + pd.tseries.offsets.BDay(cfg.h)).values
    t1_filled = t1_series.values.copy()
    mask_missing_t1 = pd.isna(t1_filled)
    t1_filled[mask_missing_t1] = fallback_t1[mask_missing_t1]
    t1_series_full = pd.Series(t1_filled, index=events_all.index, name="t1")
    weights_aligned = pd.Series(np.nan, index=events_all.index)
    # Align uniqueness weights from labels onto events_all by (t, instrument).
    w_keyed = pd.Series(weights.values, index=labels.set_index(key_cols).index)
    weights_aligned = w_keyed.reindex(events_all_keyed.index)
    weights_aligned.index = events_all.index  # restore positional alignment

    # 4. Boundary split --------------------------------------------------- #
    embargo = pd.Timedelta(days=cfg.embargo_days)
    train_idx_all, predict_idx_all = split_by_boundary(
        t_series, cfg.train_predict_boundary, embargo_td=embargo
    )
    # Training set: events with both a label AND t < boundary - embargo.
    has_label = ~label_aligned.isna()
    train_mask = np.zeros(len(events_all), dtype=bool)
    train_mask[train_idx_all] = True
    train_mask &= has_label.values
    # Prediction set: events with t in [boundary, predict_end].
    predict_mask = (
        (t_series.values >= cfg.train_predict_boundary)
        & (t_series.values < cfg.predict_end)
    )
    train_idx = np.where(train_mask)[0]
    predict_idx = np.where(predict_mask)[0]
    if verbose:
        print(f"[4/7] split: train={len(train_idx):,}, predict={len(predict_idx):,}")

    # 5. Train ------------------------------------------------------------ #
    X_train = features_all.iloc[train_idx]
    y_train = pd.Series(label_aligned.values[train_idx].astype(int), index=X_train.index)
    sw_train = pd.Series(weights_aligned.values[train_idx], index=X_train.index)
    t_train = pd.Series(t_series.values[train_idx], index=X_train.index)
    t1_train = pd.Series(t1_series_full.values[train_idx], index=X_train.index)

    # Drop any rows with NaN features (extremely rare after stage 3a, possible
    # near start of history for some instruments).
    feat_nan_mask = X_train.isna().any(axis=1)
    if feat_nan_mask.any():
        if verbose:
            print(f"    dropping {feat_nan_mask.sum()} training rows with NaN features")
        X_train = X_train.loc[~feat_nan_mask]
        y_train = y_train.loc[X_train.index]
        sw_train = sw_train.loc[X_train.index]
        t_train = t_train.loc[X_train.index]
        t1_train = t1_train.loc[X_train.index]

    model = ElasticNetLogReg(
        n_splits_inner=cfg.inner_n_splits,
        n_iter=cfg.n_iter,
        embargo_td=embargo,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train, t=t_train, t1=t1_train, sample_weight=sw_train)
    if verbose:
        print(f"[5/7] model fit. best_params={model.best_params_}")

    # 6. In-sample + out-of-sample evaluation ----------------------------- #
    # In-sample (sanity check — should be moderate, not too perfect)
    proba_train = model.predict_proba(X_train)
    in_sample_report = classification_report(y_train, proba_train)
    if verbose:
        print(f"    IN-sample: AUC={in_sample_report['auc']:.3f}, F1={in_sample_report['f1']:.3f}, "
              f"Brier={in_sample_report['brier']:.3f}")

    # Out-of-sample (only labelable events in the predict window — i.e.
    # events with t in [boundary, predict_end] AND a label). For our window
    # all events are labelable since data ends 2022-06-30 and predict_end is
    # the same date.
    oos_mask = predict_mask & has_label.values
    oos_idx = np.where(oos_mask)[0]
    X_oos = features_all.iloc[oos_idx]
    y_oos = pd.Series(label_aligned.values[oos_idx].astype(int), index=X_oos.index)
    # Drop NaN feature rows.
    oos_nan = X_oos.isna().any(axis=1)
    X_oos = X_oos.loc[~oos_nan]
    y_oos = y_oos.loc[X_oos.index]
    proba_oos = model.predict_proba(X_oos)

    eval_report = classification_report(y_oos, proba_oos)
    events_oos = events_all.loc[X_oos.index]
    per_inst = per_instrument_breakdown(events_oos, y_oos, proba_oos)
    cm = confusion_matrix_df(y_oos, (proba_oos >= 0.5).astype(int))
    baseline = baseline_compare(y_oos, proba_oos)
    thresh_curve = threshold_sweep(y_oos, proba_oos)
    if verbose:
        print(f"[6/7] OOS: n={eval_report['n']}, AUC={eval_report['auc']:.3f}, "
              f"F1={eval_report['f1']:.3f}, Brier={eval_report['brier']:.3f}")
        print(f"      label_1_share = {eval_report['label_1_share']:.3f}")

    # 7. Build deliverable predictions.csv -------------------------------- #
    # Format: one row per (date, instrument) in the prediction window.
    # signal == 0 rows get prediction = 0.0.
    if "date" in signals.columns:
        sig_indexed = signals.set_index("date")
    else:
        sig_indexed = signals
    instruments = cfg.instruments or list(sig_indexed.columns)
    pred_window = sig_indexed.loc[
        (sig_indexed.index >= cfg.train_predict_boundary)
        & (sig_indexed.index < cfg.predict_end),
        instruments,
    ]
    pred_rows = []
    # Predict on ALL predict_idx events (not just OOS labelable).
    X_predict_all = features_all.iloc[predict_idx]
    # Drop rows with NaN features.
    nan_mask_pred = X_predict_all.isna().any(axis=1)
    X_predict = X_predict_all.loc[~nan_mask_pred]
    proba_predict = model.predict_proba(X_predict)
    pred_lookup = pd.Series(proba_predict, index=X_predict.index)
    # For predict-window events with NaN features: emit 0.5 (neutral).
    for pos in predict_idx:
        if pos not in pred_lookup.index:
            pred_lookup.loc[pos] = 0.5
    events_predict = events_all.iloc[predict_idx]
    keyed_preds = pd.Series(pred_lookup.reindex(events_predict.index).values,
                            index=pd.MultiIndex.from_arrays(
                                [events_predict["t"], events_predict["instrument"]],
                                names=["date", "instrument"],
                            ))

    for d, row in pred_window.iterrows():
        for inst in instruments:
            sig_val = int(row[inst]) if not pd.isna(row[inst]) else 0
            if sig_val == 0:
                pred_rows.append({"date": d, "instrument": inst, "prediction": 0.0})
            else:
                p = keyed_preds.get((d, inst), 0.5)
                pred_rows.append({"date": d, "instrument": inst, "prediction": float(p)})
    predictions_df = pd.DataFrame(pred_rows)
    predictions_df["date"] = predictions_df["date"].dt.strftime("%Y-%m-%d")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / cfg.predictions_filename
    predictions_df.to_csv(out_path, index=False, float_format="%.4f")
    if verbose:
        print(f"[7/7] wrote {len(predictions_df)} prediction rows → {out_path}")

    return PipelineResult(
        config=cfg,
        labels=labels,
        features=features_all,
        weights=weights_aligned,
        train_idx=train_idx,
        predict_idx=predict_idx,
        model=model,
        predictions=predictions_df,
        eval_report=eval_report,
        per_instrument=per_inst,
        confusion=cm,
        baseline=baseline,
        threshold_curve=thresh_curve,
        in_sample_report=in_sample_report,
    )
