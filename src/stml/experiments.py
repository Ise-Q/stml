"""
experiments.py
==============
Multi-model experiments + per-sector ablation + cluster importance +
deep evaluation orchestrated as one callable function.

This is the bridge between the modular components (labeling / features /
regimes / models / importance / evaluation) and the actual marked
deliverables.

Public API:
  - :func:`run_all_models`        -- train logreg + XGBoost + MLP on the same
    split, report a head-to-head OOS comparison
  - :func:`per_sector_ablation`   -- train XGBoost per asset class, compare
    pooled vs per-sector
  - :func:`build_v2_artifacts`    -- full Stage 4-5 deliverable: trains all,
    computes importance, picks the best model for predictions_v2.csv,
    returns rich result dict for the report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stml.cv import PurgedKFold, split_by_boundary
from stml.evaluation import (
    classification_report,
    per_instrument_breakdown,
    confusion_matrix_df,
    threshold_sweep,
    calibration_table,
    optimal_threshold,
    regime_conditional_performance,
    baseline_compare,
    filtered_strategy_metrics,
)
from stml.features import compute_features, feature_groups
from stml.importance import (
    cluster_features,
    clustered_mdi,
    clustered_mda,
    cluster_economic_overlap,
)
from stml.io import load_clean_data
from stml.labeling import (
    extract_signal_events,
    get_meta_labels,
    get_uniqueness_weights,
)
from stml.models import ElasticNetLogReg, XGBoostMeta, MlpMeta
from stml.regimes import compute_regime_features


# --------------------------------------------------------------------------- #
ASSET_CLASSES = {
    "es1s": "equity", "nq1s": "equity", "fesx1s": "equity",
    "cl1s": "energy", "ho1s": "energy", "rb1s": "energy", "ng1s": "energy",
    "gc1s": "metals", "si1s": "metals", "hg1s": "metals", "pl1s": "metals",
}


@dataclass
class _Data:
    """Pre-computed shared inputs to avoid redoing labeling/features for each
    model run."""
    ohlcv: pd.DataFrame
    signals: pd.DataFrame
    events_all: pd.DataFrame          # all ±1 events
    events_lab: pd.DataFrame          # labelable subset
    X_lab: pd.DataFrame               # features for labelable events
    X_all: pd.DataFrame               # features for ALL ±1 events (for prediction)
    y_lab: pd.Series                  # binary labels
    t_lab: pd.Series                  # event start dates
    t1_lab: pd.Series                 # first-touch end dates
    w_lab: pd.Series                  # uniqueness weights
    side_lab: pd.Series               # primary signal direction
    ret_lab: pd.Series                # realised signed return at t1
    regimes_lab: pd.DataFrame         # HMM/GMM features aligned to events_lab
    boundary: pd.Timestamp
    predict_end: pd.Timestamp


def _build_data(boundary: pd.Timestamp, predict_end: pd.Timestamp,
                h: int = 10, pt_mult: float = 1.0, sl_mult: float = 1.0,
                vol_span: int = 100) -> _Data:
    ohlcv, signals = load_clean_data()
    labels = get_meta_labels(ohlcv, signals, h=h, pt_mult=pt_mult, sl_mult=sl_mult,
                             vol_span=vol_span, verbose=False)
    events_all = extract_signal_events(signals).reset_index(drop=True)
    feats = compute_features(
        ohlcv, events_all, signals,
        include_groups=("G1", "G2", "G3", "G4", "G5", "G7"),
    )
    regs = compute_regime_features(ohlcv, events_all, boundary=boundary)
    X_all = feats.join(regs, how="left").fillna(0.0)
    key = ["t", "instrument"]
    events_all_k = events_all.set_index(key)
    labels_k = labels.set_index(key)
    events_all["label"] = labels_k["label"].reindex(events_all_k.index).reset_index(drop=True).values
    events_all["t1_orig"] = labels_k["t1"].reindex(events_all_k.index).reset_index(drop=True).values
    events_all["ret"] = labels_k["ret"].reindex(events_all_k.index).reset_index(drop=True).values
    mask = ~events_all["label"].isna()
    events_lab = events_all.loc[mask].reset_index(drop=True)
    X_lab = X_all.loc[mask].reset_index(drop=True)
    regimes_lab = regs.loc[mask].reset_index(drop=True)
    weights = get_uniqueness_weights(labels)
    w_aligned = pd.Series(weights.values, index=labels.set_index(key).index)\
        .reindex(events_all.set_index(key).index).reset_index(drop=True)
    return _Data(
        ohlcv=ohlcv, signals=signals, events_all=events_all,
        events_lab=events_lab, X_lab=X_lab, X_all=X_all,
        y_lab=events_lab["label"].astype(int),
        t_lab=events_lab["t"],
        t1_lab=events_lab["t1_orig"],
        w_lab=w_aligned.loc[mask].reset_index(drop=True),
        side_lab=events_lab["side"],
        ret_lab=events_lab["ret"],
        regimes_lab=regimes_lab,
        boundary=boundary,
        predict_end=predict_end,
    )


def _split_train_oos(data: _Data, embargo: pd.Timedelta):
    """Return (train_idx, oos_idx) positional arrays into data.X_lab."""
    tr_pos, _ = split_by_boundary(data.t_lab, data.boundary, embargo_td=embargo)
    predict_mask = (data.t_lab.values >= data.boundary) & (data.t_lab.values < data.predict_end)
    oos_pos = np.where(predict_mask)[0]
    return tr_pos, oos_pos


# --------------------------------------------------------------------------- #
def _train_one_model(model, data: _Data, tr_pos: np.ndarray, embargo: pd.Timedelta):
    """Train one model on the training slice. Returns the fitted model."""
    X_tr = data.X_lab.iloc[tr_pos]
    y_tr = pd.Series(data.y_lab.values[tr_pos], index=X_tr.index)
    t_tr = pd.Series(data.t_lab.values[tr_pos], index=X_tr.index)
    t1_tr = pd.Series(data.t1_lab.values[tr_pos], index=X_tr.index)
    w_tr = pd.Series(data.w_lab.values[tr_pos], index=X_tr.index)
    model.fit(X_tr, y_tr, t=t_tr, t1=t1_tr, sample_weight=w_tr)
    return model


def _eval_one(
    model, data: _Data, oos_pos: np.ndarray,
) -> dict:
    """Evaluate a fitted model on the OOS slice."""
    X_oos = data.X_lab.iloc[oos_pos]
    y_oos = pd.Series(data.y_lab.values[oos_pos], index=X_oos.index)
    proba = model.predict_proba(X_oos)
    rep = classification_report(y_oos, proba)
    events_oos = data.events_lab.iloc[oos_pos]
    per_inst = per_instrument_breakdown(events_oos, y_oos, proba)
    return {
        "report": rep,
        "per_instrument": per_inst,
        "proba": proba,
        "y_true": y_oos,
        "events_oos": events_oos,
    }


# --------------------------------------------------------------------------- #
def run_all_models(
    boundary: pd.Timestamp = pd.Timestamp("2022-01-01"),
    predict_end: pd.Timestamp = pd.Timestamp("2022-07-01"),
    embargo_days: int = 10,
    n_iter: int = 15,
    n_splits_inner: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Train all three model families on the same data; return everything."""
    embargo = pd.Timedelta(days=embargo_days)
    data = _build_data(boundary=boundary, predict_end=predict_end)
    if verbose:
        print(f"Data: {len(data.X_lab)} labelable events, {data.X_lab.shape[1]} features")

    tr_pos, oos_pos = _split_train_oos(data, embargo)
    if verbose:
        print(f"Split: train={len(tr_pos)}, OOS={len(oos_pos)}")

    models = {}
    results = {}
    for name, ctor in [
        ("logreg",  lambda: ElasticNetLogReg(n_splits_inner=n_splits_inner, n_iter=n_iter,
                                              embargo_td=embargo, random_state=random_state)),
        ("xgboost", lambda: XGBoostMeta(n_splits_inner=n_splits_inner, n_iter=n_iter,
                                        embargo_td=embargo, random_state=random_state)),
        ("mlp",     lambda: MlpMeta(n_splits_inner=n_splits_inner, n_iter=n_iter,
                                    embargo_td=embargo, random_state=random_state)),
    ]:
        if verbose:
            print(f"\n--- Training {name} ---")
        m = _train_one_model(ctor(), data, tr_pos, embargo)
        models[name] = m
        res = _eval_one(m, data, oos_pos)
        results[name] = res
        if verbose:
            r = res["report"]
            print(f"{name}: AUC={r['auc']:.3f}  F1={r['f1']:.3f}  "
                  f"Brier={r['brier']:.3f}  LogLoss={r['log_loss']:.3f}")

    return {
        "data": data,
        "models": models,
        "results": results,
        "tr_pos": tr_pos,
        "oos_pos": oos_pos,
        "embargo": embargo,
    }


# --------------------------------------------------------------------------- #
def per_sector_ablation(
    boundary: pd.Timestamp = pd.Timestamp("2022-01-01"),
    predict_end: pd.Timestamp = pd.Timestamp("2022-07-01"),
    embargo_days: int = 10,
    n_iter: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Train XGBoost separately per asset class. Compare to pooled.

    Pooled vs per-sector tells us whether the panel is genuinely a 'shared
    function' or whether each sector has its own meta-model dynamics.
    """
    embargo = pd.Timedelta(days=embargo_days)
    data = _build_data(boundary=boundary, predict_end=predict_end)
    tr_pos, oos_pos = _split_train_oos(data, embargo)

    # Pooled (over everything).
    pooled = _train_one_model(
        XGBoostMeta(n_iter=n_iter, embargo_td=embargo, random_state=random_state),
        data, tr_pos, embargo,
    )
    pooled_eval = _eval_one(pooled, data, oos_pos)

    # Per-sector.
    sector_results = {}
    for sector in ["equity", "energy", "metals"]:
        instruments = [k for k, c in ASSET_CLASSES.items() if c == sector]
        sector_mask = data.events_lab["instrument"].isin(instruments).values
        tr_sec = np.array([i for i in tr_pos if sector_mask[i]])
        oos_sec = np.array([i for i in oos_pos if sector_mask[i]])
        if len(tr_sec) < 100 or len(oos_sec) < 10:
            sector_results[sector] = None
            continue
        m = _train_one_model(
            XGBoostMeta(n_iter=n_iter, embargo_td=embargo, random_state=random_state),
            data, tr_sec, embargo,
        )
        sector_results[sector] = {
            "model": m,
            "eval": _eval_one(m, data, oos_sec),
            "n_train": int(len(tr_sec)),
            "n_oos": int(len(oos_sec)),
        }
        if verbose:
            r = sector_results[sector]["eval"]["report"]
            print(f"{sector:8s}: n_train={len(tr_sec)} n_oos={len(oos_sec)}  "
                  f"AUC={r['auc']:.3f}  F1={r['f1']:.3f}")

    # Compare AUC across all instruments under pooled vs per-sector.
    pooled_per_inst = pooled_eval["per_instrument"]["auc"]
    sector_per_inst = pd.Series(dtype=float)
    for sec, sr in sector_results.items():
        if sr is None:
            continue
        for inst, aucv in sr["eval"]["per_instrument"]["auc"].items():
            sector_per_inst[inst] = aucv
    compare = pd.DataFrame({
        "pooled_auc": pooled_per_inst,
        "sector_auc": sector_per_inst,
        "sector": [ASSET_CLASSES.get(i) for i in pooled_per_inst.index],
    }).round(4)
    compare["sector_minus_pooled"] = (compare["sector_auc"] - compare["pooled_auc"]).round(4)
    return {
        "pooled": {"model": pooled, "eval": pooled_eval},
        "sectors": sector_results,
        "compare": compare,
    }


# --------------------------------------------------------------------------- #
def build_v2_artifacts(
    boundary: pd.Timestamp = pd.Timestamp("2022-01-01"),
    predict_end: pd.Timestamp = pd.Timestamp("2022-07-01"),
    embargo_days: int = 10,
    n_iter_main: int = 20,
    n_iter_sector: int = 10,
    random_state: int = 42,
    output_dir: Path = Path("results/sreeram"),
    predictions_filename: str = "predictions_v2.csv",
    verbose: bool = True,
) -> dict:
    """Stage 4 + Stage 5 master experiment. Returns a fat dict with:
      - all 3 model fits (logreg, xgboost, mlp)
      - per-sector ablation
      - cluster importance (MDI on xgboost, MDA on xgboost, cross-checked)
      - deep evaluation (calibration, threshold tuning, regime-conditional)
      - filtered-strategy metrics (meta-sized vs blind primary)
      - predictions_v2.csv written to disk (best model by OOS log-loss)
    """
    if verbose:
        print(f"=== Stage 4+5 v2 build :: boundary={boundary.date()} ===")

    out = run_all_models(
        boundary=boundary, predict_end=predict_end, embargo_days=embargo_days,
        n_iter=n_iter_main, random_state=random_state, verbose=verbose,
    )
    data = out["data"]
    embargo = out["embargo"]
    tr_pos = out["tr_pos"]
    oos_pos = out["oos_pos"]

    # === Per-sector ablation ===
    if verbose:
        print("\n=== Per-sector ablation (XGBoost) ===")
    sector_out = per_sector_ablation(
        boundary=boundary, predict_end=predict_end, embargo_days=embargo_days,
        n_iter=n_iter_sector, random_state=random_state, verbose=verbose,
    )

    # === Cluster-level importance with XGBoost ===
    if verbose:
        print("\n=== Cluster-level importance ===")
    xgb = out["models"]["xgboost"]
    X_tr = data.X_lab.iloc[tr_pos]
    X_oos = data.X_lab.iloc[oos_pos]
    y_oos = pd.Series(data.y_lab.values[oos_pos], index=X_oos.index)
    f2c, cluster_info = cluster_features(X_tr, max_k=12)
    n_clusters = len(set(f2c.values()))
    if verbose:
        print(f"Clusters: {n_clusters}")

    mdi = xgb.feature_importance("gain")
    cmdi = clustered_mdi(mdi, f2c)
    cmda = clustered_mda(xgb, X_oos, y_oos, f2c, scoring="neg_log_loss",
                        n_repeats=5, random_state=random_state)
    overlap = cluster_economic_overlap(f2c, feature_groups())

    # Cross-check: VSN attention-equivalent importance from the MLP (permutation).
    mlp = out["models"]["mlp"]
    mlp_imp = mlp.feature_importance(X_oos, y_oos, n_repeats=3)
    cmdi_mlp = clustered_mdi(mlp_imp, f2c)

    # === Deep evaluation on the BEST model (lowest OOS log-loss) ===
    rep_by_model = {k: v["report"] for k, v in out["results"].items()}
    best_name = min(rep_by_model.keys(), key=lambda k: rep_by_model[k]["log_loss"])
    best_model = out["models"][best_name]
    best_eval = out["results"][best_name]
    proba_oos = best_eval["proba"]
    if verbose:
        print(f"\nBest model by OOS log-loss: {best_name}")

    # Calibration
    cal_table = calibration_table(y_oos, proba_oos, n_bins=10)

    # Threshold tuning per instrument
    thr_global, thr_metrics = optimal_threshold(y_oos, proba_oos, metric="f1")
    thr_per_inst = {}
    for inst, idx in best_eval["events_oos"].groupby("instrument").groups.items():
        y_sub = y_oos.loc[idx]
        p_sub = pd.Series(proba_oos, index=y_oos.index).loc[idx].values
        if y_sub.nunique() < 2:
            continue
        thr_i, _ = optimal_threshold(y_sub, p_sub, metric="f1")
        thr_per_inst[inst] = thr_i

    # Regime-conditional (HMM state at each event, aligned per (date, instrument)).
    regime_state = data.regimes_lab["hmm_state_argmax"].iloc[oos_pos]
    regime_state.index = y_oos.index  # align indices
    regime_state = regime_state.dropna()
    if len(regime_state):
        regime_perf = regime_conditional_performance(
            y_oos.loc[regime_state.index],
            pd.Series(proba_oos, index=y_oos.index).loc[regime_state.index].values,
            regime_state,
        )
    else:
        regime_perf = pd.DataFrame()

    # Confusion matrix + threshold sweep + baseline
    cm = confusion_matrix_df(y_oos, (proba_oos >= 0.5).astype(int))
    cm_opt = confusion_matrix_df(y_oos, (proba_oos >= thr_global).astype(int))
    thr_sweep = threshold_sweep(y_oos, proba_oos)
    baseline = baseline_compare(y_oos, proba_oos, threshold=0.5)

    # Filtered-strategy metrics (very rough — full strategy in Stage 6)
    side_oos = data.side_lab.iloc[oos_pos].values
    ret_oos = data.ret_lab.iloc[oos_pos].values
    side_oos_s = pd.Series(side_oos, index=y_oos.index)
    ret_oos_s = pd.Series(ret_oos, index=y_oos.index)
    strat = filtered_strategy_metrics(y_oos, proba_oos, side_oos_s, ret_oos_s,
                                       threshold=thr_global)

    # === Write predictions_v2.csv using the best model ===
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = _build_predictions_csv(
        best_model, data, boundary=boundary, predict_end=predict_end,
    )
    out_path = out_dir / predictions_filename
    predictions_df.to_csv(out_path, index=False, float_format="%.4f")
    if verbose:
        print(f"\nWrote {len(predictions_df)} rows → {out_path}")

    return {
        "data": data,
        "models": out["models"],
        "results": out["results"],
        "best_name": best_name,
        "best_eval": best_eval,
        "sector": sector_out,
        "cluster": {
            "feature_to_cluster": f2c,
            "info": cluster_info,
            "mdi": cmdi,
            "mda": cmda,
            "mdi_mlp": cmdi_mlp,
            "overlap_with_groups": overlap,
        },
        "evaluation": {
            "calibration": cal_table,
            "threshold_global": thr_global,
            "threshold_per_instrument": thr_per_inst,
            "regime_conditional": regime_perf,
            "confusion_at_0_5": cm,
            "confusion_at_optimal": cm_opt,
            "threshold_sweep": thr_sweep,
            "baseline": baseline,
            "strategy_metrics": strat,
        },
        "predictions": predictions_df,
        "output_path": str(out_path),
    }


def _build_predictions_csv(
    model, data: _Data, boundary: pd.Timestamp, predict_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build the deliverable CSV from a fitted model.

    Predicts for every (date, instrument) in the prediction window. Rows with
    primary signal = 0 emit 0.0 by convention.
    """
    if "date" in data.signals.columns:
        sig_indexed = data.signals.set_index("date")
    else:
        sig_indexed = data.signals
    instruments = list(sig_indexed.columns)
    pred_window = sig_indexed.loc[
        (sig_indexed.index >= boundary) & (sig_indexed.index < predict_end),
        instruments,
    ]

    # Predict on ALL ±1 events in the window — features were already computed.
    predict_mask = (
        (data.events_all["t"].values >= boundary)
        & (data.events_all["t"].values < predict_end)
    )
    pos = np.where(predict_mask)[0]
    X_pred = data.X_all.iloc[pos]

    proba = model.predict_proba(X_pred)
    events_pred = data.events_all.iloc[pos]
    key_to_proba = pd.Series(
        proba,
        index=pd.MultiIndex.from_arrays(
            [events_pred["t"], events_pred["instrument"]],
            names=["date", "instrument"],
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
    return df
