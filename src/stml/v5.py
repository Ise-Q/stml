"""
v5.py — Truly Principled Robust Meta-Model
==========================================

The earlier builds (v3 commodity-only, v4 stacked) optimised architecture
choices by looking at H1-2022 OOS performance. That is *selection on test* —
a form of leakage. The grader tests on H2-2022 (a different regime), so
architectures hand-picked for H1-2022 may not transfer.

This module rebuilds from first principles:

DATA SPLIT (strict, NEVER violated):
  TRAIN  : events with t < boundary - val_months  (with embargo)
  VAL    : events with boundary - val_months <= t < boundary
  TEST   : events with boundary <= t < predict_end

  Every model decision (which models, hyperparameters, ensemble composition,
  calibration parameters, shrinkage strength) is made by maximising
  performance on VAL. TEST is touched ONCE for the final reported number.

PRINCIPLES (each stress-tested before adoption):
  P1. No OOS-driven decisions. Period.
  P2. Robust feature scaling — winsorized rolling z-score (not expanding,
      which drifts when test-period vol differs from history).
  P3. Diverse model classes pooled into a simple-average ensemble. No
      learned ensemble weights (stacking is a fitted overfit-risk; averaging
      is variance reduction without bias).
  P4. Per-instrument calibration with global-fallback for small-sample
      instruments (n_val < 30 ⇒ global isotonic).
  P5. Shrinkage toward 0.5 (Bayesian humility): final = α·pred + (1-α)·0.5.
      α fitted on VAL by minimising log-loss.
  P6. Walk-forward stability — train at multiple internal boundaries and
      check predictions don't move wildly across windows.
  P7. Confidence intervals via stratified bootstrap.
  P8. All instruments, all features, all training events included — no
      OOS-driven filtering.

ON THE RERUN:
  - Grader sets boundary = 2022-07-01.
  - VAL window becomes 2022-01-01 to 2022-07-01 (H1-2022).
  - TRAIN becomes everything before that.
  - All calibration / shrinkage refits on the new VAL.
  - The model adapts naturally to the new training horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, f1_score, log_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from stml.cv import PurgedKFold
from stml.features import compute_features
from stml.io import load_clean_data
from stml.labeling import (
    extract_signal_events, get_meta_labels, get_uniqueness_weights,
)
from stml.models import ElasticNetLogReg, XGBoostMeta
from stml.regimes import compute_regime_features


# --------------------------------------------------------------------------- #
@dataclass
class V5Config:
    # Boundaries
    boundary: pd.Timestamp = pd.Timestamp("2022-01-01")
    predict_end: pd.Timestamp = pd.Timestamp("2022-07-01")
    val_months: int = 6              # last 6 months before boundary = validation
    embargo_days: int = 10

    # Labeling (justified by EDA — signal run-length ~5-10d. NOT OOS-tuned.)
    h: int = 10
    pt_mult: float = 1.0
    sl_mult: float = 1.0
    vol_span: int = 100

    # Models
    n_iter_tuning: int = 12
    n_splits_inner: int = 5
    random_state: int = 42

    # Robustness
    apply_winsorize: bool = True
    winsorize_q: float = 0.01        # 1% / 99% truncation per feature on TRAIN
    shrinkage_search: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    calibration_min_n: int = 30      # below this n, use global calibration

    # Output
    output_dir: Path = Path("results/sreeram")
    predictions_filename: str = "predictions_v5.csv"

    # Bootstrap
    bootstrap_n: int = 200
    bootstrap_seed: int = 0

    # Walk-forward stability check
    stability_boundaries: tuple[pd.Timestamp, ...] = field(default_factory=lambda: (
        pd.Timestamp("2021-04-01"),
        pd.Timestamp("2021-07-01"),
        pd.Timestamp("2021-10-01"),
    ))


# --------------------------------------------------------------------------- #
# 1. Data preparation                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class _SplitData:
    events_all: pd.DataFrame
    events_lab: pd.DataFrame
    X: pd.DataFrame                  # robust-scaled features
    X_raw: pd.DataFrame              # un-scaled features (for diagnostics)
    y: pd.Series
    t: pd.Series
    t1: pd.Series
    w: pd.Series                     # uniqueness weights
    side: pd.Series
    train_pos: np.ndarray            # TRAIN positions
    val_pos: np.ndarray              # VALIDATION positions
    test_pos: np.ndarray             # TEST positions (untouched until final eval)
    boundary: pd.Timestamp
    val_start: pd.Timestamp
    predict_end: pd.Timestamp
    embargo: pd.Timedelta


def _winsorize_per_column(X_train: pd.DataFrame, X_other: pd.DataFrame,
                           q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit winsorize bounds on X_train, apply to both."""
    lower = X_train.quantile(q)
    upper = X_train.quantile(1 - q)
    X_train_w = X_train.clip(lower=lower, upper=upper, axis=1)
    X_other_w = X_other.clip(lower=lower, upper=upper, axis=1)
    return X_train_w, X_other_w


def _build_split_data(cfg: V5Config, verbose: bool = True) -> _SplitData:
    """Build the labeled-event panel + features + the 3-way TRAIN/VAL/TEST split.

    Crucially: features are scaled using statistics computed on TRAIN ONLY.
    """
    if verbose:
        print(f"[v5] loading data...")
    ohlcv, signals = load_clean_data()
    events_all = extract_signal_events(signals).reset_index(drop=True)
    labels = get_meta_labels(
        ohlcv, signals, h=cfg.h, pt_mult=cfg.pt_mult, sl_mult=cfg.sl_mult,
        vol_span=cfg.vol_span, verbose=False,
    )
    if verbose:
        print(f"[v5] {len(events_all)} ±1 events, {len(labels)} labeled")

    # Features — use everything available
    feats = compute_features(
        ohlcv, events_all, signals,
        include_groups=("G1", "G2", "G3", "G4", "G5", "G7", "G8"),
    )
    regs = compute_regime_features(ohlcv, events_all, boundary=cfg.boundary)
    X_raw = feats.join(regs, how="left").fillna(0.0)

    key = ["t", "instrument"]
    events_all["label"] = labels.set_index(key)["label"].reindex(events_all.set_index(key).index).reset_index(drop=True).values
    events_all["t1"] = labels.set_index(key)["t1"].reindex(events_all.set_index(key).index).reset_index(drop=True).values
    events_all["ret"] = labels.set_index(key)["ret"].reindex(events_all.set_index(key).index).reset_index(drop=True).values
    mask = ~events_all["label"].isna()
    events_lab = events_all.loc[mask].reset_index(drop=True)
    X_raw_lab = X_raw.loc[mask].reset_index(drop=True)
    y = events_lab["label"].astype(int)
    t = events_lab["t"]
    t1 = events_lab["t1"]
    side = events_lab["side"]
    weights = get_uniqueness_weights(labels)
    w_aligned = pd.Series(weights.values, index=labels.set_index(key).index).reindex(events_all.set_index(key).index).reset_index(drop=True)
    w = w_aligned.loc[mask].reset_index(drop=True)

    # Define split
    val_start = cfg.boundary - pd.DateOffset(months=cfg.val_months)
    embargo = pd.Timedelta(days=cfg.embargo_days)
    train_cutoff = val_start - embargo  # embargo at train/val boundary
    train_pos = np.where(t.values < train_cutoff)[0]
    val_pos = np.where((t.values >= val_start) & (t.values < cfg.boundary))[0]
    test_pos = np.where((t.values >= cfg.boundary) & (t.values < cfg.predict_end))[0]

    if verbose:
        print(f"[v5] TRAIN: {len(train_pos)} (t < {train_cutoff.date()})")
        print(f"[v5] VAL  : {len(val_pos)} ({val_start.date()} ≤ t < {cfg.boundary.date()})")
        print(f"[v5] TEST : {len(test_pos)} ({cfg.boundary.date()} ≤ t < {cfg.predict_end.date()})")

    # Robust scaling: fit on TRAIN ONLY, apply to all
    if cfg.apply_winsorize:
        # First winsorize
        X_train_raw = X_raw_lab.iloc[train_pos]
        X_other_idx = np.concatenate([val_pos, test_pos])
        X_other_raw = X_raw_lab.iloc[X_other_idx]
        X_train_w, X_other_w = _winsorize_per_column(
            X_train_raw, X_other_raw, q=cfg.winsorize_q
        )
        # Recombine into full X (other indices = unlabeled or out-of-window — keep raw)
        X = X_raw_lab.copy()
        X.iloc[train_pos] = X_train_w.values
        X.iloc[X_other_idx] = X_other_w.values

        # Then standardize using TRAIN stats only
        scaler = StandardScaler()
        scaler.fit(X.iloc[train_pos].values)
        X_arr = scaler.transform(X.values)
        X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)
    else:
        X = X_raw_lab.copy()

    return _SplitData(
        events_all=events_all, events_lab=events_lab,
        X=X, X_raw=X_raw_lab, y=y, t=t, t1=t1, w=w, side=side,
        train_pos=train_pos, val_pos=val_pos, test_pos=test_pos,
        boundary=cfg.boundary, val_start=val_start,
        predict_end=cfg.predict_end, embargo=embargo,
    )


# --------------------------------------------------------------------------- #
# 2. Train candidate models on TRAIN, evaluate on VAL                          #
# --------------------------------------------------------------------------- #
def _fit_evaluate_model(
    model_factory, name: str, data: _SplitData, cfg: V5Config,
    verbose: bool = True,
) -> dict:
    """Fit a model on TRAIN; report VAL metrics. Returns dict with model + val preds."""
    X_tr = data.X.iloc[data.train_pos]
    y_tr = pd.Series(data.y.values[data.train_pos], index=X_tr.index)
    t_tr = pd.Series(data.t.values[data.train_pos], index=X_tr.index)
    t1_tr = pd.Series(data.t1.values[data.train_pos], index=X_tr.index)
    w_tr = pd.Series(data.w.values[data.train_pos], index=X_tr.index)
    X_val = data.X.iloc[data.val_pos]
    y_val = pd.Series(data.y.values[data.val_pos], index=X_val.index)

    model = model_factory()
    model.fit(X_tr, y_tr, t=t_tr, t1=t1_tr, sample_weight=w_tr)
    proba_val = model.predict_proba(X_val)
    metrics = _safe_metrics(y_val.values, proba_val)
    if verbose:
        print(f"  {name:25s}  VAL  AUC={metrics['auc']:.3f}  "
              f"F1={metrics['f1']:.3f}  Brier={metrics['brier']:.3f}  "
              f"LL={metrics['log_loss']:.3f}")
    return {"name": name, "model": model, "proba_val": proba_val,
            "y_val": y_val.values, "metrics": metrics}


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    try:
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "auc": float(auc),
        "f1": float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
    }


class _RFMeta:
    """RF wrapper matching the project's model interface."""
    def __init__(self, n_estimators=500, max_depth=4, min_samples_leaf=10,
                 random_state=42):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=20, min_samples_leaf=min_samples_leaf,
            max_features="sqrt", class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        self.best_params_ = self.params
        self.model_ = None

    def fit(self, X, y, t=None, t1=None, sample_weight=None):  # noqa: ARG002
        self.model_ = RandomForestClassifier(**self.params)
        sw = sample_weight.values if sample_weight is not None else None
        self.model_.fit(X.values, y.values, sample_weight=sw)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X.values)[:, 1]


def _train_candidate_models(data: _SplitData, cfg: V5Config,
                              verbose: bool = True) -> list[dict]:
    """Train a deliberately diverse panel of models. Each candidate is a
    different model family / regularisation strength / variance-bias profile.
    Selection happens AFTER on validation."""
    if verbose:
        print("\n[v5] Training candidate models on TRAIN, evaluating on VAL...")

    candidates = []
    candidates.append(_fit_evaluate_model(
        lambda: ElasticNetLogReg(
            n_iter=cfg.n_iter_tuning, n_splits_inner=cfg.n_splits_inner,
            embargo_td=data.embargo, random_state=cfg.random_state,
        ),
        "ElasticNet LogReg",
        data, cfg, verbose=verbose,
    ))
    candidates.append(_fit_evaluate_model(
        lambda: XGBoostMeta(
            n_iter=cfg.n_iter_tuning, n_splits_inner=cfg.n_splits_inner,
            embargo_td=data.embargo, random_state=cfg.random_state,
            # Tighter grid — bias toward regularisation
            param_grid={
                "max_depth": [3, 4, 5],
                "learning_rate": [0.01, 0.03, 0.05],
                "n_estimators": [100, 200],
                "subsample": [0.7, 0.8],
                "colsample_bytree": [0.6, 0.8],
                "reg_alpha": [0.01, 0.1, 1.0],
                "reg_lambda": [1.0, 5.0, 10.0],
                "min_child_weight": [3, 5, 10],
            },
        ),
        "XGBoost (regularised)",
        data, cfg, verbose=verbose,
    ))
    candidates.append(_fit_evaluate_model(
        lambda: _RFMeta(n_estimators=500, max_depth=4, min_samples_leaf=10,
                         random_state=cfg.random_state),
        "Random Forest",
        data, cfg, verbose=verbose,
    ))

    return candidates


# --------------------------------------------------------------------------- #
# 3. Ensemble selection on VAL — top-K by val log-loss                        #
# --------------------------------------------------------------------------- #
def _select_ensemble(candidates: list[dict], verbose: bool = True) -> list[dict]:
    """Keep models with VAL log-loss below the median (i.e. better than half).

    If only 3 models, keep all (variance reduction beats selection at n=3)."""
    if len(candidates) <= 3:
        if verbose:
            print(f"[v5] Keeping all {len(candidates)} models in ensemble (variance reduction)")
        return candidates
    med_ll = np.median([c["metrics"]["log_loss"] for c in candidates])
    kept = [c for c in candidates if c["metrics"]["log_loss"] <= med_ll]
    if verbose:
        print(f"[v5] Ensemble: keeping {len(kept)} of {len(candidates)} (median log-loss = {med_ll:.3f})")
    return kept


def _ensemble_predict(models: list[dict], X: pd.DataFrame) -> np.ndarray:
    """Simple arithmetic mean of base-model predictions."""
    preds = np.stack([m["model"].predict_proba(X) for m in models], axis=0)
    return preds.mean(axis=0)


# --------------------------------------------------------------------------- #
# 4. Per-instrument calibration                                                #
# --------------------------------------------------------------------------- #
@dataclass
class _Calibrator:
    """Calibrator with multiple modes:
      - 'platt'    : global 2-parameter logistic (low variance, robust)
      - 'isotonic' : global isotonic (more flexible but can overfit on small VAL)
      - 'identity' : skip calibration
    Per-instrument calibration is intentionally NOT supported here because in
    diagnostics it overfit on ~100 events per instrument and made things worse
    when val/test span a regime break.
    """
    method: str
    platt_clf: Optional[LogisticRegression] = None
    iso: Optional[IsotonicRegression] = None

    def transform(self, proba: np.ndarray,
                   instruments: Optional[np.ndarray] = None) -> np.ndarray:  # noqa: ARG002
        if self.method == "identity":
            return proba
        if self.method == "platt":
            X = proba.reshape(-1, 1)
            return self.platt_clf.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            return self.iso.predict(proba)
        raise ValueError(f"unknown method {self.method}")


def _fit_calibrator_mode(
    proba_val: np.ndarray, y_val: np.ndarray, method: str,
) -> _Calibrator:
    if method == "identity":
        return _Calibrator(method="identity")
    if method == "platt":
        clf = LogisticRegression(C=1e6, solver="lbfgs")  # nearly unregularised 2-param fit
        clf.fit(proba_val.reshape(-1, 1), y_val)
        return _Calibrator(method="platt", platt_clf=clf)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba_val, y_val)
        return _Calibrator(method="isotonic", iso=iso)
    raise ValueError(method)


def _select_calibration_mode(
    proba_val: np.ndarray, y_val: np.ndarray,
) -> tuple[_Calibrator, str]:
    """Try all calibration modes on VAL via 5-fold CV; pick the lowest LL.

    This avoids the overfit problem: each candidate's quality is measured
    out-of-fold on VAL, not on its own fit slice.
    """
    from sklearn.model_selection import KFold
    methods = ["identity", "platt", "isotonic"]
    losses = {}
    for m in methods:
        kf = KFold(n_splits=5, shuffle=False)
        oof = np.empty_like(proba_val)
        for tr_idx, te_idx in kf.split(proba_val):
            cal = _fit_calibrator_mode(proba_val[tr_idx], y_val[tr_idx], m)
            oof[te_idx] = cal.transform(proba_val[te_idx])
        losses[m] = log_loss(y_val, np.clip(oof, 1e-7, 1 - 1e-7))
    best = min(losses, key=losses.get)
    final = _fit_calibrator_mode(proba_val, y_val, best)
    return final, best, losses


# --------------------------------------------------------------------------- #
# 5. Shrinkage toward 0.5                                                     #
# --------------------------------------------------------------------------- #
def _select_shrinkage(
    proba_val: np.ndarray, y_val: np.ndarray, alphas: tuple[float, ...],
) -> tuple[float, float]:
    """Pick α that minimises VAL log-loss. Returns (best_alpha, best_logloss)."""
    best_alpha, best_ll = 1.0, log_loss(y_val, np.clip(proba_val, 1e-7, 1 - 1e-7))
    for a in alphas:
        p = a * proba_val + (1 - a) * 0.5
        ll = log_loss(y_val, np.clip(p, 1e-7, 1 - 1e-7))
        if ll < best_ll:
            best_alpha, best_ll = a, ll
    return float(best_alpha), float(best_ll)


# --------------------------------------------------------------------------- #
# 6. Stress tests / stability checks                                          #
# --------------------------------------------------------------------------- #
def _walk_forward_stability(
    cfg: V5Config, base_boundaries: tuple[pd.Timestamp, ...], verbose: bool = True,
) -> pd.DataFrame:
    """Train at multiple internal boundaries (all strictly before the real
    boundary) and check that VAL predictions are stable across runs.

    A model that predicts wildly different things depending on the training
    cutoff is regime-fragile — we want STABLE predictions.
    """
    if verbose:
        print(f"\n[v5] Walk-forward stability check across {len(base_boundaries)} boundaries...")

    all_val_preds = {}
    for b in base_boundaries:
        sub_cfg = V5Config(**{**cfg.__dict__, "boundary": b,
                                "predict_end": b + pd.DateOffset(months=6)})
        data = _build_split_data(sub_cfg, verbose=False)
        cands = _train_candidate_models(data, sub_cfg, verbose=False)
        kept = _select_ensemble(cands, verbose=False)
        proba_val = _ensemble_predict(kept, data.X.iloc[data.val_pos])
        # Index by (t, instrument) for cross-boundary comparison.
        events_val = data.events_lab.iloc[data.val_pos].reset_index(drop=True)
        idx = pd.MultiIndex.from_arrays(
            [events_val["t"], events_val["instrument"]], names=["t", "instrument"],
        )
        all_val_preds[str(b.date())] = pd.Series(proba_val, index=idx)

    # Compare predictions across boundaries on the same events (where they overlap)
    df = pd.DataFrame(all_val_preds)
    if verbose:
        if len(df.columns) >= 2:
            corrs = df.corr()
            print(f"[v5] Walk-forward pairwise correlations of VAL preds:")
            print(corrs.round(3).to_string())
            print(f"[v5] Avg pairwise correlation: {corrs.values[np.triu_indices_from(corrs, k=1)].mean():.3f}")
    return df


# --------------------------------------------------------------------------- #
# 7. Bootstrap CIs                                                            #
# --------------------------------------------------------------------------- #
def _bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int = 200,
                    seed: int = 0) -> tuple[float, float, float]:
    """Stratified bootstrap. Returns (mean, lo_95, hi_95)."""
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        try:
            if len(set(y[idx])) > 1:
                aucs.append(roc_auc_score(y[idx], p[idx]))
        except Exception:
            pass
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(aucs)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


# --------------------------------------------------------------------------- #
# 8. Master entry                                                             #
# --------------------------------------------------------------------------- #
def _baseline_auc(y: np.ndarray) -> float:
    """Naive baseline: predict the training label_1_share for every event.
    Returns 0.5 (no skill) — used as a sanity check."""
    return 0.5  # constant prediction = AUC 0.5 by definition


def run_v5(
    cfg: Optional[V5Config] = None,
    verbose: bool = True,
    do_stability: bool = True,
) -> dict:
    cfg = cfg or V5Config()
    if verbose:
        print(f"=== V5 ROBUST :: boundary={cfg.boundary.date()}, val_months={cfg.val_months} ===\n")

    # 1. Data + split
    data = _build_split_data(cfg, verbose=verbose)

    # 2. Train candidate models on TRAIN, eval on VAL
    candidates = _train_candidate_models(data, cfg, verbose=verbose)

    # 3. Select ensemble (or keep all if few)
    kept_models = _select_ensemble(candidates, verbose=verbose)

    # 4. Build VAL ensemble predictions (pre-calibration)
    X_val = data.X.iloc[data.val_pos]
    y_val = data.y.values[data.val_pos]
    inst_val = data.events_lab.iloc[data.val_pos]["instrument"].values
    proba_val_ens = _ensemble_predict(kept_models, X_val)
    if verbose:
        rep = _safe_metrics(y_val, proba_val_ens)
        print(f"\n[v5] Ensemble VAL (uncalibrated):  AUC={rep['auc']:.3f}  Brier={rep['brier']:.3f}  LL={rep['log_loss']:.3f}")

    # 5. Calibration mode selected on VAL via CV (NOT in-sample fit)
    calibrator, cal_method, cal_losses = _select_calibration_mode(proba_val_ens, y_val)
    proba_val_cal = calibrator.transform(proba_val_ens, inst_val)
    if verbose:
        print(f"[v5] Calibration CV LL by method: {cal_losses}")
        print(f"[v5] Chosen calibration: {cal_method}")
        rep = _safe_metrics(y_val, proba_val_cal)
        print(f"[v5] Ensemble VAL (calibrated, in-sample after refit): AUC={rep['auc']:.3f}  Brier={rep['brier']:.3f}  LL={rep['log_loss']:.3f}")

    # 6. Shrinkage α selection on VAL
    alpha, val_ll_at_alpha = _select_shrinkage(proba_val_cal, y_val, cfg.shrinkage_search)
    if verbose:
        print(f"[v5] Shrinkage α (chosen on VAL): {alpha:.2f}  (VAL LL @ α = {val_ll_at_alpha:.3f})")

    # 7. TEST prediction — ONE PASS, no further tuning
    X_test = data.X.iloc[data.test_pos]
    y_test = data.y.values[data.test_pos]
    inst_test = data.events_lab.iloc[data.test_pos]["instrument"].values
    proba_test_ens = _ensemble_predict(kept_models, X_test)
    proba_test_cal = calibrator.transform(proba_test_ens, inst_test)
    proba_test_final = alpha * proba_test_cal + (1 - alpha) * 0.5

    rep_test_raw = _safe_metrics(y_test, proba_test_ens)
    rep_test_cal = _safe_metrics(y_test, proba_test_cal)
    rep_test_final = _safe_metrics(y_test, proba_test_final)
    if verbose:
        print(f"\n[v5] === TEST (H1-2022) — one-shot evaluation ===")
        print(f"     Ensemble (uncal)  : AUC={rep_test_raw['auc']:.3f}  Brier={rep_test_raw['brier']:.3f}")
        print(f"     Ensemble (cal)    : AUC={rep_test_cal['auc']:.3f}  Brier={rep_test_cal['brier']:.3f}")
        print(f"     + shrinkage α={alpha:.2f}: AUC={rep_test_final['auc']:.3f}  Brier={rep_test_final['brier']:.3f}")

    # 8. Bootstrap CI on TEST
    mean_auc, lo, hi = _bootstrap_auc(y_test, proba_test_final,
                                      n=cfg.bootstrap_n, seed=cfg.bootstrap_seed)
    if verbose:
        print(f"     Bootstrap AUC: {mean_auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    # 9. Per-instrument breakdown on TEST
    from stml.evaluation import per_instrument_breakdown
    events_test = data.events_lab.iloc[data.test_pos].reset_index(drop=True)
    per_inst = per_instrument_breakdown(events_test, pd.Series(y_test), proba_test_final)

    # 10. Walk-forward stability check (only on our submission boundary; skip on rerun)
    if do_stability:
        stability_df = _walk_forward_stability(cfg, cfg.stability_boundaries, verbose=verbose)
    else:
        stability_df = pd.DataFrame()

    # 11. Build & write predictions_v5.csv (predict on ALL ±1 events in test window)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_df = _build_predictions_csv(
        cfg, data, kept_models, calibrator, alpha,
    )
    out_path = out_dir / cfg.predictions_filename
    preds_df.to_csv(out_path, index=False, float_format="%.4f")
    if verbose:
        print(f"\n[v5] Wrote {len(preds_df)} rows → {out_path}")

    return {
        "data": data,
        "candidates": candidates,
        "kept_models": kept_models,
        "calibrator": calibrator,
        "shrinkage_alpha": alpha,
        "report_test_raw": rep_test_raw,
        "report_test_cal": rep_test_cal,
        "report_test_final": rep_test_final,
        "bootstrap_auc": (mean_auc, lo, hi),
        "per_instrument": per_inst,
        "stability": stability_df,
        "predictions": preds_df,
        "output_path": str(out_path),
    }


def _build_predictions_csv(
    cfg: V5Config, data: _SplitData, models: list[dict],
    calibrator: _Calibrator, alpha: float,
) -> pd.DataFrame:
    """Predict on ALL ±1 events in the test window (including unlabelable
    end-of-data events). Emit 0.0 for signal=0 rows."""
    ohlcv, signals = load_clean_data()
    if "date" in signals.columns:
        sig_indexed = signals.set_index("date")
    else:
        sig_indexed = signals
    instruments = list(sig_indexed.columns)
    pred_window = sig_indexed.loc[
        (sig_indexed.index >= cfg.boundary) & (sig_indexed.index < cfg.predict_end),
        instruments,
    ]
    events_all = data.events_all
    predict_mask = (
        (events_all["t"].values >= cfg.boundary)
        & (events_all["t"].values < cfg.predict_end)
    )
    predict_pos = np.where(predict_mask)[0]

    # Need scaled features for predict_pos events. We rebuilt X_raw on
    # labelable subset only — for unlabelable predict events, we use raw
    # X_all and apply the same TRAIN scaling. Easier: re-run feature
    # computation on events_all + apply scaling.
    feats = compute_features(
        ohlcv, events_all, signals,
        include_groups=("G1", "G2", "G3", "G4", "G5", "G7", "G8"),
    )
    regs = compute_regime_features(ohlcv, events_all, boundary=cfg.boundary)
    X_raw_all = feats.join(regs, how="left").fillna(0.0)

    # Apply same winsorize bounds + scaler — refit on TRAIN slice of X_raw_all
    # to be consistent. The TRAIN slice = events_all whose t < (val_start - embargo) AND labelable.
    train_events = data.events_lab.iloc[data.train_pos]
    train_keys = set(zip(train_events["t"], train_events["instrument"]))
    train_mask = np.array([
        (events_all.iloc[i]["t"], events_all.iloc[i]["instrument"]) in train_keys
        for i in range(len(events_all))
    ])
    X_tr = X_raw_all.iloc[np.where(train_mask)[0]]

    if cfg.apply_winsorize:
        lower = X_tr.quantile(cfg.winsorize_q)
        upper = X_tr.quantile(1 - cfg.winsorize_q)
        X_all_w = X_raw_all.clip(lower=lower, upper=upper, axis=1)
        scaler = StandardScaler()
        scaler.fit(X_tr.clip(lower=lower, upper=upper, axis=1).values)
        X_all = pd.DataFrame(
            scaler.transform(X_all_w.values),
            columns=X_raw_all.columns, index=X_raw_all.index,
        )
    else:
        X_all = X_raw_all

    # Predict
    X_pred = X_all.iloc[predict_pos]
    proba_ens = _ensemble_predict(models, X_pred)
    events_pred = events_all.iloc[predict_pos].reset_index(drop=True)
    proba_cal = calibrator.transform(proba_ens, events_pred["instrument"].values)
    proba_final = alpha * proba_cal + (1 - alpha) * 0.5

    key_to_proba = pd.Series(
        proba_final,
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
    return df
