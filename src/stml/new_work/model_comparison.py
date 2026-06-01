"""model_comparison.py — Model Development stage of the metamodel pipeline.

Pipeline order: feature engineering → triple-barrier labelling → THIS FILE
→ cluster-level feature importance → evaluation.

Builds and evaluates four classifier families (logistic, RF, XGB, MLP) for
13 instrument groups using CPCV (n_groups=6, k=2), with nested inner-fold
hyperparameter tuning, per-instrument OOS slicing, calibration curves, and
a champion selection table.

Groups
------
Individual (10): es1s nq1s fesx1s cl1s rb1s ng1s gc1s si1s hg1s pl1s
Pooled (3):
  energy_all    = [cl1s, ho1s, rb1s, ng1s]
  energy_cl_ho  = [cl1s, ho1s]
  precious      = [gc1s, si1s, pl1s]
ho1s has no individual run (too thin).

Usage
-----
    python -m stml.new_work.model_comparison
    python -m stml.new_work.model_comparison --groups es1s energy_all
    python -m stml.new_work.model_comparison --force   # re-run existing
    python -m stml.new_work.model_comparison --compile-only  # rebuild tables only

Outputs:  src/stml/new_work/outputs/model_comparison/
    {group}/{model}/oos_predictions.csv
    {group}/{model}/per_instrument_metrics.csv
    {group}/{model}/metrics.csv
    {group}/{model}/calibration.png
    {group}/{model}/hyperparams.json
    master_results.csv
    selection_table.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work.cpcv_search import CombinatorialPurgedKFold
from stml.new_work.feature_importance import (
    apply_hygiene,
    build_feature_matrix,
    load_all_data,
    _feature_cols as _fi_feature_cols,
)
from stml.na_checks import native_returns, wide_returns


# ── Configuration ─────────────────────────────────────────────────────────────

GROUPS: dict[str, list[str]] = {
    "es1s":         ["es1s"],
    "nq1s":         ["nq1s"],
    "fesx1s":       ["fesx1s"],
    "cl1s":         ["cl1s"],
    "rb1s":         ["rb1s"],
    "ng1s":         ["ng1s"],
    "gc1s":         ["gc1s"],
    "si1s":         ["si1s"],
    "hg1s":         ["hg1s"],
    "pl1s":         ["pl1s"],
    "energy_all":   ["cl1s", "ho1s", "rb1s", "ng1s"],
    "energy_cl_ho": ["cl1s", "ho1s"],
    "precious":     ["gc1s", "si1s", "pl1s"],
}

# Which regimes compete for each instrument in the selection table.
INSTRUMENT_REGIMES: dict[str, list[str]] = {
    "es1s":   ["es1s"],
    "nq1s":   ["nq1s"],
    "fesx1s": ["fesx1s"],
    "cl1s":   ["cl1s", "energy_all", "energy_cl_ho"],
    "ho1s":   ["energy_all", "energy_cl_ho"],
    "rb1s":   ["rb1s", "energy_all"],
    "ng1s":   ["ng1s", "energy_all"],
    "gc1s":   ["gc1s", "precious"],
    "si1s":   ["si1s", "precious"],
    "pl1s":   ["pl1s", "precious"],
    "hg1s":   ["hg1s"],
}

CPCV_N_GROUPS: int = 6
CPCV_K: int = 2
CPCV_EMBARGO: float = 0.01
SEED: int = 42
MODEL_NAMES: list[str] = ["logistic", "rf", "xgb", "mlp"]
OUTPUTS: Path = _HERE / "outputs" / "model_comparison"

# Metadata columns (never used as features)
_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})
# Agreed hygiene drop; not currently in the feature build but guarded here.
_DROP_ALWAYS = frozenset({"participation_60"})


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


# ── Group assembly ─────────────────────────────────────────────────────────────

def assemble_group(
    group_name: str,
    instruments: list[str],
    data: dict,
    wide_rets: pd.DataFrame,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Build, hygiene, and cache the events_df for one group.

    For pooled groups: stacks all instruments' events, applies shared hygiene,
    then appends one-hot instrument dummies as additional features so the model
    absorbs per-name base-rate and level differences.

    Folds operate on the sorted-by-date pooled set; the CPCV splitter naturally
    produces calendar-aligned test blocks, so contemporaneous cross-instrument
    events always move together.
    """
    cache_path = cache_dir / f"{group_name}_events.parquet"
    if cache_path.exists() and not force:
        print(f"  [cache hit] {group_name}")
        return pd.read_parquet(cache_path)

    print(f"  Building group '{group_name}': {instruments}")
    ev_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []

    for inst in instruments:
        daily_df, events_df = build_feature_matrix(inst, data, wide_rets)
        drops = [c for c in _DROP_ALWAYS if c in events_df.columns]
        if drops:
            events_df = events_df.drop(columns=drops)
        ev_parts.append(events_df)
        daily_parts.append(daily_df)

    if not ev_parts:
        return pd.DataFrame()

    stacked = (
        pd.concat(ev_parts, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    # For hygiene, pass a placeholder daily_df (apply_hygiene doesn't use it).
    hygiene_log: list[str] = []
    stacked, hygiene_log = apply_hygiene(stacked, pd.DataFrame(), hygiene_log)

    print(f"  {group_name}: {len(stacked)} events after hygiene")
    for line in hygiene_log:
        print(f"    {line[:120]}")

    # Instrument one-hot dummies for pooled groups (added after hygiene so they
    # are not caught by the NZV filter on the per-instrument individual case).
    if stacked["instrument"].nunique() > 1:
        dummies = pd.get_dummies(
            stacked["instrument"], prefix="inst", drop_first=True, dtype=float
        )
        stacked = pd.concat([stacked.reset_index(drop=True), dummies], axis=1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    stacked.to_parquet(cache_path)
    return stacked


# ── Inner hyperparameter tuning helpers ───────────────────────────────────────

def _inner_split(n: int, frac: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Single expanding-window inner split (train on first frac, val on rest)."""
    split = max(int(n * frac), 5)
    split = min(split, n - 5)
    if split <= 0 or split >= n:
        return np.arange(n), np.arange(0)  # degenerate: no valid inner val
    return np.arange(split), np.arange(split, n)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return -1.0
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return -1.0


# ── Model fitting: Logistic ───────────────────────────────────────────────────

def _tune_fit_logistic(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    events_tr: pd.DataFrame,
) -> tuple[StandardScaler, LogisticRegression, dict]:
    """Elastic-net logistic with nested inner time-split tuning.

    Standardises inputs within the training fold (scaler fitted here, applied
    to test data by the caller). Class weights computed from training labels.
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr)

    inner_tr, inner_val = _inner_split(len(X_sc), frac=0.75)
    best_C, best_l1 = 0.1, 0.5

    if len(inner_val) >= 5:
        X_in, X_val = X_sc[inner_tr], X_sc[inner_val]
        y_in, y_val = y_tr[inner_tr], y_tr[inner_val]
        best_auc = -1.0
        for C in [0.01, 0.1, 1.0]:
            for l1 in [0.0, 0.5, 1.0]:
                try:
                    m = LogisticRegression(
                        C=C, l1_ratio=l1, penalty="elasticnet", solver="saga",
                        max_iter=500, class_weight="balanced", random_state=SEED,
                    )
                    m.fit(X_in, y_in)
                    auc = _safe_auc(y_val, m.predict_proba(X_val)[:, 1])
                    if auc > best_auc:
                        best_auc, best_C, best_l1 = auc, C, l1
                except Exception:
                    pass

    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(len(y_tr) - n_pos, 1)
    cw = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}

    model = LogisticRegression(
        C=best_C, l1_ratio=best_l1, penalty="elasticnet", solver="saga",
        max_iter=2000, class_weight=cw, random_state=SEED,
    )
    model.fit(X_sc, y_tr)
    return scaler, model, {"C": best_C, "l1_ratio": best_l1}


# ── Model fitting: Random Forest ──────────────────────────────────────────────

def _tune_fit_rf(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    events_tr: pd.DataFrame,
) -> tuple[RandomForestClassifier, dict]:
    """Regularised RF with shallow trees + nested inner tuning for depth/leaf size.

    Uses avg_uniqueness sample weights on the final fit (AFML Ch. 4 convention).
    """
    inner_tr, inner_val = _inner_split(len(X_tr), frac=0.75)
    best_depth, best_msl = 4, 20

    if len(inner_val) >= 5:
        X_in, X_val = X_tr[inner_tr], X_tr[inner_val]
        y_in, y_val = y_tr[inner_tr], y_tr[inner_val]
        best_auc = -1.0
        for depth in [2, 4, 6]:
            for msl in [10, 20]:
                try:
                    m = RandomForestClassifier(
                        n_estimators=100, max_depth=depth, min_samples_leaf=msl,
                        max_features="sqrt", class_weight="balanced",
                        random_state=SEED, n_jobs=-1,
                    )
                    w_in = events_tr.iloc[inner_tr]["avg_uniqueness"].to_numpy(dtype=float)
                    w_in = w_in / w_in.mean() if w_in.mean() > 0 else np.ones(len(w_in))
                    m.fit(X_in, y_in, sample_weight=w_in)
                    auc = _safe_auc(y_val, m.predict_proba(X_val)[:, 1])
                    if auc > best_auc:
                        best_auc, best_depth, best_msl = auc, depth, msl
                except Exception:
                    pass

    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(len(y_tr) - n_pos, 1)
    cw = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}

    model = RandomForestClassifier(
        n_estimators=300, max_depth=best_depth, min_samples_leaf=best_msl,
        max_features="sqrt", class_weight=cw, random_state=SEED, n_jobs=-1,
    )
    w_tr = events_tr["avg_uniqueness"].to_numpy(dtype=float)
    w_tr = w_tr / w_tr.mean() if w_tr.mean() > 0 else np.ones(len(y_tr))
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    return model, {"max_depth": best_depth, "min_samples_leaf": best_msl}


# ── Model fitting: XGBoost ────────────────────────────────────────────────────

def _tune_fit_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    events_tr: pd.DataFrame,
) -> tuple[xgb.XGBClassifier, dict]:
    """XGBoost with strong regularisation; n_estimators via early stopping.

    Nested grid over (max_depth, learning_rate); early stopping on last 20%
    of the OUTER training fold determines the final model (no refit on full
    training needed — the 20% holdout is inside the outer train fold and
    never overlaps the CPCV test block).
    """
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(len(y_tr) - n_pos, 1)
    spw = n_neg / n_pos  # scale_pos_weight for imbalance

    inner_tr, inner_val = _inner_split(len(X_tr), frac=0.80)
    best_depth, best_lr, best_n = 3, 0.05, 100

    def _xgb_params(depth: int, lr: float) -> dict:
        return dict(
            n_estimators=300, max_depth=depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=2.0, reg_alpha=0.1, gamma=0.1,
            scale_pos_weight=spw, eval_metric="logloss",
            early_stopping_rounds=20, random_state=SEED,
            verbosity=0, n_jobs=-1,
        )

    if len(inner_val) >= 5 and len(np.unique(y_tr[inner_val])) > 1:
        X_in, X_val = X_tr[inner_tr], X_tr[inner_val]
        y_in, y_val = y_tr[inner_tr], y_tr[inner_val]
        best_auc = -1.0
        for depth in [3, 4]:
            for lr in [0.01, 0.05]:
                try:
                    m = xgb.XGBClassifier(**_xgb_params(depth, lr))
                    m.fit(X_in, y_in, eval_set=[(X_val, y_val)], verbose=False)
                    auc = _safe_auc(y_val, m.predict_proba(X_val)[:, 1])
                    if auc > best_auc:
                        best_auc, best_depth, best_lr = auc, depth, lr
                        best_n = max(int(getattr(m, "best_iteration", 50)) + 1, 10)
                except Exception:
                    pass

    # Final model on outer training fold with early stopping on inner 20%
    X_in2, X_val2 = X_tr[inner_tr], X_tr[inner_val]
    y_in2, y_val2 = y_tr[inner_tr], y_tr[inner_val]

    params = _xgb_params(best_depth, best_lr)

    if len(inner_val) >= 5 and len(np.unique(y_val2)) > 1:
        model = xgb.XGBClassifier(**params)
        model.fit(X_in2, y_in2, eval_set=[(X_val2, y_val2)], verbose=False)
    else:
        # Degenerate: fit on full training with fixed n_estimators
        params_fixed = {k: v for k, v in params.items()
                        if k not in ("early_stopping_rounds",)}
        params_fixed["n_estimators"] = best_n
        model = xgb.XGBClassifier(**params_fixed)
        model.fit(X_tr, y_tr)

    actual_n = max(int(getattr(model, "best_iteration", best_n - 1)) + 1, 1)
    return model, {"max_depth": best_depth, "learning_rate": best_lr, "n_estimators": actual_n}


# ── Model fitting: MLP ────────────────────────────────────────────────────────

def _tune_fit_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[StandardScaler, MLPClassifier, dict]:
    """Small MLP (sklearn) with L2 regularisation and early stopping.

    Dropout is not supported in sklearn's MLPClassifier; L2 weight decay via
    alpha and early_stopping=True provide the required regularisation for this
    'completeness' family. Inputs are standardised within the training fold.
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr)

    inner_tr, inner_val = _inner_split(len(X_sc), frac=0.75)
    best_hidden: tuple = (64,)
    best_alpha = 0.001

    if len(inner_val) >= 5:
        X_in, X_val = X_sc[inner_tr], X_sc[inner_val]
        y_in, y_val = y_tr[inner_tr], y_tr[inner_val]
        best_auc = -1.0
        for hidden in [(64,), (64, 32)]:
            for alpha in [0.001, 0.01]:
                try:
                    m = MLPClassifier(
                        hidden_layer_sizes=hidden, alpha=alpha,
                        activation="relu", solver="adam",
                        learning_rate_init=1e-3, max_iter=200,
                        early_stopping=False, random_state=SEED,
                    )
                    m.fit(X_in, y_in)
                    auc = _safe_auc(y_val, m.predict_proba(X_val)[:, 1])
                    if auc > best_auc:
                        best_auc, best_hidden, best_alpha = auc, hidden, alpha
                except Exception:
                    pass

    # Final model with early stopping on 20% holdout (sklearn handles it internally)
    model = MLPClassifier(
        hidden_layer_sizes=best_hidden, alpha=best_alpha,
        activation="relu", solver="adam", learning_rate_init=1e-3,
        max_iter=500, early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=20, random_state=SEED,
    )
    model.fit(X_sc, y_tr)
    return scaler, model, {"hidden_layer_sizes": best_hidden, "alpha": best_alpha}


# ── CPCV loop ─────────────────────────────────────────────────────────────────

def run_cpcv_model(
    model_name: str,
    events_df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Run CPCV for one model family across the full group events set.

    Returns
    -------
    oos_df : DataFrame with columns date, instrument, y_true, y_score, fold.
    hyperparams_list : per-fold chosen hyperparameters.
    """
    ev_meta = events_df[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    X = events_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = events_df["bin"].to_numpy(dtype=int)

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO
    )

    oos_rows: list[dict] = []
    hyperparams_list: list[dict] = []

    for fold_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        events_tr = ev_meta.iloc[tr_idx].reset_index(drop=True)
        events_te = ev_meta.iloc[te_idx].reset_index(drop=True)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        try:
            if model_name == "logistic":
                scaler, model, params = _tune_fit_logistic(X_tr, y_tr, events_tr)
                prob = model.predict_proba(scaler.transform(X_te))[:, 1]
            elif model_name == "rf":
                model, params = _tune_fit_rf(X_tr, y_tr, events_tr)
                prob = model.predict_proba(X_te)[:, 1]
            elif model_name == "xgb":
                model, params = _tune_fit_xgb(X_tr, y_tr, events_tr)
                prob = model.predict_proba(X_te)[:, 1]
            elif model_name == "mlp":
                scaler, model, params = _tune_fit_mlp(X_tr, y_tr)
                prob = model.predict_proba(scaler.transform(X_te))[:, 1]
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            print(f"    fold {fold_i} failed ({model_name}): {e}")
            traceback.print_exc()
            continue

        hyperparams_list.append({"fold": fold_i, **{k: str(v) for k, v in params.items()}})

        for i in range(len(te_idx)):
            oos_rows.append({
                "date":       events_te.iloc[i]["date"],
                "instrument": events_te.iloc[i]["instrument"],
                "y_true":     int(y_te[i]),
                "y_score":    float(prob[i]),
                "fold":       fold_i,
            })

    return pd.DataFrame(oos_rows), hyperparams_list


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_fold_metrics(oos_df: pd.DataFrame) -> dict[str, Any]:
    """Mean ± std of AUC, log-loss, Brier across CPCV paths."""
    if oos_df.empty:
        return {"auc_mean": np.nan, "auc_std": np.nan, "logloss_mean": np.nan,
                "logloss_std": np.nan, "brier_mean": np.nan, "brier_std": np.nan,
                "n_folds": 0}

    fold_rows: list[dict] = []
    for fold_i, fdf in oos_df.groupby("fold"):
        if fdf["y_true"].nunique() < 2 or len(fdf) < 2:
            continue
        try:
            fold_rows.append({
                "auc":    roc_auc_score(fdf["y_true"], fdf["y_score"]),
                "ll":     log_loss(fdf["y_true"], fdf["y_score"]),
                "brier":  brier_score_loss(fdf["y_true"], fdf["y_score"]),
            })
        except Exception:
            pass

    if not fold_rows:
        return {"auc_mean": np.nan, "auc_std": np.nan, "logloss_mean": np.nan,
                "logloss_std": np.nan, "brier_mean": np.nan, "brier_std": np.nan,
                "n_folds": 0}

    fm = pd.DataFrame(fold_rows)
    return {
        "auc_mean":    float(fm["auc"].mean()),
        "auc_std":     float(fm["auc"].std()),
        "logloss_mean": float(fm["ll"].mean()),
        "logloss_std":  float(fm["ll"].std()),
        "brier_mean":  float(fm["brier"].mean()),
        "brier_std":   float(fm["brier"].std()),
        "n_folds":     len(fm),
    }


def per_instrument_metrics(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument AUC/logloss/brier (slice of OOS predictions).

    AUC is computed over all OOS events for the instrument; auc_std is the
    std across CPCV paths (folds) where the instrument has 2+ classes.
    """
    rows: list[dict] = []
    for inst, idf in oos_df.groupby("instrument"):
        if idf["y_true"].nunique() < 2:
            continue

        try:
            auc_overall = roc_auc_score(idf["y_true"], idf["y_score"])
        except Exception:
            auc_overall = np.nan

        fold_aucs = []
        for _, fdf in idf.groupby("fold"):
            if fdf["y_true"].nunique() < 2:
                continue
            try:
                fold_aucs.append(roc_auc_score(fdf["y_true"], fdf["y_score"]))
            except Exception:
                pass
        auc_std = float(np.std(fold_aucs)) if len(fold_aucs) > 1 else np.nan

        try:
            ll = log_loss(idf["y_true"], idf["y_score"])
            brier = brier_score_loss(idf["y_true"], idf["y_score"])
        except Exception:
            ll = brier = np.nan

        rows.append({
            "instrument": inst,
            "auc_mean":   auc_overall,
            "auc_std":    auc_std,
            "logloss":    ll,
            "brier":      brier,
            "n_events":   len(idf),
        })
    return pd.DataFrame(rows)


# ── Calibration ───────────────────────────────────────────────────────────────

def _calibration_plot(
    oos_df: pd.DataFrame,
    group: str,
    model: str,
    out_path: Path,
) -> None:
    if oos_df.empty or oos_df["y_true"].nunique() < 2 or len(oos_df) < 20:
        return
    try:
        n_bins = min(10, max(3, len(oos_df) // 40))
        frac_pos, mean_pred = calibration_curve(
            oos_df["y_true"], oos_df["y_score"], n_bins=n_bins
        )
        brier = brier_score_loss(oos_df["y_true"], oos_df["y_score"])
        base_rate = oos_df["y_true"].mean()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mean_pred, frac_pos, "s-", label=f"{model}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.axhline(y=base_rate, color="gray", linestyle=":", linewidth=0.8,
                   label=f"Base rate={base_rate:.2f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{group}/{model}  |  Brier={brier:.4f}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  Calibration plot failed ({group}/{model}): {e}")


# ── Output saving ─────────────────────────────────────────────────────────────

def _save_group_model(
    group: str,
    model: str,
    oos_df: pd.DataFrame,
    hyperparams: list[dict],
    metrics: dict,
    out_dir: Path,
) -> None:
    gm_dir = out_dir / group / model
    gm_dir.mkdir(parents=True, exist_ok=True)

    oos_df.to_csv(gm_dir / "oos_predictions.csv", index=False)

    inst_metrics = per_instrument_metrics(oos_df)
    inst_metrics.to_csv(gm_dir / "per_instrument_metrics.csv", index=False)

    pd.DataFrame([metrics]).to_csv(gm_dir / "metrics.csv", index=False)

    _calibration_plot(oos_df, group, model, gm_dir / "calibration.png")

    with open(gm_dir / "hyperparams.json", "w") as fh:
        json.dump(hyperparams, fh, indent=2, default=str)


# ── Group × model runner ──────────────────────────────────────────────────────

def run_group_all_models(
    group_name: str,
    events_df: pd.DataFrame,
    out_dir: Path,
    force: bool = False,
) -> None:
    feat_cols = _feat_cols(events_df)
    n_events = len(events_df)

    print(f"\n{'='*62}")
    print(f"Group: {group_name}  |  {n_events} events  |  {len(feat_cols)} features")
    print("=" * 62)

    for model_name in MODEL_NAMES:
        oos_path = out_dir / group_name / model_name / "oos_predictions.csv"
        if oos_path.exists() and not force:
            print(f"  [{model_name}] already done — skipping (use --force to redo)")
            continue

        print(f"  [{model_name}] running CPCV ...", flush=True)
        try:
            oos_df, hyperparams = run_cpcv_model(model_name, events_df, feat_cols)
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            continue

        if oos_df.empty:
            print(f"    no OOS predictions produced — skipping save")
            continue

        metrics = compute_fold_metrics(oos_df)
        auc_m = metrics["auc_mean"]
        auc_s = metrics["auc_std"]
        brier = metrics["brier_mean"]
        nf = metrics["n_folds"]
        print(
            f"    AUC {auc_m:.3f}±{auc_s:.3f}  "
            f"Brier {brier:.4f}  "
            f"n_folds={nf}"
        )
        _save_group_model(group_name, model_name, oos_df, hyperparams, metrics, out_dir)


# ── Master results and selection table ────────────────────────────────────────

def compile_master_results(out_dir: Path) -> pd.DataFrame:
    """Collect per_instrument_metrics.csv from all completed (group, model) dirs."""
    rows: list[dict] = []
    for group_dir in sorted(out_dir.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        group = group_dir.name
        if group not in GROUPS:
            continue
        for model_dir in sorted(group_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if model not in MODEL_NAMES:
                continue
            inst_csv = model_dir / "per_instrument_metrics.csv"
            if not inst_csv.exists():
                continue
            inst_df = pd.read_csv(inst_csv)
            for _, row in inst_df.iterrows():
                rows.append({
                    "group":      group,
                    "model":      model,
                    "instrument": row["instrument"],
                    "auc_mean":   round(float(row["auc_mean"]), 4),
                    "auc_std":    round(float(row["auc_std"]), 4) if pd.notna(row.get("auc_std")) else np.nan,
                    "logloss":    round(float(row["logloss"]), 4),
                    "brier":      round(float(row["brier"]), 4),
                    "n_events":   int(row["n_events"]),
                })
    master = pd.DataFrame(rows)
    if not master.empty:
        master = master.sort_values(
            ["instrument", "group", "model"]
        ).reset_index(drop=True)
    master.to_csv(out_dir / "master_results.csv", index=False)
    return master


def build_selection_table(master: pd.DataFrame) -> pd.DataFrame:
    """For each instrument, rank regimes × models by per-instrument AUC.

    Produces: instrument, best_group, best_model, best_auc,
              runner_up_group, runner_up_model, runner_up_auc.
    """
    rows: list[dict] = []
    for inst, regimes in INSTRUMENT_REGIMES.items():
        candidates = master[
            (master["instrument"] == inst) & (master["group"].isin(regimes))
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values("auc_mean", ascending=False).reset_index(drop=True)
        best = candidates.iloc[0]
        row: dict = {
            "instrument":    inst,
            "best_group":    best["group"],
            "best_model":    best["model"],
            "best_auc":      best["auc_mean"],
            "best_logloss":  best["logloss"],
            "best_brier":    best["brier"],
            "n_events":      best["n_events"],
        }
        if len(candidates) > 1:
            runner = candidates.iloc[1]
            row["runner_up_group"] = runner["group"]
            row["runner_up_model"] = runner["model"]
            row["runner_up_auc"]   = runner["auc_mean"]
        else:
            row["runner_up_group"] = None
            row["runner_up_model"] = None
            row["runner_up_auc"]   = None
        rows.append(row)

    sel = pd.DataFrame(rows)
    return sel


# ── Entry points ──────────────────────────────────────────────────────────────

def run_all(
    groups: list[str] | None = None,
    force: bool = False,
    compile_only: bool = False,
) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTPUTS / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if compile_only:
        print("Compiling master results and selection table only ...\n")
        master = compile_master_results(OUTPUTS)
        sel = build_selection_table(master)
        sel.to_csv(OUTPUTS / "selection_table.csv", index=False)
        print(master.to_string(index=False))
        print("\n--- Selection Table ---")
        print(sel.to_string(index=False))
        return

    target_groups = groups if groups is not None else list(GROUPS.keys())
    invalid = [g for g in target_groups if g not in GROUPS]
    if invalid:
        print(f"Warning: unknown groups {invalid}; skipping")
        target_groups = [g for g in target_groups if g in GROUPS]

    print("Loading data ...")
    data = load_all_data()
    rets_long = native_returns(data["ohlcv"], kind="log")
    w_rets = wide_returns(rets_long).sort_index()
    print("Data loaded.\n")

    for group_name in target_groups:
        instruments = GROUPS[group_name]
        events_df = assemble_group(
            group_name, instruments, data, w_rets, cache_dir, force=force
        )
        if events_df.empty or len(events_df) < 30:
            print(f"  {group_name}: too few events ({len(events_df)}), skipping")
            continue
        run_group_all_models(group_name, events_df, OUTPUTS, force=force)

    print("\n" + "=" * 62)
    print("Compiling master results ...")
    master = compile_master_results(OUTPUTS)
    sel = build_selection_table(master)
    sel.to_csv(OUTPUTS / "selection_table.csv", index=False)

    print("\nMASTER RESULTS:")
    if not master.empty:
        print(master.to_string(index=False))
    else:
        print("  (no results yet)")

    print("\nSELECTION TABLE:")
    if not sel.empty:
        print(sel.to_string(index=False))
    else:
        print("  (no results yet)")

    print(f"\nOutputs in: {OUTPUTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Metamodel model-comparison harness"
    )
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help=f"Groups to run (default: all). Choices: {list(GROUPS.keys())}"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if OOS predictions already exist"
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Skip model runs; just recompile master_results and selection_table"
    )
    args = parser.parse_args()
    run_all(groups=args.groups, force=args.force, compile_only=args.compile_only)
