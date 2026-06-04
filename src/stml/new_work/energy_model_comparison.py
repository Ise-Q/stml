"""energy_model_comparison.py — Variant comparison for energy instruments (new labels).

Champions (selection_table.csv, new triple-barrier labels):
    cl1s  – cl1s         / logistic  (NO SIGNAL, lower_ci=0.49)
    ho1s  – energy_cl_ho / MLP       (NO SIGNAL, lower_ci=0.30)
    rb1s  – rb1s         / logistic  (SIGNAL, marginal, lower_ci=0.51)
    ng1s  – energy_all   / MLP       (SIGNAL, marginal, lower_ci=0.52)

Pooled mechanics:
    cl1s / rb1s: individual logistic models.
    ho1s: energy_cl_ho MLP pool — separate pool refit per variant on that
          instrument's feature subset.  Trained on all energy_cl_ho rows;
          scored on ho1s CPCV slice (Phase 1) or ho1s sealed-test slice (Phase 2).
    ng1s: energy_all MLP pool — as above, scored on ng1s slice.

MLP hyperparameters are LOCKED from the champion run (do not retune):
    energy_cl_ho: hidden=(64,), alpha=0.01
    energy_all:   hidden=(64,), alpha=0.01
StandardScaler is fit per fold on the pool training rows.

Variants:
    cl1s (logistic, individual):
        full    – all 101 cl1s features
        pruned  – C2_f11_lowfreq_macro + C14_f12  (the two significant clusters)
        reduced – PC1(C2_f11_lowfreq_macro) + PC1(C14_f12)
                  + f11_crude_stock_surprise + f2_vol_10 + f11_china_pmi_level

    ho1s (MLP, energy_cl_ho pool):
        full    – all 101 energy_cl_ho features
        pruned  – C8_f11_lowfreq_macro + C15_f11 + C1_f1  (top-3 raw MDA)
        reduced – f11_vix_term_slope + f4_pc3 + f1_bb_bandwidth_20 + f11_vix_5d_change
                  (f4_pc3 is the pre-computed frozen F4 latent — used as-is)

    rb1s (logistic, individual):
        full        – all 101 rb1s features
        pruned      – F5_signal + C6_f11_lowfreq_macro + C1_f1  (top-3 raw MDA)
        reduced     – f5_signal + f11_hy_oas_5d_change + f11_dist_stock_surprise
                      + f2_vol_ratio_20_60
        reduced_min – f5_signal + f11_hy_oas_5d_change  (signal purity test)

    ng1s (MLP, energy_all pool):
        full    – all 104 energy_all features
        pruned  – C15_f7  (the one significant cluster)
        reduced – f7_oi_change + f7_oi_level + f2_atr_14 + hmm_vol_next_turbulent
                  + hmm_vol_p2_turbulent + hmm_vol_p0_calm
                  (proposed via RF importance heuristic; validated here on MLP)
                  (inst_rb1s excluded — constant zero on the ng1s pool slice)

Phase 1: CPCV(n_groups=6, k=2, embargo=0.01) on TRAIN only.
         Lock = simplest variant within 1 cross-path std of full (not SE).
Phase 2: Full-train refit, single-shot OOS on sealed test; scored ONCE.

Usage
-----
    python -m stml.new_work.energy_model_comparison
    python -m stml.new_work.energy_model_comparison --phase1-only
    python -m stml.new_work.energy_model_comparison --force

Outputs: outputs/model_comparison/energy/
    cpcv_results.csv, locked_picks.csv, oos_results.csv
    {inst}_cpcv_chart.png, oos_summary_chart.png
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work.cpcv_search import CombinatorialPurgedKFold
from stml.new_work.feature_importance import build_feature_matrix, load_all_data
from stml.new_work.model_comparison import _safe_auc, _tune_fit_logistic
from stml.new_work import split_config
from stml.na_checks import native_returns, wide_returns

# ── Constants ─────────────────────────────────────────────────────────────────

CPCV_N_GROUPS = 6
CPCV_K        = 2
CPCV_EMBARGO  = 0.01
SEED          = 42

_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})

# Locked MLP hyperparameters (champion model_comparison.py run, modal fold choice)
_MLP_HIDDEN: dict[str, tuple] = {"energy_cl_ho": (64,), "energy_all": (64,)}
_MLP_ALPHA:  dict[str, float] = {"energy_cl_ho": 0.01,  "energy_all": 0.01}

# Instrument dummy values for pool columns
# energy_cl_ho pool: cl1s is reference (inst_ho1s=0)
_CL_HO_DUMMIES: dict[str, dict[str, float]] = {
    "cl1s": {"inst_ho1s": 0.0},
    "ho1s": {"inst_ho1s": 1.0},
}
# energy_all pool: cl1s is reference
_EA_DUMMIES: dict[str, dict[str, float]] = {
    "cl1s": {"inst_ho1s": 0.0, "inst_ng1s": 0.0, "inst_rb1s": 0.0},
    "ho1s": {"inst_ho1s": 1.0, "inst_ng1s": 0.0, "inst_rb1s": 0.0},
    "rb1s": {"inst_ho1s": 0.0, "inst_ng1s": 0.0, "inst_rb1s": 1.0},
    "ng1s": {"inst_ho1s": 0.0, "inst_ng1s": 1.0, "inst_rb1s": 0.0},
}

CACHE_DIR  = _HERE / "outputs" / "model_comparison" / "_cache"
ENERGY_OUT = _HERE / "outputs" / "model_comparison" / "energy"
IMP_DIR    = _HERE / "outputs" / "importance"

# Pool assignment per instrument
POOL: dict[str, str] = {
    "cl1s": "individual",
    "ho1s": "energy_cl_ho",
    "rb1s": "individual",
    "ng1s": "energy_all",
}

# Simplicity order (most regularised first → used for lock selection and charts)
SIMPLICITY: dict[str, list[str]] = {
    "cl1s": ["reduced", "pruned", "full"],
    "ho1s": ["reduced", "pruned", "full"],
    "rb1s": ["reduced_min", "reduced", "pruned", "full"],
    "ng1s": ["reduced", "pruned", "full"],
}

VARIANT_SPECS: dict[str, dict[str, dict]] = {
    "cl1s": {
        "full":   {"mode": "raw_all"},
        "pruned": {
            "mode":     "raw_clusters",
            "clusters": ["C2_f11_lowfreq_macro", "C14_f12"],
        },
        "reduced": {
            "mode":      "pca_plus_raw",
            "pca_specs": [
                {"cluster": "C2_f11_lowfreq_macro", "n_components": 1},
                {"cluster": "C14_f12",               "n_components": 1},
            ],
            "raw_features": [
                "f11_crude_stock_surprise", "f2_vol_10", "f11_china_pmi_level",
            ],
        },
    },
    "ho1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {
            "mode":     "raw_clusters",
            "clusters": ["C8_f11_lowfreq_macro", "C15_f11", "C1_f1"],
        },
        "reduced": {
            "mode":     "raw_select",
            "features": [
                "f11_vix_term_slope", "f4_pc3", "f1_bb_bandwidth_20", "f11_vix_5d_change",
            ],
        },
    },
    "rb1s": {
        "full":        {"mode": "raw_all"},
        "pruned":      {
            "mode":     "raw_clusters",
            "clusters": ["F5_signal", "C6_f11_lowfreq_macro", "C1_f1"],
        },
        "reduced":     {
            "mode":     "raw_select",
            "features": [
                "f5_signal", "f11_hy_oas_5d_change",
                "f11_dist_stock_surprise", "f2_vol_ratio_20_60",
            ],
        },
        "reduced_min": {
            "mode":     "raw_select",
            "features": ["f5_signal", "f11_hy_oas_5d_change"],
        },
    },
    "ng1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {
            "mode":     "raw_clusters",
            "clusters": ["C15_f7"],
        },
        "reduced": {
            "mode":     "raw_select",
            "features": [
                "f7_oi_change", "f7_oi_level", "f2_atr_14",
                "hmm_vol_next_turbulent", "hmm_vol_p2_turbulent", "hmm_vol_p0_calm",
            ],
        },
    },
}

# Expected champions (asserted against selection_table at startup)
_EXPECTED_CHAMPIONS: dict[str, tuple[str, str]] = {
    "cl1s": ("cl1s",         "logistic"),
    "ho1s": ("energy_cl_ho", "mlp"),
    "rb1s": ("rb1s",         "logistic"),
    "ng1s": ("energy_all",   "mlp"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


def load_cluster_membership(inst: str) -> pd.DataFrame:
    p = IMP_DIR / inst / "cluster_membership.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["cluster", "feature"])


# ── Variant transform ──────────────────────────────────────────────────────────

def transform_variant(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    feat_cols: list[str],
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_tr_v, X_te_v, names). PCA/StandardScaler always fitted on X_tr only."""
    mode = spec["mode"]

    if mode == "raw_all":
        return X_tr, X_te, list(feat_cols)

    if mode == "raw_clusters":
        members = cluster_df[
            cluster_df["cluster"].isin(spec["clusters"])
        ]["feature"].tolist()
        keep = [f for f in members if f in feat_cols]
        if not keep:
            return X_tr, X_te, list(feat_cols)
        idx = [feat_cols.index(f) for f in keep]
        return X_tr[:, idx], X_te[:, idx], keep

    if mode == "raw_select":
        keep = [f for f in spec["features"] if f in feat_cols]
        idx  = [feat_cols.index(f) for f in keep]
        return X_tr[:, idx], X_te[:, idx], keep

    if mode == "pca_plus_raw":
        parts_tr: list[np.ndarray] = []
        parts_te: list[np.ndarray] = []
        names:    list[str]        = []

        for ps in spec["pca_specs"]:
            cname   = ps["cluster"]
            n_comp  = ps["n_components"]
            members = cluster_df[cluster_df["cluster"] == cname]["feature"].tolist()
            members = [f for f in members if f in feat_cols]
            if not members:
                continue
            idx    = [feat_cols.index(f) for f in members]
            Xc_tr  = X_tr[:, idx]
            Xc_te  = X_te[:, idx]
            n_act  = min(n_comp, len(members), Xc_tr.shape[0])
            sc     = StandardScaler()
            pca    = PCA(n_components=n_act, random_state=SEED)
            pca.fit(sc.fit_transform(Xc_tr))
            parts_tr.append(pca.transform(sc.transform(Xc_tr)))
            parts_te.append(pca.transform(sc.transform(Xc_te)))
            names.extend([f"{cname}_PC{i + 1}" for i in range(n_act)])

        raw     = [f for f in spec["raw_features"] if f in feat_cols]
        raw_idx = [feat_cols.index(f) for f in raw]
        parts_tr.append(X_tr[:, raw_idx])
        parts_te.append(X_te[:, raw_idx])
        names.extend(raw)

        X_tr_v = np.hstack(parts_tr) if parts_tr else X_tr[:, []]
        X_te_v = np.hstack(parts_te) if parts_te else X_te[:, []]
        return X_tr_v, X_te_v, names

    raise ValueError(f"Unknown variant mode: {mode!r}")


# ── MLP fit with locked hyperparameters ───────────────────────────────────────

def _fit_mlp_fixed(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    pool: str,
) -> tuple[StandardScaler, MLPClassifier]:
    """Fit MLP using the champion's locked hidden/alpha. StandardScaler on X_tr."""
    hidden = _MLP_HIDDEN[pool]
    alpha  = _MLP_ALPHA[pool]
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_tr)
    model  = MLPClassifier(
        hidden_layer_sizes=hidden,
        alpha=alpha,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=SEED,
    )
    try:
        model.fit(X_sc, y_tr)
    except ValueError:
        # Fallback: validation split may become single-class on thin pools
        model = MLPClassifier(
            hidden_layer_sizes=hidden, alpha=alpha,
            activation="relu", solver="adam",
            learning_rate_init=1e-3, max_iter=300,
            early_stopping=False, random_state=SEED,
        )
        model.fit(X_sc, y_tr)
    return scaler, model


# ── CPCV — individual logistic (cl1s / rb1s) ──────────────────────────────────

def run_cpcv_individual(
    inst: str,
    variant: str,
    events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[list[float], pd.DataFrame]:
    fc      = _feat_cols(events)
    X_all   = events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_all   = events["bin"].to_numpy(dtype=int)
    ev_meta = events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO
    )
    path_aucs: list[float] = []
    oos_rows:  list[dict]  = []

    for path_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        y_tr  = y_all[tr_idx]
        y_te  = y_all[te_idx]
        ev_tr = ev_meta.iloc[tr_idx].reset_index(drop=True)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        try:
            X_tr_v, X_te_v, _ = transform_variant(
                X_all[tr_idx], X_all[te_idx], fc, spec, cluster_df
            )
            scaler, model, _ = _tune_fit_logistic(X_tr_v, y_tr, ev_tr)
            prob = model.predict_proba(scaler.transform(X_te_v))[:, 1]
        except Exception as e:
            print(f"    [{inst}/{variant}] path {path_i} failed: {e}")
            continue

        auc = _safe_auc(y_te, prob)
        if auc >= 0:
            path_aucs.append(float(auc))
        ev_te = ev_meta.iloc[te_idx].reset_index(drop=True)
        for i in range(len(te_idx)):
            oos_rows.append({
                "date":    ev_te.iloc[i]["date"],
                "y_true":  int(y_te[i]),
                "y_score": float(prob[i]),
                "path":    path_i,
            })

    return path_aucs, pd.DataFrame(oos_rows)


# ── CPCV — pooled MLP (ho1s / ng1s) ──────────────────────────────────────────

def run_cpcv_pooled_mlp(
    target_inst: str,
    variant: str,
    pool_events: pd.DataFrame,
    pool: str,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[list[float], pd.DataFrame]:
    """CPCV on full pool; score only on target_inst CPCV test slice.

    ho1s and ng1s are thin (34 and 28 train rows respectively) so many paths
    will be skipped — handled gracefully and reported.
    """
    fc       = _feat_cols(pool_events)
    X_all    = pool_events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_all    = pool_events["bin"].to_numpy(dtype=int)
    ev_meta  = pool_events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    inst_arr = pool_events["instrument"].values

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO
    )
    path_aucs: list[float] = []
    oos_rows:  list[dict]  = []
    n_skipped = 0

    for path_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        te_mask = inst_arr[te_idx] == target_inst
        te_pos  = te_idx[te_mask]
        if len(te_pos) < 2:
            n_skipped += 1
            continue
        y_te = y_all[te_pos]
        if len(np.unique(y_te)) < 2:
            n_skipped += 1
            continue
        y_tr  = y_all[tr_idx]
        ev_tr = ev_meta.iloc[tr_idx].reset_index(drop=True)
        if len(np.unique(y_tr)) < 2:
            n_skipped += 1
            continue

        try:
            X_tr_v, X_te_v, _ = transform_variant(
                X_all[tr_idx], X_all[te_pos], fc, spec, cluster_df
            )
            scaler, model = _fit_mlp_fixed(X_tr_v, y_tr, pool)
            prob = model.predict_proba(scaler.transform(X_te_v))[:, 1]
        except Exception as e:
            print(f"    [{target_inst}/{variant}] path {path_i} failed: {e}")
            n_skipped += 1
            continue

        auc = _safe_auc(y_te, prob)
        if auc >= 0:
            path_aucs.append(float(auc))
        ev_te = ev_meta.iloc[te_pos].reset_index(drop=True)
        for i in range(len(te_pos)):
            oos_rows.append({
                "date":    ev_te.iloc[i]["date"],
                "y_true":  int(y_te[i]),
                "y_score": float(prob[i]),
                "path":    path_i,
            })

    if n_skipped > 0:
        print(f"    [{target_inst}/{variant}] skipped {n_skipped}/15 paths "
              f"(thin slice / single class)")
    return path_aucs, pd.DataFrame(oos_rows)


# ── CPCV summary ──────────────────────────────────────────────────────────────

def summarise_cpcv(path_aucs: list[float], oos_df: pd.DataFrame) -> dict[str, Any]:
    if not path_aucs or oos_df.empty:
        return {"auc_mean": np.nan, "auc_std": np.nan,
                "logloss": np.nan, "brier": np.nan, "n_paths": 0}
    arr = np.array(path_aucs)
    ll = br = np.nan
    if not oos_df.empty and oos_df["y_true"].nunique() >= 2:
        try:
            ll = log_loss(oos_df["y_true"], oos_df["y_score"])
            br = brier_score_loss(oos_df["y_true"], oos_df["y_score"])
        except Exception:
            pass
    return {
        "auc_mean": float(arr.mean()),
        "auc_std":  float(arr.std()),
        "logloss":  float(ll) if not np.isnan(ll) else np.nan,
        "brier":    float(br) if not np.isnan(br) else np.nan,
        "n_paths":  len(arr),
    }


# ── Lock: simplest within 1 cross-path std of full ───────────────────────────

def select_locked_variant(cpcv_rows: list[dict], simplicity: list[str]) -> str:
    by_v  = {r["variant"]: r for r in cpcv_rows}
    full  = by_v.get("full", {})
    fm    = full.get("auc_mean", np.nan)
    fstd  = full.get("auc_std",  np.nan)
    if np.isnan(fm) or np.isnan(fstd):
        return "full"
    threshold = fm - fstd
    for v in simplicity:
        row = by_v.get(v, {})
        vm  = row.get("auc_mean", np.nan)
        if not np.isnan(vm) and vm >= threshold:
            return v
    return "full"


# ── Bootstrap AUC CI ──────────────────────────────────────────────────────────

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except Exception:
            pass
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── Phase 2 test data loading ─────────────────────────────────────────────────

def _load_test_individual(
    inst: str,
    data: dict,
    w_rets: pd.DataFrame,
    train_fc: list[str],
) -> pd.DataFrame:
    _, full_ev = build_feature_matrix(inst, data, w_rets)
    te = split_config.apply_test_mask(full_ev)
    if te.empty:
        return pd.DataFrame()
    return te.reindex(columns=list(_META) + train_fc, fill_value=0.0)


def _load_test_pooled(
    target_inst: str,
    data: dict,
    w_rets: pd.DataFrame,
    pool: str,
    pool_fc: list[str],
) -> pd.DataFrame:
    _, full_ev = build_feature_matrix(target_inst, data, w_rets)
    te = split_config.apply_test_mask(full_ev)
    if te.empty:
        return pd.DataFrame()
    dummy_map = (
        _CL_HO_DUMMIES[target_inst]
        if pool == "energy_cl_ho"
        else _EA_DUMMIES[target_inst]
    )
    for col, val in dummy_map.items():
        te[col] = val
    return te.reindex(columns=list(_META) + pool_fc, fill_value=0.0)


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def _score_probs(y_te: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    auc = _safe_auc(y_te, prob)
    ci_lo, ci_hi = bootstrap_auc_ci(y_te, prob)
    ll = br = np.nan
    if len(np.unique(y_te)) >= 2:
        try:
            ll = log_loss(y_te, prob)
            br = brier_score_loss(y_te, prob)
        except Exception:
            pass
    return {
        "auc":       float(auc) if auc >= 0 else np.nan,
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
        "logloss":   float(ll) if not np.isnan(ll) else np.nan,
        "brier":     float(br) if not np.isnan(br) else np.nan,
        "n_test":    int(len(y_te)),
    }


def _phase2_individual(
    inst: str,
    variant: str,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> dict[str, Any]:
    fc       = _feat_cols(train_events)
    X_tr_raw = train_events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_tr     = train_events["bin"].to_numpy(dtype=int)
    ev_tr    = train_events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].reset_index(drop=True)
    X_te_raw = test_events.reindex(columns=fc, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float64)
    y_te     = test_events["bin"].to_numpy(dtype=int)
    _nan: dict[str, Any] = {
        "auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan,
        "logloss": np.nan, "brier": np.nan, "n_test": int(len(y_te)),
    }
    try:
        X_tr, X_te, _ = transform_variant(X_tr_raw, X_te_raw, fc, spec, cluster_df)
        scaler, model, _ = _tune_fit_logistic(X_tr, y_tr, ev_tr)
        prob = model.predict_proba(scaler.transform(X_te))[:, 1]
    except Exception as e:
        print(f"  Phase 2 [{inst}/{variant}] failed: {e}")
        traceback.print_exc()
        return _nan
    return _score_probs(y_te, prob)


def _phase2_pooled_mlp(
    target_inst: str,
    variant: str,
    pool: str,
    pool_train: pd.DataFrame,
    test_events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> dict[str, Any]:
    fc       = _feat_cols(pool_train)
    X_tr_raw = pool_train[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_tr     = pool_train["bin"].to_numpy(dtype=int)
    X_te_raw = test_events.reindex(columns=fc, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float64)
    y_te     = test_events["bin"].to_numpy(dtype=int)
    _nan: dict[str, Any] = {
        "auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan,
        "logloss": np.nan, "brier": np.nan, "n_test": int(len(y_te)),
    }
    try:
        X_tr, X_te, _ = transform_variant(X_tr_raw, X_te_raw, fc, spec, cluster_df)
        scaler, model = _fit_mlp_fixed(X_tr, y_tr, pool)
        prob = model.predict_proba(scaler.transform(X_te))[:, 1]
    except Exception as e:
        print(f"  Phase 2 [{target_inst}/{variant}] failed: {e}")
        traceback.print_exc()
        return _nan
    return _score_probs(y_te, prob)


# ── Charts ────────────────────────────────────────────────────────────────────

def cpcv_chart(
    cpcv_rows: list[dict],
    locked: str,
    inst: str,
    simplicity: list[str],
    out_path: Path,
) -> None:
    rows     = sorted(cpcv_rows, key=lambda r: simplicity.index(r["variant"])
                      if r["variant"] in simplicity else 99)
    variants = [r["variant"] for r in rows]
    means    = [r["auc_mean"] for r in rows]
    stds     = [r.get("auc_std", 0) for r in rows]
    colors   = ["#1976D2" if v == locked else "#90CAF9" for v in variants]
    valid    = [m for m in means if not np.isnan(m)]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(max(5, len(variants) * 1.8), 4))
    ax.bar(variants, means, yerr=stds, color=colors,
           error_kw={"ecolor": "grey", "capsize": 5}, width=0.5)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="AUC = 0.5 (random)")
    full_m   = next((r["auc_mean"] for r in rows if r["variant"] == "full"), np.nan)
    full_std = next((r.get("auc_std", np.nan) for r in rows if r["variant"] == "full"), np.nan)
    if not np.isnan(full_m) and not np.isnan(full_std):
        ax.axhline(full_m - full_std, color="orange", linestyle=":",
                   linewidth=1.0, label="full − 1σ (lock threshold)")
    lo = max(0.2, min(valid) - 2.5 * max((s or 0) for s in stds))
    hi = min(1.0, max(valid) + 2.5 * max((s or 0) for s in stds))
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Variant")
    ax.set_ylabel("CPCV AUC (mean ± std across usable paths)")
    ax.set_title(f"{inst.upper()} — Phase 1 CPCV  |  locked = {locked}")
    ax.legend(fontsize=7)
    for bar, m in zip(ax.patches, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def oos_summary_chart(
    oos_rows: list[dict],
    locked_picks: dict[str, str],
    cpcv_df: pd.DataFrame,
    out_path: Path,
) -> None:
    insts    = ["cl1s", "ho1s", "rb1s", "ng1s"]
    by_inst  = {r["inst"]: r for r in oos_rows if r["variant"] == locked_picks.get(r["inst"])}
    aucs     = [by_inst.get(i, {}).get("auc",       np.nan) for i in insts]
    ci_lo    = [by_inst.get(i, {}).get("auc_ci_lo", np.nan) for i in insts]
    ci_hi    = [by_inst.get(i, {}).get("auc_ci_hi", np.nan) for i in insts]
    err_lo   = [a - lo if not (np.isnan(a) or np.isnan(lo)) else 0
                for a, lo in zip(aucs, ci_lo)]
    err_hi   = [hi - a if not (np.isnan(a) or np.isnan(hi)) else 0
                for a, hi in zip(aucs, ci_hi)]
    dev_aucs = []
    for inst in insts:
        lv  = locked_picks.get(inst, "full")
        row = cpcv_df[(cpcv_df["inst"] == inst) & (cpcv_df["variant"] == lv)]
        dev_aucs.append(float(row.iloc[0]["auc_mean"]) if not row.empty else np.nan)

    x      = np.arange(len(insts))
    labels = [f"{i}\n({locked_picks.get(i, '?')})" for i in insts]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, aucs, yerr=[err_lo, err_hi], color="#1976D2",
           error_kw={"ecolor": "grey", "capsize": 5}, width=0.35,
           label="OOS AUC ± 95% boot CI (locked)")
    ax.scatter(x, dev_aucs, marker="D", color="orange", zorder=5, s=50,
               label="CPCV dev AUC (locked)")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="AUC = 0.5")
    valid_all = [v for v in aucs + dev_aucs if not np.isnan(v)]
    if valid_all:
        ax.set_ylim(max(0.1, min(valid_all) - 0.15), min(1.0, max(valid_all) + 0.15))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_title("Energy — Phase 2 OOS (locked variants)  ◆ = CPCV dev AUC")
    ax.legend(fontsize=8)
    for xi, (a, hi) in enumerate(zip(aucs, err_hi)):
        if not np.isnan(a):
            ax.text(xi, a + hi + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(force: bool = False, phase1_only: bool = False) -> None:
    ENERGY_OUT.mkdir(parents=True, exist_ok=True)

    # ── Assert champions match selection_table ────────────────────────────────
    sel = pd.read_csv(
        _HERE / "outputs" / "model_comparison" / "selection_table.csv"
    ).set_index("instrument")
    for inst, (grp, mdl) in _EXPECTED_CHAMPIONS.items():
        actual_grp = sel.loc[inst, "best_group"]
        actual_mdl = sel.loc[inst, "best_model"]
        assert actual_grp == grp, (
            f"{inst}: expected best_group={grp!r}, got {actual_grp!r}"
        )
        assert actual_mdl == mdl, (
            f"{inst}: expected best_model={mdl!r}, got {actual_mdl!r}"
        )
    print("Selection table assertions passed.")

    # ── Load train caches ─────────────────────────────────────────────────────
    cl1s_train  = pd.read_parquet(CACHE_DIR / "cl1s_events.parquet")
    cl_ho_train = pd.read_parquet(CACHE_DIR / "energy_cl_ho_events.parquet")
    rb1s_train  = pd.read_parquet(CACHE_DIR / "rb1s_events.parquet")
    ea_train    = pd.read_parquet(CACHE_DIR / "energy_all_events.parquet")

    cl1s_fc  = _feat_cols(cl1s_train)
    cl_ho_fc = _feat_cols(cl_ho_train)
    rb1s_fc  = _feat_cols(rb1s_train)
    ea_fc    = _feat_cols(ea_train)

    print(f"  cl1s:         {len(cl1s_train)} train events, {len(cl1s_fc)} features")
    print(f"  energy_cl_ho: {len(cl_ho_train)} train events, {len(cl_ho_fc)} features"
          f"  (ho1s={int((cl_ho_train['instrument']=='ho1s').sum())} rows)")
    print(f"  rb1s:         {len(rb1s_train)} train events, {len(rb1s_fc)} features")
    print(f"  energy_all:   {len(ea_train)} train events, {len(ea_fc)} features"
          f"  (ng1s={int((ea_train['instrument']=='ng1s').sum())} rows)")

    # Pool → (train_df, feat_cols) lookup
    _pool_data = {
        "individual_cl1s": (cl1s_train,  cl1s_fc),
        "individual_rb1s": (rb1s_train,  rb1s_fc),
        "energy_cl_ho":    (cl_ho_train, cl_ho_fc),
        "energy_all":      (ea_train,    ea_fc),
    }

    cluster_dfs = {inst: load_cluster_membership(inst) for inst in VARIANT_SPECS}

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    cpcv_path   = ENERGY_OUT / "cpcv_results.csv"
    locked_path = ENERGY_OUT / "locked_picks.csv"

    if cpcv_path.exists() and not force:
        print("\n[Phase 1] Loading cached CPCV results.")
        cpcv_df = pd.read_csv(cpcv_path)
    else:
        print("\n" + "=" * 64)
        print("Phase 1 — CPCV variant scoring (TRAIN only, 15 paths)")
        print("=" * 64)
        all_rows: list[dict] = []

        for inst, spec_dict in VARIANT_SPECS.items():
            cd   = cluster_dfs[inst]
            pool = POOL[inst]
            if pool == "individual":
                train_events = cl1s_train if inst == "cl1s" else rb1s_train
            else:
                train_events = cl_ho_train if pool == "energy_cl_ho" else ea_train

            for variant, spec in spec_dict.items():
                print(f"\n  [{inst}/{variant}] …")
                if pool == "individual":
                    path_aucs, oos_df = run_cpcv_individual(
                        inst, variant, train_events, spec, cd
                    )
                else:
                    path_aucs, oos_df = run_cpcv_pooled_mlp(
                        inst, variant, train_events, pool, spec, cd
                    )
                s   = summarise_cpcv(path_aucs, oos_df)
                row = {"inst": inst, "variant": variant, **s}
                all_rows.append(row)
                print(f"    AUC {s['auc_mean']:.4f} ± {s['auc_std']:.4f}"
                      f"  n_paths={s['n_paths']}")

        cpcv_df = pd.DataFrame(all_rows)
        cpcv_df.to_csv(cpcv_path, index=False)
        print(f"\nSaved: {cpcv_path}")

    # ── Lock ──────────────────────────────────────────────────────────────────
    locked_picks: dict[str, str] = {}
    lock_rows: list[dict] = []
    for inst in VARIANT_SPECS:
        simp      = SIMPLICITY[inst]
        inst_rows = cpcv_df[cpcv_df["inst"] == inst].to_dict("records")
        locked    = select_locked_variant(inst_rows, simp)
        locked_picks[inst] = locked

        by_v   = {r["variant"]: r for r in inst_rows}
        full_r = by_v.get("full", {})
        thresh = float(full_r.get("auc_mean", np.nan)) - float(full_r.get("auc_std", np.nan))
        print(f"\n  [{inst}] LOCKED: {locked}  (threshold={thresh:.4f})")
        for v in simp + ([] if "full" in simp else ["full"]):
            r = by_v.get(v)
            if r is None:
                continue
            tag = " ← LOCKED" if v == locked else ""
            print(f"    {v:12s}  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
                  f"  n_paths={r['n_paths']}{tag}")
        lock_rows.append({"inst": inst, "locked_variant": locked})

    lock_df = pd.DataFrame(lock_rows)
    lock_df.to_csv(locked_path, index=False)
    print(f"\nLocked picks saved: {locked_path}")

    for inst in VARIANT_SPECS:
        cpcv_chart(
            cpcv_df[cpcv_df["inst"] == inst].to_dict("records"),
            locked_picks[inst], inst, SIMPLICITY[inst],
            ENERGY_OUT / f"{inst}_cpcv_chart.png",
        )

    if phase1_only:
        print("\nPhase 1 complete. --phase1-only set; stopping before test.")
        return

    # ── Phase 2 — Sealed test (scored once) ──────────────────────────────────
    oos_path = ENERGY_OUT / "oos_results.csv"
    if oos_path.exists() and not force:
        print("\n[Phase 2] Loading cached OOS results.")
        oos_df = pd.read_csv(oos_path)
    else:
        print("\n" + "=" * 64)
        print("Phase 2 — Single-shot OOS (SEALED TEST — scored once)")
        print("=" * 64)
        print("Loading full data for test events …")
        data   = load_all_data()
        rl     = native_returns(data["ohlcv"], kind="log")
        w_rets = wide_returns(rl).sort_index()

        all_oos: list[dict] = []

        for inst, spec_dict in VARIANT_SPECS.items():
            cd   = cluster_dfs[inst]
            pool = POOL[inst]
            print(f"\n  Loading test events for {inst} …")

            if pool == "individual":
                train_fc     = cl1s_fc if inst == "cl1s" else rb1s_fc
                te           = _load_test_individual(inst, data, w_rets, train_fc)
                train_events = cl1s_train if inst == "cl1s" else rb1s_train
            else:
                pool_fc      = cl_ho_fc if pool == "energy_cl_ho" else ea_fc
                te           = _load_test_pooled(inst, data, w_rets, pool, pool_fc)
                train_events = cl_ho_train if pool == "energy_cl_ho" else ea_train

            if te.empty:
                print(f"  WARNING: no test events for {inst}")
                continue
            y_te = te["bin"].to_numpy(dtype=int)
            if len(np.unique(y_te)) < 2:
                print(f"  WARNING: single-class test for {inst} — skipping")
                continue
            print(f"  {inst}: {len(te)} test events  "
                  f"(pos={int(y_te.sum())}, neg={int((1 - y_te).sum())})")

            for variant, spec in spec_dict.items():
                is_locked = locked_picks.get(inst) == variant
                tag = " [LOCKED]" if is_locked else " [diagnostic]"
                print(f"  [{inst}/{variant}]{tag} refitting …")

                if pool == "individual":
                    res = _phase2_individual(inst, variant, train_events, te, spec, cd)
                else:
                    res = _phase2_pooled_mlp(inst, variant, pool, train_events, te, spec, cd)

                row = {"inst": inst, "variant": variant, "is_locked": is_locked, **res}
                all_oos.append(row)
                print(f"    AUC {res['auc']:.4f}"
                      f"  [{res['auc_ci_lo']:.3f},{res['auc_ci_hi']:.3f}]"
                      f"  logloss={res['logloss']:.4f}"
                      f"  brier={res['brier']:.4f}"
                      f"  n_test={res['n_test']}")

        oos_df = pd.DataFrame(all_oos)
        oos_df.to_csv(oos_path, index=False)
        print(f"\nSaved: {oos_path}")

    oos_summary_chart(
        oos_df.to_dict("records"), locked_picks, cpcv_df,
        ENERGY_OUT / "oos_summary_chart.png",
    )

    print("\n" + "=" * 64)
    print("Dev-CPCV vs OOS gap (locked variants)")
    print("=" * 64)
    for inst in VARIANT_SPECS:
        locked  = locked_picks.get(inst, "full")
        c_row   = cpcv_df[(cpcv_df["inst"] == inst) & (cpcv_df["variant"] == locked)]
        o_row   = oos_df[(oos_df["inst"] == inst) & (oos_df["variant"] == locked)]
        if c_row.empty or o_row.empty:
            continue
        dev     = float(c_row.iloc[0]["auc_mean"])
        oos_auc = float(o_row.iloc[0]["auc"])
        ci_lo   = float(o_row.iloc[0]["auc_ci_lo"])
        ci_hi   = float(o_row.iloc[0]["auc_ci_hi"])
        signal  = "SIGNAL" if sel.loc[inst, "signal"] else "NO SIGNAL"
        print(f"  {inst:6s} ({locked:12s})  {signal:9s}  "
              f"dev={dev:.4f}  OOS={oos_auc:.4f}"
              f"  [{ci_lo:.3f},{ci_hi:.3f}]  gap={dev - oos_auc:+.4f}")

    print("\nDone.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Energy variant comparison (new labels)")
    ap.add_argument("--force",       action="store_true", help="Rerun even if cached")
    ap.add_argument("--phase1-only", action="store_true", help="Stop before Phase 2")
    args = ap.parse_args()
    run(force=args.force, phase1_only=args.phase1_only)
