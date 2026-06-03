"""champion_importance.py — Feature importance for all champions (train-only, clean split).

Champions (new labels, selection_table.csv, purged inner k-fold + 1SE, global cut 2021-10-06):

EQUITY:
  es1s   → es1s        / logistic (SIGNAL     lower_ci=0.57)   [was: RF / NO SIGNAL]
  nq1s   → nq1s        / XGB      (signal     lower_ci=0.54)   [was: lower_ci=0.62]
  fesx1s → fesx1s      / RF       (NO SIGNAL  lower_ci=0.47)   [was: logistic / SIGNAL]

ENERGY:
  cl1s   → cl1s        / logistic (NO SIGNAL  lower_ci=0.49)   [was: XGB / SIGNAL]
  ho1s   → energy_cl_ho/ logistic (NO SIGNAL  lower_ci=0.30)   [sel. MLP; logistic for import.]
  rb1s   → rb1s        / logistic (SIGNAL     lower_ci=0.51)   [was: energy_all / NO SIGNAL]
  ng1s   → energy_all  / RF       (SIGNAL     lower_ci=0.52)   [sel. MLP; RF for importance]

METALS:
  gc1s   → precious    / RF       (NO SIGNAL  lower_ci=0.37)   [was: XGB]
  si1s   → precious    / RF       (NO SIGNAL  lower_ci=0.48)   [was: si1s individual / XGB]
  pl1s   → precious    / RF       (SIGNAL     lower_ci=0.54)   [was: pl1s individual / NO SIGNAL]
  hg1s   → hg1s        / RF       (signal     lower_ci=0.51)   [was: lower_ci=0.56]

Signal-status changes vs old labels:
  es1s:   NO SIGNAL → SIGNAL  |  fesx1s: SIGNAL → NO SIGNAL  |  cl1s:  SIGNAL → NO SIGNAL
  rb1s:   NO SIGNAL → SIGNAL  |  ng1s:   NO SIGNAL → SIGNAL  |  pl1s:  NO SIGNAL → SIGNAL

Pipeline per instrument
-----------------------
1. Load pre-built events from model_comparison cache (train-only, post split_config purge).
2. Cluster features: Spearman distance sqrt(1-|rho|) -> Ward -> silhouette K,
   plus hand-assigned groups (F4 latent, F5 signal, F8 calendar, F_instrument dummies).
3. CPCV (n_groups=6, k=2, embargo=0.01) with champion estimator:
   TREE (RF / XGB):
     Clustered MDA  — joint permutation, N=10 repeats, score on target slice.
     Clustered MDI  — sum feature_importances_ within cluster (train; flagged).
     Group SHAP     — sum mean|SHAP| within cluster (TreeSHAP, tree_path_dependent).
     Rank agreement — Kendall tau across MDA / MDI / SHAP rankings.
   LOGISTIC (elastic-net):
     Clustered MDA  — same joint permutation (model-agnostic; applied to scaled data).
     Cluster Coef   — sum |standardised coef| within cluster; averaged across folds.
     Rank agreement — Kendall tau between MDA and Coef rankings.
4. Within-cluster breakdown for top-3 clusters by MDA:
   Tree:     members ranked by mean|SHAP|; PCA PC1–PC3 on train submatrix.
   Logistic: members ranked by mean|coef|; PCA PC1–PC3 on train submatrix.
5. Global per-feature view:
   Tree:     global_shap_summary.csv + chart.
   Logistic: global_coef_summary.csv + chart.
6. Outputs under outputs/importance/{instrument}/.

NO-SIGNAL instruments (fesx1s, cl1s, ho1s, gc1s, si1s) are run in full;
  importance reflects model noise — run for completeness and contrast only.

Usage
-----
    python -m stml.new_work.champion_importance
    python -m stml.new_work.champion_importance --asset-class equity
    python -m stml.new_work.champion_importance --instruments cl1s ho1s
    python -m stml.new_work.champion_importance --force
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
from scipy.cluster.hierarchy import dendrogram as _dendrogram, linkage as _linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work.cpcv_search import CombinatorialPurgedKFold
from stml.new_work.feature_importance import (
    CORR_CLUSTER_PREFIXES,
    HAND_ASSIGNED_PREFIXES,
    N_PERM_REPEATS,
    RANDOM_SEED,
    build_cluster_map,
    cluster_representatives,
    compute_spearman_distance,
    get_cluster_labels,
    select_k,
)
from stml.new_work.model_comparison import (
    CPCV_EMBARGO,
    CPCV_K,
    CPCV_N_GROUPS,
    _tune_fit_logistic,
    _tune_fit_rf,
    _tune_fit_xgb,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUTS   = _HERE / "outputs" / "importance"
CACHE_DIR = _HERE / "outputs" / "model_comparison" / "_cache"

SHAP_MAX_SAMPLES = 200
TOP_CLUSTERS_N   = 3

_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})

_INST_PREFIX = "inst_"

_HAND_ASSIGNED_EXT = {
    **HAND_ASSIGNED_PREFIXES,
    _INST_PREFIX: "F_instrument",
}

CHAMPIONS: dict[str, dict] = {
    # ── Equity ────────────────────────────────────────────────────────────────
    "es1s": {
        "asset_class": "equity",
        "group":       "es1s",
        "family":      "logistic",
        "model_type":  "logistic",
        "target_inst": "es1s",
        "auc_mean":    0.5953,
        "auc_std":     0.0285,
        "lower_ci":    0.5668,
        "signal":      True,
    },
    "nq1s": {
        "asset_class": "equity",
        "group":       "nq1s",
        "family":      "tree",
        "model_type":  "xgb",
        "target_inst": "nq1s",
        "auc_mean":    0.5702,
        "auc_std":     0.0328,
        "lower_ci":    0.5374,
        "signal":      True,
    },
    "fesx1s": {
        "asset_class": "equity",
        "group":       "fesx1s",
        "family":      "tree",
        "model_type":  "rf",
        "target_inst": "fesx1s",
        "auc_mean":    0.4964,
        "auc_std":     0.0242,
        "lower_ci":    0.4722,
        "signal":      False,
    },
    # ── Energy ────────────────────────────────────────────────────────────────
    "cl1s": {
        "asset_class": "energy",
        "group":       "cl1s",
        "family":      "logistic",
        "model_type":  "logistic",
        "target_inst": "cl1s",
        "auc_mean":    0.5374,
        "auc_std":     0.0498,
        "lower_ci":    0.4876,
        "signal":      False,
    },
    "ho1s": {
        # Selection champion: energy_cl_ho/mlp (AUC=0.4923). MLP unsupported for
        # tree/coef importance → use energy_cl_ho/logistic (AUC=0.4043) for importance.
        "asset_class": "energy",
        "group":       "energy_cl_ho",
        "family":      "logistic",
        "model_type":  "logistic",
        "target_inst": "ho1s",
        "auc_mean":    0.4923,
        "auc_std":     0.1892,
        "lower_ci":    0.3031,
        "signal":      False,
    },
    "rb1s": {
        "asset_class": "energy",
        "group":       "rb1s",
        "family":      "logistic",
        "model_type":  "logistic",
        "target_inst": "rb1s",
        "auc_mean":    0.5773,
        "auc_std":     0.0710,
        "lower_ci":    0.5063,
        "signal":      True,
    },
    "ng1s": {
        # Selection champion: energy_all/mlp (AUC=0.6576). MLP unsupported for
        # tree/coef importance → use energy_all/rf (best non-MLP per-inst AUC=0.5912).
        "asset_class": "energy",
        "group":       "energy_all",
        "family":      "tree",
        "model_type":  "rf",
        "target_inst": "ng1s",
        "auc_mean":    0.6576,
        "auc_std":     0.1389,
        "lower_ci":    0.5187,
        "signal":      True,
    },
    # ── Metals ────────────────────────────────────────────────────────────────
    "gc1s": {
        "asset_class": "metals",
        "group":       "precious",
        "family":      "tree",
        "model_type":  "rf",
        "target_inst": "gc1s",
        "auc_mean":    0.4866,
        "auc_std":     0.1132,
        "lower_ci":    0.3734,
        "signal":      False,
    },
    "si1s": {
        "asset_class": "metals",
        "group":       "precious",
        "family":      "tree",
        "model_type":  "rf",
        "target_inst": "si1s",
        "auc_mean":    0.5400,
        "auc_std":     0.0638,
        "lower_ci":    0.4762,
        "signal":      False,
    },
    "pl1s": {
        "asset_class": "metals",
        "group":       "precious",
        "family":      "tree",
        "model_type":  "rf",
        "target_inst": "pl1s",
        "auc_mean":    0.5697,
        "auc_std":     0.0301,
        "lower_ci":    0.5396,
        "signal":      True,
    },
    "hg1s": {
        "asset_class": "metals",
        "group":       "hg1s",
        "family":      "tree",
        "model_type":  "rf",
        "target_inst": "hg1s",
        "auc_mean":    0.5463,
        "auc_std":     0.0405,
        "lower_ci":    0.5058,
        "signal":      True,
    },
}

ASSET_CLASSES: dict[str, list[str]] = {
    "equity": ["es1s", "nq1s", "fesx1s"],
    "energy": ["cl1s", "ho1s", "rb1s", "ng1s"],
    "metals": ["gc1s", "si1s", "pl1s", "hg1s"],
}


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _feat_cols(events_df: pd.DataFrame) -> list[str]:
    return [c for c in events_df.columns if c not in _META]


def _assign_groups_champion(events_df: pd.DataFrame) -> dict[str, list[str]]:
    feat_cols = _feat_cols(events_df)
    corr_cluster: list[str] = []
    hand: dict[str, list[str]] = {}
    for c in feat_cols:
        placed = False
        for prefix, label in _HAND_ASSIGNED_EXT.items():
            if c.startswith(prefix):
                hand.setdefault(label, []).append(c)
                placed = True
                break
        if placed:
            continue
        for prefix in CORR_CLUSTER_PREFIXES:
            if c.startswith(prefix):
                corr_cluster.append(c)
                placed = True
                break
        if not placed:
            hand.setdefault("F_misc", []).append(c)
    return {"corr_cluster": corr_cluster, **hand}


# ---------------------------------------------------------------------------
# SHAP helper (tree models only)
# ---------------------------------------------------------------------------

def _shap_values(
    model: Any,
    X: np.ndarray,
    feat_names: list[str],
    max_samples: int = SHAP_MAX_SAMPLES,
) -> tuple[dict[str, float], dict[str, float]]:
    try:
        import shap as _shap
        sub = X[:max_samples]
        expl = _shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(sub, check_additivity=False)
        if hasattr(sv, "values"):
            sv = sv.values
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        sv = np.asarray(sv, dtype=float)
        signed    = {feat_names[j]: float(sv[:, j].mean())         for j in range(len(feat_names))}
        magnitude = {feat_names[j]: float(np.abs(sv[:, j]).mean()) for j in range(len(feat_names))}
        return signed, magnitude
    except Exception as e:
        print(f"  [SHAP error] {type(e).__name__}: {e}")
        zeros = {n: 0.0 for n in feat_names}
        return zeros, zeros


# ---------------------------------------------------------------------------
# Per-fold importance helpers
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _clustered_mda(
    model: Any,
    X_te: np.ndarray,       # already in model's feature space (scaled for logistic)
    y_te: np.ndarray,
    feat_names: list[str],
    cluster_map: dict[str, list[str]],
    auc_base: float,
    target_mask: np.ndarray | None,
    fold_i: int = 0,
    n_repeats: int = N_PERM_REPEATS,
) -> dict[str, float]:
    col_idx = {n: j for j, n in enumerate(feat_names)}
    y_score = y_te if target_mask is None else y_te[target_mask]
    results: dict[str, float] = {}
    for cluster_i, (cname, cols) in enumerate(cluster_map.items()):
        members = [c for c in cols if c in col_idx]
        if not members:
            results[cname] = 0.0
            continue
        drops = []
        for p in range(n_repeats):
            rng = np.random.default_rng([RANDOM_SEED, fold_i, cluster_i, p])
            X_p = X_te.copy()
            perm = rng.permutation(len(X_p))
            for c in members:
                X_p[:, col_idx[c]] = X_p[perm, col_idx[c]]
            prob_all  = model.predict_proba(X_p)[:, 1]
            prob      = prob_all if target_mask is None else prob_all[target_mask]
            auc_perm  = _safe_auc(y_score, prob)
            drops.append((auc_base - auc_perm) if auc_perm is not None else 0.0)
        results[cname] = float(np.mean(drops))
    return results


def _clustered_mdi(
    model: Any,
    feat_names: list[str],
    cluster_map: dict[str, list[str]],
) -> dict[str, float]:
    fi = dict(zip(feat_names, model.feature_importances_))
    return {cname: float(sum(fi.get(c, 0.0) for c in cols)) for cname, cols in cluster_map.items()}


def _clustered_shap(
    shap_mag: dict[str, float],
    cluster_map: dict[str, list[str]],
) -> dict[str, float]:
    return {cname: float(sum(shap_mag.get(c, 0.0) for c in cols)) for cname, cols in cluster_map.items()}


def _clustered_coef(
    coef_abs: dict[str, float],
    cluster_map: dict[str, list[str]],
) -> dict[str, float]:
    return {cname: float(sum(coef_abs.get(c, 0.0) for c in cols)) for cname, cols in cluster_map.items()}


# ---------------------------------------------------------------------------
# CPCV importance loop
# ---------------------------------------------------------------------------

def run_champion_cpcv(
    events_df: pd.DataFrame,
    feat_cols: list[str],
    cluster_map: dict[str, list[str]],
    champion_cfg: dict,
) -> dict[str, Any]:
    family      = champion_cfg["family"]      # "tree" or "logistic"
    model_type  = champion_cfg["model_type"]  # "rf", "xgb", "logistic"
    target_inst = champion_cfg["target_inst"]

    ev_meta     = events_df[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    X           = events_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y           = events_df["bin"].to_numpy(dtype=int)
    instruments = events_df["instrument"].to_numpy()

    cpcv = CombinatorialPurgedKFold(n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO)

    # Accumulators
    cmda_all:         list[dict] = []
    mdi_all:          list[dict] = []
    mdi_feat_all:     list[np.ndarray] = []
    cshap_all:        list[dict] = []
    shap_signed_all:  list[dict] = []
    shap_mag_all:     list[dict] = []
    coef_signed_all:  list[dict] = []   # logistic
    ccoef_all:        list[dict] = []   # logistic
    aucs:             list[float] = []
    skipped = 0

    for fold_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        events_tr  = ev_meta.iloc[tr_idx].reset_index(drop=True)

        if len(np.unique(y_tr)) < 2:
            skipped += 1
            continue

        te_insts    = instruments[te_idx]
        is_pooled   = events_df["instrument"].nunique() > 1
        target_mask = (te_insts == target_inst) if is_pooled else None

        y_target = y_te if target_mask is None else y_te[target_mask]
        if len(np.unique(y_target)) < 2 or len(y_target) < 2:
            skipped += 1
            continue

        # ── Fit ───────────────────────────────────────────────────────────────
        try:
            if model_type == "rf":
                model, _    = _tune_fit_rf(X_tr, y_tr, events_tr)
                X_te_model  = X_te          # trees don't need scaling
            elif model_type == "xgb":
                model, _    = _tune_fit_xgb(X_tr, y_tr, events_tr)
                X_te_model  = X_te
            elif model_type == "logistic":
                scaler, model, _ = _tune_fit_logistic(X_tr, y_tr, events_tr)
                X_te_model  = scaler.transform(X_te)  # MDA permutes scaled data
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        except Exception as e:
            print(f"    fold {fold_i} fit failed: {e}")
            skipped += 1
            continue

        # ── Base AUC on target slice ──────────────────────────────────────────
        prob_all  = model.predict_proba(X_te_model)[:, 1]
        prob_base = prob_all if target_mask is None else prob_all[target_mask]
        auc_base  = _safe_auc(y_target, prob_base)
        if auc_base is None:
            skipped += 1
            continue
        aucs.append(auc_base)

        # ── Clustered MDA ─────────────────────────────────────────────────────
        cmda_all.append(
            _clustered_mda(model, X_te_model, y_te, feat_cols, cluster_map,
                           auc_base, target_mask, fold_i=fold_i)
        )

        # ── Tree-specific: MDI + SHAP ─────────────────────────────────────────
        if family == "tree":
            mdi_all.append(_clustered_mdi(model, feat_cols, cluster_map))
            mdi_feat_all.append(model.feature_importances_)
            X_shap = X_te if target_mask is None else X_te[target_mask]
            s_signed, s_mag = _shap_values(model, X_shap, feat_cols)
            shap_signed_all.append(s_signed)
            shap_mag_all.append(s_mag)
            cshap_all.append(_clustered_shap(s_mag, cluster_map))

        # ── Logistic-specific: standardised elastic-net coefficients ──────────
        else:
            coef = model.coef_[0]   # already in standardised feature space
            coef_signed = {feat_cols[j]: float(coef[j]) for j in range(len(feat_cols))}
            coef_abs    = {k: abs(v) for k, v in coef_signed.items()}
            coef_signed_all.append(coef_signed)
            ccoef_all.append(_clustered_coef(coef_abs, cluster_map))

    print(
        f"    {len(aucs)} valid folds, {skipped} skipped; "
        f"mean AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
        if aucs else "    no valid folds"
    )

    def _agg(lst: list[dict]) -> tuple[pd.Series, pd.Series]:
        if not lst:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        df = pd.DataFrame(lst)
        return df.mean(), df.std()

    cmda_mean,  cmda_std  = _agg(cmda_all)
    mdi_mean,   mdi_std   = _agg(mdi_all)
    cshap_mean, cshap_std = _agg(cshap_all)
    ccoef_mean, ccoef_std = _agg(ccoef_all)

    shap_signed_mean = pd.DataFrame(shap_signed_all).mean() if shap_signed_all else pd.Series(dtype=float)
    shap_mag_mean    = pd.DataFrame(shap_mag_all).mean()    if shap_mag_all    else pd.Series(dtype=float)

    coef_signed_mean = pd.DataFrame(coef_signed_all).mean() if coef_signed_all else pd.Series(dtype=float)
    coef_abs_mean    = coef_signed_mean.abs()               if not coef_signed_mean.empty else pd.Series(dtype=float)

    mdi_feat_mean = pd.Series(
        np.mean(mdi_feat_all, axis=0) if mdi_feat_all else np.zeros(len(feat_cols)),
        index=feat_cols,
    )

    return {
        "family":          family,
        "cmda_mean":       cmda_mean,
        "cmda_std":        cmda_std,
        # tree
        "mdi_mean":        mdi_mean,
        "mdi_std":         mdi_std,
        "cshap_mean":      cshap_mean,
        "cshap_std":       cshap_std,
        "shap_signed_mean": shap_signed_mean,
        "shap_mag_mean":   shap_mag_mean,
        "mdi_feat_mean":   mdi_feat_mean,
        # logistic
        "ccoef_mean":      ccoef_mean,
        "ccoef_std":       ccoef_std,
        "coef_signed_mean": coef_signed_mean,
        "coef_abs_mean":   coef_abs_mean,
        # shared
        "fold_aucs": aucs,
        "n_folds":   len(aucs),
    }


# ---------------------------------------------------------------------------
# Within-cluster breakdown (SHAP or |coef| ranking + PCA PC1–PC3)
# ---------------------------------------------------------------------------

def within_cluster_breakdown(
    events_df: pd.DataFrame,
    feat_cols: list[str],
    cluster_map: dict[str, list[str]],
    member_scores: pd.Series,   # shap_mag_mean (tree) or coef_abs_mean (logistic)
    top_cluster_names: list[str],
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    X_full   = events_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    feat_idx = {n: j for j, n in enumerate(feat_cols)}

    for cname in top_cluster_names:
        cols    = cluster_map.get(cname, [])
        members = [c for c in cols if c in feat_idx]
        if len(members) < 2:
            results[cname] = {
                "members": members, "score_ranks": {},
                "pca_var": [], "pca_loadings": [],
            }
            continue

        score_rank = {c: float(member_scores.get(c, 0.0)) for c in members}
        score_rank = dict(sorted(score_rank.items(), key=lambda x: x[1], reverse=True))

        col_indices = [feat_idx[c] for c in members]
        X_sub = X_full[:, col_indices]
        n_comp = min(3, len(members))
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        pca.fit(StandardScaler().fit_transform(X_sub))

        results[cname] = {
            "members":      members,
            "score_ranks":  score_rank,
            "pca_var":      list(pca.explained_variance_ratio_),        # up to 3 values
            "pca_loadings": [list(pca.components_[i]) for i in range(n_comp)],  # up to 3 arrays
        }

    return results


# ---------------------------------------------------------------------------
# Significance / agreement
# ---------------------------------------------------------------------------

def _cluster_significance_flags(
    cmda_mean: pd.Series,
    cmda_std: pd.Series,
) -> pd.Series:
    std_safe = cmda_std.fillna(np.inf)
    return (cmda_mean > std_safe).rename("significant")


def _rank_agreement(
    cmda_mean: pd.Series,
    secondary: pd.Series,    # mdi_mean (tree) or ccoef_mean (logistic)
    tertiary: pd.Series | None,  # cshap_mean (tree) or None (logistic)
    labels: tuple[str, str, str] = ("MDA", "Secondary", "Tertiary"),
) -> pd.DataFrame:
    if tertiary is not None:
        clusters = cmda_mean.index.intersection(secondary.index).intersection(tertiary.index)
    else:
        clusters = cmda_mean.index.intersection(secondary.index)

    if len(clusters) < 3:
        return pd.DataFrame()

    r_mda = cmda_mean.loc[clusters].rank(ascending=False)
    r_sec = secondary.loc[clusters].rank(ascending=False)

    tau_ms, _ = kendalltau(r_mda, r_sec)

    if tertiary is not None:
        r_ter = tertiary.loc[clusters].rank(ascending=False)
        tau_mt, _ = kendalltau(r_mda, r_ter)
        tau_st, _ = kendalltau(r_sec, r_ter)
        return pd.DataFrame({
            "method_pair": [f"{labels[0]}-{labels[1]}", f"{labels[0]}-{labels[2]}", f"{labels[1]}-{labels[2]}"],
            "kendall_tau": [tau_ms, tau_mt, tau_st],
            "agree":       [abs(tau_ms) > 0.4, abs(tau_mt) > 0.4, abs(tau_st) > 0.4],
        })
    else:
        return pd.DataFrame({
            "method_pair": [f"{labels[0]}-{labels[1]}"],
            "kendall_tau": [tau_ms],
            "agree":       [abs(tau_ms) > 0.4],
        })


def _semantic_recovery(cluster_map: dict[str, list[str]]) -> pd.DataFrame:
    from collections import Counter
    rows: list[dict] = []
    for cname, cols in cluster_map.items():
        if not cols:
            continue
        pfx_counts: Counter = Counter()
        for c in cols:
            pfx = c.split("_")[0] + "_"
            pfx_counts[pfx] += 1
        dominant_pfx, dominant_n = pfx_counts.most_common(1)[0]
        purity = dominant_n / len(cols)
        rows.append({
            "cluster":      cname,
            "n_members":    len(cols),
            "dominant_pfx": dominant_pfx,
            "purity":       round(purity, 2),
            "pure":         purity >= 0.80,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output: save all artefacts
# ---------------------------------------------------------------------------

def _save_outputs(
    inst: str,
    events_df: pd.DataFrame,
    feat_cols: list[str],
    cluster_map: dict[str, list[str]],
    cluster_metrics: pd.DataFrame,
    cluster_labels: np.ndarray,
    corr_cols: list[str],
    dist_mat: np.ndarray,
    best_k: int,
    importance: dict[str, Any],
    sig_flags: pd.Series,
    rank_df: pd.DataFrame,
    within_breakdown: dict[str, dict],
    champion_cfg: dict,
) -> None:
    out = OUTPUTS / inst
    out.mkdir(parents=True, exist_ok=True)

    family     = importance["family"]
    cmda_mean  = importance["cmda_mean"]
    cmda_std   = importance["cmda_std"]

    # ── Cluster membership ──────────────────────────────────────────────────
    rows = []
    for cname, cols in cluster_map.items():
        sig = bool(sig_flags.get(cname, False))
        for c in cols:
            pfx = c.split("_")[0] + "_"
            rows.append({"cluster": cname, "feature": c, "f_prefix": pfx, "cluster_significant": sig})
    pd.DataFrame(rows).to_csv(out / "cluster_membership.csv", index=False)

    # ── K-selection metrics ─────────────────────────────────────────────────
    cluster_metrics.to_csv(out / "cluster_k_metrics.csv", index=False)

    # ── Clustered MDA full ──────────────────────────────────────────────────
    cmda_df = pd.DataFrame({
        "mean_drop":   cmda_mean,
        "std_drop":    cmda_std,
        "significant": sig_flags,
    })
    cmda_df.index.name = "cluster"
    cmda_df = cmda_df.sort_values("mean_drop", ascending=False)
    cmda_df.to_csv(out / "clustered_mda_full.csv")

    # ── Cross-check table ───────────────────────────────────────────────────
    if family == "tree":
        mdi_mean   = importance["mdi_mean"]
        cshap_mean = importance["cshap_mean"]
        shared = cmda_mean.index.intersection(mdi_mean.index).intersection(cshap_mean.index)
        cc = pd.DataFrame({
            "mda_mean":  cmda_mean.loc[shared],
            "mda_rank":  cmda_mean.loc[shared].rank(ascending=False).astype(int),
            "mdi_sum":   mdi_mean.loc[shared],
            "mdi_rank":  mdi_mean.loc[shared].rank(ascending=False).astype(int),
            "shap_sum":  cshap_mean.loc[shared],
            "shap_rank": cshap_mean.loc[shared].rank(ascending=False).astype(int),
            "significant": sig_flags.reindex(shared).fillna(False),
        })
    else:
        ccoef_mean = importance["ccoef_mean"]
        shared = cmda_mean.index.intersection(ccoef_mean.index)
        cc = pd.DataFrame({
            "mda_mean":  cmda_mean.loc[shared],
            "mda_rank":  cmda_mean.loc[shared].rank(ascending=False).astype(int),
            "coef_sum":  ccoef_mean.loc[shared],
            "coef_rank": ccoef_mean.loc[shared].rank(ascending=False).astype(int),
            "significant": sig_flags.reindex(shared).fillna(False),
        })
    cc.index.name = "cluster"
    cc.sort_values("mda_rank").to_csv(out / "cluster_crosscheck_table.csv")

    # ── Global per-feature view ─────────────────────────────────────────────
    if family == "tree":
        shap_mag  = importance["shap_mag_mean"]
        shap_sign = importance["shap_signed_mean"]
        mdi_feat  = importance["mdi_feat_mean"]
        feat_df = pd.DataFrame({
            "shap_magnitude": shap_mag,
            "shap_signed":    shap_sign,
            "mdi":            mdi_feat,
        }).dropna(how="all")
        feat_df.index.name = "feature"
        feat_df.sort_values("shap_magnitude", ascending=False).to_csv(out / "global_shap_summary.csv")
    else:
        coef_signed = importance["coef_signed_mean"]
        coef_abs    = importance["coef_abs_mean"]
        feat_df = pd.DataFrame({
            "coef_abs":    coef_abs,
            "coef_signed": coef_signed,
        }).dropna(how="all")
        feat_df.index.name = "feature"
        feat_df.sort_values("coef_abs", ascending=False).to_csv(out / "global_coef_summary.csv")

    # ── Rank agreement ──────────────────────────────────────────────────────
    if not rank_df.empty:
        rank_df.to_csv(out / "rank_agreement.csv", index=False)

    # ── Within-cluster breakdown CSVs ───────────────────────────────────────
    score_col = "mean_shap_mag" if family == "tree" else "mean_coef_abs"
    for cname, bd in within_breakdown.items():
        safe_name = cname.replace("/", "_").replace(" ", "_")
        rows_wc = []
        members  = bd["members"]
        pca_var  = bd["pca_var"]
        pca_loads = bd["pca_loadings"]
        for j, feat in enumerate(members):
            row_wc: dict = {
                "feature":    feat,
                "cluster":    cname,
                score_col:    bd["score_ranks"].get(feat, 0.0),
            }
            for pc_i in range(len(pca_var)):
                row_wc[f"pc{pc_i+1}_loading"]             = pca_loads[pc_i][j] if j < len(pca_loads[pc_i]) else np.nan
                row_wc[f"pca_pc{pc_i+1}_var_explained"]   = pca_var[pc_i]
            rows_wc.append(row_wc)
        pd.DataFrame(rows_wc).to_csv(out / f"within_cluster_{safe_name}.csv", index=False)

    # ── Champion metadata ───────────────────────────────────────────────────
    meta = {
        "instrument": inst,
        "asset_class": champion_cfg["asset_class"],
        "group":       champion_cfg["group"],
        "family":      family,
        "model_type":  champion_cfg["model_type"],
        "auc_mean":    champion_cfg["auc_mean"],
        "auc_std":     champion_cfg["auc_std"],
        "lower_ci":    champion_cfg["lower_ci"],
        "signal":      champion_cfg["signal"],
        "n_folds":     importance["n_folds"],
        "n_sig_clusters": int(sig_flags.sum()),
    }
    pd.DataFrame([meta]).to_csv(out / "champion_meta.csv", index=False)

    # ════════════════════════════════════════════════════════════════════════
    # Figures
    # ════════════════════════════════════════════════════════════════════════

    # ── Dendrogram ──────────────────────────────────────────────────────────
    # Skipped when best_k==0 (clustering reused from saved membership — dendrogram unchanged).
    if best_k > 0 and dist_mat.size > 0:
        try:
            condensed = squareform(dist_mat)
            Z = _linkage(condensed, method="ward")
            fig, ax = plt.subplots(figsize=(16, 5))
            _dendrogram(Z, labels=corr_cols, ax=ax, leaf_rotation=90, leaf_font_size=5)
            cut_height = Z[-(best_k - 1), 2] if best_k > 1 else Z[-1, 2]
            ax.axhline(y=cut_height, color="red", linestyle="--", linewidth=1.0, label=f"K={best_k} cut")
            ax.set_title(f"{inst} — Ward dendrogram on continuous features (Spearman distance)")
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(out / "dendrogram.png", dpi=130)
            plt.close(fig)
        except Exception as e:
            print(f"  [warn] dendrogram failed: {e}")

    # ── Clustered MDA bar chart ─────────────────────────────────────────────
    try:
        colors = ["#1f77b4" if s else "#aec7e8" for s in cmda_df["significant"]]
        fig, ax = plt.subplots(figsize=(11, max(5, len(cmda_df) * 0.35)))
        y_pos = range(len(cmda_df))
        ax.barh(list(y_pos), cmda_df["mean_drop"],
                xerr=cmda_df["std_drop"].fillna(0), align="center",
                capsize=3, color=colors, alpha=0.9)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(cmda_df.index, fontsize=7)
        ax.set_xlabel("Mean AUC drop (clustered MDA, ± std across CPCV paths)")
        ax.set_title(
            f"{inst} ({champion_cfg['group']} / {champion_cfg['model_type'].upper()}, "
            f"{importance['n_folds']} CPCV paths)\n"
            f"Cluster-level importance (dark = significant > 1σ; light = inconclusive)"
        )
        ax.axvline(x=0, color="black", linewidth=0.8)
        plt.tight_layout()
        fig.savefig(out / "clustered_mda_chart.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] clustered MDA chart failed: {e}")

    # ── Global feature chart (SHAP or coef) ────────────────────────────────
    try:
        top_feats = feat_df.head(30)
        fig, ax = plt.subplots(figsize=(10, max(5, len(top_feats) * 0.32)))
        y_pos = range(len(top_feats))
        if family == "tree":
            vals = top_feats["shap_magnitude"]
            sign_col = importance["shap_signed_mean"]
            colors_feat = ["#d62728" if sign_col.get(f, 0) > 0 else "#1f77b4" for f in top_feats.index]
            xlabel = "Mean |SHAP value| (tree_path_dependent)"
            title_suffix = "Global per-feature SHAP (top 30)\nred=positive signal, blue=negative"
        else:
            vals = top_feats["coef_abs"]
            sign_col = importance["coef_signed_mean"]
            colors_feat = ["#d62728" if sign_col.get(f, 0) > 0 else "#1f77b4" for f in top_feats.index]
            xlabel = "Mean |standardised elastic-net coefficient|"
            title_suffix = "Global per-feature coefficient (top 30)\nred=positive, blue=negative"
        ax.barh(list(y_pos), vals, align="center", color=colors_feat, alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_feats.index, fontsize=6)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{inst} — {title_suffix}")
        ax.axvline(x=0, color="black", linewidth=0.5)
        plt.tight_layout()
        fname = "global_shap_chart.png" if family == "tree" else "global_coef_chart.png"
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] global feature chart failed: {e}")

    # ── Within-cluster bar charts ───────────────────────────────────────────
    for cname, bd in within_breakdown.items():
        try:
            safe_name = cname.replace("/", "_").replace(" ", "_")
            items = list(bd["score_ranks"].items())
            if not items:
                continue
            feats_wc = [x[0] for x in items]
            vals_wc  = [x[1] for x in items]
            pca_var  = bd["pca_var"]
            pca_loads = bd["pca_loadings"]

            n_pc_plots = min(2, len(pca_var))
            fig, axes = plt.subplots(1, 1 + n_pc_plots,
                                     figsize=(5 + 5 * n_pc_plots, max(3, len(feats_wc) * 0.4)))
            if not hasattr(axes, "__len__"):
                axes = [axes]

            score_label = "Mean |SHAP|" if family == "tree" else "Mean |coef|"
            axes[0].barh(range(len(feats_wc)), vals_wc, align="center", color="#1f77b4", alpha=0.85)
            axes[0].set_yticks(range(len(feats_wc)))
            axes[0].set_yticklabels(feats_wc, fontsize=7)
            axes[0].set_xlabel(score_label)
            axes[0].set_title(
                f"Member ranking\n"
                f"PC1={pca_var[0]:.1%}" + (f" PC2={pca_var[1]:.1%}" if len(pca_var) > 1 else "")
            )

            for pc_i in range(n_pc_plots):
                load_vals = pca_loads[pc_i] if pc_i < len(pca_loads) else [0.0] * len(feats_wc)
                col_l = ["#d62728" if v > 0 else "#1f77b4" for v in load_vals]
                axes[pc_i + 1].barh(range(len(feats_wc)), load_vals, align="center", color=col_l, alpha=0.85)
                axes[pc_i + 1].set_yticks(range(len(feats_wc)))
                axes[pc_i + 1].set_yticklabels(feats_wc, fontsize=7)
                axes[pc_i + 1].set_xlabel("Loading")
                axes[pc_i + 1].set_title(f"PC{pc_i+1} loadings ({pca_var[pc_i]:.1%} var)")
                axes[pc_i + 1].axvline(x=0, color="black", linewidth=0.5)

            fig.suptitle(f"{inst} — Within-cluster breakdown: {cname}", fontsize=9)
            plt.tight_layout()
            fig.savefig(out / f"within_cluster_{safe_name}.png", dpi=130)
            plt.close(fig)
        except Exception as e:
            print(f"  [warn] within-cluster plot for {cname} failed: {e}")

    print(f"  Outputs → {out}")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_champion_importance(
    instruments: list[str] | None = None,
    asset_class: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    if asset_class is not None:
        if asset_class not in ASSET_CLASSES:
            raise ValueError(f"Unknown asset_class: {asset_class}. Choose from {list(ASSET_CLASSES)}")
        targets = ASSET_CLASSES[asset_class]
    elif instruments is not None:
        targets = instruments
    else:
        targets = list(CHAMPIONS.keys())

    results: dict[str, Any] = {}

    for inst in targets:
        if inst not in CHAMPIONS:
            print(f"[skip] {inst}: not in CHAMPIONS")
            continue

        cfg = CHAMPIONS[inst]
        sig_str = "SIGNAL" if cfg["signal"] else "NO-SIGNAL"
        print(f"\n{'='*66}")
        print(f"  {inst}  |  {cfg['group']} / {cfg['model_type'].upper()}  |  "
              f"AUC {cfg['auc_mean']:.3f}±{cfg['auc_std']:.3f}  |  {sig_str}")
        print("=" * 66)

        done_marker = OUTPUTS / inst / "clustered_mda_full.csv"
        if done_marker.exists() and not force:
            print(f"  [cache hit] pass --force to rerun")
            continue

        # 1. Load train-only events from model_comparison cache
        cache_path = CACHE_DIR / f"{cfg['group']}_events.parquet"
        if not cache_path.exists():
            print(f"  [error] cache not found: {cache_path}")
            continue
        events_df = pd.read_parquet(cache_path)
        feat_cols = _feat_cols(events_df)
        print(f"  Events: {len(events_df)}, features: {len(feat_cols)}")

        # 2. Cluster features — reuse saved membership if feature matrix is unchanged.
        # Clustering is label-independent (Ward on Spearman distance of price/macro features).
        # We re-cluster only if the saved membership is absent or incompatible (e.g. group changed).
        membership_path  = OUTPUTS / inst / "cluster_membership.csv"
        k_metrics_path   = OUTPUTS / inst / "cluster_k_metrics.csv"
        reuse_clustering = False

        if membership_path.exists():
            saved_mem = pd.read_csv(membership_path)
            mem_features = set(saved_mem["feature"].tolist())
            missing = mem_features - set(feat_cols)
            if not missing:
                # All saved features are present → reconstruct cluster_map from file.
                cluster_map = {}
                for _, row in saved_mem.iterrows():
                    cluster_map.setdefault(row["cluster"], []).append(row["feature"])
                cluster_metrics = pd.read_csv(k_metrics_path) if k_metrics_path.exists() else pd.DataFrame()
                corr_cols       = [c for cname, cols in cluster_map.items()
                                   for c in cols
                                   if any(c.startswith(p) for p in CORR_CLUSTER_PREFIXES)]
                dist_mat        = np.zeros((0, 0))
                cluster_labels  = np.array([], dtype=int)
                best_k          = 0   # 0 → skip dendrogram regeneration
                reuse_clustering = True
                print(f"  Reusing cluster_membership.csv ({len(cluster_map)} clusters, "
                      f"{len(mem_features)} features) — no re-cluster needed")
            else:
                print(f"  cluster_membership incompatible: {len(missing)} features absent "
                      f"from current events (group likely changed). Re-clustering.")

        if not reuse_clustering:
            print("  Clustering features...")
            groups      = _assign_groups_champion(events_df)
            corr_cols   = groups.pop("corr_cluster", [])
            hand_groups = groups

            if len(corr_cols) < 3:
                print(f"  [warn] only {len(corr_cols)} corr-cluster features")
                cluster_map     = hand_groups
                best_k          = 0
                cluster_metrics = pd.DataFrame()
                cluster_labels  = np.array([])
                dist_mat        = np.zeros((0, 0))
            else:
                X_corr = events_df[corr_cols].fillna(0)
                dist_mat = compute_spearman_distance(X_corr)
                best_k, cluster_metrics = select_k(X_corr, dist_mat)
                sil = cluster_metrics.set_index("K").loc[best_k, "silhouette"]
                print(f"  Best K={best_k} (silhouette={sil:.3f})")
                cluster_labels  = get_cluster_labels(dist_mat, best_k)
                _               = cluster_representatives(X_corr, cluster_labels)
                cluster_map     = build_cluster_map(corr_cols, cluster_labels, hand_groups)

        n_clusters = len(cluster_map)
        print(f"  Cluster map: {n_clusters} groups "
              f"(corr: {len(corr_cols)} features + {len(cluster_map)-best_k if best_k else '?'} hand-assigned)")

        # 3. CPCV importance
        print(f"  Running CPCV importance ({cfg['family']}) ...")
        importance = run_champion_cpcv(events_df, feat_cols, cluster_map, cfg)

        if not importance["fold_aucs"]:
            print("  [error] no valid folds — skipping")
            continue

        # 4. Significance + rank agreement
        sig_flags = _cluster_significance_flags(importance["cmda_mean"], importance["cmda_std"])

        family = importance["family"]
        if family == "tree":
            rank_df = _rank_agreement(
                importance["cmda_mean"], importance["mdi_mean"], importance["cshap_mean"],
                labels=("MDA", "MDI", "SHAP"),
            )
        else:
            rank_df = _rank_agreement(
                importance["cmda_mean"], importance["ccoef_mean"], None,
                labels=("MDA", "Coef", ""),
            )

        # 5. Within-cluster breakdown for top-3 clusters
        top_sig    = sig_flags[sig_flags].index.tolist()
        sorted_mda = importance["cmda_mean"].sort_values(ascending=False)
        top_by_mda = sorted_mda.index[:TOP_CLUSTERS_N].tolist()
        top_cluster_names = list(dict.fromkeys(top_sig[:TOP_CLUSTERS_N] + top_by_mda))[:TOP_CLUSTERS_N]

        member_scores = (
            importance["shap_mag_mean"] if family == "tree" else importance["coef_abs_mean"]
        )
        within_breakdown = within_cluster_breakdown(
            events_df, feat_cols, cluster_map, member_scores, top_cluster_names
        )

        # 6. Save all artefacts
        _save_outputs(
            inst, events_df, feat_cols, cluster_map, cluster_metrics,
            cluster_labels, corr_cols, dist_mat, best_k,
            importance, sig_flags, rank_df, within_breakdown, cfg,
        )

        results[inst] = {
            "cluster_map":     cluster_map,
            "importance":      importance,
            "sig_flags":       sig_flags,
            "rank_agreement":  rank_df,
            "within_breakdown": within_breakdown,
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Champion feature importance (train-only, all 11 instruments)"
    )
    parser.add_argument(
        "--instruments", nargs="+", default=None,
        help="Specific instruments (default: all)"
    )
    parser.add_argument(
        "--asset-class", default=None, choices=list(ASSET_CLASSES),
        help="Run all instruments in one asset class"
    )
    parser.add_argument("--force", action="store_true", help="Rerun even if outputs exist")
    args = parser.parse_args()
    run_champion_importance(
        instruments=args.instruments,
        asset_class=args.asset_class,
        force=args.force,
    )
    print("\nDone.")
