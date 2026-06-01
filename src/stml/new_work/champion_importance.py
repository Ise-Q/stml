"""champion_importance.py — Feature importance for the four signal-bearing champions.

Champions (from selection_table_v2.csv):
  cl1s  → individual cl1s,        XGB
  es1s  → individual es1s,        RF
  ho1s  → pooled energy_cl_ho,    RF   (score on ho1s test slice only)
  rb1s  → pooled energy_all,      XGB  (score on rb1s test slice only)

Pipeline per champion
---------------------
1. Load pre-built events from model_comparison cache (same hygiene / feature set).
2. Cluster features: reuse feature_importance.py clustering unchanged.
   Extended hand-assigned groups: F4 latent, F5 signal, F8 calendar, F_instrument dummies.
3. CPCV (n_groups=6, k=2, embargo=0.01) with champion estimator:
   - Clustered MDA : jointly permute entire cluster per fold; score on target
                     instrument slice for pooled champions.
   - Clustered MDI : sum feature_importances_ within cluster (train-set statistic;
                     flagged as such).
   - Group SHAP    : sum mean|SHAP| within cluster (TreeSHAP, tree_path_dependent).
   Aggregate mean ± std across CPCV paths.
4. Flag: clusters within 1σ of zero → inconclusive.
         rank disagreement across methods → inconsistent.
5. Within-cluster breakdown for top-3 significant clusters:
   - Rank members by mean|SHAP| (primary).
   - PCA on cluster submatrix: PC1 variance explained + top loadings.
6. Global per-feature SHAP summary + MDI (correlation-problem demonstration).
7. Outputs under outputs/importance/{instrument}/.

Usage
-----
    python -m stml.new_work.champion_importance
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
    _tune_fit_rf,
    _tune_fit_xgb,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUTS = _HERE / "outputs" / "importance"
CACHE_DIR = _HERE / "outputs" / "model_comparison" / "_cache"

SHAP_MAX_SAMPLES = 200
TOP_CLUSTERS_N = 3      # within-cluster breakdown depth

_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})

# Instrument-dummy prefix for pooled models (one-hot added by assemble_group)
_INST_PREFIX = "inst_"

# Extended hand-assigned groups; inst_ dummies measured as a group cluster
_HAND_ASSIGNED_EXT = {
    **HAND_ASSIGNED_PREFIXES,       # f4_, f5_, f8_
    _INST_PREFIX: "F_instrument",
}

CHAMPIONS: dict[str, dict] = {
    "cl1s": {
        "group":       "cl1s",
        "model_type":  "xgb",
        "target_inst": "cl1s",
        "notes":       "strong signal, lower CI >> 0.5",
    },
    "es1s": {
        "group":       "es1s",
        "model_type":  "rf",
        "target_inst": "es1s",
        "notes":       "marginal signal, lower CI just clears 0.5 — interpret with caution",
    },
    "ho1s": {
        "group":       "energy_cl_ho",
        "model_type":  "rf",
        "target_inst": "ho1s",
        "notes":       "strong signal but thin; single-class folds dropped; treat as indicative",
    },
    "rb1s": {
        "group":       "energy_all",
        "model_type":  "xgb",
        "target_inst": "rb1s",
        "notes":       "marginal signal, lower CI just clears 0.5 — interpret with caution",
    },
}


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _feat_cols(events_df: pd.DataFrame) -> list[str]:
    return [c for c in events_df.columns if c not in _META]


def _assign_groups_champion(events_df: pd.DataFrame) -> dict[str, list[str]]:
    """Partition feature columns into corr-cluster block and hand-assigned groups.

    Extends the standard partition with an F_instrument group for pooled model dummies.
    """
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
# SHAP helper
# ---------------------------------------------------------------------------

def _shap_values(
    model: Any,
    X: np.ndarray,
    feat_names: list[str],
    max_samples: int = SHAP_MAX_SAMPLES,
) -> tuple[dict[str, float], dict[str, float]]:
    """TreeSHAP with tree_path_dependent. Returns (signed_mean, magnitude_mean)."""
    try:
        import shap as _shap

        sub = X[:max_samples]
        expl = _shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(sub, check_additivity=False)

        # Normalise output shape across SHAP / estimator versions
        if hasattr(sv, "values"):
            sv = sv.values
        if isinstance(sv, list):
            sv = sv[1]            # binary: take class-1 array
        elif sv.ndim == 3:
            sv = sv[:, :, 1]      # (samples, features, classes)
        sv = np.asarray(sv, dtype=float)  # (samples, features)

        signed    = {feat_names[j]: float(sv[:, j].mean())         for j in range(len(feat_names))}
        magnitude = {feat_names[j]: float(np.abs(sv[:, j]).mean()) for j in range(len(feat_names))}
        return signed, magnitude
    except Exception as e:
        print(f"  [SHAP error] {type(e).__name__}: {e}")
        zeros = {n: 0.0 for n in feat_names}
        return zeros, zeros


# ---------------------------------------------------------------------------
# Per-fold importance
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
    X_te: np.ndarray,
    y_te: np.ndarray,
    feat_names: list[str],
    cluster_map: dict[str, list[str]],
    auc_base: float,
    target_mask: np.ndarray | None,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Jointly permute each cluster; score drop on target_mask slice (or all rows)."""
    col_idx = {n: j for j, n in enumerate(feat_names)}
    results: dict[str, float] = {}

    y_score = y_te if target_mask is None else y_te[target_mask]

    for cname, cols in cluster_map.items():
        members = [c for c in cols if c in col_idx]
        if not members:
            results[cname] = 0.0
            continue

        X_p = X_te.copy()
        perm = rng.permutation(len(X_p))   # one shared permutation for all members
        for c in members:
            X_p[:, col_idx[c]] = X_p[perm, col_idx[c]]

        prob_all = model.predict_proba(X_p)[:, 1]
        prob = prob_all if target_mask is None else prob_all[target_mask]
        auc_perm = _safe_auc(y_score, prob)
        results[cname] = (auc_base - auc_perm) if auc_perm is not None else 0.0

    return results


def _clustered_mdi(
    model: Any,
    feat_names: list[str],
    cluster_map: dict[str, list[str]],
) -> dict[str, float]:
    """Sum MDI (feature_importances_) within each cluster."""
    fi = dict(zip(feat_names, model.feature_importances_))
    return {
        cname: float(sum(fi.get(c, 0.0) for c in cols))
        for cname, cols in cluster_map.items()
    }


def _clustered_shap(
    shap_mag: dict[str, float],
    cluster_map: dict[str, list[str]],
) -> dict[str, float]:
    """Sum per-feature mean|SHAP| within each cluster."""
    return {
        cname: float(sum(shap_mag.get(c, 0.0) for c in cols))
        for cname, cols in cluster_map.items()
    }


# ---------------------------------------------------------------------------
# CPCV loop
# ---------------------------------------------------------------------------

def run_champion_cpcv(
    events_df: pd.DataFrame,
    feat_cols: list[str],
    cluster_map: dict[str, list[str]],
    champion_cfg: dict,
) -> dict[str, Any]:
    """Run CPCV with the champion estimator; return aggregated importance dicts."""
    model_type  = champion_cfg["model_type"]
    target_inst = champion_cfg["target_inst"]

    ev_meta = events_df[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    X = events_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = events_df["bin"].to_numpy(dtype=int)
    instruments = events_df["instrument"].to_numpy()

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO
    )

    cmda_all: list[dict] = []
    mdi_all:  list[dict] = []       # cluster-level MDI
    mdi_feat_all: list[np.ndarray] = []  # per-feature MDI (raw feature_importances_)
    cshap_all: list[dict] = []
    shap_signed_all: list[dict] = []
    shap_mag_all: list[dict] = []
    aucs: list[float] = []
    skipped = 0
    rng = np.random.default_rng(RANDOM_SEED)

    for fold_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        events_tr  = ev_meta.iloc[tr_idx].reset_index(drop=True)

        if len(np.unique(y_tr)) < 2:
            skipped += 1
            continue

        # Target instrument mask on test fold
        te_insts = instruments[te_idx]
        target_mask = (te_insts == target_inst) if instruments_are_pooled(events_df, target_inst) else None

        # Check target slice has 2 classes
        y_target = y_te if target_mask is None else y_te[target_mask]
        if len(np.unique(y_target)) < 2 or len(y_target) < 2:
            skipped += 1
            continue

        # Fit champion model
        try:
            if model_type == "rf":
                model, _ = _tune_fit_rf(X_tr, y_tr, events_tr)
            elif model_type == "xgb":
                model, _ = _tune_fit_xgb(X_tr, y_tr, events_tr)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        except Exception as e:
            print(f"    fold {fold_i} fit failed: {e}")
            skipped += 1
            continue

        # Base AUC on target slice
        prob_all  = model.predict_proba(X_te)[:, 1]
        prob_base = prob_all if target_mask is None else prob_all[target_mask]
        auc_base  = _safe_auc(y_target, prob_base)
        if auc_base is None:
            skipped += 1
            continue
        aucs.append(auc_base)

        # Clustered MDA (pass full y_te; _clustered_mda applies target_mask internally)
        cmda_all.append(
            _clustered_mda(model, X_te, y_te, feat_cols, cluster_map,
                           auc_base, target_mask, rng)
        )

        # Clustered MDI + per-feature MDI
        mdi_all.append(_clustered_mdi(model, feat_cols, cluster_map))
        mdi_feat_all.append(model.feature_importances_)

        # SHAP on target slice only
        X_shap = X_te if target_mask is None else X_te[target_mask]
        s_signed, s_mag = _shap_values(model, X_shap, feat_cols)
        shap_signed_all.append(s_signed)
        shap_mag_all.append(s_mag)
        cshap_all.append(_clustered_shap(s_mag, cluster_map))

    print(f"    {len(aucs)} valid folds, {skipped} skipped; mean AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}" if aucs else "    no valid folds")

    def _agg(lst: list[dict]) -> tuple[pd.Series, pd.Series]:
        if not lst:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        df = pd.DataFrame(lst)
        return df.mean(), df.std()

    cmda_mean,  cmda_std  = _agg(cmda_all)
    mdi_mean,   mdi_std   = _agg(mdi_all)
    cshap_mean, cshap_std = _agg(cshap_all)

    shap_signed_mean = pd.DataFrame(shap_signed_all).mean() if shap_signed_all else pd.Series(dtype=float)
    shap_mag_mean    = pd.DataFrame(shap_mag_all).mean()    if shap_mag_all    else pd.Series(dtype=float)

    mdi_feat_mean = pd.Series(
        np.mean(mdi_feat_all, axis=0) if mdi_feat_all else np.zeros(len(feat_cols)),
        index=feat_cols,
    )

    return {
        "cmda_mean":      cmda_mean,
        "cmda_std":       cmda_std,
        "mdi_mean":       mdi_mean,
        "mdi_std":        mdi_std,
        "cshap_mean":     cshap_mean,
        "cshap_std":      cshap_std,
        "shap_signed_mean": shap_signed_mean,
        "shap_mag_mean":  shap_mag_mean,
        "mdi_feat_mean":  mdi_feat_mean,
        "fold_aucs":      aucs,
        "n_folds":        len(aucs),
    }


def instruments_are_pooled(events_df: pd.DataFrame, target_inst: str) -> bool:
    """True when the events_df contains multiple instruments (pooled group)."""
    return events_df["instrument"].nunique() > 1


# ---------------------------------------------------------------------------
# Within-cluster breakdown (SHAP ranking + PCA)
# ---------------------------------------------------------------------------

def within_cluster_breakdown(
    events_df: pd.DataFrame,
    feat_cols: list[str],
    cluster_map: dict[str, list[str]],
    shap_mag_mean: pd.Series,
    top_cluster_names: list[str],
) -> dict[str, dict]:
    """For each top cluster: rank members by mean|SHAP|; PCA on full event matrix."""
    results: dict[str, dict] = {}
    X_full = events_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    feat_idx = {n: j for j, n in enumerate(feat_cols)}

    for cname in top_cluster_names:
        cols = cluster_map.get(cname, [])
        members = [c for c in cols if c in feat_idx]
        if len(members) < 2:
            results[cname] = {"members": members, "shap_ranks": {}, "pca_pc1_var": None, "pca_loadings": {}}
            continue

        # SHAP ranking
        shap_rank = {c: float(shap_mag_mean.get(c, 0.0)) for c in members}
        shap_rank = dict(sorted(shap_rank.items(), key=lambda x: x[1], reverse=True))

        # PCA on the cluster submatrix
        col_indices = [feat_idx[c] for c in members]
        X_sub = X_full[:, col_indices]
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_sub)
        n_comp = min(3, len(members))
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        pca.fit(X_sc)
        pc1_var = float(pca.explained_variance_ratio_[0])
        loadings = {
            members[j]: float(pca.components_[0, j])
            for j in range(len(members))
        }

        results[cname] = {
            "members": members,
            "shap_ranks": shap_rank,
            "pca_pc1_var": pc1_var,
            "pca_loadings": loadings,
        }

    return results


# ---------------------------------------------------------------------------
# Significance / agreement analysis
# ---------------------------------------------------------------------------

def _cluster_significance_flags(
    cmda_mean: pd.Series,
    cmda_std: pd.Series,
) -> pd.Series:
    """True where mean_drop > 1 std (significantly above zero)."""
    std_safe = cmda_std.fillna(np.inf)
    return (cmda_mean > std_safe).rename("significant")


def _rank_agreement(
    cmda_mean: pd.Series,
    mdi_mean: pd.Series,
    cshap_mean: pd.Series,
) -> pd.DataFrame:
    """Kendall tau between each pair of cluster rankings."""
    clusters = cmda_mean.index.intersection(mdi_mean.index).intersection(cshap_mean.index)
    if len(clusters) < 3:
        return pd.DataFrame()

    rank_mda  = cmda_mean.loc[clusters].rank(ascending=False)
    rank_mdi  = mdi_mean.loc[clusters].rank(ascending=False)
    rank_shap = cshap_mean.loc[clusters].rank(ascending=False)

    tau_mda_mdi,  _ = kendalltau(rank_mda, rank_mdi)
    tau_mda_shap, _ = kendalltau(rank_mda, rank_shap)
    tau_mdi_shap, _ = kendalltau(rank_mdi, rank_shap)

    return pd.DataFrame({
        "method_pair": ["MDA-MDI", "MDA-SHAP", "MDI-SHAP"],
        "kendall_tau": [tau_mda_mdi, tau_mda_shap, tau_mdi_shap],
        "agree": [abs(tau_mda_mdi) > 0.4, abs(tau_mda_shap) > 0.4, abs(tau_mdi_shap) > 0.4],
    })


def _semantic_recovery(
    cluster_map: dict[str, list[str]],
) -> pd.DataFrame:
    """For each cluster, check how purely it maps to a single F-prefix family."""
    rows: list[dict] = []
    for cname, cols in cluster_map.items():
        if not cols:
            continue
        from collections import Counter
        pfx_counts: Counter = Counter()
        for c in cols:
            pfx = c.split("_")[0] + "_"
            pfx_counts[pfx] += 1
        dominant_pfx, dominant_n = pfx_counts.most_common(1)[0]
        purity = dominant_n / len(cols)
        rows.append({
            "cluster":       cname,
            "n_members":     len(cols),
            "dominant_pfx":  dominant_pfx,
            "purity":        round(purity, 2),
            "pure":          purity >= 0.80,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Findings note
# ---------------------------------------------------------------------------

def _generate_findings_note(
    inst: str,
    champion_cfg: dict,
    fold_aucs: list[float],
    cmda_mean: pd.Series,
    cmda_std: pd.Series,
    sig_flags: pd.Series,
    rank_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    top_cluster_names: list[str],
    within_breakdown: dict[str, dict],
) -> str:
    lines: list[str] = []
    lines.append(f"=== Findings note: {inst} ===")
    lines.append(f"Champion: {champion_cfg['group']} / {champion_cfg['model_type'].upper()}")
    lines.append(f"Signal context: {champion_cfg['notes']}")
    lines.append(f"CPCV: {len(fold_aucs)} valid folds, AUC={np.mean(fold_aucs):.3f}±{np.std(fold_aucs):.3f}" if fold_aucs else "CPCV: 0 valid folds")
    lines.append("")

    # Driving clusters
    sig_clusters = sig_flags[sig_flags].index.tolist()
    insig_clusters = sig_flags[~sig_flags].index.tolist()
    lines.append("--- Cluster-level MDA (mean ± std, sorted) ---")
    sorted_cmda = cmda_mean.sort_values(ascending=False)
    for cname in sorted_cmda.index:
        flag = "*" if sig_flags.get(cname, False) else " "
        lines.append(
            f"  {flag} {cname:<40s}  "
            f"{cmda_mean.get(cname, 0):+.4f} ± {cmda_std.get(cname, np.nan):.4f}"
        )
    lines.append("")
    lines.append(f"Significant clusters (mean > 1σ): {sig_clusters or 'none'}")
    lines.append(f"Inconclusive clusters (|mean| ≤ 1σ): {insig_clusters or 'none'}")
    lines.append("")

    # Method agreement
    if not rank_df.empty:
        lines.append("--- Cross-method rank agreement (Kendall τ) ---")
        for _, row in rank_df.iterrows():
            agree_str = "AGREE" if row["agree"] else "DISAGREE"
            lines.append(f"  {row['method_pair']}: τ={row['kendall_tau']:+.2f}  [{agree_str}]")
    lines.append("")

    # Semantic recovery
    lines.append("--- Semantic F-group recovery ---")
    if not semantic_df.empty:
        for _, row in semantic_df.iterrows():
            pure_str = "pure" if row["pure"] else "mixed"
            lines.append(
                f"  {row['cluster']}: dominant={row['dominant_pfx'].rstrip('_')} "
                f"purity={row['purity']:.0%} ({pure_str})"
            )
    lines.append("")

    # Within-cluster breakdown
    if within_breakdown:
        lines.append("--- Within-cluster breakdown (top clusters) ---")
        for cname, bd in within_breakdown.items():
            lines.append(f"  {cname} (PC1 explains {bd.get('pca_pc1_var', 0) or 0:.1%}):")
            top_shap = list(bd["shap_ranks"].items())[:5]
            for feat, val in top_shap:
                lines.append(f"    mean|SHAP|={val:.4f}  {feat}")
            top_load = sorted(bd["pca_loadings"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            lines.append(f"    PC1 top loadings: " +
                         ", ".join(f"{f} ({v:+.2f})" for f, v in top_load))
    lines.append("")
    lines.append("Note: MDI is a train-set statistic (upward-biased for high-cardinality features).")
    lines.append("SHAP uses tree_path_dependent perturbation (corrects for correlated features).")
    lines.append("Clustered MDA uses a shared row permutation across all cluster members.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outputs
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
    findings_note: str,
) -> None:
    out = OUTPUTS / inst
    out.mkdir(parents=True, exist_ok=True)

    cmda_mean = importance["cmda_mean"]
    cmda_std  = importance["cmda_std"]
    mdi_mean  = importance["mdi_mean"]
    cshap_mean = importance["cshap_mean"]

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

    # ── Clustered MDA full table ────────────────────────────────────────────
    cmda_df = pd.DataFrame({
        "mean_drop": cmda_mean,
        "std_drop":  cmda_std,
        "significant": sig_flags,
    })
    cmda_df.index.name = "cluster"
    cmda_df = cmda_df.sort_values("mean_drop", ascending=False)
    cmda_df.to_csv(out / "clustered_mda_full.csv")

    # ── Cross-check table: MDA rank, MDI rank, SHAP rank ───────────────────
    shared = cmda_mean.index.intersection(mdi_mean.index).intersection(cshap_mean.index)
    crosscheck = pd.DataFrame({
        "mda_mean":   cmda_mean.loc[shared],
        "mda_rank":   cmda_mean.loc[shared].rank(ascending=False).astype(int),
        "mdi_sum":    mdi_mean.loc[shared],
        "mdi_rank":   mdi_mean.loc[shared].rank(ascending=False).astype(int),
        "shap_sum":   cshap_mean.loc[shared],
        "shap_rank":  cshap_mean.loc[shared].rank(ascending=False).astype(int),
        "significant": sig_flags.reindex(shared).fillna(False),
    })
    crosscheck.index.name = "cluster"
    crosscheck = crosscheck.sort_values("mda_rank")
    crosscheck.to_csv(out / "cluster_crosscheck_table.csv")

    # ── Global per-feature SHAP + MDI ──────────────────────────────────────
    shap_mag  = importance["shap_mag_mean"]
    shap_sign = importance["shap_signed_mean"]
    mdi_feat  = importance["mdi_feat_mean"]
    feat_df = pd.DataFrame({
        "shap_magnitude": shap_mag,
        "shap_signed":    shap_sign,
        "mdi":            mdi_feat,
    }).dropna(how="all")
    feat_df.index.name = "feature"
    feat_df = feat_df.sort_values("shap_magnitude", ascending=False)
    feat_df.to_csv(out / "global_shap_summary.csv")

    # ── Rank-agreement table ────────────────────────────────────────────────
    if not rank_df.empty:
        rank_df.to_csv(out / "rank_agreement.csv", index=False)

    # ── Within-cluster breakdown CSVs ───────────────────────────────────────
    for cname, bd in within_breakdown.items():
        safe_name = cname.replace("/", "_").replace(" ", "_")
        rows_wc = []
        for feat, shap_val in bd["shap_ranks"].items():
            load = bd["pca_loadings"].get(feat, np.nan)
            rows_wc.append({
                "feature": feat, "mean_shap_mag": shap_val,
                "pc1_loading": load,
            })
        wc_df = pd.DataFrame(rows_wc)
        wc_df["cluster"] = cname
        wc_df["pca_pc1_var_explained"] = bd.get("pca_pc1_var")
        wc_df.to_csv(out / f"within_cluster_{safe_name}.csv", index=False)

    # ── Findings note ───────────────────────────────────────────────────────
    with open(out / "findings_note.txt", "w") as fh:
        fh.write(findings_note)

    # ════════════════════════════════════════════════════════════════════════
    # Figures
    # ════════════════════════════════════════════════════════════════════════

    # ── Dendrogram ──────────────────────────────────────────────────────────
    try:
        condensed = squareform(dist_mat)
        Z = _linkage(condensed, method="ward")
        fig, ax = plt.subplots(figsize=(16, 5))
        _dendrogram(Z, labels=corr_cols, ax=ax, leaf_rotation=90, leaf_font_size=5)
        cut_height = Z[-(best_k - 1), 2] if best_k > 1 else Z[-1, 2]
        ax.axhline(y=cut_height, color="red", linestyle="--", linewidth=1.0,
                   label=f"K={best_k} cut")
        ax.set_title(f"{inst} — Ward dendrogram on continuous features (Spearman distance)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out / "dendrogram.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] dendrogram failed: {e}")

    # ── Clustered MDA bar chart (centerpiece) ───────────────────────────────
    try:
        df_plot = cmda_df.copy()
        colors = ["#1f77b4" if s else "#aec7e8" for s in df_plot["significant"]]
        fig, ax = plt.subplots(figsize=(11, max(5, len(df_plot) * 0.35)))
        y_pos = range(len(df_plot))
        ax.barh(
            list(y_pos), df_plot["mean_drop"],
            xerr=df_plot["std_drop"].fillna(0),
            align="center", capsize=3, color=colors, alpha=0.9,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df_plot.index, fontsize=7)
        ax.set_xlabel("Mean AUC drop (clustered MDA, ± std across CPCV paths)")
        ax.set_title(
            f"{inst} champion ({importance['n_folds']} CPCV paths)  "
            f"— Cluster-level importance\n"
            f"(dark = significant > 1σ; light = inconclusive)"
        )
        ax.axvline(x=0, color="black", linewidth=0.8)
        plt.tight_layout()
        fig.savefig(out / "clustered_mda_chart.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] clustered MDA chart failed: {e}")

    # ── Global SHAP bar chart ───────────────────────────────────────────────
    try:
        top_feats = feat_df.head(30)
        fig, ax = plt.subplots(figsize=(10, max(5, len(top_feats) * 0.32)))
        y_pos = range(len(top_feats))
        colors_shap = [
            "#d62728" if importance["shap_signed_mean"].get(f, 0) > 0 else "#1f77b4"
            for f in top_feats.index
        ]
        ax.barh(list(y_pos), top_feats["shap_magnitude"], align="center",
                color=colors_shap, alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_feats.index, fontsize=6)
        ax.set_xlabel("Mean |SHAP value| (tree_path_dependent)")
        ax.set_title(f"{inst} — Global per-feature SHAP (top 30)\n"
                     f"red = positive signal, blue = negative signal")
        ax.axvline(x=0, color="black", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(out / "global_shap_chart.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] global SHAP chart failed: {e}")

    # ── Within-cluster SHAP bar charts for top clusters ─────────────────────
    for cname, bd in within_breakdown.items():
        try:
            safe_name = cname.replace("/", "_").replace(" ", "_")
            items = list(bd["shap_ranks"].items())
            if not items:
                continue
            feats_wc = [x[0] for x in items]
            vals_wc  = [x[1] for x in items]
            fig, axes = plt.subplots(1, 2, figsize=(13, max(3, len(feats_wc) * 0.4)))

            # SHAP magnitude
            ax0 = axes[0]
            ax0.barh(range(len(feats_wc)), vals_wc, align="center", color="#1f77b4", alpha=0.85)
            ax0.set_yticks(range(len(feats_wc)))
            ax0.set_yticklabels(feats_wc, fontsize=7)
            ax0.set_xlabel("Mean |SHAP| within cluster")
            ax0.set_title(f"SHAP ranking\nPC1 explains {bd.get('pca_pc1_var', 0) or 0:.1%} of variance")

            # PCA PC1 loadings
            ax1 = axes[1]
            load_vals = [bd["pca_loadings"].get(f, 0.0) for f in feats_wc]
            colors_l = ["#d62728" if v > 0 else "#1f77b4" for v in load_vals]
            ax1.barh(range(len(feats_wc)), load_vals, align="center",
                     color=colors_l, alpha=0.85)
            ax1.set_yticks(range(len(feats_wc)))
            ax1.set_yticklabels(feats_wc, fontsize=7)
            ax1.set_xlabel("PC1 loading (standardised)")
            ax1.set_title(f"PCA PC1 loadings")
            ax1.axvline(x=0, color="black", linewidth=0.5)

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
    force: bool = False,
) -> dict[str, Any]:
    """Run the champion importance pipeline for each requested instrument."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    targets = instruments if instruments is not None else list(CHAMPIONS.keys())
    results: dict[str, Any] = {}

    for inst in targets:
        if inst not in CHAMPIONS:
            print(f"[skip] {inst}: not a champion instrument")
            continue

        cfg = CHAMPIONS[inst]
        print(f"\n{'='*64}")
        print(f"Champion importance: {inst}  (group={cfg['group']}, model={cfg['model_type'].upper()})")
        print("=" * 64)

        # Skip-check
        done_marker = OUTPUTS / inst / "clustered_mda_full.csv"
        if done_marker.exists() and not force:
            print(f"  [cache hit] outputs already exist; pass --force to rerun")
            continue

        # 1. Load events from model_comparison cache
        cache_path = CACHE_DIR / f"{cfg['group']}_events.parquet"
        if not cache_path.exists():
            print(f"  [error] cache not found: {cache_path}")
            continue
        events_df = pd.read_parquet(cache_path)
        feat_cols = _feat_cols(events_df)
        print(f"  Events: {len(events_df)}, features: {len(feat_cols)}")

        # 2. Cluster features
        print("  Clustering features...")
        groups = _assign_groups_champion(events_df)
        corr_cols = groups.pop("corr_cluster", [])
        hand_groups = groups   # F4_latent, F5_signal, F8_calendar, F_instrument, F_misc

        if len(corr_cols) < 3:
            print(f"  [warn] only {len(corr_cols)} corr-cluster features; skipping corr clustering")
            cluster_map = hand_groups
            best_k = 0
            cluster_metrics = pd.DataFrame()
            cluster_labels = np.array([])
            dist_mat = np.zeros((0, 0))
        else:
            X_corr = events_df[corr_cols].fillna(0)
            dist_mat = compute_spearman_distance(X_corr)
            best_k, cluster_metrics = select_k(X_corr, dist_mat)
            sil = cluster_metrics.set_index("K").loc[best_k, "silhouette"]
            print(f"  Best K={best_k} (silhouette={sil:.3f})")
            cluster_labels = get_cluster_labels(dist_mat, best_k)
            cluster_reps = cluster_representatives(X_corr, cluster_labels)
            cluster_map = build_cluster_map(corr_cols, cluster_labels, hand_groups)

        n_clusters = len(cluster_map)
        print(f"  Cluster map: {n_clusters} groups "
              f"({best_k} corr + {len(hand_groups)} hand-assigned)")

        # 3. Run CPCV importance
        print(f"  Running CPCV ({CPCV_N_GROUPS} groups, k={CPCV_K}) ...")
        importance = run_champion_cpcv(events_df, feat_cols, cluster_map, cfg)

        if not importance["fold_aucs"]:
            print("  [error] no valid folds — skipping outputs")
            continue

        # 4. Significance flags + rank agreement
        sig_flags = _cluster_significance_flags(
            importance["cmda_mean"], importance["cmda_std"]
        )
        rank_df = _rank_agreement(
            importance["cmda_mean"], importance["mdi_mean"], importance["cshap_mean"]
        )

        # 5. Semantic recovery analysis
        semantic_df = _semantic_recovery(cluster_map)

        # 6. Within-cluster breakdown for top-N significant clusters
        top_sig = sig_flags[sig_flags].index.tolist()
        # Fall back to top-N by MDA if fewer than TOP_CLUSTERS_N are significant
        sorted_by_mda = importance["cmda_mean"].sort_values(ascending=False)
        top_by_mda = sorted_by_mda.index[:TOP_CLUSTERS_N].tolist()
        top_cluster_names = list(dict.fromkeys(top_sig[:TOP_CLUSTERS_N] + top_by_mda))[:TOP_CLUSTERS_N]

        within_breakdown = within_cluster_breakdown(
            events_df, feat_cols, cluster_map,
            importance["shap_mag_mean"], top_cluster_names
        )

        # 7. Findings note
        findings_note = _generate_findings_note(
            inst, cfg, importance["fold_aucs"],
            importance["cmda_mean"], importance["cmda_std"],
            sig_flags, rank_df, semantic_df,
            top_cluster_names, within_breakdown,
        )

        # 8. Save outputs
        _save_outputs(
            inst, events_df, feat_cols, cluster_map, cluster_metrics,
            cluster_labels, corr_cols, dist_mat, best_k,
            importance, sig_flags, rank_df, within_breakdown, findings_note,
        )

        results[inst] = {
            "cluster_map":    cluster_map,
            "importance":     importance,
            "sig_flags":      sig_flags,
            "rank_agreement": rank_df,
            "semantic_df":    semantic_df,
            "within_breakdown": within_breakdown,
            "findings_note":  findings_note,
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Champion feature importance (cl1s, es1s, ho1s, rb1s)"
    )
    parser.add_argument(
        "--instruments", nargs="+", default=None,
        help="Subset of champions to run (default: all four)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun even if outputs already exist"
    )
    args = parser.parse_args()
    run_champion_importance(instruments=args.instruments, force=args.force)
    print("\nDone.")
