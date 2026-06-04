"""Phase-I+ runner — within-cluster + global SHAP + cross-check + pruned model.

Produces, per asset class, the per-instrument-importance artifacts that
Harry's branch emits:

  cluster_crosscheck_table.csv  -- cluster ranks under MDA/MDI/SHAP, significance.
  within_cluster_<CID>.csv      -- one CSV per top cluster: members ranked by
                                   mean |SHAP| + PC1/PC2/PC3 loadings + variance.
  global_shap_summary.csv       -- per-feature mean |SHAP| + signed SHAP + MDI.
  findings_note.txt             -- human-readable per-class summary.
  pruned_vs_full_auc.csv        -- val-partition AUC: full vs pruned.
                                   Importance + feature selection happen on
                                   TRAIN; the comparison is on held-out VAL.
  pruned_features.json          -- the pruned feature list and selection reasons.

Selection rule for the pruned model (per Harry's branch + user's spec):

  1. For each cluster that is *significant* (MDA mean > 1 sigma above zero):
       - If PC1 variance >= 65 percent  ->  keep PC1 component (a single
         linear combination of cluster members) as a derived feature.
       - Else if PC1+PC2 variance >= 75 percent -> keep PC1 + PC2 components.
       - Else -> keep the top 1-2 members by mean |SHAP|.
  2. Add any global top-5 SHAP features that aren't already in any top
     cluster's chosen members / components.

Honest evaluation protocol:

  * Importance + cluster-level analysis: TRAIN only.
  * Global SHAP fit: TRAIN only.
  * Pruned feature selection: TRAIN only.
  * Full-vs-pruned AUC: fit on TRAIN, score on VAL (held out).

Test partition remains sealed.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from stml.experimental.champion_pipeline import (
    _select_feature_cols, _slice_pool, _add_instrument_onehot,
    _restrict_modelling, POOL_MEMBERS,
)
from stml.experimental.config import ASSET_CLASS_MEMBERS, PipelineConfig
from stml.experimental.cv import CombinatorialPurgedCV
from stml.experimental.evaluation import cross_val_evaluate
from stml.experimental.importance import (
    HygieneConfig, ImportanceConfig, apply_hygiene,
    cluster_features_mantegna,
)
from stml.experimental.make_scope import embargo_days_map
from stml.experimental.models import balanced_sample_weight
from stml.experimental.pipeline import _fresh_estimator


_SCHEMA_COLS = frozenset({
    "instrument", "t_signal", "t_start", "t_end", "side", "ret", "label",
    "uniqueness_weight", "sigma_at_t", "barrier_hit",
    "pt", "sl", "h", "partition",
})


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError(f"Could not locate repo root from {here}")


# ---------------------------------------------------------------------------
# Cluster cross-check (Harry's cluster_crosscheck_table.csv schema).
# ---------------------------------------------------------------------------


def cluster_crosscheck(agg: pd.DataFrame) -> pd.DataFrame:
    """Add MDA/MDI/SHAP ranks per cluster + significance flag."""
    df = agg.copy()
    df = df.sort_values("mda_mean", ascending=False).reset_index(drop=True)
    df["mda_rank"] = df["mda_mean"].rank(ascending=False).astype(int)
    df["mdi_rank"] = df["mdi_sum_mean"].rank(ascending=False).astype(int)
    df["shap_rank"] = df["shap_sum_mean"].rank(ascending=False).astype(int)
    # Significant if MDA mean > 1 sigma above zero (i.e. lower CI > 0).
    df["significant"] = df["mda_mean"] > df["mda_std_across_folds"]
    return df.loc[:, [
        "cluster_id", "n_members", "members",
        "mda_mean", "mda_rank",
        "mdi_sum_mean", "mdi_rank",
        "shap_sum_mean", "shap_rank",
        "significant",
    ]].rename(columns={
        "mdi_sum_mean": "mdi_sum", "shap_sum_mean": "shap_sum",
    })


# ---------------------------------------------------------------------------
# Within-cluster breakdown (Harry's within_cluster_<CID>.csv schema).
# ---------------------------------------------------------------------------


def within_cluster_breakdown(
    cluster_id: int,
    members: list[str],
    X_modelling: pd.DataFrame,
    global_shap: pd.Series,
) -> pd.DataFrame:
    """One row per member: mean |SHAP|, PC1/2/3 loadings, PC variance explained."""
    present = [m for m in members if m in X_modelling.columns]
    if not present:
        return pd.DataFrame()
    Xn = X_modelling.loc[:, present].dropna(how="all")
    Xn = Xn.fillna(Xn.median(numeric_only=True))
    # Standardise for PCA.
    mu = Xn.mean(); sd = Xn.std(ddof=0).replace(0, 1.0)
    Xs = (Xn - mu) / sd
    n_components = min(3, max(1, Xs.shape[1]))
    pca = PCA(n_components=n_components)
    pca.fit(Xs.values)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=present,
        columns=[f"pc{i+1}_loading" for i in range(n_components)],
    )
    var_explained = {
        f"pca_pc{i+1}_var_explained": float(pca.explained_variance_ratio_[i])
        for i in range(n_components)
    }
    rows = []
    for f in present:
        row = {
            "feature": f,
            "cluster": cluster_id,
            "mean_shap_mag": float(global_shap.get(f, 0.0)),
        }
        for i in range(n_components):
            row[f"pc{i+1}_loading"] = float(loadings.loc[f, f"pc{i+1}_loading"])
            row[f"pca_pc{i+1}_var_explained"] = var_explained[
                f"pca_pc{i+1}_var_explained"
            ]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mean_shap_mag", ascending=False)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Global SHAP table (per-feature).
# ---------------------------------------------------------------------------


def compute_global_shap(
    X_train: pd.DataFrame, y_train: pd.Series, sw_train: np.ndarray,
    feature_cols: list[str], cfg: PipelineConfig,
) -> pd.DataFrame:
    """XGBoost-native TreeSHAP per feature on the asset-class training data."""
    Xa = X_train.loc[:, feature_cols].to_numpy(dtype=float)
    # XGBoost rejects inf when missing != inf; clip + NaN fill defensively.
    Xa = np.where(np.isfinite(Xa), Xa, np.nan)
    Xa = np.clip(Xa, -1e10, 1e10)
    col_medians = np.nanmedian(Xa, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    nan_mask = np.isnan(Xa)
    Xa[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    booster = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", random_state=cfg.seed,
        eval_metric="logloss", n_jobs=1,
    )
    booster.fit(Xa, y_train.values, sample_weight=sw_train)
    dmat = xgb.DMatrix(Xa)
    contribs = booster.get_booster().predict(dmat, pred_contribs=True)
    # Last column is the bias.
    shap = contribs[:, :-1]
    out = pd.DataFrame({
        "feature": feature_cols,
        "shap_magnitude": np.abs(shap).mean(axis=0),
        "shap_signed": shap.mean(axis=0),
        "mdi": booster.feature_importances_,
    }).sort_values("shap_magnitude", ascending=False)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature pruning rule (the user's spec).
# ---------------------------------------------------------------------------


@dataclass
class FeatureSelection:
    selected: list[str]
    reasons: dict[str, str]


def pruned_feature_set(
    crosscheck: pd.DataFrame,
    within_cluster_dfs: dict[int, pd.DataFrame],
    global_shap: pd.DataFrame,
    pc1_threshold: float = 0.65,
    pc12_threshold: float = 0.75,
    top_global_n: int = 5,
) -> FeatureSelection:
    """Apply the user's rule:
      * Each significant cluster -> top-1/2 SHAP members OR PC1/PC1+PC2.
      * Plus global top-N SHAP that aren't in any chosen cluster.
    """
    selected: list[str] = []
    reasons: dict[str, str] = {}
    chosen_pool: set[str] = set()
    sig = crosscheck.loc[crosscheck["significant"]].copy()

    for _, row in sig.iterrows():
        cid = int(row["cluster_id"])
        wc = within_cluster_dfs.get(cid)
        if wc is None or wc.empty:
            continue
        pc1_var = float(wc["pca_pc1_var_explained"].iloc[0])
        pc2_var = (
            float(wc["pca_pc2_var_explained"].iloc[0])
            if "pca_pc2_var_explained" in wc.columns else 0.0
        )
        # Top-N by mean |SHAP|.
        top_by_shap = wc["feature"].head(2).tolist()
        if pc1_var >= pc1_threshold:
            # One latent dimension -> keep the strongest single member as
            # a stand-in for the latent.
            keep = [top_by_shap[0]]
            why = f"cluster {cid}: PC1 {pc1_var:.0%} -> single rep"
        elif (pc1_var + pc2_var) >= pc12_threshold:
            keep = top_by_shap[:2]
            why = (f"cluster {cid}: PC1+PC2 {(pc1_var+pc2_var):.0%} -> "
                   f"two reps")
        else:
            keep = top_by_shap[:2]
            why = (f"cluster {cid}: PC1 {pc1_var:.0%} multi-dim -> "
                   f"top-2 by SHAP")
        for f in keep:
            if f not in chosen_pool:
                selected.append(f)
                reasons[f] = why
                chosen_pool.add(f)

    # Global top-N SHAP not in any chosen pool.
    global_top = global_shap.head(top_global_n)["feature"].tolist()
    for f in global_top:
        if f not in chosen_pool:
            selected.append(f)
            reasons[f] = f"global SHAP top-{top_global_n}"
            chosen_pool.add(f)

    return FeatureSelection(selected=selected, reasons=reasons)


# ---------------------------------------------------------------------------
# Full-vs-pruned AUC comparison under CPCV(6,2).
# ---------------------------------------------------------------------------


def _evaluate_feature_set(
    pool_df: pd.DataFrame, feature_cols: list[str], cfg: PipelineConfig,
    embargo_map: dict[str, int],
) -> tuple[float, float]:
    pool_df = _add_instrument_onehot(pool_df)
    cols = feature_cols + [c for c in pool_df.columns if c.startswith("inst_")]
    X = pool_df.loc[:, cols].copy()
    y = pool_df["label"].astype(int)
    t = pd.to_datetime(pool_df["t_signal"])
    t1 = pd.to_datetime(pool_df["t_end"])
    insts = pool_df["instrument"]
    uniq = pool_df["uniqueness_weight"].astype(float)
    cv = CombinatorialPurgedCV(
        n_groups=cfg.cpcv_n_groups, n_test_groups=cfg.cpcv_n_test_groups,
        t=t, t1=t1, pct_embargo=cfg.cpcv_pct_embargo,
        instruments=insts, embargo_days=embargo_map,
    )
    result = cross_val_evaluate(
        make_model=lambda: _fresh_estimator("xgboost", seed=cfg.seed),
        X=X, y=y, cv=cv,
        uniqueness_weights=uniq,
    )
    if result.oos_predictions.empty:
        return float("nan"), float("nan")
    aucs = []
    for fold_id, grp in result.oos_predictions.groupby("fold"):
        if grp["y_true"].nunique() < 2:
            continue
        try:
            aucs.append(
                roc_auc_score(grp["y_true"], grp["y_proba"],
                              sample_weight=grp["sample_weight"])
            )
        except Exception:
            pass
    if not aucs:
        return float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.std(aucs) / np.sqrt(max(len(aucs), 1)))


def _fit_predict_train_to_val(
    train_pool: pd.DataFrame, val_pool: pd.DataFrame,
    feature_cols: list[str], cfg: PipelineConfig,
) -> tuple[float, float, int]:
    """Fit XGBoost on TRAIN with the given feature set, score VAL.

    Returns (val_auc, val_sem, n_val) where val_sem is the standard binomial
    AUC SE proxy = sqrt(p(1-p)/n) on the val positive rate.
    """
    train_pool = _add_instrument_onehot(train_pool)
    val_pool = _add_instrument_onehot(val_pool)
    cols = feature_cols + [c for c in train_pool.columns if c.startswith("inst_")]
    # Ensure val has every column train has.
    for c in cols:
        if c not in val_pool.columns:
            val_pool[c] = 0.0
    X_tr = np.array(train_pool.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), copy=True)
    y_tr = train_pool["label"].astype(int).values
    X_va = np.array(val_pool.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), copy=True)
    y_va = val_pool["label"].astype(int).values
    sw_tr = train_pool["uniqueness_weight"].astype(float).values

    # Sanitise: nan/inf, large values.
    X_tr = np.clip(X_tr, -1e10, 1e10)
    X_va = np.clip(X_va, -1e10, 1e10)
    col_medians = np.nanmedian(X_tr, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    for arr in (X_tr, X_va):
        nan_mask = ~np.isfinite(arr)
        if nan_mask.any():
            arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", random_state=cfg.seed,
        eval_metric="logloss", n_jobs=1,
    )
    model.fit(X_tr, y_tr, sample_weight=sw_tr)
    proba = model.predict_proba(X_va)[:, 1]
    if len(set(y_va)) < 2:
        return float("nan"), float("nan"), int(len(y_va))
    auc = float(roc_auc_score(y_va, proba))
    # Hanley-McNeil SE proxy.
    n_pos = int(y_va.sum())
    n_neg = int(len(y_va) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return auc, float("nan"), int(len(y_va))
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    sem = float(np.sqrt(max(var, 0.0)))
    return auc, sem, int(len(y_va))


def full_vs_pruned_comparison(
    features: pd.DataFrame, asset_class: str, selected: list[str],
    cfg: PipelineConfig, verbose: bool = False,
) -> pd.DataFrame:
    """Honest eval: importance + selection used TRAIN only (in run_class).
    Here we fit on TRAIN and score on the held-out VAL partition.
    """
    members = ASSET_CLASS_MEMBERS[asset_class]
    pool = features.loc[features["instrument"].isin(members)].copy()
    train_pool = pool.loc[pool["partition"] == "train"].reset_index(drop=True)
    val_pool = pool.loc[pool["partition"] == "val"].reset_index(drop=True)
    if train_pool.empty or val_pool.empty:
        return pd.DataFrame()
    feature_cols_all = [c for c in train_pool.columns if c not in _SCHEMA_COLS
                         and not c.startswith("inst_")]
    pruned = [c for c in selected if c in train_pool.columns]
    if not pruned:
        return pd.DataFrame()
    if verbose:
        print(f"  full:   {len(feature_cols_all)} features  train n={len(train_pool)}  val n={len(val_pool)}")
        print(f"  pruned: {len(pruned)} features")
    full_auc, full_sem, n_val = _fit_predict_train_to_val(
        train_pool, val_pool, feature_cols_all, cfg,
    )
    pruned_auc, pruned_sem, _ = _fit_predict_train_to_val(
        train_pool, val_pool, pruned, cfg,
    )
    return pd.DataFrame([{
        "model": "full",
        "n_features": len(feature_cols_all),
        "n_val": n_val,
        "val_auc": full_auc, "val_sem": full_sem,
    }, {
        "model": "pruned",
        "n_features": len(pruned),
        "n_val": n_val,
        "val_auc": pruned_auc, "val_sem": pruned_sem,
    }])


# ---------------------------------------------------------------------------
# Per-asset-class driver.
# ---------------------------------------------------------------------------


def run_class(
    features: pd.DataFrame, asset_class: str, *,
    cfg: PipelineConfig, top_clusters: int = 3, verbose: bool = True,
) -> dict[str, Any]:
    members = ASSET_CLASS_MEMBERS[asset_class]
    df = features.loc[features["instrument"].isin(members)].copy()
    df = _restrict_modelling(df, cfg).reset_index(drop=True)
    if verbose:
        print(f"  modelling sample: {len(df)} events")

    feature_cols = [c for c in df.columns if c not in _SCHEMA_COLS
                     and not c.startswith("inst_")]
    X = df.loc[:, feature_cols].copy()
    X, _hygiene_log = apply_hygiene(X, config=HygieneConfig())

    # Cluster the surviving features.
    membership, _dist, _k_scores = cluster_features_mantegna(X)

    # Existing cluster summary (we already saved this in Phase E).
    root = _find_repo_root()
    out_dir = root / "results" / "sreeram_experimental" / "importance" / asset_class
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = pd.read_csv(out_dir / "clustered_importance.csv")

    # Cross-check + significance.
    crosscheck = cluster_crosscheck(agg)
    crosscheck.to_csv(out_dir / "cluster_crosscheck_table.csv",
                       index=False, float_format="%.6f")
    if verbose:
        print(f"  wrote {out_dir / 'cluster_crosscheck_table.csv'}")

    # Global per-feature SHAP table (one fit on the asset-class modelling sample).
    y = df["label"].astype(int)
    uniq = df["uniqueness_weight"].astype(float).values
    sw = balanced_sample_weight(y.values, base=uniq)
    global_shap = compute_global_shap(
        df, y, sw, feature_cols=list(X.columns), cfg=cfg,
    )
    global_shap.to_csv(out_dir / "global_shap_summary.csv",
                       index=False, float_format="%.6f")
    if verbose:
        print(f"  wrote {out_dir / 'global_shap_summary.csv'}")

    # Within-cluster CSVs for top-K clusters.
    shap_series = global_shap.set_index("feature")["shap_magnitude"]
    top = crosscheck.head(top_clusters)
    within_dfs: dict[int, pd.DataFrame] = {}
    for _, row in top.iterrows():
        cid = int(row["cluster_id"])
        members_str = str(row["members"])
        members_list = [m.strip() for m in members_str.split(",")]
        wc = within_cluster_breakdown(
            cid, members_list, X_modelling=df, global_shap=shap_series,
        )
        if wc.empty:
            continue
        within_dfs[cid] = wc
        # Use the dominant F-family suffix as Harry does (e.g., C14_f11).
        suffix = wc["feature"].iloc[0].split("_")[0]  # e.g., "f2"
        wc.to_csv(out_dir / f"within_cluster_C{cid}_{suffix}.csv",
                   index=False, float_format="%.6f")
    if verbose:
        print(f"  wrote {len(within_dfs)} within_cluster_*.csv files")

    # Pruned feature set + full-vs-pruned AUC.
    selection = pruned_feature_set(
        crosscheck, within_dfs, global_shap,
        pc1_threshold=0.65, pc12_threshold=0.75, top_global_n=5,
    )
    with open(out_dir / "pruned_features.json", "w") as f:
        json.dump({
            "asset_class": asset_class,
            "n_pruned": len(selection.selected),
            "features": selection.selected,
            "reasons": selection.reasons,
        }, f, indent=2)
    cmp_df = full_vs_pruned_comparison(
        features, asset_class, selection.selected, cfg, verbose=verbose,
    )
    if not cmp_df.empty:
        cmp_df.to_csv(out_dir / "pruned_vs_full_auc.csv",
                       index=False, float_format="%.6f")
    if verbose:
        if not cmp_df.empty:
            print("  pruned vs full AUC:")
            print(cmp_df.to_string(index=False))

    # Findings note.
    sig_clusters = crosscheck.loc[crosscheck["significant"]]
    note_lines = [
        f"=== {asset_class.upper()} — feature importance findings ===",
        "",
        f"Modelling sample: {len(df)} events; {X.shape[1]} features after hygiene.",
        f"Significant clusters (MDA mean > 1 sigma): "
        f"{len(sig_clusters)} of {len(crosscheck)}.",
        "",
    ]
    if not sig_clusters.empty:
        note_lines.append("Top significant clusters (by MDA):")
        for _, r in sig_clusters.head(3).iterrows():
            cid = int(r["cluster_id"])
            wc = within_dfs.get(cid)
            mda_str = f"MDA {r['mda_mean']:+.4f} +/- {r['mda_sum'] if 'mda_sum' in r else 0:.4f}"
            top_member = wc["feature"].iloc[0] if wc is not None and not wc.empty else "?"
            pc1 = (float(wc["pca_pc1_var_explained"].iloc[0])
                    if wc is not None and not wc.empty else 0.0)
            note_lines.append(
                f"  C{cid}: {r['n_members']} members, {mda_str}, "
                f"top SHAP member = {top_member}, PC1 var = {pc1:.0%}"
            )
        note_lines.append("")
    note_lines.append("Top 10 global features by mean |SHAP|:")
    for _, r in global_shap.head(10).iterrows():
        note_lines.append(f"  {r['feature']:<35} |SHAP| = {r['shap_magnitude']:.4f}")
    note_lines.append("")
    note_lines.append(f"Pruned feature set ({len(selection.selected)} features):")
    for f in selection.selected:
        note_lines.append(f"  - {f}  ({selection.reasons[f]})")
    note_lines.append("")
    if not cmp_df.empty:
        note_lines.append("CPCV(6,2) AUC, full vs pruned:")
        for _, r in cmp_df.iterrows():
            note_lines.append(
                f"  {r['model']:<8} n_features = {int(r['n_features']):>3}  "
                f"AUC = {r['val_auc']:.4f} +/- {r['val_sem']:.4f}"
            )
    (out_dir / "findings_note.txt").write_text("\n".join(note_lines))
    if verbose:
        print(f"  wrote {out_dir / 'findings_note.txt'}")

    return {
        "asset_class": asset_class,
        "n_features": int(X.shape[1]),
        "crosscheck": crosscheck,
        "within_cluster": within_dfs,
        "global_shap": global_shap,
        "pruned_features": selection,
        "comparison": cmp_df,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deep-importance runner")
    ap.add_argument("--top-clusters", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = PipelineConfig()
    root = _find_repo_root()
    features = pd.read_parquet(root / "data" / "sreeram_experimental_features.parquet")
    print(f"Loaded features: {features.shape}")
    summary_rows = []
    for cls in ("equity", "energy", "metals"):
        print(f"\n[deep-importance] {cls}")
        out = run_class(features, cls, cfg=cfg,
                         top_clusters=args.top_clusters,
                         verbose=not args.quiet)
        cmp_df = out["comparison"]
        full_auc = cmp_df.loc[cmp_df["model"] == "full", "val_auc"].iloc[0] if not cmp_df.empty else float("nan")
        pruned_auc = cmp_df.loc[cmp_df["model"] == "pruned", "val_auc"].iloc[0] if not cmp_df.empty else float("nan")
        summary_rows.append({
            "asset_class": cls,
            "n_significant_clusters": int(out["crosscheck"]["significant"].sum()),
            "n_pruned_features": len(out["pruned_features"].selected),
            "full_auc": full_auc,
            "pruned_auc": pruned_auc,
            "delta_auc": pruned_auc - full_auc,
        })
    summary = pd.DataFrame(summary_rows)
    out_path = root / "results" / "sreeram_experimental" / "importance" / "deep_summary.csv"
    summary.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n=== Deep-importance summary ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
