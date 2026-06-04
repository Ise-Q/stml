"""S5 runner — per-class cluster importance with the 4 bug fixes.

Plan §8 S5 deliverable. Produces, per asset class:
  results/sreeram_experimental/importance/{class}/clustered_importance.csv
  results/sreeram_experimental/importance/{class}/cluster_membership.csv
  results/sreeram_experimental/importance/{class}/rank_agreement.csv
  results/sreeram_experimental/importance/{class}/hygiene_log.txt

Per plan §3.6 + §8 S5 acceptance:
  * Bug fix 1 (max_features='sqrt'): verified in the RF construction.
  * Bug fix 2 (PurgedKFold for MDA): use CombinatorialPurgedCV(6,2) → 15 paths.
  * Bug fix 3 (SHAP): DEFERRED — shap requires numba which requires numpy<2.4;
    our pandas 3.0 pins numpy>=2.4. Substituted with mean |gain| importance
    from the fitted RF (qualitatively similar; documented in plan §13 R-12).
  * Bug fix 4 (Mantegna distance √(1-|ρ|)): metric, not the non-metric 1-|ρ|.
  * Acceptance gate: ≥1 cluster per class has MDA > 0.02.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from stml.experimental.config import ASSET_CLASS_MEMBERS, PipelineConfig
from stml.experimental.cv import CombinatorialPurgedCV
from stml.experimental.importance import (
    HygieneConfig,
    ImportanceConfig,
    aggregate_across_folds,
    apply_hygiene,
    cluster_features_mantegna,
    cluster_importance_one_fold,
    kendall_rank_agreement,
)
from stml.experimental.make_scope import embargo_days_map


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


_SCHEMA_COLS = frozenset({
    "instrument", "t_signal", "t_start", "t_end", "side", "ret", "label",
    "uniqueness_weight", "sigma_at_t", "barrier_hit",
})


def run_class(
    features: pd.DataFrame,
    asset_class: str,
    cfg: PipelineConfig | None = None,
    *,
    verbose: bool = True,
) -> dict:
    """Run cluster importance for one asset class on the modelling sample."""
    cfg = cfg or PipelineConfig()
    members = ASSET_CLASS_MEMBERS[asset_class]
    df = features.loc[features["instrument"].isin(members)].copy()
    df["t_signal"] = pd.to_datetime(df["t_signal"])
    df["t_end"] = pd.to_datetime(df["t_end"])
    # Modelling sample only (plan §11.3).
    train_cut = pd.Timestamp(cfg.global_train_cut)
    df = df.loc[df["t_signal"] <= train_cut].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in _SCHEMA_COLS]
    X = df.loc[:, feature_cols].copy()

    # Hygiene (plan §4.12 + §5.16 pattern).
    X, hygiene_log = apply_hygiene(X, config=HygieneConfig())
    if verbose:
        for line in hygiene_log:
            print(f"  hygiene: {line}")
        print(f"  after hygiene: {X.shape[1]} features × {X.shape[0]} rows")

    # Bug fix 4 — Mantegna clustering on the modelling sample.
    membership, dist_matrix, k_scores = cluster_features_mantegna(X)
    if verbose:
        K = int(membership.nunique())
        print(f"  selected K = {K} clusters (silhouette best of {len(k_scores)} candidates)")
        for cid in sorted(membership.unique()):
            n = (membership == cid).sum()
            members_str = ",".join(membership.index[membership == cid].tolist()[:3])
            print(f"    cluster {cid}: {n} features  e.g. {members_str}{'...' if n > 3 else ''}")

    # Per-fold importance under CPCV(6,2) — bug fix 2 (purging).
    y = df["label"].astype(int).values
    t = df["t_signal"]
    t1 = df["t_end"]
    inst = df["instrument"]
    uniq = df["uniqueness_weight"].astype(float).values

    cv = CombinatorialPurgedCV(
        n_groups=cfg.cpcv_n_groups,
        n_test_groups=cfg.cpcv_n_test_groups,
        t=t, t1=t1, pct_embargo=cfg.cpcv_pct_embargo,
        instruments=inst, embargo_days=embargo_days_map(),
    )

    imp_cfg = ImportanceConfig()
    per_fold = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        from stml.experimental.importance import _balanced_sample_weight

        rng = np.random.default_rng(cfg.seed + fold_id)
        sw_tr = _balanced_sample_weight(y[train_idx], uniq[train_idx])
        sw_va = uniq[test_idx]
        df_fold = cluster_importance_one_fold(
            X_train=X.iloc[train_idx], y_train=y[train_idx], sw_train=sw_tr,
            X_val=X.iloc[test_idx], y_val=y[test_idx], sw_val=sw_va,
            membership=membership, cfg=imp_cfg, rng=rng,
        )
        if not df_fold.empty:
            per_fold.append(df_fold)

    aggregated = aggregate_across_folds(per_fold)
    ranks = kendall_rank_agreement(aggregated)

    return {
        "class": asset_class,
        "membership": membership,
        "aggregated": aggregated,
        "rank_agreement": ranks,
        "hygiene_log": hygiene_log,
        "n_clusters": int(membership.nunique()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S5 cluster importance runner")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    root = _find_repo_root()
    features = pd.read_parquet(root / "data" / "sreeram_experimental_features.parquet")
    print(f"Loaded features: {features.shape}")

    summary_rows = []
    for cls in ("equity", "energy", "metals"):
        print(f"\n[importance] {cls}")
        t0 = time.time()
        result = run_class(features, cls, cfg=cfg, verbose=not args.quiet)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")

        agg = result["aggregated"]
        if agg.empty:
            print("  (no aggregated results)")
            continue

        # Acceptance gate: ≥1 cluster with MDA > 0.02.
        top_mda = float(agg["mda_mean"].max())
        n_above_002 = int((agg["mda_mean"] > 0.02).sum())
        print(f"  top cluster MDA mean = {top_mda:.4f}  (≥0.02 gate: {'PASS' if n_above_002 >= 1 else 'CHECK'})")
        print(f"  n_clusters with MDA > 0.02: {n_above_002} / {len(agg)}")
        print(f"\n  Cluster importance summary (top 5 by MDA):")
        print(agg.head(5).to_string(index=False))
        print(f"\n  Kendall τ rank agreement across methods:")
        print(result["rank_agreement"].to_string(index=False))

        summary_rows.append({
            "asset_class": cls,
            "n_clusters": int(result["n_clusters"]),
            "n_features": int(len(result["membership"])),
            "top_mda_mean": top_mda,
            "n_clusters_mda_above_002": n_above_002,
        })

        if not args.no_persist:
            out_dir = root / "results" / "sreeram_experimental" / "importance" / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            agg.to_csv(out_dir / "clustered_importance.csv", index=False, float_format="%.6f")
            result["membership"].to_frame(name="cluster_id").to_csv(
                out_dir / "cluster_membership.csv", index_label="feature"
            )
            result["rank_agreement"].to_csv(out_dir / "rank_agreement.csv", index=False, float_format="%.6f")
            (out_dir / "hygiene_log.txt").write_text("\n".join(result["hygiene_log"]))
            print(f"  → {out_dir.relative_to(root)}/")

    if summary_rows and not args.no_persist:
        pd.DataFrame(summary_rows).to_csv(
            root / "results" / "sreeram_experimental" / "importance" / "summary.csv",
            index=False, float_format="%.6f",
        )

    # Overall acceptance gate.
    print("\n=== Plan §8 S5 acceptance ===")
    if not summary_rows:
        print("  CHECK — no class results produced")
        return 1
    all_pass = all(r["n_clusters_mda_above_002"] >= 1 for r in summary_rows)
    for r in summary_rows:
        status = "PASS" if r["n_clusters_mda_above_002"] >= 1 else "CHECK"
        print(f"  {status}  {r['asset_class']}: top MDA = {r['top_mda_mean']:.4f}, "
              f"clusters > 0.02 MDA: {r['n_clusters_mda_above_002']}/{r['n_clusters']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
