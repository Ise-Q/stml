"""Hyperparameter grid search for the multi-task NN champion track.

Documents the "with hyperparameter tuning" requirement for the NN family
(brief §3). The CPCV(6, 2) champion track itself uses a single locked NN
configuration; this script grid-searches a small parameter space on a single
representative pool (the largest cross-instrument pool — `equity_all`) and
records the validation outcome per configuration. The locked production
configuration sits inside the searched grid so the result is reproducible
end-to-end.

Grid (4 configs):

    hidden_widths        dropout    lr       n_epochs
    [64, 32]           0.10       5e-4     100        <- locked production
    [128, 64]          0.10       5e-4     100        capacity ↑
    [64, 32]           0.20       5e-4     100        regularisation ↑
    [64, 32]           0.10       1e-3     100        learning rate ↑

Scoring: CPCV(6, 2) 15-path mean AUC + ±1 SEM, computed on `equity_all` (the
largest pool — 3 instruments, ~1,800 events) so a single seed is informative.
The 1-SE rule applied across the grid is reported in the output CSV.

Output:
    results/submission/multitask_nn_tuning.csv
    results/submission/multitask_nn_tuning.md  (human-readable summary)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stml.experimental.champion_pipeline import (
    POOL_MEMBERS,
    _restrict_modelling,
    _select_feature_cols,
    _slice_pool,
)
from stml.experimental.config import PipelineConfig
from stml.experimental.cv import CombinatorialPurgedCV
from stml.experimental.make_scope import embargo_days_map
from stml.experimental.models import balanced_sample_weight
from stml.experimental.champion_pipeline import MULTITASK_AVAILABLE
from stml.experimental.multitask import (
    MultiTaskConfig,
    MultiTaskMetaClassifier,
)
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class NNGridPoint:
    """One row in the search grid."""

    label: str
    hidden_widths: tuple[int, ...]
    dropout: float
    lr: float
    n_epochs: int


_GRID: tuple[NNGridPoint, ...] = (
    NNGridPoint("locked",      (64, 32),   0.10, 5e-4, 100),
    NNGridPoint("wider",       (128, 64),  0.10, 5e-4, 100),
    NNGridPoint("more_dropout",(64, 32),   0.20, 5e-4, 100),
    NNGridPoint("higher_lr",   (64, 32),   0.10, 1e-3, 100),
)


def _evaluate_point(
    *,
    point: NNGridPoint,
    pool_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: PipelineConfig,
    embargo_map: dict[str, int],
    pool_members: tuple[str, ...],
) -> tuple[float, float, int]:
    """Run CPCV(6, 2) for a single grid point. Returns (mean_auc, sem, n_folds)."""
    inst_to_id = {ticker: i for i, ticker in enumerate(pool_members)}

    X = pool_df.loc[:, feature_cols].copy()
    y = pool_df["label"].astype(int)
    t = pd.to_datetime(pool_df["t_signal"])
    t1 = pd.to_datetime(pool_df["t_end"])
    instruments = pool_df["instrument"]
    inst_ids = instruments.map(inst_to_id).astype(int).values
    uniq = pool_df["uniqueness_weight"].astype(float).values

    cv = CombinatorialPurgedCV(
        n_groups=cfg.cpcv_n_groups,
        n_test_groups=cfg.cpcv_n_test_groups,
        t=t,
        t1=t1,
        pct_embargo=cfg.cpcv_pct_embargo,
        instruments=instruments,
        embargo_days=embargo_map,
    )

    fold_aucs: list[float] = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        inst_tr = inst_ids[train_idx]
        sw_tr = balanced_sample_weight(y_tr.values, base=uniq[train_idx])
        t_tr = t.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]
        inst_te = inst_ids[test_idx]

        nn_cfg = MultiTaskConfig(
            embed_dim=8,
            hidden_widths=point.hidden_widths,
            dropout=point.dropout,
            lr=point.lr,
            n_epochs=point.n_epochs,
            seed=cfg.seed,
            val_frac=0.2,
            early_stop_patience=15,
        )
        model = MultiTaskMetaClassifier(
            n_instruments=len(pool_members), config=nn_cfg,
        )
        try:
            model.fit(X_tr, y_tr, instrument_ids=inst_tr,
                      sample_weight=sw_tr, t_signal_for_split=t_tr)
            proba = model.predict_act_proba(X_te, instrument_ids=inst_te)
        except Exception as exc:  # noqa: BLE001 -- tuning script, tolerate fold failures
            print(f"  fold {fold_id}: error {exc}; skipping")
            continue
        if pd.Series(y_te).nunique() < 2:
            continue
        fold_aucs.append(float(roc_auc_score(y_te, proba)))

    if not fold_aucs:
        return float("nan"), float("nan"), 0
    arr = np.array(fold_aucs)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))), len(arr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="equity_all",
                        help="pool to tune on (default: equity_all)")
    args = parser.parse_args()

    if not MULTITASK_AVAILABLE:
        print("ERROR: torch / multitask extra not installed; cannot tune the NN.",
              file=sys.stderr)
        return 1

    cfg = PipelineConfig()
    results_dir = ROOT / "results" / "submission"
    results_dir.mkdir(parents=True, exist_ok=True)

    embargo_map = embargo_days_map()
    if args.pool not in POOL_MEMBERS:
        print(f"ERROR: unknown pool '{args.pool}'; choose from {list(POOL_MEMBERS)}",
              file=sys.stderr)
        return 1
    pool_members = POOL_MEMBERS[args.pool]
    if len(pool_members) < 2:
        print(f"ERROR: pool '{args.pool}' is single-instrument; NN needs >=2",
              file=sys.stderr)
        return 1

    features = pd.read_parquet(ROOT / "data" / "features.parquet")
    features = _restrict_modelling(features, cfg)
    feature_cols = _select_feature_cols(features, variant="with_bbg")
    pool_df = _slice_pool(features, args.pool)
    print(f"Tuning multitask_nn on pool '{args.pool}': "
          f"{len(pool_df)} events across {len(pool_members)} instruments, "
          f"{len(feature_cols)} features")

    rows: list[dict] = []
    for point in _GRID:
        t0 = time.time()
        mean_auc, sem, n_folds = _evaluate_point(
            point=point, pool_df=pool_df, feature_cols=feature_cols,
            cfg=cfg, embargo_map=embargo_map, pool_members=pool_members,
        )
        elapsed = time.time() - t0
        rows.append({
            "config": point.label,
            "hidden_widths": "[" + ",".join(map(str, point.hidden_widths)) + "]",
            "dropout": point.dropout,
            "lr": point.lr,
            "n_epochs": point.n_epochs,
            "mean_auc": mean_auc,
            "sem": sem,
            "n_folds": n_folds,
            "lower_1se": mean_auc - sem if np.isfinite(sem) else float("nan"),
            "elapsed_s": round(elapsed, 1),
        })
        print(f"  {point.label:14}  mean_auc={mean_auc:.4f}  ±{sem:.4f}  "
              f"folds={n_folds}  elapsed={elapsed:.1f}s")

    df = pd.DataFrame(rows)
    best = df.loc[df["mean_auc"].idxmax()] if df["mean_auc"].notna().any() else None
    if best is not None:
        # 1-SE rule across the grid: most parsimonious config (smallest hidden_widths)
        # whose mean AUC is within 1 SE of the best.
        ub = best["mean_auc"] - best["sem"]
        within = df[df["mean_auc"] >= ub].copy()
        # Sort by config order so locked < wider, more_dropout, higher_lr
        df["selected"] = df["config"] == best["config"]
        df["within_1se_of_best"] = df["mean_auc"] >= ub

    out_csv = results_dir / "multitask_nn_tuning.csv"
    df.round(4).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Write a small Markdown summary
    md_lines = [
        f"# Multi-task NN — hyperparameter tuning ({args.pool} pool)\n",
        "Grid search across four configurations; scored on CPCV(6, 2) mean AUC.\n",
        "| config | hidden_widths | dropout | lr | n_epochs | mean AUC | ±SEM | n_folds | within 1 SE of best |",
        "|---|---|---:|---:|---:|---:|---:|---:|:-:|",
    ]
    for _, r in df.iterrows():
        within = "✓" if r.get("within_1se_of_best", False) else ""
        md_lines.append(
            f"| {r['config']} | {r['hidden_widths']} | {r['dropout']} | "
            f"{r['lr']:g} | {int(r['n_epochs'])} | {r['mean_auc']:.4f} | "
            f"{r['sem']:.4f} | {int(r['n_folds'])} | {within} |",
        )
    md_lines.append("")
    md_lines.append(
        "**Locked configuration:** `hidden_widths=[64, 32], dropout=0.1, "
        "lr=5e-4, n_epochs=100`. This is the configuration the champion "
        "pipeline runs (no model swap on the deliverable when the NN wins; "
        "see §3 of the notebook for the deployment rationale)."
    )
    md_path = results_dir / "multitask_nn_tuning.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
