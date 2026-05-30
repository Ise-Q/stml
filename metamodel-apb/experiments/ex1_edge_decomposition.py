"""EX.1 — why is the meta-labelling OOS edge ≈0.5? (DIAGNOSTIC-ONLY)

Decomposes the weak edge per asset class on the MODELLING sample into:
  (a) primary-signal quality — the meta-label positive rate (how often the primary side was
      barrier-correct: the metamodel's job is only to skip the wrong calls);
  (b) CPCV robustness — the best model's 15-path AUC distribution and how many paths beat 0.5
      (characterising the Equity-vs-Energy split).
Detailed raw-signal stats live in EX.5. Nothing here feeds back into the locked pt_sl/max_holding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CLASSES, imputed_modelling_X, modelling_panel, results_dir  # noqa: E402

from alken_metamodel.cross_validation import CombinatorialPurgedCV  # noqa: E402
from alken_metamodel.evaluation import cross_val_evaluate  # noqa: E402
from alken_metamodel.models import balanced_sample_weight, tree_linear_roster  # noqa: E402
from alken_metamodel.pipeline import PipelineConfig  # noqa: E402
from alken_metamodel.seeding import set_seeds  # noqa: E402


def run() -> None:
    set_seeds(42)
    cfg = PipelineConfig(use_regime=False)
    out = ["# EX.1 — Decomposing the ≈0.5 OOS edge (diagnostic, modelling sample)\n"]
    for cls in CLASSES:
        pooled, cols, mask = modelling_panel(cls, cfg)
        X, y, t1 = imputed_modelling_X(pooled, cols, mask)
        sw = balanced_sample_weight(y, base=pooled["weight"].to_numpy()[mask])
        # (b) CPCV path distribution for the best tree/linear model
        cv = CombinatorialPurgedCV(cfg.cpcv_groups, cfg.cpcv_test_groups, t1, cfg.pct_embargo)
        best_mean, best_name, best_aucs = -np.inf, None, None
        for name in tree_linear_roster(seed=42):
            res = cross_val_evaluate(
                lambda name=name: tree_linear_roster(seed=42)[name], X, y, cv, sample_weight=sw
            )
            aucs = res["auc"].to_numpy()
            m = float(np.nanmean(aucs))
            if m > best_mean:
                best_mean, best_name, best_aucs = m, name, aucs
        paths_gt_half = int(np.nansum(best_aucs > 0.5))
        out.append(
            f"## {cls} — n={int(mask.sum())}\n"
            f"- (a) primary-signal barrier hit-rate (meta-label pos_rate): {y.mean():.3f}\n"
            f"- (b) best model `{best_name}`: mean CPCV AUC {best_mean:.4f}, "
            f"{paths_gt_half}/15 paths > 0.5\n"
        )
        print(f"{cls}: pos_rate={y.mean():.3f} best={best_name} "
              f"AUC={best_mean:.4f} paths>0.5={paths_gt_half}/15")
    (results_dir() / "ex1_edge_decomposition.md").write_text("\n".join(out))
    print("wrote results/ex1_edge_decomposition.md")


if __name__ == "__main__":
    run()
