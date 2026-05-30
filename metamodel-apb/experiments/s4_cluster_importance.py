"""S4.7 — cluster-level feature importance on the REAL pooled matrix (Energy -> Equity/Metals).

Runs §4's MDI + purged-MDA + cluster-SHAP on the real modelling matrix (previously synthetic-
only), per asset class. Reports the cluster structure, which clusters drive the metamodel, and
whether pure-noise clusters score ~0 on real data. Module reused unchanged (both bug-fixes in).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CLASSES, imputed_modelling_X, modelling_panel, results_dir  # noqa: E402

from alken_metamodel.cluster_importance import cluster_feature_importance  # noqa: E402
from alken_metamodel.pipeline import PipelineConfig  # noqa: E402
from alken_metamodel.seeding import set_seeds  # noqa: E402


def run() -> None:
    set_seeds(42)
    cfg = PipelineConfig(use_regime=False)
    out = ["# S4.7 — Cluster feature importance on the real matrix\n"]
    for cls in CLASSES:
        pooled, cols, mask = modelling_panel(cls, cfg)
        X, y, t1 = imputed_modelling_X(pooled, cols, mask)
        table, clusters = cluster_feature_importance(X, y, t1, seed=42, max_clusters=10)
        table = table.sort_values("mda", ascending=False)
        out.append(f"## {cls} — n={int(mask.sum())}, {X.shape[1]} feats, {len(clusters)} clu\n")
        out.append("```\n" + table.round(4).to_string() + "\n```\n")
        top_cid = int(table.index[0].split("_")[1])
        out.append(f"Top cluster `{table.index[0]}` members: {clusters.get(top_cid, [])[:10]}\n")
        noise = table[table["mda"].abs() < 0.02]
        out.append(f"Near-zero-MDA clusters (noise): {len(noise)} of {len(table)}\n")
        print(
            f"{cls}: top={table.index[0]} mda={table['mda'].iloc[0]:.4f} "
            f"shap={table['shap'].iloc[0]:.4f} clusters={len(clusters)} noise={len(noise)}"
        )
    (results_dir() / "s4_cluster_importance.md").write_text("\n".join(out))
    print("wrote results/s4_cluster_importance.md")


if __name__ == "__main__":
    run()
