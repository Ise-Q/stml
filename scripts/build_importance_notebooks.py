"""Generate per-asset-class importance notebooks matching Harry's structure.

Outputs notebooks/sreeram_experimental/{equity,energy,metals}_importance.ipynb
each loading the artifacts emitted by make_importance_deep.py.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _cell(kind: str, source: str | list[str], outputs: list | None = None) -> dict:
    src = source if isinstance(source, list) else [source]
    cell: dict = {
        "cell_type": kind,
        "metadata": {},
        "source": src,
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs or []
    return cell


def build_notebook(asset_class: str) -> dict:
    cls = asset_class.upper()
    cells = [
        _cell("markdown", [
            f"# {cls} — Feature Importance Analysis (Sreeram_experimental)\n",
            "\n",
            "Per-asset-class cluster importance under the Jay-CSV labels.\n",
            "Follows Harry's workflow: cluster MDA + MDI + SHAP, cross-method\n",
            "rank agreement, within-cluster PCA + SHAP breakdown for top clusters,\n",
            "global per-feature SHAP, and a pruned-vs-full CPCV(6,2) AUC\n",
            "comparison.\n",
            "\n",
            "All artifacts come from\n",
            f"`results/sreeram_experimental/importance/{asset_class}/` — produced\n",
            "by `python -m stml.experimental.make_importance` (cluster summary)\n",
            "and `python -m stml.experimental.make_importance_deep` (within-cluster\n",
            "+ global SHAP + pruned-model comparison).\n",
        ]),
        _cell("code", [
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "pd.set_option('display.max_columns', None)\n",
            "pd.set_option('display.width', 240)\n",
            "\n",
            f"ASSET_CLASS = {asset_class!r}\n",
            "REPO = Path.cwd()\n",
            "while not (REPO / 'pyproject.toml').exists():\n",
            "    REPO = REPO.parent\n",
            "OUT = REPO / 'results' / 'sreeram_experimental' / 'importance' / ASSET_CLASS\n",
            "print('Loading from:', OUT)\n",
        ]),
        _cell("markdown", [
            "## 1. Cluster cross-check\n",
            "\n",
            "Cluster ranks under each importance method, plus the\n",
            "`significant` flag (MDA mean > 1 sigma above zero -> lower CI > 0).\n",
        ]),
        _cell("code", [
            "crosscheck = pd.read_csv(OUT / 'cluster_crosscheck_table.csv')\n",
            "crosscheck\n",
        ]),
        _cell("code", [
            "fig, ax = plt.subplots(figsize=(9, 4))\n",
            "x = np.arange(len(crosscheck))\n",
            "ax.bar(x, crosscheck['mda_mean'],\n",
            "       color=['#0F766E' if s else '#94a3b8' for s in crosscheck['significant']])\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels([f'C{int(c)}' for c in crosscheck['cluster_id']], rotation=45)\n",
            "ax.axhline(0, color='black', linewidth=0.5)\n",
            "ax.set_ylabel('Clustered MDA (mean)')\n",
            f"ax.set_title('{cls} — Cluster MDA (green = significant)')\n",
            "plt.tight_layout(); plt.show()\n",
        ]),
        _cell("markdown", [
            "## 2. Kendall rank agreement (MDA vs MDI vs SHAP)\n",
            "\n",
            "Rank agreement across the three importance methods.\n",
            "tau ~ 0.7+: methods agree (the typical pattern is MDI <-> SHAP).\n",
            "tau < 0.4: methods disagree (typical: MDA <-> SHAP, since MDA is\n",
            "the out-of-sample reality check).\n",
        ]),
        _cell("code", [
            "rank_agree = pd.read_csv(OUT / 'rank_agreement.csv')\n",
            "rank_agree\n",
        ]),
        _cell("markdown", [
            "## 3. Within-cluster breakdown (top clusters by MDA)\n",
            "\n",
            "For each top cluster: cluster members ranked by mean |SHAP|,\n",
            "plus the PC1/PC2/PC3 loadings from a PCA fit on standardised\n",
            "members. `pca_pc1_var_explained` >= 65% indicates a single latent\n",
            "dimension (one representative suffices); 40-65% indicates a dominant\n",
            "direction; < 40% indicates multi-dimensional structure.\n",
        ]),
        _cell("code", [
            "import glob\n",
            "wc_paths = sorted(glob.glob(str(OUT / 'within_cluster_C*.csv')))\n",
            "wc_paths\n",
        ]),
        _cell("code", [
            "for p in wc_paths:\n",
            "    name = Path(p).stem.replace('within_cluster_', '')\n",
            "    df = pd.read_csv(p)\n",
            "    pc1 = df['pca_pc1_var_explained'].iloc[0] * 100\n",
            "    pc2 = df['pca_pc2_var_explained'].iloc[0] * 100 if 'pca_pc2_var_explained' in df.columns else 0\n",
            "    print(f'--- {name}: PC1 = {pc1:.1f}%, PC1+PC2 = {pc1+pc2:.1f}%, {len(df)} members ---')\n",
            "    display(df)\n",
        ]),
        _cell("markdown", [
            "## 4. Global per-feature SHAP\n",
            "\n",
            "Mean |SHAP|, signed mean SHAP, and MDI per feature on the\n",
            "asset-class modelling sample. Top features here that aren't in any\n",
            "significant cluster are candidates for the pruned model.\n",
        ]),
        _cell("code", [
            "gshap = pd.read_csv(OUT / 'global_shap_summary.csv')\n",
            "display(gshap.head(25))\n",
        ]),
        _cell("code", [
            "fig, ax = plt.subplots(figsize=(7, 7))\n",
            "top = gshap.head(20).iloc[::-1]\n",
            "colors = ['#0F766E' if v > 0 else '#B45309' for v in top['shap_signed']]\n",
            "ax.barh(range(len(top)), top['shap_magnitude'], color=colors)\n",
            "ax.set_yticks(range(len(top)))\n",
            "ax.set_yticklabels(top['feature'])\n",
            "ax.set_xlabel('mean |SHAP|')\n",
            f"ax.set_title('{cls} — top-20 features by mean |SHAP|')\n",
            "plt.tight_layout(); plt.show()\n",
        ]),
        _cell("markdown", [
            "## 5. Pruned model — selection rule\n",
            "\n",
            "Selection rule (per Harry / user spec):\n",
            "\n",
            "1. For each significant cluster (MDA mean > 1 sigma):\n",
            "   - PC1 >= 65% -> one representative (the top SHAP member).\n",
            "   - PC1 + PC2 >= 75% -> two representatives.\n",
            "   - Otherwise -> top-2 by mean |SHAP|.\n",
            "2. Plus the global top-5 SHAP features not in any chosen cluster.\n",
        ]),
        _cell("code", [
            "with open(OUT / 'pruned_features.json') as f:\n",
            "    pruned = json.load(f)\n",
            "print(f\"Pruned feature set ({pruned['n_pruned']} features):\")\n",
            "for feat in pruned['features']:\n",
            "    print(f\"  - {feat:<35}  {pruned['reasons'][feat]}\")\n",
        ]),
        _cell("markdown", [
            "## 6. Full vs pruned model — honest held-out val AUC\n",
            "\n",
            "Importance + global SHAP + cluster selection all computed on the\n",
            "**train** partition only. Pruned features picked from train.\n",
            "Both models then fit on **train** and scored on the **held-out val**\n",
            "partition (never used for selection). The val AUC reported here is\n",
            "an honest out-of-sample estimate of the pruning effect.\n",
        ]),
        _cell("code", [
            "cmp = pd.read_csv(OUT / 'pruned_vs_full_auc.csv')\n",
            "cmp\n",
        ]),
        _cell("code", [
            "fig, ax = plt.subplots(figsize=(7, 3.5))\n",
            "ax.bar(cmp['model'], cmp['val_auc'],\n",
            "       yerr=cmp['val_sem'],\n",
            "       color=['#94a3b8', '#0F766E'])\n",
            "ax.axhline(0.5, color='black', linewidth=0.5, linestyle='--')\n",
            "ax.set_ylim(0.45, max(cmp['val_auc']) + 0.04)\n",
            "for i, (m, a, n) in enumerate(zip(cmp['model'], cmp['val_auc'], cmp['n_features'])):\n",
            "    ax.text(i, a + 0.005, f'{a:.4f}\\n(n={int(n)})', ha='center', fontsize=9)\n",
            "ax.set_ylabel('Held-out val AUC (train fit)')\n",
            f"ax.set_title('{cls} — full vs pruned model')\n",
            "plt.tight_layout(); plt.show()\n",
        ]),
        _cell("markdown", [
            "## 7. Findings note (plain-text summary)\n",
            "\n",
            "Auto-generated summary: significant clusters, top SHAP features,\n",
            "the pruned feature list, and the CPCV AUC comparison.\n",
        ]),
        _cell("code", [
            "print((OUT / 'findings_note.txt').read_text())\n",
        ]),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_dir = REPO / "notebooks" / "sreeram_experimental"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in ("equity", "energy", "metals"):
        nb = build_notebook(cls)
        path = out_dir / f"{cls}_importance.ipynb"
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"Wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
