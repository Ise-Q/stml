"""S3-fix runner — champion architecture per instrument.

Plan §4.3 / — per-instrument + alternative-pool candidate search,
champion selected by mean AUC + 1SE rule.

Produces:
* ``results/submission/champions_with_bbg.csv``
* ``results/submission/champions_without_bbg.csv``
* ``results/submission/champions_summary.csv``
* ``results/submission/champions_per_pool_per_model.csv``
   (the full 4-model × pool grid per instrument — like the master_results)

Run via::

    uv run python -m stml.experimental.make_champions
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stml.experimental.champion_pipeline import (
    INSTRUMENT_REGIMES,
    ChampionResult,
    POOL_MEMBERS,
    run_all_champions,
)
from stml.experimental.config import PipelineConfig


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def champions_to_summary_df(champions: dict[str, ChampionResult], variant: str) -> pd.DataFrame:
    rows = []
    for inst, r in champions.items():
        rows.append({
            "instrument": inst,
            "variant": variant,
            "winning_pool": r.winning_pool,
            "winning_model": r.winning_model,
            "champion_auc": r.champion_auc,
            "champion_sem": r.champion_sem,
            "lower_ci_1se": r.champion_auc - r.champion_sem
            if not np.isnan(r.champion_auc) and not np.isnan(r.champion_sem) else float("nan"),
            "selection_reason": r.selection_reason,
        })
    return pd.DataFrame(rows)


def champions_to_grid_df(champions: dict[str, ChampionResult], variant: str) -> pd.DataFrame:
    rows = []
    for inst, r in champions.items():
        for c in r.candidates:
            rows.append({
                "instrument": inst,
                "variant": variant,
                "pool": c.pool,
                "n_pool_members": len(POOL_MEMBERS.get(c.pool, ())),
                "model": c.model_name,
                "n_modelling": c.n_modelling,
                "n_oos_for_instrument": c.n_oos_for_instrument,
                "mean_auc": c.mean_auc,
                "std_auc": c.std_auc,
                "sem": c.sem,
                "is_champion": (c.pool == r.winning_pool and c.model_name == r.winning_model),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S3-fix: champion architecture per instrument")
    ap.add_argument("--variants", nargs="+",
                    default=["with_bbg", "without_bbg", "simulated_missingness"],
                    help="Ablation variants to run.")
    ap.add_argument("--no-persist", action="store_true")
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    root = _find_repo_root()
    features_path = root / "data" / "features.parquet"
    if not features_path.exists():
        print(f"FATAL: {features_path} missing — run S2 first.")
        return 1

    features = pd.read_parquet(features_path)
    print(f"Loaded features: {features.shape}")

    summaries: list[pd.DataFrame] = []
    grids: list[pd.DataFrame] = []
    decisions: dict[str, dict[str, ChampionResult]] = {}

    for variant in args.variants:
        print(f"\n{'='*80}\nVARIANT: {variant}\n{'='*80}")
        t0 = time.time()
        champions = run_all_champions(features, cfg=cfg, variant=variant, verbose=True)
        decisions[variant] = champions
        t1 = time.time()
        print(f"\n  Variant {variant} took {t1-t0:.1f}s")
        summaries.append(champions_to_summary_df(champions, variant))
        grids.append(champions_to_grid_df(champions, variant))

    summary_df = pd.concat(summaries, ignore_index=True)
    grid_df = pd.concat(grids, ignore_index=True)

    # --- Acceptance gate verdict ---
    print("\n\n" + "="*80)
    print("Plan §8 S3 acceptance gates — CHAMPION ARCHITECTURE")
    print("="*80)
    with_bbg = summary_df.loc[summary_df["variant"] == "with_bbg"].set_index("instrument")

    n_above_055 = int((with_bbg["champion_auc"] > 0.55).sum())
    n_above_060 = int((with_bbg["champion_auc"] > 0.60).sum())
    n_with_sig = int((with_bbg["lower_ci_1se"] > 0.50).sum())
    print(f"\n  Per-instrument AUC > 0.55: {n_above_055} / 11  (S3 gate: ≥6)  "
          f"{'PASS' if n_above_055 >= 6 else 'CHECK'}")
    print(f"  Per-instrument AUC > 0.60: {n_above_060} / 11")
    print(f"  Signal-flag (lower 1-SE CI > 0.50): {n_with_sig} / 11")

    # Per-class pooled AUC — weighted mean of champion AUCs.
    from stml.experimental.config import ASSET_CLASS_MEMBERS

    for cls, members in ASSET_CLASS_MEMBERS.items():
        cls_aucs = with_bbg.loc[with_bbg.index.isin(members), "champion_auc"]
        if cls_aucs.empty:
            continue
        mean_cls = cls_aucs.mean()
        print(f"  {cls}: mean champion AUC = {mean_cls:.4f}  (reference: equity 0.579, energy 0.525, metals 0.530)")

    print("\nPer-instrument champion summary (with_bbg):")
    cols = ["instrument", "winning_pool", "winning_model", "champion_auc", "champion_sem",
            "lower_ci_1se", "selection_reason"]
    print(with_bbg.reset_index().loc[:, cols].to_string(index=False))

    if not args.no_persist:
        out = root / "results" / "submission"
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "champions_summary.csv"
        grid_path = out / "champions_per_pool_per_model.csv"
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")
        grid_df.to_csv(grid_path, index=False, float_format="%.6f")
        print(f"\nWrote {summary_path.relative_to(root)}")
        print(f"Wrote {grid_path.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
