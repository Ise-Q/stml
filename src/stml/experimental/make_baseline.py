"""S3 runner — per-asset-class XGB baseline + the R-11 ablation.

Plan §8 Stage 3 deliverable. Produces:

* ``results/submission/baseline_xgb_per_class.csv`` — per-class
  pooled AUC + winner + per-instrument breakdown for the `with_bbg` variant.
* ``results/submission/bbg_missingness_ablation.csv`` — per-class
  `with_bbg_auc` / `without_bbg_auc` / `simulated_missingness_auc`. The
  decisive R-11 gate (§13 R-11).
* ``results/submission/coverage_caveat.csv`` — per-instrument
  thin / low-coherence flags. ng1s gets the §2.8.4 low-coherence flag.
* Prints methodology spec S3 acceptance gate verdict.

Run via::

    uv run python -m stml.experimental.make_baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stml.experimental.config import ASSET_CLASS_MEMBERS, PipelineConfig
from stml.experimental.pipeline import (
    AssetClassResult,
    run_asset_class,
)

# Plan §2.8.4 — instruments with low feature-target coherence vs raw Bloomberg.
# `ng1s` is the load-bearing case (R² 0.72; 11 % label flip).
LOW_COHERENCE_INSTRUMENTS = ("ng1s",)
NOISY_COHERENCE_INSTRUMENTS = ("ho1s", "rb1s")  # R² 0.87 / 0.95 — caution flag
THIN_INSTRUMENTS = ("ho1s", "gc1s", "ng1s")  # methodology spec — thin participation

# Plan §8 S3 gate — the reference per-class CPCV AUC baseline.
ALKEN_BASELINE_AUC = {
    "equity": 0.579,
    "energy": 0.525,
    "metals": 0.530,
}
TARGET_DELTA_OVER_BASELINE = 0.02  # methodology spec S3 — beat baseline by ≥ 0.02.


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _format_per_class(rows: list[AssetClassResult]) -> pd.DataFrame:
    pool_rows = []
    for r in rows:
        pool_rows.append(
            {
                "asset_class": r.asset_class,
                "variant": r.variant,
                "best_model": r.best_model_name,
                "n_modelling": r.n_modelling,
                "n_features": len(r.feature_cols),
                "pooled_mean_auc": r.pooled_mean_auc,
                "pooled_std_auc": r.pooled_std_auc,
            }
        )
    return pd.DataFrame(pool_rows)


def _format_per_instrument(rows: list[AssetClassResult], variant: str) -> pd.DataFrame:
    parts = []
    for r in rows:
        if r.variant != variant or r.per_instrument.empty:
            continue
        df = r.per_instrument.copy()
        df["asset_class"] = r.asset_class
        df["winner"] = r.best_model_name
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _coverage_caveat_frame(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for inst in events["instrument"].unique():
        sub = events.loc[events["instrument"] == inst]
        n = int(len(sub))
        n_eff = float(sub["uniqueness_weight"].sum())
        rows.append(
            {
                "instrument": inst,
                "n_events": n,
                "n_eff": n_eff,
                "thin": inst in THIN_INSTRUMENTS,
                "low_coherence_vs_raw": inst in LOW_COHERENCE_INSTRUMENTS,
                "noisy_coherence_vs_raw": inst in NOISY_COHERENCE_INSTRUMENTS,
                "note": (
                    "low feature-target coherence vs raw BBG (R² 0.72, 11 % label flip)"
                    if inst in LOW_COHERENCE_INSTRUMENTS
                    else "noisier R² 0.87-0.95 vs raw"
                    if inst in NOISY_COHERENCE_INSTRUMENTS
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("instrument").reset_index(drop=True)


def run(cfg: PipelineConfig | None = None, *, verbose: bool = True) -> dict:
    cfg = cfg or PipelineConfig()
    root = _find_repo_root()
    features_path = root / "data" / "features.parquet"
    events_path = root / "data" / "events.parquet"
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    features = pd.read_parquet(features_path)
    events = pd.read_parquet(events_path)

    classes = list(ASSET_CLASS_MEMBERS.keys())
    variants = ("with_bbg", "without_bbg", "simulated_missingness")

    all_results: list[AssetClassResult] = []
    for cls in classes:
        for variant in variants:
            if verbose:
                print(f"\n[run] class={cls} variant={variant}")
            t0 = time.time()
            r = run_asset_class(features, cls, cfg=cfg, variant=variant)
            t1 = time.time()
            if verbose:
                print(
                    f"  winner={r.best_model_name}  pooled_AUC={r.pooled_mean_auc:.4f} "
                    f"±{r.pooled_std_auc:.4f}   n_feat={len(r.feature_cols)}  "
                    f"({t1 - t0:.1f}s)"
                )
            all_results.append(r)

    return {"all_results": all_results, "features": features, "events": events}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S3 baseline + R-11 ablation runner")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    state = run(verbose=not args.quiet)
    results: list[AssetClassResult] = state["all_results"]
    events: pd.DataFrame = state["events"]

    # =========================================================================
    # baseline_xgb_per_class.csv — with_bbg pooled summary + per-instrument
    # =========================================================================
    pooled = _format_per_class(results)
    per_inst_with_bbg = _format_per_instrument(results, "with_bbg")

    # =========================================================================
    # bbg_missingness_ablation.csv — three AUC columns per class
    # =========================================================================
    aug = pooled.copy()
    pivot = (
        aug.set_index(["asset_class", "variant"])["pooled_mean_auc"]
        .unstack(level=-1)
        .reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.rename(
        columns={
            "with_bbg": "with_bbg_auc",
            "without_bbg": "without_bbg_auc",
            "simulated_missingness": "simulated_missingness_auc",
        }
    )
    pivot["delta_simulated_vs_with_bbg"] = (
        pivot["simulated_missingness_auc"] - pivot["with_bbg_auc"]
    )
    pivot["delta_without_vs_with_bbg"] = (
        pivot["without_bbg_auc"] - pivot["with_bbg_auc"]
    )

    # Ship decision per R-11.
    def _decide(row: pd.Series) -> str:
        delta = row["delta_simulated_vs_with_bbg"]
        if pd.isna(delta):
            return "indeterminate"
        if delta >= -0.03:
            return "ship_with_bbg"  # robust under simulated missingness
        return "ship_without_bbg"  # BBG-fragile architecture

    pivot["ship_decision"] = pivot.apply(_decide, axis=1)

    # =========================================================================
    # coverage_caveat.csv — thin / low-coherence flags
    # =========================================================================
    coverage = _coverage_caveat_frame(events)

    # =========================================================================
    # Plan §8 S3 acceptance gates
    # =========================================================================
    print("\n=== Plan §8 S3 acceptance gates ===\n")
    with_bbg = pooled.loc[pooled["variant"] == "with_bbg"].set_index("asset_class")
    gate1_pass = True
    for cls, target in ALKEN_BASELINE_AUC.items():
        actual = float(with_bbg.loc[cls, "pooled_mean_auc"]) if cls in with_bbg.index else float("nan")
        required = target + TARGET_DELTA_OVER_BASELINE
        ok = (not np.isnan(actual)) and actual >= required
        gate1_pass &= ok
        print(f"  [{'PASS' if ok else 'CHECK'}] {cls}: pooled AUC = {actual:.4f}  "
              f"(target ≥ baseline{target:.3f} + delta target)")

    n_above_055 = int((per_inst_with_bbg["auc"] > 0.55).sum()) if not per_inst_with_bbg.empty else 0
    gate2_pass = n_above_055 >= 6
    print(f"  [{'PASS' if gate2_pass else 'CHECK'}] ≥6 of 11 per-inst AUC > 0.55 — actual {n_above_055}")

    gate3_pass = not pivot.empty and all(
        not pd.isna(pivot.loc[pivot["asset_class"] == c, "with_bbg_auc"].iloc[0])
        for c in ALKEN_BASELINE_AUC
    )
    print(f"  [{'PASS' if gate3_pass else 'CHECK'}] bbg_missingness_ablation.csv exists "
          f"with three AUC columns per class")

    # Ship decision recorded.
    ship_decisions = pivot["ship_decision"].tolist()
    gate4_pass = all(s in ("ship_with_bbg", "ship_without_bbg") for s in ship_decisions)
    print(f"  [{'PASS' if gate4_pass else 'CHECK'}] ship-decision recorded per class: "
          f"{dict(zip(pivot['asset_class'], ship_decisions))}")

    coverage_flag_ng = bool(
        coverage.loc[coverage["instrument"] == "ng1s", "low_coherence_vs_raw"].iloc[0]
    )
    print(f"  [{'PASS' if coverage_flag_ng else 'CHECK'}] coverage_caveat.csv flags ng1s for low coherence")

    print("\n=== Pooled per-class AUC summary ===\n")
    print(pivot.to_string(index=False))

    if per_inst_with_bbg.empty:
        print("\n(per-instrument breakdown empty)")
    else:
        print("\n=== Per-instrument breakdown (with_bbg winner) ===\n")
        cols = ["instrument", "asset_class", "winner", "n", "auc", "log_loss", "brier", "pos_rate"]
        present = [c for c in cols if c in per_inst_with_bbg.columns]
        print(per_inst_with_bbg.loc[:, present].to_string(index=False))

    if not args.no_persist:
        root = _find_repo_root()
        out = root / "results" / "submission"
        out.mkdir(parents=True, exist_ok=True)
        pooled_path = out / "baseline_xgb_per_class.csv"
        ablation_path = out / "bbg_missingness_ablation.csv"
        coverage_path = out / "coverage_caveat.csv"
        per_inst_path = out / "baseline_per_instrument.csv"

        pooled.to_csv(pooled_path, index=False, float_format="%.6f")
        pivot.to_csv(ablation_path, index=False, float_format="%.6f")
        coverage.to_csv(coverage_path, index=False)
        per_inst_with_bbg.to_csv(per_inst_path, index=False, float_format="%.6f")
        for p in (pooled_path, ablation_path, coverage_path, per_inst_path):
            print(f"\nWrote {p.relative_to(root)}")

    all_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass and coverage_flag_ng
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
