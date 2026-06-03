"""sm/scope-router experiment — does dropping NG from energy training help?

Focused question. Compares 4 scope variants under CPCV(6,2) + per-instrument
embargo, on the existing Sreeram_experimental feature matrix:

    energy_all     = (cl1s, ho1s, rb1s, ng1s)   # baseline (already in POOL_MEMBERS)
    energy_ex_ng   = (cl1s, ho1s, rb1s)         # drop NG hypothesis
    cl_only        = (cl1s,)                    # crude individual (Harry-style)
    ng_separate    = (ng1s,)                    # NG isolated

Uses the existing ``champion_pipeline._evaluate_candidate`` (fair CPCV(6,2)
sample-weighted per-instrument AUC) so results are apples-to-apples vs
``baseline_xgb_per_class.csv`` and ``champions_summary.csv``.

Default model = ``lightgbm`` (energy class winner per ``baseline_xgb_per_class.csv``).
Default variant = ``without_bbg`` (since Sreeram's ablation shows without_bbg
slightly beats with_bbg on all 3 classes; we don't want to confound the scope
question with a feature-set wobble).

Outputs:
    results/sm_scope_router/scope_router_ledger.csv

Run via::

    uv run python -m stml.experimental.make_scope_router
    uv run python -m stml.experimental.make_scope_router --variant with_bbg
    uv run python -m stml.experimental.make_scope_router --models lightgbm xgboost
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from stml.experimental.champion_pipeline import (
    MIN_INDIVIDUAL_EVENTS,
    POOL_MEMBERS,
    _evaluate_candidate,
    _restrict_modelling,
    _select_feature_cols,
    _slice_pool,
)
from stml.experimental.config import PipelineConfig
from stml.experimental.make_scope import embargo_days_map

# ---------------------------------------------------------------------------
# Scope definitions — in-process extension of POOL_MEMBERS.
# energy_all already exists; we add the 3 new pools.
# ---------------------------------------------------------------------------
SCOPE_DEFS: dict[str, tuple[str, ...]] = {
    "energy_all":   ("cl1s", "ho1s", "rb1s", "ng1s"),
    "energy_ex_ng": ("cl1s", "ho1s", "rb1s"),
    "cl_only":      ("cl1s",),
    "ng_separate":  ("ng1s",),
}

# Per scope, which instruments we measure OOS AUC on.
SCOPE_TARGETS: dict[str, list[str]] = {
    "energy_all":   ["cl1s", "ho1s", "rb1s", "ng1s"],
    "energy_ex_ng": ["cl1s", "ho1s", "rb1s"],
    "cl_only":      ["cl1s"],
    "ng_separate":  ["ng1s"],
}


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def run_scope_router(
    *,
    features_path: Path,
    out_path: Path,
    cfg: PipelineConfig,
    variant: str,
    models: tuple[str, ...],
) -> pd.DataFrame:
    """Run the 4 scope variants × selected models. Append to ledger."""
    features = pd.read_parquet(features_path)
    print(f"Loaded features: {features.shape}")

    features = _restrict_modelling(features, cfg)
    print(f"After modelling restriction (≤{cfg.global_train_cut}): {features.shape}")

    embargo_map = embargo_days_map()
    feature_cols = _select_feature_cols(features, variant)
    print(f"Feature cols (variant={variant}): {len(feature_cols)}")

    # In-process extension of POOL_MEMBERS so _slice_pool can find our scopes.
    for scope, members in SCOPE_DEFS.items():
        if scope not in POOL_MEMBERS:
            POOL_MEMBERS[scope] = members

    rows: list[dict] = []
    for scope, members in SCOPE_DEFS.items():
        pool_df = _slice_pool(features, scope)
        n_modelling = len(pool_df)
        is_individual = len(members) == 1
        skipped = is_individual and n_modelling < MIN_INDIVIDUAL_EVENTS

        print(
            f"\n=== scope: {scope}  members={members}  n_pool={n_modelling}"
            f"{'  SKIPPED (<MIN_INDIVIDUAL_EVENTS=250)' if skipped else ''}"
        )

        for target in SCOPE_TARGETS[scope]:
            for model in models:
                if skipped:
                    rows.append({
                        "scope": scope,
                        "members": "+".join(members),
                        "instrument": target,
                        "model": model,
                        "variant": variant,
                        "n_modelling": n_modelling,
                        "n_oos": 0,
                        "mean_auc": float("nan"),
                        "sem": float("nan"),
                        "std_auc": float("nan"),
                        "lower_ci_1se": float("nan"),
                        "skipped": True,
                        "reason": f"individual pool < MIN_INDIVIDUAL_EVENTS={MIN_INDIVIDUAL_EVENTS}",
                        "elapsed_s": 0.0,
                    })
                    continue

                print(f"  → target={target}  model={model} ...")
                t0 = time.time()
                res = _evaluate_candidate(
                    instrument=target,
                    pool=scope,
                    model_name=model,
                    pool_df=pool_df,
                    feature_cols=feature_cols,
                    cfg=cfg,
                    nan_cols_at_test=None,
                    embargo_map=embargo_map,
                )
                elapsed = time.time() - t0
                lower_ci = (
                    res.mean_auc - res.sem
                    if not pd.isna(res.mean_auc) and not pd.isna(res.sem)
                    else float("nan")
                )
                print(
                    f"     AUC={res.mean_auc:.4f}±{res.sem:.4f}  "
                    f"n_oos={res.n_oos_for_instrument}  ({elapsed:.1f}s)"
                )
                rows.append({
                    "scope": scope,
                    "members": "+".join(members),
                    "instrument": target,
                    "model": model,
                    "variant": variant,
                    "n_modelling": res.n_modelling,
                    "n_oos": res.n_oos_for_instrument,
                    "mean_auc": res.mean_auc,
                    "sem": res.sem,
                    "std_auc": res.std_auc,
                    "lower_ci_1se": lower_ci,
                    "skipped": False,
                    "reason": "",
                    "elapsed_s": round(elapsed, 1),
                })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nWrote {out_path}")
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SCOPE-ROUTER LEDGER — per-instrument AUC by scope")
    print("=" * 80)
    keep_cols = [
        "scope", "instrument", "model", "n_modelling", "n_oos",
        "mean_auc", "sem", "lower_ci_1se",
    ]
    out = df.loc[~df["skipped"], keep_cols].copy()
    out["mean_auc"] = out["mean_auc"].map(lambda v: f"{v:.4f}")
    out["sem"] = out["sem"].map(lambda v: f"{v:.4f}")
    out["lower_ci_1se"] = out["lower_ci_1se"].map(lambda v: f"{v:.4f}")
    print(out.to_string(index=False))

    skipped = df.loc[df["skipped"], ["scope", "instrument", "reason"]]
    if not skipped.empty:
        print("\nSKIPPED runs:")
        print(skipped.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="sm/scope-router — focused NG-drop experiment")
    ap.add_argument(
        "--variant", default="without_bbg",
        choices=["with_bbg", "without_bbg"],
        help="Feature variant (default: without_bbg — Sreeram's ablation slight edge)",
    )
    ap.add_argument(
        "--models", nargs="+", default=["lightgbm"],
        help="Models to evaluate (default: lightgbm — energy class winner)",
    )
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    root = _find_repo_root()
    features_path = root / "data" / "sreeram_experimental_features.parquet"
    if not features_path.exists():
        print(f"FATAL: {features_path} missing")
        return 1

    out_path = root / "results" / "sm_scope_router" / "scope_router_ledger.csv"

    t_global = time.time()
    df = run_scope_router(
        features_path=features_path,
        out_path=out_path,
        cfg=cfg,
        variant=args.variant,
        models=tuple(args.models),
    )
    print(f"\nTotal elapsed: {time.time() - t_global:.1f}s")

    _print_summary(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
