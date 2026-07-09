"""sm/scope-router — full routing tournament.

Extends the MVP ``scope_router_ledger.csv`` (4 scopes × 1-2 models × 1 variant)
into a comprehensive routing tournament:

    25 scopes × 4 models × 2 feature variants  →  router_full_ledger.csv

Plus a per-instrument selection summary: for each of the 11 instruments, the
best validated (pool, model, feature_variant) combination, ranked by
``lower_CI_1SE`` primary, then ``mean_auc``, then simplicity (more regularised
model first, larger pool first).

This script does NOT modify ``scope_router_ledger.csv`` (the MVP checkpoint).
Output goes to:

    results/sm_scope_router/router_full_ledger.csv
    results/sm_scope_router/router_selection_summary.csv

Re-uses ``champion_pipeline._evaluate_candidate`` (same CPCV(6,2) + per-instrument
embargo + sample-weighted per-fold AUC as ``make_baseline.py``). No pipeline
modification — POOL_MEMBERS is extended in-process.

Run::

    uv run python -m stml.experimental.make_scope_router_full
    uv run python -m stml.experimental.make_scope_router_full --models lightgbm xgboost
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
from stml.experimental.config import (
    INSTRUMENT_TO_CLASS,
    INSTRUMENTS,
    PipelineConfig,
)
from stml.experimental.make_scope import embargo_days_map

# ---------------------------------------------------------------------------
# Full scope grid — 25 scopes across the three asset classes.
# In-process additions to POOL_MEMBERS (no pipeline modification).
# ---------------------------------------------------------------------------
SCOPE_DEFS: dict[str, tuple[str, ...]] = {
    # ====== Energy (10) ======
    "energy_all":   ("cl1s", "ho1s", "rb1s", "ng1s"),
    "energy_ex_ng": ("cl1s", "ho1s", "rb1s"),   # == cl_ho_rb
    "cl_only":      ("cl1s",),
    "ho_only":      ("ho1s",),
    "rb_only":      ("rb1s",),
    "ng_separate":  ("ng1s",),
    "cl_ho":        ("cl1s", "ho1s"),
    "cl_rb":        ("cl1s", "rb1s"),
    "ho_rb":        ("ho1s", "rb1s"),
    # ====== Equity (7) ======
    "equity_all":   ("es1s", "nq1s", "fesx1s"),
    "es_only":      ("es1s",),
    "nq_only":      ("nq1s",),
    "fesx_only":    ("fesx1s",),
    "us_equity":    ("es1s", "nq1s"),
    "es_fesx":      ("es1s", "fesx1s"),
    "nq_fesx":      ("nq1s", "fesx1s"),
    # ====== Metals (9) ======
    "metals_all":   ("gc1s", "si1s", "hg1s", "pl1s"),
    "precious":     ("gc1s", "si1s", "pl1s"),
    "gc_only":      ("gc1s",),
    "si_only":      ("si1s",),
    "hg_only":      ("hg1s",),
    "pl_only":      ("pl1s",),
    "gc_si":        ("gc1s", "si1s"),
    "si_pl":        ("si1s", "pl1s"),
    "metals_ex_gc": ("si1s", "hg1s", "pl1s"),
}

# Each scope evaluates per-instrument AUC on every member of its pool.
SCOPE_TARGETS: dict[str, tuple[str, ...]] = {k: v for k, v in SCOPE_DEFS.items()}

DEFAULT_MODELS = ("elasticnet_logistic", "random_forest", "xgboost", "lightgbm")
DEFAULT_VARIANTS = ("without_bbg", "with_bbg")

# Simplicity ranking — lower = simpler / more regularised.
MODEL_RANK = {"elasticnet_logistic": 0, "random_forest": 1, "lightgbm": 2, "xgboost": 3}

CHECKPOINT_EVERY = 25  # write the ledger every N evaluations for crash safety


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _row_skipped(scope, members, target, model, variant, n_pool, reason) -> dict:
    return {
        "scope": scope,
        "members": "+".join(members),
        "pool_size": len(members),
        "instrument": target,
        "asset_class": INSTRUMENT_TO_CLASS.get(target, "unknown"),
        "model": model,
        "variant": variant,
        "n_modelling": n_pool,
        "n_oos": 0,
        "mean_auc": float("nan"),
        "sem": float("nan"),
        "std_auc": float("nan"),
        "lower_ci_1se": float("nan"),
        "skipped": True,
        "reason": reason,
        "elapsed_s": 0.0,
    }


def _row_ok(scope, members, target, model, variant, res, elapsed) -> dict:
    lower_ci = (
        res.mean_auc - res.sem
        if not pd.isna(res.mean_auc) and not pd.isna(res.sem)
        else float("nan")
    )
    return {
        "scope": scope,
        "members": "+".join(members),
        "pool_size": len(members),
        "instrument": target,
        "asset_class": INSTRUMENT_TO_CLASS.get(target, "unknown"),
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
        "elapsed_s": round(elapsed, 2),
    }


def run_tournament(
    *,
    features_path: Path,
    full_ledger_path: Path,
    summary_path: Path,
    cfg: PipelineConfig,
    scopes: dict[str, tuple[str, ...]],
    models: tuple[str, ...],
    variants: tuple[str, ...],
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all (variant × scope × target × model) combinations and select per inst."""

    # Load data once.
    features_orig = pd.read_parquet(features_path)
    print(f"Loaded features: {features_orig.shape}")

    features = _restrict_modelling(features_orig, cfg)
    print(f"After modelling restriction (≤{cfg.global_train_cut}): {features.shape}")

    embargo_map = embargo_days_map()

    # In-process extension of POOL_MEMBERS.
    for scope, members in scopes.items():
        if scope not in POOL_MEMBERS:
            POOL_MEMBERS[scope] = members

    # Build the work list — (variant, scope, target, model) tuples.
    combos: list[tuple[str, str, tuple[str, ...], str, str]] = []
    for variant in variants:
        for scope, members in scopes.items():
            for target in SCOPE_TARGETS[scope]:
                for model in models:
                    combos.append((variant, scope, members, target, model))

    print(f"Total combinations: {len(combos)} "
          f"({len(variants)} variants × {len(scopes)} scopes × ~targets × {len(models)} models)")
    print(f"Models: {models}")
    print(f"Variants: {variants}")

    full_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_global = time.time()

    for i, (variant, scope, members, target, model) in enumerate(combos, start=1):
        feature_cols = _select_feature_cols(features, variant)
        pool_df = _slice_pool(features, scope)
        n_pool = len(pool_df)
        is_individual = len(members) == 1
        skipped_pool = is_individual and n_pool < MIN_INDIVIDUAL_EVENTS

        if skipped_pool:
            rows.append(_row_skipped(
                scope, members, target, model, variant, n_pool,
                f"individual pool < MIN_INDIVIDUAL_EVENTS={MIN_INDIVIDUAL_EVENTS}",
            ))
            if verbose and i % 10 == 0:
                elapsed_global = time.time() - t_global
                print(f"  [{i}/{len(combos)}] skip {scope}/{target}/{model}/{variant} "
                      f"(n={n_pool})  cum={elapsed_global:.1f}s")
            continue

        t0 = time.time()
        try:
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
        except Exception as e:  # noqa: BLE001
            rows.append(_row_skipped(
                scope, members, target, model, variant, n_pool,
                f"exception: {type(e).__name__}: {e}",
            ))
            if verbose:
                print(f"  [{i}/{len(combos)}] FAIL {scope}/{target}/{model}/{variant}: {e}")
            continue

        elapsed = time.time() - t0
        rows.append(_row_ok(scope, members, target, model, variant, res, elapsed))

        if verbose and i % 10 == 0:
            elapsed_global = time.time() - t_global
            print(f"  [{i}/{len(combos)}] {scope}/{target}/{model}/{variant}  "
                  f"AUC={res.mean_auc:.4f}±{res.sem:.4f}  "
                  f"({elapsed:.1f}s, cum={elapsed_global:.1f}s)")

        # Periodic checkpoint to disk.
        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(full_ledger_path, index=False, float_format="%.6f")

    # Final ledger write.
    full_df = pd.DataFrame(rows)
    full_df.to_csv(full_ledger_path, index=False, float_format="%.6f")
    print(f"\nWrote {full_ledger_path}  ({len(full_df)} rows)")

    # Build the per-instrument selection summary.
    summary_df = build_selection_summary(full_df)
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Wrote {summary_path}  ({len(summary_df)} rows)")

    print(f"\nTotal wall time: {time.time() - t_global:.1f}s")
    return full_df, summary_df


def build_selection_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """For each instrument, pick the best route by (lower_ci_1se, mean_auc, simplicity).

    Tiebreakers:
        1. lower_ci_1se desc (most robust lower bound)
        2. mean_auc desc
        3. model simplicity asc (elasticnet < rf < lightgbm < xgboost)
        4. pool size desc (more data preferred)  — alt: asc for tighter scope
    """
    valid = ledger.loc[~ledger["skipped"]].copy()
    valid["model_rank"] = valid["model"].map(MODEL_RANK).fillna(99).astype(int)

    rows = []
    for inst in INSTRUMENTS:
        sub = valid.loc[valid["instrument"] == inst].copy()
        cls = INSTRUMENT_TO_CLASS.get(inst, "unknown")
        if sub.empty:
            rows.append({
                "instrument": inst,
                "asset_class": cls,
                "chosen_scope": None,
                "chosen_model": None,
                "chosen_variant": None,
                "mean_auc": float("nan"),
                "sem": float("nan"),
                "lower_ci_1se": float("nan"),
                "n_modelling": 0,
                "n_oos": 0,
                "n_candidates_considered": 0,
                "n_within_1se_of_best": 0,
                "reason": "no valid candidates",
            })
            continue

        # Rank by lower_CI primary, then mean_auc, then simplicity.
        # Pool size: prefer LARGER (more data → more robust) as the tertiary
        # simplicity rule. This differs from champion_pipeline (which picks
        # smaller); we use larger for OOS-generalisability.
        sub_sorted = sub.sort_values(
            by=["lower_ci_1se", "mean_auc", "model_rank", "pool_size"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)
        best = sub_sorted.iloc[0]

        # How many candidates within 1SE of the best mean_auc?
        threshold = best["mean_auc"] - (best["sem"] if not pd.isna(best["sem"]) else 0.0)
        n_within = int((sub["mean_auc"] >= threshold).sum())

        rows.append({
            "instrument": inst,
            "asset_class": cls,
            "chosen_scope": best["scope"],
            "chosen_model": best["model"],
            "chosen_variant": best["variant"],
            "chosen_members": best["members"],
            "mean_auc": float(best["mean_auc"]),
            "sem": float(best["sem"]),
            "std_auc": float(best["std_auc"]),
            "lower_ci_1se": float(best["lower_ci_1se"]),
            "n_modelling": int(best["n_modelling"]),
            "n_oos": int(best["n_oos"]),
            "n_candidates_considered": int(len(sub)),
            "n_within_1se_of_best": n_within,
            "reason": (
                f"top of {len(sub)} valid candidates by (lower_CI, AUC, model-simplicity, pool-size); "
                f"{n_within} within 1-SE of best"
            ),
        })

    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame, full: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("PER-INSTRUMENT ROUTER (best by lower_CI_1SE)")
    print("=" * 90)
    cols = [
        "instrument", "asset_class", "chosen_scope", "chosen_model",
        "chosen_variant", "mean_auc", "sem", "lower_ci_1se", "n_within_1se_of_best",
    ]
    disp = summary.loc[:, cols].copy()
    for c in ("mean_auc", "sem", "lower_ci_1se"):
        disp[c] = disp[c].map(lambda v: f"{v:.4f}" if not pd.isna(v) else "—")
    print(disp.to_string(index=False))

    # Skipped overview.
    skipped = full.loc[full["skipped"]]
    print(f"\nSkipped: {len(skipped)} / {len(full)} runs "
          f"(mostly individual pools < MIN_INDIVIDUAL_EVENTS=250)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="sm/scope-router — full routing tournament (25 scopes × 4 models × 2 variants)"
    )
    ap.add_argument(
        "--variants", nargs="+", default=list(DEFAULT_VARIANTS),
        choices=["without_bbg", "with_bbg"],
        help="Feature variants to test (default: both)",
    )
    ap.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS),
        help="Models to evaluate (default: all 4)",
    )
    ap.add_argument(
        "--scopes", nargs="+", default=None,
        help="Subset of scope names (default: all 25)",
    )
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    root = _find_repo_root()
    features_path = root / "data" / "sreeram_experimental_features.parquet"
    if not features_path.exists():
        print(f"FATAL: {features_path} missing")
        return 1

    scopes_to_run: dict[str, tuple[str, ...]]
    if args.scopes is None:
        scopes_to_run = SCOPE_DEFS
    else:
        scopes_to_run = {k: v for k, v in SCOPE_DEFS.items() if k in args.scopes}
        if not scopes_to_run:
            print(f"FATAL: no matching scopes among {list(SCOPE_DEFS)}")
            return 1

    full_ledger_path = root / "results" / "sm_scope_router" / "router_full_ledger.csv"
    summary_path = root / "results" / "sm_scope_router" / "router_selection_summary.csv"

    full_df, summary_df = run_tournament(
        features_path=features_path,
        full_ledger_path=full_ledger_path,
        summary_path=summary_path,
        cfg=cfg,
        scopes=scopes_to_run,
        models=tuple(args.models),
        variants=tuple(args.variants),
        verbose=True,
    )

    _print_summary(summary_df, full_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
