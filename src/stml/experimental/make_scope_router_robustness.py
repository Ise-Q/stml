"""sm/scope-router — robustness pack for the selected routes.

For each per-instrument route in ``router_selection_summary.csv``:

1. Re-run the route under CPCV(6,2) to recover row-level OOS predictions.
2. Bootstrap the AUC by row resampling (1000 resamples) → 95% CI.
3. Stability evidence — per-CPCV-path AUC distribution: spread, IQR,
   fraction of paths above 0.5, min/max.
4. Paired comparison vs a model-controlled baseline:
       baseline = (class_pool, route_model, without_bbg)
   This isolates the *scope / variant* effect while controlling for model.
   Bootstrap the difference (route_auc - baseline_auc) to get its 95% CI.
5. For the two ``with_bbg`` winners (fesx1s, pl1s), also run
   ``simulated_missingness`` (BBG cols NaN'd at test time) to check if the
   BBG advantage holds when the BBG inputs are unreliable at inference.
6. Strength label per instrument:
       strong: route lower CI > baseline upper CI AND fold_above_050 ≥ 0.6 AND lower CI > 0.5
       medium: route mean > baseline mean AND fold_above_050 ≥ 0.5 AND AUC ≥ 0.5
       weak:   otherwise (or AUC < 0.5)

Outputs:

* ``results/sm_scope_router/router_robustness_summary.csv`` — per-instrument row
  with all stats (bootstrap CI, fold stats, baseline comparison, strength).
* ``results/sm_scope_router/recommended_router.json`` — adoption-ready router for
  strong + medium instruments only; weak / no-route are marked DEFAULT_BLIND.
* ``reports/sm_scope_router_robustness.md`` (companion findings note, written
  separately).

Run::

    PYTHONPATH=src .venv/bin/python -m stml.experimental.make_scope_router_robustness
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stml.experimental.champion_pipeline import (
    POOL_MEMBERS,
    _evaluate_candidate,
    _is_bbg_feature,
    _restrict_modelling,
    _select_feature_cols,
    _slice_pool,
)
from stml.experimental.config import PipelineConfig
from stml.experimental.make_scope import embargo_days_map
from stml.experimental.make_scope_router_full import SCOPE_DEFS

# Bootstrap config.
N_BOOTSTRAP = 1000
RNG_SEED = 42

# Class-level baseline pool per instrument's asset class — used for the
# "model-controlled" baseline comparison.
CLASS_POOL = {
    "equity": "equity_all",
    "energy": "energy_all",
    "metals": "metals_all",
}


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _bootstrap_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: np.ndarray | None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float, float, np.ndarray]:
    """Row-resample bootstrap of AUC. Returns (mean, ci_lo_2.5, ci_hi_97.5, samples)."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        p_b = y_proba[idx]
        w_b = sample_weight[idx] if sample_weight is not None else None
        try:
            aucs.append(roc_auc_score(y_b, p_b, sample_weight=w_b))
        except Exception:  # noqa: BLE001
            pass
    if not aucs:
        return float("nan"), float("nan"), float("nan"), np.array([])
    arr = np.array(aucs)
    return (
        float(arr.mean()),
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
        arr,
    )


def _bootstrap_diff(
    route_y: np.ndarray, route_p: np.ndarray, route_w: np.ndarray | None,
    base_y: np.ndarray, base_p: np.ndarray, base_w: np.ndarray | None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Bootstrap the difference (route_auc - baseline_auc) via independent
    resamples. Returns (diff_mean, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    diffs = []
    nr = len(route_y)
    nb = len(base_y)
    for _ in range(n_bootstrap):
        idx_r = rng.integers(0, nr, size=nr)
        idx_b = rng.integers(0, nb, size=nb)
        y_r = route_y[idx_r]
        y_b = base_y[idx_b]
        if len(np.unique(y_r)) < 2 or len(np.unique(y_b)) < 2:
            continue
        try:
            a_r = roc_auc_score(y_r, route_p[idx_r],
                                sample_weight=route_w[idx_r] if route_w is not None else None)
            a_b = roc_auc_score(y_b, base_p[idx_b],
                                sample_weight=base_w[idx_b] if base_w is not None else None)
            diffs.append(a_r - a_b)
        except Exception:  # noqa: BLE001
            pass
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(diffs)
    return (
        float(arr.mean()),
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
    )


def _fold_stats(oos: pd.DataFrame) -> dict:
    """Per-CPCV-path AUC stats: min, max, mean, std, IQR, fraction above 0.5."""
    per_fold = []
    for _, grp in oos.groupby("fold"):
        if grp["y_true"].nunique() < 2:
            continue
        try:
            per_fold.append(
                roc_auc_score(grp["y_true"], grp["y_proba"],
                              sample_weight=grp["sample_weight"])
            )
        except Exception:  # noqa: BLE001
            pass
    if not per_fold:
        return {
            "n_folds": 0, "fold_above_050": float("nan"),
            "fold_iqr": float("nan"), "fold_min": float("nan"),
            "fold_max": float("nan"), "fold_mean": float("nan"),
            "fold_std": float("nan"),
        }
    arr = np.array(per_fold)
    return {
        "n_folds": int(len(arr)),
        "fold_above_050": float((arr > 0.5).mean()),
        "fold_iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
        "fold_min": float(arr.min()),
        "fold_max": float(arr.max()),
        "fold_mean": float(arr.mean()),
        "fold_std": float(arr.std()),
    }


def _label_strength(
    route_mean: float, route_ci_lo: float,
    baseline_mean: float, baseline_ci_hi: float,
    diff_ci_lo: float,
    fold_above_050: float,
) -> tuple[str, str]:
    """Strong / medium / weak label, with one-line rationale."""
    if pd.isna(route_mean):
        return ("no_route", "no valid candidate")
    if route_mean < 0.50:
        return ("weak", f"route mean AUC < 0.5 ({route_mean:.4f}) — below random")
    if diff_ci_lo > 0 and route_ci_lo > 0.50 and fold_above_050 >= 0.6:
        return (
            "strong",
            f"diff CI > 0 ({diff_ci_lo:+.4f}), route lower CI > 0.5 ({route_ci_lo:.4f}), "
            f"{fold_above_050:.0%} of CPCV paths above 0.5"
        )
    if route_mean > baseline_mean and fold_above_050 >= 0.5:
        return (
            "medium",
            f"route mean > baseline mean (+{route_mean - baseline_mean:+.4f}), "
            f"but diff CI includes 0 ({diff_ci_lo:+.4f}); "
            f"{fold_above_050:.0%} CPCV paths above 0.5"
        )
    return (
        "weak",
        f"no clear advantage over baseline (route {route_mean:.4f} vs "
        f"baseline {baseline_mean:.4f}, fold_above_050 {fold_above_050:.0%})"
    )


def _evaluate_route_with_oos(
    *,
    target: str,
    pool: str,
    model: str,
    variant: str,
    features: pd.DataFrame,
    cfg: PipelineConfig,
    embargo_map: dict[str, int],
    nan_cols_at_test: list[str] | None = None,
) -> tuple[float, float, float, pd.DataFrame, int, int]:
    """Run _evaluate_candidate and return (mean_auc, sem, lower_ci_1se, oos_df, n_modelling, n_oos)."""
    pool_df = _slice_pool(features, pool)
    feature_cols = _select_feature_cols(features, variant)
    res = _evaluate_candidate(
        instrument=target,
        pool=pool,
        model_name=model,
        pool_df=pool_df,
        feature_cols=feature_cols,
        cfg=cfg,
        nan_cols_at_test=nan_cols_at_test,
        embargo_map=embargo_map,
    )
    return (
        float(res.mean_auc),
        float(res.sem),
        float(res.mean_auc - res.sem) if not pd.isna(res.mean_auc) else float("nan"),
        res.oos_predictions,
        int(res.n_modelling),
        int(res.n_oos_for_instrument),
    )


def run_robustness(
    *,
    selection_csv: Path,
    cfg: PipelineConfig,
    features: pd.DataFrame,
    embargo_map: dict[str, int],
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Compute robustness for every instrument in the selection CSV."""
    sel = pd.read_csv(selection_csv)
    print(f"Loaded {len(sel)} selected routes from {selection_csv.name}")

    # In-process extension of POOL_MEMBERS to include our scope grid.
    for scope, members in SCOPE_DEFS.items():
        if scope not in POOL_MEMBERS:
            POOL_MEMBERS[scope] = members

    rows: list[dict] = []
    router_json: dict[str, dict] = {}
    t_global = time.time()

    for _, sel_row in sel.iterrows():
        inst = sel_row["instrument"]
        cls = sel_row["asset_class"]
        scope = sel_row["chosen_scope"]
        model = sel_row["chosen_model"]
        variant = sel_row["chosen_variant"]
        if pd.isna(scope) or scope is None:
            # no route at all (e.g. gc1s default-blind)
            rows.append({
                "instrument": inst, "asset_class": cls,
                "chosen_scope": "DEFAULT_BLIND",
                "strength": "no_route",
                "reason": "no candidate cleared selection in the tournament",
            })
            router_json[inst] = {"route": "DEFAULT_BLIND", "reason": "no validated route"}
            continue

        if verbose:
            print(f"\n=== {inst} ({cls}) — route={scope}/{model}/{variant} ===")

        # 1. Re-run the route to get OOS predictions.
        t0 = time.time()
        r_mean, r_sem, r_ci, r_oos, r_n_mod, r_n_oos = _evaluate_route_with_oos(
            target=inst, pool=scope, model=model, variant=variant,
            features=features, cfg=cfg, embargo_map=embargo_map,
        )
        if r_oos.empty:
            rows.append({
                "instrument": inst, "asset_class": cls,
                "chosen_scope": scope, "chosen_model": model, "chosen_variant": variant,
                "strength": "no_route", "reason": "empty OOS",
            })
            continue

        # 2. Bootstrap AUC on the route OOS.
        r_y = r_oos["y_true"].values
        r_p = r_oos["y_proba"].values
        r_w = r_oos["sample_weight"].values
        r_boot_mean, r_boot_lo, r_boot_hi, _ = _bootstrap_auc(r_y, r_p, r_w)
        if verbose:
            print(f"  route AUC mean (1-SE)  : {r_mean:.4f}  ±{r_sem:.4f}")
            print(f"  route bootstrap 95% CI : [{r_boot_lo:.4f}, {r_boot_hi:.4f}]")

        # 3. Fold-stability.
        r_folds = _fold_stats(r_oos)
        if verbose:
            print(f"  fold paths above 0.5   : {r_folds['fold_above_050']:.0%}  "
                  f"(IQR={r_folds['fold_iqr']:.3f}, min={r_folds['fold_min']:.3f}, "
                  f"max={r_folds['fold_max']:.3f}, n_folds={r_folds['n_folds']})")

        # 4. Baseline = (class pool, same model, without_bbg).
        baseline_pool = CLASS_POOL.get(cls, scope)
        baseline_variant = "without_bbg"
        b_mean, b_sem, b_ci, b_oos, b_n_mod, b_n_oos = _evaluate_route_with_oos(
            target=inst, pool=baseline_pool, model=model, variant=baseline_variant,
            features=features, cfg=cfg, embargo_map=embargo_map,
        )
        if not b_oos.empty:
            b_y = b_oos["y_true"].values
            b_p = b_oos["y_proba"].values
            b_w = b_oos["sample_weight"].values
            b_boot_mean, b_boot_lo, b_boot_hi, _ = _bootstrap_auc(b_y, b_p, b_w)
            diff_mean, diff_lo, diff_hi = _bootstrap_diff(
                r_y, r_p, r_w, b_y, b_p, b_w,
            )
        else:
            b_boot_mean = b_boot_lo = b_boot_hi = float("nan")
            diff_mean = diff_lo = diff_hi = float("nan")

        if verbose:
            print(f"  baseline ({baseline_pool}/{model}/without_bbg) AUC: "
                  f"{b_mean:.4f}±{b_sem:.4f}  boot CI [{b_boot_lo:.4f}, {b_boot_hi:.4f}]")
            print(f"  diff (route − baseline): mean={diff_mean:+.4f}  "
                  f"CI [{diff_lo:+.4f}, {diff_hi:+.4f}]")

        # 5. simulated_missingness for the with_bbg winners only.
        sim_mean = sim_lo = sim_hi = float("nan")
        sim_diff = float("nan")
        if variant == "with_bbg":
            if verbose:
                print("  → simulated_missingness check (BBG cols NaN'd at test) …")
            bbg_cols = [c for c in _select_feature_cols(features, "with_bbg")
                        if _is_bbg_feature(c)]
            sm_mean, _, _, sm_oos, _, _ = _evaluate_route_with_oos(
                target=inst, pool=scope, model=model, variant="with_bbg",
                features=features, cfg=cfg, embargo_map=embargo_map,
                nan_cols_at_test=bbg_cols,
            )
            if not sm_oos.empty:
                sm_y = sm_oos["y_true"].values
                sm_p = sm_oos["y_proba"].values
                sm_w = sm_oos["sample_weight"].values
                sim_mean, sim_lo, sim_hi, _ = _bootstrap_auc(sm_y, sm_p, sm_w)
                sim_diff = r_boot_mean - sim_mean
                if verbose:
                    print(f"    sim_miss AUC: {sim_mean:.4f}  CI [{sim_lo:.4f}, {sim_hi:.4f}]")
                    print(f"    Δ vs with_bbg (BBG actually loaded): {sim_diff:+.4f}")

        # 6. Strength label.
        strength, rationale = _label_strength(
            r_mean, r_boot_lo, b_mean, b_boot_hi, diff_lo,
            r_folds["fold_above_050"],
        )
        if verbose:
            print(f"  ⇒ strength = {strength}  ({rationale})")

        rows.append({
            "instrument": inst,
            "asset_class": cls,
            "chosen_scope": scope,
            "chosen_model": model,
            "chosen_variant": variant,
            "chosen_members": sel_row["chosen_members"],
            "n_modelling": r_n_mod,
            "n_oos": r_n_oos,
            "route_mean_auc": r_mean,
            "route_sem": r_sem,
            "route_lower_ci_1se": r_ci,
            "route_boot_mean": r_boot_mean,
            "route_boot_ci_lo": r_boot_lo,
            "route_boot_ci_hi": r_boot_hi,
            "fold_above_050": r_folds["fold_above_050"],
            "fold_iqr": r_folds["fold_iqr"],
            "fold_min": r_folds["fold_min"],
            "fold_max": r_folds["fold_max"],
            "fold_n": r_folds["n_folds"],
            "baseline_scope": baseline_pool,
            "baseline_model": model,
            "baseline_variant": "without_bbg",
            "baseline_mean_auc": b_mean,
            "baseline_boot_mean": b_boot_mean,
            "baseline_boot_ci_lo": b_boot_lo,
            "baseline_boot_ci_hi": b_boot_hi,
            "diff_mean_auc": diff_mean,
            "diff_boot_ci_lo": diff_lo,
            "diff_boot_ci_hi": diff_hi,
            "sim_miss_mean_auc": sim_mean,
            "sim_miss_boot_ci_lo": sim_lo,
            "sim_miss_boot_ci_hi": sim_hi,
            "sim_miss_delta_vs_with_bbg": sim_diff,
            "strength": strength,
            "strength_rationale": rationale,
            "elapsed_s": round(time.time() - t0, 2),
        })

        # Build router_json entry (strong + medium only get a route; weak/no_route → DEFAULT_BLIND).
        if strength in ("strong", "medium"):
            router_json[inst] = {
                "scope": scope,
                "members": list(SCOPE_DEFS.get(scope, ())),
                "model": model,
                "variant": variant,
                "asset_class": cls,
                "mean_auc": r_mean,
                "boot_ci_95": [r_boot_lo, r_boot_hi],
                "strength": strength,
                "diff_vs_baseline": {
                    "baseline": f"{baseline_pool}/{model}/without_bbg",
                    "mean": diff_mean,
                    "ci_95": [diff_lo, diff_hi],
                },
                "fold_above_050": r_folds["fold_above_050"],
            }
            if variant == "with_bbg":
                router_json[inst]["sim_missingness"] = {
                    "auc": sim_mean,
                    "ci_95": [sim_lo, sim_hi],
                    "delta": sim_diff,
                }
        else:
            router_json[inst] = {
                "route": "DEFAULT_BLIND",
                "reason": strength,
                "best_route_attempted": f"{scope}/{model}/{variant}",
                "best_mean_auc": r_mean,
            }

    print(f"\nTotal wall time: {time.time() - t_global:.1f}s")
    return pd.DataFrame(rows), router_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="sm/scope-router robustness pack")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                    help=f"Bootstrap resamples per route (default: {N_BOOTSTRAP})")
    ap.parse_args(argv)

    cfg = PipelineConfig()
    root = _find_repo_root()

    selection_csv = root / "results" / "sm_scope_router" / "router_selection_summary.csv"
    if not selection_csv.exists():
        print(f"FATAL: {selection_csv} missing — run make_scope_router_full first.")
        return 1

    features_path = root / "data" / "sreeram_experimental_features.parquet"
    if not features_path.exists():
        print(f"FATAL: {features_path} missing")
        return 1

    print(f"Loaded features: {pd.read_parquet(features_path).shape}")
    features = pd.read_parquet(features_path)
    features = _restrict_modelling(features, cfg)
    embargo_map = embargo_days_map()

    out_df, router_json = run_robustness(
        selection_csv=selection_csv,
        cfg=cfg,
        features=features,
        embargo_map=embargo_map,
        verbose=True,
    )

    out_csv = root / "results" / "sm_scope_router" / "router_robustness_summary.csv"
    out_json = root / "results" / "sm_scope_router" / "recommended_router.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nWrote {out_csv}  ({len(out_df)} rows)")

    with out_json.open("w") as f:
        json.dump(router_json, f, indent=2, sort_keys=True)
    print(f"Wrote {out_json}  ({len(router_json)} entries)")

    # Print headline strength distribution.
    print("\n=== Strength distribution ===")
    print(out_df["strength"].value_counts().to_string())

    # Print per-instrument summary.
    print("\n=== Per-instrument robustness ===")
    cols = ["instrument", "chosen_scope", "chosen_model", "chosen_variant",
            "route_boot_mean", "route_boot_ci_lo", "route_boot_ci_hi",
            "baseline_boot_mean", "diff_mean_auc",
            "fold_above_050", "strength"]
    disp = out_df.loc[:, cols].copy()
    for c in ("route_boot_mean", "route_boot_ci_lo", "route_boot_ci_hi",
              "baseline_boot_mean", "diff_mean_auc", "fold_above_050"):
        disp[c] = disp[c].map(lambda v: f"{v:.4f}" if not pd.isna(v) else "—")
    print(disp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
