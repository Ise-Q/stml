"""Generate training OOF calibrated probabilities for SOPS fitting.

Runs CPCV(6,2) on each champion's training events using the same infrastructure
as run_locked_strategy.py, Platt-calibrates the OOF scores, and saves the
result to data/oof_meta_probabilities.csv.

Output schema (required by weights.py and data.py):
    date        — t_signal (the event entry date, always < BOUNDARY)
    instrument  — ticker string
    p_hat_oof   — Platt-calibrated CPCV OOF probability ∈ [0, 1]
    ret         — triple-barrier signed return (for SOPS sigmoid fitting)
    side        — primary signal direction {-1, 0, +1}
    y_true      — binary triple-barrier label {0, 1}

Typical runtime: 5–15 minutes (11 instruments × CPCV(6,2) = 165 model fits).
Must be run ONCE before SOPS evaluation. Other methods (A, B-fixed) do not
require this artifact.

Usage:
    .venv/bin/python src/stml/new_work/gen_oof_probas.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work.run_locked_strategy import (
    CACHE_DIR,
    CHAMPIONS,
    VARIANT_SPECS,
    _feat_cols,
    load_cluster_membership,
    load_pool_train,
    run_cpcv_oof,
)
from stml.new_work.config import DATA_DIR, BOUNDARY
from stml.experimental.calibration import PlattCalibrator

OUTPUT_PATH = DATA_DIR / "oof_meta_probabilities.csv"


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def generate(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Run CPCV OOF for all champions and save the calibrated probability artifact."""
    individual_events: dict[str, pd.DataFrame] = {}
    loaded_pools: dict[str, pd.DataFrame] = {}
    all_rows: list[pd.DataFrame] = []

    for inst, cfg in CHAMPIONS.items():
        family  = cfg["family"]
        pool    = cfg["pool"]
        variant = cfg["variant"]
        spec    = VARIANT_SPECS[inst][variant]
        cd      = load_cluster_membership(inst)

        # Load training events.
        if pool == "individual":
            inst_cache = CACHE_DIR / f"{inst}_events.parquet"
            if not inst_cache.exists():
                print(f"  SKIP {inst}: cache missing at {inst_cache}")
                continue
            if inst not in individual_events:
                individual_events[inst] = pd.read_parquet(inst_cache)
            tr_ev  = individual_events[inst]
            target = None
        else:
            if pool not in loaded_pools:
                loaded_pools[pool] = load_pool_train(pool)
            tr_ev  = loaded_pools[pool]
            target = inst

        print(f"  [{inst}] CPCV OOF ({family}/{pool}/{variant}) — {len(tr_ev)} rows …")
        oof_df = run_cpcv_oof(inst, tr_ev, spec, cd, family, pool, target_inst=target)

        if oof_df.empty or oof_df["y_true"].nunique() < 2:
            print(f"    WARNING: degenerate OOF for {inst} — skipping")
            continue

        # Platt-calibrate the raw OOF scores.
        cal = PlattCalibrator()
        cal.fit(oof_df["y_score"].values, oof_df["y_true"].values)
        oof_df["p_hat_oof"] = cal.transform(oof_df["y_score"].values)

        auc = _safe_auc(oof_df["y_true"].values, oof_df["y_score"].values)
        print(f"    raw OOF rows={len(oof_df)}  raw AUC={auc:.4f}  "
              f"p̂ mean={oof_df['p_hat_oof'].mean():.4f}")

        # In CPCV(6,2) each event appears in C(5,1)=5 test paths.
        # Average the calibrated probability over duplicate CPCV appearances.
        avg = (
            oof_df
            .groupby("date", as_index=False)
            .agg(y_true=("y_true", "first"), p_hat_oof=("p_hat_oof", "mean"))
        )
        avg["instrument"] = inst

        # Join with training event metadata to get ret and side.
        tr_meta = tr_ev[["date", "instrument", "ret", "side"]].copy()
        if target is not None:
            tr_meta = tr_meta[tr_meta["instrument"] == inst]
        else:
            tr_meta = tr_meta.copy()
            tr_meta["instrument"] = inst

        # Drop duplicate dates in tr_meta (keeps first; events are 1-per-date per inst).
        tr_meta = tr_meta.drop_duplicates("date")

        merged = avg.merge(tr_meta[["date", "ret", "side"]], on="date", how="left")

        n_missing_ret = merged["ret"].isna().sum()
        if n_missing_ret:
            print(f"    WARNING: {n_missing_ret} events could not be joined to ret/side")

        all_rows.append(merged)
        print(f"    → {len(merged)} unique event rows kept")

    if not all_rows:
        raise RuntimeError("No OOF data generated — check cache paths and model fits")

    result = pd.concat(all_rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])

    # Safety: enforce boundary — no training probs should be dated after BOUNDARY.
    n_before = len(result)
    result = result[result["date"] < BOUNDARY].copy()
    if len(result) < n_before:
        print(f"  Dropped {n_before - len(result)} rows dated >= BOUNDARY ({BOUNDARY.date()})")

    result = result.sort_values(["instrument", "date"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\nSaved {len(result)} OOF rows → {output_path}")
    summary = result.groupby("instrument").agg(
        n=("p_hat_oof", "count"),
        p_mean=("p_hat_oof", "mean"),
        ret_mean=("ret", "mean"),
    )
    print(summary.to_string())

    return result


def main() -> int:
    print("=" * 70)
    print("Generating OOF calibrated probabilities for all champion models")
    print("=" * 70)
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
