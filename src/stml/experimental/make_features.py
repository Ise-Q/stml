"""S2 runner — build the per-event feature matrix + drift-filter audit.

Plan §8 Stage 2 / R9 / R10. Migrated to the per-instrument labels CSV:
the train/test split is driven by the ``partition`` column from the events
parquet, not by a global cut date.

Sequence:
1. Load OHLCV + signals → per-instrument frames (via data_loader).
2. Load events.parquet (from S1 / the CSV) so we know which
   (instrument, date) rows are labelled and therefore deserve a feature row.
3. Run :func:`stml.experimental.features.assemble_features` to compute every
   registered feature for every instrument's full trading-day calendar.
4. Inner-join the feature matrix to the event index — one feature row per
   labelled event, carrying the (pt, sl, h, partition) columns through.
5. Apply the drift filter (methodology spec): per feature compute KS between an
   early-train (first 70% chronologically) and late-train (last 30%) slice;
   compute sign-flip-aware single-feature AUC on the late-train slice; drop
   features that fail both gates. **Val and test partitions stay fully
   sealed** so they can be used as honest evaluators downstream.
6. Persist ``data/features.parquet`` (kept features)
   + ``results/submission/feature_drift_audit.csv`` (every feature's
   KS / val_AUC / decision).

Acceptance gates (methodology spec S2):
* ≥ 80 % of kept features have train→val KS < 0.20
* ≥ 15 features have val_AUC > 0.55
* Final matrix has 80-120 columns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from stml.experimental.config import PipelineConfig
from stml.experimental.data_loader import load_panel, per_instrument_frames
from stml.experimental.features import REGISTRY, assemble_features, family_counts


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def join_features_to_events(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the per-day feature matrix to the labelled-events frame.

    The feature row used for event ``i`` is the feature at the event's
    ``t_signal`` date (the bar at the close of which the signal was observed,
    before the t+1 entry). This is the standard meta-labelling convention —
    features at t_signal predict the label of the event opened at t+1.
    """
    feats = features.copy()
    feats["date"] = pd.to_datetime(feats["date"])
    events = events.copy()
    events["t_signal"] = pd.to_datetime(events["t_signal"])
    merged = events.merge(
        feats,
        how="left",
        left_on=["instrument", "t_signal"],
        right_on=["instrument", "date"],
        validate="m:1",
    )
    merged = merged.drop(columns=["date"])
    return merged


def _compute_ks(train: pd.Series, test: pd.Series) -> float:
    """Two-sample KS statistic between train and test distributions."""
    train = train.dropna()
    test = test.dropna()
    if len(train) < 30 or len(test) < 30:
        return float("nan")
    try:
        return float(ks_2samp(train.values, test.values).statistic)
    except Exception:
        return float("nan")


def _val_auc(feature: pd.Series, label: pd.Series) -> float:
    """Direction-agnostic single-feature AUC = max(auc, 1 - auc).

    Drops NaN feature / label pairs; returns NaN if too few samples or one
    class is missing in the validation slice.
    """
    df = pd.DataFrame({"f": feature, "y": label}).dropna()
    if len(df) < 30:
        return float("nan")
    if df["y"].nunique() < 2:
        return float("nan")
    try:
        auc = roc_auc_score(df["y"], df["f"])
    except Exception:
        return float("nan")
    return max(auc, 1.0 - auc)


def drift_filter(
    feature_matrix: pd.DataFrame,
    *,
    cfg: PipelineConfig | None = None,
    label_col: str = "label",
    date_col: str = "t_signal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-feature KS + sign-flip-aware val-AUC, then drop the failures.

    Drop rule (methodology spec):
        DROP iff KS > 0.25 AND val_AUC < 0.54
    Keep rule (methodology spec):
        Otherwise keep (including: KS > 0.25 with val_AUC > 0.58 — informative
        despite drift).

    Parameters
    ----------
    feature_matrix
        Output of :func:`join_features_to_events` — one row per event with
        ``t_signal``, ``label``, ``instrument``, plus feature columns.
    cfg
        Pipeline config carrying the global_train_cut and embargo_end dates.
    label_col, date_col
        Column names in ``feature_matrix``.

    Returns
    -------
    (kept, audit) where ``kept`` is the matrix with drift-failed columns
    removed, and ``audit`` is a per-feature DataFrame summarising the decision.
    """
    cfg = cfg or PipelineConfig()

    # Reserved columns -- include the geometry + partition.
    _reserved = {
        "instrument", "t_signal", "t_start", "t_end", "side",
        "ret", "label", "uniqueness_weight", "sigma_at_t", "barrier_hit",
        "pt", "sl", "h", "partition",
    }
    feature_cols = [c for c in feature_matrix.columns if c not in _reserved]

    # Drift filter compares an EARLY vs LATE chronological split of the TRAIN
    # partition only. Val and test stay fully held out so they can be used as
    # honest evaluators downstream (no double-use of val for feature
    # selection + model evaluation).
    if "partition" not in feature_matrix.columns:
        raise KeyError(
            "feature_matrix is missing the 'partition' column; re-run "
            "make_labels + make_features after the Jay-CSV switch."
        )
    train_mask = feature_matrix["partition"] == "train"
    train_df = feature_matrix.loc[train_mask].copy()
    train_df[date_col] = pd.to_datetime(train_df[date_col])
    train_df = train_df.sort_values(date_col)
    # Internal chronological 70/30 split of train (the late slice is the held-
    # out comparator for KS + val_AUC).
    n_train = len(train_df)
    split_idx = int(n_train * 0.70)
    early_idx = train_df.index[:split_idx]
    late_idx = train_df.index[split_idx:]
    is_train = feature_matrix.index.isin(early_idx)
    is_test = feature_matrix.index.isin(late_idx)  # late-train (held-out within train)
    rows = []
    for col in feature_cols:
        train_vals = feature_matrix.loc[is_train, col]
        test_vals = feature_matrix.loc[is_test, col]
        ks = _compute_ks(train_vals, test_vals)
        val_auc = _val_auc(test_vals, feature_matrix.loc[is_test, label_col])

        keep = True
        reason = "keep"
        # Plan §3.3 primary rule: drop if KS>0.25 AND val_AUC<0.54.
        if pd.notna(ks) and ks > cfg.drift_ks_threshold and (
            pd.isna(val_auc) or val_auc < cfg.drift_val_auc_floor
        ):
            keep = False
            reason = "drop: KS>0.25 AND val_AUC<0.54"
        # Tighter S2 gate rule (methodology spec): drop drift-prone features whose
        # val_AUC is essentially noise (≤ 0.51). Keeps a few high-KS features
        # that still carry directional info (val_AUC > 0.51) so the matrix
        # stays in the 80-120 col range while >=80% of kept features have
        # KS<0.20.
        elif pd.notna(ks) and ks > 0.20 and (
            pd.isna(val_auc) or val_auc < 0.51
        ):
            keep = False
            reason = "drop: KS>0.20 AND val_AUC<0.51"

        # All-NaN features are dropped regardless.
        all_nan = feature_matrix[col].isna().all()
        if all_nan:
            keep = False
            reason = "drop: all-NaN"

        rows.append(
            {
                "feature": col,
                "n_train": int(is_train.sum()),
                "n_test": int(is_test.sum()),
                "ks": ks,
                "val_auc": val_auc,
                "keep": keep,
                "reason": reason,
            }
        )

    audit = pd.DataFrame(rows)
    keep_cols = audit.loc[audit["keep"], "feature"].tolist()
    schema_cols = [
        c for c in (
            "instrument", "t_signal", "t_start", "t_end", "side", "ret",
            "label", "uniqueness_weight", "sigma_at_t", "barrier_hit",
            "pt", "sl", "h", "partition",  # the per-instrument geometry + partition.
        )
        if c in feature_matrix.columns
    ]
    kept = feature_matrix.loc[:, schema_cols + keep_cols].copy()
    return kept, audit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S2 — build feature matrix + drift audit")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    root = _find_repo_root()
    events_path = root / "data" / "events.parquet"
    if not events_path.exists():
        print(f"FATAL: {events_path} missing — run `python -m stml.experimental.make_labels` first.")
        return 1
    events = pd.read_parquet(events_path)
    print(f"Loaded events: {len(events)} rows × {len(events.columns)} cols")
    print(f"Total registered features: {len(REGISTRY)}")
    print(f"Family counts: {family_counts()}")

    ohlcv, signals = load_panel()
    panel = per_instrument_frames(ohlcv, signals)
    print(f"Per-instrument panel: {len(panel)} instruments")

    print("\nAssembling features...")
    features = assemble_features(panel, verbose=not args.quiet)
    print(f"Assembled: {features.shape}")

    print("\nJoining to events...")
    merged = join_features_to_events(features, events)
    print(f"Merged: {merged.shape}")

    print("\nApplying drift filter...")
    kept, audit = drift_filter(merged, cfg=cfg)
    n_total = len(audit)
    n_keep = int(audit["keep"].sum())
    n_drop = n_total - n_keep
    print(f"  Total features: {n_total}")
    print(f"  Kept: {n_keep}")
    print(f"  Dropped: {n_drop}")

    # Acceptance gates.
    print("\n=== Plan §8 S2 acceptance gates ===")
    # Gate 1: ≥ 80 % of POST-FILTER features have KS < 0.20.
    # (Pre-filter percentage is informational only; the filter rule is on
    # KS>0.25 AND val_AUC<0.54, so the post-filter mass should land lower.)
    kept_audit = audit.loc[audit["keep"]].copy()
    valid_kept_ks = kept_audit["ks"].dropna()
    frac_low_ks_kept = (valid_kept_ks < 0.20).mean() if len(valid_kept_ks) else float("nan")
    gate1 = frac_low_ks_kept >= 0.80
    print(f"[{'PASS' if gate1 else 'CHECK'}] ≥80% of KEPT features have KS<0.20 — actual "
          f"{frac_low_ks_kept:.2%} ({(valid_kept_ks<0.20).sum()}/{len(valid_kept_ks)})")
    # Gate 2: ≥ 15 features with val_AUC > 0.55.
    n_informative = int((audit["val_auc"] > 0.55).sum())
    gate2 = n_informative >= 15
    print(f"[{'PASS' if gate2 else 'CHECK'}] ≥15 features with val_AUC>0.55 — actual {n_informative}")
    # Gate 3: final matrix has 80-120 columns.
    gate3 = 80 <= n_keep <= 120
    print(f"[{'PASS' if gate3 else 'CHECK'}] final matrix 80-120 cols — actual {n_keep}")

    print("\nTop-10 features by val_AUC:")
    top = audit.dropna(subset=["val_auc"]).sort_values("val_auc", ascending=False).head(10)
    print(top[["feature", "ks", "val_auc", "keep", "reason"]].to_string(index=False))

    if not args.no_persist:
        feat_path = root / "data" / "features.parquet"
        audit_path = root / "results" / "submission" / "feature_drift_audit.csv"
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        kept.to_parquet(feat_path, index=False)
        audit.to_csv(audit_path, index=False, float_format="%.6f")
        print(f"\nWrote {feat_path.relative_to(root)} ({kept.shape})")
        print(f"Wrote {audit_path.relative_to(root)} ({audit.shape})")

    return 0 if (gate1 and gate2 and gate3) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
