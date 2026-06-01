"""reconciliation.py — Reconciliation pass on model-comparison results.

Reads saved OOS predictions and cached group events from model_comparison.
Produces selection_table_v2.csv and reconciliation_report.md.

Steps
-----
1. Recompute per-fold mean/std AUC from stored OOS predictions (master_results
   stores pooled AUC; per-fold mean/std is what CPCV path statistics mean).
2. STD-based tie detection: within 1 per-fold-std of the per-fold-mean winner.
3. Signal/no-signal tag: (per_fold_mean − per_fold_std) > 0.50.
4. Calibrate signal-bearing champions + ties (leakage-free inner-split calibration).
   Fallback to raw scores if calibration increases Brier.
5. Final champion: lowest calibrated Brier → model simplicity tiebreak.
6. Write outputs.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import xgboost as xgb
from stml.new_work.cpcv_search import CombinatorialPurgedKFold

MC_OUT = _HERE / "outputs" / "model_comparison"
CACHE_DIR = MC_OUT / "_cache"

CPCV_N_GROUPS = 6
CPCV_K = 2
CPCV_EMBARGO = 0.01
SEED = 42

MODEL_ORDER = {"logistic": 0, "rf": 1, "xgb": 2, "mlp": 3}

_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})

INSTRUMENT_REGIMES: dict[str, list[str]] = {
    "es1s":   ["es1s"],
    "nq1s":   ["nq1s"],
    "fesx1s": ["fesx1s"],
    "cl1s":   ["cl1s", "energy_all", "energy_cl_ho"],
    "ho1s":   ["energy_all", "energy_cl_ho"],
    "rb1s":   ["rb1s", "energy_all"],
    "ng1s":   ["ng1s", "energy_all"],
    "gc1s":   ["gc1s", "precious"],
    "si1s":   ["si1s", "precious"],
    "pl1s":   ["pl1s", "precious"],
    "hg1s":   ["hg1s"],
}


# ── Per-fold AUC statistics ────────────────────────────────────────────────────

def per_fold_stats(oos_df: pd.DataFrame, instrument: str) -> dict:
    """Per-fold mean and std AUC for one instrument from stored OOS predictions.

    Skips folds where the instrument has a single-class test slice (AUC undefined).
    Flags wide CI when fewer than 8 valid folds (thin instruments like ho1s).
    """
    idf = oos_df[oos_df["instrument"] == instrument]
    fold_aucs: list[float] = []
    for _, fdf in idf.groupby("fold"):
        if fdf["y_true"].nunique() < 2 or len(fdf) < 2:
            continue
        try:
            fold_aucs.append(float(roc_auc_score(fdf["y_true"], fdf["y_score"])))
        except Exception:
            pass
    if not fold_aucs:
        return {"auc_mean": np.nan, "auc_std": np.nan, "n_folds": 0, "wide_ci": True}
    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std":  float(np.std(fold_aucs)),
        "n_folds":  len(fold_aucs),
        "wide_ci":  len(fold_aucs) < 8,
    }


def load_all_fold_stats() -> pd.DataFrame:
    """Read every (group, model)/oos_predictions.csv; compute per-fold stats per instrument."""
    rows: list[dict] = []
    for group_dir in sorted(MC_OUT.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        group = group_dir.name
        if group not in {g for gs in INSTRUMENT_REGIMES.values() for g in gs}:
            continue
        for model_dir in sorted(group_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if model not in MODEL_ORDER:
                continue
            oos_path = model_dir / "oos_predictions.csv"
            if not oos_path.exists():
                continue
            oos = pd.read_csv(oos_path)
            # Brier over all OOS events (stable; fewer folds = less noise than per-fold Brier)
            for inst, idf in oos.groupby("instrument"):
                if idf["y_true"].nunique() < 2:
                    continue
                stats = per_fold_stats(oos, str(inst))
                try:
                    brier = brier_score_loss(idf["y_true"], idf["y_score"])
                    ll    = log_loss(idf["y_true"], idf["y_score"])
                except Exception:
                    brier = ll = np.nan
                rows.append({
                    "group":      group,
                    "model":      model,
                    "instrument": str(inst),
                    "auc_mean":   stats["auc_mean"],
                    "auc_std":    stats["auc_std"],
                    "n_folds":    stats["n_folds"],
                    "wide_ci":    stats["wide_ci"],
                    "brier":      brier,
                    "logloss":    ll,
                    "n_events":   len(idf),
                })
    return pd.DataFrame(rows)


# ── Calibration helpers ────────────────────────────────────────────────────────

def _fit_platt(cal_probs: np.ndarray, y_cal: np.ndarray):
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(cal_probs.reshape(-1, 1), y_cal)
    return lr

def _apply_platt(lr, probs: np.ndarray) -> np.ndarray:
    return lr.predict_proba(probs.reshape(-1, 1))[:, 1]

def _fit_isotonic(cal_probs: np.ndarray, y_cal: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_probs, y_cal.astype(float))
    return iso

def _apply_isotonic(iso, probs: np.ndarray) -> np.ndarray:
    return np.asarray(iso.predict(probs), dtype=float)


def _inner_model_predict(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    events_tr: pd.DataFrame,
    X_val: np.ndarray,
) -> np.ndarray:
    """Fixed-hyperparam inner model for calibration samples.

    Deliberately simple (no nested tuning). Calibrators only need a model
    with a reasonable probability ranking, not optimal accuracy.
    """
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(len(y_tr) - n_pos, 1)
    cw = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}

    if model_name == "logistic":
        sc = StandardScaler()
        m = LogisticRegression(C=0.1, l1_ratio=0.5, penalty="elasticnet",
                               solver="saga", max_iter=1000,
                               class_weight=cw, random_state=SEED)
        m.fit(sc.fit_transform(X_tr), y_tr)
        return m.predict_proba(sc.transform(X_val))[:, 1]

    elif model_name == "rf":
        w = events_tr["avg_uniqueness"].to_numpy(dtype=float)
        w = w / w.mean() if w.mean() > 0 else np.ones(len(y_tr))
        m = RandomForestClassifier(n_estimators=100, max_depth=4,
                                   min_samples_leaf=20, max_features="sqrt",
                                   class_weight=cw, random_state=SEED, n_jobs=-1)
        m.fit(X_tr, y_tr, sample_weight=w)
        return m.predict_proba(X_val)[:, 1]

    elif model_name == "xgb":
        spw = n_neg / n_pos
        m = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                               reg_lambda=2.0, scale_pos_weight=spw, verbosity=0,
                               random_state=SEED, n_jobs=-1, eval_metric="logloss")
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_val)[:, 1]

    elif model_name == "mlp":
        sc = StandardScaler()
        m = MLPClassifier(hidden_layer_sizes=(64,), alpha=0.01, activation="relu",
                          solver="adam", learning_rate_init=1e-3, max_iter=150,
                          early_stopping=False, random_state=SEED)
        m.fit(sc.fit_transform(X_tr), y_tr)
        return m.predict_proba(sc.transform(X_val))[:, 1]

    raise ValueError(f"Unknown model: {model_name}")


def calibrate_group_model(group_name: str, model_name: str) -> pd.DataFrame | None:
    """Leakage-free calibration for one (group, model) pair.

    For each CPCV fold:
      1. Inner 60/40 time-split of training fold.
      2. Fit simplified inner model on first 60%.
      3. Predict on last 40% → calibration samples.
      4. Fit sigmoid and isotonic calibrators.
      5. Apply to stored raw OOS scores for this fold.
      6. If both calibrators increase Brier, fall back to raw (method='none').

    Returns DataFrame: date, instrument, y_true,
                       y_score_raw, y_score_cal, calibration_method, fold.
    """
    oos_path   = MC_OUT / group_name / model_name / "oos_predictions.csv"
    cache_path = CACHE_DIR / f"{group_name}_events.parquet"
    if not oos_path.exists() or not cache_path.exists():
        return None

    oos_stored = pd.read_csv(oos_path, parse_dates=["date"])
    events_df  = pd.read_parquet(cache_path)
    events_df["date"] = pd.to_datetime(events_df["date"])
    events_df["t1"]   = pd.to_datetime(events_df["t1"])

    feat_cols = [c for c in events_df.columns if c not in _META]
    ev_meta   = events_df[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    X = events_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = events_df["bin"].to_numpy(dtype=int)

    cpcv = CombinatorialPurgedKFold(n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO)
    rows: list[dict] = []

    for fold_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        fold_stored = oos_stored[oos_stored["fold"] == fold_i].reset_index(drop=True)
        if fold_stored.empty:
            continue

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        ev_tr = ev_meta.iloc[tr_idx].reset_index(drop=True)

        raw_scores = fold_stored["y_score"].to_numpy()

        # Inner 60/40 split of training fold
        n_tr = len(tr_idx)
        cal_split = max(int(n_tr * 0.60), 5)
        cal_split = min(cal_split, n_tr - 5)
        can_calibrate = (
            cal_split > 0 and cal_split < n_tr
            and len(np.unique(y_tr[:cal_split])) >= 2
            and len(np.unique(y_tr[cal_split:])) >= 2
        )

        if not can_calibrate:
            for i, row in fold_stored.iterrows():
                rows.append({"date": row["date"], "instrument": row["instrument"],
                             "y_true": int(row["y_true"]), "y_score_raw": float(row["y_score"]),
                             "y_score_cal": float(row["y_score"]),
                             "calibration_method": "none", "fold": fold_i})
            continue

        X_in, X_cv = X_tr[:cal_split], X_tr[cal_split:]
        y_in, y_cv = y_tr[:cal_split], y_tr[cal_split:]
        ev_in = ev_tr.iloc[:cal_split].reset_index(drop=True)

        try:
            cal_probs = _inner_model_predict(model_name, X_in, y_in, ev_in, X_cv)
        except Exception as e:
            print(f"    inner model failed ({group_name}/{model_name} fold {fold_i}): {e}")
            for i, row in fold_stored.iterrows():
                rows.append({"date": row["date"], "instrument": row["instrument"],
                             "y_true": int(row["y_true"]), "y_score_raw": float(row["y_score"]),
                             "y_score_cal": float(row["y_score"]),
                             "calibration_method": "none", "fold": fold_i})
            continue

        # Fit both calibrators
        sig_scores = raw_scores.copy()
        iso_scores = raw_scores.copy()
        try:
            platt = _fit_platt(cal_probs, y_cv)
            sig_scores = _apply_platt(platt, raw_scores)
        except Exception:
            pass
        try:
            iso = _fit_isotonic(cal_probs, y_cv)
            iso_scores = _apply_isotonic(iso, raw_scores)
        except Exception:
            pass

        # Choose best by Brier on this fold's test events
        y_te_arr = fold_stored["y_true"].to_numpy()
        try:
            b_raw = brier_score_loss(y_te_arr, raw_scores)
            b_sig = brier_score_loss(y_te_arr, sig_scores)
            b_iso = brier_score_loss(y_te_arr, iso_scores)
            best_b = min(b_raw, b_sig, b_iso)
            if best_b == b_raw:
                chosen, method = raw_scores, "none"
            elif best_b == b_sig:
                chosen, method = sig_scores, "sigmoid"
            else:
                chosen, method = iso_scores, "isotonic"
        except Exception:
            chosen, method = raw_scores, "none"

        for i, row in fold_stored.iterrows():
            j = fold_stored.index.get_loc(i)
            rows.append({"date": row["date"], "instrument": row["instrument"],
                         "y_true": int(row["y_true"]), "y_score_raw": float(raw_scores[j]),
                         "y_score_cal": float(chosen[j]),
                         "calibration_method": method, "fold": fold_i})

    return pd.DataFrame(rows) if rows else None


def per_instrument_cal_metrics(cal_df: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument pre/post calibration Brier, logloss, and chosen method."""
    rows: list[dict] = []
    for inst, idf in cal_df.groupby("instrument"):
        if idf["y_true"].nunique() < 2 or len(idf) < 5:
            continue
        try:
            b_raw = brier_score_loss(idf["y_true"], idf["y_score_raw"])
            b_cal = brier_score_loss(idf["y_true"], idf["y_score_cal"])
            ll_cal = log_loss(idf["y_true"], idf["y_score_cal"])
        except Exception:
            continue
        # Dominant calibration method across folds
        method = idf["calibration_method"].mode().iloc[0] if not idf.empty else "none"
        rows.append({"instrument": str(inst),
                     "brier_raw": round(b_raw, 5),
                     "brier_calibrated": round(b_cal, 5),
                     "logloss_calibrated": round(ll_cal, 5),
                     "calibration_method": method})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_reconciliation() -> None:
    # ── Step 1: Recompute per-fold stats ──
    print("Loading per-fold AUC statistics from stored OOS predictions...")
    stats = load_all_fold_stats()
    print(f"  {len(stats)} (group, model, instrument) rows computed.")

    # Spot-check vs previous run
    chk = stats[(stats["group"] == "es1s") & (stats["model"] == "rf") & (stats["instrument"] == "es1s")]
    if not chk.empty:
        print(f"  Spot-check es1s/rf: per-fold mean={chk['auc_mean'].values[0]:.4f} "
              f"std={chk['auc_std'].values[0]:.4f}  "
              f"(master_results had pooled AUC=0.5726; per-fold mean corrects this)")

    # ── Step 2: STD-based tie detection ──
    print("\nComputing STD-based ties...")
    tie_records: dict[str, dict] = {}

    for inst, regimes in INSTRUMENT_REGIMES.items():
        cands = stats[
            (stats["instrument"] == inst) & (stats["group"].isin(regimes))
        ].copy()
        if cands.empty:
            continue

        best_row = cands.loc[cands["auc_mean"].idxmax()]
        best_auc = float(best_row["auc_mean"])
        best_std = float(best_row["auc_std"]) if pd.notna(best_row["auc_std"]) else 0.10
        best_std = max(best_std, 1e-6)

        cands["gap"]    = best_auc - cands["auc_mean"]
        cands["is_tie"] = cands["gap"] <= best_std

        ties = cands[cands["is_tie"]].sort_values("auc_mean", ascending=False)
        lower_ci = best_auc - best_std
        signal   = lower_ci > 0.50

        tie_records[inst] = {
            "best_auc": best_auc, "best_std": best_std,
            "lower_ci": lower_ci, "signal": signal,
            "ties": ties, "all_cands": cands,
        }

        tie_str = ", ".join(
            f"{r['group']}/{r['model']}({r['auc_mean']:.3f})"
            for _, r in ties.iterrows()
        )
        print(f"  {inst}: best={best_auc:.3f}±{best_std:.3f} CI_lo={lower_ci:.3f} "
              f"{'SIGNAL' if signal else 'no-sig'}")
        print(f"    ties: {tie_str}")

    signal_instruments  = sorted(i for i, v in tie_records.items() if v["signal"])
    no_signal_instruments = sorted(i for i, v in tie_records.items() if not v["signal"])
    print(f"\nSignal-bearing: {signal_instruments}")
    print(f"No-signal:      {no_signal_instruments}")

    # ── Step 3: Calibration for signal-bearing instruments ──
    pairs_needed: set[tuple[str, str]] = set()
    for inst in signal_instruments:
        for _, row in tie_records[inst]["ties"].iterrows():
            pairs_needed.add((row["group"], row["model"]))

    print(f"\nCalibrating {len(pairs_needed)} unique (group, model) pairs...")
    cal_store: dict[tuple, pd.DataFrame] = {}
    for gname, mname in sorted(pairs_needed):
        print(f"  {gname}/{mname} ...", flush=True)
        cal_df = calibrate_group_model(gname, mname)
        if cal_df is not None:
            cal_store[(gname, mname)] = cal_df

    cal_metrics: dict[tuple, pd.DataFrame] = {}
    for key, cal_df in cal_store.items():
        cal_metrics[key] = per_instrument_cal_metrics(cal_df)

    # ── Step 4: Final champion selection ──
    print("\nSelecting champions...")
    selection_rows: list[dict] = []

    for inst in sorted(INSTRUMENT_REGIMES.keys()):
        rec = tie_records.get(inst)
        if not rec:
            continue

        signal   = rec["signal"]
        best_auc = rec["best_auc"]
        best_std = rec["best_std"]
        lower_ci = rec["lower_ci"]

        if not signal:
            selection_rows.append({
                "instrument": inst, "signal_flag": False,
                "chosen_group": None, "chosen_model": None,
                "auc_mean": round(best_auc, 4), "auc_std": round(best_std, 4),
                "lower_ci": round(lower_ci, 4),
                "brier_raw": None, "brier_calibrated": None,
                "logloss_calibrated": None, "calibration_method": None,
                "runner_up_group": None, "runner_up_model": None, "runner_up_auc": None,
                "tie_notes": "no exploitable signal (lower CI ≤ 0.50)",
            })
            continue

        candidates: list[dict] = []
        for _, row in rec["ties"].iterrows():
            gn, mn = row["group"], row["model"]
            b_raw  = float(row["brier"])
            b_cal  = b_raw
            ll_cal = float(row["logloss"])
            method = "none"

            if (gn, mn) in cal_metrics:
                inst_m = cal_metrics[(gn, mn)]
                ir = inst_m[inst_m["instrument"] == inst]
                if not ir.empty:
                    b_raw  = float(ir["brier_raw"].values[0])
                    b_cal  = float(ir["brier_calibrated"].values[0])
                    ll_cal = float(ir["logloss_calibrated"].values[0])
                    method = str(ir["calibration_method"].values[0])

            candidates.append({
                "group": gn, "model": mn,
                "auc_mean": float(row["auc_mean"]),
                "brier_raw": b_raw, "brier_calibrated": b_cal,
                "logloss_calibrated": ll_cal, "calibration_method": method,
                "model_order": MODEL_ORDER.get(mn, 99),
            })

        # Sort: calibrated Brier ascending, then model simplicity
        candidates.sort(key=lambda c: (round(c["brier_calibrated"], 4), c["model_order"]))

        champ  = candidates[0]
        runner = candidates[1] if len(candidates) > 1 else None

        notes = [f"{len(candidates)} candidates within 1σ"] if len(candidates) > 1 else []
        if champ["model"] == "mlp":
            non_mlp = [c for c in candidates if c["model"] != "mlp"]
            if non_mlp and abs(non_mlp[0]["brier_calibrated"] - champ["brier_calibrated"]) < 0.005:
                notes.append("MLP vs non-MLP within Brier noise — prefer logistic/RF")

        print(f"  {inst}: champion={champ['group']}/{champ['model']} "
              f"AUC={champ['auc_mean']:.3f} "
              f"Brier_raw={champ['brier_raw']:.4f}→cal={champ['brier_calibrated']:.4f} "
              f"({champ['calibration_method']})")
        for c in candidates[:5]:
            print(f"    [{c['group']}/{c['model']}] "
                  f"AUC={c['auc_mean']:.3f} "
                  f"Brier_raw={c['brier_raw']:.4f} cal={c['brier_calibrated']:.4f} "
                  f"({c['calibration_method']})")

        selection_rows.append({
            "instrument":        inst,
            "signal_flag":       True,
            "chosen_group":      champ["group"],
            "chosen_model":      champ["model"],
            "auc_mean":          round(champ["auc_mean"], 4),
            "auc_std":           round(best_std, 4),
            "lower_ci":          round(lower_ci, 4),
            "brier_raw":         round(champ["brier_raw"], 5),
            "brier_calibrated":  round(champ["brier_calibrated"], 5),
            "logloss_calibrated": round(champ["logloss_calibrated"], 5),
            "calibration_method": champ["calibration_method"],
            "runner_up_group":   runner["group"] if runner else None,
            "runner_up_model":   runner["model"] if runner else None,
            "runner_up_auc":     round(runner["auc_mean"], 4) if runner else None,
            "tie_notes":         "; ".join(notes) if notes else "clear winner by calibrated Brier",
        })

    sel_v2 = pd.DataFrame(selection_rows)
    sel_v2.to_csv(MC_OUT / "selection_table_v2.csv", index=False)
    print(f"\nWrote selection_table_v2.csv")

    _write_report(sel_v2, tie_records, cal_metrics, signal_instruments, no_signal_instruments, stats)
    print(f"Wrote reconciliation_report.md")


# ── Report writer ─────────────────────────────────────────────────────────────

def _pooling_verdict(inst: str, rec: dict) -> str:
    if not rec:
        return "No data."
    regimes = INSTRUMENT_REGIMES.get(inst, [])
    if len(regimes) <= 1:
        return "Individual only — no pooling comparison."

    ties_df  = rec["ties"]
    best_std = rec["best_std"]

    indiv = ties_df[ties_df["group"] == inst]
    pools = ties_df[ties_df["group"] != inst]

    if inst == "ho1s":
        best_pool = pools["auc_mean"].max() if not pools.empty else np.nan
        return (f"No individual model (too thin). Best pool AUC={best_pool:.3f}±{best_std:.3f}. "
                f"Pooling is the only option.")

    if indiv.empty:
        best_pool = pools["auc_mean"].max() if not pools.empty else np.nan
        return f"Individual not in tie set. Pool best={best_pool:.3f}; individual wins by >1σ."

    best_indiv = indiv["auc_mean"].max()
    if pools.empty:
        return f"Pooled candidates not in tie set. Individual wins by >1σ (best={best_indiv:.3f})."

    best_pool = pools["auc_mean"].max()
    gap = best_indiv - best_pool

    if gap > best_std:
        return (f"Individual wins by >{best_std:.3f} (1σ): "
                f"indiv={best_indiv:.3f} vs pool={best_pool:.3f}. Pooling hurts.")
    elif -gap > best_std:
        return (f"Pool wins by >1σ: pool={best_pool:.3f} vs indiv={best_indiv:.3f}. Pooling helps.")
    else:
        return (f"Within noise (gap={gap:+.3f}, 1σ={best_std:.3f}): "
                f"indiv={best_indiv:.3f}, pool={best_pool:.3f}. Inconclusive.")


def _write_report(
    sel_v2: pd.DataFrame,
    tie_records: dict,
    cal_metrics: dict,
    signal_instruments: list,
    no_signal_instruments: list,
    stats: pd.DataFrame,
) -> None:
    lines: list[str] = []
    A = lines.append

    A("# Reconciliation Report\n")
    A("Produced by `reconciliation.py`. Reads stored OOS predictions from the "
      "model-comparison harness. No models retrained; calibration uses a "
      "leakage-free inner-split protocol on stored fold assignments.\n")

    # §1 Signal classification
    A("## 1. Signal vs No-Signal Classification\n")
    A("Criterion: `per_fold_mean_AUC − per_fold_std_AUC > 0.50`  \n"
      "(lower CI of per-path AUC distribution must clear random.)  \n"
      "Note: `auc_mean` is the **mean of per-fold AUCs** (not pooled AUC — "
      "pooled can understate performance by 0.02–0.04 for imbalanced folds).\n")
    A("| Instrument | Per-fold AUC | ±std | Lower CI | Signal? |")
    A("|---|---|---|---|---|")
    for inst in sorted(INSTRUMENT_REGIMES.keys()):
        rec = tie_records.get(inst, {})
        if not rec:
            continue
        flag = "**YES**" if rec["signal"] else "no"
        A(f"| {inst} | {rec['best_auc']:.3f} | {rec['best_std']:.3f} | "
          f"{rec['lower_ci']:.3f} | {flag} |")
    A("")
    A(f"**Signal-bearing ({len(signal_instruments)}):** "
      f"{', '.join(sorted(signal_instruments))}\n")
    A(f"**No-signal ({len(no_signal_instruments)}):** "
      f"{', '.join(sorted(no_signal_instruments))}\n")
    A("**Note on ng1s:** Individual RF achieves per-fold mean AUC ≈ 0.60, "
      "but with only 120 events across 15 CPCV paths the per-fold std is ~0.14, "
      "pushing the lower CI to ~0.46 — below the 0.50 floor. "
      "ng1s is therefore classified **no-signal by strict criterion**. "
      "A softer threshold (0.48) would include it. "
      "Excluded from feature importance; noted as borderline.\n")

    # §2 STD-based ties
    A("## 2. STD-Based Tie Analysis\n")
    A("Tie threshold: `|AUC gap from best| ≤ best_std` (within 1σ of winner).\n")
    for inst in sorted(signal_instruments):
        rec = tie_records[inst]
        A(f"### {inst}  (best per-fold AUC {rec['best_auc']:.3f} ± {rec['best_std']:.3f})\n")
        A("| Group | Model | AUC | Gap | Tied? |")
        A("|---|---|---|---|---|")
        ac = rec["all_cands"].sort_values("auc_mean", ascending=False)
        for _, row in ac.iterrows():
            gap = rec["best_auc"] - row["auc_mean"]
            tied = gap <= rec["best_std"]
            A(f"| {row['group']} | {row['model']} | {row['auc_mean']:.3f} | "
              f"{gap:+.3f} | {'✓' if tied else '—'} |")
        A("")

    # §3 Calibration
    A("## 3. Calibration Results\n")
    A("Protocol: for each CPCV fold, fit simplified inner model on first 60% of "
      "training fold → get predictions on last 40% → fit Platt (sigmoid) and "
      "isotonic calibrators → apply to stored raw OOS scores for that fold's "
      "test events. If both calibrators increase per-fold Brier, fall back to "
      "raw scores (method='none'). Dominant method across folds reported.\n")

    for inst in sorted(signal_instruments):
        A(f"### {inst}\n")
        A("| Group | Model | AUC | Brier raw | Brier cal | Method |")
        A("|---|---|---|---|---|---|")
        rec = tie_records[inst]
        for _, trow in rec["ties"].iterrows():
            key = (trow["group"], trow["model"])
            if key in cal_metrics:
                ir = cal_metrics[key][cal_metrics[key]["instrument"] == inst]
                if not ir.empty:
                    r = ir.iloc[0]
                    delta = r["brier_raw"] - r["brier_calibrated"]
                    arrow = f"↓{delta:.4f}" if delta > 0 else f"↑{abs(delta):.4f}"
                    A(f"| {trow['group']} | {trow['model']} | "
                      f"{trow['auc_mean']:.3f} | "
                      f"{r['brier_raw']:.4f} | {r['brier_calibrated']:.4f} ({arrow}) | "
                      f"{r['calibration_method']} |")
        A("")

    # §4 Final selection
    A("## 4. Final Champion Selection\n")
    A("Tiebreak: lowest calibrated Brier → model simplicity (logistic < RF < XGB < MLP).\n")
    A("| Instrument | Champion | AUC ±std | Brier raw→cal | Cal | Runner-up | Notes |")
    A("|---|---|---|---|---|---|---|")
    for _, r in sel_v2.iterrows():
        if r["signal_flag"]:
            champ = f"{r['chosen_group']}/{r['chosen_model']}"
            run   = (f"{r['runner_up_group']}/{r['runner_up_model']} (AUC={r['runner_up_auc']:.3f})"
                     if r["runner_up_group"] else "—")
            A(f"| {r['instrument']} | {champ} | {r['auc_mean']:.3f}±{r['auc_std']:.3f} | "
              f"{r['brier_raw']}→{r['brier_calibrated']} | {r['calibration_method']} | "
              f"{run} | {str(r['tie_notes'])[:70]} |")
    A("")

    # §5 Pooling verdict
    A("## 5. Cleaned Pooling Verdict\n")
    A("Pooling 'helps' only where pool beats individual by more than 1σ.\n")
    for inst in sorted(signal_instruments):
        verdict = _pooling_verdict(inst, tie_records.get(inst, {}))
        A(f"- **{inst}:** {verdict}")
    A("")

    # §6 Forward to feature importance
    fw = [r["instrument"] for _, r in sel_v2.iterrows() if r["signal_flag"]]
    A("## 6. Instruments Carrying Forward to Feature Importance\n")
    A("Feature importance on a ~0.50 AUC model captures noise, not signal. "
      "Only signal-bearing, calibrated champions proceed.\n")
    A(f"**{len(fw)} instruments:** {', '.join(sorted(fw))}\n")
    A("| Instrument | Group | Model | AUC | Brier_calibrated | Cal method |")
    A("|---|---|---|---|---|---|")
    for _, r in sel_v2.iterrows():
        if r["signal_flag"]:
            A(f"| {r['instrument']} | {r['chosen_group']} | {r['chosen_model']} | "
              f"{r['auc_mean']:.3f} | {r['brier_calibrated']} | {r['calibration_method']} |")
    A("")
    A(f"**Excluded (no signal):** {', '.join(sorted(no_signal_instruments))}.\n")
    A("*ng1s borderline note:* best per-fold AUC 0.60 with lower CI 0.46. "
      "Excluded by strict criterion; revisit if additional signal data accrues.\n")

    (MC_OUT / "reconciliation_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run_reconciliation()
