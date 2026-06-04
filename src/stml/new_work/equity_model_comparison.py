"""equity_model_comparison.py — Variant comparison for equity instruments.

Phase 1: CPCV (n_groups=6, k=2, embargo=0.01) on TRAIN only.
         Lock variant per asset by 1SE rule (simplest within 1SE of full).
Phase 2: Full-train refit, single-shot OOS on sealed test.

Instrument families (champion):
    nq1s   – XGB
    fesx1s – Logistic elastic-net
    es1s   – RF (exploratory; no confirmed signal)

Variants:
    nq1s:   full | pruned (F5_signal + C4_f11)
                | reduced (f5_signal + f11_dist_stock_surprise
                           + f11_copper_stock_z + f11_vix_5d_change)
    fesx1s: full | pruned (F5_signal + C10_f12 + C13_f7)
                | reduced (PC1–PC4 of F5_signal + f12_mra_energy_D1
                           + f7_oi_z_20 + f1_ret_reversal_40 + f1_dist_ma_sigma_10)
    es1s:   full | pruned (F5_signal + C14_f11 + C2_f2)
                | reduced (PC1+PC2 of C14_f11, PC1+PC2 of C2_f2,
                           f5_trailing_run_length, f7_oi_z_20)

PCA discipline: StandardScaler + PCA fitted per CPCV fold on TRAIN rows only;
                transform test rows with fold-fitted transformers.
                Full-train PCA for Phase 2 refit.

Usage
-----
    python -m stml.new_work.equity_model_comparison
    python -m stml.new_work.equity_model_comparison --phase1-only
    python -m stml.new_work.equity_model_comparison --force

Outputs: outputs/model_comparison/equity/
    cpcv_results.csv          – Phase 1 mean±std across 15 CPCV paths
    locked_picks.csv          – locked variant per instrument (written before Phase 2)
    oos_results.csv           – Phase 2 single-shot OOS metrics
    {inst}_cpcv_chart.png
    oos_summary_chart.png
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work.cpcv_search import CombinatorialPurgedKFold
from stml.new_work.feature_importance import build_feature_matrix, load_all_data
from stml.new_work.model_comparison import (
    _PurgedKFold,
    _safe_auc,
    _tune_fit_logistic,
    _tune_fit_rf,
    _tune_fit_xgb,
)
from stml.new_work import split_config
from stml.na_checks import native_returns, wide_returns

# ── Constants ─────────────────────────────────────────────────────────────────

CPCV_N_GROUPS = 6
CPCV_K = 2
CPCV_EMBARGO = 0.01
SEED = 42

_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})

INST_FAMILY: dict[str, str] = {
    "nq1s":   "xgb",
    "fesx1s": "logistic",
    "es1s":   "rf",
}

# Simplicity order: index 0 = most regularised; 1SE rule picks lowest index.
SIMPLICITY = ["reduced", "pruned", "full"]

CACHE_DIR = _HERE / "outputs" / "model_comparison" / "_cache"
EQUITY_OUT = _HERE / "outputs" / "model_comparison" / "equity"
IMP_DIR    = _HERE / "outputs" / "importance"

# ── Variant specifications ─────────────────────────────────────────────────────

VARIANT_SPECS: dict[str, dict[str, dict]] = {
    "nq1s": {
        "full": {"mode": "raw_all"},
        "pruned": {
            "mode": "raw_clusters",
            "clusters": ["F5_signal", "C4_f11"],
        },
        "reduced": {
            "mode": "raw_select",
            "features": [
                "f5_signal",
                "f11_dist_stock_surprise",
                "f11_copper_stock_z",
                "f11_vix_5d_change",
            ],
        },
    },
    "fesx1s": {
        "full": {"mode": "raw_all"},
        "pruned": {
            "mode": "raw_clusters",
            "clusters": ["F5_signal", "C10_f12", "C13_f7"],
        },
        "reduced": {
            "mode": "pca_plus_raw",
            "pca_specs": [{"cluster": "F5_signal", "n_components": 4}],
            "raw_features": [
                "f12_mra_energy_D1",
                "f7_oi_z_20",
                "f1_ret_reversal_40",
                "f1_dist_ma_sigma_10",
            ],
        },
    },
    "es1s": {
        "full": {"mode": "raw_all"},
        "pruned": {
            "mode": "raw_clusters",
            "clusters": ["F5_signal", "C14_f11", "C2_f2"],
        },
        "reduced": {
            "mode": "pca_plus_raw",
            "pca_specs": [
                {"cluster": "C14_f11", "n_components": 2},
                {"cluster": "C2_f2",   "n_components": 2},
            ],
            "raw_features": ["f5_trailing_run_length", "f7_oi_z_20"],
        },
    },
}


# ── Feature helpers ────────────────────────────────────────────────────────────

def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


def load_cluster_membership(inst: str) -> pd.DataFrame:
    p = IMP_DIR / inst / "cluster_membership.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["cluster", "feature"])


# ── Variant transform ──────────────────────────────────────────────────────────

def transform_variant(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    feat_cols: list[str],
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_tr_v, X_te_v, feature_names) for the given variant.

    PCA fitted on X_tr rows only; transforms applied to both.
    """
    mode = spec["mode"]

    if mode == "raw_all":
        return X_tr, X_te, list(feat_cols)

    if mode == "raw_clusters":
        members = (
            cluster_df[cluster_df["cluster"].isin(spec["clusters"])]["feature"]
            .tolist()
        )
        keep = [f for f in members if f in feat_cols]
        if not keep:
            return X_tr, X_te, list(feat_cols)
        idx = [feat_cols.index(f) for f in keep]
        return X_tr[:, idx], X_te[:, idx], keep

    if mode == "raw_select":
        keep = [f for f in spec["features"] if f in feat_cols]
        idx = [feat_cols.index(f) for f in keep]
        return X_tr[:, idx], X_te[:, idx], keep

    if mode == "pca_plus_raw":
        parts_tr: list[np.ndarray] = []
        parts_te: list[np.ndarray] = []
        names: list[str] = []

        for ps in spec["pca_specs"]:
            cname = ps["cluster"]
            n_comp = ps["n_components"]
            members = (
                cluster_df[cluster_df["cluster"] == cname]["feature"].tolist()
            )
            members = [f for f in members if f in feat_cols]
            if not members:
                continue
            idx = [feat_cols.index(f) for f in members]
            Xc_tr = X_tr[:, idx]
            Xc_te = X_te[:, idx]
            n_comp_act = min(n_comp, len(members), Xc_tr.shape[0])
            sc  = StandardScaler()
            pca = PCA(n_components=n_comp_act, random_state=SEED)
            pca.fit(sc.fit_transform(Xc_tr))
            parts_tr.append(pca.transform(sc.transform(Xc_tr)))
            parts_te.append(pca.transform(sc.transform(Xc_te)))
            names.extend([f"{cname}_PC{i+1}" for i in range(n_comp_act)])

        raw = [f for f in spec["raw_features"] if f in feat_cols]
        raw_idx = [feat_cols.index(f) for f in raw]
        parts_tr.append(X_tr[:, raw_idx])
        parts_te.append(X_te[:, raw_idx])
        names.extend(raw)

        X_tr_v = np.hstack(parts_tr) if parts_tr else X_tr[:, []]
        X_te_v = np.hstack(parts_te) if parts_te else X_te[:, []]
        return X_tr_v, X_te_v, names

    raise ValueError(f"Unknown variant mode: {mode!r}")


# ── CPCV variant scoring ───────────────────────────────────────────────────────

def run_variant_cpcv(
    inst: str,
    variant: str,
    events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float]]:
    """Run CPCV for one instrument × one variant.

    Returns
    -------
    oos_df    : per-event predictions with fold index
    path_aucs : AUC per CPCV path (length = n_valid_paths ≤ 15)
    """
    fc = _feat_cols(events)
    X_all = events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_all = events["bin"].to_numpy(dtype=int)
    ev_meta = events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    family = INST_FAMILY[inst]

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO
    )

    oos_rows: list[dict] = []
    path_aucs: list[float] = []

    for path_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        X_tr_raw = X_all[tr_idx]
        X_te_raw = X_all[te_idx]
        y_tr = y_all[tr_idx]
        y_te = y_all[te_idx]
        ev_tr = ev_meta.iloc[tr_idx].reset_index(drop=True)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        try:
            X_tr, X_te, _ = transform_variant(
                X_tr_raw, X_te_raw, fc, spec, cluster_df
            )

            if family == "logistic":
                scaler, model, _ = _tune_fit_logistic(X_tr, y_tr, ev_tr)
                prob = model.predict_proba(scaler.transform(X_te))[:, 1]
            elif family == "rf":
                model, _ = _tune_fit_rf(X_tr, y_tr, ev_tr)
                prob = model.predict_proba(X_te)[:, 1]
            elif family == "xgb":
                model, _ = _tune_fit_xgb(X_tr, y_tr, ev_tr)
                prob = model.predict_proba(X_te)[:, 1]
            else:
                raise ValueError(f"Unknown family: {family}")
        except Exception as e:
            print(f"    [{inst}/{variant}] path {path_i} failed: {e}")
            traceback.print_exc()
            continue

        auc = _safe_auc(y_te, prob)
        if auc >= 0:
            path_aucs.append(auc)

        ev_te = ev_meta.iloc[te_idx].reset_index(drop=True)
        for i in range(len(te_idx)):
            oos_rows.append({
                "date":   ev_te.iloc[i]["date"],
                "y_true": int(y_te[i]),
                "y_score": float(prob[i]),
                "path":   path_i,
            })

    return pd.DataFrame(oos_rows), path_aucs


def summarise_cpcv(path_aucs: list[float], oos_df: pd.DataFrame) -> dict[str, Any]:
    """Compute mean±std AUC + overall logloss/Brier/AP from CPCV paths."""
    if not path_aucs or oos_df.empty:
        return {
            "auc_mean": np.nan, "auc_std": np.nan, "auc_se": np.nan,
            "ap": np.nan, "logloss": np.nan, "brier": np.nan, "n_paths": 0,
        }
    arr = np.array(path_aucs)
    ll = br = ap = np.nan
    if oos_df["y_true"].nunique() >= 2:
        try:
            ll = log_loss(oos_df["y_true"], oos_df["y_score"])
            br = brier_score_loss(oos_df["y_true"], oos_df["y_score"])
            ap = average_precision_score(oos_df["y_true"], oos_df["y_score"])
        except Exception:
            pass
    return {
        "auc_mean": float(arr.mean()),
        "auc_std":  float(arr.std()),
        "auc_se":   float(arr.std() / np.sqrt(len(arr))),
        "ap":       float(ap) if not np.isnan(ap) else np.nan,
        "logloss":  float(ll) if not np.isnan(ll) else np.nan,
        "brier":    float(br) if not np.isnan(br) else np.nan,
        "n_paths":  len(arr),
    }


# ── 1SE variant selection ──────────────────────────────────────────────────────

def select_locked_variant(
    cpcv_rows: list[dict],
) -> str:
    """1SE rule: simplest variant within 1SE of full.

    cpcv_rows: list of dicts with keys 'variant', 'auc_mean', 'auc_se'.
    Simplicity order: reduced (0) < pruned (1) < full (2).
    """
    by_variant = {r["variant"]: r for r in cpcv_rows}
    full_row = by_variant.get("full", {})
    full_mean = full_row.get("auc_mean", np.nan)
    full_se   = full_row.get("auc_se",   np.nan)

    if np.isnan(full_mean) or np.isnan(full_se):
        return "full"

    threshold = full_mean - full_se

    for v in SIMPLICITY:
        row = by_variant.get(v, {})
        vm = row.get("auc_mean", np.nan)
        if not np.isnan(vm) and vm >= threshold:
            return v

    return "full"


# ── Phase 2 helpers ────────────────────────────────────────────────────────────

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except Exception:
            pass
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def load_test_events(
    inst: str,
    data: dict,
    wide_rets: pd.DataFrame,
    train_feat_cols: list[str],
) -> pd.DataFrame:
    """Build full event matrix then slice test rows (SEALED).

    Uses train_feat_cols to align columns; fills missing with 0.
    """
    _, full_events = build_feature_matrix(inst, data, wide_rets)
    test_events = split_config.apply_test_mask(full_events)
    if test_events.empty:
        return pd.DataFrame()
    return test_events.reindex(
        columns=list(_META) + train_feat_cols, fill_value=0.0
    )


def refit_and_score(
    inst: str,
    variant: str,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> dict[str, Any]:
    """Refit on full train, predict once on test; compute OOS metrics."""
    fc = _feat_cols(train_events)
    X_tr_raw = train_events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_tr = train_events["bin"].to_numpy(dtype=int)
    ev_tr = train_events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    ev_tr = ev_tr.reset_index(drop=True)

    test_fc = [c for c in fc if c in test_events.columns]
    X_te_raw_df = test_events.reindex(columns=fc, fill_value=0.0)
    X_te_raw = X_te_raw_df.fillna(0.0).to_numpy(dtype=np.float64)
    y_te = test_events["bin"].to_numpy(dtype=int)

    family = INST_FAMILY[inst]

    try:
        X_tr, X_te, _ = transform_variant(
            X_tr_raw, X_te_raw, fc, spec, cluster_df
        )

        if family == "logistic":
            scaler, model, params = _tune_fit_logistic(X_tr, y_tr, ev_tr)
            prob = model.predict_proba(scaler.transform(X_te))[:, 1]
        elif family == "rf":
            model, params = _tune_fit_rf(X_tr, y_tr, ev_tr)
            prob = model.predict_proba(X_te)[:, 1]
        elif family == "xgb":
            model, params = _tune_fit_xgb(X_tr, y_tr, ev_tr)
            prob = model.predict_proba(X_te)[:, 1]
        else:
            raise ValueError(f"Unknown family: {family}")
    except Exception as e:
        print(f"  Phase 2 [{inst}/{variant}] failed: {e}")
        traceback.print_exc()
        return {"auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan,
                "ap": np.nan, "logloss": np.nan, "brier": np.nan, "n_test": len(y_te)}

    auc = _safe_auc(y_te, prob)
    ci_lo, ci_hi = bootstrap_auc_ci(y_te, prob)
    ll = br = ap = np.nan
    if len(np.unique(y_te)) >= 2:
        try:
            ll = log_loss(y_te, prob)
            br = brier_score_loss(y_te, prob)
            ap = average_precision_score(y_te, prob)
        except Exception:
            pass

    return {
        "auc": float(auc) if auc >= 0 else np.nan,
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
        "ap":        float(ap) if not np.isnan(ap) else np.nan,
        "logloss": float(ll) if not np.isnan(ll) else np.nan,
        "brier":   float(br) if not np.isnan(br) else np.nan,
        "n_test":  int(len(y_te)),
    }


# ── Charts ─────────────────────────────────────────────────────────────────────

def _cpcv_chart(
    cpcv_rows: list[dict],
    locked: str,
    inst: str,
    out_path: Path,
) -> None:
    rows = sorted(cpcv_rows, key=lambda r: SIMPLICITY.index(r["variant"]))
    variants = [r["variant"] for r in rows]
    means    = [r["auc_mean"] for r in rows]
    stds     = [r["auc_std"]  for r in rows]
    colors   = ["#2196F3" if v == locked else "#90CAF9" for v in variants]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(variants, means, yerr=stds, color=colors,
                  error_kw={"ecolor": "grey", "capsize": 4}, width=0.5)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(max(0.3, min(means) - 3 * max(stds)), min(1.0, max(means) + 3 * max(stds)))
    ax.set_xlabel("Variant")
    ax.set_ylabel("AUC (mean ± std, 15 CPCV paths)")
    ax.set_title(f"{inst.upper()} – Phase 1 CPCV  |  locked = {locked}")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _oos_summary_chart(
    oos_rows: list[dict],
    locked_picks: dict[str, str],
    out_path: Path,
) -> None:
    headline = [r for r in oos_rows if r["variant"] == locked_picks.get(r["inst"])]
    if not headline:
        return

    insts = [r["inst"] for r in headline]
    aucs  = [r["auc"] for r in headline]
    ci_lo = [r["auc"] - r["auc_ci_lo"] for r in headline]
    ci_hi = [r["auc_ci_hi"] - r["auc"] for r in headline]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(insts, aucs, yerr=[ci_lo, ci_hi], color="#2196F3",
           error_kw={"ecolor": "grey", "capsize": 4}, width=0.4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("OOS AUC (95% bootstrap CI)")
    ax.set_title("Equity – Phase 2 OOS (locked variants)")
    for i, (inst, auc) in enumerate(zip(insts, aucs)):
        ax.text(i, auc + ci_hi[i] + 0.005, f"{auc:.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(force: bool = False, phase1_only: bool = False) -> None:
    EQUITY_OUT.mkdir(parents=True, exist_ok=True)

    # ── Load train events from cache ─────────────────────────────────────────
    train_events: dict[str, pd.DataFrame] = {}
    for inst in INST_FAMILY:
        cache = CACHE_DIR / f"{inst}_events.parquet"
        if not cache.exists():
            raise FileNotFoundError(
                f"Train cache missing: {cache}\n"
                f"Run: python -m stml.new_work.model_comparison --groups {inst}"
            )
        train_events[inst] = pd.read_parquet(cache)
        fc = _feat_cols(train_events[inst])
        print(f"  {inst}: {len(train_events[inst])} train events, {len(fc)} features")

    # ── Cluster membership ───────────────────────────────────────────────────
    cluster_dfs: dict[str, pd.DataFrame] = {
        inst: load_cluster_membership(inst) for inst in INST_FAMILY
    }

    # ── Phase 1: CPCV variant scoring ────────────────────────────────────────
    cpcv_results_path = EQUITY_OUT / "cpcv_results.csv"
    locked_picks_path = EQUITY_OUT / "locked_picks.csv"

    if cpcv_results_path.exists() and not force:
        print("\n[Phase 1] Loading cached CPCV results.")
        cpcv_df = pd.read_csv(cpcv_results_path)
    else:
        print("\n" + "=" * 60)
        print("Phase 1 — CPCV variant scoring (TRAIN only)")
        print("=" * 60)

        all_cpcv_rows: list[dict] = []

        for inst in INST_FAMILY:
            ev = train_events[inst]
            cd = cluster_dfs[inst]
            specs = VARIANT_SPECS[inst]

            for variant, spec in specs.items():
                print(f"\n  [{inst}/{variant}] CPCV …")
                oos_df, path_aucs = run_variant_cpcv(inst, variant, ev, spec, cd)
                summary = summarise_cpcv(path_aucs, oos_df)
                row = {"inst": inst, "variant": variant, **summary}
                all_cpcv_rows.append(row)
                print(
                    f"    AUC {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}"
                    f"  (SE={summary['auc_se']:.4f}, n_paths={summary['n_paths']})"
                )

        cpcv_df = pd.DataFrame(all_cpcv_rows)
        cpcv_df.to_csv(cpcv_results_path, index=False)
        print(f"\n  Saved: {cpcv_results_path}")

    # ── 1SE lock ─────────────────────────────────────────────────────────────
    locked_picks: dict[str, str] = {}
    lock_rows: list[dict] = []
    for inst in INST_FAMILY:
        inst_rows = cpcv_df[cpcv_df["inst"] == inst].to_dict("records")
        locked = select_locked_variant(inst_rows)
        locked_picks[inst] = locked
        print(f"\n  [{inst}] LOCKED VARIANT: {locked}")
        for r in inst_rows:
            print(f"    {r['variant']:8s}  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
        lock_rows.append({"inst": inst, "locked_variant": locked})

    # ── Save locked picks BEFORE reading test data ────────────────────────────
    lock_df = pd.DataFrame(lock_rows)
    lock_df.to_csv(locked_picks_path, index=False)
    print(f"\n  Locked picks saved: {locked_picks_path}")

    # ── Phase 1 charts ────────────────────────────────────────────────────────
    for inst in INST_FAMILY:
        inst_rows = cpcv_df[cpcv_df["inst"] == inst].to_dict("records")
        _cpcv_chart(
            inst_rows,
            locked=locked_picks[inst],
            inst=inst,
            out_path=EQUITY_OUT / f"{inst}_cpcv_chart.png",
        )

    if phase1_only:
        print("\nPhase 1 complete. --phase1-only flag set; stopping before test.")
        return

    # ── Phase 2: single-shot OOS ─────────────────────────────────────────────
    oos_results_path = EQUITY_OUT / "oos_results.csv"
    if oos_results_path.exists() and not force:
        print("\n[Phase 2] Loading cached OOS results.")
        oos_df = pd.read_csv(oos_results_path)
    else:
        print("\n" + "=" * 60)
        print("Phase 2 — Single-shot OOS (SEALED TEST)")
        print("=" * 60)

        print("  Loading full data for test events …")
        data = load_all_data()
        rets_long = native_returns(data["ohlcv"], kind="log")
        w_rets = wide_returns(rets_long).sort_index()

        all_oos_rows: list[dict] = []

        for inst in INST_FAMILY:
            ev_tr = train_events[inst]
            fc    = _feat_cols(ev_tr)
            cd    = cluster_dfs[inst]
            specs = VARIANT_SPECS[inst]

            print(f"\n  Loading test events for {inst} …")
            ev_te = load_test_events(inst, data, w_rets, fc)
            if ev_te.empty:
                print(f"  WARNING: no test events for {inst}")
                continue
            print(f"  {inst}: {len(ev_te)} test events")

            for variant, spec in specs.items():
                is_locked = (locked_picks.get(inst) == variant)
                tag = " [LOCKED]" if is_locked else " [diagnostic]"
                print(f"  [{inst}/{variant}]{tag} refitting …")
                result = refit_and_score(inst, variant, ev_tr, ev_te, spec, cd)
                row = {"inst": inst, "variant": variant,
                       "is_locked": is_locked, **result}
                all_oos_rows.append(row)
                print(
                    f"    OOS AUC {result['auc']:.4f}"
                    f"  [{result['auc_ci_lo']:.3f}, {result['auc_ci_hi']:.3f}]"
                    f"  logloss={result['logloss']:.4f}"
                    f"  brier={result['brier']:.4f}"
                    f"  n_test={result['n_test']}"
                )

        oos_df = pd.DataFrame(all_oos_rows)
        oos_df.to_csv(oos_results_path, index=False)
        print(f"\n  Saved: {oos_results_path}")

    # ── OOS chart ─────────────────────────────────────────────────────────────
    _oos_summary_chart(
        oos_df.to_dict("records"),
        locked_picks,
        EQUITY_OUT / "oos_summary_chart.png",
    )

    # ── Dev-vs-OOS gap ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Dev-CPCV vs OOS gap (locked variants)")
    print("=" * 60)
    for inst in INST_FAMILY:
        locked = locked_picks.get(inst, "full")
        cpcv_row = cpcv_df[(cpcv_df["inst"] == inst) & (cpcv_df["variant"] == locked)]
        oos_row  = oos_df[(oos_df["inst"] == inst) & (oos_df["variant"] == locked)]
        if cpcv_row.empty or oos_row.empty:
            continue
        dev_auc = cpcv_row.iloc[0]["auc_mean"]
        oos_auc = oos_row.iloc[0]["auc"]
        gap     = dev_auc - oos_auc
        print(
            f"  {inst:8s} ({locked:8s})  "
            f"dev={dev_auc:.4f}  OOS={oos_auc:.4f}  gap={gap:+.4f}"
        )

    print("\nDone.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",       action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--phase1-only", action="store_true", help="Run Phase 1 only; stop before test")
    args = parser.parse_args()
    run(force=args.force, phase1_only=args.phase1_only)
