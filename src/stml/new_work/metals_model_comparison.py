"""metals_model_comparison.py — Variant comparison for metals instruments (new labels).

Champions (selection_table.csv, new triple-barrier labels):
    gc1s  – precious / RF  (NO SIGNAL, exploratory)
    si1s  – precious / RF  (NO SIGNAL, ~random)
    pl1s  – precious / RF  (SIGNAL, marginal, lower_ci=0.54)
    hg1s  – hg1s     / RF  (SIGNAL, marginal, lower_ci=0.51)

Pooled mechanics (gc1s / si1s / pl1s):
    Each variant is a SEPARATE precious-pool RF refit on that instrument's
    feature subset.  Training always uses ALL pooled precious rows; scoring uses
    the TARGET instrument's own CPCV test slice (Phase 1) or sealed-test slice
    (Phase 2).  "full" = all precious-pool features including inst_ dummies.

Variants (all raw SHAP representatives; no PCA):
    gc1s  full | pruned  (C5_hmm_vol + C10_f11_lowfreq_macro + C2_f1)
               | reduced (3 HMM-vol states + 3 rates + f6_ts_momentum_20)
    si1s  full | pruned  (C5_hmm_vol + C15_f11_lowfreq_macro + C13_f11_lowfreq_macro)
               | reduced (3 HMM-vol states + f2_atr_14 + f11_ust_10y + f11_real_yield_10y)
    pl1s  full | pruned  (C5_hmm_vol + C11_f11  — the 2 significant clusters)
               | reduced (3 HMM-vol + f2_atr_14 + f6_adx_14 + f6_macd_hist + f11_ust_10y)
    hg1s  full | pruned  (C1_f1 + C12_f11  — the 2 significant clusters)
               | reduced (f1_mr_score_40 + f1_hilo_pos_10 + f1_dist_ma_sigma_20
                          + f11_ust_bund_spread + f1_rsi_14 + f11_dxy_z + f12_mra_energy_D4)

Phase 1:  CPCV(n_groups=6, k=2, embargo=0.01) on TRAIN only.
          Lock = simplest variant within 1 std of full (cross-path std).
Phase 2:  Full-train refit, single-shot OOS on sealed test; scored ONCE.

Usage
-----
    python -m stml.new_work.metals_model_comparison
    python -m stml.new_work.metals_model_comparison --phase1-only
    python -m stml.new_work.metals_model_comparison --force
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work.cpcv_search import CombinatorialPurgedKFold
from stml.new_work.feature_importance import build_feature_matrix, load_all_data
from stml.new_work.model_comparison import _safe_auc, _tune_fit_rf
from stml.new_work import split_config
from stml.na_checks import native_returns, wide_returns

# ── Constants ─────────────────────────────────────────────────────────────────

CPCV_N_GROUPS = 6
CPCV_K        = 2
CPCV_EMBARGO  = 0.01
SEED          = 42

_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})

# Dummy values for precious pool instruments (gc1s = reference, alphabetically first)
_PRECIOUS_DUMMIES: dict[str, dict[str, float]] = {
    "gc1s": {"inst_pl1s": 0.0, "inst_si1s": 0.0},
    "si1s": {"inst_pl1s": 0.0, "inst_si1s": 1.0},
    "pl1s": {"inst_pl1s": 1.0, "inst_si1s": 0.0},
}

CACHE_DIR  = _HERE / "outputs" / "model_comparison" / "_cache"
METALS_OUT = _HERE / "outputs" / "model_comparison" / "metals"
IMP_DIR    = _HERE / "outputs" / "importance"

# ── Instrument configuration ───────────────────────────────────────────────────

# Simplicity order per instrument (index 0 = most regularised / fewest features)
SIMPLICITY: dict[str, list[str]] = {
    "gc1s": ["reduced", "pruned", "full"],
    "si1s": ["reduced", "pruned", "full"],
    "pl1s": ["reduced", "pruned", "full"],
    "hg1s": ["reduced", "pruned", "full"],
}

# Whether the instrument uses the pooled precious train events
_POOLED = {"gc1s", "si1s", "pl1s"}

VARIANT_SPECS: dict[str, dict[str, dict]] = {
    "gc1s": {
        "full": {"mode": "raw_all"},
        "pruned": {
            "mode":     "raw_clusters",
            "clusters": ["C5_hmm_vol", "C10_f11_lowfreq_macro", "C2_f1"],
        },
        "reduced": {
            "mode": "raw_select",
            "features": [
                "hmm_vol_p0_calm", "hmm_vol_p2_turbulent", "hmm_vol_next_turbulent",
                "f11_ust_10y_5d_change", "f11_bund_10y_5d_change", "f11_real_yield_10y",
                "f6_ts_momentum_20",
            ],
        },
    },
    "si1s": {
        "full": {"mode": "raw_all"},
        "pruned": {
            "mode":     "raw_clusters",
            "clusters": ["C5_hmm_vol", "C15_f11_lowfreq_macro", "C13_f11_lowfreq_macro"],
        },
        "reduced": {
            "mode": "raw_select",
            "features": [
                "hmm_vol_p0_calm", "hmm_vol_p2_turbulent", "hmm_vol_next_turbulent",
                "f2_atr_14", "f11_ust_10y_5d_change", "f11_real_yield_10y",
            ],
        },
    },
    "pl1s": {
        "full": {"mode": "raw_all"},
        "pruned": {
            "mode":     "raw_clusters",
            "clusters": ["C5_hmm_vol", "C11_f11"],  # the 2 significant clusters
        },
        "reduced": {
            "mode": "raw_select",
            "features": [
                "hmm_vol_p0_calm", "hmm_vol_p2_turbulent", "hmm_vol_next_turbulent",
                "f2_atr_14", "f6_adx_14", "f6_macd_hist_12_26_9", "f11_ust_10y_5d_change",
            ],
        },
    },
    "hg1s": {
        "full": {"mode": "raw_all"},
        "pruned": {
            "mode":     "raw_clusters",
            "clusters": ["C1_f1", "C12_f11"],  # the 2 significant clusters
        },
        "reduced": {
            "mode": "raw_select",
            "features": [
                "f1_mr_score_40", "f1_hilo_pos_10", "f1_dist_ma_sigma_20",
                "f11_ust_bund_spread", "f1_rsi_14", "f11_dxy_z", "f12_mra_energy_D4",
            ],
        },
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


def load_cluster_membership(inst: str) -> pd.DataFrame:
    p = IMP_DIR / inst / "cluster_membership.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["cluster", "feature"])


# ── Variant feature selection (raw only — no PCA) ─────────────────────────────

def select_variant_features(
    fc: list[str],
    spec: dict,
    cluster_df: pd.DataFrame,
) -> list[str]:
    """Return the feature column names for this variant spec."""
    mode = spec["mode"]
    if mode == "raw_all":
        return list(fc)
    if mode == "raw_clusters":
        members = cluster_df[
            cluster_df["cluster"].isin(spec["clusters"])
        ]["feature"].tolist()
        keep = [f for f in members if f in fc]
        return keep if keep else list(fc)
    if mode == "raw_select":
        return [f for f in spec["features"] if f in fc]
    raise ValueError(f"Unknown variant mode: {mode!r}")


def apply_variant(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    fc: list[str],
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Slice columns for this variant; returns (X_tr_v, X_te_v, names)."""
    keep = select_variant_features(fc, spec, cluster_df)
    if not keep:
        return X_tr, X_te, list(fc)
    idx = [fc.index(f) for f in keep]
    return X_tr[:, idx], X_te[:, idx], keep


# ── Pooled CPCV (gc1s / si1s / pl1s via precious pool) ───────────────────────

def run_cpcv_pooled(
    target_inst: str,
    variant: str,
    precious_events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[list[float], pd.DataFrame]:
    """CPCV on precious pool; score on target_inst slice only."""
    fc       = _feat_cols(precious_events)
    X_all    = precious_events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_all    = precious_events["bin"].to_numpy(dtype=int)
    ev_meta  = precious_events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    inst_arr = precious_events["instrument"].values

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO
    )
    path_aucs: list[float] = []
    oos_rows:  list[dict]  = []
    n_skipped = 0

    for path_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        te_mask = inst_arr[te_idx] == target_inst
        te_pos  = te_idx[te_mask]
        if len(te_pos) < 2:
            n_skipped += 1
            continue
        y_te = y_all[te_pos]
        if len(np.unique(y_te)) < 2:
            n_skipped += 1
            continue
        y_tr  = y_all[tr_idx]
        ev_tr = ev_meta.iloc[tr_idx].reset_index(drop=True)
        if len(np.unique(y_tr)) < 2:
            n_skipped += 1
            continue

        try:
            X_tr_v, X_te_v, _ = apply_variant(
                X_all[tr_idx], X_all[te_pos], fc, spec, cluster_df
            )
            model, _ = _tune_fit_rf(X_tr_v, y_tr, ev_tr)
            prob = model.predict_proba(X_te_v)[:, 1]
        except Exception as e:
            print(f"    [{target_inst}/{variant}] path {path_i} failed: {e}")
            n_skipped += 1
            continue

        auc = _safe_auc(y_te, prob)
        if auc is not None and auc >= 0:
            path_aucs.append(float(auc))
        ev_te = ev_meta.iloc[te_pos].reset_index(drop=True)
        for i in range(len(te_pos)):
            oos_rows.append({
                "date":    ev_te.iloc[i]["date"],
                "y_true":  int(y_te[i]),
                "y_score": float(prob[i]),
                "path":    path_i,
            })

    if n_skipped > 0:
        print(f"    [{target_inst}/{variant}] skipped {n_skipped}/15 paths")
    return path_aucs, pd.DataFrame(oos_rows)


# ── Individual CPCV (hg1s) ────────────────────────────────────────────────────

def run_cpcv_individual(
    inst: str,
    variant: str,
    events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[list[float], pd.DataFrame]:
    """CPCV on individual instrument events."""
    fc      = _feat_cols(events)
    X_all   = events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_all   = events["bin"].to_numpy(dtype=int)
    ev_meta = events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO
    )
    path_aucs: list[float] = []
    oos_rows:  list[dict]  = []

    for path_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        y_tr = y_all[tr_idx]
        y_te = y_all[te_idx]
        ev_tr = ev_meta.iloc[tr_idx].reset_index(drop=True)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        try:
            X_tr_v, X_te_v, _ = apply_variant(
                X_all[tr_idx], X_all[te_idx], fc, spec, cluster_df
            )
            model, _ = _tune_fit_rf(X_tr_v, y_tr, ev_tr)
            prob = model.predict_proba(X_te_v)[:, 1]
        except Exception as e:
            print(f"    [{inst}/{variant}] path {path_i} failed: {e}")
            continue

        auc = _safe_auc(y_te, prob)
        if auc is not None and auc >= 0:
            path_aucs.append(float(auc))
        ev_te = ev_meta.iloc[te_idx].reset_index(drop=True)
        for i in range(len(te_idx)):
            oos_rows.append({
                "date":   ev_te.iloc[i]["date"],
                "y_true": int(y_te[i]),
                "y_score": float(prob[i]),
                "path": path_i,
            })

    return path_aucs, pd.DataFrame(oos_rows)


# ── Summarise ─────────────────────────────────────────────────────────────────

def summarise_cpcv(path_aucs: list[float], oos_df: pd.DataFrame) -> dict[str, Any]:
    if not path_aucs or oos_df.empty:
        return {"auc_mean": np.nan, "auc_std": np.nan,
                "logloss": np.nan, "brier": np.nan, "n_paths": 0}
    arr  = np.array(path_aucs)
    ll = br = np.nan
    if not oos_df.empty and oos_df["y_true"].nunique() >= 2:
        try:
            ll = log_loss(oos_df["y_true"], oos_df["y_score"])
            br = brier_score_loss(oos_df["y_true"], oos_df["y_score"])
        except Exception:
            pass
    return {
        "auc_mean": float(arr.mean()),
        "auc_std":  float(arr.std()),
        "logloss":  float(ll) if not np.isnan(ll) else np.nan,
        "brier":    float(br) if not np.isnan(br) else np.nan,
        "n_paths":  len(arr),
    }


# ── Lock: simplest within 1 std of full ──────────────────────────────────────

def select_locked_variant(cpcv_rows: list[dict], simplicity: list[str]) -> str:
    by_v  = {r["variant"]: r for r in cpcv_rows}
    full  = by_v.get("full", {})
    fm    = full.get("auc_mean", np.nan)
    fstd  = full.get("auc_std",  np.nan)
    if np.isnan(fm) or np.isnan(fstd):
        return "full"
    threshold = fm - fstd          # 1 cross-path std below full
    for v in simplicity:           # simplicity[0] = most regularised
        row = by_v.get(v, {})
        vm  = row.get("auc_mean", np.nan)
        if not np.isnan(vm) and vm >= threshold:
            return v
    return "full"


# ── Bootstrap AUC CI ──────────────────────────────────────────────────────────

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    aucs: list[float] = []
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


# ── Phase 2 test data loading ─────────────────────────────────────────────────

def _load_test_pooled(
    target_inst: str,
    data: dict,
    w_rets: pd.DataFrame,
    precious_fc: list[str],
) -> pd.DataFrame:
    """Sealed-test events for a precious-pool instrument (gc1s / si1s / pl1s)."""
    _, full_ev = build_feature_matrix(target_inst, data, w_rets)
    te = split_config.apply_test_mask(full_ev)
    if te.empty:
        return pd.DataFrame()
    # Add precious-pool instrument dummies (gc1s = reference → all zeros)
    for col, val in _PRECIOUS_DUMMIES[target_inst].items():
        te[col] = val
    return te.reindex(columns=list(_META) + precious_fc, fill_value=0.0)


def _load_test_individual(
    inst: str,
    data: dict,
    w_rets: pd.DataFrame,
    fc: list[str],
) -> pd.DataFrame:
    """Sealed-test events for an individual instrument (hg1s)."""
    _, full_ev = build_feature_matrix(inst, data, w_rets)
    te = split_config.apply_test_mask(full_ev)
    if te.empty:
        return pd.DataFrame()
    return te.reindex(columns=list(_META) + fc, fill_value=0.0)


# ── Phase 2 fit + score ───────────────────────────────────────────────────────

def fit_and_score(
    inst: str,
    variant: str,
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    ev_tr: pd.DataFrame,
    X_te_raw: np.ndarray,
    y_te: np.ndarray,
    fc: list[str],
    spec: dict,
    cluster_df: pd.DataFrame,
) -> dict[str, Any]:
    _nan: dict[str, Any] = {
        "auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan,
        "logloss": np.nan, "brier": np.nan, "n_test": int(len(y_te)),
    }
    try:
        X_tr, X_te, _ = apply_variant(X_tr_raw, X_te_raw, fc, spec, cluster_df)
        model, _ = _tune_fit_rf(X_tr, y_tr, ev_tr)
        prob = model.predict_proba(X_te)[:, 1]
    except Exception as e:
        print(f"  Phase 2 [{inst}/{variant}] failed: {e}")
        traceback.print_exc()
        return _nan

    auc = _safe_auc(y_te, prob)
    ci_lo, ci_hi = bootstrap_auc_ci(y_te, prob)
    ll = br = np.nan
    if len(np.unique(y_te)) >= 2:
        try:
            ll = log_loss(y_te, prob)
            br = brier_score_loss(y_te, prob)
        except Exception:
            pass
    return {
        "auc":      float(auc) if (auc is not None and auc >= 0) else np.nan,
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
        "logloss":  float(ll) if not np.isnan(ll) else np.nan,
        "brier":    float(br) if not np.isnan(br) else np.nan,
        "n_test":   int(len(y_te)),
    }


# ── Charts ─────────────────────────────────────────────────────────────────────

def cpcv_chart(
    cpcv_rows: list[dict],
    locked: str,
    inst: str,
    simplicity: list[str],
    out_path: Path,
) -> None:
    rows     = sorted(cpcv_rows, key=lambda r: simplicity.index(r["variant"])
                      if r["variant"] in simplicity else 99)
    variants = [r["variant"] for r in rows]
    means    = [r["auc_mean"] for r in rows]
    stds     = [r.get("auc_std", 0) for r in rows]
    colors   = ["#1976D2" if v == locked else "#90CAF9" for v in variants]
    valid    = [m for m in means if not np.isnan(m)]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(max(5, len(variants) * 1.8), 4))
    ax.bar(variants, means, yerr=stds, color=colors,
           error_kw={"ecolor": "grey", "capsize": 5}, width=0.5)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="AUC = 0.5 (random)")
    full_mean = next((r["auc_mean"] for r in rows if r["variant"] == "full"), np.nan)
    full_std  = next((r.get("auc_std", np.nan) for r in rows if r["variant"] == "full"), np.nan)
    if not np.isnan(full_mean) and not np.isnan(full_std):
        ax.axhline(full_mean - full_std, color="orange", linestyle=":",
                   linewidth=1.0, label="full − 1σ (lock threshold)")
    lo = max(0.2, min(valid) - 2.5 * max((s or 0) for s in stds))
    hi = min(1.0, max(valid) + 2.5 * max((s or 0) for s in stds))
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Variant")
    ax.set_ylabel("CPCV AUC (mean ± std across 15 paths)")
    ax.set_title(f"{inst.upper()} — Phase 1 CPCV  |  locked = {locked}")
    ax.legend(fontsize=7)
    for bar, m in zip(ax.patches, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def oos_summary_chart(
    oos_rows: list[dict],
    locked_picks: dict[str, str],
    cpcv_df: pd.DataFrame,
    out_path: Path,
) -> None:
    insts     = list(VARIANT_SPECS.keys())
    locked_r  = [r for r in oos_rows if r["variant"] == locked_picks.get(r["inst"])]
    by_inst   = {r["inst"]: r for r in locked_r}
    aucs      = [by_inst.get(i, {}).get("auc", np.nan)      for i in insts]
    ci_lo_abs = [by_inst.get(i, {}).get("auc_ci_lo", np.nan) for i in insts]
    ci_hi_abs = [by_inst.get(i, {}).get("auc_ci_hi", np.nan) for i in insts]
    err_lo    = [a - lo if not np.isnan(a) else 0 for a, lo in zip(aucs, ci_lo_abs)]
    err_hi    = [hi - a if not np.isnan(a) else 0 for a, hi in zip(aucs, ci_hi_abs)]
    dev_aucs  = []
    for inst in insts:
        lv  = locked_picks.get(inst, "full")
        row = cpcv_df[(cpcv_df["inst"] == inst) & (cpcv_df["variant"] == lv)]
        dev_aucs.append(row.iloc[0]["auc_mean"] if not row.empty else np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(insts))
    ax.bar(x, aucs, yerr=[err_lo, err_hi], color="#1976D2",
           error_kw={"ecolor": "grey", "capsize": 5}, width=0.35, label="OOS AUC")
    ax.scatter(x, dev_aucs, marker="D", color="orange", zorder=5,
               s=40, label="CPCV dev AUC (locked)")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    valid_all = [v for v in aucs + dev_aucs if not np.isnan(v)]
    if valid_all:
        ax.set_ylim(max(0.2, min(valid_all) - 0.12), min(1.0, max(valid_all) + 0.12))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n({locked_picks.get(i,'?')})" for i in insts])
    ax.set_ylabel("AUC")
    ax.set_title("Metals — Phase 2 OOS (locked variants)  |  ◆ = CPCV dev AUC")
    ax.legend(fontsize=8)
    for xi, auc in zip(x, aucs):
        if not np.isnan(auc):
            ax.text(xi, auc + err_hi[int(xi)] + 0.005, f"{auc:.3f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(force: bool = False, phase1_only: bool = False) -> None:
    METALS_OUT.mkdir(parents=True, exist_ok=True)

    # ── Load train caches ─────────────────────────────────────────────────────
    precious_train = pd.read_parquet(CACHE_DIR / "precious_events.parquet")
    hg1s_train     = pd.read_parquet(CACHE_DIR / "hg1s_events.parquet")

    precious_fc = _feat_cols(precious_train)
    hg1s_fc     = _feat_cols(hg1s_train)

    print(f"precious pool: {len(precious_train)} events, {len(precious_fc)} features")
    for inst in ["gc1s", "si1s", "pl1s"]:
        n = (precious_train["instrument"] == inst).sum()
        print(f"  {inst}: {n} pooled rows")
    print(f"hg1s: {len(hg1s_train)} events, {len(hg1s_fc)} features")

    cluster_dfs = {inst: load_cluster_membership(inst) for inst in VARIANT_SPECS}

    # Feature columns per instrument for variant resolution
    fc_map = {
        "gc1s": precious_fc,
        "si1s": precious_fc,
        "pl1s": precious_fc,
        "hg1s": hg1s_fc,
    }

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    cpcv_path   = METALS_OUT / "cpcv_results.csv"
    locked_path = METALS_OUT / "locked_picks.csv"

    if cpcv_path.exists() and not force:
        print("\n[Phase 1] Loading cached CPCV results.")
        cpcv_df = pd.read_csv(cpcv_path)
    else:
        print("\n" + "=" * 64)
        print("Phase 1 — CPCV variant scoring (TRAIN only)")
        print("=" * 64)
        all_rows: list[dict] = []

        for inst, spec_dict in VARIANT_SPECS.items():
            cd = cluster_dfs[inst]
            for variant, spec in spec_dict.items():
                print(f"\n  [{inst}/{variant}] …")
                if inst in _POOLED:
                    path_aucs, oos_df = run_cpcv_pooled(
                        inst, variant, precious_train, spec, cd
                    )
                else:
                    path_aucs, oos_df = run_cpcv_individual(
                        inst, variant, hg1s_train, spec, cd
                    )
                s = summarise_cpcv(path_aucs, oos_df)
                row = {"inst": inst, "variant": variant, **s}
                all_rows.append(row)
                print(f"    AUC {s['auc_mean']:.4f} ± {s['auc_std']:.4f}"
                      f"  n_paths={s['n_paths']}")

        cpcv_df = pd.DataFrame(all_rows)
        cpcv_df.to_csv(cpcv_path, index=False)
        print(f"\nSaved: {cpcv_path}")

    # ── Lock ──────────────────────────────────────────────────────────────────
    locked_picks: dict[str, str] = {}
    lock_rows: list[dict] = []
    for inst in VARIANT_SPECS:
        inst_rows = cpcv_df[cpcv_df["inst"] == inst].to_dict("records")
        locked    = select_locked_variant(inst_rows, SIMPLICITY[inst])
        locked_picks[inst] = locked
        print(f"\n  [{inst}] LOCKED VARIANT: {locked}")
        for r in inst_rows:
            tag = " ← LOCKED" if r["variant"] == locked else ""
            print(f"    {r['variant']:8s}  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
                  f"  n_paths={r['n_paths']}{tag}")
        lock_rows.append({"inst": inst, "locked_variant": locked})

    lock_df = pd.DataFrame(lock_rows)
    lock_df.to_csv(locked_path, index=False)
    print(f"\nLocked picks saved: {locked_path}")

    for inst in VARIANT_SPECS:
        inst_rows = cpcv_df[cpcv_df["inst"] == inst].to_dict("records")
        cpcv_chart(
            inst_rows, locked_picks[inst], inst, SIMPLICITY[inst],
            METALS_OUT / f"{inst}_cpcv_chart.png",
        )

    if phase1_only:
        print("\nPhase 1 complete. --phase1-only set; stopping before test.")
        return

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    oos_path = METALS_OUT / "oos_results.csv"
    if oos_path.exists() and not force:
        print("\n[Phase 2] Loading cached OOS results.")
        oos_df = pd.read_csv(oos_path)
    else:
        print("\n" + "=" * 64)
        print("Phase 2 — Single-shot OOS (SEALED TEST — scored once)")
        print("=" * 64)
        print("Loading full data for test events …")
        data   = load_all_data()
        rl     = native_returns(data["ohlcv"], kind="log")
        w_rets = wide_returns(rl).sort_index()

        all_oos: list[dict] = []

        for inst, spec_dict in VARIANT_SPECS.items():
            cd = cluster_dfs[inst]
            print(f"\n  Loading test events for {inst} …")
            if inst in _POOLED:
                te   = _load_test_pooled(inst, data, w_rets, precious_fc)
                fc   = precious_fc
                ev_tr = precious_train
            else:
                fc   = hg1s_fc
                te   = _load_test_individual(inst, data, w_rets, fc)
                ev_tr = hg1s_train

            if te.empty:
                print(f"  WARNING: no test events for {inst}")
                continue
            y_te = te["bin"].to_numpy(dtype=int)
            if len(np.unique(y_te)) < 2:
                print(f"  WARNING: single-class test for {inst} — skipping")
                continue
            print(f"  {inst}: {len(te)} test events")

            X_tr_raw   = ev_tr[fc].fillna(0.0).to_numpy(dtype=np.float64)
            y_tr       = ev_tr["bin"].to_numpy(dtype=int)
            ev_tr_meta = ev_tr[["date", "t1", "bin", "instrument", "avg_uniqueness"]].reset_index(drop=True)
            X_te_raw   = te.reindex(columns=fc, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float64)

            for variant, spec in spec_dict.items():
                is_locked = (locked_picks.get(inst) == variant)
                tag = " [LOCKED]" if is_locked else " [diagnostic]"
                print(f"  [{inst}/{variant}]{tag} refitting …")
                res = fit_and_score(
                    inst, variant,
                    X_tr_raw, y_tr, ev_tr_meta,
                    X_te_raw, y_te,
                    fc, spec, cd,
                )
                row = {"inst": inst, "variant": variant, "is_locked": is_locked, **res}
                all_oos.append(row)
                print(f"    AUC {res['auc']:.4f}"
                      f"  [{res['auc_ci_lo']:.3f}, {res['auc_ci_hi']:.3f}]"
                      f"  logloss={res['logloss']:.4f}"
                      f"  brier={res['brier']:.4f}"
                      f"  n_test={res['n_test']}")

        oos_df = pd.DataFrame(all_oos)
        oos_df.to_csv(oos_path, index=False)
        print(f"\nSaved: {oos_path}")

    oos_summary_chart(
        oos_df.to_dict("records"), locked_picks, cpcv_df,
        METALS_OUT / "oos_summary_chart.png",
    )

    print("\n" + "=" * 64)
    print("Dev-CPCV vs OOS gap (locked variants)")
    print("=" * 64)
    for inst in VARIANT_SPECS:
        locked   = locked_picks.get(inst, "full")
        c_row    = cpcv_df[(cpcv_df["inst"] == inst) & (cpcv_df["variant"] == locked)]
        o_row    = oos_df[(oos_df["inst"] == inst) & (oos_df["variant"] == locked)]
        if c_row.empty or o_row.empty:
            continue
        dev      = float(c_row.iloc[0]["auc_mean"])
        oos_auc  = float(o_row.iloc[0]["auc"])
        ci_lo    = float(o_row.iloc[0]["auc_ci_lo"])
        ci_hi    = float(o_row.iloc[0]["auc_ci_hi"])
        print(f"  {inst:6s} ({locked:8s})  dev={dev:.4f}  OOS={oos_auc:.4f}"
              f"  [{ci_lo:.3f},{ci_hi:.3f}]  gap={dev - oos_auc:+.4f}")

    print("\nDone.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Metals variant comparison (new labels)")
    ap.add_argument("--force",       action="store_true", help="Rerun even if cached")
    ap.add_argument("--phase1-only", action="store_true", help="Stop before Phase 2")
    args = ap.parse_args()
    run(force=args.force, phase1_only=args.phase1_only)
    print("Done.")
