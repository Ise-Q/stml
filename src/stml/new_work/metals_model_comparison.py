"""metals_model_comparison.py — Variant comparison for metals instruments.

Phase 1: CPCV (n_groups=6, k=2, embargo=0.01) on TRAIN only.
Phase 2: Full-train refit, single-shot OOS on sealed test.

Instrument families (champion):
    gc1s  – XGB precious pool (NO SIGNAL, exploratory)
    si1s  – XGB individual   (NO SIGNAL, exploratory, ~random)
    pl1s  – Logistic individual (SIGNAL, marginal)
    hg1s  – RF individual       (SIGNAL, most robust)

Pooled mechanics (gc1s only):
    Each variant is a SEPARATE precious refit on gc1s's feature subset.
    Training uses all 882 precious-pool events; scoring uses gc1s slice only.

Variants:
    gc1s:   full | pruned (C2_f1 + C9_f11_lowfreq_macro + C3_f2)
                | reduced (PC1+PC2 of C2_f1, PC1+PC2 of C3_f2 + 3 raw)
    si1s:   full | pruned (C13_f1 + C2_f11 + C15_f2) | reduced (6 raw)
    pl1s:   full | pruned (C3_hmm_vol + C5_f11 + C4_f2)
                | reduced_regime (PC1+PC2 of C3_hmm_vol only)
                | reduced_plus   (PC1+PC2 of C3_hmm_vol + 2 raw
                                   + PC1+PC2 of C5_f11 + PC1+PC2 of C4_f2)
    hg1s:   full | pruned (C15_f2 + F5_signal + C1_f1) | reduced (5 raw)

Usage
-----
    python -m stml.new_work.metals_model_comparison
    python -m stml.new_work.metals_model_comparison --phase1-only
    python -m stml.new_work.metals_model_comparison --force

Outputs: outputs/model_comparison/metals/
    cpcv_results.csv
    locked_picks.csv
    oos_results.csv
    {inst}_cpcv_chart.png
    oos_summary_chart.png
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
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.new_work.cpcv_search import CombinatorialPurgedKFold
from stml.new_work.feature_importance import build_feature_matrix, load_all_data
from stml.new_work.model_comparison import (
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

# precious pool dummies: gc1s is reference (alphabetically first)
_PRECIOUS_DUMMY_VALUES: dict[str, dict[str, float]] = {
    "gc1s": {"inst_pl1s": 0.0, "inst_si1s": 0.0},
    "pl1s": {"inst_pl1s": 1.0, "inst_si1s": 0.0},
    "si1s": {"inst_pl1s": 0.0, "inst_si1s": 1.0},
}

# Simplicity order per instrument (index 0 = most regularised)
SIMPLICITY: dict[str, list[str]] = {
    "gc1s": ["reduced", "pruned", "full"],
    "si1s": ["reduced", "pruned", "full"],
    "pl1s": ["reduced_regime", "reduced_plus", "pruned", "full"],
    "hg1s": ["reduced", "pruned", "full"],
}

CACHE_DIR  = _HERE / "outputs" / "model_comparison" / "_cache"
METALS_OUT = _HERE / "outputs" / "model_comparison" / "metals"
IMP_DIR    = _HERE / "outputs" / "importance"


# ── Instrument configuration ───────────────────────────────────────────────────

INST_CONFIG: dict[str, dict] = {
    "gc1s": {"family": "xgb",     "pool": "precious"},
    "si1s": {"family": "xgb",     "pool": "individual"},
    "pl1s": {"family": "logistic", "pool": "individual"},
    "hg1s": {"family": "rf",       "pool": "individual"},
}

VARIANT_SPECS: dict[str, dict[str, dict]] = {
    "gc1s": {
        "full":   {"mode": "raw_all"},
        "pruned": {"mode": "raw_clusters",
                   "clusters": ["C2_f1", "C9_f11_lowfreq_macro", "C3_f2"]},
        "reduced": {
            "mode": "pca_plus_raw",
            "pca_specs": [
                {"cluster": "C2_f1", "n_components": 2},
                {"cluster": "C3_f2", "n_components": 2},
            ],
            "raw_features": ["f11_bund_10y_5d_change", "f11_vix_term_slope", "f7_volume_z_20"],
        },
    },
    "si1s": {
        "full":   {"mode": "raw_all"},
        "pruned": {"mode": "raw_clusters",
                   "clusters": ["C13_f1", "C2_f11", "C15_f2"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["f1_mr_score_20", "f11_move_vix_ratio", "f2_vol_60",
                         "f4_pc3", "f2_ret_skew_60", "f11_dxy_z"],
        },
    },
    "pl1s": {
        "full":   {"mode": "raw_all"},
        "pruned": {"mode": "raw_clusters",
                   "clusters": ["C3_hmm_vol", "C5_f11", "C4_f2"]},
        "reduced_regime": {
            "mode": "pca_plus_raw",
            "pca_specs": [{"cluster": "C3_hmm_vol", "n_components": 2}],
            "raw_features": [],
        },
        "reduced_plus": {
            "mode": "pca_plus_raw",
            "pca_specs": [
                {"cluster": "C3_hmm_vol", "n_components": 2},
                {"cluster": "C5_f11",     "n_components": 2},
                {"cluster": "C4_f2",      "n_components": 2},
            ],
            "raw_features": ["f5_sign_agree_mr", "f1_mr_score_10"],
        },
    },
    "hg1s": {
        "full":   {"mode": "raw_all"},
        "pruned": {"mode": "raw_clusters",
                   "clusters": ["C15_f2", "F5_signal", "C1_f1"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["f5_signal", "f2_ret_kurt_60", "f1_mr_score_40",
                         "f2_vol_ratio_20_60", "f2_vol_of_vol_20"],
        },
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


def load_cluster_membership(inst: str) -> pd.DataFrame:
    p = IMP_DIR / inst / "cluster_membership.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["cluster", "feature"])


# ── Variant transform ──────────────────────────────────────────────────────────

def transform_variant(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    feat_cols: list[str],
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_tr_v, X_te_v, feature_names). PCA fitted on X_tr only."""
    mode = spec["mode"]

    if mode == "raw_all":
        return X_tr, X_te, list(feat_cols)

    if mode == "raw_clusters":
        members = cluster_df[cluster_df["cluster"].isin(spec["clusters"])]["feature"].tolist()
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
            cname  = ps["cluster"]
            n_comp = ps["n_components"]
            members = cluster_df[cluster_df["cluster"] == cname]["feature"].tolist()
            members = [f for f in members if f in feat_cols]
            if not members:
                continue
            idx = [feat_cols.index(f) for f in members]
            Xc_tr = X_tr[:, idx]
            Xc_te = X_te[:, idx]
            n_act = min(n_comp, len(members), Xc_tr.shape[0])
            sc  = StandardScaler()
            pca = PCA(n_components=n_act, random_state=SEED)
            pca.fit(sc.fit_transform(Xc_tr))
            parts_tr.append(pca.transform(sc.transform(Xc_tr)))
            parts_te.append(pca.transform(sc.transform(Xc_te)))
            names.extend([f"{cname}_PC{i+1}" for i in range(n_act)])

        raw = [f for f in spec.get("raw_features", []) if f in feat_cols]
        if raw:
            raw_idx = [feat_cols.index(f) for f in raw]
            parts_tr.append(X_tr[:, raw_idx])
            parts_te.append(X_te[:, raw_idx])
            names.extend(raw)

        if not parts_tr:
            return X_tr, X_te, list(feat_cols)
        return np.hstack(parts_tr), np.hstack(parts_te), names

    raise ValueError(f"Unknown variant mode: {mode!r}")


# ── Individual CPCV (si1s, pl1s, hg1s) ───────────────────────────────────────

def run_cpcv_individual(
    inst: str,
    variant: str,
    events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float]]:
    fc = _feat_cols(events)
    X_all = events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_all = events["bin"].to_numpy(dtype=int)
    ev_meta = events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    family = INST_CONFIG[inst]["family"]

    cpcv = CombinatorialPurgedKFold(n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO)
    oos_rows: list[dict] = []
    path_aucs: list[float] = []

    for path_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        X_tr_raw, X_te_raw = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        ev_tr = ev_meta.iloc[tr_idx].reset_index(drop=True)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        try:
            X_tr, X_te, _ = transform_variant(X_tr_raw, X_te_raw, fc, spec, cluster_df)
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
            continue

        auc = _safe_auc(y_te, prob)
        if auc >= 0:
            path_aucs.append(auc)
        ev_te = ev_meta.iloc[te_idx].reset_index(drop=True)
        for i in range(len(te_idx)):
            oos_rows.append({"date": ev_te.iloc[i]["date"], "y_true": int(y_te[i]),
                             "y_score": float(prob[i]), "path": path_i})

    return pd.DataFrame(oos_rows), path_aucs


# ── Pooled CPCV (gc1s via precious pool) ─────────────────────────────────────

def run_cpcv_pooled(
    target_inst: str,
    variant: str,
    precious_events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float]]:
    """Train on full precious pool; score on target_inst slice only."""
    fc = _feat_cols(precious_events)
    X_all = precious_events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_all = precious_events["bin"].to_numpy(dtype=int)
    ev_meta = precious_events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    inst_arr = precious_events["instrument"].values
    family = INST_CONFIG[target_inst]["family"]

    cpcv = CombinatorialPurgedKFold(n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO)
    oos_rows: list[dict] = []
    path_aucs: list[float] = []
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
        X_tr_raw = X_all[tr_idx]
        y_tr     = y_all[tr_idx]
        X_te_raw = X_all[te_pos]
        ev_tr    = ev_meta.iloc[tr_idx].reset_index(drop=True)
        if len(np.unique(y_tr)) < 2:
            n_skipped += 1
            continue
        try:
            X_tr, X_te, _ = transform_variant(X_tr_raw, X_te_raw, fc, spec, cluster_df)
            if family == "xgb":
                model, _ = _tune_fit_xgb(X_tr, y_tr, ev_tr)
                prob = model.predict_proba(X_te)[:, 1]
            else:
                raise ValueError(f"Unexpected family for {target_inst}: {family}")
        except Exception as e:
            print(f"    [{target_inst}/{variant}] path {path_i} failed: {e}")
            n_skipped += 1
            continue

        auc = _safe_auc(y_te, prob)
        if auc >= 0:
            path_aucs.append(auc)
        ev_te = ev_meta.iloc[te_pos].reset_index(drop=True)
        for i in range(len(te_pos)):
            oos_rows.append({"date": ev_te.iloc[i]["date"], "y_true": int(y_te[i]),
                             "y_score": float(prob[i]), "path": path_i})

    if n_skipped > 0:
        print(f"    [{target_inst}/{variant}] skipped {n_skipped}/15 paths (thin/single-class)")
    return pd.DataFrame(oos_rows), path_aucs


# ── Summary + lock ────────────────────────────────────────────────────────────

def summarise_cpcv(path_aucs: list[float], oos_df: pd.DataFrame) -> dict[str, Any]:
    if not path_aucs or oos_df.empty:
        return {"auc_mean": np.nan, "auc_std": np.nan, "auc_se": np.nan,
                "logloss": np.nan, "brier": np.nan, "n_paths": 0}
    arr = np.array(path_aucs)
    ll = br = np.nan
    if oos_df["y_true"].nunique() >= 2:
        try:
            ll = log_loss(oos_df["y_true"], oos_df["y_score"])
            br = brier_score_loss(oos_df["y_true"], oos_df["y_score"])
        except Exception:
            pass
    return {"auc_mean": float(arr.mean()), "auc_std": float(arr.std()),
            "auc_se": float(arr.std() / np.sqrt(len(arr))),
            "logloss": float(ll) if not np.isnan(ll) else np.nan,
            "brier":   float(br) if not np.isnan(br) else np.nan,
            "n_paths": len(arr)}


def select_locked_variant(cpcv_rows: list[dict], simplicity: list[str]) -> str:
    by_v = {r["variant"]: r for r in cpcv_rows}
    full = by_v.get("full", {})
    fm, fse = full.get("auc_mean", np.nan), full.get("auc_se", np.nan)
    if np.isnan(fm) or np.isnan(fse):
        return "full"
    threshold = fm - fse
    for v in simplicity:
        row = by_v.get(v, {})
        vm = row.get("auc_mean", np.nan)
        if not np.isnan(vm) and vm >= threshold:
            return v
    return "full"


# ── Bootstrap AUC CI ──────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                     n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
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


# ── Phase 2 helpers ────────────────────────────────────────────────────────────

def _load_test_individual(inst: str, data: dict, w_rets: pd.DataFrame,
                          train_fc: list[str]) -> pd.DataFrame:
    _, full_ev = build_feature_matrix(inst, data, w_rets)
    te = split_config.apply_test_mask(full_ev)
    if te.empty:
        return pd.DataFrame()
    return te.reindex(columns=list(_META) + train_fc, fill_value=0.0)


def _load_test_gc1s(data: dict, w_rets: pd.DataFrame,
                    precious_fc: list[str]) -> pd.DataFrame:
    """Load gc1s sealed test events; add precious pool dummies (gc1s = reference)."""
    _, full_ev = build_feature_matrix("gc1s", data, w_rets)
    te = split_config.apply_test_mask(full_ev)
    if te.empty:
        return pd.DataFrame()
    for col, val in _PRECIOUS_DUMMY_VALUES["gc1s"].items():
        te[col] = val
    return te.reindex(columns=list(_META) + precious_fc, fill_value=0.0)


def _fit_and_score(
    inst: str, variant: str,
    X_tr_raw: np.ndarray, y_tr: np.ndarray, ev_tr: pd.DataFrame,
    X_te_raw: np.ndarray, y_te: np.ndarray,
    fc: list[str], spec: dict, cluster_df: pd.DataFrame, family: str,
) -> dict[str, Any]:
    _nan = {"auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan,
            "logloss": np.nan, "brier": np.nan, "n_test": len(y_te)}
    try:
        X_tr, X_te, _ = transform_variant(X_tr_raw, X_te_raw, fc, spec, cluster_df)
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
    return {"auc": float(auc) if auc >= 0 else np.nan,
            "auc_ci_lo": ci_lo, "auc_ci_hi": ci_hi,
            "logloss": float(ll) if not np.isnan(ll) else np.nan,
            "brier":   float(br) if not np.isnan(br) else np.nan,
            "n_test":  int(len(y_te))}


# ── Charts ─────────────────────────────────────────────────────────────────────

def _cpcv_chart(cpcv_rows: list[dict], locked: str, inst: str,
                simplicity: list[str], out_path: Path) -> None:
    rows = sorted(cpcv_rows, key=lambda r: simplicity.index(r["variant"])
                  if r["variant"] in simplicity else 99)
    variants = [r["variant"] for r in rows]
    means    = [r["auc_mean"] for r in rows]
    stds     = [r.get("auc_std", 0) for r in rows]
    colors   = ["#2196F3" if v == locked else "#90CAF9" for v in variants]
    valid    = [m for m in means if not np.isnan(m)]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(max(5, len(variants) * 1.5), 4))
    ax.bar(variants, means, yerr=stds, color=colors,
           error_kw={"ecolor": "grey", "capsize": 4}, width=0.5)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    lo = max(0.2, min(valid) - 3 * max((s or 0) for s in stds))
    hi = min(1.0, max(valid) + 3 * max((s or 0) for s in stds))
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Variant")
    ax.set_ylabel("AUC (mean ± std)")
    ax.set_title(f"{inst.upper()} – Phase 1 CPCV  |  locked = {locked}")
    for bar, m in zip(ax.patches, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _oos_summary_chart(oos_rows: list[dict], locked_picks: dict[str, str],
                       out_path: Path) -> None:
    headline = [r for r in oos_rows if r["variant"] == locked_picks.get(r["inst"])]
    if not headline:
        return
    insts = [r["inst"] for r in headline]
    aucs  = [r["auc"]  for r in headline]
    ci_lo = [r["auc"] - r["auc_ci_lo"] for r in headline]
    ci_hi = [r["auc_ci_hi"] - r["auc"] for r in headline]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(insts, aucs, yerr=[ci_lo, ci_hi], color="#2196F3",
           error_kw={"ecolor": "grey", "capsize": 4}, width=0.4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("OOS AUC (95% bootstrap CI)")
    ax.set_title("Metals — Phase 2 OOS (locked variants)")
    for i, (inst, auc) in enumerate(zip(insts, aucs)):
        if not np.isnan(auc):
            ax.text(i, auc + ci_hi[i] + 0.005, f"{auc:.3f}",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(force: bool = False, phase1_only: bool = False) -> None:
    METALS_OUT.mkdir(parents=True, exist_ok=True)

    # ── Load caches ───────────────────────────────────────────────────────────
    precious_train = pd.read_parquet(CACHE_DIR / "precious_events.parquet")
    si1s_train     = pd.read_parquet(CACHE_DIR / "si1s_events.parquet")
    pl1s_train     = pd.read_parquet(CACHE_DIR / "pl1s_events.parquet")
    hg1s_train     = pd.read_parquet(CACHE_DIR / "hg1s_events.parquet")

    precious_fc = _feat_cols(precious_train)
    print(f"  precious: {len(precious_train)} events, {len(precious_fc)} features")
    for inst in ["gc1s", "si1s", "pl1s"]:
        n = (precious_train["instrument"] == inst).sum()
        print(f"    {inst}: {n} in pool")
    for inst, ev in [("si1s", si1s_train), ("pl1s", pl1s_train), ("hg1s", hg1s_train)]:
        print(f"  {inst}: {len(ev)} events, {len(_feat_cols(ev))} features")

    train_cache = {
        "gc1s": precious_train,
        "si1s": si1s_train,
        "pl1s": pl1s_train,
        "hg1s": hg1s_train,
    }

    cluster_dfs = {inst: load_cluster_membership(inst) for inst in INST_CONFIG}

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    cpcv_path   = METALS_OUT / "cpcv_results.csv"
    locked_path = METALS_OUT / "locked_picks.csv"

    if cpcv_path.exists() and not force:
        print("\n[Phase 1] Loading cached CPCV results.")
        cpcv_df = pd.read_csv(cpcv_path)
    else:
        print("\n" + "=" * 60)
        print("Phase 1 — CPCV variant scoring (TRAIN only)")
        print("=" * 60)
        all_rows: list[dict] = []

        for inst, spec_dict in VARIANT_SPECS.items():
            cd   = cluster_dfs[inst]
            pool = INST_CONFIG[inst]["pool"]

            for variant, spec in spec_dict.items():
                print(f"\n  [{inst}/{variant}] CPCV …")
                if pool == "precious":
                    oos_df, path_aucs = run_cpcv_pooled(
                        inst, variant, precious_train, spec, cd)
                else:
                    oos_df, path_aucs = run_cpcv_individual(
                        inst, variant, train_cache[inst], spec, cd)
                s = summarise_cpcv(path_aucs, oos_df)
                row = {"inst": inst, "variant": variant, **s}
                all_rows.append(row)
                print(f"    AUC {s['auc_mean']:.4f} ± {s['auc_std']:.4f}"
                      f"  SE={s['auc_se']:.4f}  n_paths={s['n_paths']}")

        cpcv_df = pd.DataFrame(all_rows)
        cpcv_df.to_csv(cpcv_path, index=False)
        print(f"\n  Saved: {cpcv_path}")

    # ── Lock ──────────────────────────────────────────────────────────────────
    locked_picks: dict[str, str] = {}
    lock_rows: list[dict] = []
    for inst in INST_CONFIG:
        inst_rows = cpcv_df[cpcv_df["inst"] == inst].to_dict("records")
        locked = select_locked_variant(inst_rows, SIMPLICITY[inst])
        locked_picks[inst] = locked
        print(f"\n  [{inst}] LOCKED VARIANT: {locked}")
        for r in inst_rows:
            print(f"    {r['variant']:16s}  AUC {r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
                  f"  n_paths={r['n_paths']}")
        lock_rows.append({"inst": inst, "locked_variant": locked})

    lock_df = pd.DataFrame(lock_rows)
    lock_df.to_csv(locked_path, index=False)
    print(f"\n  Locked picks saved: {locked_path}")

    for inst in INST_CONFIG:
        inst_rows = cpcv_df[cpcv_df["inst"] == inst].to_dict("records")
        _cpcv_chart(inst_rows, locked_picks[inst], inst, SIMPLICITY[inst],
                    METALS_OUT / f"{inst}_cpcv_chart.png")

    if phase1_only:
        print("\nPhase 1 complete. --phase1-only set; stopping before test.")
        return

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    oos_path = METALS_OUT / "oos_results.csv"
    if oos_path.exists() and not force:
        print("\n[Phase 2] Loading cached OOS results.")
        oos_df = pd.read_csv(oos_path)
    else:
        print("\n" + "=" * 60)
        print("Phase 2 — Single-shot OOS (SEALED TEST)")
        print("=" * 60)
        print("  Loading full data for test events …")
        data   = load_all_data()
        rl     = native_returns(data["ohlcv"], kind="log")
        w_rets = wide_returns(rl).sort_index()

        all_oos: list[dict] = []

        for inst, spec_dict in VARIANT_SPECS.items():
            pool = INST_CONFIG[inst]["pool"]
            cd   = cluster_dfs[inst]
            family = INST_CONFIG[inst]["family"]

            print(f"\n  Loading test events for {inst} …")
            if pool == "precious":
                te = _load_test_gc1s(data, w_rets, precious_fc)
                ev_tr = precious_train
                fc    = precious_fc
            else:
                fc = _feat_cols(train_cache[inst])
                te = _load_test_individual(inst, data, w_rets, fc)
                ev_tr = train_cache[inst]

            if te.empty:
                print(f"  WARNING: no test events for {inst}")
                continue
            print(f"  {inst}: {len(te)} test events")

            X_tr_raw = ev_tr[fc].fillna(0.0).to_numpy(dtype=np.float64)
            y_tr     = ev_tr["bin"].to_numpy(dtype=int)
            ev_tr_meta = ev_tr[["date","t1","bin","instrument","avg_uniqueness"]].reset_index(drop=True)
            X_te_raw = te.reindex(columns=fc, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float64)
            y_te     = te["bin"].to_numpy(dtype=int)

            for variant, spec in spec_dict.items():
                is_locked = (locked_picks.get(inst) == variant)
                tag = " [LOCKED]" if is_locked else " [diagnostic]"
                print(f"  [{inst}/{variant}]{tag} refitting …")
                res = _fit_and_score(inst, variant, X_tr_raw, y_tr, ev_tr_meta,
                                     X_te_raw, y_te, fc, spec, cd, family)
                row = {"inst": inst, "variant": variant, "is_locked": is_locked, **res}
                all_oos.append(row)
                print(f"    AUC {res['auc']:.4f}"
                      f"  [{res['auc_ci_lo']:.3f}, {res['auc_ci_hi']:.3f}]"
                      f"  logloss={res['logloss']:.4f}"
                      f"  brier={res['brier']:.4f}"
                      f"  n_test={res['n_test']}")

        oos_df = pd.DataFrame(all_oos)
        oos_df.to_csv(oos_path, index=False)
        print(f"\n  Saved: {oos_path}")

    _oos_summary_chart(oos_df.to_dict("records"), locked_picks,
                       METALS_OUT / "oos_summary_chart.png")

    print("\n" + "=" * 60)
    print("Dev-CPCV vs OOS gap (locked variants)")
    print("=" * 60)
    for inst in INST_CONFIG:
        locked = locked_picks.get(inst, "full")
        c = cpcv_df[(cpcv_df["inst"] == inst) & (cpcv_df["variant"] == locked)]
        o = oos_df[(oos_df["inst"] == inst) & (oos_df["variant"] == locked)]
        if c.empty or o.empty:
            continue
        dev, oos_auc = c.iloc[0]["auc_mean"], o.iloc[0]["auc"]
        print(f"  {inst:6s} ({locked:16s})  dev={dev:.4f}  OOS={oos_auc:.4f}"
              f"  gap={dev - oos_auc:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",       action="store_true")
    ap.add_argument("--phase1-only", action="store_true")
    args = ap.parse_args()
    run(force=args.force, phase1_only=args.phase1_only)
