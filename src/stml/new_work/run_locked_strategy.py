"""run_locked_strategy.py — OOS strategy backtest using locked meta-models.

Strategy track: refit each instrument's locked champion on TRAIN events,
apply Platt calibration (CPCV OOF), predict on sealed TEST events, size
positions via fractional Kelly + vol-target, and run net-of-cost backtest
against a primary-blind baseline.

Locked champions (locked_picks.csv + equity/energy/metals INST_FAMILY):
  nq1s   xgb  / reduced     (equity, individual)
  fesx1s log  / reduced     (equity, individual)
  es1s   rf   / pruned      (equity, individual)
  cl1s   log  / reduced     (energy, individual)
  ho1s   mlp  / reduced     (energy, energy_cl_ho pool)
  rb1s   log  / reduced_min (energy, individual)
  ng1s   mlp  / reduced     (energy, energy_all pool)
  gc1s   rf   / reduced     (metals, precious pool)
  si1s   rf   / reduced     (metals, precious pool)
  pl1s   rf   / reduced     (metals, precious pool)
  hg1s   rf   / reduced     (metals, individual)

Outputs:
  outputs/strategy_weights.csv
  outputs/metamodel_predictions.csv
  results/sreeram_experimental/strategy_daily_net_returns.csv
  results/sreeram_experimental/oos_events_with_predictions.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
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
from stml.new_work.energy_model_comparison import (
    _fit_mlp_fixed,
    CPCV_N_GROUPS,
    CPCV_K,
    CPCV_EMBARGO,
)
from stml.new_work import split_config
from stml.na_checks import native_returns, wide_returns
from stml.experimental.calibration import PlattCalibrator
from stml.experimental.sizing import position_weight
from stml.experimental.volatility import ewma_daily_sigma
from stml.experimental.backtest import barrier_backtest, performance_metrics

# ── Paths ─────────────────────────────────────────────────────────────────────

CACHE_DIR  = _HERE / "outputs" / "model_comparison" / "_cache"
IMP_DIR    = _HERE / "outputs" / "importance"
STRAT_DIR  = _REPO / "results" / "sreeram_experimental"
OUT_DIR    = _HERE / "outputs"

SEED = 42

# ── Locked champion config ────────────────────────────────────────────────────

_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})

# pool → cache parquet name
_POOL_CACHE: dict[str, str] = {
    "energy_cl_ho": "energy_cl_ho_events.parquet",
    "energy_all":   "energy_all_events.parquet",
    "precious":     "precious_events.parquet",
}

# Instrument dummy values (for test-event column alignment with pool features)
_CL_HO_DUMMIES: dict[str, dict[str, float]] = {
    "cl1s": {"inst_ho1s": 0.0},
    "ho1s": {"inst_ho1s": 1.0},
}
_EA_DUMMIES: dict[str, dict[str, float]] = {
    "cl1s": {"inst_ho1s": 0.0, "inst_ng1s": 0.0, "inst_rb1s": 0.0},
    "ho1s": {"inst_ho1s": 1.0, "inst_ng1s": 0.0, "inst_rb1s": 0.0},
    "rb1s": {"inst_ho1s": 0.0, "inst_ng1s": 0.0, "inst_rb1s": 1.0},
    "ng1s": {"inst_ho1s": 0.0, "inst_ng1s": 1.0, "inst_rb1s": 0.0},
}
_PRECIOUS_DUMMIES: dict[str, dict[str, float]] = {
    "gc1s": {"inst_pl1s": 0.0, "inst_si1s": 0.0},
    "si1s": {"inst_pl1s": 0.0, "inst_si1s": 1.0},
    "pl1s": {"inst_pl1s": 1.0, "inst_si1s": 0.0},
}

CHAMPIONS: dict[str, dict[str, Any]] = {
    "nq1s":   {"family": "xgb",      "pool": "individual", "variant": "reduced"},
    "fesx1s": {"family": "logistic",  "pool": "individual", "variant": "reduced"},
    "es1s":   {"family": "rf",        "pool": "individual", "variant": "pruned"},
    "cl1s":   {"family": "logistic",  "pool": "individual", "variant": "reduced"},
    "ho1s":   {"family": "mlp",       "pool": "energy_cl_ho", "variant": "reduced"},
    "rb1s":   {"family": "logistic",  "pool": "individual", "variant": "reduced_min"},
    "ng1s":   {"family": "mlp",       "pool": "energy_all",   "variant": "reduced"},
    "gc1s":   {"family": "rf",        "pool": "precious",     "variant": "reduced"},
    "si1s":   {"family": "rf",        "pool": "precious",     "variant": "reduced"},
    "pl1s":   {"family": "rf",        "pool": "precious",     "variant": "reduced"},
    "hg1s":   {"family": "rf",        "pool": "individual", "variant": "reduced"},
}

# Variant feature specifications — unified across all comparison scripts
VARIANT_SPECS: dict[str, dict[str, dict]] = {
    # ── equity ──────────────────────────────────────────────────────────────
    "nq1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["F5_signal", "C4_f11"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["f5_signal", "f11_dist_stock_surprise",
                         "f11_copper_stock_z", "f11_vix_5d_change"],
        },
    },
    "fesx1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["F5_signal", "C10_f12", "C13_f7"]},
        "reduced": {
            "mode": "pca_plus_raw",
            "pca_specs": [{"cluster": "F5_signal", "n_components": 4}],
            "raw_features": ["f12_mra_energy_D1", "f7_oi_z_20",
                             "f1_ret_reversal_40", "f1_dist_ma_sigma_10"],
        },
    },
    "es1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["F5_signal", "C14_f11", "C2_f2"]},
        "reduced": {
            "mode": "pca_plus_raw",
            "pca_specs": [
                {"cluster": "C14_f11", "n_components": 2},
                {"cluster": "C2_f2",   "n_components": 2},
            ],
            "raw_features": ["f5_trailing_run_length", "f7_oi_z_20"],
        },
    },
    # ── energy ──────────────────────────────────────────────────────────────
    "cl1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["C2_f11_lowfreq_macro", "C14_f12"]},
        "reduced": {
            "mode": "pca_plus_raw",
            "pca_specs": [
                {"cluster": "C2_f11_lowfreq_macro", "n_components": 1},
                {"cluster": "C14_f12",               "n_components": 1},
            ],
            "raw_features": ["f11_crude_stock_surprise", "f2_vol_10", "f11_china_pmi_level"],
        },
    },
    "ho1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["C8_f11_lowfreq_macro", "C15_f11", "C1_f1"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["f11_vix_term_slope", "f4_pc3", "f1_bb_bandwidth_20", "f11_vix_5d_change"],
        },
    },
    "rb1s": {
        "full":        {"mode": "raw_all"},
        "pruned":      {"mode": "raw_clusters", "clusters": ["F5_signal", "C6_f11_lowfreq_macro", "C1_f1"]},
        "reduced":     {
            "mode": "raw_select",
            "features": ["f5_signal", "f11_hy_oas_5d_change",
                         "f11_dist_stock_surprise", "f2_vol_ratio_20_60"],
        },
        "reduced_min": {
            "mode": "raw_select",
            "features": ["f5_signal", "f11_hy_oas_5d_change"],
        },
    },
    "ng1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["C15_f7"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["f7_oi_change", "f7_oi_level", "f2_atr_14",
                         "hmm_vol_next_turbulent", "hmm_vol_p2_turbulent", "hmm_vol_p0_calm"],
        },
    },
    # ── metals ──────────────────────────────────────────────────────────────
    "gc1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["C5_hmm_vol", "C10_f11_lowfreq_macro", "C2_f1"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["hmm_vol_p0_calm", "hmm_vol_p2_turbulent", "hmm_vol_next_turbulent",
                         "f11_ust_10y_5d_change", "f11_bund_10y_5d_change", "f11_real_yield_10y",
                         "f6_ts_momentum_20"],
        },
    },
    "si1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["C5_hmm_vol", "C15_f11_lowfreq_macro", "C13_f11_lowfreq_macro"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["hmm_vol_p0_calm", "hmm_vol_p2_turbulent", "hmm_vol_next_turbulent",
                         "f2_atr_14", "f11_ust_10y_5d_change", "f11_real_yield_10y"],
        },
    },
    "pl1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["C5_hmm_vol", "C11_f11"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["hmm_vol_p0_calm", "hmm_vol_p2_turbulent", "hmm_vol_next_turbulent",
                         "f2_atr_14", "f6_adx_14", "f6_macd_hist_12_26_9", "f11_ust_10y_5d_change"],
        },
    },
    "hg1s": {
        "full":    {"mode": "raw_all"},
        "pruned":  {"mode": "raw_clusters", "clusters": ["C1_f1", "C12_f11"]},
        "reduced": {
            "mode": "raw_select",
            "features": ["f1_mr_score_40", "f1_hilo_pos_10", "f1_dist_ma_sigma_20",
                         "f11_ust_bund_spread", "f1_rsi_14", "f11_dxy_z", "f12_mra_energy_D4"],
        },
    },
}


# ── Variant transform (unified — handles all modes) ───────────────────────────

def transform_variant(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    feat_cols: list[str],
    spec: dict,
    cluster_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_tr_v, X_te_v, names). PCA always fitted on X_tr rows only."""
    mode = spec["mode"]

    if mode == "raw_all":
        return X_tr, X_te, list(feat_cols)

    if mode == "raw_clusters":
        members = cluster_df[
            cluster_df["cluster"].isin(spec["clusters"])
        ]["feature"].tolist()
        keep = [f for f in members if f in feat_cols]
        if not keep:
            return X_tr, X_te, list(feat_cols)
        idx = [feat_cols.index(f) for f in keep]
        return X_tr[:, idx], X_te[:, idx], keep

    if mode == "raw_select":
        keep = [f for f in spec["features"] if f in feat_cols]
        idx  = [feat_cols.index(f) for f in keep]
        if not idx:
            return X_tr, X_te, list(feat_cols)
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
            n_act  = min(n_comp, len(members), Xc_tr.shape[0])
            sc  = StandardScaler()
            pca = PCA(n_components=n_act, random_state=SEED)
            pca.fit(sc.fit_transform(Xc_tr))
            parts_tr.append(pca.transform(sc.transform(Xc_tr)))
            parts_te.append(pca.transform(sc.transform(Xc_te)))
            names.extend([f"{cname}_PC{i + 1}" for i in range(n_act)])
        raw = [f for f in spec.get("raw_features", []) if f in feat_cols]
        raw_idx = [feat_cols.index(f) for f in raw]
        parts_tr.append(X_tr[:, raw_idx])
        parts_te.append(X_te[:, raw_idx])
        names.extend(raw)
        X_tr_v = np.hstack(parts_tr) if parts_tr else X_tr[:, []]
        X_te_v = np.hstack(parts_te) if parts_te else X_te[:, []]
        return X_tr_v, X_te_v, names

    raise ValueError(f"Unknown variant mode: {mode!r}")


# ── Cluster membership loader ─────────────────────────────────────────────────

def load_cluster_membership(inst: str) -> pd.DataFrame:
    p = IMP_DIR / inst / "cluster_membership.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["cluster", "feature"])


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


# ── Fit model by family ───────────────────────────────────────────────────────

def fit_model(
    family: str,
    pool: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    ev_tr: pd.DataFrame,
):
    """Return (scaler_or_None, model) pair."""
    if family == "logistic":
        scaler, model, _ = _tune_fit_logistic(X_tr, y_tr, ev_tr)
        return scaler, model
    if family == "rf":
        model, _ = _tune_fit_rf(X_tr, y_tr, ev_tr)
        return None, model
    if family == "xgb":
        model, _ = _tune_fit_xgb(X_tr, y_tr, ev_tr)
        return None, model
    if family == "mlp":
        scaler, model = _fit_mlp_fixed(X_tr, y_tr, pool)
        return scaler, model
    raise ValueError(f"Unknown family: {family!r}")


def predict_proba(scaler, model, X_te: np.ndarray) -> np.ndarray:
    if scaler is not None:
        X_te = scaler.transform(X_te)
    return model.predict_proba(X_te)[:, 1]


# ── CPCV OOF for Platt calibration ───────────────────────────────────────────

def run_cpcv_oof(
    inst: str,
    train_events: pd.DataFrame,
    spec: dict,
    cluster_df: pd.DataFrame,
    family: str,
    pool: str,
    target_inst: str | None = None,
) -> pd.DataFrame:
    """Run CPCV(6,2) on train_events; return OOF predictions for target_inst rows.

    For individual instruments: target_inst = None (all rows scored).
    For pooled instruments: target_inst = the instrument whose rows to keep.
    """
    fc      = _feat_cols(train_events)
    X_all   = train_events[fc].fillna(0.0).to_numpy(dtype=np.float64)
    y_all   = train_events["bin"].to_numpy(dtype=int)
    ev_meta = train_events[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy()
    inst_arr = train_events["instrument"].values if target_inst else None

    cpcv = CombinatorialPurgedKFold(
        n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO
    )
    oof_rows: list[dict] = []

    for path_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        if target_inst is not None:
            te_mask = inst_arr[te_idx] == target_inst
            te_pos  = te_idx[te_mask]
        else:
            te_pos = te_idx

        if len(te_pos) < 2:
            continue

        y_te = y_all[te_pos]
        y_tr = y_all[tr_idx]
        ev_tr_fold = ev_meta.iloc[tr_idx].reset_index(drop=True)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        try:
            X_tr_v, X_te_v, _ = transform_variant(
                X_all[tr_idx], X_all[te_pos], fc, spec, cluster_df
            )
            scaler, model = fit_model(family, pool, X_tr_v, y_tr, ev_tr_fold)
            prob = predict_proba(scaler, model, X_te_v)
        except Exception as e:
            print(f"    [{inst}/path {path_i}] CPCV OOF failed: {e}")
            continue

        ev_te_fold = ev_meta.iloc[te_pos].reset_index(drop=True)
        for i in range(len(te_pos)):
            oof_rows.append({
                "date":    ev_te_fold.iloc[i]["date"],
                "y_true":  int(y_te[i]),
                "y_score": float(prob[i]),
            })

    return pd.DataFrame(oof_rows)


# ── Train-pool loading for pool instruments ───────────────────────────────────

def load_pool_train(pool: str) -> pd.DataFrame:
    cache_file = _POOL_CACHE[pool]
    path = CACHE_DIR / cache_file
    if not path.exists():
        raise FileNotFoundError(f"Pool cache missing: {path}")
    return pd.read_parquet(path)


# ── Test events loading per instrument ────────────────────────────────────────

def load_test_events_for_inst(
    inst: str,
    pool: str,
    pool_fc: list[str] | None,
    data: dict,
    w_rets: pd.DataFrame,
) -> pd.DataFrame:
    """Load sealed test events for one instrument, aligned to pool feature columns."""
    _, full_ev = build_feature_matrix(inst, data, w_rets)
    te = split_config.apply_test_mask(full_ev)
    if te.empty:
        return pd.DataFrame()

    if pool == "individual":
        return te.reset_index(drop=True)

    # Pool instrument: add dummy columns then reindex to pool_fc order
    if pool == "energy_cl_ho":
        dummy_map = _CL_HO_DUMMIES[inst]
    elif pool == "energy_all":
        dummy_map = _EA_DUMMIES[inst]
    elif pool == "precious":
        dummy_map = _PRECIOUS_DUMMIES[inst]
    else:
        raise ValueError(f"Unknown pool: {pool!r}")

    for col, val in dummy_map.items():
        te[col] = val

    return te.reindex(columns=list(_META) + pool_fc, fill_value=0.0).reset_index(drop=True)


# ── EWMA volatility lookup ────────────────────────────────────────────────────

def build_vol_lookup(
    inst: str,
    ohlcv_df: pd.DataFrame,
    span: int = 100,
) -> pd.Series:
    """Return daily EWMA σ (un-annualised) indexed by date for one instrument.

    ohlcv_df is the long-format DataFrame from load_all_data()['ohlcv'].
    """
    sub = ohlcv_df[ohlcv_df["instrument"] == inst].sort_values("date")
    if sub.empty:
        raise KeyError(f"No OHLCV for {inst}")
    close = sub.set_index("date")["close"].astype(float)
    return ewma_daily_sigma(close, span=span, min_periods=20)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    STRAT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print locked champion table
    print("=" * 80)
    print("LOCKED CHAMPION TABLE")
    print("=" * 80)
    print(f"{'Inst':8s} {'Family':10s} {'Pool':16s} {'Variant':12s}")
    print("-" * 48)
    for inst, cfg in CHAMPIONS.items():
        print(f"{inst:8s} {cfg['family']:10s} {cfg['pool']:16s} {cfg['variant']:12s}")
    print()

    # Load all raw data (for test events and returns)
    print("Loading all raw data …")
    data = load_all_data()
    rets_long = native_returns(data["ohlcv"], kind="log")
    w_rets = wide_returns(rets_long).sort_index()

    # Pre-build EWMA vol per instrument (on ALL history, causal)
    print("Building EWMA volatility …")
    vol_lookup: dict[str, pd.Series] = {}
    ohlcv_df = data["ohlcv"]
    for inst in CHAMPIONS:
        try:
            vol_lookup[inst] = build_vol_lookup(inst, ohlcv_df)
        except KeyError:
            print(f"  WARNING: no OHLCV for {inst}, vol will be NaN")
            vol_lookup[inst] = pd.Series(dtype=float)

    # Collect per-instrument: train events, pool events, test events, cluster_df
    print("\nLoading train/pool event caches …")
    train_events: dict[str, pd.DataFrame] = {}      # individual or pool events
    individual_events: dict[str, pd.DataFrame] = {} # individual instrument train events only

    loaded_pools: dict[str, pd.DataFrame] = {}      # pool → pool_events (shared)

    for inst in CHAMPIONS:
        cfg  = CHAMPIONS[inst]
        pool = cfg["pool"]

        # Individual parquets: cl1s, rb1s, hg1s, nq1s, fesx1s, es1s, gc1s, si1s, pl1s
        # ho1s and ng1s are pool-only instruments — no individual parquet in cache
        inst_cache = CACHE_DIR / f"{inst}_events.parquet"
        if inst_cache.exists():
            individual_events[inst] = pd.read_parquet(inst_cache)

        if pool == "individual":
            if inst not in individual_events:
                raise FileNotFoundError(f"Train cache missing: {inst_cache}")
            train_events[inst] = individual_events[inst]
        else:
            if pool not in loaded_pools:
                loaded_pools[pool] = load_pool_train(pool)
            train_events[inst] = loaded_pools[pool]

        n = len(train_events[inst])
        print(f"  {inst}: {n} train rows  (pool={pool})")

    # Load cluster memberships
    cluster_dfs: dict[str, pd.DataFrame] = {
        inst: load_cluster_membership(inst) for inst in CHAMPIONS
    }

    # Load test events
    print("\nLoading test events …")
    test_events: dict[str, pd.DataFrame] = {}
    for inst in CHAMPIONS:
        cfg    = CHAMPIONS[inst]
        pool   = cfg["pool"]
        pool_fc = _feat_cols(train_events[inst]) if pool != "individual" else None
        te = load_test_events_for_inst(inst, pool, pool_fc, data, w_rets)
        if te.empty:
            print(f"  WARNING: no test events for {inst}")
        else:
            print(f"  {inst}: {len(te)} test events  "
                  f"({te['date'].min().date()} → {te['date'].max().date()})")
        test_events[inst] = te

    # ── Per-instrument: CPCV OOF → Platt calibration ──────────────────────────
    print("\n" + "=" * 80)
    print("PHASE A — CPCV OOF for Platt calibration")
    print("=" * 80)
    calibrators: dict[str, PlattCalibrator] = {}

    for inst in CHAMPIONS:
        cfg     = CHAMPIONS[inst]
        family  = cfg["family"]
        pool    = cfg["pool"]
        variant = cfg["variant"]
        spec    = VARIANT_SPECS[inst][variant]
        cd      = cluster_dfs[inst]
        tr_ev   = train_events[inst]
        target  = inst if pool != "individual" else None

        print(f"\n  [{inst}] CPCV OOF (family={family}, pool={pool}, variant={variant}) …")
        oof_df = run_cpcv_oof(inst, tr_ev, spec, cd, family, pool, target_inst=target)

        if oof_df.empty or oof_df["y_true"].nunique() < 2:
            print(f"    WARNING: degenerate OOF for {inst}, using uncalibrated proba")
            calibrators[inst] = PlattCalibrator()
        else:
            cal = PlattCalibrator()
            cal.fit(oof_df["y_score"].values, oof_df["y_true"].values)
            calibrators[inst] = cal
            auc_oof = _safe_auc(oof_df["y_true"].values, oof_df["y_score"].values)
            print(f"    OOF rows={len(oof_df)}  raw AUC={auc_oof:.4f}")

    # ── Per-instrument: refit on full TRAIN → predict on TEST ─────────────────
    print("\n" + "=" * 80)
    print("PHASE B — Full-train refit + test prediction")
    print("=" * 80)

    # Collect all test events with predictions for the backtest
    all_test_events: list[pd.DataFrame] = []
    meta_weights: list[float]     = []
    baseline_weights: list[float] = []

    for inst in CHAMPIONS:
        te = test_events.get(inst)
        if te is None or te.empty:
            continue

        cfg     = CHAMPIONS[inst]
        family  = cfg["family"]
        pool    = cfg["pool"]
        variant = cfg["variant"]
        spec    = VARIANT_SPECS[inst][variant]
        cd      = cluster_dfs[inst]
        tr_ev   = train_events[inst]
        cal     = calibrators[inst]
        vol_ser = vol_lookup[inst]

        print(f"\n  [{inst}] refitting {family}/{variant} on {len(tr_ev)} train rows …")

        fc_tr = _feat_cols(tr_ev)
        X_tr_raw = tr_ev[fc_tr].fillna(0.0).to_numpy(dtype=np.float64)
        y_tr     = tr_ev["bin"].to_numpy(dtype=int)
        ev_tr_df = tr_ev[["date", "t1", "bin", "instrument", "avg_uniqueness"]].reset_index(drop=True)

        X_te_raw = te.reindex(columns=fc_tr, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float64)
        y_te     = te["bin"].to_numpy(dtype=int)

        try:
            X_tr_v, X_te_v, feat_names = transform_variant(X_tr_raw, X_te_raw, fc_tr, spec, cd)
            scaler, model = fit_model(family, pool, X_tr_v, y_tr, ev_tr_df)
            raw_prob = predict_proba(scaler, model, X_te_v)
        except Exception as e:
            print(f"    ERROR: {e}; skipping {inst}")
            continue

        cal_prob = cal.transform(raw_prob)

        if len(np.unique(y_te)) >= 2:
            oos_auc = _safe_auc(y_te, raw_prob)
            print(f"    n_test={len(y_te)}  raw_OOS_AUC={oos_auc:.4f}")

        # Build per-event position weights
        for i, row in te.reset_index(drop=True).iterrows():
            event_date = pd.Timestamp(row["date"])
            side = int(row["side"]) if not np.isnan(row["side"]) else 0

            # EWMA vol at event date (un-annualised daily vol)
            if event_date in vol_ser.index:
                daily_vol = float(vol_ser.loc[event_date])
            else:
                idx = vol_ser.index.searchsorted(event_date)
                if idx > 0 and idx <= len(vol_ser):
                    daily_vol = float(vol_ser.iloc[idx - 1])
                else:
                    daily_vol = float("nan")

            p_cal = float(cal_prob[i])

            # Meta strategy: gate at p* = 0.55
            w_meta = position_weight(side, p_cal, daily_vol)

            # Primary-blind baseline: always use p = 1.0 (full kelly fraction)
            w_base = position_weight(side, 1.0, daily_vol)

            meta_weights.append(w_meta)
            baseline_weights.append(w_base)

        # Tag the test events with predictions
        te_tagged = te.reset_index(drop=True).copy()
        te_tagged["instrument"] = inst
        te_tagged["calibrated_proba"] = cal_prob
        te_tagged["raw_proba"] = raw_prob
        all_test_events.append(te_tagged)

    if not all_test_events:
        print("\nERROR: No test events found for any instrument. Aborting.")
        return

    # Combine all test events
    all_te_df = pd.concat(all_test_events, ignore_index=True)
    all_te_df["t_start"] = pd.to_datetime(all_te_df["date"])
    all_te_df["t_end"]   = pd.to_datetime(all_te_df["t1"])
    meta_w_ser   = pd.Series(meta_weights,     index=all_te_df.index, name="weight")
    baseline_w_ser = pd.Series(baseline_weights, index=all_te_df.index, name="weight")
    all_te_df["weight"] = meta_w_ser

    print(f"\nTotal test events across all instruments: {len(all_te_df)}")
    print(f"  Meta gates fired (|w|>0): {(meta_w_ser.abs() > 0).sum()}")
    print(f"  Baseline weight (|w|>0):  {(baseline_w_ser.abs() > 0).sum()}")

    # ── Backtest ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE C — Backtest (meta vs primary-blind baseline)")
    print("=" * 80)

    # Build returns panel (date × instrument)
    returns_panel = w_rets.copy()

    # Run meta strategy backtest
    print("\nRunning meta strategy backtest …")
    meta_report = barrier_backtest(all_te_df, meta_w_ser, returns_panel)
    meta_m = meta_report.metrics

    # Run baseline backtest
    print("Running baseline backtest …")
    base_report = barrier_backtest(all_te_df, baseline_w_ser, returns_panel)
    base_m = base_report.metrics

    # Print results
    print("\n" + "=" * 80)
    print("OOS PERFORMANCE SUMMARY (net of Grinold-Kahn costs)")
    print("=" * 80)
    print(f"\n{'Metric':30s}  {'Meta':>12s}  {'Baseline':>12s}")
    print("-" * 58)
    for key in ["n", "ann_return", "ann_vol", "sharpe", "sortino",
                "max_dd", "total_cost_bps", "turnover_per_year", "avg_holding_days"]:
        mv = meta_m.get(key, float("nan"))
        bv = base_m.get(key, float("nan"))
        if key == "n":
            print(f"  {key:28s}  {int(mv):>12d}  {int(bv):>12d}")
        else:
            print(f"  {key:28s}  {mv:>12.4f}  {bv:>12.4f}")

    gross_meta = meta_report.gross_returns
    gross_base = base_report.gross_returns
    print(f"\n  {'gross_ann_return':28s}  {meta_m.get('gross_ann_return', float('nan')):>12.4f}"
          f"  {base_m.get('gross_ann_return', float('nan')):>12.4f}")
    print(f"  {'net_ann_return':28s}  {meta_m.get('net_ann_return', float('nan')):>12.4f}"
          f"  {base_m.get('net_ann_return', float('nan')):>12.4f}")

    # Hit rate (fraction of positive-return days)
    meta_hit = float((meta_report.net_returns > 0).mean())
    base_hit = float((base_report.net_returns > 0).mean())
    print(f"  {'hit_rate (daily)':28s}  {meta_hit:>12.4f}  {base_hit:>12.4f}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Writing outputs …")

    # strategy_weights.csv — daily positions (date × instrument)
    weights_path = OUT_DIR / "strategy_weights.csv"
    meta_report.weights.to_csv(weights_path)
    print(f"  {weights_path.relative_to(_REPO)}")

    # metamodel_predictions.csv — per-event predictions
    preds_path = OUT_DIR / "metamodel_predictions.csv"
    pred_cols = ["instrument", "date", "t1", "side", "ret", "bin",
                 "raw_proba", "calibrated_proba", "weight"]
    all_te_df[pred_cols].to_csv(preds_path, index=False)
    print(f"  {preds_path.relative_to(_REPO)}")

    # strategy_daily_net_returns.csv — for make_significance.py
    net_path = STRAT_DIR / "strategy_daily_net_returns.csv"
    meta_report.net_returns.to_csv(net_path, header=True)
    print(f"  {net_path.relative_to(_REPO)}")

    # oos_events_with_predictions.csv — for PT/TM/HM in make_significance.py
    oos_ev_path = STRAT_DIR / "oos_events_with_predictions.csv"
    all_te_df[pred_cols].rename(columns={"weight": "weight"}).to_csv(oos_ev_path, index=False)
    print(f"  {oos_ev_path.relative_to(_REPO)}")

    # ── Significance (S7) ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE D — S7 significance tests")
    print("=" * 80)
    from stml.experimental.make_significance import run as run_significance
    run_significance(verbose=True)

    print("\n" + "=" * 80)
    print("Strategy track complete.")
    print("=" * 80)


if __name__ == "__main__":
    run()
