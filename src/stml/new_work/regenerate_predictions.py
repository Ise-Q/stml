"""regenerate_predictions.py — Rebuild OOF and OOS predictions using the
correct locked champion models from each asset-class model comparison notebook.

Champion spec (from locked_picks.csv + notebook headers):

    Instrument  Pool            Model     Variant      Signal
    ----------  ----            -----     -------      ------
    nq1s        nq1s            xgb       reduced      YES
    fesx1s      fesx1s          logistic  reduced      NO
    es1s        es1s            rf        pruned       NO
    cl1s        cl1s            logistic  reduced      NO
    ho1s        energy_cl_ho    mlp       reduced      NO
    rb1s        rb1s            logistic  reduced_min  YES
    ng1s        energy_all      mlp       reduced      YES
    gc1s        precious        rf        reduced      NO
    si1s        precious        rf        reduced      NO
    pl1s        precious        rf        reduced      YES
    hg1s        hg1s            rf        reduced      YES

Outputs:
    data/oof_meta_probabilities.csv   — calibrated OOF probabilities (training set)
    outputs/metamodel_predictions.csv — calibrated OOS probabilities (test set)

Usage:
    .venv/bin/python src/stml/new_work/regenerate_predictions.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

# ── Import per-asset-class modules under aliases ──────────────────────────────
import stml.new_work.equity_model_comparison as _eq
import stml.new_work.energy_model_comparison as _en
import stml.new_work.metals_model_comparison as _me
from stml.new_work import split_config
from stml.new_work.feature_importance import load_all_data, native_returns, wide_returns
from stml.new_work.cpcv_search import CombinatorialPurgedKFold

# ── Champion spec ─────────────────────────────────────────────────────────────

CHAMPIONS: dict[str, dict] = {
    "nq1s":   {"pool": "nq1s",         "model": "xgb",      "variant": "reduced",     "module": _eq},
    "fesx1s": {"pool": "fesx1s",       "model": "logistic",  "variant": "reduced",     "module": _eq},
    "es1s":   {"pool": "es1s",         "model": "rf",        "variant": "pruned",      "module": _eq},
    "cl1s":   {"pool": "cl1s",         "model": "logistic",  "variant": "reduced",     "module": _en},
    "ho1s":   {"pool": "energy_cl_ho", "model": "mlp",       "variant": "reduced",     "module": _en},
    "rb1s":   {"pool": "rb1s",         "model": "logistic",  "variant": "reduced_min", "module": _en},
    "ng1s":   {"pool": "energy_all",   "model": "mlp",       "variant": "reduced",     "module": _en},
    "gc1s":   {"pool": "precious",     "model": "rf",        "variant": "reduced",     "module": _me},
    "si1s":   {"pool": "precious",     "model": "rf",        "variant": "reduced",     "module": _me},
    "pl1s":   {"pool": "precious",     "model": "rf",        "variant": "reduced",     "module": _me},
    "hg1s":   {"pool": "hg1s",        "model": "rf",        "variant": "reduced",     "module": _me},
}

CACHE_DIR = _HERE / "outputs" / "model_comparison" / "_cache"
OUTPUTS   = _HERE / "outputs"

CPCV_N_GROUPS = 6
CPCV_K        = 2
CPCV_EMBARGO  = 0.01
SEED          = 42

_META = frozenset({
    "date", "instrument", "side", "t1", "ret", "bin",
    "trgt", "h", "pt_mult", "sl_mult", "sigma_method", "avg_uniqueness",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


def _apply_variant(
    inst: str,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    fc: list[str],
    variant: str,
    module,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the variant feature transform using the instrument's module.

    All three asset-class modules expose either transform_variant or apply_variant
    with the same signature: (X_tr, X_te, fc, spec, cluster_df) → (X_tr_v, X_te_v, names).
    """
    spec       = module.VARIANT_SPECS[inst][variant]
    cluster_df = module.load_cluster_membership(inst)
    fn         = getattr(module, "transform_variant", None) or module.apply_variant
    X_tr_v, X_te_v, _ = fn(X_tr, X_te, fc, spec, cluster_df)
    return X_tr_v, X_te_v


def _fit_predict(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    ev_tr: pd.DataFrame,
    pool: str,
) -> np.ndarray:
    """Fit model and return predicted probabilities on X_te."""
    if model_name == "logistic":
        scaler, model, _ = _eq._tune_fit_logistic(X_tr, y_tr, ev_tr)
        return model.predict_proba(scaler.transform(X_te))[:, 1]
    elif model_name == "rf":
        model, _ = _eq._tune_fit_rf(X_tr, y_tr, ev_tr)
        return model.predict_proba(X_te)[:, 1]
    elif model_name == "xgb":
        model, _ = _eq._tune_fit_xgb(X_tr, y_tr, ev_tr)
        return model.predict_proba(X_te)[:, 1]
    elif model_name == "mlp":
        # _fit_mlp_fixed handles its own StandardScaler internally, returns (scaler, model)
        scaler, model = _en._fit_mlp_fixed(X_tr, y_tr, pool)
        return model.predict_proba(scaler.transform(X_te))[:, 1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _platt_calibrate(raw_oof: np.ndarray, y_oof: np.ndarray, raw_oos: np.ndarray) -> np.ndarray:
    """Fit Platt sigmoid on OOF (raw, y) and apply to OOS raw probabilities."""
    if len(np.unique(y_oof)) < 2:
        return raw_oos
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    cal.fit(raw_oof.reshape(-1, 1), y_oof)
    return cal.predict_proba(raw_oos.reshape(-1, 1))[:, 1]


def _add_pool_dummies(
    df: pd.DataFrame,
    target_inst: str,
    pool: str,
) -> pd.DataFrame:
    """Add the correct instrument dummy columns for a pooled test event row."""
    if pool == "energy_cl_ho":
        for col, val in _en._CL_HO_DUMMIES[target_inst].items():
            df = df.copy()
            df[col] = val
    elif pool == "energy_all":
        for col, val in _en._EA_DUMMIES[target_inst].items():
            df = df.copy()
            df[col] = val
    elif pool == "precious":
        for col, val in _me._PRECIOUS_DUMMIES[target_inst].items():
            df = df.copy()
            df[col] = val
    return df


# ── Per-instrument prediction generator ──────────────────────────────────────

def generate_instrument_predictions(
    inst: str,
    spec: dict,
    data: dict,
    w_rets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate OOF (training CPCV) and OOS predictions for one instrument.

    Returns
    -------
    oof_df : columns [date, y_true, raw_proba, calibrated_proba, instrument, ret, side]
    oos_df : columns [instrument, date, t1, side, ret, bin, raw_proba, calibrated_proba]
    """
    pool     = spec["pool"]
    model    = spec["model"]
    variant  = spec["variant"]
    module   = spec["module"]
    is_pooled = pool not in {inst}

    print(f"  [{inst}] pool={pool}  model={model}  variant={variant}")

    # ── Load pool training events (pre-BOUNDARY, already cached) ─────────────
    cache_path = CACHE_DIR / f"{pool}_events.parquet"
    pool_train = pd.read_parquet(cache_path)

    # ── Restrict CPCV / full-train to the pool's training rows ───────────────
    fc_pool = _feat_cols(pool_train)
    X_pool  = pool_train[fc_pool].fillna(0.0).to_numpy(dtype=np.float64)
    y_pool  = pool_train["bin"].to_numpy(dtype=int)
    ev_meta = pool_train[["date", "t1", "bin", "instrument", "avg_uniqueness", "ret", "side"]].copy().reset_index(drop=True)

    inst_mask = pool_train["instrument"].values == inst

    # ── CPCV to generate OOF predictions on target instrument ─────────────────
    cpcv = CombinatorialPurgedKFold(n_groups=CPCV_N_GROUPS, k=CPCV_K, embargo=CPCV_EMBARGO)
    oof_rows: list[dict] = []

    for path_i, (tr_idx, te_idx) in enumerate(cpcv.split(ev_meta)):
        # Target instrument rows that fall in the test slice
        te_inst_mask = inst_mask[te_idx]
        if not te_inst_mask.any():
            continue

        y_tr = y_pool[tr_idx]
        if len(np.unique(y_tr)) < 2:
            continue

        X_tr_raw = X_pool[tr_idx]
        X_te_raw = X_pool[te_idx]
        ev_tr    = ev_meta.iloc[tr_idx].reset_index(drop=True)

        try:
            X_tr_v, X_te_v = _apply_variant(inst, X_tr_raw, X_te_raw, fc_pool, variant, module)
        except Exception as e:
            print(f"    path {path_i}: variant transform failed: {e}")
            continue

        # Predict only target-instrument test rows
        X_te_inst = X_te_v[te_inst_mask]
        y_te_inst = y_pool[te_idx][te_inst_mask]
        ev_te_inst = ev_meta.iloc[te_idx].reset_index(drop=True).loc[te_inst_mask].reset_index(drop=True)

        try:
            prob_full = _fit_predict(model, X_tr_v, y_tr, X_te_v, ev_tr, pool)
            prob_inst = prob_full[te_inst_mask]
        except Exception as e:
            print(f"    path {path_i}: model fit failed: {e}")
            continue

        for i in range(len(prob_inst)):
            oof_rows.append({
                "date":      ev_te_inst.iloc[i]["date"],
                "y_true":    int(y_te_inst[i]),
                "raw_proba": float(prob_inst[i]),
                "instrument": inst,
                "ret":       float(ev_te_inst.iloc[i]["ret"]),
                "side":      float(ev_te_inst.iloc[i]["side"]),
            })

    oof_df = pd.DataFrame(oof_rows).sort_values("date").reset_index(drop=True)
    print(f"    OOF rows: {len(oof_df)}")

    # ── Full-train refit on pool training data → predict OOS ─────────────────
    # Build pool feature matrix for OOS events
    from stml.new_work.feature_importance import build_feature_matrix

    if is_pooled:
        # Build OOS events for target instrument, add pool dummies
        _, full_ev = build_feature_matrix(inst, data, w_rets)
        te_ev = split_config.apply_test_mask(full_ev)
        if not te_ev.empty:
            te_ev = _add_pool_dummies(te_ev, inst, pool)
        # Align to pool feature columns
        te_ev = te_ev.reindex(columns=list(_META) + fc_pool, fill_value=0.0) if not te_ev.empty else te_ev
    else:
        _, full_ev = build_feature_matrix(inst, data, w_rets)
        te_ev = split_config.apply_test_mask(full_ev)
        if not te_ev.empty:
            te_ev = te_ev.reindex(columns=list(_META) + fc_pool, fill_value=0.0)

    oos_rows: list[dict] = []
    if te_ev.empty:
        print(f"    OOS: no test events found")
    else:
        X_te_oos = te_ev[fc_pool].fillna(0.0).to_numpy(dtype=np.float64)
        y_te_oos = te_ev["bin"].to_numpy(dtype=int)

        # Full-train X / y (whole pool training)
        ev_tr_full = ev_meta[["date", "t1", "bin", "instrument", "avg_uniqueness"]].copy().reset_index(drop=True)

        try:
            X_tr_v, X_te_v = _apply_variant(inst, X_pool, X_te_oos, fc_pool, variant, module)
            ev_tr_full_aligned = ev_meta[["date", "t1", "bin", "instrument", "avg_uniqueness"]].reset_index(drop=True)
            raw_oos = _fit_predict(model, X_tr_v, y_pool, X_te_v, ev_tr_full_aligned, pool)
        except Exception as e:
            print(f"    OOS refit failed: {e}")
            raw_oos = np.full(len(te_ev), 0.5)

        # Platt calibration: fit on OOF raw → apply to OOS raw
        if len(oof_df) > 10 and oof_df["y_true"].nunique() >= 2:
            cal_oos = _platt_calibrate(
                oof_df["raw_proba"].values,
                oof_df["y_true"].values,
                raw_oos,
            )
        else:
            cal_oos = raw_oos

        print(f"    OOS rows: {len(te_ev)}  mean_cal_p={cal_oos.mean():.3f}  pct_over_50={(cal_oos>0.5).mean():.0%}")

        for i, row in enumerate(te_ev.itertuples(index=False)):
            oos_rows.append({
                "instrument":      inst,
                "date":            row.date,
                "t1":              row.t1,
                "side":            float(row.side),
                "ret":             float(row.ret),
                "bin":             int(row.bin),
                "raw_proba":       float(raw_oos[i]),
                "calibrated_proba": float(cal_oos[i]),
            })

    oos_out = pd.DataFrame(oos_rows)

    # Apply Platt calibration to OOF probabilities too
    if len(oof_df) > 10 and oof_df["y_true"].nunique() >= 2:
        cal_oof = _platt_calibrate(
            oof_df["raw_proba"].values,
            oof_df["y_true"].values,
            oof_df["raw_proba"].values,
        )
        oof_df["calibrated_proba"] = cal_oof
    else:
        oof_df["calibrated_proba"] = oof_df["raw_proba"]

    # Rename for oof output format
    oof_out = oof_df.rename(columns={"y_true": "y_true", "calibrated_proba": "p_hat_oof"})
    oof_out = oof_out[["date", "y_true", "p_hat_oof", "instrument", "ret", "side"]]

    return oof_out, oos_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Regenerate OOF + OOS predictions using locked champion models")
    print("=" * 70)

    print("\nLoading data ...")
    data = load_all_data()
    rets_long = native_returns(data["ohlcv"], kind="log")
    w_rets    = wide_returns(rets_long).sort_index()
    print("Data loaded.\n")

    all_oof: list[pd.DataFrame] = []
    all_oos: list[pd.DataFrame] = []

    for inst, spec in CHAMPIONS.items():
        print(f"\n{'─'*60}")
        print(f"Instrument: {inst}")
        oof, oos = generate_instrument_predictions(inst, spec, data, w_rets)
        all_oof.append(oof)
        all_oos.append(oos)

    # ── Combine and save OOF ─────────────────────────────────────────────────
    # Average p_hat_oof across CPCV paths (each event appears ~k=2 paths)
    oof_raw = pd.concat(all_oof, ignore_index=True)
    agg     = oof_raw.groupby(["instrument", "date"], as_index=False).agg(
        y_true    = ("y_true",    "first"),
        p_hat_oof = ("p_hat_oof", "mean"),
        ret       = ("ret",       "first"),
        side      = ("side",      "first"),
    )
    oof_combined = (
        agg[["date", "y_true", "p_hat_oof", "instrument", "ret", "side"]]
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )
    oof_path = _REPO / "data" / "oof_meta_probabilities.csv"
    oof_combined.to_csv(oof_path, index=False)
    print(f"\nSaved OOF → {oof_path.relative_to(_REPO)}  ({len(oof_combined)} rows)")

    # ── Combine and save OOS ─────────────────────────────────────────────────
    oos_combined = (
        pd.concat(all_oos, ignore_index=True)
        .sort_values(["instrument", "date"])
        .reset_index(drop=True)
    )
    oos_path = OUTPUTS / "metamodel_predictions.csv"
    oos_combined.to_csv(oos_path, index=False)
    print(f"Saved OOS → {oos_path.relative_to(_REPO)}  ({len(oos_combined)} rows)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"{'Instrument':10s}  {'OOF rows':>8s}  {'OOS rows':>8s}  "
          f"{'OOS mean p̂':>10s}  {'OOS % taken':>10s}")
    print("-" * 60)
    for inst in CHAMPIONS:
        oof_sub = oof_combined[oof_combined["instrument"] == inst]
        oos_sub = oos_combined[oos_combined["instrument"] == inst]
        p = oos_sub["calibrated_proba"] if not oos_sub.empty else pd.Series(dtype=float)
        mean_p = p.mean() if len(p) else float("nan")
        pct    = (p > 0.5).mean() if len(p) else float("nan")
        print(f"{inst:10s}  {len(oof_sub):>8d}  {len(oos_sub):>8d}  "
              f"{mean_p:>10.3f}  {pct:>10.0%}")


if __name__ == "__main__":
    main()
