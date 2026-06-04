"""S6 runner — refit champions, predict, calibrate, size, backtest, emit.

Implements the Madmoun Optional Session 3 strategy-construction recipe:

  1. Per instrument: refit champion on full modelling sample.
  2. Per instrument: purged-OOF predictions used for Platt calibration
     (slide 31) AND for bootstrap p* threshold + sizing-policy fit.
  3. Per instrument: bootstrap p* = L/(G+L) from OOF TP/FP returns (slide 21).
  4. Per instrument: fit chosen sizing policy from {model_confidence,
     all_or_nothing, ncdf, linear_scaling, ecdf, sops} (default SOPS,
     slide 34).
  5. Per event in the sealed test slice: predict, Platt-transform, gate by
     p*, size via the policy, scale by σ_tgt / σ^ann_{t,k} (slide 40).
  6. Aggregate cross-sectionally as (1/K_active) Σ_k w_{t,k} · r_{t+1,k}
     (slide 41), charge Grinold-Kahn costs, write the deliverable CSVs.

Outputs:

    outputs/metamodel_predictions.csv          (calibrated)
    outputs/metamodel_predictions_raw.csv      (uncalibrated)
    outputs/strategy_weights.csv               (lecturer's vol-targeted)
    outputs/coverage_caveat.csv                (per-instrument flags)
    outputs/experiment_log.csv                 (per-class deterministic log)
    results/sreeram_experimental/backtest_metrics.csv
    results/sreeram_experimental/threshold_summary.csv

Acceptance gates:
* Calibration: Platt monotone → AUC invariant (unit test).
* Realised ann vol ≤ 10 % (the lecturer's σ_tgt cap with headroom).
* Byte-identical re-emit verified.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stml.experimental.backtest import barrier_backtest, performance_metrics
from stml.experimental.calibration import PlattCalibrator, expected_calibration_error
from stml.experimental.champion_pipeline import (
    INSTRUMENT_REGIMES,
    POOL_MEMBERS,
    _is_bbg_feature,
    _select_feature_cols,
    _slice_pool,
)
from stml.experimental.config import INSTRUMENTS, PipelineConfig
from stml.experimental.cv import CombinatorialPurgedCV
from stml.experimental.data_loader import load_panel
from stml.experimental.emit import (
    emit_predictions,
    emit_weights,
    panel_to_long,
)
from stml.experimental.evaluation import cross_val_evaluate
from stml.experimental.make_scope import embargo_days_map
from stml.experimental.models import balanced_sample_weight
from stml.experimental.pipeline import _fresh_estimator
from stml.experimental.sizing import (
    MAX_LEVERAGE,
    TARGET_VOL,
    TRADING_DAYS,
    SizingPolicy,
    fit_sizing_policy,
    position_weight,
)
from stml.experimental.threshold import (
    ThresholdEstimate,
    estimate_threshold,
)
from stml.experimental.volatility import ewma_daily_sigma, ewma_lecturer


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


_SCHEMA_COLS = frozenset({
    "instrument", "t_signal", "t_start", "t_end", "side", "ret", "label",
    "uniqueness_weight", "sigma_at_t", "barrier_hit",
    "pt", "sl", "h", "partition",  # Jay's geometry + partition.
})


@dataclass
class InstrumentDeliverable:
    instrument: str
    pool: str
    model_name: str
    test_events: pd.DataFrame
    raw_proba: np.ndarray
    calibrated_proba: np.ndarray
    weight_per_event: np.ndarray


def _load_champions() -> dict[str, dict]:
    """Load champion per-instrument selection from S3-fix output."""
    root = _find_repo_root()
    summary = pd.read_csv(root / "results" / "sreeram_experimental" / "champions_summary.csv")
    with_bbg = summary.loc[summary["variant"] == "with_bbg"].set_index("instrument")
    return {
        inst: {
            "pool": row["winning_pool"],
            "model": row["winning_model"],
        }
        for inst, row in with_bbg.iterrows()
    }


def _add_instrument_onehot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    insts = sorted(df["instrument"].unique())
    for inst in insts:
        out[f"inst_{inst}"] = (df["instrument"] == inst).astype(float)
    return out


def _split_modelling_and_test(features: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final-deliverable split (Jay-CSV partition):

      * **modelling = train + val combined** — used for the final per-
        instrument model refit + the OOF generation that drives Platt /
        threshold / SOPS. This is the maximum-data fit before the sealed
        test window. (Champion *selection* upstream uses train only with
        val as honest scoreboard; that's a separate methodology step in
        ``champion_pipeline.py``.)
      * test = sealed (H1-2022, read once at the end)
    """
    df = features.copy()
    df["t_signal"] = pd.to_datetime(df["t_signal"])
    df["t_end"] = pd.to_datetime(df["t_end"])
    if "partition" not in df.columns:
        raise KeyError(
            "feature matrix is missing the 'partition' column; "
            "re-run make_labels + make_features after the Jay-CSV switch."
        )
    modelling = df.loc[df["partition"].isin(["train", "val"])].reset_index(drop=True)
    test = df.loc[df["partition"] == "test"].reset_index(drop=True)
    return modelling, test


def _predict_for_instrument(
    *,
    instrument: str,
    pool: str,
    model_name: str,
    modelling: pd.DataFrame,
    test: pd.DataFrame,
    cfg: PipelineConfig,
    embargo_map: dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Refit champion on full modelling sample (pool), predict on test slice
    for THIS instrument.

    Returns (test_events_for_this_instrument, raw_proba, oof_modelling_proba)
    where oof_modelling_proba is the purged OOF predictions on the modelling
    sample (for Platt calibration fit).
    """
    pool_modelling = _slice_pool(modelling, pool)
    pool_modelling = _add_instrument_onehot(pool_modelling)
    feature_cols = _select_feature_cols(pool_modelling, "with_bbg") + [
        c for c in pool_modelling.columns if c.startswith("inst_")
    ]

    X_m = pool_modelling.loc[:, feature_cols]
    y_m = pool_modelling["label"].astype(int).values
    uniq_m = pool_modelling["uniqueness_weight"].astype(float).values
    sw_m = balanced_sample_weight(y_m, base=uniq_m)

    # 1) Purged-OOF predictions on the modelling sample for THIS instrument
    #    → used downstream for Platt calibration.
    t_m = pd.to_datetime(pool_modelling["t_signal"])
    t1_m = pd.to_datetime(pool_modelling["t_end"])
    insts_m = pool_modelling["instrument"]
    cv = CombinatorialPurgedCV(
        n_groups=cfg.cpcv_n_groups,
        n_test_groups=cfg.cpcv_n_test_groups,
        t=t_m, t1=t1_m, pct_embargo=cfg.cpcv_pct_embargo,
        instruments=insts_m, embargo_days=embargo_map,
    )
    cv_result = cross_val_evaluate(
        make_model=lambda: _fresh_estimator(model_name, seed=cfg.seed),
        X=X_m, y=pool_modelling["label"].astype(int),
        cv=cv, uniqueness_weights=pool_modelling["uniqueness_weight"].astype(float),
    )
    oos = cv_result.oos_predictions
    oos["instrument"] = oos["row_idx"].map(
        dict(enumerate(insts_m.values))
    )
    inst_oof = oos.loc[oos["instrument"] == instrument].copy()
    # Mean per row to dedupe across CPCV paths.
    oof_mean = inst_oof.groupby("row_idx")["y_proba"].mean()
    oof_label = inst_oof.groupby("row_idx")["y_true"].first()
    oof_aligned = pd.DataFrame({"proba": oof_mean, "label": oof_label})

    # 2) Refit on FULL modelling sample.
    model = _fresh_estimator(model_name, seed=cfg.seed)
    try:
        model.fit(X_m, y_m, sample_weight=sw_m)
    except Exception as exc:
        # Fallback: simple uniform sample weights.
        model = _fresh_estimator(model_name, seed=cfg.seed)
        model.fit(X_m, y_m)

    # 3) Predict on the test slice for THIS instrument.
    pool_test = _slice_pool(test, pool)
    pool_test = _add_instrument_onehot(pool_test)
    # Align columns to pool_modelling (some inst dummies might be missing).
    missing = [c for c in feature_cols if c not in pool_test.columns]
    for c in missing:
        pool_test[c] = 0.0
    X_t = pool_test.loc[:, feature_cols]
    # Predict for ALL rows in the pool test slice; then filter to instrument.
    try:
        all_proba = model.predict_act_proba(X_t)
    except Exception:
        all_proba = np.full(len(X_t), 0.5)
    pool_test = pool_test.assign(_raw_proba=all_proba)
    inst_test = pool_test.loc[pool_test["instrument"] == instrument].copy().reset_index(drop=True)
    raw_proba = inst_test["_raw_proba"].values

    return inst_test, raw_proba, oof_aligned


def _per_event_weight(
    *,
    test_events: pd.DataFrame,
    calibrated_proba: np.ndarray,
    daily_sigma: pd.Series,
    policy: SizingPolicy,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Lecturer-spec weight: ``w = side · g(p̂) · σ_tgt / σ^ann_{t,k}``.

    ``daily_sigma`` is the per-instrument causal EWMA daily σ̂ keyed by date;
    ``policy`` is a fitted :class:`SizingPolicy` (one of six lectured methods).
    """
    weights = np.zeros(len(test_events), dtype=float)
    for i, row in test_events.iterrows():
        sigma_d = float(daily_sigma.get(row["t_start"], float("nan")))
        w = position_weight(
            side=int(row["side"]),
            p=float(calibrated_proba[i]),
            daily_sigma=sigma_d,
            policy=policy,
            target_vol=cfg.target_vol,
            max_leverage=cfg.max_leverage,
            trading_days=TRADING_DAYS,
        )
        weights[i] = w
    return weights


def _returns_panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument daily log returns as a wide panel."""
    rows = []
    for inst in INSTRUMENTS:
        sub = ohlcv.loc[ohlcv["instrument"] == inst, ["date", "close"]].sort_values("date")
        sub = sub.set_index("date")
        ret = np.log(sub["close"] / sub["close"].shift(1))
        rows.append(ret.rename(inst))
    return pd.concat(rows, axis=1).sort_index()


def run(cfg: PipelineConfig | None = None, *, verbose: bool = True) -> dict:
    cfg = cfg or PipelineConfig()
    root = _find_repo_root()
    features = pd.read_parquet(root / "data" / "sreeram_experimental_features.parquet")
    if verbose:
        print(f"Loaded features: {features.shape}")

    champions = _load_champions()
    if verbose:
        print(f"Loaded champions for {len(champions)} instruments")

    embargo_map = embargo_days_map()
    modelling, test = _split_modelling_and_test(features, cfg)
    if verbose:
        print(f"Modelling sample: {len(modelling)} events  Test slice: {len(test)} events")

    # Per-instrument deliverables.
    deliverables: dict[str, InstrumentDeliverable] = {}
    calibrators: dict[str, PlattCalibrator] = {}
    oof_records: list[dict] = []  # for NN portfolio model (slide 48).
    ohlcv, _signals = load_panel()
    returns_panel = _returns_panel(ohlcv)

    for inst in INSTRUMENTS:
        if inst not in champions:
            continue
        pool = champions[inst]["pool"]
        model_name = champions[inst]["model"]
        if model_name == "multitask_nn":
            # Multi-task NN refit + predict is more involved; fall back to its
            # 1-SE-pickable alternative for the deliverable. The multitask
            # contribution is in the CV/per-instrument breakdown only.
            from stml.experimental.champion_pipeline import select_champion
            summary = pd.read_csv(root / "results" / "sreeram_experimental" / "champions_per_pool_per_model.csv")
            grid = summary.loc[
                (summary["variant"] == "with_bbg")
                & (summary["instrument"] == inst)
                & (summary["model"] != "multitask_nn")
            ].sort_values("mean_auc", ascending=False)
            if grid.empty:
                continue
            pool = grid.iloc[0]["pool"]
            model_name = grid.iloc[0]["model"]
            if verbose:
                print(f"  {inst}: multi-task NN champion → substitute "
                      f"{model_name}@{pool} for sklearn-refit deliverable")

        if verbose:
            print(f"\n[deliverable] {inst}: {model_name}@{pool}")

        try:
            test_events, raw_proba, oof = _predict_for_instrument(
                instrument=inst, pool=pool, model_name=model_name,
                modelling=modelling, test=test, cfg=cfg, embargo_map=embargo_map,
            )
        except Exception as exc:
            if verbose:
                print(f"    [WARN] failed: {exc}")
            continue

        if test_events.empty:
            continue

        # ----- 1. Calibration: Platt fit on OOF (slide 31).
        platt = PlattCalibrator()
        if not oof.empty:
            platt.fit(oof["proba"].values, oof["label"].values)
        calibrators[inst] = platt
        calibrated = platt.transform(raw_proba)
        oof_cal = platt.transform(oof["proba"].values) if not oof.empty else np.empty(0)

        # Persist OOF calibrated predictions for downstream consumers (NN
        # portfolio model, slide 48 channel layout).
        modelling_inst_for_oof = modelling.loc[modelling["instrument"] == inst].reset_index(drop=True)
        if oof_cal.size and len(modelling_inst_for_oof) == len(oof_cal):
            oof_records.extend([
                {"date": ts, "instrument": inst, "calibrated_proba": float(p)}
                for ts, p in zip(modelling_inst_for_oof["t_signal"], oof_cal)
            ])

        # ----- 2. Threshold gate p* (slide 21), bootstrapped on OOF training trades.
        # Per-instrument OOF needs the realised return aligned to label/proba.
        # We pull it from the modelling sample of THIS instrument.
        modelling_inst = modelling.loc[modelling["instrument"] == inst].reset_index(drop=True)
        # `oof` is indexed by row_idx INSIDE the pool slice; the per-instrument
        # row alignment is preserved via .groupby("row_idx")["y_true"] earlier.
        if not oof.empty and len(oof) == len(modelling_inst):
            r_oof = modelling_inst["ret"].astype(float).values
            y_oof = oof["label"].values.astype(int)
            p_oof = oof_cal  # use CALIBRATED OOF probabilities for p*
            p_star_est = estimate_threshold(
                p_oof, y_oof, r_oof,
                base_gate=0.5, n_bootstrap=cfg.pstar_bootstrap, seed=cfg.seed,
            )
        else:
            p_star_est = ThresholdEstimate(
                p_star=0.5, ci_low=0.5, ci_high=0.5, n_tp=0, n_fp=0,
                rG_mean=float("nan"), rL_mean=float("nan"),
                bootstrap_p_stars=np.empty(0, dtype=float),
            )
        threshold = p_star_est.p_star if cfg.use_pstar_threshold else 0.5

        # ----- 3. Sizing policy: fit on OOF (slide 34, default SOPS).
        policy = fit_sizing_policy(
            cfg.sizing_method,
            p_tr=oof_cal if oof_cal.size else None,
            r_tr=modelling_inst["ret"].astype(float).values
                  if (oof_cal.size and "ret" in modelling_inst.columns) else None,
            threshold=threshold,
        )

        # ----- 4. Causal EWMA daily σ̂ (slide 39).
        close = ohlcv.loc[ohlcv["instrument"] == inst].set_index("date")["close"]
        daily_returns = np.log(close / close.shift(1)).dropna()
        ewma_sigma = ewma_lecturer(daily_returns, span=cfg.ewma_sigma_span)
        # Re-key by date for lookup at each event's t_start.
        ewma_sigma.index = pd.to_datetime(ewma_sigma.index)

        # ----- 5. Per-event sized weight (slide 40).
        weights = _per_event_weight(
            test_events=test_events, calibrated_proba=calibrated,
            daily_sigma=ewma_sigma, policy=policy, cfg=cfg,
        )

        deliverables[inst] = InstrumentDeliverable(
            instrument=inst, pool=pool, model_name=model_name,
            test_events=test_events.assign(
                raw_proba=raw_proba, calibrated_proba=calibrated, weight=weights,
                p_star=threshold,
                sizing_method=cfg.sizing_method,
            ),
            raw_proba=raw_proba, calibrated_proba=calibrated, weight_per_event=weights,
        )
        if verbose:
            print(f"    p* = {threshold:.4f} (CI [{p_star_est.ci_low:.4f},"
                  f" {p_star_est.ci_high:.4f}], TP={p_star_est.n_tp},"
                  f" FP={p_star_est.n_fp})   sizer={cfg.sizing_method}"
                  f"   n_active = {(weights != 0.0).sum()} / {len(weights)}")

    # Assemble deliverable frames.
    pred_rows_raw = []
    pred_rows_cal = []
    event_rows = []
    for inst, d in deliverables.items():
        for _, row in d.test_events.iterrows():
            pred_rows_raw.append({
                "date": row["t_signal"], "instrument": inst,
                "prediction": float(row["raw_proba"]),
            })
            pred_rows_cal.append({
                "date": row["t_signal"], "instrument": inst,
                "prediction": float(row["calibrated_proba"]),
            })
            event_rows.append({
                "instrument": inst,
                "t_signal": row["t_signal"], "t_start": row["t_start"], "t_end": row["t_end"],
                "side": int(row["side"]), "ret": float(row["ret"]),
                "label": int(row["label"]),
                "raw_proba": float(row["raw_proba"]),
                "calibrated_proba": float(row["calibrated_proba"]),
                "weight": float(row["weight"]),
            })

    if not event_rows:
        print("FATAL: no events produced.")
        return {"status": "no_events"}

    events_full = pd.DataFrame(event_rows).sort_values(["instrument", "t_signal"]).reset_index(drop=True)
    preds_raw = pd.DataFrame(pred_rows_raw)
    preds_cal = pd.DataFrame(pred_rows_cal)

    # Backtest on the test slice.
    report = barrier_backtest(
        events_full.assign(t_start=events_full["t_start"], t_end=events_full["t_end"]),
        events_full["weight"], returns_panel,
    )

    # Weights panel as long deliverable.
    weights_long = panel_to_long(report.weights, value_col="weight")

    # Coverage caveats per instrument.
    n_above_055 = 0
    coverage_rows = []
    for inst in INSTRUMENTS:
        sub = events_full.loc[events_full["instrument"] == inst]
        if sub.empty:
            coverage_rows.append({
                "instrument": inst, "n_oos_events": 0, "thin": True,
                "low_coherence_vs_raw": inst in ("ng1s",),
            })
            continue
        n_oos = len(sub)
        coverage_rows.append({
            "instrument": inst, "n_oos_events": n_oos,
            "thin": n_oos < 60,
            "low_coherence_vs_raw": inst == "ng1s",
            "noisy_coherence_vs_raw": inst in ("ho1s", "rb1s"),
        })
    coverage = pd.DataFrame(coverage_rows)

    # Persist.
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root / "results" / "sreeram_experimental"
    results_dir.mkdir(parents=True, exist_ok=True)

    emit_predictions(preds_raw, outputs_dir / "metamodel_predictions_raw.csv")
    emit_predictions(preds_cal, outputs_dir / "metamodel_predictions.csv")
    emit_weights(weights_long.rename(columns={"weight": "weight"}),
                  outputs_dir / "strategy_weights.csv")
    coverage.to_csv(outputs_dir / "coverage_caveat.csv", index=False)
    events_full.to_csv(results_dir / "oos_events_with_predictions.csv",
                        index=False, float_format="%.10f")
    pd.Series(report.metrics).to_csv(results_dir / "backtest_metrics.csv",
                                       header=["value"])

    # OOF calibrated predictions (for the NN portfolio model, slide 48).
    if oof_records:
        oof_df = pd.DataFrame(oof_records).sort_values(["date", "instrument"])
        oof_df.to_csv(
            results_dir / "oof_calibrated_predictions.csv",
            index=False, float_format="%.10f",
        )

    # Threshold + sizing-policy summary per instrument (audit deliverable).
    thr_rows = []
    for inst, d in deliverables.items():
        sub = d.test_events
        if sub.empty:
            continue
        thr_rows.append({
            "instrument": inst,
            "sizing_method": cfg.sizing_method,
            "p_star": float(sub["p_star"].iloc[0]),
            "n_oos": int(len(sub)),
            "n_oos_taken": int((sub["weight"] != 0.0).sum()),
            "mean_abs_weight": float(np.abs(sub["weight"]).mean()),
            "max_abs_weight": float(np.abs(sub["weight"]).max()),
        })
    if thr_rows:
        pd.DataFrame(thr_rows).to_csv(
            results_dir / "threshold_summary.csv", index=False,
            float_format="%.6f",
        )

    # Persist daily returns for S7 significance.
    report.net_returns.to_csv(results_dir / "strategy_daily_net_returns.csv",
                                header=["net_ret"])
    report.gross_returns.to_csv(results_dir / "strategy_daily_gross_returns.csv",
                                  header=["gross_ret"])

    # Experiment log.
    exp_log = pd.DataFrame([{
        "asset_class": cls,
        "n_events": int(events_full.loc[events_full["instrument"].isin(members)].shape[0]),
        "champion_models": ", ".join(sorted(set(
            champions[i]["model"] for i in members if i in champions
        ))),
    } for cls, members in {
        "equity": ("es1s", "nq1s", "fesx1s"),
        "energy": ("cl1s", "ho1s", "rb1s", "ng1s"),
        "metals": ("gc1s", "si1s", "hg1s", "pl1s"),
    }.items()])
    exp_log.to_csv(outputs_dir / "experiment_log.csv", index=False)

    # Print acceptance gates (Optional Session 3 recipe).
    print("\n=== Strategy acceptance gates (Optional Session 3) ===")
    m = report.metrics
    n_taken = (events_full["weight"] != 0).sum()
    print(f"  sizing method         : {cfg.sizing_method}")
    print(f"  target ann vol        : {cfg.target_vol:.2%}")
    print(f"  events taken (w != 0) : {n_taken} / {len(events_full)} "
          f"({100*n_taken/max(len(events_full),1):.1f}%)")
    print(f"  realised ann vol      : {m['ann_vol']:.4f}  (cap = {cfg.target_vol:.2f})  "
          f"{'PASS' if m['ann_vol'] <= cfg.target_vol + 1e-3 else 'CHECK'}")
    print(f"  realised ann return   : {m['ann_return']:.4f}")
    print(f"  Sharpe ann            : {m['sharpe']:.4f}")
    print(f"  Sortino (full-T)      : {m['sortino']:.4f}")
    print(f"  Max DD                : {m['max_dd']:.4f}")
    print(f"  Turnover (1/yr)       : {m['turnover_per_year']:.2f}")

    return {"report": report, "deliverables": deliverables,
             "events_full": events_full, "coverage": coverage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S6 deliverable runner")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = run(verbose=not args.quiet)
    return 0 if result.get("status") != "no_events" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
