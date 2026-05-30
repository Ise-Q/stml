"""End-to-end orchestration: one meta-labelling metamodel per asset class (Stage 5).

Data flows in one direction (CLAUDE.md):
  load_clean_data -> per-instrument causal features (+ regime) -> triple-barrier meta-labels
  on non-zero-signal days -> pool the class (+ instrument-id) -> purged-CV horse-race to
  SELECT a model -> refit on the locked modelling sample -> predict P(act) on the
  config-driven window.

Leakage discipline enforced here:
- Features are computed on each instrument's FULL history then right-sliced (``features.py``);
  regime fitted blocks fit on the contiguous ``fe_train_end`` prefix (``regime.py``).
- The model is SELECTED and TRAINED only on the modelling sample (dates <= ``modelling_end``),
  locking the feature set before the final OOS window — no snooping on the prediction window.
- The prediction window is **config-driven** (``predict_start``/``predict_end``), never hardcoded
  to Jan–Jun 2022, so the grader can swap in the hidden Jul–Dec 2022 half.
- The pooled feature matrix keeps the event-date index so ``PurgedKFold`` purges concurrent
  labels across instruments by their ``t1`` spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from stml.metamodel.scope import ASSET_CLASS_MAP

from .calibration import PlattCalibrator
from .cross_validation import CombinatorialPurgedCV, PurgedKFold, nested_cpcv
from .evaluation import cross_val_evaluate, evaluate_predictions, oos_predictions
from .features import (
    assemble_instrument_features,
    attach_instrument,
    daily_barrier_sigma,
    filter_signal_days,
)
from .models import balanced_sample_weight, tree_linear_roster
from .regime import assemble_regime_features
from .seeding import set_seeds
from .triple_barrier import triple_barrier_labels

#: Friendly class name -> the ASSET_CLASS_MAP code (the authoritative 11-instrument universe;
#: NOT ``io.INSTRUMENTS``, which omits nq1s).
ASSET_CLASS_CODES = {"equity": "EQ", "energy": "EN", "metals": "ME"}
_LABEL_COLS = ("side", "t1", "ret", "bin", "weight")
_NON_FEATURE = set(_LABEL_COLS) | {"instrument", "date"}


def class_members(asset_class: str) -> list[str]:
    """Instruments in an asset class, in ASSET_CLASS_MAP order."""
    code = ASSET_CLASS_CODES[asset_class]
    return [inst for inst, c in ASSET_CLASS_MAP.items() if c == code]


@dataclass(frozen=True)
class PipelineConfig:
    """Dates, barrier, CV and sizing settings. The prediction window is config-driven."""

    fe_train_end: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2021-07-01"))
    modelling_end: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2021-12-31"))
    predict_start: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2022-01-01"))
    predict_end: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2022-06-30"))
    pt_sl: tuple[float, float] = (1.0, 1.0)
    max_holding: int = 10
    n_splits: int = 5
    pct_embargo: float = 0.01
    seed: int = 42
    use_regime: bool = True
    use_macro: bool = False  # join the PIT-lagged macro block (§1/§3) into the feature panel
    #: "tree_linear" (fast default), "default" (tree/linear + reduced torch NNs, the shipped
    #: deliverable roster), or "full" (also adds the off-path KerasVSN comparison).
    roster: str = "tree_linear"
    cv_scheme: str = "purged"  # "purged" (PurgedKFold) or "cpcv" (15-path selection distribution)
    cpcv_groups: int = 6  # N for CombinatorialPurgedCV -> C(6,2)=15 paths
    cpcv_test_groups: int = 2


def _roster_factory(config: PipelineConfig):
    """Resolve the horse-race roster factory; ``full`` lazily pulls in the neural variants."""
    if config.roster == "default":
        from .neural import default_roster

        return default_roster
    if config.roster == "full":
        from .neural import full_roster

        return full_roster
    return tree_linear_roster


@dataclass
class AssetClassResult:
    asset_class: str
    predictions: pd.DataFrame  # date, instrument, prediction, prediction_calibrated, side, ann_vol
    best_model: str
    cv_scores: dict[str, float]
    n_modelling: int
    diagnostics: pd.DataFrame  # per-instrument OOS metrics (printed before the aggregate)
    calibrator: object  # per-class Platt map fit on modelling-OOS preds (pass-3 S3.9)
    oos_brier: float  # class-level modelling-OOS Brier (XT.2 experiment-log backfill)
    oos_precision: float  # class-level modelling-OOS precision (XT.2)


class _IdentityCalibrator:
    """Pass-through calibrator used when the modelling-OOS sample is too degenerate to fit Platt."""

    def transform(self, proba) -> np.ndarray:
        return np.asarray(proba, dtype=float)


def fit_oos_calibrator(make_model, X, y, t1: pd.Series, sample_weight, config: PipelineConfig):
    """Fit a Platt calibrator on purged-OOS predictions of the **modelling** sample (S3.9/S6.11).

    The calibrator is fit only on the rows passed here — the caller restricts these to dates
    ``<= modelling_end``, so applying the map to the deliverable (``>= predict_start``) cannot leak.
    Falls back to an identity map when the OOS sample is single-class or too thin to calibrate.
    Returns ``(calibrator, oos_predictions, finite_mask)`` so the caller reuses the OOS array.
    """
    cv = PurgedKFold(n_splits=config.n_splits, t1=t1, pct_embargo=config.pct_embargo)
    oos = oos_predictions(make_model, X, y, cv, sample_weight=sample_weight)
    finite = np.isfinite(oos)
    yv = np.asarray(y)[finite]
    if finite.sum() < 20 or len(np.unique(yv)) < 2:
        return _IdentityCalibrator(), oos, finite
    return PlattCalibrator().fit(oos[finite], yv), oos, finite


def _close_of(ohlcv_inst: pd.DataFrame) -> pd.Series:
    s = ohlcv_inst.set_index("date")["close"].sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s.astype(float)


def build_instrument_panel(
    ohlcv: pd.DataFrame, signals: pd.DataFrame, instrument: str, config: PipelineConfig
) -> pd.DataFrame:
    """Per-instrument modelling table: causal features (+regime) joined to triple-barrier
    meta-labels on the non-zero-signal trade days, keyed by event date and instrument."""
    ohlcv_inst = ohlcv[ohlcv["instrument"] == instrument]
    signal = signals.set_index("date")[instrument].sort_index()
    signal.index = pd.DatetimeIndex(signal.index)

    feats = assemble_instrument_features(ohlcv_inst, signal)
    if config.use_regime:
        regime = assemble_regime_features(
            ohlcv_inst, fit_end=config.fe_train_end, seed=config.seed
        )
        feats = pd.concat([feats, regime.reindex(feats.index)], axis=1)
    if config.use_macro:
        from .macro import macro_features

        macro = macro_features(feats.index)  # PIT-lagged, causal -> truncation-invariant
        feats = pd.concat([feats, macro.reindex(feats.index)], axis=1)

    sigma = daily_barrier_sigma(feats)  # de-annualised daily barrier width
    labels = triple_barrier_labels(
        _close_of(ohlcv_inst),
        signal,
        sigma,
        pt_sl=config.pt_sl,
        max_holding=config.max_holding,
    )

    feats_on_events = filter_signal_days(feats, signal)
    panel = feats_on_events.join(labels, how="inner").dropna(subset=["bin"])
    panel = attach_instrument(panel, instrument)
    panel.index.name = None  # keep the event-date index (for t1 alignment) but free the name
    panel["date"] = panel.index
    return panel


def build_class_panel(
    ohlcv: pd.DataFrame, signals: pd.DataFrame, instruments: list[str], config: PipelineConfig
) -> pd.DataFrame:
    """Pool the class's instrument panels and add instrument-id one-hot columns.

    Keeps the event-date index (duplicated across instruments) so the purged CV can purge
    concurrent cross-instrument labels by their ``t1`` spans.
    """
    panels = [build_instrument_panel(ohlcv, signals, i, config) for i in instruments]
    pooled = pd.concat(panels, axis=0)
    for inst in instruments:
        pooled[f"inst_{inst}"] = (pooled["instrument"] == inst).astype(float)
    return pooled.sort_values(["date", "instrument"])


def feature_columns(pooled: pd.DataFrame) -> list[str]:
    return [c for c in pooled.columns if c not in _NON_FEATURE]


def _make_cv(t1: pd.Series, config: PipelineConfig):
    """The selection splitter: PurgedKFold (default) or 15-path CombinatorialPurgedCV (S3.8)."""
    if config.cv_scheme == "cpcv":
        return CombinatorialPurgedCV(
            config.cpcv_groups, config.cpcv_test_groups, t1, config.pct_embargo
        )
    return PurgedKFold(n_splits=config.n_splits, t1=t1, pct_embargo=config.pct_embargo)


def _horse_race(factory, X, y, cv, sample_weight, seed) -> dict[str, float]:
    """Mean OOS AUC per roster estimator over ``cv`` (NaN folds -> ignored by nanmean)."""
    scores: dict[str, float] = {}
    for name in factory(seed=seed):
        res = cross_val_evaluate(
            lambda name=name: factory(seed=seed)[name], X, y, cv, sample_weight=sample_weight
        )
        scores[name] = float(np.nanmean(res["auc"].to_numpy()))
    return scores


def select_model(X, y, t1, sample_weight, config: PipelineConfig) -> tuple[str, dict[str, float]]:
    """Horse-race the roster by mean OOS AUC under ``config.cv_scheme``; return winner + scores."""
    cv = _make_cv(t1, config)
    scores = _horse_race(_roster_factory(config), X, y, cv, sample_weight, config.seed)
    best = max(scores, key=lambda k: scores[k] if np.isfinite(scores[k]) else -np.inf)
    return best, scores


def nested_cpcv_select_and_evaluate(
    X, y, t1: pd.Series, sample_weight, config: PipelineConfig
) -> pd.DataFrame:
    """Headline nested-CPCV: inner CPCV picks the estimator, outer CPCV scores it (S3.8).

    Each outer fold runs the full inner horse-race on the outer-train rows ONLY (no tuning
    leakage — the inner splits are built from ``t1.iloc[outer_train]``), refits the winner, and
    scores it on the held-out outer fold. The returned per-outer-fold distribution is the
    selection-bias-aware OOS estimate reported in the §3/§6 write-up (small-N variance is the
    documented cost). Diagnostic only — never feeds the locked deliverable config.
    """
    factory = _roster_factory(config)
    y = np.asarray(y)
    sample_weight = np.asarray(sample_weight)
    rows = []
    for i, (otr, ote, inner_cv) in enumerate(
        nested_cpcv(
            X,
            t1,
            outer_groups=config.cpcv_groups,
            outer_test_groups=config.cpcv_test_groups,
            pct_embargo=config.pct_embargo,
        )
    ):
        x_tr, y_tr, sw_tr = X.iloc[otr], y[otr], sample_weight[otr]
        scores = _horse_race(factory, x_tr, y_tr, inner_cv, sw_tr, config.seed)
        best = max(scores, key=lambda k: scores[k] if np.isfinite(scores[k]) else -np.inf)
        model = factory(seed=config.seed)[best]
        model.fit(x_tr, y_tr, sample_weight=sw_tr)
        proba = model.predict_act_proba(X.iloc[ote])
        y_te = y[ote]
        auc = (
            evaluate_predictions(y_te, proba)["auc"]
            if len(np.unique(y_te)) == 2
            else float("nan")
        )
        rows.append(
            {"outer_fold": i, "selected": best, "auc": round(float(auc), 6), "n_test": len(ote)}
        )
    return pd.DataFrame(rows)


def per_instrument_diagnostics(
    x_model,
    y_model,
    t1_model,
    instrument_model,
    sample_weight,
    best_name,
    config: PipelineConfig,
    *,
    oos=None,
) -> pd.DataFrame:
    """Per-instrument purged-OOS metrics for the selected model (§5 per-instrument reporting).

    One purged-CV pass of the winning estimator collects OOS P(act) for every modelling row;
    metrics are then grouped by instrument so a strong pooled number can't hide a weak member.
    The OOS array may be supplied (reused from calibration) to avoid recomputing the CV pass.
    """
    if oos is None:
        cv = PurgedKFold(n_splits=config.n_splits, t1=t1_model, pct_embargo=config.pct_embargo)
        factory = _roster_factory(config)
        oos = oos_predictions(
            lambda: factory(seed=config.seed)[best_name],
            x_model,
            y_model,
            cv,
            sample_weight=sample_weight,
        )
    rows = []
    for inst in sorted(set(instrument_model)):
        in_inst = instrument_model == inst
        scored = in_inst & np.isfinite(oos)
        yi, pi = y_model[scored], oos[scored]
        metrics = evaluate_predictions(yi, pi) if len(yi) and len(np.unique(yi)) == 2 else {}
        rows.append(
            {
                "instrument": inst,
                "n": int(in_inst.sum()),
                "pos_rate": round(float(y_model[in_inst].mean()), 4) if in_inst.any() else np.nan,
                "auc": round(metrics.get("auc", float("nan")), 4),
                "precision": round(metrics.get("precision", float("nan")), 4),
            }
        )
    return pd.DataFrame(rows)


def run_asset_class(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    asset_class: str,
    config: PipelineConfig | None = None,
) -> AssetClassResult:
    """Select + train one metamodel for ``asset_class`` and predict the config-driven window."""
    config = config or PipelineConfig()
    set_seeds(config.seed)
    instruments = class_members(asset_class)

    pooled = build_class_panel(ohlcv, signals, instruments, config)
    cols = feature_columns(pooled)
    X = pooled[cols]
    y = pooled["bin"].to_numpy()
    t1 = pooled["t1"]
    uniqueness = pooled["weight"].to_numpy()
    dates = pd.DatetimeIndex(pooled["date"])

    model_mask = np.asarray(dates <= config.modelling_end)
    pred_mask = np.asarray((dates >= config.predict_start) & (dates <= config.predict_end))

    x_model, y_model = X[model_mask], y[model_mask]
    sw_model = balanced_sample_weight(y_model, base=uniqueness[model_mask])
    best, scores = select_model(x_model, y_model, t1[model_mask], sw_model, config)

    # One purged-OOS pass of the winner over the modelling sample feeds calibration, the per-
    # instrument diagnostics, and the class-level Brier/precision — all strictly pre-predict_start.
    def _make_best():
        return _roster_factory(config)(seed=config.seed)[best]

    calibrator, oos_model, finite = fit_oos_calibrator(
        _make_best, x_model, y_model, t1[model_mask], sw_model, config
    )
    yv = y_model[finite]
    calibratable = finite.any() and len(np.unique(yv)) == 2
    class_metrics = evaluate_predictions(yv, oos_model[finite]) if calibratable else {}

    diagnostics = per_instrument_diagnostics(
        x_model,
        y_model,
        t1[model_mask],
        pooled["instrument"].to_numpy()[model_mask],
        sw_model,
        best,
        config,
        oos=oos_model,
    )

    model = _make_best()
    model.fit(x_model, y_model, sample_weight=sw_model)

    x_pred = X[pred_mask]
    proba = model.predict_act_proba(x_pred)
    proba_cal = np.clip(calibrator.transform(proba), 0.0, 1.0)  # calibrated p̂ feeds Kelly sizing
    predictions = pd.DataFrame(
        {
            "date": dates[pred_mask],
            "instrument": pooled["instrument"].to_numpy()[pred_mask],
            "prediction": proba,
            "prediction_calibrated": proba_cal,
            "side": pooled["side"].to_numpy()[pred_mask],
            "ann_vol": pooled["f2_vol_20"].to_numpy()[pred_mask],
        }
    )
    return AssetClassResult(
        asset_class=asset_class,
        predictions=predictions,
        best_model=best,
        cv_scores=scores,
        n_modelling=int(model_mask.sum()),
        diagnostics=diagnostics,
        calibrator=calibrator,
        oos_brier=float(class_metrics.get("brier", float("nan"))),
        oos_precision=float(class_metrics.get("precision", float("nan"))),
    )
