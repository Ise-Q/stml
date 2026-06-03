"""Per-asset-class orchestrator — plan §8 Stage 3.

Lifted with attribution from
``metamodel-apb/src/alken_metamodel/pipeline.py`` (alken parity).

The flow per asset class:

1. Slice the feature matrix to the asset-class member instruments.
2. Restrict to the modelling sample (``date ≤ global_train_cut``).
3. Run a CPCV(6,2) horse-race across the model roster, sample-weighted by
   uniqueness × balanced-class weight.
4. Pick the winner by 15-path mean OOS AUC.
5. Refit the winner on the full modelling sample.
6. Compute per-instrument purged-OOS metrics so a strong pooled number can't
   hide a weak member.
7. Return :class:`AssetClassResult` with everything downstream needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from stml.experimental.config import ASSET_CLASS_MEMBERS, PipelineConfig
from stml.experimental.cv import CombinatorialPurgedCV
from stml.experimental.evaluation import (
    CVResult,
    cross_val_evaluate,
    per_instrument_breakdown,
)
from stml.experimental.make_scope import embargo_days_map
from stml.experimental.models import MetaClassifier, default_roster


# Columns considered "schema" in the features parquet — never used as features.
_SCHEMA_COLS = frozenset({
    "instrument", "t_signal", "t_start", "t_end", "side", "ret", "label",
    "uniqueness_weight", "sigma_at_t", "barrier_hit",
})


# Bloomberg-derived feature families to drop in the "without_bbg" ablation.
_BBG_FAMILY_PREFIXES = ("f18_", "f19_", "f22_")


def _is_bbg_feature(col: str) -> bool:
    return any(col.startswith(p) for p in _BBG_FAMILY_PREFIXES)


@dataclass
class AssetClassResult:
    asset_class: str
    variant: str  # "with_bbg" | "without_bbg" | "simulated_missingness"
    best_model_name: str
    feature_cols: list[str]
    roster_cv: dict[str, CVResult] = field(default_factory=dict)
    cv_winner: CVResult | None = None
    per_instrument: pd.DataFrame = field(default_factory=pd.DataFrame)
    pooled_mean_auc: float = float("nan")
    pooled_std_auc: float = float("nan")
    n_modelling: int = 0


def _select_feature_cols(features: pd.DataFrame, variant: str) -> list[str]:
    """Return the feature column list for a given ablation variant."""
    all_feats = [
        c for c in features.columns
        if c not in _SCHEMA_COLS and not c.startswith("inst_")
    ]
    if variant == "without_bbg":
        return [c for c in all_feats if not _is_bbg_feature(c)]
    return all_feats  # with_bbg + simulated_missingness use the full set


def _add_instrument_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """Append per-instrument one-hot dummies (plan §3.1 — required for per-class XGB).

    Within an asset class, the dummies let the tree learn `instrument × feature`
    interactions without per-instrument training; instrument is just another
    feature the tree can split on first.
    """
    out = df.copy()
    insts = sorted(df["instrument"].unique())
    for inst in insts:
        out[f"inst_{inst}"] = (df["instrument"] == inst).astype(float)
    return out


def _slice_class(features: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    members = ASSET_CLASS_MEMBERS.get(asset_class, ())
    if not members:
        return pd.DataFrame()
    return features.loc[features["instrument"].isin(members)].reset_index(drop=True)


def _restrict_modelling(features: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Keep only modelling-period rows (date ≤ global_train_cut).

    The held-out test slice (> embargo_end) is sealed and consumed only by
    the simulated-missingness ablation and by the S8 final OOS read.
    """
    train_cut = pd.Timestamp(cfg.global_train_cut)
    df = features.copy()
    df["t_signal"] = pd.to_datetime(df["t_signal"])
    df["t_end"] = pd.to_datetime(df["t_end"])
    return df.loc[df["t_signal"] <= train_cut].reset_index(drop=True)


def run_asset_class(
    features: pd.DataFrame,
    asset_class: str,
    *,
    cfg: PipelineConfig | None = None,
    variant: str = "with_bbg",
    roster: dict[str, MetaClassifier] | None = None,
    instrument_scope_path: object = None,
) -> AssetClassResult:
    """Run the full per-class horse-race on the modelling sample.

    Parameters
    ----------
    features
        Output of :mod:`stml.experimental.make_features` — long-form with
        schema cols + feature cols.
    asset_class
        ``'equity'`` | ``'energy'`` | ``'metals'``.
    cfg
        Frozen :class:`PipelineConfig`; defaults if None.
    variant
        ``'with_bbg'`` (full feature matrix), ``'without_bbg'`` (drops F18 / F19
        / F22), or ``'simulated_missingness'`` (full features at train,
        F18/F19/F22 forced to NaN at test — the R-11 ablation).
    roster
        Optional override; default uses :func:`default_roster`.
    instrument_scope_path
        Optional path to a non-default ``instrument_scope.json``.
    """
    cfg = cfg or PipelineConfig()

    cls_features = _slice_class(features, asset_class)
    cls_features = _restrict_modelling(cls_features, cfg)
    if cls_features.empty:
        return AssetClassResult(
            asset_class=asset_class,
            variant=variant,
            best_model_name="(no data)",
            feature_cols=[],
        )

    # Append per-instrument one-hot dummies (plan §3.1 — required for per-class
    # XGB to learn `instrument × feature` interactions without per-instrument
    # training).
    cls_features = _add_instrument_onehot(cls_features)

    feature_cols = _select_feature_cols(cls_features, variant)
    onehot_cols = [c for c in cls_features.columns if c.startswith("inst_")]
    feature_cols = feature_cols + onehot_cols

    X = cls_features.loc[:, feature_cols].copy()
    y = cls_features["label"].astype(int)
    t = pd.to_datetime(cls_features["t_signal"])
    t1 = pd.to_datetime(cls_features["t_end"])
    instruments = cls_features["instrument"]
    uniq = cls_features["uniqueness_weight"].astype(float)

    embargo_map = embargo_days_map(instrument_scope_path)

    cv = CombinatorialPurgedCV(
        n_groups=cfg.cpcv_n_groups,
        n_test_groups=cfg.cpcv_n_test_groups,
        t=t,
        t1=t1,
        pct_embargo=cfg.cpcv_pct_embargo,
        instruments=instruments,
        embargo_days=embargo_map,
    )

    nan_cols: list[str] | None = None
    if variant == "simulated_missingness":
        nan_cols = [c for c in X.columns if _is_bbg_feature(c)]

    roster = roster or default_roster(seed=cfg.seed)

    # Horse-race: per-estimator CV.
    # For XGBoost: pick hyperparameters via a tiny inner grid + 1SE rule on
    # the modelling slice (plan §4.19 — Harry's PM refresh refinement).
    # Other estimators use their plan defaults.
    if "xgboost" in roster:
        best_xgb_overrides = _tune_xgb_via_grid(
            X=X, y=y, t=t, t1=t1, uniq=uniq, instruments=instruments,
            embargo_map=embargo_map, cfg=cfg,
        )
    else:
        best_xgb_overrides = {}

    roster_cv: dict[str, CVResult] = {}
    for name, model in roster.items():
        def _factory(name=name, overrides=best_xgb_overrides):
            return _fresh_estimator(name, seed=cfg.seed, xgb_overrides=overrides)

        result = cross_val_evaluate(
            make_model=_factory,
            X=X,
            y=y,
            cv=cv,
            uniqueness_weights=uniq,
            nan_columns_at_test=nan_cols,
        )
        roster_cv[name] = result

    # Pick winner by mean AUC.
    auc_by_model = {
        name: float(res.mean_scores.get("auc", float("nan")))
        for name, res in roster_cv.items()
    }
    valid = {n: v for n, v in auc_by_model.items() if not np.isnan(v)}
    best_name = max(valid, key=valid.get) if valid else next(iter(roster_cv))
    winner = roster_cv[best_name]

    # Also build a simple-average ensemble across the three estimators
    # (defensive variance reduction, NOT stacking — plan §3.10 only rejects
    # stacked ensembles with learned weights). Tracked as "ensemble_simple"
    # in roster_cv and considered for the per-class winner pick.
    ens_oos = _ensemble_simple_oos(roster_cv)
    if ens_oos is not None and not ens_oos.empty:
        from sklearn.metrics import roc_auc_score
        # Pool-fold AUC for the ensemble (one row per OOS row after averaging).
        ens_auc = float("nan")
        if ens_oos["y_true"].nunique() >= 2:
            try:
                ens_auc = float(roc_auc_score(
                    ens_oos["y_true"], ens_oos["y_proba"],
                    sample_weight=ens_oos["sample_weight"],
                ))
            except Exception:
                pass
        if not np.isnan(ens_auc) and ens_auc > auc_by_model.get(best_name, -1):
            # Replace the winner with the ensemble if it scores higher.
            best_name = "ensemble_simple"
            winner = CVResult(
                fold_scores=pd.DataFrame(),
                oos_predictions=ens_oos,
                mean_scores={"auc": ens_auc},
                std_scores={"auc": float("nan")},
            )
            roster_cv["ensemble_simple"] = winner

    per_inst = per_instrument_breakdown(
        instruments=instruments, oos_predictions=winner.oos_predictions
    )

    return AssetClassResult(
        asset_class=asset_class,
        variant=variant,
        best_model_name=best_name,
        feature_cols=feature_cols,
        roster_cv=roster_cv,
        cv_winner=winner,
        per_instrument=per_inst,
        pooled_mean_auc=float(winner.mean_scores.get("auc", float("nan"))),
        pooled_std_auc=float(winner.std_scores.get("auc", float("nan"))),
        n_modelling=int(len(cls_features)),
    )


def _ensemble_simple_oos(roster_cv: dict[str, CVResult]) -> pd.DataFrame | None:
    """Build a simple-average ensemble OOS prediction frame across estimators.

    For each (row_idx, fold) tuple present in ALL estimators' OOS predictions,
    take the simple mean of `y_proba`. Then dedupe by row_idx (mean across
    folds) and return one row per OOS event.

    Returns ``None`` if fewer than 2 estimators have non-empty OOS frames.
    """
    frames = [
        res.oos_predictions for res in roster_cv.values()
        if res.oos_predictions is not None and not res.oos_predictions.empty
    ]
    if len(frames) < 2:
        return None

    # Align by (row_idx, fold) — every estimator was fit on the same CPCV
    # splits, so the (row_idx, fold) sets should match.
    base = frames[0][["row_idx", "fold", "y_true", "sample_weight"]].copy()
    base["y_proba_sum"] = frames[0]["y_proba"].astype(float)
    base["n_models"] = 1
    for f in frames[1:]:
        merge = f[["row_idx", "fold", "y_proba"]]
        merged = base.merge(merge, on=["row_idx", "fold"], how="inner", suffixes=("", "_x"))
        merged["y_proba_sum"] = merged["y_proba_sum"] + merged["y_proba"]
        merged["n_models"] = merged["n_models"] + 1
        base = merged.drop(columns=["y_proba"])
    base["y_proba"] = base["y_proba_sum"] / base["n_models"]
    # Dedupe by row_idx — mean across folds.
    out = base.groupby("row_idx", as_index=False).agg(
        y_true=("y_true", "first"),
        sample_weight=("sample_weight", "first"),
        y_proba=("y_proba", "mean"),
        fold=("fold", "first"),
    )
    return out[["fold", "row_idx", "y_true", "y_proba", "sample_weight"]]


def _fresh_estimator(
    name: str,
    *,
    seed: int = 42,
    xgb_overrides: dict | None = None,
) -> MetaClassifier:
    """Build a fresh MetaClassifier by name."""
    from stml.experimental.models import (
        make_elasticnet_logistic,
        make_lightgbm,
        make_random_forest,
        make_xgb,
    )

    if name == "elasticnet_logistic":
        return make_elasticnet_logistic(seed=seed)
    if name == "xgboost":
        return make_xgb(seed=seed, **(xgb_overrides or {}))
    if name == "lightgbm":
        return make_lightgbm(seed=seed)
    if name == "random_forest":
        return make_random_forest(seed=seed)
    raise ValueError(f"Unknown estimator name: {name}")


def _tune_xgb_via_grid(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    t: pd.Series,
    t1: pd.Series,
    uniq: pd.Series,
    instruments: pd.Series,
    embargo_map: dict[str, int],
    cfg: PipelineConfig,
) -> dict:
    """Inner purged k-fold + 1SE rule (plan §4.19) over a small XGB grid.

    Grid: max_depth × n_estimators × reg_lambda. The "1SE" rule picks the
    MOST REGULARISED config within 1 SE of the best mean inner-fold AUC —
    strongly favours generalisation over in-sample fit.
    """
    from stml.experimental.models import make_xgb

    inner_cv = CombinatorialPurgedCV(
        n_groups=cfg.inner_kfold_k + 1,
        n_test_groups=1,
        t=t,
        t1=t1,
        pct_embargo=cfg.cpcv_pct_embargo,
        instruments=instruments,
        embargo_days=embargo_map,
    )

    # Compact grid — emphasizing regularisation.
    grid = [
        {"max_depth": 3, "n_estimators": 100, "reg_lambda": 1.0, "reg_alpha": 0.1},
        {"max_depth": 3, "n_estimators": 200, "reg_lambda": 2.0, "reg_alpha": 0.5},
        {"max_depth": 4, "n_estimators": 100, "reg_lambda": 1.0, "reg_alpha": 0.1},
        {"max_depth": 4, "n_estimators": 200, "reg_lambda": 2.0, "reg_alpha": 0.5},
        {"max_depth": 5, "n_estimators": 100, "reg_lambda": 2.0, "reg_alpha": 0.5},
    ]
    # Sort grid from most-regularised to least (1SE pick = first config within
    # 1 SE of the best).
    grid = sorted(grid, key=lambda d: (-d["reg_lambda"], -d["reg_alpha"], d["max_depth"], d["n_estimators"]))

    configs_scores: list[dict] = []
    for params in grid:
        result = cross_val_evaluate(
            make_model=lambda p=params: make_xgb(seed=cfg.seed, **p),
            X=X, y=y, cv=inner_cv, uniqueness_weights=uniq,
        )
        mean_auc = float(result.mean_scores.get("auc", float("nan")))
        std_auc = float(result.std_scores.get("auc", float("nan")))
        n_folds = len(result.fold_scores)
        sem = std_auc / max(np.sqrt(n_folds), 1.0) if not np.isnan(std_auc) else float("nan")
        configs_scores.append({
            "params": params,
            "mean_auc": mean_auc,
            "sem": sem,
        })

    valid = [c for c in configs_scores if not np.isnan(c["mean_auc"])]
    if not valid:
        return {}  # fall back to defaults
    best = max(valid, key=lambda c: c["mean_auc"])
    threshold = best["mean_auc"] - (best["sem"] if not np.isnan(best["sem"]) else 0.0)
    # First config (already sorted most-regularised first) within 1 SE of best.
    pick = next((c for c in valid if c["mean_auc"] >= threshold), best)
    return pick["params"]
