"""Champion architecture — per-instrument + alternative-pool model selection.

Plan §4.3 + §4.4 + branch_descriptions §4.11 (Harry's `INSTRUMENT_REGIMES`).

Per instrument, evaluate multiple candidate `(pool, model)` combinations under
CPCV(6,2) + per-instrument embargo, pick the **champion** by mean AUC + 1-SE
rule (most regularised within 1 SE of the best). Aggregate champion predictions
into one per-instrument OOS frame.

This replaces the prior per-class-only pipeline (`pipeline.run_asset_class`)
for the final reported per-instrument breakdown.

The pool options per instrument mirror Harry's PM-refresh `INSTRUMENT_REGIMES`:

    es1s / nq1s / fesx1s / hg1s : individual only (already specialised within
        their class; pooling would dilute the signal).
    cl1s   : individual OR energy_all OR energy_cl_ho
    ho1s   : energy_all OR energy_cl_ho (too thin for individual)
    rb1s   : individual OR energy_all
    ng1s   : energy_all (Harry's PM refresh; ng1s individual dropped after the
        clean-split discipline showed it didn't survive)
    gc1s   : individual OR precious (gc + si + pl trio)
    si1s   : individual OR precious
    pl1s   : individual OR precious

Plus a fourth coarse pool — full asset class (equity / energy / metals) — as a
fallback the champion can fall back to when neither individual nor narrow
pools win.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from stml.experimental.config import PipelineConfig
from stml.experimental.cv import CombinatorialPurgedCV
from stml.experimental.evaluation import CVResult, cross_val_evaluate
from stml.experimental.make_scope import embargo_days_map
from stml.experimental.models import balanced_sample_weight, default_roster

# Multi-task NN (S4) — torch is in optional `multitask` extra.
try:
    from stml.experimental.multitask import MultiTaskConfig, MultiTaskMetaClassifier
    MULTITASK_AVAILABLE = True
except ImportError:
    MULTITASK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pool definitions — Harry §4.11 INSTRUMENT_REGIMES + the full asset class.
# ---------------------------------------------------------------------------

# Each entry is a tuple of instruments that constitute a pool.
POOL_MEMBERS: dict[str, tuple[str, ...]] = {
    # Asset-class pools (alken-style)
    "equity_all":    ("es1s", "nq1s", "fesx1s"),
    "energy_all":    ("cl1s", "ho1s", "rb1s", "ng1s"),
    "metals_all":    ("gc1s", "si1s", "hg1s", "pl1s"),
    # Sub-class pools (Harry-style)
    "energy_cl_ho":  ("cl1s", "ho1s"),
    "precious":      ("gc1s", "si1s", "pl1s"),
    # Per-instrument pools (one entry per instrument).
    "es1s":   ("es1s",),
    "nq1s":   ("nq1s",),
    "fesx1s": ("fesx1s",),
    "cl1s":   ("cl1s",),
    "ho1s":   ("ho1s",),
    "rb1s":   ("rb1s",),
    "ng1s":   ("ng1s",),
    "gc1s":   ("gc1s",),
    "si1s":   ("si1s",),
    "pl1s":   ("pl1s",),
    "hg1s":   ("hg1s",),
}

# Plan §4.3 / Harry §4.11 — per-instrument candidate pools.
INSTRUMENT_REGIMES: dict[str, list[str]] = {
    "es1s":   ["es1s", "equity_all"],
    "nq1s":   ["nq1s", "equity_all"],
    "fesx1s": ["fesx1s", "equity_all"],
    "cl1s":   ["cl1s", "energy_cl_ho", "energy_all"],
    "ho1s":   ["energy_cl_ho", "energy_all"],
    "rb1s":   ["rb1s", "energy_all"],
    "ng1s":   ["energy_all"],
    "gc1s":   ["gc1s", "precious", "metals_all"],
    "si1s":   ["si1s", "precious", "metals_all"],
    "pl1s":   ["pl1s", "precious", "metals_all"],
    "hg1s":   ["hg1s", "metals_all"],
}

# Minimum modelling events required to train on an individual instrument.
MIN_INDIVIDUAL_EVENTS = 250


# ---------------------------------------------------------------------------
# Schema utilities.
# ---------------------------------------------------------------------------


_SCHEMA_COLS = frozenset({
    "instrument", "t_signal", "t_start", "t_end", "side", "ret", "label",
    "uniqueness_weight", "sigma_at_t", "barrier_hit",
})

_BBG_FAMILY_PREFIXES = ("f18_", "f19_", "f22_")


def _is_bbg_feature(col: str) -> bool:
    return any(col.startswith(p) for p in _BBG_FAMILY_PREFIXES)


def _select_feature_cols(features: pd.DataFrame, variant: str) -> list[str]:
    all_feats = [
        c for c in features.columns
        if c not in _SCHEMA_COLS and not c.startswith("inst_")
    ]
    if variant == "without_bbg":
        return [c for c in all_feats if not _is_bbg_feature(c)]
    return all_feats


def _add_instrument_onehot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    insts = sorted(df["instrument"].unique())
    for inst in insts:
        out[f"inst_{inst}"] = (df["instrument"] == inst).astype(float)
    return out


def _restrict_modelling(features: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    train_cut = pd.Timestamp(cfg.global_train_cut)
    df = features.copy()
    df["t_signal"] = pd.to_datetime(df["t_signal"])
    df["t_end"] = pd.to_datetime(df["t_end"])
    return df.loc[df["t_signal"] <= train_cut].reset_index(drop=True)


def _slice_pool(features: pd.DataFrame, pool: str) -> pd.DataFrame:
    members = POOL_MEMBERS.get(pool, ())
    if not members:
        return pd.DataFrame()
    return features.loc[features["instrument"].isin(members)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Champion selection.
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    instrument: str
    pool: str
    model_name: str
    mean_auc: float = float("nan")
    std_auc: float = float("nan")
    sem: float = float("nan")
    n_modelling: int = 0
    n_oos_for_instrument: int = 0
    oos_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_cols: list[str] = field(default_factory=list)


@dataclass
class ChampionResult:
    instrument: str
    winning_pool: str
    winning_model: str
    champion_auc: float
    champion_sem: float
    candidates: list[CandidateResult] = field(default_factory=list)
    selection_reason: str = ""


def _filter_oos_to_instrument(
    oos: pd.DataFrame, pool_instruments: pd.Series, instrument: str
) -> pd.DataFrame:
    """Reduce a pool's OOS predictions to rows for the target instrument only."""
    if oos.empty:
        return oos
    inst_by_idx = dict(enumerate(pool_instruments.values))
    out = oos.copy()
    out["instrument"] = out["row_idx"].map(inst_by_idx)
    return out.loc[out["instrument"] == instrument].reset_index(drop=True)


def _evaluate_multitask_candidate(
    *,
    instrument: str,
    pool: str,
    pool_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: PipelineConfig,
    nan_cols_at_test: list[str] | None,
    embargo_map: dict[str, int],
) -> CandidateResult:
    """Multi-task NN evaluator — only makes sense on multi-instrument pools.

    Manually iterates CPCV(6,2) folds (instead of using cross_val_evaluate which
    is built for sklearn-style estimators without instrument_ids).
    """
    if not MULTITASK_AVAILABLE:
        return CandidateResult(
            instrument=instrument, pool=pool, model_name="multitask_nn",
        )
    pool_members = POOL_MEMBERS[pool]
    if len(pool_members) < 2:
        return CandidateResult(
            instrument=instrument, pool=pool, model_name="multitask_nn",
        )
    inst_to_id = {ticker: i for i, ticker in enumerate(pool_members)}
    if instrument not in inst_to_id:
        return CandidateResult(
            instrument=instrument, pool=pool, model_name="multitask_nn",
        )

    pool_df = pool_df.copy()
    X = pool_df.loc[:, feature_cols].copy()
    y = pool_df["label"].astype(int)
    t = pd.to_datetime(pool_df["t_signal"])
    t1 = pd.to_datetime(pool_df["t_end"])
    instruments = pool_df["instrument"]
    inst_ids = instruments.map(inst_to_id).astype(int).values
    uniq = pool_df["uniqueness_weight"].astype(float).values

    cv = CombinatorialPurgedCV(
        n_groups=cfg.cpcv_n_groups,
        n_test_groups=cfg.cpcv_n_test_groups,
        t=t, t1=t1,
        pct_embargo=cfg.cpcv_pct_embargo,
        instruments=instruments, embargo_days=embargo_map,
    )

    oos_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        inst_tr = inst_ids[train_idx]
        sw_tr = balanced_sample_weight(y_tr.values, base=uniq[train_idx])
        t_tr = t.iloc[train_idx]
        X_te = X.iloc[test_idx].copy()
        y_te = y.iloc[test_idx]
        inst_te = inst_ids[test_idx]
        if nan_cols_at_test:
            for col in nan_cols_at_test:
                if col in X_te.columns:
                    X_te[col] = np.nan

        m = MultiTaskMetaClassifier(
            n_instruments=len(pool_members),
            config=MultiTaskConfig(seed=cfg.seed, val_frac=0.2,
                                     n_epochs=100, early_stop_patience=15),
        )
        try:
            m.fit(X_tr, y_tr, instrument_ids=inst_tr, sample_weight=sw_tr,
                  t_signal_for_split=t_tr)
            proba = m.predict_act_proba(X_te, instrument_ids=inst_te)
        except Exception as exc:
            continue

        # Record OOS rows for the target instrument.
        for j, (event_idx, prob, label) in enumerate(zip(test_idx, proba, y_te.values)):
            if instruments.iloc[event_idx] != instrument:
                continue
            oos_rows.append({
                "fold": fold_id,
                "row_idx": int(event_idx),
                "y_true": int(label),
                "y_proba": float(prob),
                "sample_weight": float(uniq[event_idx]),
                "instrument": instrument,
            })

    if not oos_rows:
        return CandidateResult(
            instrument=instrument, pool=pool, model_name="multitask_nn",
            n_modelling=len(pool_df),
        )
    oos_df = pd.DataFrame(oos_rows)
    # Per-fold AUC on this instrument's slices.
    from sklearn.metrics import roc_auc_score
    per_fold = []
    for fold_id, grp in oos_df.groupby("fold"):
        if grp["y_true"].nunique() < 2:
            continue
        try:
            a = roc_auc_score(grp["y_true"], grp["y_proba"], sample_weight=grp["sample_weight"])
            per_fold.append(a)
        except Exception:
            pass
    if not per_fold:
        return CandidateResult(
            instrument=instrument, pool=pool, model_name="multitask_nn",
            n_modelling=len(pool_df),
            n_oos_for_instrument=len(oos_df),
            oos_predictions=oos_df,
        )
    return CandidateResult(
        instrument=instrument, pool=pool, model_name="multitask_nn",
        mean_auc=float(np.mean(per_fold)),
        std_auc=float(np.std(per_fold)),
        sem=float(np.std(per_fold) / max(np.sqrt(len(per_fold)), 1.0)),
        n_modelling=len(pool_df),
        n_oos_for_instrument=len(oos_df),
        oos_predictions=oos_df,
        feature_cols=feature_cols,
    )


def _evaluate_candidate(
    *,
    instrument: str,
    pool: str,
    model_name: str,
    pool_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: PipelineConfig,
    nan_cols_at_test: list[str] | None,
    embargo_map: dict[str, int],
) -> CandidateResult:
    """Run one (pool, model) candidate under CPCV(6,2), return its OOS slice for the instrument."""
    if pool_df.empty or len(pool_df) < 40:
        return CandidateResult(instrument=instrument, pool=pool, model_name=model_name)

    pool_df = _add_instrument_onehot(pool_df)
    feature_cols_with_onehot = feature_cols + [c for c in pool_df.columns if c.startswith("inst_")]

    X = pool_df.loc[:, feature_cols_with_onehot].copy()
    y = pool_df["label"].astype(int)
    t = pd.to_datetime(pool_df["t_signal"])
    t1 = pd.to_datetime(pool_df["t_end"])
    instruments = pool_df["instrument"]
    uniq = pool_df["uniqueness_weight"].astype(float)

    cv = CombinatorialPurgedCV(
        n_groups=cfg.cpcv_n_groups,
        n_test_groups=cfg.cpcv_n_test_groups,
        t=t,
        t1=t1,
        pct_embargo=cfg.cpcv_pct_embargo,
        instruments=instruments,
        embargo_days=embargo_map,
    )

    from stml.experimental.pipeline import _fresh_estimator

    def _factory(name=model_name):
        # Per-instrument modelling: stronger regularisation to fight overfit.
        if name == "xgboost":
            return _fresh_estimator(
                "xgboost",
                seed=cfg.seed,
                xgb_overrides=dict(max_depth=3, n_estimators=100, reg_lambda=2.0, reg_alpha=0.5),
            )
        return _fresh_estimator(name, seed=cfg.seed)

    result = cross_val_evaluate(
        make_model=_factory,
        X=X, y=y, cv=cv,
        uniqueness_weights=uniq,
        nan_columns_at_test=nan_cols_at_test,
    )

    # Filter the OOS predictions to the target instrument only and compute its AUC.
    inst_oos = _filter_oos_to_instrument(result.oos_predictions, instruments, instrument)

    if inst_oos.empty or inst_oos["y_true"].nunique() < 2:
        return CandidateResult(
            instrument=instrument, pool=pool, model_name=model_name,
            n_modelling=int(len(pool_df)),
            n_oos_for_instrument=int(len(inst_oos)),
        )

    # Per-fold AUC ON THE INSTRUMENT'S OOS ROWS — alken-style "fair" metric.
    from sklearn.metrics import roc_auc_score
    per_fold = []
    for fold_id, grp in inst_oos.groupby("fold"):
        if grp["y_true"].nunique() < 2:
            continue
        try:
            a = roc_auc_score(
                grp["y_true"], grp["y_proba"], sample_weight=grp["sample_weight"]
            )
            per_fold.append(a)
        except Exception:
            pass
    if not per_fold:
        return CandidateResult(
            instrument=instrument, pool=pool, model_name=model_name,
            n_modelling=int(len(pool_df)),
            n_oos_for_instrument=int(len(inst_oos)),
        )
    mean_auc = float(np.mean(per_fold))
    std_auc = float(np.std(per_fold))
    sem = std_auc / max(np.sqrt(len(per_fold)), 1.0)

    return CandidateResult(
        instrument=instrument,
        pool=pool,
        model_name=model_name,
        mean_auc=mean_auc,
        std_auc=std_auc,
        sem=sem,
        n_modelling=int(len(pool_df)),
        n_oos_for_instrument=int(len(inst_oos)),
        oos_predictions=inst_oos,
        feature_cols=feature_cols,
    )


def select_champion(candidates: list[CandidateResult]) -> ChampionResult:
    """Plan §4.19 1SE rule — pick the most regularised within 1 SE of best.

    Ordering for "most regularised" (least flexible first):
        elasticnet_logistic > random_forest > lightgbm > xgboost
        smaller pool (individual) > larger pool (asset class)
    """
    instrument = candidates[0].instrument if candidates else "?"
    valid = [c for c in candidates if not np.isnan(c.mean_auc)]
    if not valid:
        return ChampionResult(
            instrument=instrument,
            winning_pool="(none)",
            winning_model="(none)",
            champion_auc=float("nan"),
            champion_sem=float("nan"),
            candidates=candidates,
            selection_reason="no valid candidates",
        )
    best = max(valid, key=lambda c: c.mean_auc)
    threshold = best.mean_auc - (best.sem if not np.isnan(best.sem) else 0.0)

    # Models, more-regularised → less.
    # The multi-task NN sits at the most-complex end because it has many
    # parameters; we prefer simpler models when they're within 1 SE.
    model_rank = {
        "elasticnet_logistic": 0,
        "random_forest": 1,
        "lightgbm": 2,
        "xgboost": 3,
        "multitask_nn": 4,
    }
    # Pools, smaller → larger.
    def _pool_size(pool: str) -> int:
        return len(POOL_MEMBERS.get(pool, ()))

    within_se = [c for c in valid if c.mean_auc >= threshold]
    within_se.sort(key=lambda c: (model_rank.get(c.model_name, 99), _pool_size(c.pool)))
    pick = within_se[0]
    reason = (
        "best mean AUC"
        if pick == best
        else f"1SE pick (best={best.model_name}@{best.pool} {best.mean_auc:.4f}±{best.sem:.4f})"
    )

    return ChampionResult(
        instrument=instrument,
        winning_pool=pick.pool,
        winning_model=pick.model_name,
        champion_auc=pick.mean_auc,
        champion_sem=pick.sem,
        candidates=candidates,
        selection_reason=reason,
    )


def run_champion_for_instrument(
    *,
    instrument: str,
    features: pd.DataFrame,
    cfg: PipelineConfig,
    variant: str,
    embargo_map: dict[str, int],
    nan_cols_at_test: list[str] | None,
) -> ChampionResult:
    """Evaluate all (pool, model) candidates for ``instrument`` and pick champion."""
    pools = INSTRUMENT_REGIMES.get(instrument, [instrument])
    feature_cols = _select_feature_cols(features, variant)

    candidates: list[CandidateResult] = []
    for pool in pools:
        pool_df = _slice_pool(features, pool)
        # Skip individual pools that are too thin.
        if len(POOL_MEMBERS[pool]) == 1 and len(pool_df) < MIN_INDIVIDUAL_EVENTS:
            continue
        # nan columns at test for the R-11 ablation.
        nan_at_test = None
        if variant == "simulated_missingness":
            nan_at_test = [c for c in feature_cols if _is_bbg_feature(c)]
        elif nan_cols_at_test is not None:
            nan_at_test = nan_cols_at_test

        for model_name in ("elasticnet_logistic", "random_forest", "lightgbm", "xgboost"):
            c = _evaluate_candidate(
                instrument=instrument,
                pool=pool,
                model_name=model_name,
                pool_df=pool_df,
                feature_cols=feature_cols,
                cfg=cfg,
                nan_cols_at_test=nan_at_test,
                embargo_map=embargo_map,
            )
            candidates.append(c)

        # Multi-task NN — only on multi-instrument pools (sharing makes no
        # sense for individual pools).
        if MULTITASK_AVAILABLE and len(POOL_MEMBERS[pool]) >= 2:
            mt = _evaluate_multitask_candidate(
                instrument=instrument,
                pool=pool,
                pool_df=pool_df,
                feature_cols=feature_cols,
                cfg=cfg,
                nan_cols_at_test=nan_at_test,
                embargo_map=embargo_map,
            )
            candidates.append(mt)

    return select_champion(candidates)


def run_all_champions(
    features: pd.DataFrame,
    *,
    cfg: PipelineConfig | None = None,
    variant: str = "with_bbg",
    verbose: bool = True,
) -> dict[str, ChampionResult]:
    """Run champion selection for every instrument in INSTRUMENT_REGIMES."""
    cfg = cfg or PipelineConfig()
    embargo_map = embargo_days_map()
    features = _restrict_modelling(features, cfg)

    results: dict[str, ChampionResult] = {}
    for instrument in INSTRUMENT_REGIMES:
        if verbose:
            print(f"\n[champion] {instrument} — pools {INSTRUMENT_REGIMES[instrument]} × 4 models")
        r = run_champion_for_instrument(
            instrument=instrument,
            features=features,
            cfg=cfg,
            variant=variant,
            embargo_map=embargo_map,
            nan_cols_at_test=None,
        )
        results[instrument] = r
        if verbose:
            print(f"    champion = {r.winning_model}@{r.winning_pool}  "
                  f"AUC = {r.champion_auc:.4f} ± {r.champion_sem:.4f}  ({r.selection_reason})")
    return results
