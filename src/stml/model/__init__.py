"""Meta-model prediction layer: triple-barrier meta-labeling + leakage-safe model search.

Consumes the frozen feature-engineering outputs (``results/feature_matrix.parquet`` and the
scope / redundancy registries) read-only, turns each primary signal into a binary
profitable/not-profitable label via the triple-barrier method (:mod:`stml.model.labels`), and
fits & compares linear, tree-based and neural meta-models (one per family) under purged +
embargoed walk-forward cross-validation (:mod:`stml.model.cv`) with Optuna.

Pipeline order: ``labels`` -> ``dataset`` -> ``cv`` -> ``barrier_search`` -> ``optuna_objective``
(+ ``linear`` / ``trees`` / ``mlp`` / ``vsn``) -> ``importance`` -> ``evaluate``.
"""

from stml.model.barrier_search import BarrierResult, search_barriers
from stml.model.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier,
    calibrate_oof,
    calibration_report,
    reliability_table,
)
from stml.model.cv import CombinatorialPurgedCV, PurgedWalkForward
from stml.model.dataset import (
    Preprocessor,
    asset_class_map,
    attach_bar_pos,
    close_panel,
    embargo_map,
    events_frame,
    load_matrix,
    load_scope,
    make_xy,
    scope_iter,
    select_features,
)
from stml.model.evaluate import (
    adding_zeros_eval,
    bootstrap_returns,
    decision_threshold,
    evaluate_predictions,
    nav_sharpe,
    per_instrument_breakdown,
    per_instrument_vs_baseline,
    plot_calibration,
    plot_roc_pr,
    predictions_grid,
    release_test,
)
from stml.model.importance import (
    clustered_mda,
    clustered_mdi,
    clustered_shap,
    iterative_cluster_drop,
    load_feature_clusters,
    map_model_features_to_clusters,
    nn_importance,
    noise_injection_check,
    permutation_importance_auc,
    single_feature_importance,
    tree_importance,
)
from stml.model.labels import class_balance, sample_uniqueness, triple_barrier_labels
from stml.model.linear import LogRegModel, logreg_param_space
from stml.model.mlp import MLPModel, mlp_param_space
from stml.model.optuna_objective import (
    MODEL_REGISTRY,
    cpcv_oof_auc,
    cross_val_auc,
    make_objective,
    make_objective_cpcv,
    run_study,
    two_stage_select,
)
from stml.model.trees import (
    HAS_LIGHTGBM,
    AdaBoostModel,
    LGBMModel,
    RFModel,
    XGBModel,
    adaboost_param_space,
    lgbm_param_space,
    rf_param_space,
    xgb_param_space,
)
from stml.model.vsn import VSNModel, vsn_param_space

__all__ = [
    # labels
    "triple_barrier_labels",
    "sample_uniqueness",
    "class_balance",
    # dataset
    "load_matrix",
    "load_scope",
    "asset_class_map",
    "embargo_map",
    "close_panel",
    "attach_bar_pos",
    "events_frame",
    "select_features",
    "make_xy",
    "scope_iter",
    "Preprocessor",
    # cv
    "PurgedWalkForward",
    "CombinatorialPurgedCV",
    # barrier search
    "search_barriers",
    "BarrierResult",
    # models (linear / tree / neural)
    "LogRegModel",
    "XGBModel",
    "RFModel",
    "AdaBoostModel",
    "LGBMModel",
    "MLPModel",
    "VSNModel",
    "HAS_LIGHTGBM",
    "logreg_param_space",
    "xgb_param_space",
    "rf_param_space",
    "adaboost_param_space",
    "lgbm_param_space",
    "mlp_param_space",
    "vsn_param_space",
    # optuna + CPCV scoring / two-stage selection
    "MODEL_REGISTRY",
    "cross_val_auc",
    "make_objective",
    "run_study",
    "cpcv_oof_auc",
    "make_objective_cpcv",
    "two_stage_select",
    # calibration
    "PlattCalibrator",
    "IsotonicCalibrator",
    "calibrate_oof",
    "reliability_table",
    "brier",
    "calibration_report",
    # importance (incl. cluster-level)
    "tree_importance",
    "nn_importance",
    "permutation_importance_auc",
    "load_feature_clusters",
    "map_model_features_to_clusters",
    "clustered_mdi",
    "clustered_mda",
    "clustered_shap",
    "single_feature_importance",
    "noise_injection_check",
    "iterative_cluster_drop",
    # evaluate (incl. economic eval + deliverable grid)
    "evaluate_predictions",
    "per_instrument_breakdown",
    "per_instrument_vs_baseline",
    "plot_roc_pr",
    "plot_calibration",
    "release_test",
    "bootstrap_returns",
    "decision_threshold",
    "adding_zeros_eval",
    "nav_sharpe",
    "predictions_grid",
]
