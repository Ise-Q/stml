"""Meta-model prediction layer: triple-barrier meta-labeling + leakage-safe model search.

Consumes the frozen feature-engineering outputs (``results/feature_matrix.parquet`` and the
scope / redundancy registries) read-only, turns each primary signal into a binary
profitable/not-profitable label via the triple-barrier method (:mod:`stml.model.labels`), and
fits & compares tree-based and neural meta-models under purged + embargoed walk-forward
cross-validation (:mod:`stml.model.cv`) with Optuna.

Pipeline order: ``labels`` -> ``dataset`` -> ``cv`` -> ``barrier_search`` -> ``optuna_objective``
(+ ``trees`` / ``mlp`` / ``vsn``) -> ``importance`` -> ``evaluate``.
"""

from stml.model.barrier_search import BarrierResult, search_barriers
from stml.model.cv import PurgedWalkForward
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
    evaluate_predictions,
    per_instrument_breakdown,
    plot_calibration,
    plot_roc_pr,
    release_test,
)
from stml.model.importance import nn_importance, permutation_importance_auc, tree_importance
from stml.model.labels import class_balance, sample_uniqueness, triple_barrier_labels
from stml.model.mlp import MLPModel, mlp_param_space
from stml.model.optuna_objective import MODEL_REGISTRY, cross_val_auc, make_objective, run_study
from stml.model.trees import RFModel, XGBModel, rf_param_space, xgb_param_space
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
    # barrier search
    "search_barriers",
    "BarrierResult",
    # models
    "XGBModel",
    "RFModel",
    "MLPModel",
    "VSNModel",
    "xgb_param_space",
    "rf_param_space",
    "mlp_param_space",
    "vsn_param_space",
    # optuna
    "MODEL_REGISTRY",
    "cross_val_auc",
    "make_objective",
    "run_study",
    # importance
    "tree_importance",
    "nn_importance",
    "permutation_importance_auc",
    # evaluate
    "evaluate_predictions",
    "per_instrument_breakdown",
    "plot_roc_pr",
    "plot_calibration",
    "release_test",
]
