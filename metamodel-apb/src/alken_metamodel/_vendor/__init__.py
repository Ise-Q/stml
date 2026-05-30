"""Vendored, runnable course modules — provenance and patch log.

These files are copied **verbatim** from the ``sts-ml`` skill
(``~/.claude/skills/sts-ml/scripts/``, the T3.03 course archive) so the
metamodel subproject runs cold (the skill directory is not part of the repo the
grader re-runs). They are kept byte-identical to source on first import; the
known bug fixes are applied on top in a separate commit so each fix is visible
in the diff (brief: "show each bug fix").

| file | source | role | pending fixes (Stage 3) |
|------|--------|------|--------------------------|
| ``vsn.py`` | sts-ml/scripts/vsn.py | Keras VSN (GLU/GRN/InputTransformation/VariableSelectionNetwork) — the Keras neural variant | — |
| ``cluster_feature_importance.py`` | sts-ml/scripts/cluster_feature_importance.py | OptimalClusterer + Spearman distance + cluster MDI/PFI | #4 distance ``1-|rho|`` -> Mantegna ``sqrt(1-|rho|)`` (line ~30); #2 ``KFold(shuffle=True)`` -> PurgedKFold (line ~105) |
| ``trend_scanning.py`` | sts-ml/scripts/trend_scanning.py | ``tValLinR``, ``trend_labels`` — used as a backward-trend *feature* (``look_forward=False``), not the label | — |
| ``regression_metrics.py`` | sts-ml/scripts/regression_metrics.py | MAE/MSE/RMSE/MAPE/R2 report | — |

Import note: ``vsn.py`` imports TensorFlow (slow). This package ``__init__`` does
**not** import the modules eagerly — import the specific vendored module where needed.
"""
