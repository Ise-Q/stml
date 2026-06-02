"""Two-stage HP selection (plan item 6): mean path-AUC (Stage 1) then lowest Sharpe-spread
(Stage 2, ``std(path_AUC)`` -- lower is more robust). Hand-built studies pin the formula AND the
direction so a lucky high-variance spike cannot beat a consistent plateau.
"""

from __future__ import annotations

import optuna
import pytest

from stml.model.optuna_objective import two_stage_select

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _study_from_specs(specs: list[tuple[float, float]]) -> optuna.Study:
    """Build a finished study whose trial i returns ``specs[i][0]`` with ``path_auc_std`` set."""
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial: optuna.Trial) -> float:
        mean, std = specs[trial.number]
        trial.set_user_attr("path_auc_std", std)
        return mean

    study.optimize(objective, n_trials=len(specs))
    return study


def test_stage1_keeps_top_frac_then_stage2_min_spread():
    # means: 0.70 0.69 0.68 0.55 0.50 ; top_frac 0.5 of 5 -> ceil = 3 survivors: the 0.70/0.69/0.68
    # spreads among survivors: 0.010 0.005 0.020 -> lowest is the 0.69 trial (std 0.005)
    specs = [(0.70, 0.010), (0.69, 0.005), (0.68, 0.020), (0.55, 0.001), (0.50, 0.100)]
    best = two_stage_select(_study_from_specs(specs), top_frac=0.5)
    assert best.value == pytest.approx(0.69)
    assert best.user_attrs["path_auc_std"] == pytest.approx(0.005)


def test_lucky_spike_loses_to_robust_plateau():
    # the highest-mean trial (0.72) is also the highest-variance (0.09); a slightly lower-mean but
    # tight trial (0.71, std 0.004) must win once both survive Stage 1.
    specs = [(0.72, 0.090), (0.71, 0.004), (0.60, 0.050), (0.58, 0.002)]
    best = two_stage_select(_study_from_specs(specs), top_frac=0.5)  # ceil(0.5*4)=2 -> 0.72,0.71
    assert best.value == pytest.approx(0.71)


def test_tie_on_spread_breaks_to_higher_mean():
    specs = [(0.70, 0.01), (0.66, 0.01), (0.50, 0.20)]  # top_frac 1.0 -> all; equal spread on top 2
    best = two_stage_select(_study_from_specs(specs), top_frac=1.0)
    assert best.value == pytest.approx(0.70)


def test_raises_with_no_scorable_trials():
    study = optuna.create_study(direction="maximize")
    with pytest.raises(ValueError):
        two_stage_select(study)
