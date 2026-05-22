"""
search.py
=========
Tiered, guided hyperparameter search for one archetype on one *cell* (US-010).

A *cell* is whatever unit the caller decides to replicate -- a standalone
instrument or a pooled asset class. :func:`search_cell` stays **agnostic** to
that choice: the caller supplies a ``generate_fn(params) -> pd.Series`` that
returns a replica signal already aligned to ``target``'s index (for a standalone
instrument it wraps :meth:`Archetype.generate` on that instrument; for a pool it
concatenates per-instrument replicas). search.py never touches OHLCV directly.

Objective: train discrepancy ONLY
----------------------------------
The :func:`composite_skill` of a replica against the target is the mean of two
**chance-corrected** metrics from :func:`stml.replication.metrics.panel`:

    composite_skill = mean( panel(...)["kappa"],
                            panel(...)["ordinal_skill"]["vs_flat"] )

Both summands are 0 for any marginal-only guess (always-flat, majority,
stratified-random) and 1 for a perfect replica, so their mean inherits the same
chance-correction. The optimiser minimises the discrepancy

    discrepancy = 1 - composite_skill              (lower = better replica)

computed **on the train slice only** (``target.iloc[train_idx]`` vs
``replica.iloc[train_idx]``). Selection is by this train objective. Validation
metrics are computed via :func:`metrics.panel` on ``val_idx`` and recorded to the
ledger (and returned as ``best_metrics``) purely for downstream gating /
reporting -- they NEVER drive selection, which keeps validation a genuine
out-of-sample generalisation check. The **test** split is never read: there is
no call path from :func:`search_cell` to :func:`stml.replication.splits.get_test`.

Tiered by effective sample size
-------------------------------
``n_eff`` is the post-embargo number of independent regime-calls
(:func:`stml.replication.splits.n_eff`). The search tier is chosen from it:

* ``n_eff >= FLOOR`` (=10): **Optuna TPE** -- a seeded
  :class:`optuna.samplers.TPESampler` minimises the train discrepancy over a
  frozen ``budget`` of trials. Each parameter is suggested categorically from
  its ``param_space`` list, so the search space matches the discrete archetype
  grid and the seeded sampler is exactly reproducible.
* ``n_eff < FLOOR``: **coarse deterministic grid** -- every point in the
  Cartesian product of the ``param_space`` lists is evaluated and the minimum
  train discrepancy wins. Deterministic by construction (no sampler), the honest
  choice when there are too few regime-calls to justify a guided search.

Read-before-propose
-------------------
Before searching, :func:`search_cell` consults
``ledger.prior_trials(archetype, cell)`` and threads a ``motivated_by`` note
(the ids of prior best trials, or a first-iteration marker) into every recorded
trial, so the iterative learning loop stays auditable. Every evaluated config is
recorded to the ledger with its ``tier``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import optuna
import pandas as pd

from stml.replication import metrics
from stml.replication.ledger import Ledger

__all__ = ["FLOOR", "composite_skill", "discrepancy", "search_cell"]

# Below this many post-embargo regime-calls a guided (TPE) search is not
# justified; fall back to an exhaustive deterministic grid (mirrors gates.FLOOR
# and the CONTRACT pooling floor).
FLOOR: int = 10

# Keep Optuna quiet: the per-trial INFO logging is noise for a frozen-budget
# study driven from inside a larger pipeline.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _composite_one(target: pd.Series, replica: pd.Series) -> float:
    """The single-series composite ``mean(kappa, ordinal_skill_vs_flat)``.

    The original (group-agnostic) skill: one :func:`metrics.panel` over the whole
    aligned ``(target, replica)`` pair. Used directly for standalone cells and as
    the per-group kernel of the group-averaged variant below.
    """
    panel = metrics.panel(target.to_numpy(), replica.to_numpy())
    kappa = float(panel["kappa"])
    ordinal = float(panel["ordinal_skill"]["vs_flat"])
    return 0.5 * (kappa + ordinal)


def composite_skill(
    target: pd.Series,
    replica: pd.Series,
    groups: Sequence | np.ndarray | None = None,
) -> float:
    """Chance-corrected replication skill of ``replica`` against ``target``.

    The mean of two :func:`stml.replication.metrics.panel` co-primary metrics,
    both chance-corrected so a marginal-only predictor scores ~0 and a perfect
    replica scores 1:

    * ``kappa`` -- Cohen's kappa (agreement beyond chance);
    * ``ordinal_skill["vs_flat"]`` -- the chance-corrected ordinal skill score,
      which weights a full sign flip (``-1 <-> +1``) as twice as costly as a move
      to/from flat.

    Both inputs are coerced to integer label arrays by :func:`metrics.panel`.

    Parameters
    ----------
    target, replica : pd.Series
        Equal-length label series over ``{-1, 0, +1}`` (already restricted to the
        slice being scored). Their values are compared positionally.
    groups : array-like or None
        Optional per-row group label aligned to ``target`` / ``replica`` (e.g. the
        member-instrument id for a concatenated pool). When provided, the
        composite is computed **per unique group** and the equal-weight MEAN across
        groups is returned -- a WITHIN-instrument skill measure that a single
        concatenated panel cannot give. Concatenating members and scoring one panel
        lets cross-instrument base-rate matching inflate kappa (a momentum replica
        that anti-replicates two of three energy members still scored a positive
        pooled kappa); group-averaging removes that artifact. ``None`` (the default)
        reproduces the original single-panel skill exactly, so the standalone path
        is unchanged.

    Returns
    -------
    float
        ``mean(kappa, ordinal_skill_vs_flat)`` -- over the whole pair when
        ``groups is None``, else the equal-weight mean of that quantity computed
        per group.
    """
    if groups is None:
        return _composite_one(target, replica)
    return _group_mean(target, replica, groups, _composite_one)


def discrepancy(
    target: pd.Series,
    replica: pd.Series,
    groups: Sequence | np.ndarray | None = None,
) -> float:
    """Replication discrepancy ``1 - composite_skill`` (lower = better replica).

    A perfect replica scores ``composite_skill == 1`` and so ``discrepancy == 0``;
    a chance-level replica scores ``~0`` skill and so ``discrepancy ~ 1``. This is
    the quantity the search MINIMISES on the train slice. ``groups`` is threaded
    through to :func:`composite_skill`: a pooled cell minimises the MEAN per-member
    (within-instrument) discrepancy, never the concatenated one.
    """
    return 1.0 - composite_skill(target, replica, groups)


def _group_mean(
    target: pd.Series,
    replica: pd.Series,
    groups: Sequence | np.ndarray,
    kernel: Callable[[pd.Series, pd.Series], float],
) -> float:
    """Equal-weight mean of ``kernel(target_g, replica_g)`` over unique groups.

    ``groups`` is positionally aligned to ``target`` / ``replica`` (one label per
    row). Each unique group's rows are sliced out POSITIONALLY (so the kernel sees
    a within-instrument pair) and scored, then the per-group scores are averaged
    with equal weight -- a member with more rows does not dominate. Groups are
    visited in first-appearance order so the average is deterministic. An empty
    input yields ``0.0`` (no group to score), matching the chance-level convention
    of a marginal-only guess.
    """
    grp = np.asarray(groups)
    if grp.shape[0] != len(target):
        raise ValueError(
            f"groups length {grp.shape[0]} != target length {len(target)}"
        )
    if grp.size == 0:
        return 0.0
    # First-appearance order keeps the equal-weight mean deterministic.
    _, first_idx = np.unique(grp, return_index=True)
    unique_in_order = grp[np.sort(first_idx)]
    scores: list[float] = []
    for g in unique_in_order:
        mask = grp == g
        scores.append(kernel(target.iloc[mask], replica.iloc[mask]))
    return float(np.mean(scores))


def _train_discrepancy(
    generate_fn: Callable[[dict], pd.Series],
    target: pd.Series,
    train_idx: Sequence[int] | np.ndarray,
    params: dict,
    groups: Sequence | np.ndarray | None = None,
) -> tuple[float, pd.Series]:
    """Train-slice discrepancy for one parameter set (the search objective).

    Builds the replica via ``generate_fn(params)`` (the caller wires it to an
    archetype on an instrument or a pool), positionally slices both ``target``
    and the replica to ``train_idx``, and returns ``(discrepancy, replica)``. The
    full replica is returned too so the caller can re-slice it for val metrics
    without regenerating. Only ``train_idx`` rows enter the objective; ``test`` is
    never indexed. When ``groups`` (aligned to the FULL ``target``) is supplied,
    the train-slice ``groups`` are sliced alongside so a pooled cell minimises the
    MEAN per-member discrepancy, not the concatenated one.
    """
    replica = generate_fn(params)
    train_list = list(train_idx)
    tgt_train = target.iloc[train_list]
    rep_train = replica.iloc[train_list]
    grp_train = None if groups is None else np.asarray(groups)[train_list]
    return discrepancy(tgt_train, rep_train, grp_train), replica


def _val_metrics(
    target: pd.Series,
    replica: pd.Series,
    val_idx: Sequence[int] | np.ndarray,
    groups: Sequence | np.ndarray | None = None,
) -> dict:
    """Full :func:`metrics.panel` on the val slice (for the ledger / reporting).

    Recorded and returned as ``best_metrics`` but NEVER used for selection, so
    validation stays a genuine out-of-sample check. An empty ``val_idx`` (a thin
    embargoed window) yields an empty dict rather than raising. When ``groups``
    (aligned to the FULL ``target``) is supplied, the scalar co-primary metrics
    are reported as the MEAN of the per-group panels (within-instrument
    aggregation) so the recorded pool ``kappa`` / ``ordinal_skill`` are the
    per-member averages, never the concatenated values.
    """
    val_list = list(val_idx)
    if not val_list:
        return {}
    if groups is None:
        return metrics.panel(
            target.iloc[val_list].to_numpy(), replica.iloc[val_list].to_numpy()
        )
    return _group_mean_panel(
        target.iloc[val_list],
        replica.iloc[val_list],
        np.asarray(groups)[val_list],
    )


def _group_mean_panel(
    target: pd.Series,
    replica: pd.Series,
    groups: Sequence | np.ndarray,
) -> dict:
    """A panel-shaped dict whose co-primary metrics are per-group means.

    For a pooled cell the recorded ``val_metrics`` must reflect WITHIN-instrument
    skill, so ``kappa`` and ``ordinal_skill['vs_flat']`` are the equal-weight mean
    of each member's own :func:`metrics.panel` value. The dict mirrors the keys a
    single :func:`metrics.panel` produces that downstream code reads (``kappa``,
    ``ordinal_skill``); the ``confusion`` matrix is meaningless across members and
    is omitted (the ledger drops it anyway). ``n_groups`` records how many members
    were averaged.
    """
    grp = np.asarray(groups)
    _, first_idx = np.unique(grp, return_index=True)
    unique_in_order = grp[np.sort(first_idx)]
    kappas: list[float] = []
    ordinals: list[float] = []
    for g in unique_in_order:
        mask = grp == g
        p = metrics.panel(
            target.iloc[mask].to_numpy(), replica.iloc[mask].to_numpy()
        )
        kappas.append(float(p["kappa"]))
        ordinals.append(float(p["ordinal_skill"]["vs_flat"]))
    return {
        "kappa": float(np.mean(kappas)),
        "ordinal_skill": {"vs_flat": float(np.mean(ordinals))},
        "n_groups": int(unique_in_order.size),
    }


def _grid_points(param_space: dict[str, list]) -> list[dict]:
    """Every point in the Cartesian product of the ``param_space`` lists.

    Iteration order follows ``param_space`` key/value order, so the enumeration
    -- and therefore the grid winner under ties -- is deterministic.
    """
    keys = list(param_space.keys())
    value_lists = [list(param_space[k]) for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


def _suggest_params(trial: optuna.Trial, param_space: dict[str, list]) -> dict:
    """Suggest one config from ``param_space`` via categorical choices.

    Every archetype axis is a discrete list of candidates, so categorical
    suggestion matches the search space exactly and a seeded
    :class:`optuna.samplers.TPESampler` reproduces the same proposals run over
    run.
    """
    return {
        name: trial.suggest_categorical(name, list(choices))
        for name, choices in param_space.items()
    }


def _motivation(prior: list[dict]) -> list:
    """A ``motivated_by`` note from prior trials (read-before-propose).

    Returns the ids of up to the three most-skillful prior trials (by recorded
    ``composite_skill``, else lowest ``discrepancy``), so each new proposal links
    back to what motivated the cell's search. A first-iteration cell with no
    history returns a single explanatory marker string.
    """
    if not prior:
        return ["seed: no prior trials for this cell"]

    def _key(t: dict) -> float:
        vm = t.get("val_metrics") or {}
        if isinstance(vm.get("composite_skill"), (int, float)):
            return -float(vm["composite_skill"])
        if isinstance(t.get("train_discrepancy"), (int, float)):
            return float(t["train_discrepancy"])
        return math.inf

    ranked = sorted(prior, key=_key)
    return [t["id"] for t in ranked[:3] if "id" in t]


def _record_trial(
    *,
    ledger: Ledger,
    archetype: str,
    cell: str,
    tier: str,
    params: dict,
    train_disc: float,
    val_metrics: dict,
    motivated_by: list,
) -> dict:
    """Append one evaluated config to the ledger with its tier and provenance.

    The recorded ``val_metrics`` is augmented with ``composite_skill`` (so a later
    iteration's read-before-propose can rank prior trials) and the trial carries
    ``train_discrepancy`` (the actual selection objective) alongside the standard
    ledger fields.
    """
    vm = _ledger_safe_metrics(val_metrics)
    if vm:
        vm["composite_skill"] = _safe_val_composite(vm)
    return ledger.record(
        {
            "archetype": archetype,
            "cell": cell,
            "tier": tier,
            "params": dict(params),
            "val_metrics": vm,
            "train_discrepancy": float(train_disc),
            "gate_result": {},
            "motivated_by": list(motivated_by),
        }
    )


def _ledger_safe_metrics(val_metrics: dict) -> dict:
    """A JSON-serialisable view of a :func:`metrics.panel` dict for the ledger.

    :func:`metrics.panel` returns a ``confusion`` 2-D ndarray, which the ledger's
    scalar-oriented JSON encoder cannot serialise (it ``.item()``-coerces numpy
    scalars). The confusion matrix is a reporting artifact, not part of the
    auditable trial schema (the ledger renders only ``kappa`` /
    ``ordinal_skill``), so it is dropped from the recorded copy; the full panel is
    still returned to the caller as ``best_metrics``. The returned dict shares no
    mutable state with ``val_metrics``.
    """
    return {k: v for k, v in val_metrics.items() if k != "confusion"}


def _safe_val_composite(val_metrics: dict) -> float | None:
    """Composite skill from a recorded val ``metrics.panel`` dict, or ``None``.

    Mirrors :func:`composite_skill` but reads from an already-computed panel dict
    (the ledger stores panels, not series), so prior-trial ranking can reuse the
    val skill without recomputation.
    """
    kappa = val_metrics.get("kappa")
    ordinal = val_metrics.get("ordinal_skill")
    if isinstance(ordinal, dict):
        ordinal = ordinal.get("vs_flat")
    if isinstance(kappa, (int, float)) and isinstance(ordinal, (int, float)):
        return 0.5 * (float(kappa) + float(ordinal))
    return None


def search_cell(
    *,
    archetype: str,
    cell: str,
    param_space: dict[str, list],
    generate_fn: Callable[[dict], pd.Series],
    target: pd.Series,
    train_idx: Sequence[int] | np.ndarray,
    val_idx: Sequence[int] | np.ndarray,
    n_eff: int,
    ledger: Ledger,
    budget: int = 64,
    seed: int = 0,
    groups: Sequence | np.ndarray | None = None,
) -> dict:
    """Search one archetype's params on one cell, minimising train discrepancy.

    The tier is chosen from ``n_eff``: ``>= FLOOR`` runs a seeded Optuna TPE study
    for ``budget`` trials; below the floor runs an exhaustive deterministic grid
    over ``param_space``. Either way the objective is the **train**-slice
    discrepancy (:func:`discrepancy` on ``target.iloc[train_idx]`` vs the replica),
    selection is by that objective, and validation metrics are computed +
    recorded only for the ledger / downstream gating. The test split is never
    touched.

    Parameters
    ----------
    archetype, cell : str
        Identifiers recorded with every ledger trial (e.g. ``"mean_reversion"``,
        ``"es1s"`` or ``"pool:energy"``). search.py does not interpret them.
    param_space : dict[str, list]
        ``param -> candidate values`` (from :meth:`Archetype.param_space`). Each
        axis is a discrete list; TPE suggests categorically and the grid takes the
        Cartesian product.
    generate_fn : callable
        ``params -> pd.Series``: a replica signal in ``{-1, 0, +1}`` aligned to
        ``target``'s index. The caller wires this to an archetype on an instrument
        or to a pool concatenation; search.py stays agnostic.
    target : pd.Series
        The target signal aligned to the same index ``generate_fn`` returns.
    train_idx, val_idx : sequence of int
        Positional indices into ``target`` / the replica for the train and val
        slices. ``train_idx`` defines the objective; ``val_idx`` is reporting-only
        and may be empty.
    n_eff : int
        Post-embargo effective sample size; selects the tier against ``FLOOR``.
    ledger : Ledger
        Receives every evaluated config (with ``tier`` and ``motivated_by``).
    budget : int
        Number of TPE trials (ignored on the grid tier, which is exhaustive).
    seed : int
        Seeds the TPE sampler so the search is reproducible.
    groups : array-like or None
        Optional per-row group label aligned to the FULL ``target`` (one label per
        row; e.g. the member-instrument id for a concatenated pool). When provided,
        the train objective minimises the MEAN per-group (within-instrument)
        discrepancy and the recorded val metrics are likewise group-averaged, so a
        pooled cell can never pass on cross-instrument base-rate matching. ``None``
        (the default, used by every standalone cell) leaves the single-series
        behaviour exactly as before.

    Returns
    -------
    dict with keys
        ``best_params``      : the params minimising train discrepancy.
        ``best_metrics``     : :func:`metrics.panel` on the val slice for the
                               winner (reporting/gating only; ``{}`` if no val).
                               When ``groups`` is given this is the group-averaged
                               (within-instrument) panel summary instead.
        ``best_discrepancy`` : the winning **train** discrepancy.
        ``tier``             : ``"tpe"`` or ``"grid"``.
        ``trials``           : the recorded ledger trial dicts (in eval order).

    Raises
    ------
    ValueError
        If ``param_space`` is empty or ``budget`` is non-positive on the TPE tier.
    """
    if not param_space:
        raise ValueError("param_space must be non-empty")

    prior = ledger.prior_trials(archetype, cell)
    motivated_by = _motivation(prior)

    tier = "tpe" if n_eff >= FLOOR else "grid"
    recorded: list[dict] = []

    best_params: dict | None = None
    best_disc = math.inf
    best_replica: pd.Series | None = None

    if tier == "grid":
        # Exhaustive, deterministic: evaluate every grid point, keep the min. The
        # product order is fixed, so ties break to the first-enumerated config.
        for params in _grid_points(param_space):
            train_disc, replica = _train_discrepancy(
                generate_fn, target, train_idx, params, groups
            )
            vm = _val_metrics(target, replica, val_idx, groups)
            recorded.append(
                _record_trial(
                    ledger=ledger,
                    archetype=archetype,
                    cell=cell,
                    tier=tier,
                    params=params,
                    train_disc=train_disc,
                    val_metrics=vm,
                    motivated_by=motivated_by,
                )
            )
            if train_disc < best_disc:
                best_disc = train_disc
                best_params = params
                best_replica = replica
    else:
        if budget <= 0:
            raise ValueError(f"budget must be positive for TPE, got {budget}")
        # Seeded TPE over the discrete grid; reproducible for fixed seed+budget.
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        # The objective records each evaluated config to the ledger as a side
        # effect, so the ledger captures the full TPE trajectory (not just best).
        def _objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial, param_space)
            train_disc, replica = _train_discrepancy(
                generate_fn, target, train_idx, params, groups
            )
            vm = _val_metrics(target, replica, val_idx, groups)
            recorded.append(
                _record_trial(
                    ledger=ledger,
                    archetype=archetype,
                    cell=cell,
                    tier=tier,
                    params=params,
                    train_disc=train_disc,
                    val_metrics=vm,
                    motivated_by=motivated_by,
                )
            )
            # Stash so the winning replica's val metrics need no regeneration.
            trial.set_user_attr("train_disc", float(train_disc))
            return train_disc

        study.optimize(_objective, n_trials=budget)

        best_params = dict(study.best_params)
        best_disc = float(study.best_value)
        # Regenerate the winner's replica once to compute its val metrics.
        best_replica = generate_fn(best_params)

    assert best_params is not None and best_replica is not None  # noqa: S101
    best_metrics = _val_metrics(target, best_replica, val_idx, groups)

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_discrepancy": best_disc,
        "tier": tier,
        "trials": recorded,
    }
