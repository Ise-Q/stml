"""
gates.py
========
Robustness gates (US-008) for the primary-signal replication search.

A candidate replica only "passes" if it clears *four* independent gates on the
TRAIN+VAL window (the held-out test block is never touched here -- see
:func:`stml.replication.splits.get_test`). The four gates attack the four ways a
replica can look good for the wrong reason:

* **G1 beats-baseline (+ multiplicity).** The replica must beat the *chance*
  cutoffs calibrated on train (``thresholds.json``'s ``suggested_cutoffs``) on
  the val slice, on BOTH co-primary metrics (``kappa`` and
  ``ordinal_skill.vs_flat``). To guard against search-induced optimism, the
  required exceedance margin grows with the number of configurations tried,
  ``margin_required = base_margin * (1 + log1p(n_configs) / K)``
  (Bonferroni-flavoured). On thin cells (``n_eff < FLOOR``) PBO/Bonferroni are
  low-power -- there are too few independent regime-calls for a multiplicity
  correction to bite -- so the n_eff FLOOR and asset-class pooling
  (:func:`gate_cell`) do the real anti-overfit work there; the margin term is a
  cheap, monotone extra check, not the primary defence.

* **G2 drift-aware generalization.** The released signal's base rates DRIFT
  across the split (e.g. ng1s participation 0.07 -> 0.31 -> 0.43). A replica
  that merely tracks whatever the *current* base rate is would look like it
  generalizes if compared against a single fixed (train) chance level. We
  therefore subtract a chance baseline computed on **each split's own** base
  rates: ``skill_split = metric_split - metric(baseline_on_that_split)``. A
  base-rate-only replica nets to ~0 on both splits, so it cannot pass; a
  genuinely skillful replica keeps a positive, transferable skill.

* **G3 perturbation plateau.** A robust optimum is a plateau, not a knife-edge
  spike. Given the val composite metric evaluated over a neighbourhood of
  parameter settings (supplied by the caller), the replica passes only if the
  worst neighbour still clears the cutoff AND the neighbourhood spread is small.

* **G4 multi-metric consistency.** No single metric may carry a failing
  ensemble: ``kappa``, ``ordinal_skill.vs_flat`` and the NAV increment
  correlation must all agree in sign (> 0).

The ``passed`` flag is the logical AND of the four gates; a below-FLOOR cell is
gated on its asset-class POOL (:func:`gate_cell`) and never returns a standalone
``passed=True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stml.replication import baselines, metrics, nav
from stml.replication.splits import Split

__all__ = ["GateResult", "evaluate", "gate_cell", "FLOOR"]

# Post-embargo val n_eff below this is "low power": gate on the asset-class pool
# instead of the standalone instrument (LOCKED C1 fact; see CONTRACT2.md).
FLOOR = 10

# G1 multiplicity knobs. ``BASE_MARGIN`` is the floor exceedance over the
# chance cutoff with a single config; ``MARGIN_K`` damps how fast the margin
# grows with the number of configurations tried.
BASE_MARGIN = 0.02
MARGIN_K = 10.0

# G2 generalization fraction: val skill must retain at least this share of the
# train skill (and be strictly positive) to count as transferable.
GEN_FRAC = 0.5

# G3 plateau tolerance: the spread of the neighbourhood composite metric must be
# below this for the optimum to count as a plateau rather than a spike.
PLATEAU_STD_TOL = 0.15


@dataclass
class GateResult:
    """Verdict of the four robustness gates for one replica on one cell.

    Attributes
    ----------
    g1, g2, g3, g4 : bool
        Individual gate outcomes (beats-baseline, drift-aware generalization,
        perturbation plateau, multi-metric consistency).
    passed : bool
        Logical AND of the four gates. A below-FLOOR cell evaluated via
        :func:`gate_cell` carries ``details['low_power']=True`` and never
        returns a standalone ``passed=True``.
    details : dict
        Diagnostic payload (per-gate scalars, cutoffs, margins, pooling flags)
        so a verdict is fully auditable.
    """

    g1: bool
    g2: bool
    g3: bool
    g4: bool
    passed: bool
    details: dict = field(default_factory=dict)


def _to_label_array(signal: pd.Series, dates: pd.Index) -> np.ndarray:
    """Slice ``signal`` to ``dates`` (intersection, order-preserving) as ints.

    The split is defined on the full released date axis; a signal carries those
    same dates, so intersecting with ``split.*_dates`` yields the per-split label
    vector without imputing anything.
    """
    common = signal.index.intersection(dates)
    return signal.loc[common].to_numpy().astype(int)


def _group_labels(signal: pd.Series, dates: pd.Index, groups: pd.Series) -> np.ndarray:
    """Per-row group labels for the same rows :func:`_to_label_array` selects.

    ``groups`` is a Series sharing ``signal``'s index (one group label per row,
    e.g. the pool member id). Intersecting with ``dates`` in the SAME order as
    :func:`_to_label_array` keeps the group vector positionally aligned to the
    sliced label vector, so a per-group split is exact.
    """
    common = signal.index.intersection(dates)
    return groups.loc[common].to_numpy()


def _unique_in_order(groups: np.ndarray) -> np.ndarray:
    """Unique group labels in first-appearance order (deterministic averaging)."""
    _, first_idx = np.unique(groups, return_index=True)
    return groups[np.sort(first_idx)]


def _mean_over_groups(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    groups: np.ndarray,
    kernel,
) -> float:
    """Equal-weight mean of ``kernel(true_g, pred_g)`` over unique groups.

    Each group's rows are sliced out positionally and scored by ``kernel`` (a
    within-instrument metric); the per-group scores are averaged with equal
    weight, so a larger member never dominates. An empty input returns ``0.0``
    (chance level). Non-finite per-group kernels (e.g. a constant-series NAV corr)
    are dropped from the average; if NONE are finite the mean is ``nan`` so the
    downstream ``> 0`` test fails, exactly as a degenerate ensemble should.
    """
    if labels_true.size == 0:
        return 0.0
    scores: list[float] = []
    for g in _unique_in_order(groups):
        mask = groups == g
        scores.append(float(kernel(labels_true[mask], labels_pred[mask])))
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _margin_required(n_configs: int) -> float:
    """Bonferroni-flavoured exceedance margin that grows with search breadth.

    ``margin = BASE_MARGIN * (1 + log1p(n_configs) / MARGIN_K)``. With a single
    config this is ``BASE_MARGIN``; it inflates slowly (log) as more
    configurations are tried, so a candidate plucked from a large search must
    clear the chance cutoff by a wider margin. On thin cells this term is
    low-power by design -- the n_eff FLOOR + pooling carry the real load.
    """
    n = max(int(n_configs), 1)
    return BASE_MARGIN * (1.0 + math.log1p(n) / MARGIN_K)


def _skill_vs_split_baseline(
    target: np.ndarray, replica: np.ndarray
) -> float:
    """Replica skill on a split, net of that split's OWN majority chance level.

    ``skill = kappa(target, replica) - kappa(target, majority_on_this_split)``.
    Computing the baseline from *this* split's base rates is what makes G2
    drift-aware: a replica that only echoes the current majority class nets to
    ~0 regardless of how the base rate drifts between splits, whereas a genuine
    replica keeps a positive skill. Cohen's kappa of a constant (majority)
    prediction is 0 by construction, so the subtraction is explicit and honest
    rather than a hidden no-op.
    """
    if target.size == 0:
        return 0.0
    metric = metrics.panel(target, replica)["kappa"]
    base_pred = baselines.majority_class(target)
    base_metric = metrics.panel(target, base_pred)["kappa"]
    return metric - base_metric


def _kappa(target: np.ndarray, replica: np.ndarray) -> float:
    """Cohen's kappa via :func:`metrics.panel` (per-group kernel for pools)."""
    return float(metrics.panel(target, replica)["kappa"])


def _ordinal_vs_flat(target: np.ndarray, replica: np.ndarray) -> float:
    """Ordinal skill vs flat via :func:`metrics.panel` (per-group kernel)."""
    return float(metrics.panel(target, replica)["ordinal_skill"]["vs_flat"])


def _gate1(
    target_val: np.ndarray,
    replica_val: np.ndarray,
    cutoffs: dict,
    n_configs: int,
    groups_val: np.ndarray | None = None,
) -> tuple[bool, dict]:
    """G1: val co-primary metrics beat the chance cutoffs by a growing margin.

    With ``groups_val`` the co-primary metrics are the MEAN of each member's own
    panel (within-instrument), so a pooled cell cannot clear G1 on cross-instrument
    base-rate matching.
    """
    if groups_val is None:
        panel_val = metrics.panel(target_val, replica_val)
        kappa = panel_val["kappa"]
        osk = panel_val["ordinal_skill"]["vs_flat"]
    else:
        kappa = _mean_over_groups(target_val, replica_val, groups_val, _kappa)
        osk = _mean_over_groups(target_val, replica_val, groups_val, _ordinal_vs_flat)
    margin = _margin_required(n_configs)
    kappa_bar = cutoffs["kappa"] + margin
    osk_bar = cutoffs["ordinal_skill"] + margin
    passed = bool(kappa > kappa_bar and osk > osk_bar)
    return passed, {
        "kappa": kappa,
        "ordinal_skill_vs_flat": osk,
        "kappa_cutoff": kappa_bar,
        "ordinal_skill_cutoff": osk_bar,
        "margin_required": margin,
        "n_configs": int(n_configs),
        # PBO / Bonferroni are low-power below FLOOR; floor+pooling do the work.
        "multiplicity_note": (
            "margin grows with n_configs (Bonferroni-flavoured); on thin cells "
            "(n_eff < FLOOR) the floor + pooling carry the anti-overfit load"
        ),
    }


def _gate2(
    target_train: np.ndarray,
    replica_train: np.ndarray,
    target_val: np.ndarray,
    replica_val: np.ndarray,
    gen_frac: float,
    groups_train: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
) -> tuple[bool, dict]:
    """G2: per-split-baseline skill on val transfers from train (drift-aware).

    With group labels the per-split skill is the MEAN of each member's own
    per-split-baseline skill (within-instrument, per-member/per-split baselines),
    so a pooled cell's generalisation is judged member-by-member rather than on a
    base-rate-matched concatenation.
    """
    if groups_train is None:
        skill_train = _skill_vs_split_baseline(target_train, replica_train)
    else:
        skill_train = _mean_over_groups(
            target_train, replica_train, groups_train, _skill_vs_split_baseline
        )
    if groups_val is None:
        skill_val = _skill_vs_split_baseline(target_val, replica_val)
    else:
        skill_val = _mean_over_groups(
            target_val, replica_val, groups_val, _skill_vs_split_baseline
        )
    # Pass requires BOTH a positive transferable skill AND retention of at least
    # gen_frac of the train skill. The strict ``> 0`` is what fails a replica
    # that merely matches each split's drifting base rate (skill ~ 0 on both).
    passed = bool(skill_val > 0.0 and skill_val >= gen_frac * skill_train)
    return passed, {
        "skill_train": skill_train,
        "skill_val": skill_val,
        "gen_frac": gen_frac,
        "required_val_skill": gen_frac * skill_train,
    }


def _gate3(
    perturbed_metrics: list[float] | None,
    cutoff: float,
    std_tol: float,
) -> tuple[bool, dict]:
    """G3: the neighbourhood composite metric is a plateau, not a spike."""
    if perturbed_metrics is None:
        # The caller owns neighbour generation; with none supplied G3 is simply
        # not evaluated (treated as non-blocking) and flagged as such.
        return True, {"not_evaluated": True}
    arr = np.asarray(perturbed_metrics, dtype=float)
    if arr.size == 0:
        return True, {"not_evaluated": True, "reason": "empty_neighbourhood"}
    worst = float(arr.min())
    spread = float(arr.std())
    passed = bool(worst > cutoff and spread < std_tol)
    return passed, {
        "min_perturbed": worst,
        "std_perturbed": spread,
        "cutoff": cutoff,
        "std_tol": std_tol,
        "n_neighbours": int(arr.size),
    }


def _gate4(
    target_val: np.ndarray,
    replica_val: np.ndarray,
    replica_signal: pd.Series,
    target_signal: pd.Series,
    aligned_ret: pd.Series,
    val_dates: pd.Index,
    groups_val: np.ndarray | None = None,
    groups: pd.Series | None = None,
) -> tuple[bool, dict]:
    """G4: kappa, ordinal skill and NAV increment-corr all agree (> 0).

    With group labels every co-primary signal is the MEAN of the per-member
    within-instrument values (kappa, ordinal skill, and NAV increment-corr), so a
    pooled cell passes only when the members agree in sign on AVERAGE -- not when a
    cross-instrument concatenation happens to correlate.
    """
    if groups_val is None:
        panel_val = metrics.panel(target_val, replica_val)
        kappa = panel_val["kappa"]
        osk = panel_val["ordinal_skill"]["vs_flat"]
    else:
        kappa = _mean_over_groups(target_val, replica_val, groups_val, _kappa)
        osk = _mean_over_groups(target_val, replica_val, groups_val, _ordinal_vs_flat)

    # NAV increment correlation on the val window only (three-way aligned inside
    # nav_discrepancy). A constant val series -> nan corr, which fails the > 0
    # test, exactly as a degenerate replica should.
    if groups is None:
        rep_val = replica_signal.loc[replica_signal.index.intersection(val_dates)]
        tgt_val = target_signal.loc[target_signal.index.intersection(val_dates)]
        inc_corr = nav.nav_discrepancy(rep_val, tgt_val, aligned_ret)["increment_corr"]
    else:
        inc_corr = _mean_group_increment_corr(
            replica_signal, target_signal, aligned_ret, val_dates, groups
        )
    inc_ok = bool(np.isfinite(inc_corr) and inc_corr > 0.0)
    passed = bool(kappa > 0.0 and osk > 0.0 and inc_ok)
    return passed, {
        "kappa": kappa,
        "ordinal_skill_vs_flat": osk,
        "increment_corr": inc_corr,
    }


def _mean_group_increment_corr(
    replica_signal: pd.Series,
    target_signal: pd.Series,
    aligned_ret: pd.Series,
    val_dates: pd.Index,
    groups: pd.Series,
) -> float:
    """Equal-weight mean of per-member val NAV increment-correlation.

    For each unique member the signal rows are restricted to that member's rows
    (``groups == member``) intersected with the val window, then scored by
    :func:`nav.nav_discrepancy` -- a genuinely WITHIN-instrument PnL correlation.
    Members whose val NAV correlation is non-finite (a constant member series) are
    dropped; if none are finite the result is ``nan`` so G4's ``> 0`` test fails.
    """
    val_common = replica_signal.index.intersection(val_dates)
    corrs: list[float] = []
    for g in _unique_in_order(groups.loc[val_common].to_numpy()):
        member_idx = groups.index[groups == g]
        rows = member_idx.intersection(val_common)
        if len(rows) == 0:
            continue
        c = nav.nav_discrepancy(
            replica_signal.loc[rows], target_signal.loc[rows], aligned_ret.loc[rows]
        )["increment_corr"]
        if math.isfinite(c):
            corrs.append(float(c))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def evaluate(
    replica_signal: pd.Series,
    target_signal: pd.Series,
    aligned_ret: pd.Series,
    split: Split,
    thresholds_entry: dict,
    n_eff: int,
    n_configs: int,
    perturbed_metrics: list[float] | None = None,
    groups: pd.Series | np.ndarray | None = None,
) -> GateResult:
    """Run the four robustness gates on a replica over the TRAIN+VAL window.

    Parameters
    ----------
    replica_signal, target_signal : pd.Series
        Date-indexed signals in ``{-1, 0, 1}`` on the released date axis.
    aligned_ret : pd.Series
        Date-indexed aligned log-returns (``next_day`` convention already
        applied by :func:`stml.replication.align.align_instrument`).
    split : Split
        The chronological split whose ``train_dates`` / ``val_dates`` slice the
        per-split label vectors. The test block is NEVER read here.
    thresholds_entry : dict
        One ``thresholds.json`` entry (per-instrument or per-asset-class). G1
        reads ``thresholds_entry['suggested_cutoffs']`` for ``kappa`` and
        ``ordinal_skill``.
    n_eff : int
        Post-embargo val effective sample size (number of regime-calls). Carried
        into ``details`` and consulted by :func:`gate_cell` for the FLOOR rule.
    n_configs : int
        Number of configurations evaluated in the search that produced this
        replica; inflates the G1 exceedance margin (multiplicity control).
    perturbed_metrics : list[float] or None
        Val composite metrics over a parameter neighbourhood for G3. ``None``
        means the caller did not supply neighbours, so G3 is not evaluated. For a
        pooled cell the caller supplies group-AVERAGED composites here.
    groups : array-like or None
        Optional per-row group label aligned to the signal index (one label per
        row; e.g. the pool member id). When provided, G1/G2/G4 are computed on
        PER-GROUP-then-AVERAGED metrics (within-instrument), so a pooled cell can
        never clear a gate by cross-instrument base-rate matching. ``None`` (the
        default, every standalone cell) leaves the single-series gates unchanged.

    Returns
    -------
    GateResult
    """
    cutoffs = thresholds_entry["suggested_cutoffs"]

    target_val = _to_label_array(target_signal, split.val_dates)
    replica_val = _to_label_array(replica_signal, split.val_dates)
    target_train = _to_label_array(target_signal, split.train_dates)
    replica_train = _to_label_array(replica_signal, split.train_dates)

    # A pooled cell carries one group label per signal row; align it to the same
    # index as the signals so each split's group vector is positionally exact.
    groups_series = (
        None
        if groups is None
        else pd.Series(np.asarray(groups), index=replica_signal.index)
    )
    groups_val = (
        None
        if groups_series is None
        else _group_labels(replica_signal, split.val_dates, groups_series)
    )
    groups_train = (
        None
        if groups_series is None
        else _group_labels(replica_signal, split.train_dates, groups_series)
    )

    g1, d1 = _gate1(target_val, replica_val, cutoffs, n_configs, groups_val)
    g2, d2 = _gate2(
        target_train,
        replica_train,
        target_val,
        replica_val,
        GEN_FRAC,
        groups_train,
        groups_val,
    )
    # G3 plateau is judged against the same chance kappa cutoff as G1.
    g3, d3 = _gate3(perturbed_metrics, cutoffs["kappa"], PLATEAU_STD_TOL)
    g4, d4 = _gate4(
        target_val,
        replica_val,
        replica_signal,
        target_signal,
        aligned_ret,
        split.val_dates,
        groups_val,
        groups_series,
    )

    passed = bool(g1 and g2 and g3 and g4)
    details = {
        "g1": d1,
        "g2": d2,
        "g3": d3,
        "g4": d4,
        "n_eff": int(n_eff),
        "floor": FLOOR,
    }
    if groups_series is not None:
        details["grouped"] = True
    return GateResult(g1=g1, g2=g2, g3=g3, g4=g4, passed=passed, details=details)


def gate_cell(
    replica_signal: pd.Series,
    target_signal: pd.Series,
    aligned_ret: pd.Series,
    split: Split,
    thresholds_entry: dict,
    n_eff: int,
    n_configs: int,
    *,
    pooled_replica_signal: pd.Series | None = None,
    pooled_target_signal: pd.Series | None = None,
    pooled_aligned_ret: pd.Series | None = None,
    pooled_split: Split | None = None,
    pooled_thresholds_entry: dict | None = None,
    pooled_n_eff: int | None = None,
    perturbed_metrics: list[float] | None = None,
    groups: pd.Series | np.ndarray | None = None,
) -> GateResult:
    """Gate one cell, routing thin (``n_eff < FLOOR``) cells through pooling.

    A standalone instrument with enough leakage-free regime-calls
    (``n_eff >= FLOOR``) is gated directly via :func:`evaluate`. Below the FLOOR
    there are too few independent observations to trust a standalone verdict, so
    the caller supplies the concatenated asset-class POOL series and the cell is
    gated on the pool instead. Such a verdict is always tagged
    ``details['low_power']=True`` / ``details['pooled']=True`` and can NEVER be a
    standalone pass: even if the pooled gates clear, ``passed`` is forced to
    ``False`` for the instrument itself -- the pool result is advisory for the
    class, not a green light for the thin member.

    The ``pooled_*`` arguments default to ``None``; the caller (US-010/US-011) is
    responsible for building and passing the concatenated asset-class series.
    ``groups`` (a per-row member id aligned to the pooled signals) makes the gates
    measure WITHIN-instrument skill: a pool with summed n_eff >= FLOOR is gated
    first-class via :func:`evaluate` but on per-member-then-averaged metrics, so
    cross-instrument base-rate matching can no longer manufacture a pass.
    """
    if n_eff >= FLOOR:
        return evaluate(
            replica_signal,
            target_signal,
            aligned_ret,
            split,
            thresholds_entry,
            n_eff,
            n_configs,
            perturbed_metrics=perturbed_metrics,
            groups=groups,
        )

    # Below FLOOR: gate on the asset-class pool (caller-supplied). Fall back to
    # the standalone series only if no pool was provided, but still mark the
    # verdict low-power and strip any standalone pass.
    p_replica = pooled_replica_signal if pooled_replica_signal is not None else replica_signal
    p_target = pooled_target_signal if pooled_target_signal is not None else target_signal
    p_ret = pooled_aligned_ret if pooled_aligned_ret is not None else aligned_ret
    p_split = pooled_split if pooled_split is not None else split
    p_entry = pooled_thresholds_entry if pooled_thresholds_entry is not None else thresholds_entry
    p_n_eff = pooled_n_eff if pooled_n_eff is not None else n_eff

    pooled = evaluate(
        p_replica,
        p_target,
        p_ret,
        p_split,
        p_entry,
        p_n_eff,
        n_configs,
        perturbed_metrics=perturbed_metrics,
        groups=groups,
    )

    pooled.details["low_power"] = True
    pooled.details["pooled"] = True
    pooled.details["standalone_n_eff"] = int(n_eff)
    pooled.details["pool_provided"] = pooled_replica_signal is not None
    # A below-floor cell never earns a standalone pass, whatever the pool says.
    pooled.passed = False
    return pooled
