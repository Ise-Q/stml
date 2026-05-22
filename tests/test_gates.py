"""
test_gates.py
=============
Tests for :mod:`stml.replication.gates` (US-008).

The four robustness gates are exercised on small, deterministic synthetic
signals so the invariants hold without depending on brittle magic numbers:

- A near-perfect replica (``replica == target``) passes ALL four gates.
- A baseline-level replica (always-flat / majority predictions) FAILS G1.
- A below-FLOOR ``n_eff`` cell routes through pooling: :func:`gates.gate_cell`
  sets ``low_power=True`` / ``pooled=True`` and never grants a standalone pass.
- G2 is drift-aware: a replica that only matches each split's *own* (drifting)
  majority class has ~0 skill after per-split baseline subtraction and so does
  NOT pass G2.

A final integration smoke test gates one real instrument (cl1s) end to end via
``load_clean_data`` + ``align_instrument`` + ``chronological_split``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.replication import baselines, gates
from stml.replication.splits import chronological_split

# Chance cutoffs in the shape of one thresholds.json entry. The exact floor
# values mirror the per-instrument ``suggested_cutoffs`` minimum (0.05).
_THRESHOLDS_ENTRY = {
    "suggested_cutoffs": {
        "kappa": 0.05,
        "mcc": 0.05,
        "macro_f1": 0.3,
        "ordinal_skill": 0.05,
    }
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _axis(n: int = 100, start: str = "2021-01-04") -> pd.DatetimeIndex:
    """A contiguous daily date axis the chronological split is built on."""
    return pd.date_range(start=start, periods=n, freq="D")


def _structured_target(dates: pd.DatetimeIndex, seed: int = 7) -> pd.Series:
    """A non-degenerate {-1,0,1} target with a real (skillful) structure.

    Built as a slowly-varying run-based signal so that a perfect replica earns a
    clearly positive kappa / ordinal skill rather than sitting at chance.
    """
    rng = np.random.default_rng(seed)
    vals: list[int] = []
    while len(vals) < len(dates):
        lab = int(rng.choice([-1, 0, 1]))
        run = int(rng.integers(2, 6))
        vals.extend([lab] * run)
    return pd.Series(np.asarray(vals[: len(dates)], dtype=int), index=dates)


def _returns(dates: pd.DatetimeIndex, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, size=len(dates)), index=dates)


# --------------------------------------------------------------------------- #
# GateResult shape                                                             #
# --------------------------------------------------------------------------- #

def test_gate_result_dataclass_fields() -> None:
    """GateResult carries the four bools, a passed flag, and a details dict."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    gr = gates.evaluate(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=10,
        perturbed_metrics=[0.8, 0.82, 0.79],
    )
    assert isinstance(gr.g1, bool)
    assert isinstance(gr.g2, bool)
    assert isinstance(gr.g3, bool)
    assert isinstance(gr.g4, bool)
    assert isinstance(gr.passed, bool)
    assert isinstance(gr.details, dict)
    assert gr.passed == (gr.g1 and gr.g2 and gr.g3 and gr.g4)


# --------------------------------------------------------------------------- #
# Near-perfect replica passes all four gates                                  #
# --------------------------------------------------------------------------- #

def test_perfect_replica_passes_all_gates() -> None:
    """replica == target clears G1-G4 (the contract's headline invariant)."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    gr = gates.evaluate(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=50,
        # neighbourhood clears the cutoff with tiny spread -> plateau
        perturbed_metrics=[0.80, 0.81, 0.82, 0.79],
    )
    assert gr.g1, gr.details["g1"]
    assert gr.g2, gr.details["g2"]
    assert gr.g3, gr.details["g3"]
    assert gr.g4, gr.details["g4"]
    assert gr.passed


# --------------------------------------------------------------------------- #
# Baseline-level replica fails G1                                             #
# --------------------------------------------------------------------------- #

def test_always_flat_replica_fails_g1() -> None:
    """An always-flat replica scores at chance on val and so fails G1."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    flat = pd.Series(np.zeros(len(dates), dtype=int), index=dates)
    gr = gates.evaluate(
        flat, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=50,
    )
    assert not gr.g1
    assert not gr.passed


def test_majority_replica_fails_g1() -> None:
    """A majority-class replica also scores at chance on val and fails G1."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    mode = baselines.majority_class(target.to_numpy())[0]
    maj = pd.Series(np.full(len(dates), mode, dtype=int), index=dates)
    gr = gates.evaluate(
        maj, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=50,
    )
    assert not gr.g1
    assert not gr.passed


# --------------------------------------------------------------------------- #
# Below-floor n_eff routes through pooling, never a standalone pass           #
# --------------------------------------------------------------------------- #

def test_below_floor_pools_and_blocks_standalone_pass() -> None:
    """n_eff=2 (ng1s-like) -> gate_cell pools, sets low_power, no standalone pass.

    Even with a perfect replica (which would otherwise pass all four gates), a
    below-FLOOR cell must NOT return a standalone passed=True.
    """
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    gr = gates.gate_cell(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=2, n_configs=50,
        perturbed_metrics=[0.80, 0.81, 0.82],
    )
    assert gr.details.get("low_power") is True
    assert gr.details.get("pooled") is True
    assert gr.passed is False
    assert gr.details.get("standalone_n_eff") == 2


def test_at_floor_evaluates_standalone() -> None:
    """n_eff == FLOOR is gated directly (no pooling flag, may pass)."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    gr = gates.gate_cell(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=gates.FLOOR,
        n_configs=50, perturbed_metrics=[0.80, 0.81, 0.82],
    )
    assert "low_power" not in gr.details
    assert "pooled" not in gr.details
    assert gr.passed is True


def test_below_floor_uses_pooled_series_when_provided() -> None:
    """When the caller supplies pool series, gate_cell evaluates on the pool."""
    dates = _axis()
    split = chronological_split(dates)
    # Standalone (thin) cell: a flat signal that would fail G1 on its own.
    flat = pd.Series(np.zeros(len(dates), dtype=int), index=dates)
    ret = _returns(dates)
    # Pool: a richer structured target/replica pair.
    pool_target = _structured_target(dates, seed=3)

    gr = gates.gate_cell(
        flat, flat, ret, split, _THRESHOLDS_ENTRY, n_eff=2, n_configs=10,
        pooled_replica_signal=pool_target,
        pooled_target_signal=pool_target,
        pooled_aligned_ret=ret,
        pooled_split=split,
        pooled_thresholds_entry=_THRESHOLDS_ENTRY,
        pooled_n_eff=48,
        perturbed_metrics=[0.80, 0.81],
    )
    assert gr.details.get("pool_provided") is True
    assert gr.details.get("low_power") is True
    # Pool gates may clear, but the thin member still gets no standalone pass.
    assert gr.passed is False
    # The G1 verdict reflects the POOL (perfect replica), not the flat standalone.
    assert gr.g1 is True


# --------------------------------------------------------------------------- #
# G2 is drift-aware: matching a drifting per-split base rate fails G2          #
# --------------------------------------------------------------------------- #

def test_drift_base_rate_replica_fails_g2() -> None:
    """A replica that predicts each split's OWN majority has ~0 per-split skill.

    The target's majority class differs between train and val (base-rate drift).
    A replica that simply emits each split's majority therefore tracks the
    drifting base rate but carries no genuine skill: after subtracting that
    split's own majority-baseline kappa, skill_train and skill_val are both ~0,
    so G2 (which requires a strictly positive, transferable skill) must fail.
    """
    dates = _axis(120)
    split = chronological_split(dates)

    # Construct a target whose train and val majorities differ (drift).
    train_dates = split.train_dates
    val_dates = split.val_dates
    test_dates = split.test_dates
    target = pd.Series(0, index=dates, dtype=int)
    # train: mostly +1 (majority +1), with some structure
    target.loc[train_dates] = np.resize([1, 1, 1, 0, -1], len(train_dates))
    # val: mostly -1 (majority flips to -1 -> drift)
    target.loc[val_dates] = np.resize([-1, -1, -1, 0, 1], len(val_dates))
    target.loc[test_dates] = 0
    ret = _returns(dates)

    # Verify the drift premise: train and val majorities really differ.
    train_majority = baselines.majority_class(target.loc[train_dates].to_numpy())[0]
    val_majority = baselines.majority_class(target.loc[val_dates].to_numpy())[0]
    assert train_majority != val_majority, "test premise: base rate must drift"

    # Replica = each split's own majority class.
    replica = pd.Series(0, index=dates, dtype=int)
    replica.loc[train_dates] = train_majority
    replica.loc[val_dates] = val_majority

    gr = gates.evaluate(
        replica, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=10,
    )
    assert not gr.g2, gr.details["g2"]
    # skill nets to ~0 on both splits after per-split baseline subtraction.
    assert abs(gr.details["g2"]["skill_train"]) < 1e-9
    assert abs(gr.details["g2"]["skill_val"]) < 1e-9


def test_genuine_replica_passes_g2() -> None:
    """A perfect replica keeps a positive, transferable per-split skill -> G2."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    gr = gates.evaluate(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=10,
    )
    assert gr.g2
    assert gr.details["g2"]["skill_val"] > 0.0


# --------------------------------------------------------------------------- #
# Group-aware gates: cross-instrument base-rate matching cannot pass           #
# --------------------------------------------------------------------------- #

def _base_rate_only_pool_signals(
    n_per: int = 60,
) -> tuple[pd.Series, pd.Series, pd.Series, np.ndarray, "object"]:
    """A 2-member pool: replica matches ONLY each member's (drifting) base rate.

    Each member's target has within-member structure but a DIFFERENT majority; the
    replica is constant at that member's majority. Concatenated, the two constant
    blocks line up with the cross-member base-rate split (spurious positive kappa);
    per member each block is a marginal-only guess (kappa ~0). Returns
    ``(replica, target, ret, groups, split)`` on a synthetic integer index with a
    split whose train/val each span BOTH members.
    """
    from stml.replication.splits import Split

    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(12)

    def member(maj: int, rng: np.random.Generator) -> np.ndarray:
        vals: list[int] = []
        while len(vals) < n_per:
            vals.append(maj if rng.random() < 0.7 else int(rng.choice([-1, 0, 1])))
        return np.asarray(vals[:n_per], dtype=int)

    t_a = member(1, rng_a)
    t_b = member(-1, rng_b)
    idx = pd.RangeIndex(2 * n_per)
    target = pd.Series(np.concatenate([t_a, t_b]), index=idx)
    replica = pd.Series(
        np.concatenate([np.full(n_per, 1), np.full(n_per, -1)]), index=idx
    )
    groups = np.asarray(["A"] * n_per + ["B"] * n_per, dtype=object)
    ret = pd.Series(np.sign(replica.to_numpy()).astype(float) * 0.01, index=idx)
    cut = int(n_per * 0.6)
    train = np.concatenate([np.arange(cut), np.arange(n_per, n_per + cut)])
    val = np.concatenate([np.arange(cut, n_per), np.arange(n_per + cut, 2 * n_per)])
    split = Split(
        train_idx=train,
        val_idx=val,
        test_idx=np.empty(0, dtype=int),
        train_dates=pd.Index(train),
        val_dates=pd.Index(val),
        test_dates=pd.Index(np.empty(0, dtype=int)),
    )
    return replica, target, ret, groups, split


def test_groups_block_cross_instrument_base_rate_pass() -> None:
    """A base-rate-only pooled replica clears G1 concatenated but fails it grouped.

    Locks the artifact fix at the gate level: WITHOUT ``groups`` the concatenated
    co-primary kappa is spuriously positive and G1 passes; WITH ``groups`` the
    co-primary metrics are per-member-then-averaged (each member chance-level), so
    G1 fails and the verdict carries ``grouped=True``.
    """
    replica, target, ret, groups, split = _base_rate_only_pool_signals()

    no_groups = gates.evaluate(
        replica, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=10,
    )
    grouped = gates.evaluate(
        replica, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=10,
        groups=groups,
    )

    # Concatenation rewards base-rate matching -> spurious G1 pass.
    assert no_groups.g1 is True
    assert no_groups.details["g1"]["kappa"] > 0.1
    # Group-averaging measures within-instrument skill -> chance -> G1 fails.
    assert grouped.g1 is False
    assert abs(grouped.details["g1"]["kappa"]) < 0.05
    assert grouped.details.get("grouped") is True
    assert grouped.passed is False


# --------------------------------------------------------------------------- #
# G3 perturbation plateau                                                     #
# --------------------------------------------------------------------------- #

def test_g3_not_evaluated_when_no_neighbours() -> None:
    """perturbed_metrics=None -> G3 passes but is flagged not_evaluated."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    gr = gates.evaluate(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=10,
        perturbed_metrics=None,
    )
    assert gr.g3 is True
    assert gr.details["g3"].get("not_evaluated") is True


def test_g3_spike_fails() -> None:
    """A knife-edge optimum (high spread / low worst neighbour) fails G3."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    # Neighbourhood collapses to chance at the edges -> not a plateau.
    gr = gates.evaluate(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=10,
        perturbed_metrics=[0.80, 0.01, -0.02, 0.79],
    )
    assert gr.g3 is False


# --------------------------------------------------------------------------- #
# G4 multi-metric consistency                                                 #
# --------------------------------------------------------------------------- #

def test_g4_fails_when_nav_corr_nonpositive() -> None:
    """G4 fails if the NAV increment correlation is not strictly positive.

    An opposite-sign replica keeps a nonzero kappa structure but its PnL
    increments anti-correlate with the target, so G4 (which needs all three
    co-primary signals > 0) must fail even if other gates might not.
    """
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    opposite = -target
    gr = gates.evaluate(
        opposite, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=10,
        perturbed_metrics=[0.8],
    )
    assert gr.g4 is False
    assert gr.details["g4"]["increment_corr"] <= 0.0 or not np.isfinite(
        gr.details["g4"]["increment_corr"]
    )


# --------------------------------------------------------------------------- #
# G1 multiplicity: margin grows with n_configs                                #
# --------------------------------------------------------------------------- #

def test_g1_margin_grows_with_n_configs() -> None:
    """The required exceedance margin is monotone non-decreasing in n_configs."""
    dates = _axis()
    split = chronological_split(dates)
    target = _structured_target(dates)
    ret = _returns(dates)

    g_small = gates.evaluate(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=1,
    )
    g_large = gates.evaluate(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=1000,
    )
    assert (
        g_large.details["g1"]["margin_required"]
        >= g_small.details["g1"]["margin_required"]
    )


# --------------------------------------------------------------------------- #
# Integration smoke: gate one real instrument end to end                      #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_data():
    from stml.io import load_clean_data
    return load_clean_data()


def test_gate_real_instrument_runs(real_data) -> None:
    """A real instrument (cl1s) gates end to end without crashing.

    Uses the released signal as its own (perfect) replica so the run is
    deterministic; the point is that align + split + gates compose on real data
    and return a well-formed GateResult.
    """
    from stml.replication.align import align_instrument

    ohlcv, signals = real_data
    inst = "cl1s"
    aligned = align_instrument(signals, ohlcv, inst, convention="next_day")
    frame = aligned.frame.set_index("date")
    target = frame["signal"].astype(int)
    ret = frame["ret"]

    split = chronological_split(signals["date"])

    gr = gates.evaluate(
        target, target, ret, split, _THRESHOLDS_ENTRY, n_eff=20, n_configs=20,
        perturbed_metrics=[0.7, 0.71, 0.69],
    )
    assert isinstance(gr.passed, bool)
    assert set(gr.details) >= {"g1", "g2", "g3", "g4", "n_eff", "floor"}
    # Perfect self-replica on real data should clear the consistency gate.
    assert gr.g4 is True
