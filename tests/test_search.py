"""
test_search.py
==============
Tests for :mod:`stml.replication.search` (US-010).

The tiered guided search is exercised on small, deterministic synthetic data
(plus one real-instrument integration smoke test) so the contract invariants
hold without depending on brittle magic numbers:

* **TPE reproducibility** -- two ``search_cell`` runs with the SAME seed+budget
  on the same data return the SAME ``best_params`` (the ``n_eff >= FLOOR`` path);
* **grid determinism** -- a forced low-``n_eff`` cell evaluates its whole grid
  and returns a deterministic best;
* **ledger provenance** -- the ledger receives every evaluated config, each with
  a ``tier`` in ``{"tpe", "grid"}`` and a ``motivated_by`` note;
* **NO test access** -- monkeypatching :func:`stml.replication.splits.get_test`
  to raise does not perturb ``search_cell`` (it must never reach the test split);
* **train-based objective** -- a ``generate_fn`` whose replica perfectly matches
  the target on TRAIN yields ``best_discrepancy ~ 0`` (selection is by the train
  objective, not val).

All ledger writes go to a ``tmp_path`` so the real ``results/jj/ledger.json`` is
never touched. Budgets are kept small (8-16) for speed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stml.replication.ledger import Ledger
from stml.replication.search import (
    FLOOR,
    composite_skill,
    discrepancy,
    search_cell,
)
from stml.replication.splits import chronological_split

# A small TPE budget -- enough to exercise the sampler, fast in CI.
_BUDGET = 12


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _structured_target(n: int = 90, seed: int = 7) -> pd.Series:
    """A non-degenerate run-based ``{-1, 0, +1}`` target on a daily date axis.

    Built from short constant runs so a perfect replica earns a clearly positive
    composite skill (and hence ~0 discrepancy) rather than sitting at chance.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="D")
    vals: list[int] = []
    while len(vals) < n:
        lab = int(rng.choice([-1, 0, 1]))
        run = int(rng.integers(2, 6))
        vals.extend([lab] * run)
    return pd.Series(np.asarray(vals[:n], dtype=int), index=idx)


def _shift_generate_fn(target: pd.Series) -> Callable[[dict], pd.Series]:
    """A ``generate_fn`` over a discrete ``shift`` axis (and an inert ``k`` axis).

    ``shift == 0`` reproduces the target EXACTLY (so its train discrepancy is 0);
    any other shift degrades it. The inert ``k`` axis widens the grid so the
    Cartesian-product enumeration is non-trivial. The returned series is aligned
    to ``target``'s index (NaNs from the shift are filled flat).
    """

    def gen(params: dict) -> pd.Series:
        shifted = target.shift(int(params["shift"]))
        return shifted.fillna(0).astype(int)

    return gen


# Discrete search space matching the synthetic generate_fn above.
_PARAM_SPACE: dict[str, list] = {"shift": [0, 1, 2, 3], "k": [0, 1]}


# --------------------------------------------------------------------------- #
# composite_skill / discrepancy identities                                     #
# --------------------------------------------------------------------------- #
def test_perfect_replica_zero_discrepancy() -> None:
    """A perfect replica scores composite_skill 1 and discrepancy 0."""
    target = _structured_target()
    assert composite_skill(target, target) == pytest.approx(1.0)
    assert discrepancy(target, target) == pytest.approx(0.0)


def test_flat_replica_chance_discrepancy() -> None:
    """An always-flat replica scores ~0 skill (discrepancy ~ 1), not negative."""
    target = _structured_target()
    flat = pd.Series(np.zeros(len(target), dtype=int), index=target.index)
    skill = composite_skill(target, flat)
    # Chance-corrected: a marginal-only guess sits at ~0 skill.
    assert abs(skill) < 0.15
    assert discrepancy(target, flat) == pytest.approx(1.0 - skill)


# --------------------------------------------------------------------------- #
# Cross-instrument pooling artifact: group-averaging vs concatenation          #
# --------------------------------------------------------------------------- #
def _base_rate_only_pool() -> tuple[pd.Series, pd.Series, np.ndarray]:
    """A two-member pool whose replica matches ONLY each member's base-rate level.

    Each member's target has genuine within-member structure (short runs) but a
    DIFFERENT majority class; the replica is CONSTANT at that member's majority.
    So per member the replica is a marginal-only guess (within-instrument kappa
    ~0), yet the concatenation of the two constant blocks lines up with the
    cross-member base-rate split and earns a spuriously positive concatenated
    kappa -- exactly the artifact the ``groups`` aggregation removes.
    """
    n_per = 60
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(12)

    def member(maj: int, rng: np.random.Generator) -> np.ndarray:
        vals: list[int] = []
        while len(vals) < n_per:
            lab = maj if rng.random() < 0.7 else int(rng.choice([-1, 0, 1]))
            vals.append(lab)
        return np.asarray(vals[:n_per], dtype=int)

    t_a = member(1, rng_a)  # member A majority +1
    t_b = member(-1, rng_b)  # member B majority -1
    target = pd.Series(np.concatenate([t_a, t_b]))
    replica = pd.Series(np.concatenate([np.full(n_per, 1), np.full(n_per, -1)]))
    groups = np.asarray(["A"] * n_per + ["B"] * n_per, dtype=object)
    return target, replica, groups


def test_groups_neutralise_base_rate_matching_artifact() -> None:
    """A base-rate-only replica scores ~0 WITH groups, spuriously positive without.

    Locks the cross-instrument pooling fix: ``composite_skill`` (and hence
    ``discrepancy``) on a replica that only matches each member's base-rate level
    must be ~chance once ``groups`` measures WITHIN-instrument skill, even though
    the CONCATENATED skill (``groups=None``) is clearly positive from
    cross-member base-rate matching.
    """
    target, replica, groups = _base_rate_only_pool()

    concat_skill = composite_skill(target, replica)  # old single-metric behaviour
    grouped_skill = composite_skill(target, replica, groups)  # within-instrument

    # The artifact is real: concatenation rewards base-rate matching.
    assert concat_skill > 0.1
    # Group-averaging removes it: each member's constant replica is chance-level.
    assert abs(grouped_skill) < 0.05
    # discrepancy threads groups through identically.
    assert discrepancy(target, replica, groups) == pytest.approx(1.0 - grouped_skill)
    assert discrepancy(target, replica, groups) > discrepancy(target, replica)


def test_groups_equal_weight_mean_of_per_member_skill() -> None:
    """With groups the composite is the equal-weight mean of per-member composites.

    Equal weight (not row-weighted): two members of different sizes each count
    once, so the grouped composite equals the plain mean of the two members'
    standalone composites.
    """
    target, replica, groups = _base_rate_only_pool()
    a = groups == "A"
    b = groups == "B"
    comp_a = composite_skill(target[a], replica[a])
    comp_b = composite_skill(target[b], replica[b])
    grouped = composite_skill(target, replica, groups)
    assert grouped == pytest.approx(0.5 * (comp_a + comp_b))


# --------------------------------------------------------------------------- #
# TPE reproducibility (n_eff >= FLOOR)                                          #
# --------------------------------------------------------------------------- #
def test_tpe_reproducible_same_seed_budget(tmp_path: Path) -> None:
    """Two TPE runs with the SAME seed+budget on the same data agree on best_params.

    This is the contract's headline reproducibility invariant for the guided
    tier: a seeded :class:`optuna.samplers.TPESampler` must propose the identical
    trajectory, so the selected configuration is deterministic.
    """
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    common = dict(
        archetype="ts_momentum",
        cell="es1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=20,  # >= FLOOR -> TPE
        budget=_BUDGET,
        seed=0,
    )

    r1 = search_cell(ledger=Ledger(tmp_path / "a.json"), **common)
    r2 = search_cell(ledger=Ledger(tmp_path / "b.json"), **common)

    assert r1["tier"] == "tpe"
    assert r2["tier"] == "tpe"
    assert r1["best_params"] == r2["best_params"]
    assert r1["best_discrepancy"] == pytest.approx(r2["best_discrepancy"])
    # Same number of trials recorded each run (the frozen budget).
    assert len(r1["trials"]) == _BUDGET
    assert len(r2["trials"]) == _BUDGET


def test_tpe_different_seed_may_explore_differently(tmp_path: Path) -> None:
    """A different seed yields its own (still valid) recorded trajectory.

    Not a strict 'must differ' assertion (small discrete spaces can coincide);
    the point is that both runs complete and stay on the TPE tier.
    """
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    common = dict(
        archetype="ts_momentum",
        cell="es1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=20,
        budget=_BUDGET,
    )
    r_a = search_cell(ledger=Ledger(tmp_path / "a.json"), seed=1, **common)
    r_b = search_cell(ledger=Ledger(tmp_path / "b.json"), seed=2, **common)
    assert r_a["tier"] == "tpe" and r_b["tier"] == "tpe"
    assert len(r_a["trials"]) == _BUDGET and len(r_b["trials"]) == _BUDGET


# --------------------------------------------------------------------------- #
# Grid determinism (n_eff < FLOOR)                                             #
# --------------------------------------------------------------------------- #
def test_grid_deterministic_below_floor(tmp_path: Path) -> None:
    """A forced low-n_eff cell enumerates its whole grid and is deterministic.

    Two runs of the same low-``n_eff`` cell must return identical ``best_params``
    and ``best_discrepancy`` (no sampler, exhaustive product), and must evaluate
    exactly ``|product(param_space)|`` configs.
    """
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    common = dict(
        archetype="mean_reversion",
        cell="ng1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=FLOOR - 1,  # below floor -> grid
        budget=_BUDGET,
        seed=0,
    )
    r1 = search_cell(ledger=Ledger(tmp_path / "g1.json"), **common)
    r2 = search_cell(ledger=Ledger(tmp_path / "g2.json"), **common)

    n_grid = len(_PARAM_SPACE["shift"]) * len(_PARAM_SPACE["k"])
    assert r1["tier"] == "grid"
    assert len(r1["trials"]) == n_grid
    assert r1["best_params"] == r2["best_params"]
    assert r1["best_discrepancy"] == pytest.approx(r2["best_discrepancy"])
    # The exhaustive grid finds the exact optimum (shift == 0 -> perfect train).
    assert r1["best_params"]["shift"] == 0
    assert r1["best_discrepancy"] == pytest.approx(0.0, abs=1e-9)


def test_floor_boundary_selects_tpe(tmp_path: Path) -> None:
    """``n_eff == FLOOR`` is the guided (TPE) tier; one below is the grid tier."""
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    base = dict(
        archetype="ts_momentum",
        cell="es1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        budget=_BUDGET,
        seed=0,
    )
    at_floor = search_cell(ledger=Ledger(tmp_path / "at.json"), n_eff=FLOOR, **base)
    below = search_cell(ledger=Ledger(tmp_path / "below.json"), n_eff=FLOOR - 1, **base)
    assert at_floor["tier"] == "tpe"
    assert below["tier"] == "grid"


# --------------------------------------------------------------------------- #
# Train-based objective: perfect-on-train replica -> best_discrepancy ~ 0       #
# --------------------------------------------------------------------------- #
def test_objective_is_train_based(tmp_path: Path) -> None:
    """A generate_fn that perfectly matches the target on TRAIN yields disc ~ 0.

    Selection is by the TRAIN-slice discrepancy, so a replica that is exact on
    train (``shift == 0`` here) must be the winner with ``best_discrepancy ~ 0``
    on BOTH tiers, regardless of how it scores on val.
    """
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    base = dict(
        archetype="ts_momentum",
        cell="es1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        budget=_BUDGET,
        seed=0,
    )
    tpe = search_cell(ledger=Ledger(tmp_path / "t.json"), n_eff=20, **base)
    grid = search_cell(ledger=Ledger(tmp_path / "g.json"), n_eff=FLOOR - 1, **base)

    assert tpe["best_params"]["shift"] == 0
    assert tpe["best_discrepancy"] == pytest.approx(0.0, abs=1e-9)
    assert grid["best_params"]["shift"] == 0
    assert grid["best_discrepancy"] == pytest.approx(0.0, abs=1e-9)


def test_train_objective_independent_of_val_window(tmp_path: Path) -> None:
    """The selected best is driven by train; an empty val window still selects it.

    With ``val_idx`` empty, ``best_metrics`` is ``{}`` (no val to report) but the
    train-based selection is unchanged -- proving val never drives selection.
    """
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    res = search_cell(
        ledger=Ledger(tmp_path / "l.json"),
        archetype="ts_momentum",
        cell="es1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=np.array([], dtype=int),
        n_eff=FLOOR - 1,
        budget=_BUDGET,
        seed=0,
    )
    assert res["best_params"]["shift"] == 0
    assert res["best_discrepancy"] == pytest.approx(0.0, abs=1e-9)
    assert res["best_metrics"] == {}


# --------------------------------------------------------------------------- #
# Ledger receives trials, each tagged with a tier in {tpe, grid}               #
# --------------------------------------------------------------------------- #
def test_ledger_records_every_trial_with_tier(tmp_path: Path) -> None:
    """Every evaluated config is recorded with a tier in {tpe, grid}."""
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    ledger = Ledger(tmp_path / "ledger.json")
    res = search_cell(
        ledger=ledger,
        archetype="ts_momentum",
        cell="es1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=20,
        budget=_BUDGET,
        seed=0,
    )
    recorded = ledger.prior_trials("ts_momentum", "es1s")
    assert len(recorded) == _BUDGET
    assert len(res["trials"]) == _BUDGET
    for t in recorded:
        assert t["tier"] in {"tpe", "grid"}
        assert t["tier"] == "tpe"  # this cell is on the guided tier
        assert "params" in t and "shift" in t["params"]
        assert "train_discrepancy" in t
        # motivated_by present (read-before-propose provenance).
        assert isinstance(t["motivated_by"], list) and t["motivated_by"]


def test_ledger_grid_trials_tagged_grid(tmp_path: Path) -> None:
    """A below-floor cell records its trials with tier == 'grid'."""
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    ledger = Ledger(tmp_path / "ledger.json")
    search_cell(
        ledger=ledger,
        archetype="mean_reversion",
        cell="ng1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=2,
        budget=_BUDGET,
        seed=0,
    )
    recorded = ledger.prior_trials("mean_reversion", "ng1s")
    assert recorded
    assert all(t["tier"] == "grid" for t in recorded)


def test_read_before_propose_motivated_by_references_prior(tmp_path: Path) -> None:
    """A second search on the same cell cites prior trial ids in motivated_by."""
    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)
    ledger = Ledger(tmp_path / "ledger.json")

    common = dict(
        archetype="ts_momentum",
        cell="es1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=20,
        budget=_BUDGET,
        seed=0,
    )
    first = search_cell(ledger=ledger, **common)
    second = search_cell(ledger=ledger, **common)

    # The second run's recorded trials must cite prior (first-run) trial ids.
    first_ids = {t["id"] for t in first["trials"]}
    cited = set()
    for t in second["trials"]:
        cited.update(x for x in t["motivated_by"] if isinstance(x, int))
    assert cited & first_ids, "second search did not reference any prior trial id"


# --------------------------------------------------------------------------- #
# NO TEST ACCESS: get_test raising must not affect search_cell                 #
# --------------------------------------------------------------------------- #
def test_search_never_touches_test_split(tmp_path: Path, monkeypatch) -> None:
    """Monkeypatching get_test to ALWAYS raise leaves search_cell unaffected.

    ``search.py`` must never read the test block. We replace
    :func:`stml.replication.splits.get_test` with a tripwire that raises on ANY
    call; if the search reached test the run would error. It completes normally.
    """
    import stml.replication.splits as splits_mod

    def _boom(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("search_cell must never call get_test")

    monkeypatch.setattr(splits_mod, "get_test", _boom)

    target = _structured_target()
    gen = _shift_generate_fn(target)
    split = chronological_split(target.index)

    # Both tiers must complete with the tripwire armed.
    tpe = search_cell(
        ledger=Ledger(tmp_path / "tpe.json"),
        archetype="ts_momentum",
        cell="es1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=20,
        budget=_BUDGET,
        seed=0,
    )
    grid = search_cell(
        ledger=Ledger(tmp_path / "grid.json"),
        archetype="mean_reversion",
        cell="ng1s",
        param_space=_PARAM_SPACE,
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=2,
        budget=_BUDGET,
        seed=0,
    )
    assert tpe["tier"] == "tpe"
    assert grid["tier"] == "grid"
    # And the guard is genuinely armed: calling get_test now raises.
    with pytest.raises(AssertionError):
        splits_mod.get_test(split, final_confirmation=True)


# --------------------------------------------------------------------------- #
# Integration smoke: search one real instrument end to end                     #
# --------------------------------------------------------------------------- #
def test_search_real_instrument_end_to_end(tmp_path: Path) -> None:
    """A real instrument (cl1s) searches end to end via load_clean_data + align.

    Wires ``generate_fn`` to the ``mean_reversion`` archetype on cl1s, aligned to
    the released signal, and runs a small TPE search. The point is that align +
    archetype + search compose on real data and return a well-formed result that
    never touches test.
    """
    from stml.io import load_clean_data
    from stml.replication.align import align_instrument
    from stml.replication.archetypes import ARCHETYPES

    ohlcv, signals = load_clean_data()
    inst = "cl1s"

    aligned = align_instrument(signals, ohlcv, inst, convention="next_day")
    frame = aligned.frame.set_index("date")
    target = frame["signal"].astype(int)

    inst_ohlcv = ohlcv[ohlcv["instrument"] == inst].copy()
    arc = ARCHETYPES["mean_reversion"]

    def gen(params: dict) -> pd.Series:
        # Archetype signal reindexed onto the aligned target's index (no fill of
        # structural gaps beyond flat for warm-up rows the target requires).
        sig = arc.generate(inst_ohlcv, params)
        return sig.reindex(target.index).fillna(0).astype(int)

    split = chronological_split(target.index)
    res = search_cell(
        ledger=Ledger(tmp_path / "ledger.json"),
        archetype="mean_reversion",
        cell=inst,
        param_space=arc.param_space(),
        generate_fn=gen,
        target=target,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        n_eff=20,
        budget=8,
        seed=0,
    )
    assert res["tier"] == "tpe"
    assert set(res) == {"best_params", "best_metrics", "best_discrepancy", "tier", "trials"}
    assert "deadband" in res["best_params"]
    assert np.isfinite(res["best_discrepancy"])
    assert "kappa" in res["best_metrics"]
    assert len(res["trials"]) == 8
