"""
test_run_replicate.py
=====================
LIGHT smoke test for the replication-search orchestrator
:mod:`stml.replication.run_replicate` (US-011).

Kept fast by running ONE family (``mean_reversion`` -- the C1 prior-best) with a
tiny TPE budget. Every artifact is written under ``tmp_path`` (and the ledger is
pinned to a temp file), so the smoke run never clobbers the real, full deliverables
in ``reports/`` and ``results/jj/``. The test asserts STRUCTURE and INVARIANTS --
the artifacts exist, the matrix/top-candidates are well-formed, the gate booleans
are consistent, and the pool cell is never a standalone pass -- not exact metric
numbers, which come from the already-tested upstream modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stml.replication import gates
from stml.replication.run_replicate import (
    POOL_CLASS,
    POOL_MEMBERS,
    POOL_NAME,
    STANDALONE_CELLS,
    _neighbours,
    _pool_cell,
    _run_cell,
    main,
    run,
)
from stml.replication.archetypes import ARCHETYPES
from stml.replication.ledger import Ledger
from stml.replication.splits import Split, chronological_split

SMOKE_FAMILY = "mean_reversion"
SMOKE_BUDGET = 4

# A second family for the multi-family consistency cross-check. Kept to two
# families with a tiny budget so the smoke run stays fast and tmp-path isolated.
CONSISTENCY_FAMILIES = ["mean_reversion", "ts_momentum"]


def test_run_writes_artifacts_under_tmp_path(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.json"
    result = run(
        [SMOKE_FAMILY],
        out_dir=str(tmp_path),
        budget=SMOKE_BUDGET,
        seed=0,
        ledger_path=str(ledger_path),
    )

    paths = result["paths"]
    summary_path = paths["summary"]
    top_path = paths["top_candidates"]
    ledger_md_path = paths["ledger_md"]
    family_reports = paths["family_reports"]

    # Isolation guarantee: every artifact lives under the temp dir, not the repo.
    assert isinstance(summary_path, Path)
    assert tmp_path in summary_path.parents
    assert tmp_path in top_path.parents
    assert tmp_path in ledger_md_path.parents
    assert ledger_path.exists()  # the ledger went to the temp file, not the repo

    # The summary + per-family report exist and are non-empty.
    assert summary_path.exists() and summary_path.stat().st_size > 0
    assert summary_path.name == "replication-summary.md"
    assert SMOKE_FAMILY in family_reports
    fam_report = family_reports[SMOKE_FAMILY]
    assert tmp_path in fam_report.parents
    assert fam_report.exists() and fam_report.stat().st_size > 0
    assert fam_report.name == f"{SMOKE_FAMILY}.md"

    # The summary names the family and renders the pass/fail matrix + headline.
    summary_text = summary_path.read_text(encoding="utf-8")
    assert SMOKE_FAMILY in summary_text
    assert "pass/fail matrix" in summary_text.lower()
    assert "headline" in summary_text.lower()


def test_top_candidates_is_valid_json_with_gate_booleans(tmp_path) -> None:
    result = run(
        [SMOKE_FAMILY],
        out_dir=str(tmp_path),
        budget=SMOKE_BUDGET,
        seed=0,
        ledger_path=str(tmp_path / "ledger.json"),
    )
    top_path = result["paths"]["top_candidates"]

    payload = json.loads(top_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    # Every entry carries the contract schema with the four gate booleans.
    for entry in payload:
        assert entry["archetype"] == SMOKE_FAMILY
        assert "cell" in entry and "params" in entry and "tier" in entry
        assert "n_eff" in entry and "val_metrics" in entry
        for g in ("g1", "g2", "g3", "g4", "passed"):
            assert isinstance(entry["gates"][g], bool)
        # A listed top candidate must actually have passed all four gates.
        assert entry["gates"]["passed"] is True


def test_invariants_pool_never_standalone_pass_and_matrix_shape(tmp_path) -> None:
    result = run(
        [SMOKE_FAMILY],
        out_dir=str(tmp_path),
        budget=SMOKE_BUDGET,
        seed=0,
        ledger_path=str(tmp_path / "ledger.json"),
    )
    summary = result["summary"]

    # The success flag agrees with the count (one family can never reach 5).
    assert summary["success"] is (summary["n_families_passed"] >= 5)
    assert summary["n_families_passed"] == len(summary["families_passed"])

    # Passing cells are drawn only from the LOCKED universe: the eight standalone
    # instruments plus the single energy pool. The three thin energy members
    # (cl1s/ho1s/ng1s) are never evaluated as standalone cells -- they live only
    # inside pool:energy -- so none of them can appear here.
    passing_cells = summary["passing_cells"][SMOKE_FAMILY]
    assert isinstance(passing_cells, list)
    assert set(passing_cells) <= ({*STANDALONE_CELLS, POOL_NAME})
    assert not ({"cl1s", "ho1s", "ng1s"} & set(passing_cells))

    # If the single family passed, the one-shot test confirmation ran for it
    # (val/test composites present); otherwise no test read happened.
    tc = summary["test_consistency"]
    if SMOKE_FAMILY in summary["families_passed"]:
        assert len(tc) == 1
        assert tc[0]["family"] == SMOKE_FAMILY
        assert "val_composite" in tc[0] and "test_composite" in tc[0]
    else:
        assert tc == []


def test_main_cli_smoke(tmp_path) -> None:
    # Drive the argparse entry point exactly as the CLI would, into tmp_path.
    result = main(
        [
            "--families",
            SMOKE_FAMILY,
            "--out-dir",
            str(tmp_path),
            "--budget",
            str(SMOKE_BUDGET),
            "--seed",
            "0",
        ]
    )
    assert result["summary"]["n_families_passed"] == len(
        result["summary"]["families_passed"]
    )
    assert (tmp_path / "reports" / "replication-summary.md").exists()
    # FLOOR is the shared anti-overfit constant; the orchestrator imports it
    # from gates rather than redefining it.
    assert gates.FLOOR == 10


# --------------------------------------------------------------------------- #
# FIX 1: G3 neighbourhood perturbs NUMERIC axes only, holds CATEGORICAL fixed. #
# --------------------------------------------------------------------------- #
def test_neighbours_hold_categorical_axes_fixed() -> None:
    """The +/-1 G3 neighbourhood must vary only numeric axes.

    Flipping ``base`` (mean_reversion <-> ts_momentum), ``regime`` (high <->
    low) or ``score`` (momentum <-> reversal) is a strategy switch, not a tuning
    perturbation, so every neighbour must keep those at the winner's value while
    still moving at least one numeric axis.
    """
    # vol_regime_gated carries two categorical axes (base, regime) + numerics.
    vrg = ARCHETYPES["vol_regime_gated"].param_space()
    best = {
        "base": "mean_reversion",
        "lookback": 20,
        "z_window": 40,
        "vol_window": 20,
        "regime": "high",
        "vol_quantile": 0.5,
        "q_window": 120,
        "deadband": 0.5,
    }
    neighbours = _neighbours(best, vrg)
    assert neighbours, "expected a non-empty numeric neighbourhood"
    for n in neighbours:
        assert n["base"] == best["base"]
        assert n["regime"] == best["regime"]
        # A neighbour differs from the winner in exactly the numeric axes.
        diffs = {k for k in best if n[k] != best[k]}
        assert diffs and diffs <= {"lookback", "z_window", "vol_window",
                                   "vol_quantile", "q_window", "deadband"}

    # xsect_rank: the categorical 'score' axis is held fixed too.
    xr = ARCHETYPES["xsect_rank"].param_space()
    xbest = {"lookback": 20, "top_frac": 0.3, "score": "momentum"}
    xneigh = _neighbours(xbest, xr)
    assert xneigh
    assert all(n["score"] == "momentum" for n in xneigh)

    # A fully-numeric family still gets a real neighbourhood (no regression).
    mr = ARCHETYPES["mean_reversion"].param_space()
    mneigh = _neighbours({"lookback": 20, "z_window": 40, "deadband": 0.5}, mr)
    assert mneigh


# --------------------------------------------------------------------------- #
# FIX 3: pooled-cell gateable-n_eff semantics.                                #
# --------------------------------------------------------------------------- #
def _perfect_cell(n_rows: int = 60) -> tuple[pd.Series, pd.Series, pd.Series, Split]:
    """A perfectly-replicated 3-state cell + aligned returns + a train/val split.

    Returns ``(replica, target, aligned_ret, split)`` where the replica equals
    the target (kappa ~ 1), the returns make the target's PnL positive (so the
    NAV increment correlation is > 0), and the split puts the first 60% of dates
    in train, the rest in val. This is the substrate for asserting the pooled
    gating semantics independent of any archetype/search.
    """
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="D")
    pattern = np.array([1, 0, -1, 1, -1, 0], dtype=int)
    sig = np.resize(pattern, n_rows)
    target = pd.Series(sig, index=idx, name="signal")
    replica = target.copy()
    # Returns aligned so s_t * r_{t+1} > 0 on every nonzero day -> positive corr.
    aligned_ret = pd.Series(np.sign(sig).astype(float) * 0.01, index=idx)
    cut = int(n_rows * 0.6)
    split = Split(
        train_idx=np.arange(cut),
        val_idx=np.arange(cut, n_rows),
        test_idx=np.empty(0, dtype=int),
        train_dates=idx[:cut],
        val_dates=idx[cut:],
        test_dates=idx[:0],
    )
    return replica, target, aligned_ret, split


_THIN_THRESHOLDS = {"suggested_cutoffs": {"kappa": 0.05, "ordinal_skill": 0.05}}


def test_pool_summed_n_eff_gated_first_class() -> None:
    """A pooled cell with summed n_eff >= FLOOR is gated first-class.

    The energy pool's gateable n_eff is the SUM of its members' post-embargo val
    regime-calls (9 + 9 + 2 = 20 >= FLOOR). At/above the FLOOR, ``gate_cell``
    routes straight through ``evaluate`` -- the verdict is NOT tagged low-power
    and CAN be a pass (here a perfect replica clears all four gates).
    """
    replica, target, aligned_ret, split = _perfect_cell()
    pooled_n_eff = 9 + 9 + 2  # the documented energy-pool sum
    assert pooled_n_eff >= gates.FLOOR

    res = gates.gate_cell(
        replica,
        target,
        aligned_ret,
        split,
        _THIN_THRESHOLDS,
        pooled_n_eff,
        n_configs=24,
        perturbed_metrics=None,
    )
    # First-class: not forced low-power, and a perfect replica passes.
    assert res.details.get("low_power") is None
    assert res.passed is True


def test_single_below_floor_instrument_is_low_power_never_standalone() -> None:
    """A lone below-FLOOR instrument is low-power and never a standalone pass.

    Even a perfect replica on a thin standalone cell (n_eff = 2 < FLOOR) is
    routed through pooling, tagged ``low_power=True``, and has ``passed`` forced
    to ``False`` -- a thin member can never earn a standalone pass, whatever the
    pool says.
    """
    replica, target, aligned_ret, split = _perfect_cell()
    res = gates.gate_cell(
        replica,
        target,
        aligned_ret,
        split,
        _THIN_THRESHOLDS,
        n_eff=2,  # below FLOOR
        n_configs=24,
        pooled_replica_signal=replica,
        pooled_target_signal=target,
        pooled_aligned_ret=aligned_ret,
        pooled_split=split,
        pooled_thresholds_entry=_THIN_THRESHOLDS,
        pooled_n_eff=20,
        perturbed_metrics=None,
    )
    assert res.details.get("low_power") is True
    assert res.passed is False


# --------------------------------------------------------------------------- #
# FIX 7: matrix <-> top_candidates.json <-> ledger winner records agree.       #
# --------------------------------------------------------------------------- #
def test_matrix_top_candidates_ledger_agree_on_passes(tmp_path) -> None:
    """The summary matrix, top_candidates.json and ledger agree on passes.

    Runs two families on a tiny budget into tmp_path, then cross-checks that the
    set of passing (family, cell) is identical across (a) the in-memory summary
    (the matrix source), (b) ``top_candidates.json``, and (c) the ledger's
    per-(family, cell) gate-result winner records. This would have caught the
    xsect_rank contradiction class (a cell shown PASS in the matrix but narrated
    as a failure, or disagreeing artifacts).
    """
    ledger_path = tmp_path / "ledger.json"
    result = run(
        CONSISTENCY_FAMILIES,
        out_dir=str(tmp_path),
        budget=SMOKE_BUDGET,
        seed=0,
        ledger_path=str(ledger_path),
    )

    # (a) Passing (family, cell) from the in-memory summary (matrix source).
    passing_cells = result["summary"]["passing_cells"]
    matrix_pass = {
        (fam, cell) for fam, cells in passing_cells.items() for cell in cells
    }

    # (b) Passing (family, cell) from top_candidates.json.
    top = json.loads(
        result["paths"]["top_candidates"].read_text(encoding="utf-8")
    )
    top_pass = {
        (e["archetype"], e["cell"]) for e in top if e["gates"]["passed"] is True
    }
    # Every top candidate is a genuine pass present in the matrix.
    assert top_pass <= matrix_pass
    # top_candidates keeps the single best passing cell per passing family, so
    # its family set must match the matrix's passing-family set exactly.
    assert {f for f, _ in top_pass} == {f for f, _ in matrix_pass}

    # (c) The ledger's winner records (the post-search gate trials carrying a
    # populated gate_result) must agree on which (family, cell) passed.
    led = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_pass = {
        (t["archetype"], t["cell"])
        for t in led
        if isinstance(t.get("gate_result"), dict)
        and t["gate_result"].get("passed") is True
    }
    assert ledger_pass == matrix_pass


# --------------------------------------------------------------------------- #
# FIX: pool cells are gated with WITHIN-instrument (group-averaged) skill.      #
# --------------------------------------------------------------------------- #
def test_pool_recorded_kappa_is_group_averaged_not_concatenated(tmp_path) -> None:
    """A pool cell's recorded val kappa is the per-member mean, not concatenated.

    Builds the real energy pool for ``ts_momentum`` (the family proven to
    ANTI-replicate cl1s/ho1s while the concatenated pool kappa looked positive),
    searches it on a tiny budget, and asserts the FROZEN winner's recorded pool
    val kappa equals the equal-weight MEAN of its per-member within-instrument val
    kappas -- i.e. group-averaging is what is gated. It must NOT equal the old
    concatenated-panel kappa whenever the two differ, locking the artifact fix.
    """
    from stml.io import load_clean_data
    from stml.replication.run_replicate import _read_thresholds, _repo_root

    ohlcv, signals = load_clean_data()
    ohlcv = ohlcv[ohlcv["date"] >= "2019-01-01"].copy()
    thresholds = _read_thresholds(_repo_root())
    split = chronological_split(signals["date"])
    archetype = ARCHETYPES["ts_momentum"]

    pool_data = _pool_cell(
        archetype=archetype,
        signals=signals,
        ohlcv=ohlcv,
        split=split,
        thresholds_entry=thresholds["per_asset_class"][POOL_CLASS],
    )
    # The pool carries one group label per row, aligned to its synthetic index.
    assert pool_data.groups is not None
    assert len(pool_data.groups) == len(pool_data.target)
    assert set(pool_data.groups.tolist()) == set(POOL_MEMBERS)

    ledger = Ledger(tmp_path / "ledger.json")
    res = _run_cell(
        family="ts_momentum",
        archetype=archetype,
        cell_data=pool_data,
        pool_cell_data=pool_data,
        ledger=ledger,
        budget=SMOKE_BUDGET,
        seed=0,
    )

    assert res.cell == POOL_NAME
    pm = res.pool_member_metrics
    assert pm is not None
    # The recorded pool val kappa equals the equal-weight mean of the per-member
    # within-instrument val kappas (group-averaging), to floating tolerance.
    member_kappas = [v for v in pm["members"].values() if v is not None]
    assert member_kappas
    expected_mean = sum(member_kappas) / len(member_kappas)
    assert res.val_metrics["kappa"] == pytest.approx(expected_mean, abs=1e-9)
    assert res.val_metrics["kappa"] == pytest.approx(pm["group_avg_kappa"], abs=1e-9)

    # And it is genuinely NOT the concatenated metric (the old artifact), which
    # differs here because the members do not share a base rate.
    assert pm["concatenated_kappa"] is not None
    assert res.val_metrics["kappa"] != pytest.approx(
        pm["concatenated_kappa"], abs=1e-6
    )

    # The pool is gated first-class (summed n_eff >= FLOOR) but on grouped metrics.
    assert res.gate_result.details.get("grouped") is True
    assert res.gate_result.details.get("low_power") is None
