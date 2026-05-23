"""tests/test_features_scope.py
==============================
AC-8: D5 InstrumentScope registry tests.

All assertions run against real data via ``stml.io.load_clean_data``.
The verified ground-truth n_eff_gate table is pinned from CONTRACT_FE §2
and must match exactly.
"""

from __future__ import annotations

import json

import pytest

from stml.io import load_clean_data
from stml.metamodel.scope import (
    FLOOR,
    InstrumentScope,
    build_scope,
    persist_scope,
)
from stml.replication.splits import run_length_p90

# ---------------------------------------------------------------------------
# Verified ground-truth table (CONTRACT_FE §2, baked in per spec)
# ---------------------------------------------------------------------------

VERIFIED_N_EFF: dict[str, int] = {
    "es1s": 35,
    "nq1s": 20,
    "fesx1s": 25,
    "cl1s": 9,
    "ho1s": 9,
    "rb1s": 13,
    "ng1s": 2,
    "gc1s": 11,
    "si1s": 19,
    "hg1s": 29,
    "pl1s": 26,
}

LOW_POWER_INSTRUMENTS: frozenset[str] = frozenset({"cl1s", "ho1s", "ng1s"})

# Verified full-period run-length p90 (the downstream-CV embargo width), AC-6e.
VERIFIED_EMBARGO_P90: dict[str, int] = {
    "es1s": 10,
    "nq1s": 8,
    "fesx1s": 9,
    "cl1s": 14,
    "ho1s": 26,
    "rb1s": 19,
    "ng1s": 33,
    "gc1s": 12,
    "si1s": 7,
    "hg1s": 11,
    "pl1s": 8,
}

ASSET_CLASS_EXPECTED: dict[str, str] = {
    "es1s": "EQ",
    "nq1s": "EQ",
    "fesx1s": "EQ",
    "cl1s": "EN",
    "ho1s": "EN",
    "rb1s": "EN",
    "ng1s": "EN",
    "gc1s": "ME",
    "si1s": "ME",
    "hg1s": "ME",
    "pl1s": "ME",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scope() -> dict[str, InstrumentScope]:
    """Build scope once from real data; shared across tests in this module."""
    _, signals = load_clean_data()
    return build_scope(signals)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scope_has_exactly_11_instruments(scope: dict[str, InstrumentScope]) -> None:
    """build_scope returns exactly 11 InstrumentScope entries."""
    assert len(scope) == 11, f"Expected 11 instruments, got {len(scope)}: {list(scope)}"


def test_all_instruments_present(scope: dict[str, InstrumentScope]) -> None:
    """Every instrument in VERIFIED_N_EFF is present in the scope dict."""
    missing = set(VERIFIED_N_EFF) - set(scope)
    assert not missing, f"Missing instruments: {missing}"


def test_fit_scope_regime_per_instrument(scope: dict[str, InstrumentScope]) -> None:
    """Every instrument must have fit_scope_regime == 'per_instrument'."""
    wrong = {
        inst: sc.fit_scope_regime
        for inst, sc in scope.items()
        if sc.fit_scope_regime != "per_instrument"
    }
    assert not wrong, f"fit_scope_regime != 'per_instrument': {wrong}"


def test_fit_scope_latent_pooled_within_class(scope: dict[str, InstrumentScope]) -> None:
    """Every instrument must have fit_scope_latent == 'pooled_within_class'."""
    wrong = {
        inst: sc.fit_scope_latent
        for inst, sc in scope.items()
        if sc.fit_scope_latent != "pooled_within_class"
    }
    assert not wrong, f"fit_scope_latent != 'pooled_within_class': {wrong}"


def test_n_eff_gate_matches_verified_table(scope: dict[str, InstrumentScope]) -> None:
    """n_eff_gate must match the pinned verified table from CONTRACT_FE §2."""
    mismatches: dict[str, dict] = {}
    for inst, expected in VERIFIED_N_EFF.items():
        actual = scope[inst].n_eff_gate
        if actual != expected:
            mismatches[inst] = {"expected": expected, "actual": actual}
    assert not mismatches, (
        "n_eff_gate mismatches (CONTRACT_FE §2 verified table):\n"
        + "\n".join(f"  {k}: {v}" for k, v in mismatches.items())
    )


def test_low_power_flag_matches_floor(scope: dict[str, InstrumentScope]) -> None:
    """low_power is True iff instrument is in {cl1s, ho1s, ng1s} (n_eff_gate < FLOOR=10)."""
    wrong: dict[str, dict] = {}
    for inst, sc in scope.items():
        expected_low = inst in LOW_POWER_INSTRUMENTS
        if sc.low_power != expected_low:
            wrong[inst] = {
                "expected_low_power": expected_low,
                "actual_low_power": sc.low_power,
                "n_eff_gate": sc.n_eff_gate,
                "FLOOR": FLOOR,
            }
    assert not wrong, f"low_power flag mismatches: {wrong}"


def test_low_power_consistent_with_n_eff_gate(scope: dict[str, InstrumentScope]) -> None:
    """low_power == (n_eff_gate < FLOOR) for every instrument."""
    wrong = {
        inst: {"n_eff_gate": sc.n_eff_gate, "low_power": sc.low_power}
        for inst, sc in scope.items()
        if sc.low_power != (sc.n_eff_gate < FLOOR)
    }
    assert not wrong, f"low_power inconsistent with n_eff_gate < FLOOR: {wrong}"


def test_asset_class_assignment(scope: dict[str, InstrumentScope]) -> None:
    """asset_class matches the §2 map for every instrument."""
    wrong = {
        inst: {"expected": ASSET_CLASS_EXPECTED[inst], "actual": sc.asset_class}
        for inst, sc in scope.items()
        if sc.asset_class != ASSET_CLASS_EXPECTED.get(inst)
    }
    assert not wrong, f"asset_class mismatches: {wrong}"


def test_embargo_p90_persisted_and_matches_run_length(
    scope: dict[str, InstrumentScope],
) -> None:
    """AC-6e: per-instrument embargo (run_length_p90) is exposed for downstream CV.

    Each ``embargo_p90`` must be positive and equal the full-period
    ``run_length_p90`` of that instrument's signal, and match the pinned
    verified table.
    """
    _, signals = load_clean_data()
    wrong: dict[str, dict] = {}
    for inst, sc in scope.items():
        expected = run_length_p90(signals[inst])
        if sc.embargo_p90 != expected or sc.embargo_p90 <= 0:
            wrong[inst] = {"embargo_p90": sc.embargo_p90, "run_length_p90": expected}
    assert not wrong, f"embargo_p90 mismatch / non-positive: {wrong}"

    actual = {i: scope[i].embargo_p90 for i in VERIFIED_EMBARGO_P90}
    assert actual == VERIFIED_EMBARGO_P90, (
        f"embargo_p90 != verified table: {actual}"
    )


def test_persist_scope_round_trips(
    scope: dict[str, InstrumentScope], tmp_path
) -> None:
    """persist_scope writes valid JSON with all 11 instruments; all fields round-trip."""
    dest = tmp_path / "instrument_scope.json"
    persist_scope(scope, path=dest)

    assert dest.exists(), "persist_scope did not create the output file"
    raw = json.loads(dest.read_text())

    # Must have exactly 11 instruments.
    assert len(raw) == 11, f"JSON has {len(raw)} entries, expected 11"

    # Every instrument and every field must round-trip.
    for inst, sc in scope.items():
        assert inst in raw, f"Instrument {inst!r} missing from JSON"
        rec = raw[inst]
        assert rec["instrument"] == sc.instrument
        assert rec["asset_class"] == sc.asset_class
        assert rec["n_eff_gate"] == sc.n_eff_gate
        assert rec["embargo_p90"] == sc.embargo_p90
        assert rec["fit_scope_regime"] == sc.fit_scope_regime
        assert rec["fit_scope_latent"] == sc.fit_scope_latent
        assert rec["low_power"] == sc.low_power


def test_persist_scope_creates_parent_dirs(
    scope: dict[str, InstrumentScope], tmp_path
) -> None:
    """persist_scope creates parent directories when they do not exist."""
    dest = tmp_path / "nested" / "deep" / "scope.json"
    persist_scope(scope, path=dest)
    assert dest.exists()


def test_instrument_scope_dataclass_fields(scope: dict[str, InstrumentScope]) -> None:
    """Each entry is an InstrumentScope with the expected field names."""
    for inst, sc in scope.items():
        assert isinstance(sc, InstrumentScope), f"{inst} is not InstrumentScope"
        assert hasattr(sc, "instrument")
        assert hasattr(sc, "asset_class")
        assert hasattr(sc, "n_eff_gate")
        assert hasattr(sc, "fit_scope_regime")
        assert hasattr(sc, "fit_scope_latent")
        assert hasattr(sc, "low_power")
        assert sc.instrument == inst
