"""S0 scaffold verification — methodology spec acceptance gates for Stage 0.

These tests do not exercise any modelling logic; they verify the package
imports cleanly, the determinism env is pinned, and the seed plumbing works.
A failure here means a downstream stage cannot be byte-stable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from stml.experimental import _env, config, seeding


def test_package_imports() -> None:
    """``from stml.experimental import _env, seeding, config`` must succeed."""
    # The imports at the top of this file are the test — if they fail, pytest
    # collection itself errors and we never reach the body. The explicit
    # ``hasattr`` calls let pytest report a friendlier message if the symbols
    # were ever moved.
    assert hasattr(_env, "applied_env")
    assert hasattr(seeding, "set_seeds")
    assert hasattr(seeding, "derive_seed")
    assert hasattr(config, "PipelineConfig")


def test_env_vars_pinned() -> None:
    """Single-thread env vars must be present after ``import stml.experimental``."""
    env = _env.applied_env()
    # The methodology spec contract: each pool pinned to 1 thread + libomp fix.
    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert env[key] == "1", f"{key} should be pinned to 1 thread"
    assert env["KMP_DUPLICATE_LIB_OK"] == "TRUE", "libomp dup-load fix missing"
    # PYTHONHASHSEED only takes effect when set before python launch; we still
    # set it for downstream subprocess inheritance.
    assert env["PYTHONHASHSEED"] == "42"


def test_set_seeds_makes_numpy_deterministic() -> None:
    """Two calls to ``set_seeds`` should produce identical numpy draws."""
    seeding.set_seeds(seed=42)
    a = np.random.rand(5)

    seeding.set_seeds(seed=42)
    b = np.random.rand(5)

    np.testing.assert_array_equal(a, b)


def test_derive_seed_is_position_unique_and_byte_stable() -> None:
    """``derive_seed(base, t)`` must depend on both args AND be replay-stable.

    This is the invariant that makes per-row bootstrap MC right-edge truncation
    invariant — see methodology spec A value at position ``t`` derived from
    ``derive_seed(base, t)`` is identical whether computed on the truncated or
    full panel.
    """
    base = 42
    # Same args -> same result, every time.
    assert seeding.derive_seed(base, 100) == seeding.derive_seed(base, 100)
    # Different positions -> different seeds.
    assert seeding.derive_seed(base, 100) != seeding.derive_seed(base, 101)
    # Different bases -> different seeds.
    assert seeding.derive_seed(base, 100) != seeding.derive_seed(base + 1, 100)
    # The seed is an int, fits in int64.
    s = seeding.derive_seed(42, 12345)
    assert isinstance(s, int)
    assert -(2**63) <= s < 2**63


def test_pipeline_config_is_frozen() -> None:
    """``PipelineConfig`` is a frozen dataclass — mutation must raise."""
    cfg = config.PipelineConfig()
    with pytest.raises(Exception):
        cfg.seed = 99  # type: ignore[misc]


def test_pipeline_config_defaults_match_plan() -> None:
    """Spot-check that defaults are wired to plan values (methodology spec / §6 / §10)."""
    cfg = config.PipelineConfig()
    # Plan §3.2 — t+1 entry; symmetric default pt=sl=0.5 (tighter than h=10 EWMA).
    assert cfg.max_holding == 10
    assert cfg.pt_mult == 0.5
    assert cfg.sl_mult == 0.5
    # Plan §3.2 — GARCH(1,1) refit cadence + sample bounds.
    assert cfg.sigma_source == "garch"
    assert cfg.garch_refit_every == 21
    assert cfg.garch_min_obs == 500
    # Plan §3.4 — CPCV (6,2) -> 15 paths.
    assert (cfg.cpcv_n_groups, cfg.cpcv_n_test_groups) == (6, 2)
    # Plan §4.19 — purged inner k-fold + 1SE.
    assert cfg.inner_kfold_k == 4
    assert cfg.selection_rule == "one_se"
    # Plan §4.19 — clean global cut + embargo.
    assert cfg.global_train_cut == "2021-10-06"
    assert cfg.embargo_end == "2021-10-20"
    # Madmoun Optional Session 3 -- 10 % target vol (slide 40); default
    # sizing = SOPS (slide 34); p* gate enabled by default (slide 21).
    assert cfg.target_vol == 0.10
    assert cfg.sizing_method == "sops"
    assert cfg.use_pstar_threshold is True
    assert cfg.ewma_sigma_span == 60
    # Grinold-Kahn cost model unchanged.
    assert cfg.half_spread_bps == 2.0
    assert cfg.impact_bps == 10.0


def test_asset_class_membership_covers_universe() -> None:
    """Every instrument in ``INSTRUMENTS`` belongs to exactly one asset class."""
    members_flat = [
        inst for members in config.ASSET_CLASS_MEMBERS.values() for inst in members
    ]
    assert sorted(members_flat) == sorted(config.INSTRUMENTS)
    # No duplicates within any class.
    for cls, members in config.ASSET_CLASS_MEMBERS.items():
        assert len(set(members)) == len(members), f"duplicate in {cls}"
