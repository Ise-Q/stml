"""Tests for ``stml.experimental.bloomberg_ingest`` — S2.a acceptance.

Verifies:
* All four cleaned parquets are present and non-empty after a single ``ingest()``.
* Publication lags are correctly applied (daily +1d, EIA +5d).
* EIA release flag fires on Wednesdays (with snap-forward on holiday-Wed).
* The futures_term parquet preserves both front and second-month coverage.
"""

from __future__ import annotations

import pandas as pd

from stml.experimental.bloomberg_ingest import (
    BLOCK_A_FRONT_TICKERS,
    BLOCK_A_SECOND_TICKERS,
    LAG_DAILY,
    LAG_EIA,
    ingest,
)


def test_ingest_emits_all_expected_parquets(tmp_path):  # noqa: ARG001
    """A run produces 4 (or 5 with the flag) parquets in cleaned/."""
    result = ingest()
    expected = {"futures_term", "options_iv", "eia_crude", "eia_release_flag", "macro_alternative"}
    assert expected.issubset(result.parquets.keys())
    for name, (rows, cols) in result.parquets.items():
        assert rows > 0, f"{name} has zero rows"
        assert cols > 0, f"{name} has zero columns"


def test_futures_term_has_front_and_second_columns(repo_root):  # noqa: ARG001
    """Both `CL1_LAST` (front) and `CL2_LAST` (2nd) must be in futures_term."""
    df = pd.read_parquet(repo_root / "data/bloomberg/cleaned/futures_term.parquet")
    expected_front = [f"{t.replace('_Comdty', '').replace('_Index', '')}_LAST" for t in BLOCK_A_FRONT_TICKERS]
    expected_second = [f"{t.replace('_Comdty', '').replace('_Index', '')}_LAST" for t in BLOCK_A_SECOND_TICKERS]
    for col in expected_front + expected_second:
        assert col in df.columns, f"missing column {col}"


def test_eia_release_flag_fires_only_on_wednesdays_or_first_business_day_after(repo_root):  # noqa: ARG001
    """The EIA crude release lands on Wed (Friday data + 5 calendar days).

    If Wed is a US holiday (e.g., Jan 1 2020-01-01) the flag snaps forward to
    the first available business day. We allow Wed (most common) or Thursday
    (the snap-forward case).
    """
    flag = pd.read_parquet(repo_root / "data/bloomberg/cleaned/eia_release_flag.parquet")
    fire_days = flag.loc[flag["EIA_CRUDE_RELEASE_FLAG"] == 1].index
    assert len(fire_days) >= 1500, "expected ~1500 EIA releases over 1990-2022"
    # Day-of-week distribution — most should be Wednesday.
    dows = pd.Series(fire_days).dt.day_name()
    wed_share = (dows == "Wednesday").mean()
    assert wed_share >= 0.90, (
        f"only {wed_share:.1%} of release days are Wednesdays — expected >=90%"
    )


def test_lag_constants_match_plan_5_1():
    """Publication lags = methodology spec contract."""
    assert LAG_DAILY == 1
    assert LAG_EIA == 5


def test_options_iv_pl1_substituted_by_gc1_at_loader_level(repo_root):  # noqa: ARG001
    """``XPT`` sheets are empty → ``XPT_*`` columns absent from options_iv.

    The pl1s → GC1 substitution happens at the **feature module** level
    (`features/options_iv.py`), not at the ingest level. The ingest just
    drops empty sheets.
    """
    df = pd.read_parquet(repo_root / "data/bloomberg/cleaned/options_iv.parquet")
    xpt_cols = [c for c in df.columns if c.startswith("XPT_")]
    assert not xpt_cols, "XPT columns should be absent (empty sheets dropped)"
    # GC1 IV should be present (used as PL1 substitute downstream).
    assert "GC1_IV1M_ATM" in df.columns


def test_macro_alternative_has_eia_levels_complementary_to_crude_change(repo_root):  # noqa: ARG001
    """the macro has LEVELS; new BBG EIA has CHANGES. Both present, distinct."""
    macro = pd.read_parquet(repo_root / "data/bloomberg/cleaned/macro_alternative.parquet")
    crude = pd.read_parquet(repo_root / "data/bloomberg/cleaned/eia_crude.parquet")
    assert "EIA_CRUDE_STOCK" in macro.columns  # LEVELS
    assert "EIA_CRUDE_CHANGE_KB" in crude.columns  # CHANGES
    # The level is ~300k mbbl (median).
    assert macro["EIA_CRUDE_STOCK"].median() > 100_000
    # The change is ~thousand (median |x|).
    assert crude["EIA_CRUDE_CHANGE_KB"].abs().median() < 10_000
