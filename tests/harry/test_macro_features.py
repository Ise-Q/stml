"""test_macro_features.py — unit tests for macro feature groups M1–M6.

Structure
---------
Each group has:
  - A shape-preservation test (output rows == input rows, correct columns).
  - A toy-example sanity check (known-input, known-output).
  - A z-score mean-0 property check (after warmup, abs(mean) < 0.5 on a
    stationary AR(1) panel).

Truncation-invariance for every feature is covered by the universal
harness in test_causality.py (via CAUSALITY_REGISTRATIONS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stml.harry.features.macro_features import (
    MACRO_INSTRUMENT_TARGETS,
    m1_volatility_term_structure,
    m2_rates_curve,
    m3_credit,
)


# --------------------------------------------------------------------------- #
# Shared synthetic panel (stationary AR(1) for z-score mean tests)            #
# --------------------------------------------------------------------------- #
def _ar1(rng: np.random.Generator, mu: float, sigma: float, n: int, phi: float = 0.95) -> np.ndarray:
    """Stationary AR(1): mean ``mu``, stationary std ``sigma``, persistence ``phi``."""
    innov = rng.normal(0.0, sigma * np.sqrt(1.0 - phi ** 2), n)
    x = np.empty(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = mu + phi * (x[i - 1] - mu) + innov[i]
    return x


def _make_macro_panel(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Stationary AR(1) macro panel — all columns non-NaN.

    Uses mean-reverting AR(1) processes (phi=0.95) so that z-scores
    have mean close to zero, making the 'mean-0' property tests valid.
    EIA columns change every 5 rows; PMI columns every 21 rows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n, freq="B")

    # M1 — volatility (AR(1), stationary)
    vix   = np.clip(_ar1(rng, 16.0, 4.0,  n), 8.0,  80.0)
    vix3m = np.clip(vix + rng.normal(0.5, 0.3, n), 8.0,  85.0)
    move  = np.clip(_ar1(rng, 85.0, 12.0, n), 40.0, 200.0)
    skew  = np.clip(_ar1(rng, 115.0, 3.0, n), 100.0, 150.0)

    # M2 — rates (AR(1))
    ust10 = np.clip(_ar1(rng, 3.0, 0.5, n), 0.1, 10.0)
    ust2  = np.clip(_ar1(rng, 2.0, 0.4, n), 0.05, 9.0)
    bund  = np.clip(_ar1(rng, 1.0, 0.4, n), -0.9, 5.0)
    tips  = np.clip(_ar1(rng, 1.0, 0.4, n), -2.0, 5.0)
    be    = np.clip(_ar1(rng, 2.0, 0.2, n), 0.5,  4.0)

    # M3 — credit (AR(1))
    hy = np.clip(_ar1(rng, 450.0, 100.0, n), 100.0, 2000.0)
    ig = np.clip(_ar1(rng, 120.0, 30.0,  n),  20.0,  500.0)

    # M4 — FX (AR(1))
    dxy    = np.clip(_ar1(rng, 95.0, 3.0,  n), 70.0, 120.0)
    eurusd = np.clip(_ar1(rng, 1.10, 0.04, n),  0.8,   1.6)

    # M5 — EIA weekly (change every 5 rows)
    def _eia_ar1(start: float, step_std: float) -> np.ndarray:
        vals = np.empty(n, dtype=float)
        level = start
        for i in range(n):
            if i % 5 == 0:
                level += rng.normal(0, step_std)
            vals[i] = level
        return vals

    crude    = _eia_ar1(450_000.0, 2_000.0)
    dist_    = _eia_ar1(120_000.0, 1_000.0)
    gasoline = _eia_ar1(220_000.0, 1_500.0)
    ng       = _eia_ar1(3_000.0,   100.0)
    copper   = np.clip(_ar1(rng, 200_000.0, 20_000.0, n), 10_000.0, 600_000.0)
    bdi      = np.clip(_ar1(rng, 1_500.0,  300.0,    n),    200.0,  10_000.0)

    # M6 — PMI monthly (change every 21 rows)
    def _pmi_seq(center: float, std: float) -> np.ndarray:
        vals = np.empty(n, dtype=float)
        level = center + rng.normal(0, std)
        for i in range(n):
            if i % 21 == 0:
                level = center + rng.normal(0, std)
            vals[i] = level
        return vals

    ism   = _pmi_seq(52.0, 4.0)
    china = _pmi_seq(51.0, 3.0)

    df = pd.DataFrame(
        {
            "VIX": vix, "VIX3M": vix3m, "MOVE": move, "CBOE_SKEW": skew,
            "10Y_UST": ust10, "2Y_UST": ust2, "10Y_BUND": bund,
            "TIPS10Y": tips, "BE10Y": be,
            "HY_OAS": hy, "IG_OAS": ig,
            "DXY": dxy, "EURUSD": eurusd,
            "EIA_CRUDE_STOCK": crude, "EIA_DIST_STOCK": dist_,
            "EIA_GASOLINE_STOCK": gasoline, "EIA_NG_STOCK": ng,
            "LME_COPPER_STOCK": copper, "BAL_DRY_INDEX": bdi,
            "US_ISM_MFG_PMI": ism, "CHINA_PMI_MFG": china,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _tiny_panel(**cols: np.ndarray) -> pd.DataFrame:
    """Build a minimal macro panel from keyword column arrays."""
    n = len(next(iter(cols.values())))
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(cols, index=dates)


# --------------------------------------------------------------------------- #
# M1 — volatility / term structure                                            #
# --------------------------------------------------------------------------- #
class TestM1VolatilityTermStructure:
    def test_shape(self):
        df = _make_macro_panel()
        out = m1_volatility_term_structure(df, window=60)
        assert out.shape == (len(df), 6)
        assert list(out.columns) == [
            "vix_level_z", "vix_5d_change", "vix_term_slope",
            "move_z", "move_vix_ratio", "skew_z",
        ]

    def test_index_preserved(self):
        df = _make_macro_panel()
        out = m1_volatility_term_structure(df, window=60)
        assert out.index.equals(df.index)

    def test_vix_term_slope_sign(self):
        """When VIX3M > VIX, term slope must be positive (contango)."""
        n = 50
        vix   = np.full(n, 15.0)
        vix3m = np.full(n, 17.0)
        move  = np.full(n, 80.0)
        skew  = np.full(n, 115.0)
        df = _tiny_panel(VIX=vix, VIX3M=vix3m, MOVE=move, CBOE_SKEW=skew)
        out = m1_volatility_term_structure(df, window=10)
        slope = out["vix_term_slope"].dropna()
        assert (slope > 0).all(), "vix_term_slope must be positive when VIX3M > VIX"

    def test_vix_term_slope_exact(self):
        """vix_term_slope = VIX3M - VIX; exact value check."""
        n = 50
        df = _tiny_panel(
            VIX=np.full(n, 20.0), VIX3M=np.full(n, 23.0),
            MOVE=np.full(n, 80.0), CBOE_SKEW=np.full(n, 115.0),
        )
        out = m1_volatility_term_structure(df, window=10)
        assert np.allclose(out["vix_term_slope"].dropna(), 3.0, atol=1e-10)

    def test_vix_term_slope_backwardation(self):
        """When VIX3M < VIX, slope is negative (spot stress)."""
        n = 50
        df = _tiny_panel(
            VIX=np.full(n, 30.0), VIX3M=np.full(n, 25.0),
            MOVE=np.full(n, 120.0), CBOE_SKEW=np.full(n, 110.0),
        )
        out = m1_volatility_term_structure(df, window=10)
        assert (out["vix_term_slope"].dropna() < 0).all()

    def test_z_scores_mean_zero_after_warmup(self):
        df = _make_macro_panel(n=400)
        out = m1_volatility_term_structure(df, window=60)
        warmup = 60
        for col in ("vix_level_z", "move_z", "skew_z"):
            tail = out[col].iloc[warmup:]
            assert tail.notna().all(), f"{col} has NaN past warmup"
            assert abs(tail.mean()) < 0.5, f"{col} z-score mean={tail.mean():.3f}"

    def test_ratios_finite_after_warmup(self):
        df = _make_macro_panel(n=400)
        out = m1_volatility_term_structure(df, window=60)
        ratio = out["move_vix_ratio"]
        assert ratio.notna().all()
        assert np.isfinite(ratio).all()
        assert (ratio > 0).all()


# --------------------------------------------------------------------------- #
# M2 — rates / curve                                                          #
# --------------------------------------------------------------------------- #
class TestM2RatesCurve:
    def test_shape(self):
        df = _make_macro_panel()
        out = m2_rates_curve(df, window=60)
        assert out.shape == (len(df), 7)
        assert list(out.columns) == [
            "us_2s10s_slope", "ust_10y_5d_change", "bund_10y_5d_change",
            "ust_bund_spread", "real_yield_10y", "breakeven_10y", "be_5d_change",
        ]

    def test_index_preserved(self):
        df = _make_macro_panel()
        out = m2_rates_curve(df, window=60)
        assert out.index.equals(df.index)

    def test_2s10s_slope_exact(self):
        """us_2s10s_slope = 10Y - 2Y: exact value when constant."""
        n = 50
        df = _tiny_panel(
            **{"10Y_UST": np.full(n, 3.5), "2Y_UST": np.full(n, 1.5),
               "10Y_BUND": np.full(n, 1.0), "TIPS10Y": np.full(n, 1.0),
               "BE10Y": np.full(n, 2.5)},
        )
        out = m2_rates_curve(df, window=10)
        assert np.allclose(out["us_2s10s_slope"].dropna(), 2.0, atol=1e-10)

    def test_2s10s_positive_upward_curve(self):
        """10Y > 2Y → positive slope."""
        n = 50
        df = _tiny_panel(
            **{"10Y_UST": np.full(n, 3.5), "2Y_UST": np.full(n, 1.5),
               "10Y_BUND": np.full(n, 1.0), "TIPS10Y": np.full(n, 1.0),
               "BE10Y": np.full(n, 2.5)},
        )
        out = m2_rates_curve(df, window=10)
        assert (out["us_2s10s_slope"].dropna() > 0).all()

    def test_2s10s_negative_inverted_curve(self):
        """2Y > 10Y (inverted) → negative slope."""
        n = 50
        df = _tiny_panel(
            **{"10Y_UST": np.full(n, 3.0), "2Y_UST": np.full(n, 4.0),
               "10Y_BUND": np.full(n, 1.0), "TIPS10Y": np.full(n, 1.0),
               "BE10Y": np.full(n, 2.0)},
        )
        out = m2_rates_curve(df, window=10)
        assert (out["us_2s10s_slope"].dropna() < 0).all()

    def test_z_scores_mean_zero_after_warmup(self):
        df = _make_macro_panel(n=400)
        out = m2_rates_curve(df, window=60)
        for col in ("real_yield_10y", "breakeven_10y"):
            tail = out[col].iloc[60:]
            assert tail.notna().all(), f"{col} NaN past warmup"
            assert abs(tail.mean()) < 0.5, f"{col} mean={tail.mean():.3f}"

    def test_no_nan_past_warmup(self):
        df = _make_macro_panel(n=400)
        out = m2_rates_curve(df, window=60)
        for col in out.columns:
            tail = out[col].iloc[60:]
            assert tail.notna().all(), f"{col} NaN past warmup"
            assert np.isfinite(tail).all(), f"{col} Inf past warmup"


# --------------------------------------------------------------------------- #
# M3 — credit                                                                 #
# --------------------------------------------------------------------------- #
class TestM3Credit:
    def test_shape(self):
        df = _make_macro_panel()
        out = m3_credit(df, window=60)
        assert out.shape == (len(df), 4)
        assert list(out.columns) == [
            "hy_oas_z", "hy_oas_5d_change", "ig_oas_z", "hy_ig_ratio",
        ]

    def test_index_preserved(self):
        df = _make_macro_panel()
        out = m3_credit(df, window=60)
        assert out.index.equals(df.index)

    def test_hy_ig_ratio_positive(self):
        """HY spreads always > IG in realistic data → ratio > 0."""
        df = _make_macro_panel(n=400)
        out = m3_credit(df, window=60)
        assert (out["hy_ig_ratio"].dropna() > 0).all()

    def test_hy_ig_ratio_exact(self):
        """HY=400, IG=100 → ratio = 4.0."""
        n = 50
        df = _tiny_panel(HY_OAS=np.full(n, 400.0), IG_OAS=np.full(n, 100.0))
        out = m3_credit(df, window=10)
        assert np.allclose(out["hy_ig_ratio"].dropna(), 4.0, atol=1e-10)

    def test_z_scores_mean_zero_after_warmup(self):
        df = _make_macro_panel(n=400)
        out = m3_credit(df, window=60)
        for col in ("hy_oas_z", "ig_oas_z"):
            tail = out[col].iloc[60:]
            assert tail.notna().all()
            assert abs(tail.mean()) < 0.5, f"{col} mean={tail.mean():.3f}"

    def test_no_nan_past_warmup(self):
        df = _make_macro_panel(n=400)
        out = m3_credit(df, window=60)
        for col in out.columns:
            tail = out[col].iloc[60:]
            assert tail.notna().all(), f"{col} NaN past warmup"
            assert np.isfinite(tail).all(), f"{col} Inf past warmup"


# --------------------------------------------------------------------------- #
# Catalog sanity (M1–M3)                                                      #
# --------------------------------------------------------------------------- #
class TestMacroInstrumentTargets:
    def test_m1_m2_m3_features_documented(self):
        features = {
            "vix_level_z", "vix_5d_change", "vix_term_slope",
            "move_z", "move_vix_ratio", "skew_z",
            "us_2s10s_slope", "ust_10y_5d_change", "bund_10y_5d_change",
            "ust_bund_spread", "real_yield_10y", "breakeven_10y", "be_5d_change",
            "hy_oas_z", "hy_oas_5d_change", "ig_oas_z", "hy_ig_ratio",
        }
        missing = features - set(MACRO_INSTRUMENT_TARGETS.keys())
        assert not missing, f"Features missing from catalog: {missing}"

    def test_target_instruments_known(self):
        universe = {
            "es1s", "nq1s", "fesx1s",
            "cl1s", "ho1s", "rb1s", "ng1s",
            "gc1s", "si1s", "hg1s", "pl1s",
        }
        for feat, instruments in MACRO_INSTRUMENT_TARGETS.items():
            unknown = set(instruments) - universe
            assert not unknown, f"{feat} targets unknown instruments: {unknown}"
