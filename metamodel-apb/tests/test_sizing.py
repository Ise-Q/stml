"""Known-value tests for fractional-Kelly sizing + vol targeting (Stage 1, RED-first).

Closed forms (nlr-cw §7; Kelly 1956; MacLean-Ziemba-Blazenko 1992; Carver 2015):
  Kelly fraction  f* = (p̂·b − (1−p̂)·d) / (b·d)
  Fractional size = κ·f*, zero when p̂ < floor (0.55), clipped to [0, cap].
  Vol-target leverage = target_vol / realised_vol.
"""

from __future__ import annotations

import numpy as np
import pytest

from alken_metamodel.sizing import (
    CONFIDENCE_FLOOR,
    cer_improves,
    confidence_taper,
    fractional_kelly,
    kappa_baker_mchale,
    kelly_fraction,
    position_weight,
    vol_target_leverage,
)


@pytest.mark.parametrize(
    ("p", "b", "d", "expected"),
    [
        (0.6, 1.0, 1.0, 0.2),  # (0.6-0.4)/1
        (0.55, 2.0, 1.0, 0.325),  # (1.1-0.45)/2
        (0.7, 1.5, 1.0, 0.5),  # (1.05-0.3)/1.5
        (0.5, 1.0, 1.0, 0.0),  # no edge
    ],
)
def test_kelly_fraction_closed_form(p, b, d, expected):
    assert kelly_fraction(p, b, d) == pytest.approx(expected)


def test_below_confidence_floor_is_zero():
    assert fractional_kelly(0.5499, 1.0, 1.0) == 0.0
    assert fractional_kelly(0.54, 2.0, 1.0) == 0.0
    # at/above the floor it sizes
    assert fractional_kelly(0.55, 2.0, 1.0) > 0.0


def test_kappa_scales_linearly():
    quarter = fractional_kelly(0.6, 1.0, 1.0, kappa=0.25)
    half = fractional_kelly(0.6, 1.0, 1.0, kappa=0.5)
    assert quarter == pytest.approx(0.25 * 0.2)  # κ·f*
    assert half == pytest.approx(2.0 * quarter)


def test_fractional_kelly_is_bounded():
    # tiny stop-loss d blows up raw f*; fractional size must clip to cap
    raw = kelly_fraction(0.9, 1.0, 0.01)
    assert raw > 10  # unbounded
    assert fractional_kelly(0.9, 1.0, 0.01, kappa=0.25, cap=1.0) == 1.0
    # never negative for an "act" decision
    assert fractional_kelly(0.56, 1.0, 5.0, kappa=0.25, cap=1.0) >= 0.0


def test_vol_target_leverage():
    assert vol_target_leverage(0.50, target_vol=0.25) == pytest.approx(0.5)
    assert vol_target_leverage(0.125, target_vol=0.25) == pytest.approx(2.0)
    assert vol_target_leverage(0.25, target_vol=0.25) == pytest.approx(1.0)


def test_vol_target_leverage_capped_and_safe():
    assert vol_target_leverage(0.001, target_vol=0.25, max_leverage=5.0) == 5.0  # capped
    assert vol_target_leverage(0.0, target_vol=0.25, max_leverage=5.0) == 5.0  # no div-by-zero


def test_position_weight_signs_with_side_and_zeros_below_floor():
    # long, confident -> positive weight
    w_long = position_weight(side=1, p=0.7, b=1.0, d=1.0, realised_vol=0.25, target_vol=0.25)
    assert w_long > 0
    # short, confident -> negative weight, same magnitude
    w_short = position_weight(side=-1, p=0.7, b=1.0, d=1.0, realised_vol=0.25, target_vol=0.25)
    assert w_short == pytest.approx(-w_long)
    # below floor -> flat regardless of side
    assert position_weight(side=1, p=0.5, b=1.0, d=1.0, realised_vol=0.25) == 0.0


# --- S6.15: CER-gated sizing (smooth taper + per-instrument kappa) -----------


def test_confidence_taper_endpoints_and_monotone():
    width = 0.05
    assert confidence_taper(CONFIDENCE_FLOOR - width, width=width) == pytest.approx(0.0)
    assert confidence_taper(CONFIDENCE_FLOOR + width, width=width) == pytest.approx(1.0)
    assert confidence_taper(CONFIDENCE_FLOOR, width=width) == pytest.approx(0.5)  # smoothstep mid
    grid = np.linspace(0.45, 0.65, 50)
    t = confidence_taper(grid, width=width)
    assert np.all(np.diff(t) >= -1e-12)  # monotone non-decreasing
    assert np.all((t >= 0.0) & (t <= 1.0))


def test_taper_is_continuous_where_hard_floor_jumps():
    eps = 1e-4
    hard_jump = abs(fractional_kelly(0.55 + eps, 1.0, 1.0) - fractional_kelly(0.55 - eps, 1.0, 1.0))
    taper_jump = abs(
        fractional_kelly(0.55 + eps, 1.0, 1.0, taper_width=0.05)
        - fractional_kelly(0.55 - eps, 1.0, 1.0, taper_width=0.05)
    )
    assert hard_jump > 0.02  # hard floor discontinuously jumps to ~κ·f*(0.55)
    assert taper_jump < 1e-3  # the taper is continuous at the old floor


def test_kappa_baker_mchale_formula_and_monotone():
    assert kappa_baker_mchale(0.1, 0.0) == pytest.approx(1.0)  # no residual var -> full conviction
    assert kappa_baker_mchale(0.0, 0.1) == pytest.approx(0.0)  # no edge -> no bet
    assert kappa_baker_mchale(0.1, 0.01) == pytest.approx(0.5)  # e²=σ²=0.01 -> ½
    resid_var = np.array([0.001, 0.01, 0.1, 1.0])
    k = kappa_baker_mchale(0.1, resid_var)
    assert np.all(np.diff(k) < 0)  # strictly decreasing in residual variance
    assert np.all((k >= 0.0) & (k <= 1.0))


def test_cer_gate_adopts_only_on_strict_gain():
    assert cer_improves(0.50, 0.40) is True
    assert cer_improves(0.40, 0.50) is False
    assert cer_improves(0.40, 0.40) is False  # tie -> revert to flat kappa
    assert cer_improves(0.401, 0.40, min_gain=0.005) is False  # below margin -> revert
