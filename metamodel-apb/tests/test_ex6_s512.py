"""EX.6 (leakage-safe κᵢ) + S5.12 (standardise-then-repool TM) — RED-first known-answer tests.

The two helpers live in the Stage-6 runner ``experiments/s6_barrier_backtest.py`` (a non-deliverable
diagnostic). The runner inserts its own directory on import, so we add ``experiments/`` to the path
and import it as a top-level module — exactly how it imports ``_common``.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_EXPERIMENTS = pathlib.Path(__file__).resolve().parent.parent / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import s6_barrier_backtest as s6  # noqa: E402

from alken_metamodel.signal_analysis import treynor_mazuy  # noqa: E402
from alken_metamodel.sizing import kappa_baker_mchale  # noqa: E402

_OUTPUTS = pathlib.Path(__file__).resolve().parent.parent / "outputs"


# --- EX.6: leakage-safe per-instrument κᵢ ------------------------------------


def _model_oof(dates, instruments, p_hat, y) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "instrument": list(instruments),
            "p_hat": np.asarray(p_hat, dtype=float),
            "y": np.asarray(y, dtype=float),
        }
    )


def test_leakage_safe_kappa_window_strictly_pre_predict_start():
    """Non-circularity guard: κᵢ must REJECT any row dated >= predict_start (the pass-4 trap)."""
    predict_start = pd.Timestamp("2022-01-01")
    bad = _model_oof(["2021-06-01", "2022-02-01"], ["es1s", "es1s"], [0.60, 0.60], [1, 1])
    with pytest.raises(AssertionError):
        s6.leakage_safe_kappa(bad, predict_start)
    good = _model_oof(["2021-06-01", "2021-09-01"], ["es1s", "es1s"], [0.62, 0.58], [1, 0])
    out = s6.leakage_safe_kappa(good, predict_start)
    assert set(out) == {"es1s"}
    assert 0.0 <= out["es1s"] <= 1.0


def test_leakage_safe_kappa_matches_baker_mchale_and_decreases_in_sigma():
    """Same formula as the circular diagnostic (np.var, ddof=0) — only the window moves. The
    tight (low-variance) sleeve earns a higher κ than the noisy one at equal edge."""
    ps = pd.Timestamp("2022-01-01")
    tight = _model_oof(["2021-01-01"] * 4, ["a"] * 4, [0.60, 0.61, 0.59, 0.60], [1, 1, 0, 1])
    noisy = _model_oof(["2021-01-01"] * 4, ["b"] * 4, [0.60, 0.95, 0.25, 0.60], [1, 1, 0, 1])
    frame = pd.concat([tight, noisy], ignore_index=True)
    out = s6.leakage_safe_kappa(frame, ps)
    for inst, g in frame.groupby("instrument"):
        p = g["p_hat"].to_numpy()
        assert out[inst] == pytest.approx(
            float(kappa_baker_mchale(abs(float(np.mean(p)) - 0.5), float(np.var(p))))
        )
    assert out["a"] > out["b"]  # κ decreases in the residual/variance dispersion


def test_revert_path_emit_is_byte_identical(tmp_path):
    """The expected EX.6 revert means NO re-emit; the flat-κ emit is order-invariant + byte-stable
    (the determinism contract that makes 'revert ⇒ byte-identical' trivially true)."""
    from alken_metamodel.emit import WEIGHT_COLUMNS, _emit

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-03"]),
            "instrument": ["es1s", "cl1s"],
            "weight": [0.1234567891, -0.05],
        }
    )
    p1, p2 = tmp_path / "w1.csv", tmp_path / "w2.csv"
    _emit(df, p1, WEIGHT_COLUMNS)
    _emit(df.sample(frac=1.0, random_state=1), p2, WEIGHT_COLUMNS)  # row order must not matter
    assert p1.read_bytes() == p2.read_bytes()


# --- S5.12: standardise-then-repool TM ---------------------------------------


def _sleeve(scale, gamma_sign, n=400, seed=0):
    """A synthetic sleeve with pnl=side·mkt+planted convexity at a chosen return scale (side=+1)."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0, scale, size=n)
    side = np.ones(n)
    pnl = side * mkt + gamma_sign * (mkt**2)
    return mkt, pnl


def test_s512_standardisation_is_per_sleeve_common_scale():
    """Each sleeve is vol-targeted to unit return-std by dividing BOTH mkt and pnl by the SAME
    factor — so the pnl=side·mkt+conv identity is preserved and large-scale sleeves stop
    dominating the pooled quadratic."""
    big = _sleeve(0.10, +1.0, seed=1)
    small = _sleeve(0.01, -1.0, seed=2)
    sleeves = [("big", *big, 0.0, len(big[0])), ("small", *small, 0.0, len(small[0]))]
    _g, _t, _p, _wavg, dbg = s6.standardise_then_repool_tm(sleeves, _return_debug=True)
    for m_std, p_std in dbg["per_sleeve"]:
        assert np.std(m_std, ddof=1) == pytest.approx(1.0, rel=1e-6)  # common unit scale
        # the standardised pnl is the standardised mkt plus the rescaled convexity (side=+1):
        # p_std = m_std + gamma_sign*scale*m_std**2 → quadratic-in-m_std identity holds
        coef = np.polyfit(m_std, p_std, 2)
        assert coef[1] == pytest.approx(1.0, abs=1e-6)  # linear term == side == +1


def _sleeve_artefact(scale, baseline, gamma=-0.3, n=600, seed=0):
    """A concave sleeve (per-sleeve γ<0) with a scale-correlated PnL baseline. Pooling two such
    sleeves of different scale manufactures a POSITIVE quadratic term no sleeve has — the
    Jagannathan–Korajczyk / Simpson's-paradox artefact the standardise-then-repool step removes."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0, scale, size=n)
    pnl = baseline + mkt + gamma * (mkt**2)  # intercept absorbs `baseline` in a per-sleeve fit
    return mkt, pnl


def test_s512_collapse_statistic_vs_weighted_average():
    """Simpson's paradox: each sleeve is concave (γ<0), but the large-scale sleeve's elevated PnL
    baseline coincides with its larger mkt², so the RAW pooled γ turns POSITIVE. Common-scale
    standardisation equalises the mkt² ranges and the pooled γ sign collapses back to negative —
    the in-data analogue of the +1.18 → negative collapse (sign collapse, not magnitude)."""
    big = _sleeve_artefact(0.10, 0.06, seed=3)  # large scale + elevated baseline
    small = _sleeve_artefact(0.01, 0.0, seed=4)  # small scale, no baseline
    sleeves = [
        ("big", *big, float(treynor_mazuy(*big)[0]), len(big[0])),
        ("small", *small, float(treynor_mazuy(*small)[0]), len(small[0])),
    ]
    raw_pool_g = float(
        treynor_mazuy(np.concatenate([big[0], small[0]]), np.concatenate([big[1], small[1]]))[0]
    )
    g_std, _t, _p, g_wavg = s6.standardise_then_repool_tm(sleeves)
    assert raw_pool_g > 0.0  # heterogeneous scales manufacture a positive pooled γ (the artefact)
    assert g_wavg < 0.0  # the trade-count-weighted average of per-sleeve γ is negative
    assert g_std < 0.0  # standardise-then-repool collapses the sign back to negative
    assert np.sign(g_std) == np.sign(g_wavg)


def test_s512_writes_nothing_to_deliverables():
    """Diagnostic-only: calling the repool must not mutate any deliverable CSV in outputs/."""
    before = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in _OUTPUTS.glob("*.csv")}
    s = _sleeve(0.05, -1.0, seed=5)
    s6.standardise_then_repool_tm([("x", *s, 0.0, len(s[0])), ("y", *s, 0.0, len(s[0]))])
    after = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in _OUTPUTS.glob("*.csv")}
    assert before == after
