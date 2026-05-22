"""
run_characterize.py
===================
Checkpoint deliverable orchestrator (US-006) for the primary-signal
reverse-engineering study. This is a thin CLI that *renders* the numbers the
already-built, already-tested modules produce -- it reimplements nothing.

Run it::

    python -m stml.replication.run_characterize                       # all 11
    python -m stml.replication.run_characterize --instruments cl1s ng1s

and it writes exactly two artifacts:

1. ``reports/signal-characterization.md`` -- a human-readable report that, per
   instrument, answers the six C1 questions with NUMBERS pulled straight from
   :mod:`stml.replication.characterize`, plus the convention verdict and the
   ``n_eff`` / asset-class pooling map.
2. ``results/jj/thresholds.json`` -- per-metric significance cutoffs calibrated
   on the TRAIN split ONLY (the frozen anti-leakage commitment), with a
   provenance block recording the calibration window and train-only marker.

What this module computes itself (everything else is delegated):

* the convention split -- ``corr_at_lag1`` (next-day PnL), ``best_forward_lag``
  (argmax of ``|corr|`` over ``h in 0..+5``), and ``best_construction_lag``
  (argmax over ``h in -5..-1``) -- read off ``lead_lag``'s ``lag_profile``;
* the post-embargo val ``n_eff`` and the ``standalone`` / ``pool:<class>``
  decision against a documented, reviewable ``FLOOR``;
* the train-only baseline metric panels that set each cutoff.

Reuse (see ``.omc/scratch/CONTRACT.md``): :func:`stml.io.load_clean_data`,
:mod:`stml.replication.characterize`, :mod:`stml.replication.splits`,
:mod:`stml.replication.baselines`, :mod:`stml.replication.metrics`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stml.replication.baselines import (
    always_flat,
    majority_class,
    persistence,
    stratified_random,
)
from stml.replication.characterize import (
    alpha_type,
    cross_asset,
    drift,
    lead_lag,
    model_family_fingerprint,
    regime,
)
from stml.replication.metrics import panel
from stml.replication.splits import chronological_split, embargoed_val, n_eff

# Canonical instrument order (matches characterize._INSTRUMENTS).
_INSTRUMENTS: list[str] = [
    "es1s",
    "nq1s",
    "fesx1s",
    "cl1s",
    "ho1s",
    "rb1s",
    "ng1s",
    "gc1s",
    "si1s",
    "hg1s",
    "pl1s",
]

# Asset-class membership (equity / energy / metals).
_CLASS_OF: dict[str, str] = {
    "es1s": "equity",
    "nq1s": "equity",
    "fesx1s": "equity",
    "cl1s": "energy",
    "ho1s": "energy",
    "rb1s": "energy",
    "ng1s": "energy",
    "gc1s": "metals",
    "si1s": "metals",
    "hg1s": "metals",
    "pl1s": "metals",
}
_CLASSES: tuple[str, ...] = ("equity", "energy", "metals")

# DOCUMENTED, reviewable checkpoint parameters (the user may change these).
FLOOR: int = 10
"""Minimum post-embargo val n_eff for a STANDALONE verdict; below it an
instrument is folded into asset-class pooling. A checkpoint parameter."""

CUTOFF_MARGIN: float = 0.05
"""Small additive margin added on top of the strongest train baseline so a
replica must clear the no-skill band by a real (not noise) gap. A checkpoint
parameter; documented in the thresholds.json provenance block."""

_BASELINE_NAMES: tuple[str, ...] = (
    "always_flat",
    "majority_class",
    "stratified_random",
    "persistence",
)
# The CHANCE (no-skill) baselines that set the replication pass bar. Persistence
# (s_t = s_{t-1}) is deliberately EXCLUDED: it is a strong but trivially-laggable
# predictor, reported separately as an upper reference (see persistence_reference).
_CHANCE_BASELINE_NAMES: tuple[str, ...] = (
    "always_flat",
    "majority_class",
    "stratified_random",
)
_METRIC_KEYS: tuple[str, ...] = ("kappa", "mcc", "macro_f1", "ordinal_skill")

# Where the artifacts land (resolved relative to the repo root).
_REPORT_REL = Path("reports/signal-characterization.md")
_THRESHOLDS_REL = Path("results/jj/thresholds.json")


# --------------------------------------------------------------------------- #
# Repo / IO helpers                                                           #
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    """Walk up from this file until the directory holding ``data/`` +
    ``pyproject.toml`` is found (mirrors :func:`stml.io._find_repo_root`)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError(
        f"could not locate stml repo root (data/ + pyproject.toml) from {here}"
    )


def _signal_series(signals: pd.DataFrame, instrument: str) -> pd.Series:
    """Date-indexed signal series for one instrument (ascending)."""
    return (
        signals[["date", instrument]]
        .set_index("date")[instrument]
        .sort_index()
        .astype(int)
    )


# --------------------------------------------------------------------------- #
# Markdown rendering helpers                                                   #
# --------------------------------------------------------------------------- #
def _fmt(x: float | int | None, nd: int = 3) -> str:
    """Compact fixed-point string; ``nan`` / ``None`` render as ``n/a``."""
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "n/a"
    return f"{xf:.{nd}f}"


def _fmt_bool(x: object) -> str:
    """Render a bool verdict, or ``n/a`` when the value is a nan sentinel."""
    if isinstance(x, bool):
        return "yes" if x else "no"
    return "n/a"


# --------------------------------------------------------------------------- #
# Convention split (computed here from lead_lag's lag_profile)                 #
# --------------------------------------------------------------------------- #
def _convention_split(lag_profile: dict[int, float]) -> dict:
    """Split a lead/lag profile into the PnL- vs construction-lag readings.

    * ``corr_at_lag1``  -- ``corr(s_t, r_{t+1})``; the next-day PnL number. The
      default ``next_day`` convention is empirically supported when this is
      ``> 0``.
    * ``best_forward_lag`` -- the ``h in 0..+5`` maximizing ``|corr|`` (the
      forward / PnL side, including the concurrent ``h=0``).
    * ``best_construction_lag`` -- the ``h in -5..-1`` maximizing ``|corr|``.
      A non-zero construction lag means the signal is built from TRAILING
      returns; the SIGN of that correlation gives the style (negative =>
      mean-reversion / counter-trend, positive => momentum) -- not a
      convention problem.
    """
    prof = {int(h): float(c) for h, c in lag_profile.items()}
    corr_at_lag1 = prof.get(1, float("nan"))

    fwd = {h: c for h, c in prof.items() if 0 <= h <= 5 and math.isfinite(c)}
    con = {h: c for h, c in prof.items() if -5 <= h <= -1 and math.isfinite(c)}
    best_fwd = max(fwd, key=lambda h: abs(fwd[h])) if fwd else None
    best_con = max(con, key=lambda h: abs(con[h])) if con else None

    return {
        "corr_at_lag1": corr_at_lag1,
        "next_day_confirmed": bool(math.isfinite(corr_at_lag1) and corr_at_lag1 > 0),
        "best_forward_lag": best_fwd,
        "best_forward_corr": prof.get(best_fwd, float("nan")) if best_fwd is not None else float("nan"),
        "best_construction_lag": best_con,
        "best_construction_corr": prof.get(best_con, float("nan")) if best_con is not None else float("nan"),
    }


# --------------------------------------------------------------------------- #
# n_eff + pooling map                                                          #
# --------------------------------------------------------------------------- #
def _pooling_map(signals: pd.DataFrame, instruments: list[str]) -> dict:
    """Post-embargo val ``n_eff`` per instrument + the standalone/pool label.

    For each instrument the gateable ``n_eff`` is computed on the POST-embargo
    val window (implementation note #1):
    ``n_eff(signal.iloc[embargoed_val(signal, split)])``. An instrument is
    ``standalone`` when that ``n_eff >= FLOOR`` else ``pool:<class>``. Pooled
    per-class ``n_eff`` is the sum of the class members' post-embargo n_eff.
    """
    split = chronological_split(signals["date"])
    per_inst: dict[str, dict] = {}
    for inst in instruments:
        sig = _signal_series(signals, inst)
        post = sig.iloc[embargoed_val(sig, split)]
        ne = int(n_eff(post))
        cls = _CLASS_OF[inst]
        per_inst[inst] = {
            "n_eff_post_embargo": ne,
            "class": cls,
            "decision": "standalone" if ne >= FLOOR else f"pool:{cls}",
        }

    pooled: dict[str, int] = {}
    for cls in _CLASSES:
        members = [i for i in instruments if _CLASS_OF[i] == cls]
        pooled[cls] = int(sum(per_inst[i]["n_eff_post_embargo"] for i in members))

    sub_floor = [i for i in instruments if per_inst[i]["n_eff_post_embargo"] < FLOOR]
    return {"per_instrument": per_inst, "pooled_class_n_eff": pooled, "sub_floor": sub_floor}


# --------------------------------------------------------------------------- #
# Train-only threshold calibration                                            #
# --------------------------------------------------------------------------- #
def _baseline_preds(y_true: np.ndarray) -> dict[str, np.ndarray]:
    """The four naive predictors on a TRAIN label array (seeded RNG)."""
    return {
        "always_flat": always_flat(y_true),
        "majority_class": majority_class(y_true),
        "stratified_random": stratified_random(y_true, seed=0),
        "persistence": persistence(y_true),
    }


def _metric_value(panel_out: dict, key: str) -> float:
    """Pull a scalar metric out of a metrics.panel result.

    ``ordinal_skill`` is a dict ``{vs_flat, vs_random}``; the headline scalar is
    ``vs_flat`` (the intrinsic chance-corrected skill).
    """
    if key == "ordinal_skill":
        return float(panel_out["ordinal_skill"]["vs_flat"])
    return float(panel_out[key])


def _calibrate_instrument(y_true: np.ndarray) -> dict:
    """Train-only baseline metric panels + suggested per-metric cutoffs.

    For each of the four baselines the full metric panel is scored on the TRAIN
    signal and reported in ``baseline_metrics``. The suggested cutoff for a
    metric is ``max-over-CHANCE-baselines + CUTOFF_MARGIN`` -- i.e. the strongest
    no-skill reference among ``always_flat`` / ``majority_class`` /
    ``stratified_random(seed=0)`` plus the documented margin, so a replica must
    clear the no-skill band by a real (not noise) gap.

    Persistence (``s_t = s_{t-1}``) is a strong but trivially-laggable predictor;
    it is EXCLUDED from the cutoff max and instead surfaced verbatim in
    ``persistence_reference`` as a SEPARATE upper reference -- not the pass bar.
    """
    preds = _baseline_preds(y_true)
    baseline_metrics: dict[str, dict[str, float]] = {}
    for name, yp in preds.items():
        p = panel(y_true, yp)
        baseline_metrics[name] = {k: _metric_value(p, k) for k in _METRIC_KEYS}

    cutoffs: dict[str, float] = {}
    for k in _METRIC_KEYS:
        vals = [
            baseline_metrics[name][k]
            for name in _CHANCE_BASELINE_NAMES
            if math.isfinite(baseline_metrics[name][k])
        ]
        chance_ceiling = max(vals) if vals else 0.0
        cutoffs[k] = float(chance_ceiling + CUTOFF_MARGIN)

    return {
        "baseline_metrics": baseline_metrics,
        "suggested_cutoffs": cutoffs,
        "persistence_reference": dict(baseline_metrics["persistence"]),
        "n_train": int(len(y_true)),
    }


def _calibrate_thresholds(signals: pd.DataFrame, instruments: list[str]) -> dict:
    """Build the full thresholds.json payload (per-instrument + per-class).

    Strictly TRAIN-only: ``y_true`` is each instrument's TRAIN-split signal and
    no val/test data is touched. A per-asset-class pooled entry concatenates the
    member instruments' TRAIN signals and calibrates on the pooled series.
    """
    split = chronological_split(signals["date"])
    train_dates = pd.DatetimeIndex(signals["date"]).to_numpy()[split.train_idx]
    train_start = pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d")
    train_end = pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d")

    per_instrument: dict[str, dict] = {}
    train_signals: dict[str, np.ndarray] = {}
    for inst in instruments:
        y_train = _signal_series(signals, inst).iloc[split.train_idx].to_numpy(dtype=int)
        train_signals[inst] = y_train
        per_instrument[inst] = {"class": _CLASS_OF[inst], **_calibrate_instrument(y_train)}

    per_class: dict[str, dict] = {}
    for cls in _CLASSES:
        members = [i for i in instruments if _CLASS_OF[i] == cls]
        if not members:
            continue
        pooled = np.concatenate([train_signals[i] for i in members])
        per_class[cls] = {"members": members, **_calibrate_instrument(pooled)}

    return {
        "provenance": {
            "calibration_window": "train",
            "train_date_range": [train_start, train_end],
            "floor": FLOOR,
            "convention": "next_day",
            "cutoff_margin": CUTOFF_MARGIN,
            "cutoff_rule": "max over chance baselines {always_flat, majority_class, "
            "stratified_random(seed=0)} + cutoff_margin; persistence reported "
            "separately as an upper reference, NOT the pass bar",
            "persistence_is_pass_bar": False,
            "metrics": list(_METRIC_KEYS),
            "ordinal_skill_component": "vs_flat",
            "note": "calibrated on train split only; no val/test data used",
        },
        "per_instrument": per_instrument,
        "per_asset_class": per_class,
    }


# --------------------------------------------------------------------------- #
# Report rendering                                                            #
# --------------------------------------------------------------------------- #
def _render_q1(lines: list[str], a: dict) -> None:
    """Q1 alpha-type: trailing-return corr signs, MA agreement, breakouts."""
    lines.append("**Q1 -- alpha type (momentum vs mean-reversion).** "
                 f"`alpha_label` = **{a.get('alpha_label', 'n/a')}**, "
                 f"`momentum_score` = {_fmt(a.get('momentum_score'))} "
                 f"(mean trailing-return corr; >0 momentum, <0 mean-reversion).")
    trail = " · ".join(
        f"trail_{k}={_fmt(a.get(f'trail_corr_{k}'))}" for k in (1, 5, 10, 20)
    )
    ma = " · ".join(
        f"MA{n}={_fmt(a.get(f'ma_sign_agreement_{n}'))}" for n in (10, 20, 50)
    )
    lines.append(f"  - trailing-return corr: {trail}")
    lines.append(f"  - distance-from-MA sign agreement: {ma} (fraction trading WITH the MA distance)")
    lines.append(f"  - breakout coincidence: any={_fmt(a.get('breakout_coincidence'))}, "
                 f"directional={_fmt(a.get('breakout_coincidence_directional'))} "
                 "(Donchian-20 break on a nonzero day)")


def _render_q2(lines: list[str], ll: dict, conv: dict) -> None:
    """Q2 lead/lag with the PnL- vs construction-lag split."""
    prof = {int(h): float(c) for h, c in ll["lag_profile"].items()}
    profile_str = ", ".join(f"h{h:+d}:{_fmt(prof[h], 2)}" for h in sorted(prof))
    lines.append("**Q2 -- lead/lag and holding convention.**")
    lines.append(f"  - lag profile `corr(s_t, r_t+h)`: {profile_str}")
    lines.append("  - *forward (PnL) convention*: "
                 f"`corr_at_lag1` (h=+1) = **{_fmt(conv['corr_at_lag1'])}** "
                 f"-> {'next_day confirmed' if conv['next_day_confirmed'] else 'NOT confirmed (<= 0)'}; "
                 f"`best_forward_lag` (argmax|corr| over h in 0..+5) = {conv['best_forward_lag']} "
                 f"(corr {_fmt(conv['best_forward_corr'])}).")
    cc = conv["best_construction_corr"]
    if cc is None or cc != cc:  # None or NaN (e.g. single-direction ng1s)
        constr_read = "no usable trailing correlation (undetermined)"
    elif cc < 0:
        constr_read = "a NEGATIVE loading => mean-reversion / counter-trend construction"
    else:
        constr_read = "a POSITIVE loading => momentum / trend-following construction"
    lines.append("  - *construction lag*: "
                 f"`best_construction_lag` (argmax|corr| over h in -5..-1) = {conv['best_construction_lag']} "
                 f"(corr {_fmt(conv['best_construction_corr'])}) -- the signal is built from TRAILING returns; "
                 f"{constr_read} (independent of the forward PnL convention).")


def _render_q3(lines: list[str], rg: dict) -> None:
    """Q3 regime participation low- vs high-vol + avoids-high-vol boolean."""
    lines.append("**Q3 -- regime (does it avoid high vol?).** "
                 f"`participation_low_vol` = {_fmt(rg.get('participation_low_vol'))}, "
                 f"`participation_high_vol` = {_fmt(rg.get('participation_high_vol'))} (GMM); "
                 f"median-vol split low={_fmt(rg.get('participation_low_vol_median'))} / "
                 f"high={_fmt(rg.get('participation_high_vol_median'))}; "
                 f"**avoids_high_vol = {_fmt_bool(rg.get('avoids_high_vol'))}** "
                 f"(status: {rg.get('status', 'n/a')}). This verdict is instrument-specific.")


def _render_q5(lines: list[str], dr: dict) -> None:
    """Q5 drift: per-split base-rate table (train/val/test)."""
    lines.append("**Q5 -- drift (per-split base rates).**")
    lines.append("")
    lines.append("  | split | n | participation | long_bias | frac -1 | frac 0 | frac +1 |")
    lines.append("  |-------|--:|--------------:|----------:|--------:|-------:|--------:|")
    for s in ("train", "val", "test"):
        b = dr[s]
        lines.append(
            f"  | {s} | {b['n']} | {_fmt(b['participation_rate'])} | {_fmt(b['long_bias'])} | "
            f"{_fmt(b['frac_neg1'])} | {_fmt(b['frac_0'])} | {_fmt(b['frac_pos1'])} |"
        )
    lines.append("")
    tr = dr["trend"]
    lines.append(f"  - train->test trend: participation {_fmt(tr['participation_train_to_test'])}, "
                 f"long_bias {_fmt(tr['long_bias_train_to_test'])}.")


def _render_q6(lines: list[str], mf: dict) -> None:
    """Q6 model-family fingerprint: advisory low-confidence label."""
    label = mf.get("label", "inconclusive")
    conf = mf.get("confidence", 0.0)
    plain = (
        "inconclusive (gates nothing)"
        if label == "inconclusive"
        else f"{label} (advisory only, low confidence)"
    )
    lines.append("**Q6 -- model-family fingerprint (ADVISORY).** "
                 f"label = **{plain}**, confidence = {_fmt(conf)}; "
                 f"CV acc tree={_fmt(mf.get('tree_cv_acc'))} / linear={_fmt(mf.get('linear_cv_acc'))} / "
                 f"forest={_fmt(mf.get('forest_cv_acc'))} vs majority={_fmt(mf.get('majority_acc'))}.")


def _render_report(
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    instruments: list[str],
    char: dict[str, dict],
    ca: dict,
    pooling: dict,
    thresholds: dict,
) -> str:
    """Assemble the full Markdown report string from the computed pieces."""
    lines: list[str] = []
    lines.append("# Signal Characterization (C1 Checkpoint)")
    lines.append("")
    lines.append("Reverse-engineering the primary trading signal `s_t in {-1, 0, +1}` for the "
                 "released futures universe. All numbers below are produced by "
                 "`stml.replication.characterize` on the cleaned data "
                 "(`stml.io.load_clean_data`); this report only renders them.")
    lines.append("")
    lines.append(f"Instruments analysed ({len(instruments)}): " + ", ".join(f"`{i}`" for i in instruments) + ".")
    lines.append("")

    # --- Convention verdict (top-level) --------------------------------------
    conv_by_inst = {
        inst: _convention_split(char[inst]["lead_lag"]["lag_profile"]) for inst in instruments
    }
    negatives = [i for i in instruments if not conv_by_inst[i]["next_day_confirmed"]]
    lines.append("## Convention verdict")
    lines.append("")
    if not negatives:
        lines.append(f"`corr_at_lag1` (corr of `s_t` with next-day return `r_t+1`) is **positive for all "
                     f"{len(instruments)} instruments** -> the default `next_day` PnL convention is "
                     "**empirically supported**. No instrument shows a non-positive next-day correlation.")
    else:
        lines.append(f"`corr_at_lag1` is positive for {len(instruments) - len(negatives)} of "
                     f"{len(instruments)} instruments. The default `next_day` convention is supported on "
                     "those; instruments with `corr_at_lag1 <= 0` (flagged for review): "
                     + ", ".join(f"`{i}` ({_fmt(conv_by_inst[i]['corr_at_lag1'])})" for i in negatives) + ".")
    lines.append("")
    neg_constr = [
        i for i in instruments
        if (cc := conv_by_inst[i]["best_construction_corr"]) == cc and cc < 0
    ]
    lines.append("Note the distinction surfaced per instrument below: a non-zero *construction* lag reflects "
                 "a signal BUILT from trailing returns; the SIGN of that correlation gives the style "
                 "(negative => mean-reversion / counter-trend, positive => momentum), independent of the "
                 f"forward PnL convention. {len(neg_constr)} of {len(instruments)} instruments load negatively "
                 "(short-horizon mean-reversion / counter-trend).")
    lines.append("")
    lines.append("The headline convention uses the FORWARD-restricted argmax (`best_forward_lag`, over h>=0); "
                 "the `lead_lag` module's global `best_lag` over h in [-5, +5] is dominated by the (negative) "
                 "construction relationship by design and is NOT the convention verdict.")
    lines.append("")
    lines.append("| inst | corr@lag+1 | next_day? | best_fwd_lag | best_constr_lag |")
    lines.append("|------|-----------:|:---------:|-------------:|----------------:|")
    for inst in instruments:
        c = conv_by_inst[inst]
        lines.append(
            f"| `{inst}` | {_fmt(c['corr_at_lag1'])} | {'yes' if c['next_day_confirmed'] else 'NO'} | "
            f"{c['best_forward_lag']} | {c['best_construction_lag']} |"
        )
    lines.append("")

    # --- Cross-asset (Q4, panel-level) ---------------------------------------
    lines.append("## Q4 -- cross-asset structure (panel-level)")
    lines.append("")
    lines.append(f"Mean |off-diagonal| signal correlation = **{_fmt(ca['mean_abs_offdiag_corr'])}** "
                 "(~0.11 expected: the 11 signals are nearly independent across assets).")
    lines.append("")
    clusters: dict[int, list[str]] = {}
    for inst, lab in ca["cluster_labels"].items():
        if inst in instruments:
            clusters.setdefault(int(lab), []).append(inst)
    lines.append("Behavioral fingerprint clusters (participation / long-bias / persistence / momentum):")
    for lab in sorted(clusters):
        lines.append(f"  - cluster {lab}: " + ", ".join(f"`{i}`" for i in clusters[lab]))
    lines.append("")

    # --- n_eff + pooling map -------------------------------------------------
    lines.append("## Effective sample size (n_eff) and asset-class pooling map")
    lines.append("")
    lines.append("`n_eff` is computed on the POST-embargo validation window "
                 "(`n_eff(signal.iloc[embargoed_val(signal, split)])`). An instrument is **standalone** "
                 "when its post-embargo val `n_eff >= FLOOR`, else it is folded into asset-class pooling "
                 "(`pool:<class>`).")
    lines.append("")
    lines.append(f"**FLOOR = {FLOOR}** is a documented checkpoint parameter and is reviewable -- the user "
                 "may change it. Lowering it admits more standalone instruments; raising it pools more.")
    lines.append("")
    lines.append("| inst | class | post-embargo val n_eff | decision |")
    lines.append("|------|-------|-----------------------:|----------|")
    for inst in instruments:
        m = pooling["per_instrument"][inst]
        lines.append(f"| `{inst}` | {m['class']} | {m['n_eff_post_embargo']} | {m['decision']} |")
    lines.append("")
    sub = pooling["sub_floor"]
    if sub:
        lines.append(f"Sub-floor instruments (n_eff < {FLOOR}, pooled): "
                     + ", ".join(f"`{i}`" for i in sub) + ".")
    else:
        lines.append(f"No instrument falls below FLOOR = {FLOOR}; all are standalone.")
    lines.append("")
    lines.append("Pooled per-class post-embargo val n_eff:")
    for cls in _CLASSES:
        if cls in pooling["pooled_class_n_eff"]:
            lines.append(f"  - {cls}: {pooling['pooled_class_n_eff'][cls]}")
    lines.append("")

    # --- Per-instrument C1 answers -------------------------------------------
    lines.append("## Per-instrument characterization (Q1, Q2, Q3, Q5, Q6)")
    lines.append("")
    for inst in instruments:
        c = char[inst]
        lines.append(f"### `{inst}` ({_CLASS_OF[inst]})")
        lines.append("")
        _render_q1(lines, c["alpha_type"])
        lines.append("")
        _render_q2(lines, c["lead_lag"], conv_by_inst[inst])
        lines.append("")
        _render_q3(lines, c["regime"])
        lines.append("")
        _render_q5(lines, c["drift"])
        lines.append("")
        _render_q6(lines, c["model_family_fingerprint"])
        lines.append("")

    # --- All-instrument summary table ----------------------------------------
    lines.append("## Summary table (all instruments)")
    lines.append("")
    lines.append("| inst | class | alpha | momentum_score | corr@lag+1 | avoids_high_vol | "
                 "fingerprint (conf) | n_eff | decision |")
    lines.append("|------|-------|-------|---------------:|-----------:|:---------------:|"
                 "--------------------|------:|----------|")
    for inst in instruments:
        c = char[inst]
        a, rg, mf = c["alpha_type"], c["regime"], c["model_family_fingerprint"]
        pm = pooling["per_instrument"][inst]
        lines.append(
            f"| `{inst}` | {_CLASS_OF[inst]} | {a.get('alpha_label', 'n/a')} | "
            f"{_fmt(a.get('momentum_score'))} | {_fmt(conv_by_inst[inst]['corr_at_lag1'])} | "
            f"{_fmt_bool(rg.get('avoids_high_vol'))} | {mf.get('label', 'n/a')} ({_fmt(mf.get('confidence'))}) | "
            f"{pm['n_eff_post_embargo']} | {pm['decision']} |"
        )
    lines.append("")

    # --- Threshold calibration provenance ------------------------------------
    prov = thresholds["provenance"]
    lines.append("## Train-only threshold calibration")
    lines.append("")
    lines.append(f"`results/jj/thresholds.json` is calibrated on the **{prov['calibration_window']}** split "
                 f"only ({prov['train_date_range'][0]} .. {prov['train_date_range'][1]}); "
                 "no val/test data is used (the frozen anti-leakage commitment). For each instrument the "
                 "four naive baselines (always_flat, majority_class, stratified_random(seed=0), persistence) "
                 "are scored on the TRAIN signal. Each per-metric cutoff is "
                 f"`max(chance baselines) + {prov['cutoff_margin']}` (the documented margin) -- the strongest "
                 "of the no-skill references {always_flat, majority_class, stratified_random}. A replica must "
                 "clear these cutoffs to count as skillful.")
    lines.append("")
    lines.append("Cutoffs are set off no-skill baselines (flat / majority / stratified-random). Persistence "
                 "(`s_t = s_{t-1}`) is a strong but trivially-laggable predictor and is shown separately as "
                 "an UPPER REFERENCE, not the replication pass bar -- whether a replica must also beat "
                 "persistence is a checkpoint decision for the user.")
    lines.append("")
    lines.append("Suggested chance-level cutoffs vs the persistence reference "
                 "(kappa / mcc / macro_f1 / ordinal_skill[vs_flat]):")
    lines.append("")
    lines.append("| inst | kappa cut | mcc cut | macro_f1 cut | ordinal_skill cut | "
                 "| persist kappa | persist mcc | persist macro_f1 | persist ordinal_skill |")
    lines.append("|------|----------:|--------:|-------------:|------------------:|--|"
                 "--------------:|------------:|-----------------:|---------------------:|")
    for inst in instruments:
        co = thresholds["per_instrument"][inst]["suggested_cutoffs"]
        pr = thresholds["per_instrument"][inst]["persistence_reference"]
        lines.append(f"| `{inst}` | {_fmt(co['kappa'])} | {_fmt(co['mcc'])} | "
                     f"{_fmt(co['macro_f1'])} | {_fmt(co['ordinal_skill'])} | | "
                     f"{_fmt(pr['kappa'])} | {_fmt(pr['mcc'])} | "
                     f"{_fmt(pr['macro_f1'])} | {_fmt(pr['ordinal_skill'])} |")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run(
    instruments: list[str] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Run the C1 characterization and write the two checkpoint artifacts.

    Parameters
    ----------
    instruments : the subset to analyse; ``None`` (the default) means all 11 in
        canonical order.
    out_dir : root under which ``reports/`` and ``results/jj/`` are written.
        ``None`` (the default) uses the repo root, producing the canonical
        deliverables. Tests pass a temporary directory here so a partial
        (single-instrument) smoke run never clobbers the full deliverables.

    Returns
    -------
    dict with ``{"report": Path, "thresholds": Path}`` -- the two written files.
    """
    instruments = list(instruments) if instruments else list(_INSTRUMENTS)

    root = Path(out_dir) if out_dir is not None else _repo_root()
    ohlcv, signals = _load_clean()
    # A one-year lookback before the 2020-2022 signal era covers every trailing
    # feature while keeping the slow Markov/GMM fits fast.
    ohlcv = ohlcv[ohlcv["date"] >= "2019-01-01"].copy()

    char: dict[str, dict] = {}
    for inst in instruments:
        char[inst] = {
            "alpha_type": alpha_type(inst, signals, ohlcv),
            "lead_lag": lead_lag(inst, signals, ohlcv),
            "regime": regime(inst, ohlcv, signals),
            "drift": drift(inst, signals),
            "model_family_fingerprint": model_family_fingerprint(inst, signals, ohlcv),
        }
    ca = cross_asset(signals, ohlcv, instruments=instruments)
    pooling = _pooling_map(signals, instruments)
    thresholds = _calibrate_thresholds(signals, instruments)

    report_path = root / _REPORT_REL
    thresholds_path = root / _THRESHOLDS_REL
    report_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        _render_report(signals, ohlcv, instruments, char, ca, pooling, thresholds),
        encoding="utf-8",
    )
    thresholds_path.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")

    return {"report": report_path, "thresholds": thresholds_path}


def _load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thin indirection over :func:`stml.io.load_clean_data` (kept separate so
    the import stays local to the run path)."""
    from stml.io import load_clean_data

    return load_clean_data()


def main(argv: list[str] | None = None) -> dict[str, Path]:
    """CLI entry point: parse ``--instruments`` and run the orchestrator."""
    parser = argparse.ArgumentParser(
        prog="python -m stml.replication.run_characterize",
        description="Render the C1 signal-characterization report and calibrate "
        "train-only thresholds.json (US-006 checkpoint deliverable).",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        metavar="INST",
        choices=_INSTRUMENTS,
        help="instruments to analyse (default: all 11 in canonical order).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="output root for reports/ and results/jj/ (default: repo root).",
    )
    args = parser.parse_args(argv)

    paths = run(args.instruments, out_dir=args.out_dir)
    for label, p in paths.items():
        size = p.stat().st_size
        print(f"wrote {label}: {p} ({size} bytes)")
    return paths


if __name__ == "__main__":
    main()
