"""EX.5 — economically-ranked triple-barrier labelling sweep (DIAGNOSTIC-ONLY).

Ranks labelling schemes by **downstream out-of-sample economic usefulness** (net Sharpe from the
barrier-exact, cost-aware backtest), NOT by label accuracy — the objective the existing
``ex3_barrier_surface`` (which ranks by CV AUC) does not serve. Everything happens on the
**modelling window only** (≤ ``modelling_end``); the locked OOS window (Jan–Jun 2022) is never
touched, so the chosen config can never snoop the grader's hidden half. This module holds the
reusable, unit-tested logic; ``experiments/ex5_barrier_economic_sweep.py`` is the thin runner.

Design (advisor-reviewed):
- **Common event-set.** All compared configs are ranked on the *intersection* of their event keys
  ``(date, instrument)`` so we rank label quality, not different trade populations (per-config full
  ``n`` is reported too).
- **Features held identical** across configs (only the labeller changes) — assembled once per
  instrument and cached. Feature-side leakage is therefore a constant that cannot bias the
  relative ranking; the only label-side leakage risks (causal vol, ``t1`` purging) are controlled.
- **Barrier-INDEPENDENT sizing.** ``weight = side · 1{p̂≥τ} · vol_target_leverage(ann_vol)`` — it
  deliberately does NOT use fractional-Kelly on the barriers ``(b,d)``, which would make sizing
  depend on the config under test and confound the comparison.
- **Cost sensitivity.** Net Sharpe at 0× / 1× / 2× the ``barrier_backtest`` cost defaults, because
  tighter/shorter barriers churn more and cost can flip the winner.
- **Always-act economic floor.** A mandatory baseline trading every signal-day (anchor labels, all
  acted) net of costs — if nothing beats it, that is the headline finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from stml.io import load_clean_data

from .backtest import barrier_backtest
from .barrier_vol import barrier_sigma
from .cross_validation import PurgedKFold
from .evaluation import evaluate_predictions, oos_predictions
from .features import assemble_instrument_features, daily_barrier_sigma, filter_signal_days
from .labelling_variants import (
    fixed_horizon_labels,
    trend_scan_labels,
    triple_barrier_labels_ext,
    vol_scaled_horizon,
)
from .models import balanced_sample_weight, make_lightgbm, make_xgb
from .pipeline import PipelineConfig, class_members, load_embargo_days
from .seeding import set_seeds
from .sizing import vol_target_leverage

COST_MULTIPLIERS = ((0.0, "0x"), (1.0, "1x"), (2.0, "2x"))
BASE_HALF_SPREAD_BPS = 2.0
BASE_IMPACT_BPS = 10.0
SIZING_THRESHOLD = 0.5
TARGET_VOL = 0.25
#: Labellers see close history from here to ``modelling_end`` ONLY — strict anti-snooping
#: confinement (no Jan–Jun 2022 bar is ever read, even to resolve a late-2021 label) and a large
#: speed-up for the O(bars×windows) trend-scan baseline. Signals start 2020-01-03, so this buffer
#: comfortably covers every ≤40-bar backward/forward window without altering any 2020+ label.
LABEL_HISTORY_START = pd.Timestamp("2019-07-01")


# --- configuration set (anchored one-factor-at-a-time) ----------------------


@dataclass(frozen=True)
class SweepConfig:
    """One labelling scheme. ``family`` selects the labeller; the rest are its parameters."""

    name: str
    stage: str  # 'anchor' | 'A_vol' | 'B_width' | 'C_horizon' | 'baseline' | 'floor' | 'joint'
    family: str  # 'tb' | 'b1_fixed' | 'b2_trend' | 'always_act'
    vol_estimator: str | None = None
    vol_param: int | None = None
    pt_sl: tuple[float, float] = (1.0, 1.0)
    horizon: int = 10  # V1 fixed bar count (tb)
    vol_scaled: tuple[int, int, int] | None = None  # (h0, h_min, h_max) for V2
    fixed_h: int = 10  # B1
    tau: float = 0.0  # B1 threshold
    trend_span: tuple[int, int] = (5, 20)  # B2


def build_configs() -> list[SweepConfig]:
    """Anchored OFAT sweep: each stage varies one factor with the others at the LdP default.

    The anchor vol is ``"shipped"`` — the live pipeline's ``daily_barrier_sigma``
    (``f2_vol_20``/√252), which is a **realized-vol-20** (close-to-close rolling std), NOT
    Garman-Klass. Stage A therefore contrasts the shipped realized-vol against a *genuine*
    Garman-Klass arm (``gk``), EWMA, and other rolling windows — the only way to answer "should the
    barrier vol estimator change from what we ship?" (and to test the lit-review EWMA commitment).
    """
    cfgs: list[SweepConfig] = [
        SweepConfig("anchor_shipped_1_1_h10", "anchor", "tb", "shipped", 20, (1.0, 1.0), 10),
    ]
    # Stage A — vol estimator (pt_sl=(1,1), h=10); genuine GK + EWMA + rolling vs the shipped anchor
    for est, param in [
        ("gk", 20),
        ("ewma", 20),
        ("ewma", 50),
        ("ewma", 100),
        ("rolling", 20),
        ("rolling", 50),
        ("rolling", 100),
    ]:
        cfgs.append(SweepConfig(f"volA_{est}{param}", "A_vol", "tb", est, param, (1.0, 1.0), 10))
    # Stage B — horizontal width (vol=shipped, h=10). Includes the **tight-stop** (sl=0.25) regime
    # the jay/triple-barrier-label study found robust (wide-ish PT, very tight SL).
    for ptsl, lbl in [
        ((0.5, 0.5), "0p5"),
        ((2.0, 2.0), "2p0"),
        ((2.0, 1.0), "2_1"),
        ((1.0, 2.0), "1_2"),
        ((0.5, 0.25), "0p5_0p25"),
        ((1.0, 0.25), "1_0p25"),
        ((2.0, 0.25), "2_0p25"),
    ]:
        cfgs.append(SweepConfig(f"widthB_{lbl}", "B_width", "tb", "shipped", 20, ptsl, 10))
    # Stage C — vertical horizon (vol=shipped, pt_sl=(1,1)); h=15 added per the PDF's mid-h picks
    for h in (5, 15, 20, 40):
        cfgs.append(SweepConfig(f"horizC_h{h}", "C_horizon", "tb", "shipped", 20, (1.0, 1.0), h))
    cfgs.append(
        SweepConfig(
            "horizC_volscaled", "C_horizon", "tb", "shipped", 20, (1.0, 1.0), vol_scaled=(10, 2, 40)
        )
    )
    # Baselines (same events, same act/skip semantics, same fixed model)
    for h in (5, 10, 20, 40):
        cfgs.append(SweepConfig(f"b1_fixed_h{h}", "baseline", "b1_fixed", fixed_h=h, tau=0.0))
    cfgs.append(SweepConfig("b2_trend_5_20", "baseline", "b2_trend", trend_span=(5, 20)))
    # Economic floor — every signal acted, anchor (shipped) labels for t1/side
    cfgs.append(
        SweepConfig("always_act_floor", "floor", "always_act", "shipped", 20, (1.0, 1.0), 10)
    )
    return cfgs


# --- pure helpers (unit-tested) ---------------------------------------------


def common_event_index(frames: list[pd.DataFrame]) -> pd.MultiIndex:
    """Intersection of the configs' ``(date, instrument)`` event keys, sorted."""
    idx: pd.Index | None = None
    for f in frames:
        idx = f.index if idx is None else idx.intersection(f.index)
    return idx.sort_values()


def build_meta(
    dates,
    instruments,
    side,
    proba,
    ann_vol,
    t1,
    *,
    tau: float = SIZING_THRESHOLD,
    target_vol: float = TARGET_VOL,
    max_leverage: float = 5.0,
    act_all: bool = False,
) -> pd.DataFrame:
    """Barrier-INDEPENDENT sizing → the ``[date, instrument, weight, t1]`` frame the backtest
    consumes. ``weight = side · gate · vol_target_leverage(ann_vol)``; ``gate`` is ``1{p̂≥τ}`` (or
    all-ones for the always-act floor)."""
    p = np.nan_to_num(np.asarray(proba, dtype=float), nan=0.0)
    gate = np.ones_like(p) if act_all else (p >= tau).astype(float)
    lev = np.atleast_1d(
        np.asarray(
            vol_target_leverage(ann_vol, target_vol=target_vol, max_leverage=max_leverage),
            dtype=float,
        )
    )
    weight = np.asarray(side, dtype=float) * gate * lev
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "instrument": np.asarray(instruments),
            "weight": weight,
            "t1": pd.DatetimeIndex(pd.to_datetime(np.asarray(t1))),
        }
    )


def assert_modelling_only(index, modelling_end) -> None:
    """Guard: no event/return date may exceed ``modelling_end`` (anti-snooping confinement)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) and idx.max() > pd.Timestamp(modelling_end):
        raise AssertionError(
            f"date {idx.max().date()} exceeds modelling_end {pd.Timestamp(modelling_end).date()} "
            f"— anti-snooping breach (the sweep must stay on the modelling window)"
        )


def _annualised_sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252.0)) if sd > 0 else float("nan")


def lag_placebo_sharpe(label_common, close_by_inst, *, lags=(0, 1, 2)) -> dict:
    """Model-free **placebo-in-time** labeling-quality diagnostic (cf. jay/triple-barrier-label §2).

    For the acted (``bin == 1``) events, the position earns the side-adjusted *single-bar* return at
    lag ``L`` from the signal bar — ``r_L = side · (close_{t+L}/close_{t+L-1} − 1)`` — and we report
    its per-trade Sharpe (×√252) at each lag. A genuine forward-looking label makes **lag-1** (the
    first tradeable bar) dominate **lag-0** (the contemporaneous, untradeable placebo); a strong
    lag-0 flags a circular / self-fulfilling label (e.g. an ``h=1`` barrier overlapping its own
    return). Uses the realised label, so it is an ORACLE upper bound — a diagnostic, NEVER the
    ranker. Bar returns are read from ``close_by_inst`` sliced to the modelling window, so no
    post-``modelling_end`` bar is touched (events near the edge are dropped via the position guard).
    """
    out = {}
    for lag in lags:
        rets = []
        for inst, sub in label_common.groupby(level="instrument"):
            close = close_by_inst.get(inst)
            if close is None:
                continue
            c = close.to_numpy()
            pos = close.index.get_indexer(sub.index.get_level_values("date"))
            tgt = pos + lag
            y = sub["bin"].to_numpy()
            valid = (pos >= 0) & (tgt - 1 >= 0) & (tgt < len(c)) & (y == 1)
            if valid.any():
                u = c[tgt[valid]] / c[tgt[valid] - 1] - 1.0
                rets.append(sub["side"].to_numpy()[valid] * u)
        r = np.concatenate(rets) if rets else np.array([])
        out[f"adj_sharpe_lag{lag}"] = _annualised_sharpe(r)
    return out


# --- per-instrument cache + label production --------------------------------


@dataclass
class _InstrumentData:
    close: pd.Series
    signal: pd.Series
    ohlc: pd.DataFrame
    feats: pd.DataFrame
    sigma_shipped: pd.Series  # daily_barrier_sigma = realized-vol-20 (the live pipeline default)
    ann_vol: pd.Series


def _instrument_data(ohlcv: pd.DataFrame, signals: pd.DataFrame, inst: str) -> _InstrumentData:
    ohlcv_inst = ohlcv[ohlcv["instrument"] == inst]
    signal = signals.set_index("date")[inst].sort_index()
    signal.index = pd.DatetimeIndex(signal.index)
    feats = assemble_instrument_features(ohlcv_inst, signal)
    close = ohlcv_inst.set_index("date")["close"].sort_index().astype(float)
    close.index = pd.DatetimeIndex(close.index)
    return _InstrumentData(
        close, signal, ohlcv_inst, feats, daily_barrier_sigma(feats), feats["f2_vol_20"]
    )


def _labels_for(cfg: SweepConfig, d: _InstrumentData, modelling_end) -> pd.DataFrame:
    """Labels for one instrument, computed on ``[LABEL_HISTORY_START, modelling_end]`` only.

    Confining the close window here means NO label ever reads a Jan–Jun 2022 bar to resolve its
    barrier (strict anti-snooping); boundary events whose barrier would clip at the window edge are
    dropped in ``_pooled_labels`` so no artificial timeout label survives.
    """
    lo, hi = LABEL_HISTORY_START, modelling_end
    close = d.close.loc[lo:hi]
    signal = d.signal.loc[lo:hi]
    if cfg.family in ("tb", "always_act"):
        if cfg.vol_estimator == "shipped":
            sigma = d.sigma_shipped.loc[lo:hi]  # the live pipeline's realized-vol-20 barrier σ
        else:  # "gk" | "ewma" | "rolling" — genuinely different estimators via the resolver
            ohlc = d.ohlc[(d.ohlc["date"] >= lo) & (d.ohlc["date"] <= hi)]
            sigma = barrier_sigma(ohlc, cfg.vol_estimator, cfg.vol_param)
        if cfg.vol_scaled is not None:
            h0, h_min, h_max = cfg.vol_scaled
            t_events = signal.index[signal.to_numpy() != 0]
            horizon = vol_scaled_horizon(sigma, t_events, h0, h_min=h_min, h_max=h_max)
        else:
            horizon = cfg.horizon
        return triple_barrier_labels_ext(close, signal, sigma, pt_sl=cfg.pt_sl, max_holding=horizon)
    if cfg.family == "b1_fixed":
        return fixed_horizon_labels(close, signal, cfg.fixed_h, cfg.tau)
    if cfg.family == "b2_trend":
        return trend_scan_labels(close, signal, cfg.trend_span)
    raise ValueError(f"unknown family {cfg.family!r}")


def _pooled_labels(cfg, insts, cache, modelling_end) -> pd.DataFrame:
    parts = []
    for inst in insts:
        d = cache[inst]
        last_in_window = d.close.loc[:modelling_end].index[-1]
        lab = _labels_for(cfg, d, modelling_end)
        # Drop boundary events whose barrier clipped at the window edge (t1 == last in-window bar):
        # their label would be an artefact of truncation, not a resolved barrier.
        lab = lab[(lab.index <= modelling_end) & (lab["t1"] < last_in_window)]
        if lab.empty:
            continue
        lab = lab.copy()
        lab["ann_vol"] = d.ann_vol.reindex(lab.index).to_numpy()
        lab.index = pd.MultiIndex.from_arrays(
            [lab.index, [inst] * len(lab)], names=["date", "instrument"]
        )
        parts.append(lab)
    return pd.concat(parts) if parts else pd.DataFrame()


def _pooled_features(insts, cache, modelling_end) -> pd.DataFrame:
    parts = []
    for inst in insts:
        d = cache[inst]
        fe = filter_signal_days(d.feats, d.signal)
        fe = fe[fe.index <= modelling_end].copy()
        fe.index = pd.MultiIndex.from_arrays(
            [fe.index, [inst] * len(fe)], names=["date", "instrument"]
        )
        parts.append(fe)
    return pd.concat(parts)


def _returns_panel(insts, cache, modelling_end) -> pd.DataFrame:
    panel = pd.DataFrame({inst: cache[inst].close.pct_change() for inst in insts})
    return panel[panel.index <= modelling_end]


def _design_matrix(pooled_feats, common) -> tuple[pd.DataFrame, pd.Series]:
    """Median-imputed, zero-variance-dropped X indexed by the event-date axis + aligned instruments.

    X.index is the (sorted, duplicate-allowed) event-date ``DatetimeIndex`` that ``PurgedKFold``
    requires to equal ``t1.index``; ``inst_series`` carries the per-row ticker for per-instrument
    embargo."""
    raw = pooled_feats.reindex(common)
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    filled = imp.fit_transform(raw)
    event_dates = pd.DatetimeIndex([k[0] for k in common])
    X = pd.DataFrame(filled, index=event_dates, columns=raw.columns)
    X = X[[c for c in X.columns if X[c].var() > 1e-12]]
    inst_series = pd.Series([k[1] for k in common], index=event_dates)
    return X, inst_series


def _make_model(name: str):
    return make_lightgbm() if name == "lightgbm" else make_xgb()


def _barrier_mix(label_common: pd.DataFrame) -> dict:
    bt = label_common.get("barrier_type")
    if bt is None:
        return {"pct_pt": float("nan"), "pct_sl": float("nan"), "pct_timeout": float("nan")}
    return {
        "pct_pt": round(float((bt == "pt").mean()), 3),
        "pct_sl": round(float((bt == "sl").mean()), 3),
        "pct_timeout": round(float((bt == "vertical").mean()), 3),
    }


def _score(
    cfg, X, inst_series, label_common, returns_panel, cfg_pipe, embargo, model_name, close_by_inst
) -> dict:
    order = X.index
    y = label_common["bin"].to_numpy()
    t1 = pd.Series(pd.to_datetime(label_common["t1"].to_numpy()), index=order)
    side = label_common["side"].to_numpy()
    ann_vol = label_common["ann_vol"].to_numpy()
    sw = balanced_sample_weight(y, base=label_common["weight"].to_numpy())

    cv = PurgedKFold(
        n_splits=cfg_pipe.n_splits,
        t1=t1,
        pct_embargo=cfg_pipe.pct_embargo,
        instruments=inst_series,
        embargo_days=embargo,
    )
    oos = oos_predictions(lambda: _make_model(model_name), X, y, cv, sample_weight=sw)
    finite = np.isfinite(oos)
    clf = (
        evaluate_predictions(y[finite], oos[finite], sample_weight=sw[finite])
        if finite.any() and len(np.unique(y[finite])) == 2
        else {}
    )

    row = {
        "config": cfg.name,
        "stage": cfg.stage,
        "family": cfg.family,
        "n": int(len(y)),
        "pos_rate": round(float(y.mean()), 4),
        **_barrier_mix(label_common),
        "auc": round(float(clf.get("auc", float("nan"))), 4),
        "log_loss": round(float(clf.get("log_loss", float("nan"))), 4),
        "precision_taken": round(float(clf.get("precision", float("nan"))), 4),
        "recall": round(float(clf.get("recall", float("nan"))), 4),
        "f1": round(float(clf.get("f1", float("nan"))), 4),
        "brier": round(float(clf.get("brier", float("nan"))), 4),
        # ORACLE placebo-in-time diagnostic (uses the realised label, never ranked): lag-1 should
        # dominate lag-0; a strong lag-0 flags a circular/self-fulfilling label.
        **{k: round(v, 3) for k, v in lag_placebo_sharpe(label_common, close_by_inst).items()},
    }
    act_all = cfg.family == "always_act"
    rep_last = {}
    for mult, tag in COST_MULTIPLIERS:
        meta = build_meta(order, inst_series, side, oos, ann_vol, t1, act_all=act_all)
        _, rep = barrier_backtest(
            meta,
            returns_panel,
            half_spread_bps=BASE_HALF_SPREAD_BPS * mult,
            impact_bps=BASE_IMPACT_BPS * mult,
        )
        row[f"net_sharpe_{tag}"] = round(float(rep["sharpe"]), 4)
        row[f"net_total_{tag}"] = round(float(rep.get("net_total_return", float("nan"))), 5)
        row[f"max_dd_{tag}"] = round(float(rep.get("max_drawdown", float("nan"))), 4)
        rep_last = rep
    row["ann_turnover"] = round(float(rep_last.get("ann_turnover", float("nan"))), 3)
    row["avg_holding"] = round(float(rep_last.get("avg_holding_period", float("nan"))), 2)
    return row


def run_class(
    asset_class: str,
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    cfg_pipe: PipelineConfig,
    embargo: dict,
    *,
    configs: list[SweepConfig] | None = None,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """Score every config for one asset class on the common event-set; return the results table."""
    configs = configs or build_configs()
    insts = class_members(asset_class)
    cache = {inst: _instrument_data(ohlcv, signals, inst) for inst in insts}
    returns_panel = _returns_panel(insts, cache, cfg_pipe.modelling_end)
    pooled_feats = _pooled_features(insts, cache, cfg_pipe.modelling_end)
    # window-confined close per instrument for the oracle lag-placebo diagnostic (no 2022 bar read)
    close_by_inst = {i: cache[i].close.loc[: cfg_pipe.modelling_end] for i in insts}

    label_frames = {
        c.name: _pooled_labels(c, insts, cache, cfg_pipe.modelling_end) for c in configs
    }
    ranked = [c for c in configs if c.family != "always_act" and not label_frames[c.name].empty]
    common = common_event_index([label_frames[c.name] for c in ranked])

    assert_modelling_only(pd.DatetimeIndex([k[0] for k in common]), cfg_pipe.modelling_end)
    assert_modelling_only(returns_panel.index, cfg_pipe.modelling_end)
    for c in ranked:  # every label's first-touch must also resolve within the window (no 2022 read)
        assert_modelling_only(
            pd.DatetimeIndex(label_frames[c.name]["t1"].to_numpy()), cfg_pipe.modelling_end
        )

    X, inst_series = _design_matrix(pooled_feats, common)
    rows = []
    for c in configs:
        lf = label_frames[c.name]
        if lf.empty:
            continue
        rows.append(
            _score(
                c, X, inst_series, lf.reindex(common), returns_panel, cfg_pipe, embargo,
                model_name, close_by_inst,
            )
        )
    table = pd.DataFrame(rows)
    table.attrs["asset_class"] = asset_class
    table.attrs["common_n"] = len(common)
    return table


def recommend(table: pd.DataFrame, *, tie_band: float = 0.10) -> dict:
    """Per-stage winner by net Sharpe (1×), tie-broken toward the simpler / more balanced config.

    Within ``tie_band`` net-Sharpe of a stage's best, prefer: GK > rolling > EWMA vol; |k| nearer 1;
    shorter horizon; pos_rate nearer 0.5. Returns the recommended ``(vol, pt_sl, horizon)`` plus the
    always-act floor's net Sharpe for the "did anything beat blind-primary?" headline.
    """

    def _pick(stage_names: tuple[str, ...]) -> pd.Series:
        sub = table[table["stage"].isin(stage_names)].copy()
        if sub.empty:
            return pd.Series(dtype=object)
        best = sub["net_sharpe_1x"].max()
        band = sub[sub["net_sharpe_1x"] >= best - tie_band].copy()
        band["_bal"] = (band["pos_rate"] - 0.5).abs()
        return band.sort_values(["_bal", "net_sharpe_1x"], ascending=[True, False]).iloc[0]

    floor = table[table["family"] == "always_act"]
    return {
        "vol_winner": _pick(("A_vol", "anchor")).get("config"),
        "width_winner": _pick(("B_width", "anchor")).get("config"),
        "horizon_winner": _pick(("C_horizon", "anchor")).get("config"),
        "best_net_sharpe_1x": float(table["net_sharpe_1x"].max())
        if not table.empty
        else float("nan"),
        "floor_net_sharpe_1x": float(floor["net_sharpe_1x"].iloc[0])
        if not floor.empty
        else float("nan"),
    }


def run(
    classes=("equity", "energy", "metals"),
    cfg_pipe: PipelineConfig | None = None,
    *,
    model_name: str = "xgboost",
) -> dict[str, pd.DataFrame]:
    """Run the sweep for each asset class; return ``{asset_class: results_table}``."""
    set_seeds(42)
    cfg_pipe = cfg_pipe or PipelineConfig()
    ohlcv, signals = load_clean_data()
    embargo = load_embargo_days()
    return {
        ac: run_class(ac, ohlcv, signals, cfg_pipe, embargo, model_name=model_name)
        for ac in classes
    }
