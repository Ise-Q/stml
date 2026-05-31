"""S6.8 — deployment deflation gate (DSR range + MinBTL + CSCV-PBO) on the calibrated default path.

The §6 Sharpe is selected from the horse-race, so it overstates skill. This gate asks whether it
survives that selection bias — it makes NO "strategy works" claim. For each class we re-fit every
roster candidate, barrier-backtest it to a net daily series (the same basis as §6), and feed the
selected (calibrated) series + the candidate dispersion to the tested ``deflation`` functions:

- DSR over a RANGE N ∈ [N_eff → N_raw] (N_eff via ONC on the trial-return matrix) — a single
  backtest cannot pin N, so we report the sensitivity;
- MinBTL vs the actual ~half-year OOS length;
- CSCV-PBO on the (days × candidates) trial-return matrix.

The "doesn't clear" outcome is the expected, honest finding, not a number to tune toward.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CLASSES, results_dir  # noqa: E402
from stml.io import load_clean_data, load_returns_panel  # noqa: E402

from alken_metamodel.backtest import barrier_backtest  # noqa: E402
from alken_metamodel.deflation import (  # noqa: E402
    deflated_sharpe_ratio,
    effective_n_trials,
    min_backtest_length,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from alken_metamodel.emit import strategy_weights  # noqa: E402
from alken_metamodel.models import balanced_sample_weight  # noqa: E402
from alken_metamodel.pipeline import (  # noqa: E402
    PipelineConfig,
    _roster_factory,
    build_class_panel,
    class_members,
    feature_columns,
    fit_oos_calibrator,
    select_model,
)
from alken_metamodel.seeding import set_seeds  # noqa: E402

TARGET_ANN_SHARPE = 1.0  # the annualised Sharpe a deployable strategy would need to justify
ANN = 252


def class_trials(cls: str, cfg: PipelineConfig, ohlcv, signals):
    """(best, selected-calibrated meta, {candidate: raw meta}) on the OOS window for a class."""
    set_seeds(cfg.seed)
    pooled = build_class_panel(ohlcv, signals, class_members(cls), cfg)
    cols = feature_columns(pooled)
    X, y, t1 = pooled[cols], pooled["bin"].to_numpy(), pooled["t1"]
    dates = pd.DatetimeIndex(pooled["date"])
    inst = pooled["instrument"].to_numpy()
    inst_s = pd.Series(inst, index=X.index)  # S2.6 per-instrument embargo
    side = pooled["side"].to_numpy()
    vol = pooled["f2_vol_20"].to_numpy()
    mmask = np.asarray(dates <= cfg.modelling_end)
    pmask = np.asarray((dates >= cfg.predict_start) & (dates <= cfg.predict_end))
    sw = balanced_sample_weight(y[mmask], base=pooled["weight"].to_numpy()[mmask])
    best, _ = select_model(X[mmask], y[mmask], t1[mmask], sw, cfg, instruments=inst_s[mmask])
    factory = _roster_factory(cfg)
    t1_pred = pd.DatetimeIndex(t1[pmask].to_numpy())

    def meta_for(proba):
        preds = pd.DataFrame(
            {"date": dates[pmask], "instrument": inst[pmask], "prediction": proba,
             "side": side[pmask], "ann_vol": vol[pmask]}
        )
        m = strategy_weights(preds, cfg)
        m["t1"] = t1_pred
        return m

    trial_meta = {}
    for name in factory(seed=cfg.seed):
        model = factory(seed=cfg.seed)[name]
        model.fit(X[mmask], y[mmask], sample_weight=sw)
        trial_meta[name] = meta_for(model.predict_act_proba(X[pmask]))

    cal, _, _ = fit_oos_calibrator(
        lambda: factory(seed=cfg.seed)[best], X[mmask], y[mmask], t1[mmask], sw, cfg,
        instruments=inst_s[mmask],
    )
    sel = factory(seed=cfg.seed)[best]
    sel.fit(X[mmask], y[mmask], sample_weight=sw)
    sel_meta = meta_for(np.clip(cal.transform(sel.predict_act_proba(X[pmask])), 0.0, 1.0))
    return best, sel_meta, trial_meta


def _trial_matrix(trial_nets: dict[str, pd.Series]) -> np.ndarray:
    df = pd.DataFrame(trial_nets).dropna(how="all").fillna(0.0)
    return df.to_numpy()


def gate_block(
    label: str, selected: str, sel_net: pd.Series, trial_nets: dict[str, pd.Series]
) -> str:
    names = list(trial_nets)
    n_raw = len(names)
    sharpes = [sharpe_ratio(trial_nets[n].dropna().to_numpy()) for n in names]
    sharpes = [s for s in sharpes if np.isfinite(s)]
    tstd = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else float("nan")
    matrix = _trial_matrix(trial_nets)
    n_eff = effective_n_trials(matrix)
    net = sel_net.dropna().to_numpy()
    sr_d = sharpe_ratio(net)
    sr_ann = sr_d * np.sqrt(ANN) if np.isfinite(sr_d) else float("nan")
    # DSR ladder: N_eff (ONC, optimistic) → N_raw (roster horse-race) → 2·, 4·N_raw. The roster
    # count UNDER-states the search: the data-driven cluster-rep reducer + the S1.8-b F16 expansion
    # add an implicit feature-selection search not counted in N_raw, so the higher rungs are the
    # honest upper bound and DSR (monotone-decreasing in N) only falls — the gate fails a fortiori.
    dsr = {k: deflated_sharpe_ratio(net, n_trials=k, trials_sharpe_std=tstd)
           for k in (n_eff, n_raw, 2 * n_raw, 4 * n_raw)}
    pbo = probability_of_backtest_overfitting(matrix, n_blocks=10)["pbo"]
    minbtl_raw = min_backtest_length(n_raw, target_sharpe=TARGET_ANN_SHARPE)
    minbtl_eff = min_backtest_length(n_eff, target_sharpe=TARGET_ANN_SHARPE)
    oos_years = len(net) / ANN
    clears = "CLEARS" if max(dsr.values()) > 0.95 else "does NOT clear"
    ladder = " → ".join(f"{dsr[k]:.3f}" for k in (n_eff, n_raw, 2 * n_raw, 4 * n_raw))
    line = (
        f"## {label} — selected `{selected}`  (deflation gate)\n"
        f"- Sharpe (ann)={sr_ann:.3f}  trial-Sharpe-dispersion={tstd:.4f}  "
        f"N_raw={n_raw}  N_eff(ONC)={n_eff}\n"
        f"- **DSR ladder** N∈[{n_eff}, {n_raw}, {2 * n_raw}, {4 * n_raw}] = [{ladder}]  "
        f"(0.95 threshold: {clears})\n"
        f"- N_raw counts only the roster horse-race; the cluster-rep reducer + F16 add an "
        f"implicit feature-selection search, so the higher rungs are the honest upper bound.\n"
        f"- CSCV-PBO={pbo:.3f}  (high = overfit selection)\n"
        f"- MinBTL @ ann-Sharpe {TARGET_ANN_SHARPE:.1f}: "
        f"[{minbtl_eff:.2f}y → {minbtl_raw:.2f}y]  vs OOS≈{oos_years:.2f}y\n"
    )
    print(f"{label}: DSR[{dsr[n_eff]:.3f}->{dsr[4 * n_raw]:.3f}] PBO={pbo:.3f} "
          f"MinBTL[{minbtl_eff:.2f}->{minbtl_raw:.2f}]y OOS={oos_years:.2f}y Sharpe={sr_ann:.2f}")
    return line


def run() -> None:
    cfg = PipelineConfig(
        roster="default", cv_scheme="cpcv", use_macro=True,
        per_instrument_embargo=True, use_drift=True,  # pass-4: matches the emit deliverable
    )
    ohlcv, signals = load_clean_data()
    rets = load_returns_panel(kind="simple")
    rets = rets[rets.index >= cfg.predict_start]

    out = [
        "# S6.8 — deployment deflation gate (calibrated default path)\n",
        "No 'strategy works' claim. The gate deflates the §6 Sharpe for selection bias; it is "
        "reported as a RANGE over N ∈ [N_eff → N_raw] because a single backtest cannot pin the "
        "effective trial count. The trial universe is the roster candidates actually horse-raced "
        "(5 per class); the pooled gate treats the full 5×3 program as the search space (a "
        "conservative upper N). DSR uses Bailey–López de Prado (2014); PBO uses CSCV "
        "(Bailey–Borwein–LdP–Zhu 2017).\n",
    ]
    pooled_trials: dict[str, pd.Series] = {}
    sel_metas = []
    for cls in CLASSES:
        best, sel_meta, trial_meta = class_trials(cls, cfg, ohlcv, signals)
        sel_metas.append(sel_meta)
        sub = rets[[c for c in class_members(cls) if c in rets.columns]]
        sel_net, _ = barrier_backtest(sel_meta, sub)
        trial_nets = {n: barrier_backtest(m, sub)[0] for n, m in trial_meta.items()}
        for n, s in trial_nets.items():
            pooled_trials[f"{cls}:{n}"] = s
        out.append(gate_block(cls, best, sel_net, trial_nets))

    all_net, all_rep = barrier_backtest(pd.concat(sel_metas, ignore_index=True), rets)
    out.append(gate_block("all-11 (pooled, N over the full 5×3 program)", "per-class winners",
                          all_net, pooled_trials))
    out.append(
        f"\n_Pooled net ann-Sharpe={all_rep['sharpe']:.3f}, "
        f"net total={all_rep['net_total_return']:.4f}, "
        f"cost drag (compounded)={all_rep['cost_drag_compounded']:.4f}._\n"
    )
    (results_dir() / "s6_deflation_gate.md").write_text("\n".join(out))
    print("wrote results/s6_deflation_gate.md")


if __name__ == "__main__":
    run()
