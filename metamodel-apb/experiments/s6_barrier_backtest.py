"""S6 — barrier-exact + cost-aware backtest on real OOS data (feeds §6).

Refits the shipped default-path model per class, predicts the OOS window, sizes with fractional-
Kelly × vol-target, then runs BOTH the simple fixed-horizon backtest and the new barrier-exact
backtest (exit on the actual t1 touch, overlapping labels netted) with the Grinold–Kahn cost
model. Reports Sharpe/Sortino/vol/MaxDD + the brief's turnover & average-holding-period + the
gross-vs-net cost split, per class and pooled.
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

from alken_metamodel.backtest import (  # noqa: E402
    backtest_strategy,
    barrier_backtest,
    certainty_equivalent,
)
from alken_metamodel.deflation import probabilistic_sharpe_ratio, sharpe_ratio  # noqa: E402
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
from alken_metamodel.signal_analysis import (  # noqa: E402
    henriksson_merton,
    pesaran_timmermann,
    treynor_mazuy,
)
from alken_metamodel.significance import (  # noqa: E402
    ljung_box_test,
    min_track_record_length,
    sharpe_ci_analytic,
    stationary_bootstrap_sharpe_ci,
    t_statistic,
)
from alken_metamodel.sizing import (  # noqa: E402
    KAPPA,
    TAPER_WIDTH,
    TARGET_VOL,
    cer_improves,
    kappa_baker_mchale,
    position_weight,
)

ANN = 252


def class_oos_meta(cls: str, cfg: PipelineConfig, ohlcv, signals):
    """(best_model, meta) where meta = OOS [date, instrument, weight, t1] for the class."""
    set_seeds(cfg.seed)
    pooled = build_class_panel(ohlcv, signals, class_members(cls), cfg)
    cols = feature_columns(pooled)
    X = pooled[cols]
    y = pooled["bin"].to_numpy()
    t1 = pooled["t1"]
    inst = pd.Series(pooled["instrument"].to_numpy(), index=X.index)  # S2.6 per-instrument embargo
    dates = pd.DatetimeIndex(pooled["date"])
    mmask = np.asarray(dates <= cfg.modelling_end)
    pmask = np.asarray((dates >= cfg.predict_start) & (dates <= cfg.predict_end))
    sw = balanced_sample_weight(y[mmask], base=pooled["weight"].to_numpy()[mmask])
    best, _ = select_model(X[mmask], y[mmask], t1[mmask], sw, cfg, instruments=inst[mmask])
    model = _roster_factory(cfg)(seed=cfg.seed)[best]
    model.fit(X[mmask], y[mmask], sample_weight=sw)
    # size on the CALIBRATED p̂ (matches the shipped deliverable; pass-3 S6.11)
    cal, _, _ = fit_oos_calibrator(
        lambda: _roster_factory(cfg)(seed=cfg.seed)[best], X[mmask], y[mmask], t1[mmask], sw, cfg,
        instruments=inst[mmask],
    )
    proba = np.clip(cal.transform(model.predict_act_proba(X[pmask])), 0.0, 1.0)
    preds = pd.DataFrame(
        {
            "date": dates[pmask],
            "instrument": pooled["instrument"].to_numpy()[pmask],
            "prediction": proba,
            "side": pooled["side"].to_numpy()[pmask],
            "ann_vol": pooled["f2_vol_20"].to_numpy()[pmask],
        }
    )
    meta = strategy_weights(preds, cfg)  # date, instrument, weight (row-aligned to preds)
    meta["t1"] = pd.DatetimeIndex(t1[pmask].to_numpy())
    preds = preds.assign(t1=pd.DatetimeIndex(t1[pmask].to_numpy()))
    return best, meta, preds


def _fmt(d: dict, keys) -> str:
    return "  ".join(f"{k}={d[k]:.4f}" for k in keys if k in d and pd.notna(d[k]))


def directional_calls(meta: pd.DataFrame, returns_panel: pd.DataFrame):
    """OOS acted-trade timing series: (realised_sign, predicted_sign, market_ret, signed_pnl).

    The signs feed the directional tests (PT primary, H–M proxy); the continuous market return
    and the signed-side PnL feed the Treynor–Mazuy convexity regression (S5.10).
    """
    real, pred, mkt, pnl = [], [], [], []
    cal = returns_panel.index
    for row in meta.itertuples(index=False):
        if row.weight == 0 or row.instrument not in returns_panel.columns:
            continue
        seg = returns_panel[row.instrument][(cal >= row.date) & (cal < row.t1)]
        if seg.empty:
            continue
        r = float((1.0 + seg).prod() - 1.0)
        side = float(np.sign(row.weight))
        real.append(np.sign(r))
        pred.append(side)
        mkt.append(r)
        pnl.append(side * r)
    return (
        np.asarray(real, dtype=float),
        np.asarray(pred, dtype=float),
        np.asarray(mkt, dtype=float),
        np.asarray(pnl, dtype=float),
    )


def utility_line(label: str, net: pd.Series, meta: pd.DataFrame, returns: pd.DataFrame) -> str:
    """Timing tests (PT primary, TM convexity, H–M proxy) + certainty-equivalent (S5.10)."""
    real, pred, mkt, pnl = directional_calls(meta, returns)
    if real.size:
        hit, z, _ = henriksson_merton(real, pred)
        pt_stat, pt_p = pesaran_timmermann(real, pred)
        tm_g, tm_t, tm_p = treynor_mazuy(mkt, pnl)
    else:
        hit = z = pt_stat = pt_p = tm_g = tm_t = tm_p = float("nan")
    cer = certainty_equivalent(net)
    print(f"{label} utility: PT={pt_stat:.2f} (p={pt_p:.3f}) | TM g={tm_g:.4f} (t={tm_t:.2f}) | "
          f"H-M(proxy) hit={hit:.3f} z={z:.2f} | CER(daily)={cer:.6f}")
    return (
        f"- utility (S5.10): **Pesaran–Timmermann** stat={pt_stat:.2f} (p={pt_p:.3f}, *primary*); "
        f"**Treynor–Mazuy** γ={tm_g:.4f} (t={tm_t:.2f}, p={tm_p:.3f}); Henriksson–Merton proxy "
        f"hit={hit:.3f} (z={z:.2f}, base-rate-sensitive); "
        f"certainty-equivalent (daily, γ=5)={cer:.6f}\n"
    )


def significance_block(net: pd.Series) -> str:
    """S6.14 significance-first inference (the PRIMARY §6 read): t-stat → studentised stationary
    block-bootstrap CI → Lo/Opdyke band → PSR(0)/MinTRL. Ljung–Box gates the √252 annualisation;
    DSR/PBO are demoted to s6_deflation_gate.md."""
    from scipy.stats import kurtosis as _ku
    from scipy.stats import skew as _sk

    r = net.dropna().to_numpy()
    n = len(r)
    sr_d = sharpe_ratio(r)
    t = t_statistic(r)
    _, lb_p = ljung_box_test(r, lags=10)
    boot_lo, boot_hi = stationary_bootstrap_sharpe_ci(r, alpha=0.05, reps=2000, seed=42)
    an_lo, an_hi = sharpe_ci_analytic(r, alpha=0.05)
    sk, ku = float(_sk(r)), float(_ku(r, fisher=False))
    psr0 = probabilistic_sharpe_ratio(sr_d, 0.0, n, skew=sk, kurt=ku)
    mintrl = min_track_record_length(sr_d, 0.0, skew=sk, kurt=ku, prob=0.95)
    ann = float(np.sqrt(ANN))
    contains0 = "contains 0 (no significant edge)" if boot_lo <= 0 <= boot_hi else "EXCLUDES 0"
    sig = "significant" if abs(t) > 1.96 else "NOT significant"
    lb_note = "IID-consistent" if lb_p > 0.05 else "serial-correlated (√252 overstates ann)"
    print(f"SIGNIFICANCE: t={t:.3f} (n={n}, {sig}) | boot95%=[{boot_lo:.3f},{boot_hi:.3f}] "
          f"{contains0} | PSR(0)={psr0:.3f} MinTRL={mintrl:.0f}d LB_p={lb_p:.3f}")
    return (
        "## Significance-first inference (S6.14 — the PRIMARY §6 read)\n"
        f"- per-period Sharpe={sr_d:.4f}; **t = SR·√n = {t:.3f}** (n={n}) — **{sig}** at 5% "
        f"before any deflation.\n"
        f"- **Studentised stationary block-bootstrap 95% CI (PRIMARY)**: per-period "
        f"[{boot_lo:.3f}, {boot_hi:.3f}] → ann ×√252 [{boot_lo * ann:.3f}, {boot_hi * ann:.3f}] "
        f"— {contains0}.\n"
        f"- Lo/Opdyke analytic band (per-period): [{an_lo:.3f}, {an_hi:.3f}].\n"
        f"- PSR(0)={psr0:.3f}; **MinTRL={mintrl:.0f} days** vs {n} available.\n"
        f"- Ljung–Box(10) p={lb_p:.3f} ({lb_note}); DSR/PBO demoted (see s6_deflation_gate.md).\n"
    )


def cer_gate_block(all_preds: pd.DataFrame, cfg: PipelineConfig, rets, net_base: pd.Series) -> str:
    """S6.15 CER gate. The DECISION is made only on the **leakage-safe** smooth taper (a fixed
    function of p̂, no estimation): adopt iff it strictly raises the OOS certainty-equivalent over
    the flat-κ / hard-floor deliverable. The per-instrument Baker–McHale κᵢ variant is reported as
    a **circular diagnostic only** — its κᵢ is estimated on the OOS window itself, so any CER gain
    is look-ahead, not a valid out-of-sample improvement (a leakage-safe κᵢ needs modelling-sample
    residuals — future work)."""
    pt, sl = cfg.pt_sl

    def resize(kappa_map: dict | None) -> pd.Series:
        w = []
        for row in all_preds.itertuples(index=False):
            if not (pd.notna(row.ann_vol) and pd.notna(row.side)):
                w.append(0.0)
                continue
            k = kappa_map.get(row.instrument, KAPPA) if kappa_map else KAPPA
            w.append(position_weight(side=row.side, p=row.prediction, b=pt, d=sl,
                                     realised_vol=row.ann_vol, target_vol=TARGET_VOL,
                                     kappa=k, taper_width=TAPER_WIDTH))
        alt = all_preds[["date", "instrument"]].copy()
        alt["weight"] = w
        alt["t1"] = all_preds["t1"].to_numpy()
        return barrier_backtest(alt, rets)[0]

    kap = {  # OOS-estimated κᵢ — CIRCULAR, diagnostic only
        inst: float(kappa_baker_mchale(abs(float(np.mean(g["prediction"])) - 0.5),
                                       float(np.var(g["prediction"]))))
        for inst, g in all_preds.groupby("instrument")
    }
    net_taper = resize(None)        # leakage-safe: flat κ + smooth taper
    net_kappa = resize(kap)         # circular diagnostic: + OOS-fit κᵢ
    cer_base = certainty_equivalent(net_base)
    cer_taper = certainty_equivalent(net_taper)
    cer_kappa = certainty_equivalent(net_kappa)
    # Decision on the LEAKAGE-SAFE variant, with a materiality margin: sizing complexity must earn a
    # gain larger than the noise of a 127-day CER (≥10% of the baseline), else a tiny blip is overfit.
    margin = 0.10 * abs(cer_base)
    adopt = cer_improves(cer_taper, cer_base, min_gain=margin)
    decision = ("ADOPT smooth taper" if adopt
                else "REVERT to flat κ=0.25 / hard floor (leakage-safe taper gain immaterial)")
    print(f"CER GATE: base CER={cer_base:.6f} | taper(leak-safe) CER={cer_taper:.6f} "
          f"| +κᵢ(circular) CER={cer_kappa:.6f} -> {decision}")
    return (
        "## CER-gated sizing (S6.15)\n"
        f"- flat κ=0.25 / hard floor (deliverable): OOS CER(daily, γ=5)={cer_base:.6f}\n"
        f"- **leakage-safe smooth taper** (decision basis): OOS CER={cer_taper:.6f}\n"
        f"- +per-instrument Baker–McHale κᵢ (κᵢ∈[{min(kap.values()):.3f},{max(kap.values()):.3f}], "
        f"**OOS-estimated → CIRCULAR diagnostic, not an adopt signal**): OOS CER={cer_kappa:.6f}\n"
        f"- **Decision: {decision}.** Shipped weights keep flat κ unless the *leakage-safe* taper "
        f"strictly improves OOS CER.\n"
    )


def run() -> None:
    cfg = PipelineConfig(
        roster="default", cv_scheme="cpcv", use_macro=True,
        per_instrument_embargo=True, use_drift=True,  # pass-4: matches the emit deliverable
    )
    ohlcv, signals = load_clean_data()
    rets = load_returns_panel(kind="simple")
    rets = rets[rets.index >= cfg.predict_start]  # restrict to the backtest window forward

    out = ["# S6 — barrier-exact + cost-aware backtest (real OOS, pass-4 path)\n"]
    metas = []
    all_preds = []
    for cls in CLASSES:
        best, meta, preds = class_oos_meta(cls, cfg, ohlcv, signals)
        metas.append(meta)
        all_preds.append(preds)
        sub = rets[[c for c in class_members(cls) if c in rets.columns]]
        _, simple = backtest_strategy(meta[["date", "instrument", "weight"]], sub,
                                      max_holding=cfg.max_holding)
        net_bex, bex = barrier_backtest(meta, sub)
        out.append(
            f"## {cls} — model `{best}` (calibrated sizing)\n"
            f"- simple (max_holding={cfg.max_holding}): "
            f"{_fmt(simple, ['sharpe','sortino','ann_vol','max_drawdown','total_return'])}\n"
            f"- barrier-exact: {_fmt(bex, ['sharpe','sortino','ann_vol','max_drawdown'])}  "
            f"turnover_ann={bex['ann_turnover']:.2f}  hold_days={bex['avg_holding_period']:.1f}  "
            f"cost={bex['total_cost']:.4f}  gross={bex['gross_total_return']:.4f} "
            f"net={bex['net_total_return']:.4f}  "
            f"cost_drag_compounded={bex['cost_drag_compounded']:.4f}\n"
            + utility_line(cls, net_bex, meta, sub)
        )
        print(f"{cls} [{best}] simple Sharpe={simple['sharpe']:.3f} | barrier-exact "
              f"Sharpe={bex['sharpe']:.3f} Sortino={bex['sortino']:.3f} "
              f"hold={bex['avg_holding_period']:.1f}d turn={bex['ann_turnover']:.1f} "
              f"gross={bex['gross_total_return']:.4f} net={bex['net_total_return']:.4f}")

    allmeta = pd.concat(metas, ignore_index=True)
    net_all, bex_all = barrier_backtest(allmeta, rets)
    out.append(
        f"## all 11 (pooled) — barrier-exact (calibrated sizing)\n"
        f"- {_fmt(bex_all, ['sharpe','sortino','ann_vol','max_drawdown'])}  "
        f"turnover_ann={bex_all['ann_turnover']:.2f}  "
        f"hold_days={bex_all['avg_holding_period']:.1f}  "
        f"cost={bex_all['total_cost']:.4f}  gross={bex_all['gross_total_return']:.4f} "
        f"net={bex_all['net_total_return']:.4f}  "
        f"cost_drag_compounded={bex_all['cost_drag_compounded']:.4f}\n"
        + utility_line("all-11", net_all, allmeta, rets)
    )
    print(f"ALL11 barrier-exact Sharpe={bex_all['sharpe']:.3f} "
          f"Sortino={bex_all['sortino']:.3f} net={bex_all['net_total_return']:.4f} "
          f"cost={bex_all['total_cost']:.4f} hold={bex_all['avg_holding_period']:.1f}d")

    pooled_preds = pd.concat(all_preds, ignore_index=True)
    out.append(significance_block(net_all))
    out.append(cer_gate_block(pooled_preds, cfg, rets, net_all))
    # persist the pooled net return series so §6 inference is reproducible from an artifact
    net_all.rename("net_return").to_frame().to_csv(results_dir() / "s6_net_returns.csv")
    (results_dir() / "s6_barrier_backtest.md").write_text("\n".join(out))
    print("wrote results/s6_barrier_backtest.md + s6_net_returns.csv")


if __name__ == "__main__":
    run()
