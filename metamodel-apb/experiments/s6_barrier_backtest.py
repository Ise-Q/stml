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
    stationary_bootstrap_cer_diff_ci,
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
    """(best, meta, preds, model_oof). meta = OOS [date, instrument, weight, t1]; model_oof =
    the calibrated MODELLING-sample OOF preds (dates <= modelling_end), the leakage-safe κᵢ inputs
    for EX.6 (already computed for calibration here, previously discarded)."""
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
    cal, oos_model, finite = fit_oos_calibrator(
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
    # EX.6: calibrated modelling-sample OOF preds (purged-OOS, dates <= modelling_end, i.e.
    # < predict_start → non-circular). Same edge/variance formula as the circular diagnostic.
    model_oof = pd.DataFrame(
        {
            "date": dates[mmask][finite],
            "instrument": pooled["instrument"].to_numpy()[mmask][finite],
            "p_hat": np.clip(cal.transform(oos_model[finite]), 0.0, 1.0),
            "y": y[mmask][finite],
        }
    )
    return best, meta, preds, model_oof


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


def standardise_then_repool_tm(sleeves, *, _return_debug: bool = False):
    """S5.12 (LR-9 rec #5) — vol-target each sleeve to a COMMON scale by dividing BOTH its market
    return and its signed PnL by that sleeve's market return-std (the SAME factor, so the
    ``pnl = side·mkt`` identity is preserved), then pool and re-estimate the Treynor–Mazuy γ.

    Standardising the regressor rescales TM's γ by the sleeve's σ, so large-scale sleeves stop
    dominating the pooled quadratic — the mechanism behind the +1.18 Simpson's-paradox artefact.
    The diagnostic is the **sign collapse** of the positive raw-pooled γ toward the (negative)
    trade-count-weighted average of the per-sleeve γ. Diagnostic-only: writes nothing.
    ``sleeves`` = list of ``(label, mkt, pnl, gamma_sleeve, n)``. Returns
    ``(gamma_std, t_std, p_std, gamma_weighted_avg)`` (+ a debug dict if requested).
    """
    mkts, pnls, gammas, ns, dbg = [], [], [], [], []
    for _label, mkt, pnl, g_sleeve, n in sleeves:
        m = np.asarray(mkt, dtype=float)
        p = np.asarray(pnl, dtype=float)
        s = float(np.std(m, ddof=1)) if m.size > 1 else 0.0
        if not np.isfinite(s) or s == 0.0 or m.size < 4:
            continue
        m_std, p_std_sleeve = m / s, p / s  # divide both by the SAME factor → preserves identity
        mkts.append(m_std)
        pnls.append(p_std_sleeve)
        gammas.append(float(g_sleeve))
        ns.append(int(n))
        dbg.append((m_std, p_std_sleeve))
    gamma_std, t_std, p_std = treynor_mazuy(np.concatenate(mkts), np.concatenate(pnls))
    gamma_wavg = float(np.average(gammas, weights=np.asarray(ns, dtype=float)))
    if _return_debug:
        return gamma_std, t_std, p_std, gamma_wavg, {"per_sleeve": dbg}
    return gamma_std, t_std, p_std, gamma_wavg


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


def resize(all_preds: pd.DataFrame, cfg: PipelineConfig, rets, kappa_map: dict | None) -> pd.Series:
    """Re-size every OOS bet with an optional per-instrument κ map → barrier-exact net returns.
    NON-deliverable (writes nothing); shared by the S6.15 taper gate and the EX.6 κᵢ gate."""
    pt, sl = cfg.pt_sl
    w = []
    for row in all_preds.itertuples(index=False):
        if not (pd.notna(row.ann_vol) and pd.notna(row.side)):
            w.append(0.0)
            continue
        k = kappa_map.get(row.instrument, KAPPA) if kappa_map else KAPPA
        w.append(
            position_weight(
                side=row.side, p=row.prediction, b=pt, d=sl, realised_vol=row.ann_vol,
                target_vol=TARGET_VOL, kappa=k, taper_width=TAPER_WIDTH,
            )
        )
    alt = all_preds[["date", "instrument"]].copy()
    alt["weight"] = w
    alt["t1"] = all_preds["t1"].to_numpy()
    return barrier_backtest(alt, rets)[0]


def leakage_safe_kappa(model_oof: pd.DataFrame, predict_start) -> dict:
    """Per-instrument Baker–McHale κᵢ from MODELLING-sample OOF preds only (EX.6, non-circular).

    Identical edge/variance formula to the circular OOS diagnostic in ``cer_gate_block`` — only the
    estimation window moves (dates strictly < predict_start). The window check is the leakage guard
    the pass-4 OOS-estimated κᵢ failed."""
    if not (pd.DatetimeIndex(model_oof["date"]) < pd.Timestamp(predict_start)).all():
        raise AssertionError("EX.6 κᵢ estimation window leaks into the OOS deliverable window")
    return {
        inst: float(
            kappa_baker_mchale(abs(float(np.mean(g["p_hat"])) - 0.5), float(np.var(g["p_hat"])))
        )
        for inst, g in model_oof.groupby("instrument")
    }


def ex6_gate_block(model_oof_all, all_preds, cfg, rets, net_base) -> tuple[str, bool]:
    """EX.6 — leakage-safe per-instrument κᵢ, gated on (i) a >5% relative OOS-CER gain AND (ii) a
    paired studentised CER-difference bootstrap CI excluding 0. Returns ``(markdown, adopt)``; the
    decision is surfaced either way and adoption HALTs before any re-emit (expected: revert)."""
    kap_ls = leakage_safe_kappa(model_oof_all, cfg.predict_start)
    net_kappa_ls = resize(all_preds, cfg, rets, kap_ls)
    cer_base = certainty_equivalent(net_base)
    cer_ls = certainty_equivalent(net_kappa_ls)
    rel_gain = (cer_ls - cer_base) / abs(cer_base) if cer_base else float("nan")
    ci_lo, ci_hi = stationary_bootstrap_cer_diff_ci(
        net_kappa_ls.to_numpy(), net_base.to_numpy(), risk_aversion=5.0, seed=cfg.seed
    )
    adopt = bool(rel_gain > 0.05 and ci_lo > 0)
    decision = (
        "ADOPT leakage-safe κᵢ"
        if adopt
        else "REVERT to flat κ=0.25 (leakage-safe κᵢ gain immaterial / CI contains 0)"
    )
    print(
        f"EX.6 GATE: base CER={cer_base:.6f} | κᵢ(leak-safe) CER={cer_ls:.6f} "
        f"(rel {rel_gain:+.1%}) | CER-diff 95% CI=[{ci_lo:.6f},{ci_hi:.6f}] -> {decision}"
    )
    md = (
        "## EX.6 — leakage-safe per-instrument κᵢ (sizing follow-up)\n"
        f"- flat κ=0.25 (deliverable): OOS CER(daily, γ=5)={cer_base:.6f}\n"
        f"- **leakage-safe κᵢ** (modelling-sample OOF, dates < predict_start; "
        f"κᵢ∈[{min(kap_ls.values()):.3f},{max(kap_ls.values()):.3f}]): OOS CER={cer_ls:.6f} "
        f"(rel {rel_gain:+.1%})\n"
        f"- paired studentised CER-difference 95% CI = [{ci_lo:.6f}, {ci_hi:.6f}] "
        f"({'EXCLUDES 0' if ci_lo > 0 else 'contains 0'})\n"
        f"- **Decision: {decision}.** Adoption requires BOTH a +5% relative gain and a CI "
        f"excluding 0; contrast the **circular OOS κᵢ** (look-ahead) reported in S6.15.\n"
    )
    return md, adopt


def cer_gate_block(all_preds: pd.DataFrame, cfg: PipelineConfig, rets, net_base: pd.Series) -> str:
    """S6.15 CER gate. The DECISION is made only on the **leakage-safe** smooth taper (a fixed
    function of p̂, no estimation): adopt iff it strictly raises the OOS certainty-equivalent over
    the flat-κ / hard-floor deliverable. The per-instrument Baker–McHale κᵢ variant is reported as
    a **circular diagnostic only** — its κᵢ is estimated on the OOS window itself, so any CER gain
    is look-ahead, not a valid out-of-sample improvement (a leakage-safe κᵢ is EX.6 below)."""
    kap = {  # OOS-estimated κᵢ — CIRCULAR, diagnostic only
        inst: float(kappa_baker_mchale(abs(float(np.mean(g["prediction"])) - 0.5),
                                       float(np.var(g["prediction"]))))
        for inst, g in all_preds.groupby("instrument")
    }
    net_taper = resize(all_preds, cfg, rets, None)   # leakage-safe: flat κ + smooth taper
    net_kappa = resize(all_preds, cfg, rets, kap)    # circular diagnostic: + OOS-fit κᵢ
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
    model_oofs = []  # EX.6: leakage-safe modelling-sample OOF preds per class
    sleeves = []  # S5.12: per-sleeve (mkt, pnl, γ, n) for the standardise-then-repool TM
    for cls in CLASSES:
        best, meta, preds, model_oof = class_oos_meta(cls, cfg, ohlcv, signals)
        metas.append(meta)
        all_preds.append(preds)
        model_oofs.append(model_oof)
        sub = rets[[c for c in class_members(cls) if c in rets.columns]]
        _, simple = backtest_strategy(meta[["date", "instrument", "weight"]], sub,
                                      max_holding=cfg.max_holding)
        net_bex, bex = barrier_backtest(meta, sub)
        mkt_c, pnl_c = directional_calls(meta, sub)[2:]  # S5.12: acted-trade timing series
        g_c = float(treynor_mazuy(mkt_c, pnl_c)[0]) if mkt_c.size else float("nan")
        sleeves.append((cls, mkt_c, pnl_c, g_c, int(mkt_c.size)))
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

    # EX.6 — leakage-safe per-instrument κᵢ (non-circular), gated on the two-part meaningful bar.
    model_oof_all = pd.concat(model_oofs, ignore_index=True)
    ex6_md, ex6_adopt = ex6_gate_block(model_oof_all, pooled_preds, cfg, rets, net_all)
    out.append(ex6_md)
    if ex6_adopt:
        print("HALT: EX.6 cleared the meaningful-gain bar — do NOT re-emit; surface for approval.")
        (results_dir() / "EX6_ADOPT_HALT.txt").write_text(ex6_md)

    # S5.12 — standardise-then-repool TM (in-data aggregation-artefact proof). Diagnostic-only.
    g_std, t_std, p_std, g_wavg = standardise_then_repool_tm(sleeves)
    raw_pool_g = float(treynor_mazuy(*directional_calls(allmeta, rets)[2:])[0])
    collapsed = bool(
        np.sign(g_std) == np.sign(g_wavg) or abs(g_std - g_wavg) < abs(raw_pool_g - g_wavg)
    )
    verdict = (
        "COLLAPSES toward the negative sleeve average → artefact confirmed in our own data"
        if collapsed
        else "DOES NOT collapse — HALT (contradicts the S5.11 resolution)"
    )
    out.append(
        "## S5.12 — standardise-then-repool TM (in-data aggregation-artefact proof, LR-9 rec #5)\n"
        f"- raw pooled γ (unstandardised) = {raw_pool_g:+.4f} (the apparent 'timing' artefact)\n"
        f"- **standardised-pooled γ = {g_std:+.4f}** (t={t_std:.2f}, p={p_std:.3f})\n"
        f"- trade-count-weighted average of per-sleeve γ = {g_wavg:+.4f}\n"
        f"- **{verdict}**\n"
    )
    print(
        f"S5.12: raw pooled γ={raw_pool_g:+.4f} -> standardised-pooled γ={g_std:+.4f} "
        f"(weighted avg {g_wavg:+.4f}) | {'COLLAPSE' if collapsed else 'NO-COLLAPSE (HALT)'}"
    )
    if not collapsed:
        print("HALT: S5.12 γ did not collapse — surface; contradicts the S5.11 resolution.")
        (results_dir() / "S512_NO_COLLAPSE_HALT.txt").write_text(out[-1])

    # persist the pooled net return series so §6 inference is reproducible from an artifact
    net_all.rename("net_return").to_frame().to_csv(results_dir() / "s6_net_returns.csv")
    (results_dir() / "s6_barrier_backtest.md").write_text("\n".join(out))
    print("wrote results/s6_barrier_backtest.md + s6_net_returns.csv")


if __name__ == "__main__":
    run()
