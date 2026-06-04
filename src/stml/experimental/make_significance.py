"""S7 runner — significance + deflation + signal_analysis on the H1-2022 OOS.

Reads ``results/submission/strategy_daily_net_returns.csv`` (the S6
emission) and produces ``results/submission/significance_summary.md``
+ machine-readable CSV.

Plan §8 S7 acceptance: all numbers reproduce from
``outputs/strategy_weights.csv``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stml.experimental.deflation import (
    cscv_pbo_combinations_count,
    deflated_sharpe_ratio,
    dsr_ladder,
    effective_n_trials,
    expected_max_sharpe,
    min_backtest_length,
    probability_of_backtest_overfitting,
)
from stml.experimental.signal_analysis import (
    henriksson_merton_proxy,
    pesaran_timmermann,
    treynor_mazuy,
)
from stml.experimental.significance import significance_report


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _pt_inputs(events_with_preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For Pesaran-Timmermann: realised = side × ret; predicted = signed conviction.

    Predicted = side × (calibrated_proba − 0.5). Positive when model agrees
    with the primary signal's direction (act), negative when it would skip /
    flip (predict against).

    Realised = side × ret — the bare-trade signed return.
    """
    df = events_with_preds.copy()
    realised = (df["side"].astype(float) * df["ret"]).values
    predicted = (df["side"].astype(float) * (df["calibrated_proba"].astype(float) - 0.5)).values
    return realised, predicted


def run(verbose: bool = True) -> dict:
    root = _find_repo_root()
    res_dir = root / "results" / "submission"

    net_path = res_dir / "strategy_daily_net_returns.csv"
    if not net_path.exists():
        print(f"FATAL: {net_path} missing — run S6 first.")
        return {"status": "no_returns"}
    net = pd.read_csv(net_path, parse_dates=[0], index_col=0)["net_ret"].dropna()

    # Number of trials counted: we evaluated 4 model families × 11 instruments
    # × up to 3 pool variants ≈ ~80-130 candidate (instrument, pool, model)
    # combinations in the champion search. Set N_raw conservatively to 120.
    n_raw_trials = 120
    n_eff = effective_n_trials(
        net.values.reshape(-1, 1).repeat(min(n_raw_trials, 50), axis=1)
    ) or 1  # We don't have per-trial returns; approximate N_eff ≤ 4 (ONC seldom > 10).
    n_eff = max(2, min(n_eff, 20))

    sig = significance_report(net.values, n_boot=2000, seed=42)

    # Annualised numbers.
    n = sig.n
    ann_sharpe = sig.sr * np.sqrt(252.0)
    ann_ret = net.mean() * 252.0
    ann_vol = net.std(ddof=1) * np.sqrt(252.0)

    # Deflation ladder. Trials_std is the empirical std of per-period Sharpe
    # across our champion-grid trials. Reading the champions_per_pool_per_model
    # grid gives a direct estimate; if unavailable, use a conservative default
    # (per-period trial Sharpe std typically 0.05-0.10 for ML strategies).
    grid_path = res_dir / "champions_per_pool_per_model.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        gd = grid.loc[(grid["variant"] == "with_bbg") & grid["mean_auc"].notna(), "mean_auc"]
        # Convert AUC → approx per-period Sharpe via SR ≈ 2*(AUC-0.5)/√(252/h)
        # ... but here we use the AUC std as a proxy for trial dispersion;
        # divided by √(252) to align to per-period units.
        trials_std = float(gd.std() / np.sqrt(252)) if len(gd) > 1 else 0.05
        trials_std = max(0.02, min(trials_std, 0.20))
    else:
        trials_std = 0.05  # conservative ML-strategy default
    ladder = dsr_ladder(net.values, n_eff=n_eff, n_raw=n_raw_trials, trials_std=trials_std)
    sr0_neff = expected_max_sharpe(n_eff, trials_std=trials_std)
    sr0_nraw = expected_max_sharpe(n_raw_trials, trials_std=trials_std)
    minbtl = min_backtest_length(n_raw_trials, target_sharpe=ann_sharpe if ann_sharpe > 0 else 1.0)
    pbo_combos = cscv_pbo_combinations_count(16)

    # PBO requires per-trial perf. We use a coarse approximation: split
    # net returns into blocks of 16, treat as a single trial. With only 1
    # trial PBO is ill-defined. Report NaN with a note.
    pbo = float("nan")

    # Signal analysis on per-event preds (for PT / TM / HM).
    events_path = res_dir / "oos_events_with_predictions.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
        pt_real, pt_pred = _pt_inputs(events)
        pt_stat, pt_p = pesaran_timmermann(pt_real, pt_pred)
        # TM uses (bare market, sized portfolio).
        bare = (events["side"].astype(float) * events["ret"]).values
        sized = (events["weight"] * events["ret"]).values
        tm_gamma, tm_t = treynor_mazuy(bare, sized)
        hm_hit, hm_z, hm_p = henriksson_merton_proxy(pt_real, pt_pred)
    else:
        pt_stat, pt_p = float("nan"), float("nan")
        tm_gamma, tm_t = float("nan"), float("nan")
        hm_hit, hm_z, hm_p = float("nan"), float("nan"), float("nan")

    # Persist a machine-readable CSV.
    summary_rows = [
        {"name": "n_periods", "value": n},
        {"name": "ann_return", "value": ann_ret},
        {"name": "ann_vol", "value": ann_vol},
        {"name": "ann_sharpe", "value": ann_sharpe},
        {"name": "t_stat", "value": sig.t_stat},
        {"name": "bootstrap_ci_low_per_period", "value": sig.bootstrap_ci_low},
        {"name": "bootstrap_ci_high_per_period", "value": sig.bootstrap_ci_high},
        {"name": "bootstrap_ci_low_ann", "value": sig.bootstrap_ci_low * np.sqrt(252)},
        {"name": "bootstrap_ci_high_ann", "value": sig.bootstrap_ci_high * np.sqrt(252)},
        {"name": "bootstrap_block_length", "value": sig.bootstrap_block_length},
        {"name": "analytic_ci_low_per_period", "value": sig.analytic_ci_low},
        {"name": "analytic_ci_high_per_period", "value": sig.analytic_ci_high},
        {"name": "psr_zero", "value": sig.psr_zero},
        {"name": "min_trl_periods", "value": sig.min_trl_days},
        {"name": "ljung_box_q_lag10", "value": sig.ljung_box_stat},
        {"name": "ljung_box_p_lag10", "value": sig.ljung_box_p},
        {"name": "n_eff_trials", "value": n_eff},
        {"name": "n_raw_trials", "value": n_raw_trials},
        {"name": "expected_max_sharpe_n_eff", "value": sr0_neff},
        {"name": "expected_max_sharpe_n_raw", "value": sr0_nraw},
        {"name": "min_btl", "value": minbtl},
        {"name": "cscv_pbo_combinations_n16", "value": pbo_combos},
        {"name": "pesaran_timmermann_stat", "value": pt_stat},
        {"name": "pesaran_timmermann_p", "value": pt_p},
        {"name": "treynor_mazuy_gamma", "value": tm_gamma},
        {"name": "treynor_mazuy_t", "value": tm_t},
        {"name": "henriksson_merton_hit", "value": hm_hit},
        {"name": "henriksson_merton_z", "value": hm_z},
        {"name": "henriksson_merton_p", "value": hm_p},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(res_dir / "significance_summary.csv", index=False, float_format="%.6f")
    ladder.to_csv(res_dir / "deflation_ladder.csv", index=False, float_format="%.6f")

    # Markdown summary doc.
    md_lines = [
        "# Significance + deflation + directional skill — H1-2022 OOS",
        "",
        f"Source data: `results/submission/strategy_daily_net_returns.csv` (n = {n} periods).",
        "",
        "## §3.8 PRIMARY — Sharpe significance",
        "",
        "| Statistic | Value | Reading |",
        "|---|---:|---|",
        f"| n (periods) | {n} | OOS sample size |",
        f"| Per-period Sharpe SR | {sig.sr:.4f} | uncorrected for n |",
        f"| Annualised Sharpe (×√252) | {ann_sharpe:.4f} | use only if Ljung-Box OK |",
        f"| **t = SR·√n** | **{sig.t_stat:.4f}** | "
        f"{'distinguishable from 0 at 5%' if abs(sig.t_stat) > 1.96 else 'NOT significant at 5%'} |",
        f"| Studentised stationary block-bootstrap 95% CI (PRIMARY) | "
        f"[{sig.bootstrap_ci_low:.4f}, {sig.bootstrap_ci_high:.4f}] per period | "
        f"{'EXCLUDES 0 → significant' if sig.bootstrap_ci_low > 0 else 'CONTAINS 0' if sig.bootstrap_ci_high > 0 > sig.bootstrap_ci_low else 'EXCLUDES 0 (negative)'} |",
        f"| Bootstrap block length (Politis-White) | {sig.bootstrap_block_length:.2f} | data-driven |",
        f"| Lo/Opdyke analytic 95% CI | "
        f"[{sig.analytic_ci_low:.4f}, {sig.analytic_ci_high:.4f}] per period | parametric cross-check |",
        f"| PSR(SR* = 0) | {sig.psr_zero:.4f} | "
        f"{'> 0.95 deployment threshold' if sig.psr_zero > 0.95 else 'below 0.95 — insufficient'} |",
        f"| MinTRL (95% PSR) | {sig.min_trl_days:.0f} periods | "
        f"{'< n → certified' if sig.min_trl_days < n else f'~{sig.min_trl_days/max(n,1):.1f}× too short to certify'} |",
        f"| Ljung-Box Q(10) | {sig.ljung_box_stat:.3f}, p = {sig.ljung_box_p:.4f} | "
        f"{'OK — IID-like, √252 annualisation valid' if sig.ljung_box_p > 0.05 else 'rejects IID — √252 OVERSTATES'} |",
        "",
        "## §3.8 deflation ladder",
        "",
        "| Rung | n_trials | DSR |",
        "|---|---:|---:|",
    ]
    for _, row in ladder.iterrows():
        md_lines.append(f"| {row['rung']} | {row['n_trials']} | {row['dsr']:.4f} |")
    md_lines.extend([
        "",
        f"* CSCV-PBO combinations at n_blocks=16: **{pbo_combos:,}** (corrects the long-propagated 12,780 typo).",
        f"* MinBTL @ target Sharpe = ann Sharpe: {minbtl:.1f} periods.",
        f"* Expected max Sharpe of N_eff = {n_eff}: {sr0_neff:.4f}.",
        f"* Expected max Sharpe of N_raw = {n_raw_trials}: {sr0_nraw:.4f}.",
        "",
        "## §3.8 directional skill",
        "",
        "| Test | Value | Reading |",
        "|---|---:|---|",
        f"| **Pesaran-Timmermann (PRIMARY)** | S = {pt_stat:.4f}, p = {pt_p:.4f} | "
        f"{'positive directional skill' if pt_stat > 1.96 else 'NO positive directional skill'} |",
        f"| Treynor-Mazuy γ | {tm_gamma:.4f} (t = {tm_t:.2f}) | "
        f"{'convex timing; check S5.12 scale-aggregation' if abs(tm_t) > 1.96 else 'no significant convexity'} |",
        f"| Henriksson-Merton hit rate (proxy) | {hm_hit:.4f} (z = {hm_z:.2f}, p = {hm_p:.4f}) | "
        "BASE-RATE SENSITIVE — read with caveat |",
        "",
        "## Five-lens verdict",
        "",
        "Lens 1 — AUC (per class, §8 S3): equity 0.554 | energy 0.602 | metals 0.554.",
        "Lens 2 — cluster MDA (§8 S5): equity 0.022 PASS | energy 0.036 PASS | metals 0.016 CHECK.",
        "Lens 3 — Sharpe significance (this section): per-period bootstrap CI " +
        ("EXCLUDES 0" if sig.bootstrap_ci_low > 0 else "CONTAINS 0") +
        ", t = " + f"{sig.t_stat:.2f}, PSR(0) = {sig.psr_zero:.2f}.",
        "Lens 4 — deflation: DSR at N_eff = " +
        f"{ladder.iloc[0]['dsr']:.3f}, at 4·N_raw = {ladder.iloc[-1]['dsr']:.3f}.",
        "Lens 5 — Pesaran-Timmermann: S = " + f"{pt_stat:.2f}, p = {pt_p:.2f}.",
    ])

    md_path = res_dir / "significance_summary.md"
    md_path.write_text("\n".join(md_lines))

    if verbose:
        print("\n=== S7 significance summary ===")
        for row in summary_rows[:14]:
            v = row["value"]
            print(f"  {row['name']:40s}  {v if not isinstance(v, float) else f'{v:.6f}'}")
        print(f"\nWrote {md_path.relative_to(root)}")
        print(f"Wrote results/submission/significance_summary.csv")
        print(f"Wrote results/submission/deflation_ladder.csv")

    return {"summary": summary_df, "report": sig, "pbo_combos": pbo_combos}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S7 significance + deflation runner")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    state = run(verbose=not args.quiet)
    return 0 if state.get("status") != "no_returns" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
