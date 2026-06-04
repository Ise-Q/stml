"""OOS evaluation — lecture-aligned strategy comparison.

Methods:
    A          — Benchmark: ŷ = side (primary-only, full conviction)
    B-aon      — all_or_nothing: ŷ = side · 1(p̂ > 0.5)
    B-sops     — SOPS sigmoid fitted on OOF Sharpe
    C-vsn-lstm — VSN+LSTM neural model (loads saved weights)
    D-tft      — Temporal Fusion Transformer (loads saved weights)

Portfolio construction follows StrategyWeights lecture (slides 38–43):
    - Simple returns r_t = (P_t − P_{t-1}) / P_{t-1}
    - EWMA vol (span=60), exact lecture recurrence, annualised ×√252, σ_tgt=10%
    - 1/K cross-sectional average (K = 11 instruments, flat = cash)
    - Lag: w_t earns r_{t+1} (position stamped at t_start, shift(-1) in backtest)
    - Net of Grinold-Kahn costs: 2 bps half-spread + 10 bps × |Δw|

Outputs:
    results/strategy_eval/
        eval_summary.csv / .md
        net_returns_<method>.csv

Usage:
    .venv/bin/python src/stml/new_work/evaluate.py [--no-sops] [--no-neural] [--quiet]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from stml.io import load_data, load_returns_panel
from stml.new_work import config
from stml.new_work.weights import make_weights, load_oos_events, MethodName
from stml.new_work.targeting import build_vol_panel
from stml.experimental.backtest import barrier_backtest, performance_metrics, BacktestReport
from stml.experimental.significance import significance_report

EVAL_DIR = _REPO / "results" / "strategy_eval"

METHODS_BASE: list[MethodName] = ["A", "B-aon"]

METHOD_LABELS: dict[str, str] = {
    "A":          "A Benchmark",
    "B-aon":      "B all_or_nothing",
    "B-sops":     "B SOPS",
    "C-vsn-lstm": "C VSN+LSTM",
    "D-tft":      "D TFT",
}


# ---------------------------------------------------------------------------
# Breakeven cost
# ---------------------------------------------------------------------------


def breakeven_halfspread(
    net_returns: pd.Series,
    weights: pd.DataFrame,
) -> float:
    """Half-spread (bps) at which net Sharpe hits zero."""
    from stml.experimental.cost_model import annualised_turnover
    turnover = annualised_turnover(weights)
    gross_mean = net_returns.mean() * 252.0
    if turnover <= 0:
        return float("nan")
    be_bps = (gross_mean / turnover) * 10_000
    return float(be_bps)


# ---------------------------------------------------------------------------
# Single-method evaluation
# ---------------------------------------------------------------------------


def evaluate_method(
    method: MethodName,
    *,
    oos_events: pd.DataFrame,
    vol_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    oof_df: pd.DataFrame | None = None,
    ohlcv: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
    verbose: bool = True,
) -> tuple[BacktestReport, dict]:
    """Run barrier_backtest for one method and return (report, metrics_dict)."""
    events, weights_series = make_weights(
        method,
        oos_events=oos_events,
        vol_panel=vol_panel,
        oof_df=oof_df,
        ohlcv=ohlcv,
        signals=signals,
    )

    report = barrier_backtest(events, weights_series, returns_panel)
    m = report.metrics.copy()

    sr_report = significance_report(report.net_returns.values, n_boot=2000, seed=42)
    m["bootstrap_ci_low"]  = float(sr_report.bootstrap_ci_low * np.sqrt(252))
    m["bootstrap_ci_high"] = float(sr_report.bootstrap_ci_high * np.sqrt(252))
    m["t_stat"]            = float(sr_report.t_stat)
    m["psr_zero"]          = float(sr_report.psr_zero)

    m["breakeven_halfspread_bps"] = breakeven_halfspread(
        report.net_returns, report.weights
    )

    n_total = len(events)
    n_active = int((weights_series.abs() > 0).sum())
    m["frac_events_taken"] = n_active / max(n_total, 1)
    m["n_events_active"]   = n_active
    m["n_events_total"]    = n_total

    if verbose:
        _print_method_summary(method, m, report)

    return report, m


def _print_method_summary(method: MethodName, m: dict, report: BacktestReport) -> None:
    label = METHOD_LABELS.get(method, method)
    net = report.net_returns
    print(f"\n  ── {label} ──────────────────────────────────────────")
    print(f"    OOS period:  {net.index.min().date()} → {net.index.max().date()} "
          f"({m['n']} days)")
    print(f"    Ann return:  {m['ann_return']*100:+.1f}%")
    print(f"    Ann vol:     {m['ann_vol']*100:.1f}%")
    print(f"    Sharpe:      {m['sharpe']:.3f}   t={m['t_stat']:.2f}")
    print(f"    Max DD:      {m['max_dd']*100:.1f}%")
    print(f"    Events taken: {m['n_events_active']}/{m['n_events_total']} "
          f"({m['frac_events_taken']*100:.0f}%)")
    print(f"    Breakeven ½-spread: {m['breakeven_halfspread_bps']:.1f} bps")
    print(f"    Ann Sharpe CI (bootstrap 95%): "
          f"[{m['bootstrap_ci_low']:.3f}, {m['bootstrap_ci_high']:.3f}]")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


DISPLAY_COLS = [
    ("ann_return",               "Ann return",         "{:+.2%}"),
    ("ann_vol",                  "Ann vol",            "{:.2%}"),
    ("sharpe",                   "Sharpe",             "{:.3f}"),
    ("t_stat",                   "t-stat",             "{:.2f}"),
    ("bootstrap_ci_low",         "SR 95% CI low",      "{:.3f}"),
    ("bootstrap_ci_high",        "SR 95% CI high",     "{:.3f}"),
    ("psr_zero",                 "PSR(SR*=0)",         "{:.3f}"),
    ("max_dd",                   "Max DD",             "{:.2%}"),
    ("total_cost_bps",           "Total cost (bps)",   "{:.1f}"),
    ("turnover_per_year",        "Annual turnover",    "{:.2f}"),
    ("breakeven_halfspread_bps", "Breakeven ½-spread", "{:.1f} bps"),
    ("frac_events_taken",        "Events taken",       "{:.0%}"),
]


def comparison_table(
    results: dict[str, dict],
    label_map: dict[str, str],
    display_cols: list[tuple],
) -> pd.DataFrame:
    rows = []
    for key, label, fmt in display_cols:
        row = {"Metric": label}
        for meth, m in results.items():
            v = m.get(key, float("nan"))
            try:
                row[label_map.get(meth, meth)] = fmt.format(v)
            except (ValueError, TypeError):
                row[label_map.get(meth, meth)] = "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Metric")


def _to_markdown(df: pd.DataFrame) -> str:
    col_header = " | ".join(["Metric"] + list(df.columns))
    sep = "|".join(["---"] * (len(df.columns) + 1))
    lines = [f"| {col_header} |", f"| {sep} |"]
    for idx, row in df.iterrows():
        line = " | ".join([str(idx)] + [str(v) for v in row.values])
        lines.append(f"| {line} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(
    include_sops: bool = True,
    include_neural: bool = True,
    verbose: bool = True,
) -> dict[MethodName, dict]:
    """Run OOS evaluation and return metrics dict keyed by method name."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Strategy evaluation — OOS comparison (lecture-aligned)")
    print(f"Vol: {config.VOL_METHOD}(span={config.EWMA_SPAN})  "
          f"σ_tgt={config.SIGMA_TGT:.0%}  max_lev={config.MAX_LEVERAGE}  "
          f"returns=simple  1/K={11}")
    print("=" * 70)

    print("\nLoading data …")
    ohlcv, _ = load_data()
    returns_panel = load_returns_panel(kind="simple")
    oos_events = load_oos_events()

    print(f"  OOS events: {len(oos_events)} across "
          f"{oos_events['instrument'].nunique()} instruments")
    print(f"  OOS period: {oos_events['t_start'].min().date()} → "
          f"{oos_events['t_start'].max().date()}")

    print(f"\nBuilding {config.VOL_METHOD} vol panel (span={config.EWMA_SPAN}) …")
    vol_panel = build_vol_panel(
        ohlcv,
        method=config.VOL_METHOD,
        window=config.LOOKBACK_L,
        span=config.EWMA_SPAN,
    )

    signals = None
    if include_neural:
        from stml.new_work.data import load_signals
        signals = load_signals()

    oof_df: pd.DataFrame | None = None
    methods: list[MethodName] = list(METHODS_BASE)

    if include_sops:
        oof_path = config.OOF_PROBA_PATH
        if oof_path.exists():
            from stml.new_work.data import load_oof_probabilities
            oof_df = load_oof_probabilities()
            methods.append("B-sops")
            print(f"  OOF probs loaded: {len(oof_df)} training events")
        else:
            print(f"  OOF probs not found at {oof_path} — skipping B-sops")

    if include_neural:
        _outputs = Path(__file__).parent / "outputs"
        if (_outputs / "vsn_lstm_weights_cp5.pt").exists() or (_outputs / "vsn_lstm_weights.pt").exists():
            methods.append("C-vsn-lstm")
        else:
            print("  vsn_lstm weights not found — skipping C-VSN+LSTM")
        if (_outputs / "tft_weights_cp5.pt").exists() or (_outputs / "tft_weights.pt").exists():
            methods.append("D-tft")
        else:
            print("  tft weights not found — skipping D-TFT")

    print("\n" + "=" * 70)
    print("OOS RESULTS")
    print(f"  Boundary: GLOBAL_CUT={config.BOUNDARY.date()}  "
          f"Embargo end: 2021-10-20")
    print("=" * 70)

    all_results: dict[MethodName, dict] = {}
    all_reports: dict[MethodName, BacktestReport] = {}

    for method in methods:
        report, m = evaluate_method(
            method,
            oos_events=oos_events,
            vol_panel=vol_panel,
            returns_panel=returns_panel,
            oof_df=oof_df,
            ohlcv=ohlcv,
            signals=signals,
            verbose=verbose,
        )
        all_results[method] = m
        all_reports[method] = report

        ret_name = method.replace("-", "_").replace("+", "_")
        ret_path = EVAL_DIR / f"net_returns_{ret_name}.csv"
        report.net_returns.to_csv(ret_path, header=True)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    tbl = comparison_table(all_results, METHOD_LABELS, DISPLAY_COLS)
    print(tbl.to_string())

    csv_path = EVAL_DIR / "eval_summary.csv"
    tbl.to_csv(csv_path)
    print(f"\nSaved → {csv_path.relative_to(_REPO)}")

    md_lines = [
        "# Strategy evaluation — OOS results",
        "",
        f"OOS period: {oos_events['t_start'].min().date()} → "
        f"{oos_events['t_start'].max().date()}",
        f"Vol: {config.VOL_METHOD}(span={config.EWMA_SPAN})  "
        f"σ_tgt={config.SIGMA_TGT:.0%}  max_lev={config.MAX_LEVERAGE}",
        "",
        "## Lecture conventions (StrategyWeights slides 38–43)",
        "- Returns: simple  r_t = (P_t − P_{t-1}) / P_{t-1}",
        "- EWMA vol: λ=2/(span+1), exact lecture recurrence, ×√252, floor=2%",
        "- Weight: w_t,k = ŷ_t,k × σ_tgt / σ̂_t,k",
        "- Lag: w_t earns r_{t+1} (position at close of t, return from t→t+1)",
        "- Aggregate: R^port = (1/K) Σ_k w_t,k r_{t+1,k}  (K=11, flat=cash)",
        "- Costs: 2bps half-spread + 10bps×|Δw| Grinold-Kahn",
        "",
        "## Results",
        "",
        _to_markdown(tbl),
    ]

    md_path = EVAL_DIR / "eval_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved → {md_path.relative_to(_REPO)}")

    return all_results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate strategy methods A, B-aon, B-sops, C-VSN+LSTM, D-TFT"
    )
    ap.add_argument("--no-sops",   action="store_true", help="Skip B-sops")
    ap.add_argument("--no-neural", action="store_true", help="Skip C/D neural models")
    ap.add_argument("--quiet",     action="store_true")
    args = ap.parse_args(argv)
    run(
        include_sops=not args.no_sops,
        include_neural=not args.no_neural,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
