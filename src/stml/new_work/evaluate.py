"""OOS evaluation — Checkpoint 5: full four-way comparison + vol comparison.

Methods compared:
    A          — Benchmark (primary-only, full conviction)
    B-mc       — model_confidence: ŷ = side · p̂  (p̂ > 0.5)
    B-aon      — all_or_nothing:  ŷ = side · 1(p̂ > 0.5)
    B-ncdf     — NCDF:            ŷ = side · Φ(z)
    B-sops     — SOPS sigmoid fitted on OOF Sharpe
    C-vsn-lstm — VSN+LSTM neural model (loads saved weights)
    D-tft      — Temporal Fusion Transformer (loads saved weights)

Vol comparison (Section 2):
    Methods A and B-aon re-run with yang_zhang / ewma_close / garch / gjr estimators.

Outputs:
    results/strategy_eval/
        eval_summary.csv / .md     — full seven-column comparison table
        vol_comparison.csv / .md   — A and B-aon × four vol estimators
        net_returns_<method>.csv   — per-method daily net returns

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

METHODS_BASE: list[MethodName] = ["A", "B-mc", "B-aon", "B-ncdf"]

METHOD_LABELS: dict[str, str] = {
    "A":          "A Benchmark",
    "B-mc":       "B model_confidence",
    "B-aon":      "B all_or_nothing",
    "B-ncdf":     "B ncdf",
    "B-sops":     "B SOPS",
    "C-vsn-lstm": "C VSN+LSTM",
    "D-tft":      "D TFT",
}

VOL_METHOD_LABELS = {
    "yang_zhang": "Yang-Zhang (default)",
    "ewma_close": "EWMA(60)",
    "garch":      "GARCH(1,1)",
    "gjr":        "GJR-GARCH (equity only)",
}


# ---------------------------------------------------------------------------
# Breakeven cost
# ---------------------------------------------------------------------------


def breakeven_halfspread(
    net_returns: pd.Series,
    weights: pd.DataFrame,
) -> float:
    """Half-spread (bps) at which net Sharpe hits zero.

    Approximation: gross_mean_daily / (annualised_turnover / 252) / 10000.
    """
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
    if verbose:
        print(f"\n  Sizing conviction for {METHOD_LABELS.get(method, method)} …")

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
# Comparison tables
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

VOL_CMP_COLS = [
    ("ann_return", "Ann return",  "{:+.2%}"),
    ("ann_vol",    "Ann vol",     "{:.2%}"),
    ("sharpe",     "Sharpe",      "{:.3f}"),
    ("t_stat",     "t-stat",      "{:.2f}"),
    ("max_dd",     "Max DD",      "{:.2%}"),
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
# Vol comparison
# ---------------------------------------------------------------------------


def run_vol_comparison(
    *,
    oos_events: pd.DataFrame,
    ohlcv: pd.DataFrame,
    returns_panel: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, dict[str, dict]]:
    """Re-run methods A and B-aon with four vol estimators.

    Returns nested dict: results[method][vol_method] = metrics_dict.
    """
    vol_methods = ["yang_zhang", "ewma_close", "garch", "gjr"]
    cmp_methods: list[MethodName] = ["A", "B-aon"]

    results: dict[str, dict[str, dict]] = {m: {} for m in cmp_methods}

    if verbose:
        print("\n" + "=" * 70)
        print("VOL COMPARISON — methods A and B-aon across four estimators")
        print("=" * 70)

    for vm in vol_methods:
        if verbose:
            print(f"\n  Building {VOL_METHOD_LABELS[vm]} vol panel …")

        vp = build_vol_panel(
            ohlcv,
            method=vm,
            window=config.LOOKBACK_L,
            span=config.EWMA_SPAN,
        )

        for meth in cmp_methods:
            if verbose:
                print(f"    {METHOD_LABELS[meth]} × {VOL_METHOD_LABELS[vm]} …")
            _, m = evaluate_method(
                meth,
                oos_events=oos_events,
                vol_panel=vp,
                returns_panel=returns_panel,
                verbose=False,
            )
            results[meth][vm] = m

    return results


def _vol_cmp_table(
    vol_results: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """Build a vol-comparison table: rows = metrics, columns = method×vol combos."""
    label_map = {}
    ordered: dict[str, dict] = {}
    for meth in ["A", "B-aon"]:
        for vm in ["yang_zhang", "ewma_close", "garch", "gjr"]:
            col_key = f"{meth}_{vm}"
            col_label = f"{METHOD_LABELS[meth]} / {VOL_METHOD_LABELS[vm]}"
            label_map[col_key] = col_label
            ordered[col_key] = vol_results[meth][vm]

    return comparison_table(ordered, label_map, VOL_CMP_COLS)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(
    include_sops: bool = True,
    include_neural: bool = True,
    verbose: bool = True,
) -> dict[MethodName, dict]:
    """Run the full four-way evaluation and vol comparison."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Strategy evaluation — Checkpoint 5: four-way + vol comparison")
    print(f"Vol method: {config.VOL_METHOD}  σ_tgt={config.SIGMA_TGT:.0%}  "
          f"max_lev={config.MAX_LEVERAGE}")
    print("=" * 70)

    # Shared inputs — load once.
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

    # Load signals once for neural methods.
    signals = None
    if include_neural:
        from stml.new_work.data import load_signals
        signals = load_signals()

    # OOF probabilities — required for B-sops.
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
        from pathlib import Path as _Path
        _outputs = _Path(__file__).parent / "outputs"
        if (_outputs / "vsn_lstm_weights.pt").exists():
            methods.append("C-vsn-lstm")
        else:
            print("  vsn_lstm_weights.pt not found — skipping C-VSN+LSTM")
        if (_outputs / "tft_weights.pt").exists():
            methods.append("D-tft")
        else:
            print("  tft_weights.pt not found — skipping D-TFT")

    # ── Section 1: Full method comparison ────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 1: Full method comparison (Yang-Zhang vol)")
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
    print("SECTION 1 RESULTS")
    print("=" * 70)
    tbl = comparison_table(all_results, METHOD_LABELS, DISPLAY_COLS)
    print(tbl.to_string())

    csv_path = EVAL_DIR / "eval_summary.csv"
    tbl.to_csv(csv_path)
    print(f"\nSaved → {csv_path.relative_to(_REPO)}")

    md_lines = [
        "# Strategy evaluation — methods A–D",
        "",
        f"OOS period: {oos_events['t_start'].min().date()} → "
        f"{oos_events['t_start'].max().date()}",
        f"Vol method: {config.VOL_METHOD}  "
        f"σ_tgt={config.SIGMA_TGT:.0%}  max_lev={config.MAX_LEVERAGE}",
        "",
        "## Section 1: Method comparison (Yang-Zhang vol)",
        "",
        _to_markdown(tbl),
        "",
        "## Notes",
        "- All Sharpe ratios are annualised (×√252).",
        "- Bootstrap CI: Politis-Romano stationary block bootstrap (n_boot=2000).",
        "- Transaction costs: 2bps half-spread + 10bps×|Δw| Grinold-Kahn impact.",
        "- Vol targeting: Yang-Zhang(20-bar) annualised σ̂, σ_tgt=10%, max_lev=10×.",
        "- C-VSN+LSTM / D-TFT: trained on pre-BOUNDARY OOF data; best checkpoint by val Sharpe.",
    ]

    # ── Section 2: Vol comparison ─────────────────────────────────────────
    print("\n")
    vol_results = run_vol_comparison(
        oos_events=oos_events,
        ohlcv=ohlcv,
        returns_panel=returns_panel,
        verbose=verbose,
    )

    vol_tbl = _vol_cmp_table(vol_results)

    print("\n" + "=" * 70)
    print("SECTION 2 RESULTS — Vol estimator comparison")
    print("=" * 70)
    print(vol_tbl.to_string())

    vol_csv = EVAL_DIR / "vol_comparison.csv"
    vol_tbl.to_csv(vol_csv)
    print(f"\nSaved → {vol_csv.relative_to(_REPO)}")

    md_lines += [
        "",
        "## Section 2: Vol estimator comparison (methods A and B-aon)",
        "",
        _to_markdown(vol_tbl),
        "",
        "### Vol estimators",
        "- **Yang-Zhang**: bias-minimised OHLC estimator, window=20 bars.",
        "- **EWMA(60)**: exponentially weighted close-to-close returns, span=60.",
        "- **GARCH(1,1)**: refitted every 21 bars on up to 2000 bars of history.",
        "- **GJR-GARCH**: asymmetric GARCH for equity instruments (es1s, nq1s, fesx1s); "
        "  GARCH(1,1) for commodity instruments.",
    ]

    md_path = EVAL_DIR / "eval_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved → {md_path.relative_to(_REPO)}")

    return all_results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate strategy methods A–D + vol comparison"
    )
    ap.add_argument("--no-sops",   action="store_true",
                    help="Skip B-sops")
    ap.add_argument("--no-neural", action="store_true",
                    help="Skip C-VSN+LSTM and D-TFT (neural models)")
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
