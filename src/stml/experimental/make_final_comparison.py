"""Phase H runner — primary-blind baseline + consolidated comparison + caveats.

Produces three artifacts:

  1. ``outputs/strategy_weights_primary_blind.csv`` -- the baseline that takes
     every primary signal at full vol-targeted size (no meta filter, no
     calibration, no sizing curve). The honest comparator for "did the
     meta-model add value?".
  2. ``results/submission/strategy_variant_comparison.csv`` -- the
     existing comparison file, augmented with a `primary_blind` row.
  3. ``outputs/coverage_caveat.csv`` -- explicit flags for instruments whose
     meta-model produced zero positions on the sealed test, plus thin-OOS
     warnings (ho1s with 2 events).

Run after make_deliverables + make_nn_strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stml.experimental.backtest import (
    BacktestReport, performance_metrics, strategy_returns,
)
from stml.experimental.config import INSTRUMENTS, PipelineConfig
from stml.experimental.cost_model import (
    annualised_turnover, transaction_costs,
)
from stml.experimental.data_loader import load_panel
from stml.experimental.emit import emit_weights, panel_to_long
from stml.experimental.sizing import TARGET_VOL, TRADING_DAYS
from stml.experimental.volatility import ewma_lecturer


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _returns_panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        sub = ohlcv.loc[ohlcv["instrument"] == inst, ["date", "close"]].sort_values("date")
        sub = sub.set_index("date")
        ret = np.log(sub["close"] / sub["close"].shift(1))
        rows.append(ret.rename(inst))
    return pd.concat(rows, axis=1).sort_index()


def _primary_blind_weights(
    events_test: pd.DataFrame,
    *,
    ohlcv: pd.DataFrame,
    target_vol: float = TARGET_VOL,
    ewma_span: int = 60,
    max_leverage: float = 10.0,
) -> pd.Series:
    """Per-event weight for the primary-blind baseline.

    ``w = side * sigma_target / sigma_t_annualised`` for every event in the
    test partition, where sigma_t is the causal EWMA daily std at the entry
    bar, annualised by sqrt(252). Clipped to ``max_leverage``. No meta filter.
    """
    weights = np.zeros(len(events_test), dtype=float)
    sigma_by_inst: dict[str, pd.Series] = {}
    for inst in events_test["instrument"].unique():
        sub = ohlcv.loc[ohlcv["instrument"] == inst, ["date", "close"]].sort_values("date")
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.set_index("date")
        log_ret = np.log(sub["close"] / sub["close"].shift(1)).dropna()
        sigma = ewma_lecturer(log_ret, span=ewma_span)
        sigma.index = pd.to_datetime(sigma.index)
        sigma_by_inst[inst] = sigma

    for i, row in events_test.reset_index(drop=True).iterrows():
        inst = row["instrument"]
        side = int(row["side"])
        if side == 0:
            continue
        sigma = sigma_by_inst.get(inst)
        if sigma is None:
            continue
        sigma_d = sigma.get(pd.Timestamp(row["t_start"]), float("nan"))
        if not np.isfinite(sigma_d) or sigma_d <= 0:
            continue
        sigma_ann = float(sigma_d) * np.sqrt(TRADING_DAYS)
        lev = min(target_vol / sigma_ann, max_leverage)
        weights[i] = np.sign(side) * lev

    return pd.Series(weights, name="weight")


def _build_position_panel_from_events(
    events: pd.DataFrame, weights: pd.Series, business_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Stamp per-event weights onto a (date x instrument) panel."""
    events = events.copy().reset_index(drop=True)
    events["t_start"] = pd.to_datetime(events["t_start"])
    events["t_end"] = pd.to_datetime(events["t_end"])
    instruments = sorted(events["instrument"].unique())
    panel = pd.DataFrame(0.0, index=business_calendar, columns=instruments)
    for i, row in events.iterrows():
        w = float(weights.iloc[i])
        if w == 0.0:
            continue
        dates = business_calendar[
            (business_calendar >= row["t_start"]) & (business_calendar < row["t_end"])
        ]
        if len(dates) == 0:
            continue
        col = row["instrument"]
        panel.loc[dates, col] = panel.loc[dates, col] + w
    return panel


def run(
    *, target_vol: float = TARGET_VOL, max_leverage: float = 10.0,
    verbose: bool = True,
) -> dict:
    root = _find_repo_root()
    cfg = PipelineConfig()
    events_path = root / "data" / "events.parquet"
    events_all = pd.read_parquet(events_path)
    events_test = events_all.loc[events_all["partition"] == "test"].copy()
    events_test["t_start"] = pd.to_datetime(events_test["t_start"])
    events_test["t_end"] = pd.to_datetime(events_test["t_end"])

    ohlcv, _signals = load_panel()
    returns_panel = _returns_panel(ohlcv)

    if verbose:
        print(f"Primary-blind baseline: {len(events_test)} sealed-test events")

    weights_per_event = _primary_blind_weights(
        events_test, ohlcv=ohlcv, target_vol=target_vol,
        max_leverage=max_leverage,
    )

    # Backtest.
    cal_min = events_test["t_start"].min()
    cal_max = events_test["t_end"].max()
    business_calendar = pd.bdate_range(cal_min, cal_max)
    panel = _build_position_panel_from_events(events_test, weights_per_event, business_calendar)
    panel = panel.reindex(columns=INSTRUMENTS).fillna(0.0)

    gross = strategy_returns(panel, returns_panel)
    costs = transaction_costs(
        panel,
        half_spread_bps=cfg.half_spread_bps,
        impact_bps=cfg.impact_bps,
        impact_exponent=cfg.impact_exponent,
    )
    costs_aligned = costs.reindex(gross.index).fillna(0.0)
    net = (gross - costs_aligned).rename("net_ret")
    metrics = performance_metrics(net)
    metrics["turnover_per_year"] = annualised_turnover(panel)
    metrics["gross_ann_return"] = float(gross.mean() * 252.0)
    metrics["net_ann_return"] = float(net.mean() * 252.0)
    metrics["total_cost_bps"] = float(costs.sum() * 10_000)

    if verbose:
        print(f"  ann ret (net): {metrics['ann_return']:+.4f}")
        print(f"  ann vol:       {metrics['ann_vol']:.4f}")
        print(f"  Sharpe:        {metrics['sharpe']:+.4f}")
        print(f"  Sortino:       {metrics['sortino']:+.4f}")
        print(f"  Max DD:        {metrics['max_dd']:.4f}")
        print(f"  Turnover/yr:   {metrics['turnover_per_year']:.2f}")

    # Persist primary-blind weights as a deliverable CSV.
    weights_long = panel_to_long(panel, value_col="weight")
    outputs_dir = root / "outputs"
    emit_weights(weights_long, outputs_dir / "strategy_weights_primary_blind.csv")

    # Append to strategy_variant_comparison.csv.
    cmp_path = root / "results" / "submission" / "strategy_variant_comparison.csv"
    cmp = pd.read_csv(cmp_path)
    # Drop any stale primary_blind row.
    cmp = cmp.loc[cmp["variant"] != "primary_blind"].copy()
    new_row = {
        "variant": "primary_blind",
        "val_sharpe": float("nan"),
        "test_sharpe": metrics["sharpe"],
        "test_ann_ret_net": metrics["ann_return"],
        "test_ann_vol": metrics["ann_vol"],
        "test_sortino": metrics["sortino"],
        "test_max_dd": metrics["max_dd"],
        "test_turnover": metrics["turnover_per_year"],
        "test_n": metrics["n"],
        "hidden_dim": float("nan"),
        "n_epochs": 0,
    }
    cmp = pd.concat([cmp, pd.DataFrame([new_row])], ignore_index=True)
    cmp = cmp.sort_values("test_sharpe", ascending=False)
    cmp.to_csv(cmp_path, index=False, float_format="%.6f")
    if verbose:
        print(f"\nUpdated {cmp_path.relative_to(root)} with primary_blind row.")

    # Update coverage_caveat.csv with the structural flags surfaced by the
    # SOPS deliverable (instruments where Platt+p* zeroed every position).
    coverage_rows = []
    # Recompute counts.
    sw_sops = pd.read_csv(root / "outputs" / "strategy_weights_sops.csv")
    sw_sops["date"] = pd.to_datetime(sw_sops["date"])
    for inst in INSTRUMENTS:
        n_oos = int((events_test["instrument"] == inst).sum())
        sub_w = sw_sops.loc[sw_sops["instrument"] == inst, "weight"]
        n_w_nz = int((sub_w.abs() > 1e-9).sum()) if len(sub_w) else 0
        coverage_rows.append({
            "instrument": inst,
            "n_oos_events": n_oos,
            "thin_oos": bool(n_oos < 5),
            "no_meta_positions": bool(n_oos > 0 and n_w_nz == 0),
            "low_coherence_vs_raw": inst == "ng1s",
            "noisy_coherence_vs_raw": inst in ("ho1s", "rb1s"),
            "documented_failure_in_jay_pdf": inst == "ho1s",
            "self_fulfilling_h1_label": inst in (
                "cl1s", "fesx1s", "ho1s", "ng1s", "nq1s", "pl1s"
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(outputs_dir / "coverage_caveat.csv", index=False)
    if verbose:
        print(f"Wrote {(outputs_dir / 'coverage_caveat.csv').relative_to(root)}")

    return {"metrics": metrics, "comparison": cmp, "coverage": coverage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase H — final comparison runner")
    ap.add_argument("--target-vol", type=float, default=TARGET_VOL)
    ap.add_argument("--max-leverage", type=float, default=10.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out = run(
        target_vol=args.target_vol, max_leverage=args.max_leverage,
        verbose=not args.quiet,
    )
    return 0 if np.isfinite(out["metrics"]["sharpe"]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
