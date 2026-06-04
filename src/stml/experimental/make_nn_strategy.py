"""NN portfolio strategy runner -- Madmoun Optional Session 3 §4.

Trains all three NN backbones (linear / lstm / vlstm) on the modelling slice
using the lecturer's Sharpe-loss recipe (slide 42), selects the winner by
validation Sharpe (chronologically-last 20% of modelling), refits on
train + val combined, then applies to the SEALED test slice exactly once.

Plus the existing SOPS sizing-rules variant from ``make_deliverables`` is
re-run on the same window so the four variants can be compared head-to-head.

Outputs:

    outputs/strategy_weights_nn_linear.csv
    outputs/strategy_weights_nn_lstm.csv
    outputs/strategy_weights_nn_vlstm.csv
    outputs/strategy_weights_sops.csv                    (copy of SOPS deliverable)
    results/submission/nn_training_history_<variant>.csv
    results/submission/strategy_variant_comparison.csv
    results/submission/strategy_winner.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stml.experimental.backtest import (
    BacktestReport, performance_metrics, strategy_returns,
)
from stml.experimental.config import INSTRUMENTS, PipelineConfig
from stml.experimental.cost_model import (
    HALF_SPREAD_BPS, IMPACT_BPS, IMPACT_EXPONENT,
    annualised_turnover, transaction_costs,
)
from stml.experimental.data_loader import load_panel
from stml.experimental.emit import emit_weights, panel_to_long
from stml.experimental.nn_dataset import build_portfolio_panel
from stml.experimental.nn_portfolio import (
    BackboneName, TrainConfig, build_portfolio_model, chronological_split,
    predict_weights, train_portfolio_model,
)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not locate repo root from {here}")


def _returns_panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        sub = ohlcv.loc[ohlcv["instrument"] == inst, ["date", "close"]].sort_values("date")
        sub = sub.set_index("date")
        ret = np.log(sub["close"] / sub["close"].shift(1))
        rows.append(ret.rename(inst))
    return pd.concat(rows, axis=1).sort_index()


# ---------------------------------------------------------------------------
# Variant runner.
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    name: str
    val_sharpe: float
    test_metrics: dict
    weights: pd.DataFrame  # date x instrument
    extras: dict


def _filter_weights_to_event_days(
    weights_panel: pd.DataFrame, signals: pd.DataFrame, instruments: list[str],
) -> pd.DataFrame:
    """Zero the weight on days where the primary signal is 0 (no event)."""
    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals.set_index("date").reindex(weights_panel.index)
    out = weights_panel.copy()
    for inst in instruments:
        if inst not in signals.columns:
            continue
        zero = signals[inst].fillna(0) == 0
        out.loc[zero, inst] = 0.0
    return out


def _backtest_from_weights(
    weights_panel: pd.DataFrame, returns_panel: pd.DataFrame,
    *, cost_cfg: PipelineConfig,
) -> BacktestReport:
    """Run the strategy_returns + costs + metrics machinery on a weights panel."""
    weights = weights_panel.copy()
    rp = returns_panel.reindex(weights.index).reindex(columns=weights.columns)
    gross = strategy_returns(weights, rp)
    costs = transaction_costs(
        weights,
        half_spread_bps=cost_cfg.half_spread_bps,
        impact_bps=cost_cfg.impact_bps,
        impact_exponent=cost_cfg.impact_exponent,
    )
    costs_aligned = costs.reindex(gross.index).fillna(0.0)
    net = (gross - costs_aligned).rename("net_ret")
    metrics = performance_metrics(net)
    metrics["turnover_per_year"] = annualised_turnover(weights)
    metrics["total_cost_bps"] = float(costs.sum() * 10_000)
    metrics["gross_ann_return"] = float(gross.mean() * 252.0)
    metrics["net_ann_return"] = float(net.mean() * 252.0)
    return BacktestReport(
        weights=weights, gross_returns=gross, net_returns=net,
        daily_costs=costs_aligned, metrics=metrics,
    )


def _run_nn_variant(
    backbone_name: BackboneName,
    *,
    train_panel,
    val_panel,
    full_train_panel,  # train+val combined, for refit
    test_panel,
    cfg: PipelineConfig,
    train_cfg: TrainConfig,
    hidden_dim: int,
    signals: pd.DataFrame,
    returns_panel: pd.DataFrame,
    instruments: list[str],
    results_dir: Path,
    verbose: bool,
    n_seeds: int = 1,
) -> VariantResult:
    """Train + validate + refit + test for one NN backbone.

    If ``n_seeds > 1``, train ``n_seeds`` independently-seeded models, refit
    each on train+val, and average their inference weights -- the classic
    small-data NN variance reduction (Lakshminarayanan et al. 2017).
    """
    if verbose:
        print(f"\n[variant] nn_{backbone_name} (hidden={hidden_dim}, "
              f"seeds={n_seeds})")

    val_sharpes = []
    weights_runs = []
    epochs_runs = []
    history_first = None

    for seed_idx in range(n_seeds):
        seed = cfg.seed + 17 * seed_idx

        # Train on train slice; early-stop on val Sharpe.
        model = build_portfolio_model(
            backbone_name,
            lookback=train_panel.L, n_channels=train_panel.C,
            hidden_dim=hidden_dim, target_vol=cfg.target_vol,
            max_leverage=cfg.max_leverage, seed=seed,
        )
        tcfg = TrainConfig(
            lr=train_cfg.lr, epochs=train_cfg.epochs, patience=train_cfg.patience,
            grad_clip=train_cfg.grad_clip, weight_decay=train_cfg.weight_decay,
            seed=seed, device=train_cfg.device,
            verbose=(verbose and seed_idx == 0),
        )
        train_out = train_portfolio_model(model, train_panel, val_panel, cfg=tcfg)
        val_sharpes.append(train_out["best_val_sharpe"])
        epochs_runs.append(train_out["n_epochs_run"])
        if seed_idx == 0:
            history_first = train_out["history"]
        if verbose:
            print(f"  seed {seed}: val Sharpe {train_out['best_val_sharpe']:+.3f}  "
                  f"({train_out['n_epochs_run']} epochs)")

        # Refit on train+val combined.
        model_final = build_portfolio_model(
            backbone_name,
            lookback=full_train_panel.L, n_channels=full_train_panel.C,
            hidden_dim=hidden_dim, target_vol=cfg.target_vol,
            max_leverage=cfg.max_leverage, seed=seed,
        )
        refit_cfg = TrainConfig(
            lr=train_cfg.lr, epochs=train_out["n_epochs_run"],
            grad_clip=train_cfg.grad_clip, patience=train_cfg.epochs,
            weight_decay=train_cfg.weight_decay, seed=seed,
            device=train_cfg.device, verbose=False,
        )
        train_portfolio_model(model_final, full_train_panel, None, cfg=refit_cfg)
        weights_runs.append(predict_weights(model_final, test_panel))

    # Persist first-seed history for the audit log.
    if history_first is not None:
        history_first.to_csv(
            results_dir / f"nn_training_history_{backbone_name}.csv",
            index=False, float_format="%.6f",
        )

    # Average weights across seeds.
    if n_seeds > 1:
        avg_w = sum(w.fillna(0.0) for w in weights_runs) / len(weights_runs)
        weights = avg_w
    else:
        weights = weights_runs[0]

    weights_eventday = _filter_weights_to_event_days(weights, signals, instruments)
    report = _backtest_from_weights(weights_eventday, returns_panel, cost_cfg=cfg)

    if verbose:
        m = report.metrics
        print(f"  TEST Sharpe={m['sharpe']:+.3f}  "
              f"ann_vol={m['ann_vol']:.4f}  "
              f"ann_ret(net)={m['ann_return']:+.4f}  "
              f"max_dd={m['max_dd']:.4f}  "
              f"turnover={m['turnover_per_year']:.1f}")

    val_sharpe_med = float(np.median(val_sharpes))
    return VariantResult(
        name=f"nn_{backbone_name}",
        val_sharpe=val_sharpe_med,
        test_metrics=report.metrics,
        weights=weights_eventday,
        extras={
            "hidden_dim": hidden_dim,
            "n_epochs": int(np.median(epochs_runs)),
            "n_seeds": n_seeds,
            "val_sharpes_all": [float(s) for s in val_sharpes],
        },
    )


def _load_sops_weights(root: Path, instruments: list[str]) -> pd.DataFrame:
    """Read back the SOPS deliverable weights produced by make_deliverables."""
    sw = pd.read_csv(root / "outputs" / "strategy_weights.csv")
    sw["date"] = pd.to_datetime(sw["date"])
    panel = sw.pivot_table(index="date", columns="instrument",
                            values="weight", aggfunc="sum").fillna(0.0)
    for inst in instruments:
        if inst not in panel.columns:
            panel[inst] = 0.0
    return panel[instruments].sort_index()


def _run_sops_variant(
    *, returns_panel: pd.DataFrame, instruments: list[str],
    cfg: PipelineConfig, root: Path, verbose: bool,
) -> VariantResult:
    """Re-backtest the SOPS deliverable on the same window for comparison."""
    weights = _load_sops_weights(root, instruments)
    report = _backtest_from_weights(weights, returns_panel, cost_cfg=cfg)
    if verbose:
        m = report.metrics
        print(f"\n[variant] sops (reference)")
        print(f"  TEST Sharpe={m['sharpe']:+.3f}  ann_vol={m['ann_vol']:.4f}  "
              f"ann_ret(net)={m['ann_return']:+.4f}  max_dd={m['max_dd']:.4f}")
    return VariantResult(
        name="sops", val_sharpe=float("nan"),
        test_metrics=report.metrics, weights=weights,
        extras={"hidden_dim": None, "n_epochs": 0},
    )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def run(
    cfg: PipelineConfig | None = None,
    *,
    lookback: int = 60,
    hidden_dim: int = 32,
    backbones: tuple[BackboneName, ...] = ("linear", "lstm", "vlstm", "tft"),
    epochs: int = 120,
    patience: int = 20,
    lr: float = 1e-3,
    n_seeds: int = 1,
    verbose: bool = True,
) -> dict:
    cfg = cfg or PipelineConfig()
    root = _find_repo_root()
    results_dir = root / "results" / "submission"
    outputs_dir = root / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # --- Build panels.
    if verbose:
        print("Loading OOF calibrated predictions...")
    oof_path = results_dir / "oof_calibrated_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"{oof_path} not found. Run `python -m stml.experimental.make_deliverables` first."
        )
    p_hat_oof = pd.read_csv(oof_path)

    # Combine OOF (training period) + sealed-test predictions so the panel
    # has p̂ for every event day; otherwise p̂ stays at 0.5 on the test slice.
    test_preds = pd.read_csv(outputs_dir / "metamodel_predictions.csv")
    test_preds = test_preds.rename(columns={"prediction": "calibrated_proba"})
    full_p_hat = pd.concat([p_hat_oof, test_preds], ignore_index=True).drop_duplicates(
        subset=["date", "instrument"], keep="last",
    )

    if verbose:
        print(f"Building portfolio panel (L={lookback})...")
    # Train + val + test window. Use a wider start so EWMA σ̂ warms up.
    # End boundary is data-driven from the events parquet's max t_end so the
    # marker's H2-2022 re-run extends automatically.
    events_path = root / "data" / "events.parquet"
    events_df_for_end = pd.read_parquet(events_path)
    panel_end = pd.to_datetime(events_df_for_end["t_end"]).max()
    panel_full = build_portfolio_panel(
        p_hat_oof=full_p_hat, lookback=lookback,
        start="2020-01-01",  # signals start 2020-01-01
        end=str(panel_end.date()),
        include_inst_id=True,
    )
    instruments = panel_full.instruments

    # Split by partition dates inferred from the events parquet (the CSV).
    events_df = pd.read_parquet(events_path)
    events_df["t_signal"] = pd.to_datetime(events_df["t_signal"])
    train_max_date = events_df.loc[events_df["partition"] == "train", "t_signal"].max()
    val_max_date = events_df.loc[events_df["partition"] == "val", "t_signal"].max()
    test_min_date = events_df.loc[events_df["partition"] == "test", "t_signal"].min()
    if verbose:
        print(f"  partition boundaries: train≤{train_max_date.date()}  "
              f"val∈({train_max_date.date()},{val_max_date.date()}]  "
              f"test≥{test_min_date.date()}")

    mask_modelling = panel_full.dates <= val_max_date
    mask_test = panel_full.dates >= test_min_date
    mask_train_only = panel_full.dates <= train_max_date
    mask_val_only = (panel_full.dates > train_max_date) & (panel_full.dates <= val_max_date)

    def _slice(p, mask):
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise RuntimeError("empty slice")
        from stml.experimental.nn_portfolio import PortfolioPanel
        return PortfolioPanel(
            dates=p.dates[idx], instruments=p.instruments,
            X=p.X[idx], sigma_daily=p.sigma_daily[idx],
            r_next=p.r_next[idx], mask=p.mask[idx],
            feature_names=p.feature_names,
        )

    full_train_panel = _slice(panel_full, mask_modelling)
    test_panel = _slice(panel_full, mask_test)
    train_panel = _slice(panel_full, mask_train_only)
    val_panel = _slice(panel_full, mask_val_only)

    if verbose:
        print(f"  train: T={train_panel.T}  val: T={val_panel.T}  test: T={test_panel.T}")
        print(f"  K instruments = {len(instruments)}, channels = {panel_full.C}")

    # Load OHLCV for backtest returns.
    ohlcv, signals = load_panel()
    signals["date"] = pd.to_datetime(signals["date"])
    returns_panel = _returns_panel(ohlcv)

    # --- Run NN variants.
    train_cfg = TrainConfig(
        lr=lr, epochs=epochs, patience=patience,
        grad_clip=1.0, weight_decay=1e-5, seed=cfg.seed,
        device="cpu", verbose=verbose,
    )

    variant_results: list[VariantResult] = []
    for bb in backbones:
        try:
            res = _run_nn_variant(
                bb, train_panel=train_panel, val_panel=val_panel,
                full_train_panel=full_train_panel, test_panel=test_panel,
                cfg=cfg, train_cfg=train_cfg, hidden_dim=hidden_dim,
                signals=signals, returns_panel=returns_panel,
                instruments=instruments, results_dir=results_dir,
                verbose=verbose, n_seeds=n_seeds,
            )
        except Exception as exc:
            if verbose:
                print(f"  nn_{bb} failed: {exc}")
            continue
        variant_results.append(res)

    # --- SOPS reference.
    try:
        sops_res = _run_sops_variant(
            returns_panel=returns_panel, instruments=instruments,
            cfg=cfg, root=root, verbose=verbose,
        )
        variant_results.append(sops_res)
    except Exception as exc:
        if verbose:
            print(f"  sops baseline failed: {exc}")

    # --- Persist per-variant weights.
    for res in variant_results:
        long = panel_to_long(res.weights, value_col="weight")
        long.to_csv(
            outputs_dir / f"strategy_weights_{res.name}.csv",
            index=False, float_format="%.10f",
        )

    # --- Comparison table.
    rows = []
    for res in variant_results:
        m = res.test_metrics
        rows.append({
            "variant": res.name,
            "val_sharpe": res.val_sharpe,
            "test_sharpe": m.get("sharpe", float("nan")),
            "test_ann_ret_net": m.get("ann_return", float("nan")),
            "test_ann_vol": m.get("ann_vol", float("nan")),
            "test_sortino": m.get("sortino", float("nan")),
            "test_max_dd": m.get("max_dd", float("nan")),
            "test_turnover": m.get("turnover_per_year", float("nan")),
            "test_n": m.get("n", 0),
            "hidden_dim": res.extras.get("hidden_dim"),
            "n_epochs": res.extras.get("n_epochs"),
        })
    cmp_df = pd.DataFrame(rows).sort_values("test_sharpe", ascending=False)
    cmp_path = results_dir / "strategy_variant_comparison.csv"
    cmp_df.to_csv(cmp_path, index=False, float_format="%.6f")
    if verbose:
        print("\n=== Strategy variant comparison (sealed-test metrics) ===")
        print(cmp_df.to_string(index=False))

    # --- Winner.
    # NN variants are picked by VAL Sharpe (the lecturer's stop criterion).
    nn_only = cmp_df.loc[cmp_df["variant"].str.startswith("nn_")]
    if not nn_only.empty and nn_only["val_sharpe"].notna().any():
        winner_row = nn_only.sort_values("val_sharpe", ascending=False).iloc[0]
        winner_name = str(winner_row["variant"])
    else:
        winner_row = cmp_df.iloc[0]
        winner_name = str(winner_row["variant"])
    winner_payload = {
        "winner": winner_name,
        "selection_criterion": "max validation Sharpe (slide 51)",
        "val_sharpe": float(winner_row["val_sharpe"]),
        "test_sharpe": float(winner_row["test_sharpe"]),
        "test_ann_vol": float(winner_row["test_ann_vol"]),
        "test_ann_ret_net": float(winner_row["test_ann_ret_net"]),
        "test_sortino": float(winner_row["test_sortino"]),
        "test_max_dd": float(winner_row["test_max_dd"]),
        "hidden_dim": winner_row["hidden_dim"]
            if pd.notna(winner_row["hidden_dim"]) else None,
        "n_epochs": int(winner_row["n_epochs"]) if pd.notna(winner_row["n_epochs"]) else None,
        "lookback": lookback,
    }
    with open(results_dir / "strategy_winner.json", "w") as f:
        json.dump(winner_payload, f, indent=2, default=str)
    if verbose:
        print(f"\n[winner] {winner_name}  -- val Sharpe {winner_row['val_sharpe']:+.3f}  "
              f"-> test Sharpe {winner_row['test_sharpe']:+.3f}")

    return {
        "variant_results": variant_results,
        "comparison": cmp_df,
        "winner": winner_payload,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NN portfolio strategy runner")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbones", nargs="+",
                     default=["linear", "lstm", "vlstm", "tft"])
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds to ensemble")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out = run(
        lookback=args.lookback, hidden_dim=args.hidden,
        backbones=tuple(args.backbones), epochs=args.epochs,
        patience=args.patience, lr=args.lr, n_seeds=args.seeds,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
