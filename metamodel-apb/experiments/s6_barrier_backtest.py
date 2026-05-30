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
from alken_metamodel.signal_analysis import henriksson_merton  # noqa: E402


def class_oos_meta(cls: str, cfg: PipelineConfig, ohlcv, signals):
    """(best_model, meta) where meta = OOS [date, instrument, weight, t1] for the class."""
    set_seeds(cfg.seed)
    pooled = build_class_panel(ohlcv, signals, class_members(cls), cfg)
    cols = feature_columns(pooled)
    X = pooled[cols]
    y = pooled["bin"].to_numpy()
    t1 = pooled["t1"]
    dates = pd.DatetimeIndex(pooled["date"])
    mmask = np.asarray(dates <= cfg.modelling_end)
    pmask = np.asarray((dates >= cfg.predict_start) & (dates <= cfg.predict_end))
    sw = balanced_sample_weight(y[mmask], base=pooled["weight"].to_numpy()[mmask])
    best, _ = select_model(X[mmask], y[mmask], t1[mmask], sw, cfg)
    model = _roster_factory(cfg)(seed=cfg.seed)[best]
    model.fit(X[mmask], y[mmask], sample_weight=sw)
    # size on the CALIBRATED p̂ (matches the shipped deliverable; pass-3 S6.11)
    cal, _, _ = fit_oos_calibrator(
        lambda: _roster_factory(cfg)(seed=cfg.seed)[best], X[mmask], y[mmask], t1[mmask], sw, cfg
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
    return best, meta


def _fmt(d: dict, keys) -> str:
    return "  ".join(f"{k}={d[k]:.4f}" for k in keys if k in d and pd.notna(d[k]))


def directional_calls(meta: pd.DataFrame, returns_panel: pd.DataFrame):
    """(realised, predicted) signs for the OOS acted trades: did each call time direction (S5.8)?"""
    real, pred = [], []
    cal = returns_panel.index
    for row in meta.itertuples(index=False):
        if row.weight == 0 or row.instrument not in returns_panel.columns:
            continue
        seg = returns_panel[row.instrument][(cal >= row.date) & (cal < row.t1)]
        if seg.empty:
            continue
        real.append(np.sign((1.0 + seg).prod() - 1.0))
        pred.append(np.sign(row.weight))
    return np.asarray(real, dtype=float), np.asarray(pred, dtype=float)


def utility_line(label: str, net: pd.Series, meta: pd.DataFrame, returns: pd.DataFrame) -> str:
    """Henriksson–Merton timing test + mean-variance certainty-equivalent (S5.8)."""
    real, pred = directional_calls(meta, returns)
    hit, z, p = henriksson_merton(real, pred) if real.size else (float("nan"),) * 3
    cer = certainty_equivalent(net)
    print(f"{label} utility: H-M hit={hit:.3f} z={z:.2f} p={p:.3f}  CER(daily)={cer:.6f}")
    return (f"- utility (S5.8): Henriksson–Merton hit={hit:.3f} (z={z:.2f}, p={p:.3f}), "
            f"certainty-equivalent (daily, γ=5)={cer:.6f}\n")


def run() -> None:
    cfg = PipelineConfig(roster="default", cv_scheme="cpcv", use_macro=True)
    ohlcv, signals = load_clean_data()
    rets = load_returns_panel(kind="simple")
    rets = rets[rets.index >= cfg.predict_start]  # restrict to the backtest window forward

    out = ["# S6 — barrier-exact + cost-aware backtest (real OOS, default path)\n"]
    metas = []
    for cls in CLASSES:
        best, meta = class_oos_meta(cls, cfg, ohlcv, signals)
        metas.append(meta)
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
    (results_dir() / "s6_barrier_backtest.md").write_text("\n".join(out))
    print("wrote results/s6_barrier_backtest.md")


if __name__ == "__main__":
    run()
