"""Tune strategy configuration for the competition track."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from stml.io import load_clean_data
from stml.strategy import backtest, blind_baseline_strategy, write_strategy_weights, StrategyConfig

ohlcv, signals = load_clean_data()
predictions_v4 = pd.read_csv('results/sreeram/predictions_v4.csv')

print('Tuning meta-strategy configurations...')
print(f"{'config':<60s} {'CAGR':>7s} {'Sharpe':>7s} {'Sortino':>8s} {'MDD':>7s} {'AvgPos':>7s}")

configs = [
    # threshold, target_vol, max_per, gross_cap, label
    (0.55, 0.10, 0.30, 2.0, 'baseline (cfg defaults)'),
    (0.50, 0.10, 0.30, 2.0, 'thr=0.50 (lower bar)'),
    (0.50, 0.15, 0.40, 2.5, 'thr=0.50 + larger sizes'),
    (0.40, 0.15, 0.40, 2.5, 'thr=0.40 + larger sizes'),
    (0.50, 0.20, 0.50, 3.0, 'thr=0.50 + aggressive sizing'),
    (0.45, 0.18, 0.45, 3.0, 'thr=0.45 + leveraged'),
    (0.50, 0.10, 0.50, 3.0, 'thr=0.50 + big gross/per-inst'),
    (0.55, 0.10, 0.40, 3.0, 'baseline + bigger caps'),
]
for thr, vol, max_p, gross, label in configs:
    cfg = StrategyConfig(
        threshold=thr, target_vol=vol, target_portfolio_vol=0.15,
        max_per_instrument=max_p, gross_cap=gross, net_cap=2.0,
    )
    r = backtest(predictions_v4, signals, ohlcv, cfg=cfg)
    m = r['metrics']
    print(f"{label:<60s} {m['CAGR']:>7.3f} {m['Sharpe']:>7.2f} {m['Sortino']:>8.2f} {m['MDD']:>+7.3f} {m['n_positions_avg']:>7.2f}")

# Blind
print()
b = blind_baseline_strategy(predictions_v4, signals, ohlcv,
                            cfg=StrategyConfig(target_vol=0.10, target_portfolio_vol=0.15,
                                                max_per_instrument=0.30, gross_cap=2.0, net_cap=2.0))
m = b['metrics']
print(f"{'BLIND BASELINE':<60s} {m['CAGR']:>7.3f} {m['Sharpe']:>7.2f} {m['Sortino']:>8.2f} {m['MDD']:>+7.3f} {m['n_positions_avg']:>7.2f}")

# Pick best by Sharpe
print()
print('Picking best by Sharpe...')
best_sharpe = -1
best_cfg = None
best_label = ''
for thr, vol, max_p, gross, label in configs:
    cfg = StrategyConfig(
        threshold=thr, target_vol=vol, target_portfolio_vol=0.15,
        max_per_instrument=max_p, gross_cap=gross, net_cap=2.0,
    )
    r = backtest(predictions_v4, signals, ohlcv, cfg=cfg)
    if r['metrics']['Sharpe'] > best_sharpe:
        best_sharpe = r['metrics']['Sharpe']
        best_cfg = cfg
        best_label = label
        best_r = r

print(f'Best: "{best_label}"  Sharpe={best_sharpe:.2f}')
write_strategy_weights(best_r['weights'], 'results/sreeram/strategy_weights.csv')
print(f'Wrote results/sreeram/strategy_weights.csv with {len(best_r["weights"])} rows')

# Also show per-instrument exposure stats
print()
print('Per-instrument weight stats (best config):')
ws = best_r['weights']
print((ws.abs().mean()).round(3).to_string())
