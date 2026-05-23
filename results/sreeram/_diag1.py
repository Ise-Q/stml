"""Diagnostic battery — runs as a script. Output captured separately."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from stml.io import load_clean_data
from stml.labeling import extract_signal_events, get_meta_labels, get_uniqueness_weights
from stml.features import compute_features
from stml.regimes import compute_regime_features
from stml.cv import split_by_boundary
from sklearn.metrics import roc_auc_score
from scipy import stats

boundary = pd.Timestamp('2022-01-01'); embargo = pd.Timedelta(days=10)
ohlcv, signals = load_clean_data()
labels = get_meta_labels(ohlcv, signals, h=10, pt_mult=1.0, sl_mult=1.0, verbose=False)
events_all = extract_signal_events(signals).reset_index(drop=True)
feats = compute_features(ohlcv, events_all, signals, include_groups=('G1','G2','G3','G4','G5','G7'))
regs = compute_regime_features(ohlcv, events_all, boundary=boundary)
X_all = feats.join(regs, how='left').fillna(0.0)
key=['t','instrument']
events_all['label'] = labels.set_index(key)['label'].reindex(events_all.set_index(key).index).reset_index(drop=True).values
events_all['t1_orig'] = labels.set_index(key)['t1'].reindex(events_all.set_index(key).index).reset_index(drop=True).values
mask = ~events_all['label'].isna()
events_lab = events_all.loc[mask].reset_index(drop=True)
X_lab = X_all.loc[mask].reset_index(drop=True)
y_lab = events_lab['label'].astype(int)
t_lab = events_lab['t']

tr_pos, _ = split_by_boundary(t_lab, boundary, embargo_td=embargo)
predict_mask = (t_lab.values >= boundary) & (t_lab.values < pd.Timestamp('2022-07-01'))
oos_pos = np.where(predict_mask)[0]
X_tr, y_tr = X_lab.iloc[tr_pos], y_lab.iloc[tr_pos]
X_oos, y_oos = X_lab.iloc[oos_pos], y_lab.iloc[oos_pos]
print(f'TR: {X_tr.shape}, OOS: {X_oos.shape}\n')

print('=== DIAG 1: Single-feature OOS AUC (top 20) ===')
single_auc = {}
for col in X_tr.columns:
    vals = X_oos[col].values
    if np.unique(vals).size < 2: continue
    try:
        auc = roc_auc_score(y_oos, vals)
        single_auc[col] = max(auc, 1-auc)
    except: pass
top_feats = pd.Series(single_auc).sort_values(ascending=False)
print(top_feats.head(20).round(3).to_string())
print(f'\nBEST single feature AUC: {top_feats.iloc[0]:.3f}')
print(f'Top-10 mean: {top_feats.head(10).mean():.3f}')
print(f'How many features have AUC > 0.55: {(top_feats > 0.55).sum()}')

print('\n=== DIAG 2: Train vs OOS distribution shift (top 15 most-shifted features) ===')
ks_results = {}
for col in X_tr.columns:
    if X_tr[col].std() == 0: continue
    ks_stat, _ = stats.ks_2samp(X_tr[col].values, X_oos[col].values)
    ks_results[col] = ks_stat
print(pd.Series(ks_results).sort_values(ascending=False).head(15).round(3).to_string())
print(f'\nMean KS stat (all features): {pd.Series(ks_results).mean():.3f}')
print(f'Features with KS > 0.3 (severe shift): {(pd.Series(ks_results) > 0.3).sum()}')

print('\n=== DIAG 3: Label balance over time ===')
events_lab['year_month'] = pd.to_datetime(events_lab['t']).dt.to_period('M')
month_summary = events_lab.groupby('year_month')['label'].agg(['count','mean']).round(3)
print(month_summary.to_string())
print(f'\nTrain label_1_share: {y_tr.mean():.3f}')
print(f'OOS label_1_share: {y_oos.mean():.3f}')

print('\n=== DIAG 4: Per-instrument event counts and label balance ===')
tr_summary = events_lab.iloc[tr_pos].groupby('instrument').agg(train_n=('label','count'), train_pos=('label','mean'))
oos_summary = events_lab.iloc[oos_pos].groupby('instrument').agg(oos_n=('label','count'), oos_pos=('label','mean'))
inst_summary = pd.concat([tr_summary, oos_summary], axis=1).round(3)
print(inst_summary.to_string())

print('\n=== DIAG 5: Feature scale check (vol-related) ===')
key_feats = ['vol_5d','vol_21d','vol_63d','ewma_vol_50','side_signal','mom_21d','trend_tval_21d']
for f in key_feats:
    if f in X_tr.columns:
        print(f'{f:25s}  TR: mean={X_tr[f].mean():.3f} std={X_tr[f].std():.3f}  OOS: mean={X_oos[f].mean():.3f} std={X_oos[f].std():.3f}')
