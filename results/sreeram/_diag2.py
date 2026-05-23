"""DIAG 2: Try a battery of fixes. Each is a one-line config change."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from stml.io import load_clean_data
from stml.labeling import extract_signal_events, get_meta_labels, get_uniqueness_weights
from stml.features import compute_features
from stml.regimes import compute_regime_features
from stml.cv import split_by_boundary, PurgedKFold
from stml.models import ElasticNetLogReg, XGBoostMeta
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

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
events_all['side'] = events_all['side']
mask = ~events_all['label'].isna()
events_lab = events_all.loc[mask].reset_index(drop=True)
X_lab = X_all.loc[mask].reset_index(drop=True)
y_lab = events_lab['label'].astype(int)
t_lab = events_lab['t']
weights_full = get_uniqueness_weights(labels)
w_aligned = pd.Series(weights_full.values, index=labels.set_index(key).index).reindex(events_all.set_index(key).index).reset_index(drop=True)
w_lab = w_aligned.loc[mask].reset_index(drop=True)

tr_pos, _ = split_by_boundary(t_lab, boundary, embargo_td=embargo)
oos_pos = np.where((t_lab.values >= boundary) & (t_lab.values < pd.Timestamp('2022-07-01')))[0]
X_tr_full, y_tr = X_lab.iloc[tr_pos], y_lab.iloc[tr_pos]
X_oos_full, y_oos = X_lab.iloc[oos_pos], y_lab.iloc[oos_pos]
t_tr = t_lab.iloc[tr_pos]
t1_tr = events_lab.iloc[tr_pos]['t1_orig']
w_tr = w_lab.iloc[tr_pos]

def score(y, p, label='?'):
    auc = roc_auc_score(y, p) if y.nunique()>1 else float('nan')
    ll = log_loss(y, np.clip(p, 1e-7, 1-1e-7))
    br = brier_score_loss(y, p)
    print(f'  {label:35s} AUC={auc:.3f}  LL={ll:.3f}  Brier={br:.3f}')

# ---- BASELINE: Full features XGBoost ----
print('=== FIX 0: BASELINE (full features, XGBoost) ===')
m = XGBoostMeta(n_iter=15, embargo_td=embargo, random_state=42)
m.fit(X_tr_full, pd.Series(y_tr.values, index=X_tr_full.index),
      t=pd.Series(t_tr.values, index=X_tr_full.index),
      t1=pd.Series(t1_tr.values, index=X_tr_full.index),
      sample_weight=pd.Series(w_tr.values, index=X_tr_full.index))
score(y_oos, m.predict_proba(X_oos_full), 'baseline')
baseline_proba = m.predict_proba(X_oos_full)

# ---- FIX 1: DROP z-scored features (use only raw + bounded) ----
print('\n=== FIX 1: Drop z-scored features ===')
non_z_cols = [c for c in X_tr_full.columns if not c.startswith('z_')]
X_tr_nz, X_oos_nz = X_tr_full[non_z_cols], X_oos_full[non_z_cols]
print(f'  Features: {X_tr_nz.shape[1]} (was {X_tr_full.shape[1]})')
m = XGBoostMeta(n_iter=15, embargo_td=embargo, random_state=42)
m.fit(X_tr_nz, pd.Series(y_tr.values, index=X_tr_nz.index),
      t=pd.Series(t_tr.values, index=X_tr_nz.index),
      t1=pd.Series(t1_tr.values, index=X_tr_nz.index),
      sample_weight=pd.Series(w_tr.values, index=X_tr_nz.index))
score(y_oos, m.predict_proba(X_oos_nz), 'no z-scored features')

# ---- FIX 2: DROP equity instruments ----
print('\n=== FIX 2: Drop equity instruments ===')
equity = ['es1s','nq1s','fesx1s']
tr_non_eq = np.array([i for i in tr_pos if events_lab.iloc[i]['instrument'] not in equity])
oos_non_eq = np.array([i for i in oos_pos if events_lab.iloc[i]['instrument'] not in equity])
X_tr_ne, y_tr_ne = X_lab.iloc[tr_non_eq], y_lab.iloc[tr_non_eq]
X_oos_ne, y_oos_ne = X_lab.iloc[oos_non_eq], y_lab.iloc[oos_non_eq]
t_tr_ne, t1_tr_ne, w_tr_ne = t_lab.iloc[tr_non_eq], events_lab.iloc[tr_non_eq]['t1_orig'], w_lab.iloc[tr_non_eq]
print(f'  Train: {len(tr_non_eq)} (was {len(tr_pos)}), OOS: {len(oos_non_eq)} (was {len(oos_pos)})')
m = XGBoostMeta(n_iter=15, embargo_td=embargo, random_state=42)
m.fit(X_tr_ne, pd.Series(y_tr_ne.values, index=X_tr_ne.index),
      t=pd.Series(t_tr_ne.values, index=X_tr_ne.index),
      t1=pd.Series(t1_tr_ne.values, index=X_tr_ne.index),
      sample_weight=pd.Series(w_tr_ne.values, index=X_tr_ne.index))
score(y_oos_ne, m.predict_proba(X_oos_ne), 'commodities only')

# ---- FIX 3: RECENCY-WEIGHTED training ----
print('\n=== FIX 3: Recency-weight training (decay 0.5 per year) ===')
# weight = decay ^ (years before boundary)
decay = 0.5
days_before = (boundary - pd.to_datetime(t_tr).values).astype('timedelta64[D]').astype(int)
years_before = days_before / 365.25
rec_w = (decay ** years_before) * w_tr.values
rec_w = rec_w / rec_w.mean()
print(f'  Recency weights: min={rec_w.min():.3f}, max={rec_w.max():.3f}, mean=1.0')
m = XGBoostMeta(n_iter=15, embargo_td=embargo, random_state=42)
m.fit(X_tr_full, pd.Series(y_tr.values, index=X_tr_full.index),
      t=pd.Series(t_tr.values, index=X_tr_full.index),
      t1=pd.Series(t1_tr.values, index=X_tr_full.index),
      sample_weight=pd.Series(rec_w, index=X_tr_full.index))
score(y_oos, m.predict_proba(X_oos_full), 'recency-weighted')

# ---- FIX 4: ENSEMBLE — LogReg + XGBoost ----
print('\n=== FIX 4: Ensemble LogReg + XGBoost (geometric mean) ===')
m_lr = ElasticNetLogReg(n_iter=15, embargo_td=embargo)
m_lr.fit(X_tr_full, pd.Series(y_tr.values, index=X_tr_full.index),
         t=pd.Series(t_tr.values, index=X_tr_full.index),
         t1=pd.Series(t1_tr.values, index=X_tr_full.index),
         sample_weight=pd.Series(w_tr.values, index=X_tr_full.index))
p_lr = m_lr.predict_proba(X_oos_full)
p_xgb = baseline_proba
p_ens = (p_lr * p_xgb) ** 0.5
p_ens_avg = (p_lr + p_xgb) / 2
score(y_oos, p_ens, 'geo-mean ensemble')
score(y_oos, p_ens_avg, 'arith-mean ensemble')

# ---- FIX 5: PER-INSTRUMENT MODELS (XGBoost per instrument) ----
print('\n=== FIX 5: Per-instrument models ===')
per_inst_preds = np.full(len(y_oos), 0.5)
inst_results = {}
for inst in events_lab['instrument'].unique():
    tr_inst = np.array([i for i, idx in enumerate(tr_pos) if events_lab.iloc[idx]['instrument'] == inst])
    if len(tr_inst) < 30: continue
    oos_inst = np.array([i for i, idx in enumerate(oos_pos) if events_lab.iloc[idx]['instrument'] == inst])
    if len(oos_inst) == 0: continue
    X_tr_i = X_tr_full.iloc[tr_inst]
    y_tr_i = pd.Series(y_tr.values[tr_inst], index=X_tr_i.index)
    t_tr_i = pd.Series(t_tr.values[tr_inst], index=X_tr_i.index)
    t1_tr_i = pd.Series(t1_tr.values[tr_inst], index=X_tr_i.index)
    w_tr_i = pd.Series(w_tr.values[tr_inst], index=X_tr_i.index)
    try:
        m = XGBoostMeta(n_iter=10, embargo_td=embargo, n_splits_inner=3 if len(tr_inst)<200 else 5)
        m.fit(X_tr_i, y_tr_i, t=t_tr_i, t1=t1_tr_i, sample_weight=w_tr_i)
        X_oos_i = X_oos_full.iloc[oos_inst]
        y_oos_i = y_oos.values[oos_inst]
        p = m.predict_proba(X_oos_i)
        per_inst_preds[oos_inst] = p
        if len(y_oos_i) >= 5 and len(set(y_oos_i)) > 1:
            inst_results[inst] = roc_auc_score(y_oos_i, p)
    except Exception as e:
        print(f'  {inst}: failed ({e})')
print(f'  Per-instrument AUCs: {pd.Series(inst_results).round(3).to_string()}')
score(y_oos, per_inst_preds, 'per-instrument combined')

# ---- FIX 6: TRAIN WITH BOUNDARY=2022-04-01 (3 more months of training) ----
print('\n=== FIX 6: Closer boundary (2022-04-01) ===')
b2 = pd.Timestamp('2022-04-01')
tr_pos2, _ = split_by_boundary(t_lab, b2, embargo_td=embargo)
oos_pos2 = np.where((t_lab.values >= b2) & (t_lab.values < pd.Timestamp('2022-07-01')))[0]
X_tr2, y_tr2 = X_lab.iloc[tr_pos2], y_lab.iloc[tr_pos2]
X_oos2, y_oos2 = X_lab.iloc[oos_pos2], y_lab.iloc[oos_pos2]
t_tr2, t1_tr2, w_tr2 = t_lab.iloc[tr_pos2], events_lab.iloc[tr_pos2]['t1_orig'], w_lab.iloc[tr_pos2]
m = XGBoostMeta(n_iter=15, embargo_td=embargo)
m.fit(X_tr2, pd.Series(y_tr2.values, index=X_tr2.index),
      t=pd.Series(t_tr2.values, index=X_tr2.index),
      t1=pd.Series(t1_tr2.values, index=X_tr2.index),
      sample_weight=pd.Series(w_tr2.values, index=X_tr2.index))
score(y_oos2, m.predict_proba(X_oos2), 'boundary=2022-04-01 (predicts Q2 only)')
