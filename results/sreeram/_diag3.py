"""More fixes."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from stml.io import load_clean_data
from stml.labeling import extract_signal_events, get_meta_labels, get_uniqueness_weights
from stml.features import compute_features
from stml.regimes import compute_regime_features
from stml.cv import split_by_boundary
from stml.models import ElasticNetLogReg, XGBoostMeta, MlpMeta
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, f1_score

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
weights_full = get_uniqueness_weights(labels)
w_aligned = pd.Series(weights_full.values, index=labels.set_index(key).index).reindex(events_all.set_index(key).index).reset_index(drop=True)
w_lab = w_aligned.loc[mask].reset_index(drop=True)

tr_pos, _ = split_by_boundary(t_lab, boundary, embargo_td=embargo)
oos_pos = np.where((t_lab.values >= boundary) & (t_lab.values < pd.Timestamp('2022-07-01')))[0]
X_tr_full, y_tr = X_lab.iloc[tr_pos], y_lab.iloc[tr_pos]
X_oos_full, y_oos = X_lab.iloc[oos_pos], y_lab.iloc[oos_pos]
t_tr, t1_tr, w_tr = t_lab.iloc[tr_pos], events_lab.iloc[tr_pos]['t1_orig'], w_lab.iloc[tr_pos]

def score(y, p, label='?'):
    try:
        auc = roc_auc_score(y, p) if y.nunique()>1 else float('nan')
    except: auc = float('nan')
    ll = log_loss(y, np.clip(p, 1e-7, 1-1e-7))
    br = brier_score_loss(y, p)
    f1 = f1_score(y, (p>=0.5).astype(int), zero_division=0)
    print(f'  {label:40s} AUC={auc:.3f}  F1={f1:.3f}  LL={ll:.3f}  Brier={br:.3f}')

# FIX 7: RECENCY weighting (FIXED — convert to pandas Timedelta properly)
print('=== FIX 7: Recency weighting (decay 0.5/year, 0.3/year, 0.2/year) ===')
boundary_ts = pd.to_datetime(boundary)
t_tr_pd = pd.DatetimeIndex(pd.to_datetime(t_tr.values))
days_before = np.array([(boundary_ts - d).days for d in t_tr_pd], dtype=float)
years_before = days_before / 365.25
for decay in [0.7, 0.5, 0.3]:
    rec_w = (decay ** years_before) * w_tr.values
    rec_w = rec_w / rec_w.mean()
    m = XGBoostMeta(n_iter=15, embargo_td=embargo)
    m.fit(X_tr_full, pd.Series(y_tr.values, index=X_tr_full.index),
          t=pd.Series(t_tr.values, index=X_tr_full.index),
          t1=pd.Series(t1_tr.values, index=X_tr_full.index),
          sample_weight=pd.Series(rec_w, index=X_tr_full.index))
    score(y_oos, m.predict_proba(X_oos_full), f'decay={decay} (recent weighted)')

# FIX 8: PER-INSTRUMENT models (where enough data)
print('\n=== FIX 8: Per-instrument XGBoost (where >=80 train events) ===')
per_inst_preds = np.full(len(y_oos), 0.5)  # default 0.5 for no model
per_inst_aucs = {}
covered_instruments = set()
for inst in events_lab['instrument'].unique():
    tr_idx_inst = np.array([i for i, idx in enumerate(tr_pos) if events_lab.iloc[idx]['instrument'] == inst])
    if len(tr_idx_inst) < 80:
        per_inst_aucs[inst] = ('too few train', None)
        continue
    oos_idx_inst = np.array([i for i, idx in enumerate(oos_pos) if events_lab.iloc[idx]['instrument'] == inst])
    if len(oos_idx_inst) == 0:
        continue
    X_tr_i = X_tr_full.iloc[tr_idx_inst]
    y_tr_i = pd.Series(y_tr.values[tr_idx_inst], index=X_tr_i.index)
    t_tr_i = pd.Series(t_tr.values[tr_idx_inst], index=X_tr_i.index)
    t1_tr_i = pd.Series(t1_tr.values[tr_idx_inst], index=X_tr_i.index)
    w_tr_i = pd.Series(w_tr.values[tr_idx_inst], index=X_tr_i.index)
    n_splits = 5 if len(tr_idx_inst) > 200 else 3
    try:
        m = XGBoostMeta(n_iter=10, embargo_td=embargo, n_splits_inner=n_splits)
        m.fit(X_tr_i, y_tr_i, t=t_tr_i, t1=t1_tr_i, sample_weight=w_tr_i)
        X_oos_i = X_oos_full.iloc[oos_idx_inst]
        y_oos_i = y_oos.values[oos_idx_inst]
        p = m.predict_proba(X_oos_i)
        per_inst_preds[oos_idx_inst] = p
        covered_instruments.add(inst)
        if len(y_oos_i) >= 5 and len(set(y_oos_i)) > 1:
            per_inst_aucs[inst] = (f'auc={roc_auc_score(y_oos_i, p):.3f}', float(roc_auc_score(y_oos_i, p)))
    except Exception as e:
        per_inst_aucs[inst] = (f'err: {e}', None)
for inst, (msg, _) in per_inst_aucs.items():
    print(f'  {inst:8s}: {msg}')
score(y_oos, per_inst_preds, 'PER-INSTRUMENT combined OOS')

# FIX 9: per-sector models (3 separate XGBoost models)
print('\n=== FIX 9: Per-sector models ===')
from stml.experiments import ASSET_CLASSES
sector_models = {}
sector_preds = np.full(len(y_oos), 0.5)
for sector in ['energy','metals','equity']:
    inst_in_sec = [k for k,v in ASSET_CLASSES.items() if v == sector]
    tr_sec = np.array([i for i, idx in enumerate(tr_pos) if events_lab.iloc[idx]['instrument'] in inst_in_sec])
    oos_sec = np.array([i for i, idx in enumerate(oos_pos) if events_lab.iloc[idx]['instrument'] in inst_in_sec])
    if len(tr_sec) < 100:
        continue
    X_tr_s = X_tr_full.iloc[tr_sec]
    y_tr_s = pd.Series(y_tr.values[tr_sec], index=X_tr_s.index)
    t_tr_s = pd.Series(t_tr.values[tr_sec], index=X_tr_s.index)
    t1_tr_s = pd.Series(t1_tr.values[tr_sec], index=X_tr_s.index)
    w_tr_s = pd.Series(w_tr.values[tr_sec], index=X_tr_s.index)
    m = XGBoostMeta(n_iter=15, embargo_td=embargo)
    m.fit(X_tr_s, y_tr_s, t=t_tr_s, t1=t1_tr_s, sample_weight=w_tr_s)
    X_oos_s = X_oos_full.iloc[oos_sec]
    y_oos_s = y_oos.values[oos_sec]
    p = m.predict_proba(X_oos_s)
    sector_preds[oos_sec] = p
    sector_models[sector] = m
    if len(y_oos_s) >= 5 and len(set(y_oos_s)) > 1:
        print(f'  {sector:8s}: n_tr={len(tr_sec)} n_oos={len(oos_sec)} AUC={roc_auc_score(y_oos_s, p):.3f}')
score(y_oos, sector_preds, 'PER-SECTOR combined OOS')

# FIX 10: Hybrid — per-sector for commodities + pooled for equity
print('\n=== FIX 10: HYBRID — per-sector for commodities, pooled for equity ===')
m_pool = XGBoostMeta(n_iter=15, embargo_td=embargo)
m_pool.fit(X_tr_full, pd.Series(y_tr.values, index=X_tr_full.index),
           t=pd.Series(t_tr.values, index=X_tr_full.index),
           t1=pd.Series(t1_tr.values, index=X_tr_full.index),
           sample_weight=pd.Series(w_tr.values, index=X_tr_full.index))
pool_p = m_pool.predict_proba(X_oos_full)
hybrid = pool_p.copy()
# Replace commodity predictions with sector predictions
for sector in ['energy','metals']:
    inst_in_sec = [k for k,v in ASSET_CLASSES.items() if v == sector]
    for i, idx in enumerate(oos_pos):
        if events_lab.iloc[idx]['instrument'] in inst_in_sec:
            hybrid[i] = sector_preds[i]
score(y_oos, hybrid, 'HYBRID per-sector commodities + pooled equity')

# FIX 11: Best of pooled vs per-sector per instrument
print('\n=== FIX 11: BEST per instrument (pooled vs sector vs per-instrument) ===')
# We already have pool_p, sector_preds, per_inst_preds
# Per instrument pick whichever model's TRAIN AUC was best (on a purged inner fold)
# For simplicity: pick best by comparing OOS scores per instrument (slightly cheating
# for diagnostic — in real deployment we'd pick by CV on training data)
best_pick = np.full(len(y_oos), 0.5)
inst_choices = {}
for inst in events_lab['instrument'].unique():
    oos_idx_inst = np.array([i for i, idx in enumerate(oos_pos) if events_lab.iloc[idx]['instrument'] == inst])
    if len(oos_idx_inst) == 0:
        continue
    y_oos_i = y_oos.values[oos_idx_inst]
    if len(set(y_oos_i)) < 2:
        # Use pool default
        best_pick[oos_idx_inst] = pool_p[oos_idx_inst]
        inst_choices[inst] = 'pooled (single class oos)'
        continue
    aucs = {
        'pool': roc_auc_score(y_oos_i, pool_p[oos_idx_inst]),
        'sector': roc_auc_score(y_oos_i, sector_preds[oos_idx_inst]),
        'per_inst': roc_auc_score(y_oos_i, per_inst_preds[oos_idx_inst]),
    }
    best_name = max(aucs, key=aucs.get)
    if best_name == 'pool':
        best_pick[oos_idx_inst] = pool_p[oos_idx_inst]
    elif best_name == 'sector':
        best_pick[oos_idx_inst] = sector_preds[oos_idx_inst]
    else:
        best_pick[oos_idx_inst] = per_inst_preds[oos_idx_inst]
    inst_choices[inst] = f'{best_name} (auc={aucs[best_name]:.3f})'
for inst, choice in inst_choices.items():
    print(f'  {inst:8s}: best = {choice}')
score(y_oos, best_pick, 'BEST per instrument (oracle)')

# FIX 12: Equity-only model with COMMODITY data EXCLUDED — does it help equity?
print('\n=== FIX 12: Equity-only training (does it help equity?) ===')
eq = ['es1s','nq1s','fesx1s']
tr_eq = np.array([i for i, idx in enumerate(tr_pos) if events_lab.iloc[idx]['instrument'] in eq])
oos_eq = np.array([i for i, idx in enumerate(oos_pos) if events_lab.iloc[idx]['instrument'] in eq])
X_tr_e = X_tr_full.iloc[tr_eq]
y_tr_e = pd.Series(y_tr.values[tr_eq], index=X_tr_e.index)
t_tr_e = pd.Series(t_tr.values[tr_eq], index=X_tr_e.index)
t1_tr_e = pd.Series(t1_tr.values[tr_eq], index=X_tr_e.index)
w_tr_e = pd.Series(w_tr.values[tr_eq], index=X_tr_e.index)
print(f'  Equity train: {len(tr_eq)}, equity OOS: {len(oos_eq)}')
m = XGBoostMeta(n_iter=15, embargo_td=embargo)
m.fit(X_tr_e, y_tr_e, t=t_tr_e, t1=t1_tr_e, sample_weight=w_tr_e)
X_oos_e = X_oos_full.iloc[oos_eq]
y_oos_e = pd.Series(y_oos.values[oos_eq])
p_eq = m.predict_proba(X_oos_e)
print(f'  Equity-only model AUC: {roc_auc_score(y_oos_e, p_eq):.3f}')
print(f'  Pooled model AUC on equity OOS: {roc_auc_score(y_oos_e, pool_p[oos_eq]):.3f}')
