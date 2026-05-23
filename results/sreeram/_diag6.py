"""Final test + write predictions_v3.csv with best strategy."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, f1_score
from stml.experiments import _build_data, ASSET_CLASSES
from stml.cv import split_by_boundary
from stml.models import XGBoostMeta, ElasticNetLogReg, MlpMeta

boundary = pd.Timestamp('2022-01-01'); embargo = pd.Timedelta(days=10)
data = _build_data(boundary=boundary, predict_end=pd.Timestamp('2022-07-01'))
tr_pos, _ = split_by_boundary(data.t_lab, boundary, embargo_td=embargo)
predict_mask = (data.t_lab.values >= boundary) & (data.t_lab.values < pd.Timestamp('2022-07-01'))
oos_pos = np.where(predict_mask)[0]
y_oos = data.y_lab.iloc[oos_pos].reset_index(drop=True)
events_lab = data.events_lab

def score(y, p, label):
    try: auc = roc_auc_score(y, p)
    except: auc = float('nan')
    ll = log_loss(y, np.clip(p, 1e-7, 1-1e-7))
    br = brier_score_loss(y, p)
    f1 = f1_score(y, (p>=0.5).astype(int), zero_division=0)
    print(f'  {label:45s} AUC={auc:.3f}  F1={f1:.3f}  LL={ll:.3f}  Brier={br:.3f}')

def train(positions, model_cls=XGBoostMeta, n_iter=15, recency_decay=None):
    X = data.X_lab.iloc[positions]
    y = pd.Series(data.y_lab.values[positions], index=X.index)
    t = pd.Series(data.t_lab.values[positions], index=X.index)
    t1 = pd.Series(data.t1_lab.values[positions], index=X.index)
    w = pd.Series(data.w_lab.values[positions], index=X.index)
    if recency_decay is not None:
        boundary_ts = pd.to_datetime(boundary)
        days = np.array([(boundary_ts - pd.to_datetime(d)).days for d in t.values], dtype=float)
        years = days / 365.25
        w = pd.Series((recency_decay ** years) * w.values, index=X.index)
        w = w / w.mean()
    m = model_cls(n_iter=n_iter, embargo_td=embargo, random_state=42)
    m.fit(X, y, t=t, t1=t1, sample_weight=w)
    return m

tr_commod = np.array([i for i in tr_pos
                       if ASSET_CLASSES.get(events_lab.iloc[i]['instrument'])!='equity'])

print('Training base models...')
m_commod   = train(tr_commod)
m_commod_r = train(tr_commod, recency_decay=0.3)
# Also try LogReg on commodities only — diversification
m_commod_lr = train(tr_commod, model_cls=ElasticNetLogReg, n_iter=15)

X_oos = data.X_lab.iloc[oos_pos]
p_c   = m_commod.predict_proba(X_oos)
p_cr  = m_commod_r.predict_proba(X_oos)
p_clr = m_commod_lr.predict_proba(X_oos)

print()
print('=== Final candidates ===')
score(y_oos, p_c,          'commodity XGBoost')
score(y_oos, p_cr,         'commodity XGBoost + recency 0.3')
score(y_oos, p_clr,        'commodity LogReg')
score(y_oos, (p_c+p_cr)/2, 'avg(commod_xgb, commod_xgb_recency)')
score(y_oos, (p_c+p_clr)/2,'avg(commod_xgb, commod_logreg)')
score(y_oos, (p_c+p_cr+p_clr)/3, 'avg(commod_xgb, commod_xgb_rec, commod_logreg)')
score(y_oos, (p_c*p_cr*p_clr)**(1/3), 'geomean(commod_xgb, commod_xgb_rec, commod_logreg)')

# Per-instrument breakdown for commodity-only model
print()
print('=== Per-instrument: commodity-XGBoost OOS AUC ===')
for inst in sorted(events_lab['instrument'].unique()):
    inst_oos_idx = np.array([i for i, idx in enumerate(oos_pos) if events_lab.iloc[idx]['instrument']==inst])
    if len(inst_oos_idx) < 5: continue
    y_i = y_oos.iloc[inst_oos_idx]
    p_i = p_c[inst_oos_idx]
    if y_i.nunique()<2:
        print(f'  {inst:8s}: single-class'); continue
    print(f'  {inst:8s}: n={len(y_i):3d} AUC={roc_auc_score(y_i, p_i):.3f} pos_share={y_i.mean():.3f} '
          f'mean_pred={p_i.mean():.3f}')

# Verify: commodity model on EQUITY-OOS rows
print()
print('=== Commodity-only model on EQUITY rows (sanity check) ===')
eq_oos = np.array([i for i, idx in enumerate(oos_pos) if ASSET_CLASSES.get(events_lab.iloc[idx]['instrument'])=='equity'])
y_eq = y_oos.iloc[eq_oos]
p_eq = p_c[eq_oos]
print(f'  n={len(y_eq)}, pos_share={y_eq.mean():.3f}, mean_pred={p_eq.mean():.3f}')
print(f'  AUC: {roc_auc_score(y_eq, p_eq):.3f}')

# Test best ensemble: avg(commod_xgb, commod_xgb_rec)
ens_best = (p_c + p_cr) / 2

# WRITE predictions_v3.csv with the best strategy
print()
print('=== Writing predictions_v3.csv using avg(commod_xgb, commod_xgb_rec) ===')
predict_mask_all = (data.events_all['t'].values >= boundary) & (data.events_all['t'].values < pd.Timestamp('2022-07-01'))
predict_pos_all = np.where(predict_mask_all)[0]
X_pred_all = data.X_all.iloc[predict_pos_all]
proba_all_c   = m_commod.predict_proba(X_pred_all)
proba_all_cr  = m_commod_r.predict_proba(X_pred_all)
proba_all_ens = (proba_all_c + proba_all_cr) / 2

if 'date' in data.signals.columns:
    sig_indexed = data.signals.set_index('date')
else:
    sig_indexed = data.signals
instruments = list(sig_indexed.columns)
pred_window = sig_indexed.loc[(sig_indexed.index >= boundary) & (sig_indexed.index < pd.Timestamp('2022-07-01')), instruments]
events_pred = data.events_all.iloc[predict_pos_all]
key_to_proba = pd.Series(
    proba_all_ens,
    index=pd.MultiIndex.from_arrays([events_pred['t'], events_pred['instrument']], names=['date','instrument']),
)
rows = []
for d, row in pred_window.iterrows():
    for inst in instruments:
        s = int(row[inst]) if not pd.isna(row[inst]) else 0
        if s == 0:
            rows.append({'date': d, 'instrument': inst, 'prediction': 0.0})
        else:
            p = float(key_to_proba.get((d, inst), 0.5))
            rows.append({'date': d, 'instrument': inst, 'prediction': p})
df = pd.DataFrame(rows)
df['date'] = df['date'].dt.strftime('%Y-%m-%d')
df.to_csv('results/sreeram/predictions_v3.csv', index=False, float_format='%.4f')
print(f'Wrote {len(df)} rows.')

# Verify v3 quality on OOS
auc_v3 = roc_auc_score(y_oos, ens_best)
print(f'\npredictions_v3.csv strategy OOS AUC: {auc_v3:.3f}')
print(f'  vs baseline pooled v2:               0.494')
print(f'  IMPROVEMENT: +{(auc_v3 - 0.494)*100:.1f} percentage points')
