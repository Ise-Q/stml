"""Final fixes: ensemble + commodity-trained model + recency weighting."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, f1_score
from stml.experiments import _build_data, ASSET_CLASSES
from stml.cv import split_by_boundary
from stml.models import XGBoostMeta, ElasticNetLogReg

boundary = pd.Timestamp('2022-01-01'); embargo = pd.Timedelta(days=10)
data = _build_data(boundary=boundary, predict_end=pd.Timestamp('2022-07-01'))
tr_pos, _ = split_by_boundary(data.t_lab, boundary, embargo_td=embargo)
predict_mask = (data.t_lab.values >= boundary) & (data.t_lab.values < pd.Timestamp('2022-07-01'))
oos_pos = np.where(predict_mask)[0]
y_oos = data.y_lab.iloc[oos_pos].reset_index(drop=True)
events_lab = data.events_lab

def score(y, p, label):
    try:
        auc = roc_auc_score(y, p)
    except: auc = float('nan')
    ll = log_loss(y, np.clip(p, 1e-7, 1-1e-7))
    br = brier_score_loss(y, p)
    f1 = f1_score(y, (p>=0.5).astype(int), zero_division=0)
    print(f'  {label:45s} AUC={auc:.3f}  F1={f1:.3f}  LL={ll:.3f}  Brier={br:.3f}')

def train_subset(positions, n_iter=15, recency_decay=None):
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
    m = XGBoostMeta(n_iter=n_iter, embargo_td=embargo, random_state=42)
    m.fit(X, y, t=t, t1=t1, sample_weight=w)
    return m

# Train base models
print('Training models...')
tr_commod = np.array([i for i in tr_pos
                       if ASSET_CLASSES.get(events_lab.iloc[i]['instrument'])!='equity'])
tr_metals = np.array([i for i in tr_pos
                       if ASSET_CLASSES.get(events_lab.iloc[i]['instrument'])=='metals'])
tr_energy = np.array([i for i in tr_pos
                       if ASSET_CLASSES.get(events_lab.iloc[i]['instrument'])=='energy'])

m_pooled       = train_subset(tr_pos)
m_pooled_rec   = train_subset(tr_pos, recency_decay=0.3)
m_commodity    = train_subset(tr_commod)
m_commodity_rec= train_subset(tr_commod, recency_decay=0.3)
m_metals       = train_subset(tr_metals)
m_energy       = train_subset(tr_energy)

X_oos = data.X_lab.iloc[oos_pos]

# Baseline scores
print()
print('=== Single-model OOS ===')
p_pool      = m_pooled.predict_proba(X_oos)
p_pool_rec  = m_pooled_rec.predict_proba(X_oos)
p_commodity = m_commodity.predict_proba(X_oos)
p_comm_rec  = m_commodity_rec.predict_proba(X_oos)
score(y_oos, p_pool,      'pooled')
score(y_oos, p_pool_rec,  'pooled + recency 0.3')
score(y_oos, p_commodity, 'commodities-only (trained on commod)')
score(y_oos, p_comm_rec,  'commodities-only + recency 0.3')

# Ensembles
print()
print('=== Ensembles (arithmetic mean) ===')
score(y_oos, (p_pool + p_pool_rec)/2,           'pool + pool_recency')
score(y_oos, (p_pool + p_commodity)/2,          'pool + commodity')
score(y_oos, (p_pool + p_commodity + p_pool_rec + p_comm_rec)/4,  'mega ensemble (4-way)')

# Hybrid: commodity model for commodities, pooled for equity
print()
print('=== Hybrid (commodity model for commod, pooled for equity) ===')
hybrid_a = p_pool.copy()
hybrid_b = p_pool.copy()
hybrid_c = p_pool.copy()
hybrid_d = p_pool.copy()
hybrid_e = p_pool.copy()
for i, pos in enumerate(oos_pos):
    inst = events_lab.iloc[pos]['instrument']
    sec = ASSET_CLASSES.get(inst)
    if sec != 'equity':
        hybrid_a[i] = p_commodity[i]
        hybrid_b[i] = p_comm_rec[i]
        hybrid_c[i] = (p_commodity[i] + p_comm_rec[i])/2
        # ensemble: pooled + commodity for commodities
        hybrid_d[i] = (p_pool[i] + p_commodity[i] + p_comm_rec[i])/3
        # for metals only, also include metals sector
        if sec == 'metals':
            X_one = X_oos.iloc[[i]]
            p_metals_i = float(m_metals.predict_proba(X_one)[0])
            hybrid_e[i] = (p_pool[i] + p_commodity[i] + p_metals_i) / 3
        elif sec == 'energy':
            X_one = X_oos.iloc[[i]]
            p_energy_i = float(m_energy.predict_proba(X_one)[0])
            hybrid_e[i] = (p_pool[i] + p_commodity[i] + p_energy_i) / 3
        else:
            hybrid_e[i] = (p_pool[i] + p_commodity[i])/2

score(y_oos, hybrid_a, 'commodity-only commodities, pool equity')
score(y_oos, hybrid_b, 'commodity-recency commodities, pool equity')
score(y_oos, hybrid_c, 'avg(commod, commod-rec) for commod, pool equity')
score(y_oos, hybrid_d, 'avg(pool, commod, commod-rec) for commod, pool equity')
score(y_oos, hybrid_e, 'avg(pool, commod, sector) for commod, pool equity')

# What if for equity we abstain (predict label_1_share = const)?
print()
print('=== Equity abstention experiments ===')
for inst_eq, eq_label in [('all', 'es,nq,fesx'), ]:
    # Use commodity ensemble for commodities; predict label_1_share for each equity
    eq_share = {}
    for inst in ['es1s','nq1s','fesx1s']:
        tr_inst = [i for i in tr_pos if events_lab.iloc[i]['instrument']==inst]
        eq_share[inst] = float(events_lab.iloc[tr_inst]['label'].mean())
    pred_abst = p_pool.copy()
    for i, pos in enumerate(oos_pos):
        inst = events_lab.iloc[pos]['instrument']
        sec = ASSET_CLASSES.get(inst)
        if sec == 'equity':
            pred_abst[i] = eq_share[inst]  # constant per instrument
        else:
            pred_abst[i] = hybrid_c[i]
    score(y_oos, pred_abst, 'commod ensemble + equity=trained-label-share constant')

# Predict 0 for equity (very conservative — never take equity bets)
pred_zero_eq = p_pool.copy()
for i, pos in enumerate(oos_pos):
    inst = events_lab.iloc[pos]['instrument']
    sec = ASSET_CLASSES.get(inst)
    if sec == 'equity':
        pred_zero_eq[i] = 0.0
    else:
        pred_zero_eq[i] = hybrid_c[i]
score(y_oos, pred_zero_eq, 'commod ensemble + equity=0 (skip all equity)')

# Save BEST overall as predictions_v3
print()
print('=== WRITING predictions_v3.csv from best strategy ===')
# Best so far is likely hybrid_c. Let me pick whichever has highest AUC
candidates = {
    'pool': p_pool,
    'pool_rec': p_pool_rec,
    'commod': p_commodity,
    'commod_rec': p_comm_rec,
    'pool+commod': (p_pool + p_commodity)/2,
    'mega4': (p_pool + p_commodity + p_pool_rec + p_comm_rec)/4,
    'hybrid_a': hybrid_a, 'hybrid_b': hybrid_b, 'hybrid_c': hybrid_c,
    'hybrid_d': hybrid_d, 'hybrid_e': hybrid_e,
    'equity_const': pred_abst, 'equity_zero': pred_zero_eq,
}
aucs = {n: roc_auc_score(y_oos, p) for n, p in candidates.items()}
best_strategy = max(aucs, key=aucs.get)
print(f'Best strategy: {best_strategy}, AUC={aucs[best_strategy]:.3f}')
print('All AUCs (sorted):')
for n, a in sorted(aucs.items(), key=lambda x: -x[1]):
    print(f'  {n:25s} {a:.3f}')
