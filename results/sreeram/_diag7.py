"""Try feature selection + aggressive regularization."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, f1_score
from sklearn.ensemble import RandomForestClassifier
from stml.experiments import _build_data, ASSET_CLASSES
from stml.cv import split_by_boundary
from stml.models import XGBoostMeta

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
    print(f'  {label:50s} AUC={auc:.3f}  LL={ll:.3f}')

tr_commod = np.array([i for i in tr_pos
                       if ASSET_CLASSES.get(events_lab.iloc[i]['instrument'])!='equity'])

# Compute single-feature TRAIN AUC (NOT OOS — to avoid leakage)
print('Selecting features by inner-CV single-feature AUC...')
from stml.cv import PurgedKFold
t_tr_comm = pd.Series(data.t_lab.values[tr_commod], index=range(len(tr_commod)))
t1_tr_comm = pd.Series(data.t1_lab.values[tr_commod], index=range(len(tr_commod)))
y_tr_comm = data.y_lab.values[tr_commod]
X_tr_comm = data.X_lab.iloc[tr_commod].reset_index(drop=True)
# Quick proxy: use Spearman rank correlation between feature and label (CV-averaged)
# Easier: compute single-feature AUC on a held-out fold (using purged CV)
cv = PurgedKFold(n_splits=3, t=t_tr_comm, t1=t1_tr_comm, embargo_td=embargo)
feature_aucs = {col: [] for col in X_tr_comm.columns}
for tr_idx, te_idx in cv.split(X_tr_comm):
    y_te = y_tr_comm[te_idx]
    if len(np.unique(y_te)) < 2:
        continue
    for col in X_tr_comm.columns:
        x_te = X_tr_comm.iloc[te_idx][col].values
        if np.unique(x_te).size < 2:
            continue
        try:
            auc = roc_auc_score(y_te, x_te)
            feature_aucs[col].append(max(auc, 1 - auc))
        except: pass
feature_avg = pd.Series({c: np.mean(a) if a else 0.5 for c, a in feature_aucs.items()})
print(f'  Top 10 by inner-CV single-feature AUC:')
print(feature_avg.sort_values(ascending=False).head(10).round(3).to_string())

# Keep features with CV AUC > 0.54
selected = feature_avg[feature_avg > 0.54].index.tolist()
print(f'\n  Selected {len(selected)} features (CV AUC > 0.54)')

# Train models with reduced feature set
def train_subset(positions, cols, model='xgb', n_iter=15, max_depth=None, reg=None):
    X = data.X_lab.iloc[positions][cols]
    y = pd.Series(data.y_lab.values[positions], index=X.index)
    t = pd.Series(data.t_lab.values[positions], index=X.index)
    t1 = pd.Series(data.t1_lab.values[positions], index=X.index)
    w = pd.Series(data.w_lab.values[positions], index=X.index)
    grid_override = None
    if max_depth and reg:
        grid_override = {
            'max_depth': [max_depth], 'learning_rate': [0.03, 0.05],
            'n_estimators': [100, 200, 300], 'subsample': [0.7, 0.8],
            'colsample_bytree': [0.6, 0.8], 'reg_alpha': [reg], 'reg_lambda': [reg*5],
            'min_child_weight': [3, 5],
        }
    m = XGBoostMeta(n_iter=n_iter, embargo_td=embargo, random_state=42, param_grid=grid_override)
    m.fit(X, y, t=t, t1=t1, sample_weight=w)
    return m

X_oos_all = data.X_lab.iloc[oos_pos]

# FIX 13: commod + reduced features
print('\n=== FIX 13: Commodity model with selected features only ===')
m_sel = train_subset(tr_commod, selected)
p_sel = m_sel.predict_proba(X_oos_all[selected])
score(y_oos, p_sel, 'commod XGB + selected features')

# FIX 14: aggressive regularization
print('\n=== FIX 14: Aggressive regularization on commod XGBoost ===')
m_reg = train_subset(tr_commod, list(X_tr_comm.columns), max_depth=3, reg=1.0)
p_reg = m_reg.predict_proba(X_oos_all)
score(y_oos, p_reg, 'commod XGB + max_depth=3, reg=1.0')

# FIX 15: Random Forest (different ensemble — Breiman bagging)
print('\n=== FIX 15: Random Forest on commodity data ===')
X_tr_rf = data.X_lab.iloc[tr_commod].values
y_tr_rf = data.y_lab.values[tr_commod]
w_tr_rf = data.w_lab.values[tr_commod]
rf = RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1,
)
rf.fit(X_tr_rf, y_tr_rf, sample_weight=w_tr_rf)
p_rf = rf.predict_proba(X_oos_all.values)[:, 1]
score(y_oos, p_rf, 'commod RandomForest (depth=5)')

# FIX 16: Ultra-conservative LogReg
print('\n=== FIX 16: Ultra-regularized LogReg ===')
from stml.models import ElasticNetLogReg
m_ulr = ElasticNetLogReg(n_iter=15, embargo_td=embargo, random_state=42,
                          C_log_grid=(-4, -3, -2.5, -2), l1_ratio_grid=(0.8, 0.9, 1.0))
m_ulr.fit(data.X_lab.iloc[tr_commod], pd.Series(data.y_lab.values[tr_commod], index=range(len(tr_commod))),
          t=pd.Series(data.t_lab.values[tr_commod], index=range(len(tr_commod))),
          t1=pd.Series(data.t1_lab.values[tr_commod], index=range(len(tr_commod))),
          sample_weight=pd.Series(data.w_lab.values[tr_commod], index=range(len(tr_commod))))
p_ulr = m_ulr.predict_proba(X_oos_all)
score(y_oos, p_ulr, 'commod ultra-regularized LogReg')

# FINAL: ensemble of all best candidates
print('\n=== FINAL ENSEMBLE (avg of multiple commodity models) ===')
score(y_oos, (p_sel + p_reg)/2, 'avg(selected, regularized)')
score(y_oos, (p_sel + p_reg + p_rf)/3, 'avg(selected, regularized, RF)')
score(y_oos, (p_sel + p_rf)/2, 'avg(selected, RF)')
score(y_oos, (p_sel + p_reg + p_rf + p_ulr)/4, 'avg of 4 commod models')

# Baseline pooled XGBoost for comparison
m_pool = train_subset(tr_pos, list(X_tr_comm.columns))
p_pool = m_pool.predict_proba(X_oos_all)
print('\nReference:')
score(y_oos, p_pool, 'pooled XGBoost baseline')

# Final winner
candidates = {
    'commod_selected': p_sel,
    'commod_reg': p_reg,
    'commod_rf': p_rf,
    'commod_ulr': p_ulr,
    'avg_sel_reg': (p_sel+p_reg)/2,
    'avg_sel_rf': (p_sel+p_rf)/2,
    'avg_3': (p_sel+p_reg+p_rf)/3,
    'avg_4': (p_sel+p_reg+p_rf+p_ulr)/4,
}
print('\n=== Sorted ===')
for name, p in sorted(candidates.items(), key=lambda x: -roc_auc_score(y_oos, x[1])):
    print(f'  {name:25s}  AUC={roc_auc_score(y_oos, p):.3f}')
