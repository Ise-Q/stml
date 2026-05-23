"""DIAG 4: Run the best-of-per-instrument model and write predictions_v3.csv."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, f1_score
from stml.best_of import build_best_of
from stml.experiments import _build_data
from stml.cv import split_by_boundary

boundary = pd.Timestamp('2022-01-01')
predict_end = pd.Timestamp('2022-07-01')
result = build_best_of(boundary=boundary, predict_end=predict_end, verbose=True)

# Evaluate
data = _build_data(boundary=boundary, predict_end=predict_end)
embargo = pd.Timedelta(days=10)
predict_mask = (data.t_lab.values >= boundary) & (data.t_lab.values < predict_end)
oos_pos = np.where(predict_mask)[0]
y_oos = data.y_lab.iloc[oos_pos].reset_index(drop=True)
p_oos = result.oos_predictions

print()
print('=== Per-instrument CV-AUC table (chosen by inner-CV) ===')
print(result.cv_aucs_per_instrument.to_string())
print()
print('=== Chosen model per instrument ===')
for inst, (choice, auc) in result.chosen_per_instrument.items():
    print(f'  {inst:8s}: {choice:18s} (CV AUC={auc:.3f})')

print()
print('=== Best-of OOS H1-2022 ===')
auc = roc_auc_score(y_oos, p_oos)
f1 = f1_score(y_oos, (p_oos>=0.5).astype(int), zero_division=0)
br = brier_score_loss(y_oos, p_oos)
ll = log_loss(y_oos, np.clip(p_oos, 1e-7, 1-1e-7))
print(f'  AUC: {auc:.3f}')
print(f'  F1: {f1:.3f}')
print(f'  Brier: {br:.3f}')
print(f'  LogLoss: {ll:.3f}')

# Per-instrument
print()
print('=== Best-of per-instrument OOS AUC ===')
events_lab = data.events_lab
for inst in sorted(events_lab['instrument'].unique()):
    inst_oos_idx = np.array([i for i, idx in enumerate(oos_pos) if events_lab.iloc[idx]['instrument']==inst])
    if len(inst_oos_idx) == 0:
        continue
    y_i = y_oos.iloc[inst_oos_idx]
    p_i = p_oos[inst_oos_idx]
    if y_i.nunique()<2:
        print(f'  {inst:8s}: n={len(y_i)} single-class')
        continue
    print(f'  {inst:8s}: n={len(y_i):3d} auc={roc_auc_score(y_i, p_i):.3f}')

# Build predictions_v3.csv
print()
print('=== Writing predictions_v3.csv ===')
# Use the full events_all (incl unlabelable ones at end)
from stml.experiments import ASSET_CLASSES
predict_mask_all = (data.events_all['t'].values >= boundary) & (data.events_all['t'].values < predict_end)
predict_pos_all = np.where(predict_mask_all)[0]
proba_all = np.full(len(predict_pos_all), 0.5)
for j, ev_idx in enumerate(predict_pos_all):
    inst = data.events_all.iloc[ev_idx]['instrument']
    choice, _ = result.chosen_per_instrument.get(inst, ('pooled', None))
    if choice == 'per_instrument' and result.per_instrument_models.get(inst) is not None:
        m = result.per_instrument_models[inst]
    elif choice == 'sector':
        m = result.sector_models.get(ASSET_CLASSES.get(inst), result.pooled_model)
    else:
        m = result.pooled_model
    proba_all[j] = float(m.predict_proba(data.X_all.iloc[[ev_idx]])[0])

# Build the CSV in the required format
if 'date' in data.signals.columns:
    sig_indexed = data.signals.set_index('date')
else:
    sig_indexed = data.signals
instruments = list(sig_indexed.columns)
pred_window = sig_indexed.loc[(sig_indexed.index >= boundary) & (sig_indexed.index < predict_end), instruments]
events_pred = data.events_all.iloc[predict_pos_all]
key_to_proba = pd.Series(
    proba_all,
    index=pd.MultiIndex.from_arrays(
        [events_pred['t'], events_pred['instrument']], names=['date', 'instrument'],
    ),
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
print(f'  Wrote {len(df)} rows.')
print(f'  Mean prob (non-zero): {df[df.prediction>0].prediction.mean():.3f}')
