# Multi-task NN — hyperparameter tuning (equity_all pool)

Grid search across four configurations; scored on CPCV(6, 2) mean AUC.

| config | hidden_widths | dropout | lr | n_epochs | mean AUC | ±SEM | n_folds | within 1 SE of best |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| locked | [64,32] | 0.1 | 0.0005 | 100 | 0.5312 | 0.0077 | 14 | ✓ |
| wider | [128,64] | 0.1 | 0.0005 | 100 | 0.5173 | 0.0077 | 14 |  |
| more_dropout | [64,32] | 0.2 | 0.0005 | 100 | 0.5344 | 0.0082 | 14 | ✓ |
| higher_lr | [64,32] | 0.1 | 0.001 | 100 | 0.5320 | 0.0074 | 14 | ✓ |

**Locked configuration:** `hidden_widths=[64, 32], dropout=0.1, lr=5e-4, n_epochs=100`. This is the configuration the champion pipeline runs (no model swap on the deliverable when the NN wins; see §3 of the notebook for the deployment rationale).
