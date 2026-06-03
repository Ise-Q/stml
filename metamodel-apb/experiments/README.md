# `experiments/` — diagnostic & reproducibility probes

These scripts are **not part of the locked deliverable pipeline.** Each builds a real pooled panel
per asset class and writes a markdown (and sometimes CSV) report to the **gitignored**
`experiments/results/` directory. They feed **nothing** back into `emit` — the shipped deliverable
([`../README.md`](../README.md) §3) is untouched by anything here.

Run one from the repo root:

```bash
uv run --directory metamodel-apb python experiments/<name>.py   # writes experiments/results/<name>.md
```

`experiments/results/` is gitignored, so every artifact below is regenerated on demand.

| Script | Supports | Output(s) in `results/` |
| :---: | :---: | :---: |
| `ex1_edge_decomposition.py` | EX.1 — CPCV path-robustness / edge decomposition | `ex1_edge_decomposition.md` |
| `ex3_barrier_surface.py` | barrier-hyperparameter surface | `ex3_barrier_surface.md` |
| `ex4_calibration.py` | EX.4 — calibration diagnostics (reliability, ECE) | `ex4_calibration.md` |
| `ex5_signal_characterisation.py` | EX.5 — primary-signal ceiling characterisation | `ex5_signal_characterisation.md` |
| `s3_calibration_selected.py` | S3.9 — calibration on the *selected* model | `s3_calibration_selected.md` |
| `s4_cluster_importance.py` | §4 — cluster-level feature importance | `s4_cluster_importance.md` |
| `s6_barrier_backtest.py` | §6 — barrier-exact, cost-aware backtest (+ S5.12 check) | `s6_barrier_backtest.md`, `s6_net_returns.csv`, `EX6_ADOPT_HALT.txt`, `S512_NO_COLLAPSE_HALT.txt` |
| `s6_deflation_gate.py` | S6.8 — deflation gate (DSR / MinBTL / CSCV-PBO) | `s6_deflation_gate.md` |
| `x8_feature_counts.py` | X.8 — per-class feature-family column census | `x8_feature_counts.md` |
| `_common.py` | shared helpers (`results_dir()`, modelling-panel / imputed-X builders) | — (imported, not run) |

The two `*_HALT.txt` files from `s6_barrier_backtest.py` are gate sentinels (EX.6 adopt/halt and the
S5.12 no-collapse check), not reports.
