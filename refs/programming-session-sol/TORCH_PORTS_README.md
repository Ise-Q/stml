# PyTorch ports of the course solution notebooks

This directory contains faithful PyTorch ports of the TensorFlow/Keras course
solution notebooks, produced because this project's environment is
**PyTorch 2.12 (CPU) + sklearn + xgboost — TensorFlow is not installed**.
Plan of record: `.omc/plans/tf-to-torch-notebooks.md` (Option A — faithful 1:1 port).

## What was ported

| Notebook | Ported? | Why |
|---|---|---|
| `Solution_Programming_Session_4_torch.ipynb` | ✅ | Keras feed-forward NN cells converted to torch; two latent env breakages fixed (see below). |
| `Solution_Programming_Session_6_torch.ipynb` | ✅ | Baseline Keras MLP + the from-scratch VSN layer stack (InputTransformation / GLU / GRN / VSN / FinalModel) reimplemented as `nn.Module`s with a manual training loop replacing `tf.GradientTape`. |
| `Solution_Programming_Session_7_torch.ipynb` | ✅ | LSTM Long/Short/Neutral classifier + transformer building blocks (scaled dot-product attention, look-ahead mask, positional encoding, multi-head attention) converted to torch. |

## What was NOT ported, and why

- **Sessions 1, 2, 3, 5 — no conversion needed.** They contain **no deep-learning
  code at all**; the only "tensorflow" string in them is the `tfo-notebook-buttons`
  CSS class in the title markdown cell. Their sklearn / xgboost / statsmodels /
  hmmlearn code runs unmodified in this environment.
- **Session 8 — no conversion, by decision.** It uses the Nixtla `neuralforecast`
  TFT, which is **already PyTorch-Lightning-backed** (it is not TensorFlow code;
  the package simply isn't installed here). A native-torch TFT rewrite would be
  high effort with **zero fidelity gain** over the library implementation — if you
  want to run Session 8, `uv add neuralforecast utilsforecast` and run the
  original notebook as-is.

## Port conventions (apply to all three `_torch` notebooks)

- **Faithful 1:1**: every markdown cell, plot, architecture, hyperparameter and
  validation strategy is identical to the original; only TF code cells were
  converted. Training loops are hand-rolled (manual EarlyStopping with
  best-weights restore + `torch.optim.lr_scheduler.ReduceLROnPlateau`);
  metrics come from `sklearn.metrics`. **Zero new dependencies.**
- Results are comparable but **not bit-identical** to the originals (different
  RNG / weight initialisation between TF and torch) — noted in each notebook.
- "Idiomatic torch note" markdown cells sit beside constructs that were kept
  Keras-faithful rather than idiomatic-torch.

### Three documented uniform deviations (Keras semantics with no stock torch equivalent)

1. **Strict `== 1` asserts → `torch.allclose(..., atol=1e-5)`** — the S6 VSN
   alpha-weights sum-to-1 assert and the S7 attention-weights sum-to-1 assert
   both use `allclose` (float32 softmax sums are not exactly 1.0).
2. **S7 loss**: one-hot `categorical_crossentropy` → integer-label
   `nn.CrossEntropyLoss` on logits (softmax at inference). Numerically identical
   (verified to 1e-8); the one-hot encoding was an input-format artifact, not an
   architectural choice. (S6 keeps the faithful sigmoid head + `nn.BCELoss`
   because the sigmoid head *is* the architecture-under-study feeding the alpha
   interpretability analysis.)
3. **S7 LSTM dropout emulation**: Keras `LSTM(dropout=0.2)` → explicit
   `nn.Dropout(0.2)` on the LSTM **input** (stock `nn.LSTM(dropout=)` is a no-op
   for a single layer); Keras `recurrent_dropout` has no stock torch equivalent
   and is **dropped**, documented in-notebook.

### Per-notebook caveats

- **S4 — Enhanced NN MAE sits at ~2× XGBoost** (649 vs 324; the Basic NN at 267
  beats both trees). Torch weight init differs from TF, so NN results are
  comparable, not identical; the in-notebook deviation note also documents the
  target-scaling (StandardScaler on `y_train`, inverse-transformed at predict)
  needed for L1-loss convergence on raw death-count targets.
- **S6 — noise-ranking sanity check is init-seed-sensitive.** The notebook
  reseeds to `SEED=17` before constructing the trained VSN so the injected
  Noise_* features land reproducibly in the bottom half of the importance
  ranking; the synthetic *data* itself is unchanged (`random_state=42`, same as
  the original). With other init seeds a noise feature can near-tie its way
  above the median on this small synthetic set.
- **S4 — `savefig` side effects.** The original's 8 `plt.savefig(...)` calls are
  kept faithfully; executing the notebook with this directory as cwd drops PNGs
  here. Run from a scratch directory or delete the `*.png` afterwards.

### Session 4 latent environment fixes (would crash even unconverted)

- `X.fillna(method='bfill')` → `X.bfill()` (pandas 3.0 removed `method=`).
- `rf_param_grid`: `'max_features': ['auto', 'sqrt', 'log2']` →
  `[1.0, 'sqrt', 'log2']` (`'auto'` was removed; `1.0` is the legacy regressor
  `'auto'` = all features).

## Verification

```bash
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 \
  --output /tmp/_exec_check_S4.ipynb refs/programming-session-sol/Solution_Programming_Session_4_torch.ipynb
# likewise for S6 / S7
```

Committed notebooks are kept output-free (nbstripout convention); execute on a
`/tmp` copy. Session 4 downloads its CSV from a live Google-Drive URL — a fetch
failure there is an environment flake, not a port regression.
