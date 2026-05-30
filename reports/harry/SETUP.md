# Setup — Harry's Branch

Harry's contribution mostly works with the same environment everyone else uses
(``uv sync`` from the repo root). One module needs an extra wheel.

## Wavelet feature module

``src/stml/harry/features/wavelet.py`` requires
[PyWavelets](https://pywavelets.readthedocs.io/) for the MODWT-style energy
bands. PyWavelets is declared as an optional extra so that teammates who
never run Harry's pipeline don't have to install it.

```bash
uv sync --extra harry-features
```

After ``uv sync --extra harry-features``, ``import pywt`` works in the
project venv and the wavelet module loads. Without it, importing the
module raises a clean ``ImportError`` that points back here.

## Verifying Harry's tests after setup

```bash
uv run pytest tests/harry/ -q
```

Should print ``N passed`` (no skips other than the legitimate skip when
``pywt`` is absent). To run only the wavelet tests:

```bash
uv run pytest tests/harry/test_wavelet.py -v
```

## What's NOT a setup requirement (yet)

* **ripser / TDA** — Harry's optional ``tda.py`` module is deferred. It
  has C++ build dependencies that can blow up on some platforms, so it
  is not added to the optional extras until it ships.
* **xgboost / hmmlearn / lightgbm** — Sreeram's branch carries these.
  Harry's pipeline (Step 4) will use scikit-learn's
  ``HistGradientBoostingClassifier`` and a re-implemented VSN in
  PyTorch, so the wider extras stay opt-in.
