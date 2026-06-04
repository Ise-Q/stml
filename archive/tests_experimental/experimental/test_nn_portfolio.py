"""Tests for ``stml.experimental.nn_portfolio`` -- Madmoun §4 recipe."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

torch = pytest.importorskip("torch")

from stml.experimental.nn_portfolio import (
    LinearBackbone, LSTMBackbone, VLSTMBackbone, TFTBackbone,
    PortfolioModel, PortfolioPanel, TrainConfig,
    aggregate_portfolio, build_portfolio_model, chronological_split,
    predict_weights, sharpe_loss, train_portfolio_model,
)
from stml.experimental.nn_portfolio import (
    _GatedLinearUnit, _GatedResidualNetwork, _PerStepVSN,
    _InterpretableMultiHeadAttention,
)


# ---------------------------------------------------------------------------
# Sharpe-loss + aggregation primitives (slides 41-42).
# ---------------------------------------------------------------------------


def test_sharpe_loss_sign_and_value_known():
    """Negative-Sharpe loss on a known-stable series."""
    # Deterministic: constant per-day returns → SR = mean / std = inf, so use
    # a tiny noise. Sharpe is sensitive to small-sample mean/std jitter; we
    # only check sign and approximate magnitude.
    rng = np.random.default_rng(0)
    r = 0.001 + 0.01 * rng.standard_normal(5000)
    t = torch.tensor(r, dtype=torch.float32)
    loss = sharpe_loss(t).item()
    # Loss = -SR; SR > 0 expected (positive drift); loss < 0.
    assert loss < 0
    # Annualised SR should be in a wide reasonable band.
    sr = -loss
    assert 0.5 < sr < 3.0


def test_aggregate_portfolio_one_over_k():
    """(1/K_active) aggregation -- slide 41."""
    # 2 assets, 3 days: day 0 only asset A active; day 1 both; day 2 neither.
    W = torch.tensor([[0.5, 0.0], [0.3, 0.4], [0.0, 0.0]])
    R = torch.tensor([[0.01, 0.02], [-0.01, 0.03], [0.05, 0.05]])
    port = aggregate_portfolio(W, R).detach().tolist()
    # Day 0: K_active = 1, sum = 0.5*0.01 = 0.005, R = 0.005.
    # Day 1: K_active = 2, sum = -0.003 + 0.012 = 0.009, R = 0.0045.
    # Day 2: K_active = 0 -> 0.
    expected = np.array([0.005, 0.0045, 0.0])
    np.testing.assert_allclose(port, expected, atol=1e-6)


def test_aggregate_handles_no_active_day():
    """Days with K_active=0 must produce 0, not nan/inf."""
    W = torch.zeros((5, 3))
    R = torch.tensor([[0.01]*3]*5)
    port = aggregate_portfolio(W, R).detach().tolist()
    np.testing.assert_array_equal(port, np.zeros(5))


# ---------------------------------------------------------------------------
# Backbones produce correct output shapes.
# ---------------------------------------------------------------------------


def test_linear_backbone_shape():
    bb = LinearBackbone(lookback=20, n_channels=4, hidden_dim=8)
    x = torch.randn(7, 20, 4)
    out = bb(x)
    assert out.shape == (7, 8)


def test_lstm_backbone_shape():
    bb = LSTMBackbone(lookback=20, n_channels=4, hidden_dim=8)
    x = torch.randn(7, 20, 4)
    out = bb(x)
    assert out.shape == (7, 8)


def test_tft_backbone_shape_and_interpretability_hooks():
    """TFT output shape + VSN channel weights + attention pattern populated."""
    from stml.experimental.nn_portfolio import TFTBackbone
    bb = TFTBackbone(lookback=12, n_channels=5, hidden_dim=8, n_heads=2)
    x = torch.randn(4, 12, 5)
    out = bb(x)
    assert out.shape == (4, 8)
    # VSN per-step weights: (B, L, C), softmax over C.
    assert bb.last_channel_weights is not None
    assert bb.last_channel_weights.shape == (4, 12, 5)
    sums = bb.last_channel_weights.sum(dim=2)
    np.testing.assert_allclose(
        np.asarray(sums.detach().tolist()), np.ones((4, 12)), atol=1e-5,
    )
    # Attention: averaged across heads -> (B, L, L) with causal mask.
    assert bb.last_attention is not None
    assert bb.last_attention.shape == (4, 12, 12)
    # Causal mask: position t can only attend to ≤ t, so the upper triangle
    # (k > t) must be ~0 (modulo numerical floor of the softmax).
    upper = torch.triu(bb.last_attention, diagonal=1)
    assert upper.abs().max().item() < 1e-5


def test_tft_gradient_flow():
    """Loss backprop touches every parameter group in the TFT."""
    from stml.experimental.nn_portfolio import TFTBackbone
    bb = TFTBackbone(lookback=8, n_channels=4, hidden_dim=8, n_heads=2)
    head = torch.nn.Linear(8, 1)
    x = torch.randn(3, 8, 4)
    y = torch.tanh(head(bb(x)).squeeze(-1))
    target = torch.tensor([0.5, -0.3, 0.2])
    loss = ((y - target) ** 2).mean()
    loss.backward()
    # Every parameter should have a non-None gradient with non-zero norm
    # somewhere -- otherwise that branch is dead.
    for name, p in bb.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum().item() > 0, f"zero grad for {name}"


def test_grn_gate_closed_passes_residual_through():
    """When the GLU gate inside the GRN is fully closed, output = LayerNorm(residual)."""
    from stml.experimental.nn_portfolio import _GatedResidualNetwork
    torch.manual_seed(0)
    grn = _GatedResidualNetwork(input_dim=4, hidden_dim=4, output_dim=4)
    # Force the gate to ~0 by setting its weights & bias to very negative.
    grn.glu.gate.weight.data.fill_(-1e3)
    grn.glu.gate.bias.data.fill_(-1e3)
    x = torch.randn(5, 4)
    out = grn(x)
    expected = grn.norm(x)  # residual through, GLU contribution ≈ 0.
    np.testing.assert_allclose(
        np.asarray(out.detach().tolist()),
        np.asarray(expected.detach().tolist()),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# TFT (Temporal Fusion Transformer) — Lim et al. 2021 building blocks.
# ---------------------------------------------------------------------------


def test_glu_gate_can_skip_residual():
    """GLU(x) = (W_1 x) ⊙ σ(W_2 x). When the gate bias is very negative, the
    output should be ~0 -- the skip-residual mechanism in §5.1."""
    glu = _GatedLinearUnit(dim=4)
    # Set gate bias to a very negative value -> σ ≈ 0 -> output ≈ 0.
    with torch.no_grad():
        glu.gate.bias.fill_(-20.0)
    out = glu(torch.randn(3, 4))
    assert out.abs().max().item() < 1e-4


def test_grn_layernorm_output_shape():
    """GRN with input_dim != output_dim must project residual via skip-projection."""
    grn = _GatedResidualNetwork(input_dim=5, hidden_dim=8, output_dim=3)
    out = grn(torch.randn(7, 5))
    assert out.shape == (7, 3)
    # With context.
    grn_c = _GatedResidualNetwork(
        input_dim=5, hidden_dim=8, output_dim=3, context_dim=4,
    )
    out_c = grn_c(torch.randn(7, 5), torch.randn(7, 4))
    assert out_c.shape == (7, 3)


def test_per_step_vsn_softmax_weights():
    """Per-step VSN must produce softmax weights summing to 1 at every step."""
    vsn = _PerStepVSN(n_channels=4, embed_dim=8)
    x = torch.randn(3, 10, 4)
    selected, weights = vsn(x)
    assert selected.shape == (3, 10, 8)
    assert weights.shape == (3, 10, 4)
    sums = weights.sum(dim=-1).detach().tolist()
    sums = np.asarray(sums)
    np.testing.assert_allclose(sums, np.ones((3, 10)), atol=1e-5)


def test_interpretable_attention_causality_and_shape():
    """Multi-head attention must be causal (no future leakage) and return
    averaged-across-heads attention weights as the interpretability hook."""
    attn = _InterpretableMultiHeadAttention(d_model=8, n_heads=2)
    x = torch.randn(3, 6, 8)
    out, weights = attn(x)
    assert out.shape == (3, 6, 8)
    assert weights.shape == (3, 6, 6)
    # Causality: weights[t, u] should be 0 for u > t (strict upper triangle).
    w0 = weights[0].detach().tolist()
    w0 = np.asarray(w0)
    upper = np.triu(np.ones_like(w0), k=1).astype(bool)
    assert np.all(np.abs(w0[upper]) < 1e-6)
    # Rows still sum to 1 (softmax along last dim).
    row_sums = w0.sum(axis=-1)
    np.testing.assert_allclose(row_sums, np.ones(6), atol=1e-5)


def test_tft_backbone_shape_and_interpretability_hooks():
    """TFT backbone: output shape correct, interpretability hooks populated."""
    bb = TFTBackbone(lookback=10, n_channels=5, hidden_dim=8, n_heads=2)
    x = torch.randn(4, 10, 5)
    out = bb(x)
    assert out.shape == (4, 8)
    # Hooks populated after forward.
    assert bb.last_channel_weights is not None
    assert bb.last_channel_weights.shape == (4, 10, 5)
    assert bb.last_attention is not None
    assert bb.last_attention.shape == (4, 10, 10)
    # Channel weights are softmax → rows sum to 1 at every (sample, step).
    sums = np.asarray(bb.last_channel_weights.sum(dim=-1).tolist())
    np.testing.assert_allclose(sums, np.ones((4, 10)), atol=1e-5)


def test_tft_backbone_rounds_hidden_to_n_heads_multiple():
    """If hidden_dim isn't a multiple of n_heads, TFT rounds up."""
    bb = TFTBackbone(lookback=8, n_channels=3, hidden_dim=7, n_heads=4)
    # 7 isn't a multiple of 4; should round up to 8.
    assert bb.H == 8
    out = bb(torch.randn(2, 8, 3))
    assert out.shape == (2, 8)


def test_tft_via_build_portfolio_model():
    """``build_portfolio_model('tft', ...)`` returns a PortfolioModel."""
    model = build_portfolio_model(
        "tft", lookback=8, n_channels=4, hidden_dim=8, seed=42,
    )
    assert isinstance(model.backbone, TFTBackbone)
    x = torch.randn(3, 8, 4)
    sigma = torch.tensor([0.01, 0.02, 0.015])
    y, w = model(x, sigma)
    assert y.shape == (3,)
    assert w.shape == (3,)


def test_tft_loss_decreases_during_training():
    """Verify TFT is trainable: Sharpe loss decreases over a few epochs on a
    toy panel with a planted signal. (Note: GLU-gated residuals can have some
    parameters at zero gradient at step 0 — initialisation effect — but the
    network as a whole must still be trainable.)"""
    panel = _make_panel(seed=3)
    train, val = chronological_split(panel, val_frac=0.20)
    model = build_portfolio_model(
        "tft", lookback=panel.L, n_channels=panel.C,
        hidden_dim=8, seed=42,
    )
    cfg = TrainConfig(epochs=20, patience=20, lr=5e-3, verbose=False)
    out = train_portfolio_model(model, train, val, cfg=cfg)
    # Sharpe should improve from random init across 20 epochs on toy data.
    history = out["history"]
    initial_train_sr = float(history["train_sharpe"].iloc[0])
    final_train_sr = float(history["train_sharpe"].iloc[-1])
    assert final_train_sr > initial_train_sr, (
        f"TFT did not train: SR went {initial_train_sr:+.3f} -> {final_train_sr:+.3f}"
    )


def test_vlstm_backbone_shape_and_channel_weights():
    bb = VLSTMBackbone(lookback=20, n_channels=4, hidden_dim=8, embed_dim=6)
    x = torch.randn(7, 20, 4)
    out = bb(x)
    assert out.shape == (7, 8)
    assert bb.last_channel_weights is not None
    assert bb.last_channel_weights.shape == (7, 4)
    # Channel weights are a softmax → rows sum to 1.
    sums = np.asarray(bb.last_channel_weights.sum(dim=1).detach().tolist())
    np.testing.assert_allclose(sums, np.ones(7), atol=1e-5)


# ---------------------------------------------------------------------------
# Vol-targeting layer (slide 40).
# ---------------------------------------------------------------------------


def test_portfolio_model_vol_target_formula():
    """Weight = ŷ · σ_tgt / (σ_daily · √252), clipped to max_leverage."""
    torch.manual_seed(0)
    bb = LinearBackbone(lookback=4, n_channels=2, hidden_dim=4)
    model = PortfolioModel(bb, target_vol=0.10, max_leverage=10.0)
    # Force tanh head to roughly identity by zero-init bias.
    model.head.weight.data.zero_()
    model.head.bias.data[:] = 0.5  # tanh(0.5) ≈ 0.4621.
    x = torch.randn(3, 4, 2)
    sigma = torch.tensor([0.01, 0.02, 0.0126])  # daily.
    y, w = model(x, sigma)
    # tanh(0.5) on every sample.
    expected_y = np.tanh(0.5)
    np.testing.assert_allclose(np.asarray(y.detach().tolist()), expected_y, atol=1e-6)
    # Leverage per sample.
    sigma_np = np.asarray(sigma.tolist())
    lev = np.minimum(0.10 / (sigma_np * np.sqrt(252)), 10.0)
    expected_w = expected_y * lev
    np.testing.assert_allclose(np.asarray(w.detach().tolist()), expected_w, atol=1e-5)


def test_portfolio_model_zero_weight_on_nan_sigma():
    """NaN / zero σ̂ → zero weight (no NaN propagation)."""
    bb = LinearBackbone(lookback=4, n_channels=2, hidden_dim=4)
    model = PortfolioModel(bb, target_vol=0.10)
    x = torch.randn(3, 4, 2)
    sigma = torch.tensor([float("nan"), 0.0, 0.01])
    _y, w = model(x, sigma)
    w_np = np.asarray(w.detach().tolist())
    assert w_np[0] == 0.0
    assert w_np[1] == 0.0
    assert w_np[2] != 0.0


# ---------------------------------------------------------------------------
# Training overfit on a toy panel — Sharpe should rise.
# ---------------------------------------------------------------------------


def _make_panel(T=120, K=3, L=8, C=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, K, L, C)).astype(np.float32)
    sigma = (0.01 + 0.002 * np.abs(rng.standard_normal((T, K)))).astype(np.float32)
    r_next = 0.001 + 0.005 * rng.standard_normal((T, K)).astype(np.float32)
    # Embed a strong signal: r_{t+1, k} ∝ X[t, k, -1, 0].
    for k in range(K):
        r_next[:, k] += 0.02 * X[:, k, -1, 0]
    mask = np.ones((T, K), dtype=bool)
    mask[:L - 1] = False
    return PortfolioPanel(
        dates=pd.date_range("2020-01-01", periods=T),
        instruments=[f"a{i}" for i in range(K)],
        X=X, sigma_daily=sigma, r_next=r_next, mask=mask,
        feature_names=[f"c{i}" for i in range(C)],
    )


def test_train_overfit_linear():
    panel = _make_panel(seed=0)
    train, val = chronological_split(panel, val_frac=0.20)
    model = build_portfolio_model("linear", lookback=panel.L,
                                    n_channels=panel.C, hidden_dim=16, seed=42)
    cfg = TrainConfig(epochs=80, patience=80, lr=5e-3, verbose=False)
    out = train_portfolio_model(model, train, val, cfg=cfg)
    # On a toy with a strong signal, val Sharpe should clear 1 within 80 epochs.
    assert out["best_val_sharpe"] > 1.0


def test_predict_weights_shape():
    panel = _make_panel(seed=1)
    model = build_portfolio_model("linear", lookback=panel.L,
                                    n_channels=panel.C, hidden_dim=8, seed=0)
    w = predict_weights(model, panel)
    assert w.shape == (panel.T, panel.K)
    assert (w.columns == panel.instruments).all()


def test_deterministic_with_seed():
    """Same seed → bitwise-identical weights for the same panel."""
    panel = _make_panel(seed=2)
    m1 = build_portfolio_model("lstm", lookback=panel.L,
                                 n_channels=panel.C, hidden_dim=8, seed=123)
    m2 = build_portfolio_model("lstm", lookback=panel.L,
                                 n_channels=panel.C, hidden_dim=8, seed=123)
    w1 = predict_weights(m1, panel).values
    w2 = predict_weights(m2, panel).values
    np.testing.assert_allclose(w1, w2, atol=1e-9)


# ---------------------------------------------------------------------------
# Sharpe gradient sign.
# ---------------------------------------------------------------------------


def test_sharpe_loss_gradient_pushes_weights_in_the_right_direction():
    """If R = w * r and r > 0, ∂Sharpe/∂w should be positive."""
    w = torch.tensor([0.5], requires_grad=True)
    r = torch.tensor([0.001 * i for i in range(1, 101)])
    R = w * r
    loss = sharpe_loss(R)
    loss.backward()
    # Loss = -Sharpe, so to MINIMISE loss we move w in the −grad direction.
    # ∂Sharpe/∂w > 0 means ∂loss/∂w < 0; gradient on w should be NEGATIVE.
    assert w.grad.item() < 0.0
