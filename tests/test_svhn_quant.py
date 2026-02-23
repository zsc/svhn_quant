from __future__ import annotations

import os

import numpy as np
import torch

from datasets.svhn_mat import SVHNMatDataset, SVHNTransformConfig
from models.svhn_cnn import QuantConfig
from models.svhn_vit import SVHNViT
from quantization.balanced import balanced_quantize_weight, equalize_k
from quantization.ops import quantize_w_bitutils, round_half_away_from_zero, round_to_zero


def test_svhn_mat_loader_basic() -> None:
    if not os.path.exists("train_32x32.mat"):
        raise RuntimeError("Expected train_32x32.mat in repo root for this test.")

    tf = SVHNTransformConfig(normalize=True, random_crop=False, horizontal_flip=False)
    ds = SVHNMatDataset(".", "train", transform=tf, train=False, max_samples=64)

    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert x.dtype == torch.float32
    assert 0 <= y <= 9

    labels = ds.labels
    assert isinstance(labels, np.ndarray)
    assert labels.min() >= 0
    assert labels.max() <= 9
    assert not np.any(labels == 10), "SVHN label 10 must be mapped to 0"


def test_round_to_zero_ties() -> None:
    x = torch.tensor([1.5, -1.5, 0.5, -0.5, 2.5, -2.5])
    y = round_to_zero(x)
    assert torch.equal(y, torch.tensor([1.0, -1.0, 0.0, -0.0, 2.0, -2.0]))


def test_round_half_away_from_zero_ties() -> None:
    x = torch.tensor([1.5, -1.5, 0.5, -0.5, 2.5, -2.5])
    y = round_half_away_from_zero(x)
    assert torch.equal(y, torch.tensor([2.0, -2.0, 1.0, -1.0, 3.0, -3.0]))


def test_equalize_k_range() -> None:
    w = torch.randn(256, 64)
    for k in (4, 8):
        we = equalize_k(w, k, mode="recursive_mean")
        assert torch.isfinite(we).all()
        assert float(we.min().item()) >= -1e-6
        assert float(we.max().item()) <= 1.0 + 1e-6


def test_balanced_quantize_weight_discrete_and_backward() -> None:
    torch.manual_seed(0)
    w = torch.randn(128, 128, requires_grad=True)
    wq = balanced_quantize_weight(w, 4, scale_mode="maxabs", equalize_mode="recursive_mean", ste=True)

    scale = w.detach().abs().max().clamp_min(1e-8)
    assert float(wq.abs().max().item()) <= float(scale.item()) + 1e-4

    uniq = torch.unique(wq.detach().cpu())
    assert uniq.numel() <= 2**4

    loss = wq.pow(2).mean()
    loss.backward()
    assert w.grad is not None
    assert w.grad.shape == w.shape


def test_balanced_quantize_weight_1bit_discrete_and_backward() -> None:
    torch.manual_seed(0)
    w = torch.randn(128, 128, requires_grad=True)
    wq = balanced_quantize_weight(w, 1, scale_mode="maxabs", equalize_mode="recursive_mean", ste=True)

    scale = w.detach().abs().max().clamp_min(1e-8)
    assert float(wq.abs().max().item()) <= float(scale.item()) + 1e-4

    uniq = torch.unique(wq.detach().cpu())
    assert uniq.numel() <= 2

    loss = wq.pow(2).mean()
    loss.backward()
    assert w.grad is not None
    assert w.grad.shape == w.shape


def test_quantize_w_bitutils_discrete_and_backward() -> None:
    torch.manual_seed(0)
    w = torch.randn(256, 256, requires_grad=True)
    x = torch.tanh(w)
    wq = quantize_w_bitutils(x, 4)

    uniq = torch.unique(wq.detach().cpu())
    assert uniq.numel() <= 2**4

    loss = wq.pow(2).mean()
    loss.backward()
    assert w.grad is not None


def test_vit_forward_backward_smoke() -> None:
    cfg = QuantConfig(quant="none", w_bits=32, a_bits=32)
    model = SVHNViT(cfg, patch_size=8, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
    y.mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
