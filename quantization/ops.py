from __future__ import annotations

from typing import Literal

import torch

ScaleMode = Literal["maxabs", "meanabs2.5"]


def round_to_zero(x: torch.Tensor) -> torch.Tensor:
    """
    Paper "round half towards zero" tie-breaking rule.

    round-to-zero(x) := sgn(x) * ceil(|x| - 1/2)

    This differs from torch.round (ties-to-even) and is required by the BQ paper.
    """

    return torch.sign(x) * torch.ceil(torch.abs(x) - 0.5)


class RoundToZeroSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return round_to_zero(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Straight-Through Estimator (STE): pass gradient through.
        return grad_output


def round_to_zero_ste(x: torch.Tensor) -> torch.Tensor:
    return RoundToZeroSTE.apply(x)


def round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """
    TensorFlow-like rounding used in bit-rnn's `bit_utils.py`:
    round half away from zero.

    tf.round( 0.5) ->  1
    tf.round(-0.5) -> -1
    """

    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


class RoundHalfAwayFromZeroSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return round_half_away_from_zero(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad_output


def round_half_away_from_zero_ste(x: torch.Tensor) -> torch.Tensor:
    return RoundHalfAwayFromZeroSTE.apply(x)


def round_bit(x: torch.Tensor, bit: int) -> torch.Tensor:
    """
    bit-rnn `round_bit`: tf.round(x * k) / k with STE in training.
    Here we provide the pure forward (no STE) variant.
    """

    if bit >= 32:
        return x
    k = float((2**bit) - 1)
    return round_half_away_from_zero(x * k) / k


def round_bit_ste(x: torch.Tensor, bit: int) -> torch.Tensor:
    if bit >= 32:
        return x
    k = float((2**bit) - 1)
    return round_half_away_from_zero_ste(x * k) / k


def clip_by_value_ste(x: torch.Tensor, minv: float, maxv: float) -> torch.Tensor:
    """
    Straight-through clip (bit-rnn uses TF gradient overrides for Minimum/Maximum
    inside `clip_by_value` so the gradient w.r.t. x is identity).
    """

    clipped = x.clamp(minv, maxv)
    return x + (clipped - x).detach()


def qk(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    k-bit uniform quantizer Q_k from the paper.
    Input must be in [0, 1].

    Q_k(x) = round-to-zero((2^k - 1) * x) / (2^k - 1)
    """

    if k >= 32:
        return x
    n = (2**k) - 1
    x = x.clamp(0.0, 1.0)
    return round_to_zero(x * float(n)) / float(n)


def qk_ste(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= 32:
        return x
    n = (2**k) - 1
    x = x.clamp(0.0, 1.0)
    return round_to_zero_ste(x * float(n)) / float(n)


def estimate_scale(w: torch.Tensor, scale_mode: ScaleMode, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Estimate quantization scale with stop-gradient (detach).

    - maxabs: scale = max(|W|)                 (paper default)
    - meanabs2.5: scale = mean(|W|) * 2.5      (engineering trick from bit_utils.py)
    """

    if scale_mode == "maxabs":
        scale = w.detach().abs().max()
    elif scale_mode == "meanabs2.5":
        scale = w.detach().abs().mean() * 2.5
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode!r}")

    return scale.clamp_min(eps)


def quantize_w_bitutils(x: torch.Tensor, bit: int) -> torch.Tensor:
    """
    Strict translation of bit-rnn `bit_utils.quantize_w`:

    scale = stop_gradient(mean(abs(x)) * 2.5)
    y = (round_bit(clip(x/scale, -0.5, 0.5) + 0.5) - 0.5) * scale

    Notes:
    - `scale` is detached (stop-gradient).
    - `clip` uses a straight-through gradient (identity), matching TF's custom
      gradients for Minimum/Maximum.
    - `round_bit` uses round-half-away-from-zero with STE, matching tf.round +
      gradient override.
    """

    if bit >= 32:
        return x
    scale = (x.detach().abs().mean() * 2.5).clamp_min(1e-8)
    u = clip_by_value_ste(x / scale, -0.5, 0.5) + 0.5
    return (round_bit_ste(u, bit) - 0.5) * scale


def uniform_symmetric_quantize_weight(
    w: torch.Tensor,
    k: int,
    *,
    scale_mode: ScaleMode = "maxabs",
    ste: bool = True,
) -> torch.Tensor:
    """
    Symmetric uniform quantization baseline (paper-style), kept for reference.

    The default `--quant uniform` path uses `quantize_w_bitutils` in the model,
    which matches the bit-rnn reference implementation more closely.
    """

    if k >= 32:
        return w

    scale = estimate_scale(w, scale_mode)
    if scale_mode == "meanabs2.5":
        w = w.clamp(min=-scale, max=scale)

    x = (w / scale).clamp(-1.0, 1.0)
    u = (x + 1.0) * 0.5  # [-1,1] -> [0,1]
    q = qk_ste(u, k) if ste else qk(u, k)
    xq = q * 2.0 - 1.0
    return xq * scale


def uniform_quantize_activation(
    x: torch.Tensor,
    k: int,
    *,
    ste: bool = True,
) -> torch.Tensor:
    """
    Optional activation quantization.

    bit-rnn style: `round_bit(relu(x), bit)` (no clipping), with STE.
    """

    if k >= 32:
        return x
    return round_bit_ste(x, k) if ste else round_bit(x, k)
