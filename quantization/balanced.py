from __future__ import annotations

from typing import Literal

import torch

from .ops import ScaleMode, estimate_scale, round_to_zero, round_to_zero_ste

EqualizeMode = Literal["recursive_mean"]


def _equalize_recursive_mean(w: torch.Tensor, k: int, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Algorithm 2 (mean-recursive histogram equalization), implemented efficiently.

    The paper presents Algorithm 2 with recursion + full-size masks. A literal
    implementation does ~O(2^k) full-tensor passes, which becomes impractical for
    higher bitwidths (e.g. k=8).

    This implementation is equivalent but runs in ~O(k) passes:
    - We recursively partition the working set by thresholds T = mean(S_W).
    - Each element accumulates a k-bit prefix code (left=0, right=1).
    - At the leaf (after k splits), we linearly map within each leaf subset to
      [0,1] (min/max detached), producing a "residual" in [0,1).
    - Final equalized value: W_e = (prefix + residual) / 2^k  in [0,1].

    Engineering detail: statistics (mean/min/max) are detached (stop-gradient) to
    reduce instability, similar to TF's stop_gradient practice.
    """

    w_flat = w.reshape(-1)
    n = int(w_flat.numel())
    if n == 0:
        return w

    device = w_flat.device
    dtype = w_flat.dtype

    group_id = torch.zeros(n, device=device, dtype=torch.int64)
    w_det = w_flat.detach()
    ones = torch.ones(n, device=device, dtype=dtype)

    for level in range(int(k)):
        num_groups = 2**level

        sum_g = torch.zeros(num_groups, device=device, dtype=dtype).scatter_add(0, group_id, w_det)
        cnt_g = torch.zeros(num_groups, device=device, dtype=dtype).scatter_add(0, group_id, ones)
        mean_g = (sum_g / cnt_g.clamp_min(1.0)).detach()
        thresh = mean_g[group_id]

        right = w_flat >= thresh
        group_id = group_id * 2 + right.to(dtype=torch.int64)

    num_leaf = 2**int(k)

    min_g = torch.full((num_leaf,), float("inf"), device=device, dtype=dtype).scatter_reduce(
        0, group_id, w_det, reduce="amin", include_self=True
    )
    max_g = torch.full((num_leaf,), float("-inf"), device=device, dtype=dtype).scatter_reduce(
        0, group_id, w_det, reduce="amax", include_self=True
    )
    value_range_g = (max_g - min_g).detach()

    min_per = min_g[group_id].detach()
    range_per = value_range_g[group_id]
    valid_per = torch.isfinite(range_per) & (range_per > eps)
    denom = torch.where(valid_per, range_per, torch.ones_like(range_per))

    residual = ((w_flat - min_per) / denom).clamp(0.0, 1.0)
    residual = torch.where(valid_per, residual, torch.zeros_like(residual))

    w_e = (group_id.to(dtype=dtype) + residual) / float(num_leaf)
    return w_e.reshape_as(w).clamp(0.0, 1.0)


def equalize_k(w: torch.Tensor, k: int, *, mode: EqualizeMode = "recursive_mean") -> torch.Tensor:
    if k >= 32:
        raise ValueError("equalize_k expects k in {2,3,4,8} (bits), not 32.")
    if mode != "recursive_mean":
        raise ValueError(f"Unsupported equalize mode: {mode}")
    return _equalize_recursive_mean(w, k)


def balanced_quantize_weight(
    w: torch.Tensor,
    k: int,
    *,
    scale_mode: ScaleMode = "maxabs",
    equalize_mode: EqualizeMode = "recursive_mean",
    ste: bool = True,
) -> torch.Tensor:
    """
    Algorithm 1: k-bit Balanced Quantization for weights.

    scale = max(|W|) (default) or mean(|W|)*2.5 (engineering option), detached.
    W_e = equalize_k(W) in [0,1]
    W_f = 1/(2^k-1) * round-to-zero(2^k * W_e - 1/2) - 1/2
    W_q = 2 * scale * W_f    in [-scale, +scale]
    """

    if k >= 32:
        return w

    scale = estimate_scale(w, scale_mode)
    if scale_mode == "meanabs2.5":
        # Optional outlier suppression for the mean-based scale estimate.
        w = w.clamp(min=-scale, max=scale)

    w_e = equalize_k(w, k, mode=equalize_mode).clamp(0.0, 1.0)
    n = 2**k
    rtz = round_to_zero_ste if ste else round_to_zero
    w_f = rtz(w_e * float(n) - 0.5).div(float(n - 1)).sub(0.5)
    w_q = w_f.mul(2.0).mul(scale)
    return w_q
