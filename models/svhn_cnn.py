from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.balanced import EqualizeMode, balanced_quantize_weight
from quantization.ops import ScaleMode, quantize_w_bitutils, uniform_quantize_activation, uniform_symmetric_quantize_weight


QuantMode = Literal["none", "balanced", "uniform"]


@dataclass(frozen=True)
class QuantConfig:
    quant: QuantMode = "none"
    w_bits: int = 32
    a_bits: int = 32
    equalize: EqualizeMode = "recursive_mean"
    scale_mode: ScaleMode = "maxabs"
    fp32_first_last: bool = False


class QuantActivation(nn.Module):
    def __init__(self, a_bits: int) -> None:
        super().__init__()
        self.a_bits = int(a_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return uniform_quantize_activation(x, self.a_bits, ste=True)


class QuantConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        quant: QuantMode,
        w_bits: int,
        equalize: EqualizeMode,
        scale_mode: ScaleMode,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.quant = quant
        self.w_bits = int(w_bits)
        self.equalize = equalize
        self.scale_mode = scale_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w = self.weight
        if self.quant == "balanced":
            w = balanced_quantize_weight(w, self.w_bits, scale_mode=self.scale_mode, equalize_mode=self.equalize, ste=True)
        elif self.quant == "uniform":
            if self.scale_mode == "meanabs2.5":
                # Align with bit-rnn: quantize_w(tanh(W)) with meanabs*2.5 scale + STE for clip/round.
                w = quantize_w_bitutils(torch.tanh(w), self.w_bits)
            else:
                w = uniform_symmetric_quantize_weight(w, self.w_bits, scale_mode=self.scale_mode, ste=True)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    def __init__(
        self,
        *args,
        quant: QuantMode,
        w_bits: int,
        equalize: EqualizeMode,
        scale_mode: ScaleMode,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.quant = quant
        self.w_bits = int(w_bits)
        self.equalize = equalize
        self.scale_mode = scale_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w = self.weight
        if self.quant == "balanced":
            w = balanced_quantize_weight(w, self.w_bits, scale_mode=self.scale_mode, equalize_mode=self.equalize, ste=True)
        elif self.quant == "uniform":
            if self.scale_mode == "meanabs2.5":
                w = quantize_w_bitutils(torch.tanh(w), self.w_bits)
            else:
                w = uniform_symmetric_quantize_weight(w, self.w_bits, scale_mode=self.scale_mode, ste=True)
        return F.linear(x, w, self.bias)


def _conv(
    in_ch: int,
    out_ch: int,
    cfg: QuantConfig,
    *,
    kernel_size: int = 3,
    padding: int = 1,
    quant_override: QuantMode | None = None,
    w_bits_override: int | None = None,
) -> QuantConv2d:
    return QuantConv2d(
        in_ch,
        out_ch,
        kernel_size=kernel_size,
        padding=padding,
        bias=True,
        quant=quant_override or cfg.quant,
        w_bits=w_bits_override if w_bits_override is not None else cfg.w_bits,
        equalize=cfg.equalize,
        scale_mode=cfg.scale_mode,
    )


def _linear(
    in_dim: int,
    out_dim: int,
    cfg: QuantConfig,
    *,
    quant_override: QuantMode | None = None,
    w_bits_override: int | None = None,
) -> QuantLinear:
    return QuantLinear(
        in_dim,
        out_dim,
        bias=True,
        quant=quant_override or cfg.quant,
        w_bits=w_bits_override if w_bits_override is not None else cfg.w_bits,
        equalize=cfg.equalize,
        scale_mode=cfg.scale_mode,
    )


class SVHNCNN(nn.Module):
    def __init__(self, cfg: QuantConfig, *, num_classes: int = 10) -> None:
        super().__init__()
        self.cfg = cfg
        qa = QuantActivation(cfg.a_bits) if cfg.a_bits < 32 else nn.Identity()

        first_quant = "none" if cfg.fp32_first_last else cfg.quant
        first_w_bits = 32 if cfg.fp32_first_last else cfg.w_bits
        last_quant = "none" if cfg.fp32_first_last else cfg.quant
        last_w_bits = 32 if cfg.fp32_first_last else cfg.w_bits

        self.conv1 = _conv(3, 64, cfg, quant_override=first_quant, w_bits_override=first_w_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = _conv(64, 64, cfg)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = _conv(64, 128, cfg)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = _conv(128, 128, cfg)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = _conv(128, 256, cfg)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = _conv(256, 256, cfg)
        self.bn6 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = _linear(256 * 4 * 4, 512, cfg)
        self.fc2 = _linear(512, int(num_classes), cfg, quant_override=last_quant, w_bits_override=last_w_bits)

        self.qa = qa

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.pool(self.qa(F.relu(self.bn2(self.conv2(self.qa(F.relu(self.bn1(self.conv1(x)))))))))
        x = self.pool(self.qa(F.relu(self.bn4(self.conv4(self.qa(F.relu(self.bn3(self.conv3(x)))))))))
        x = self.pool(self.qa(F.relu(self.bn6(self.conv6(self.qa(F.relu(self.bn5(self.conv5(x)))))))))

        x = torch.flatten(x, 1)
        x = self.qa(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
