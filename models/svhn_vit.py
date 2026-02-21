from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.svhn_cnn import QuantActivation, QuantConfig, QuantConv2d, QuantLinear


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, cfg: QuantConfig, *, drop: float) -> None:
        super().__init__()
        self.fc1 = QuantLinear(dim, hidden_dim, quant=cfg.quant, w_bits=cfg.w_bits, equalize=cfg.equalize, scale_mode=cfg.scale_mode)
        self.fc2 = QuantLinear(hidden_dim, dim, quant=cfg.quant, w_bits=cfg.w_bits, equalize=cfg.equalize, scale_mode=cfg.scale_mode)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, cfg: QuantConfig, *, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.scale = float(self.head_dim**-0.5)

        self.qkv = QuantLinear(dim, dim * 3, quant=cfg.quant, w_bits=cfg.w_bits, equalize=cfg.equalize, scale_mode=cfg.scale_mode)
        self.proj = QuantLinear(dim, dim, quant=cfg.quant, w_bits=cfg.w_bits, equalize=cfg.equalize, scale_mode=cfg.scale_mode)
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj_drop = nn.Dropout(float(proj_drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cfg: QuantConfig,
        *,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.qa = QuantActivation(cfg.a_bits) if cfg.a_bits < 32 else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, cfg, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(round(dim * float(mlp_ratio)))
        self.mlp = MLP(dim, hidden, cfg, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x + self.qa(self.attn(self.norm1(x)))
        x = x + self.qa(self.mlp(self.norm2(x)))
        return x


class SVHNViT(nn.Module):
    """
    Minimal ViT for SVHN (32x32) with 8x8 patches by default (16 tokens).

    Patch embedding uses a Conv2d with kernel_size=stride=patch_size.
    """

    def __init__(
        self,
        cfg: QuantConfig,
        *,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        patch_norm: bool = False,
        pool: Literal["cls", "mean"] = "cls",
        drop: float = 0.0,
        attn_drop: float = 0.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.qa = QuantActivation(cfg.a_bits) if cfg.a_bits < 32 else nn.Identity()

        patch = int(patch_size)
        if 32 % patch != 0:
            raise ValueError(f"patch_size must divide 32, got patch_size={patch}")
        grid = 32 // patch
        num_patches = grid * grid

        first_quant = "none" if cfg.fp32_first_last else cfg.quant
        first_w_bits = 32 if cfg.fp32_first_last else cfg.w_bits
        last_quant = "none" if cfg.fp32_first_last else cfg.quant
        last_w_bits = 32 if cfg.fp32_first_last else cfg.w_bits

        self.patch_embed = QuantConv2d(
            3,
            int(embed_dim),
            kernel_size=patch,
            stride=patch,
            padding=0,
            bias=True,
            quant=first_quant,
            w_bits=first_w_bits,
            equalize=cfg.equalize,
            scale_mode=cfg.scale_mode,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, int(embed_dim)))
        self.pos_drop = nn.Dropout(float(drop))
        self.patch_norm = nn.LayerNorm(int(embed_dim)) if bool(patch_norm) else nn.Identity()
        self.pool = str(pool)
        if self.pool not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pool={pool!r} (expected 'cls' or 'mean')")

        self.blocks = nn.ModuleList(
            [
                Block(
                    int(embed_dim),
                    int(num_heads),
                    cfg,
                    mlp_ratio=float(mlp_ratio),
                    drop=float(drop),
                    attn_drop=float(attn_drop),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))

        self.head = QuantLinear(
            int(embed_dim),
            int(num_classes),
            bias=True,
            quant=last_quant,
            w_bits=last_w_bits,
            equalize=cfg.equalize,
            scale_mode=cfg.scale_mode,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Simple initialization (kept lightweight; good enough for SVHN-scale experiments).
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b = x.size(0)
        x = self.patch_embed(x)  # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.patch_norm(x)
        x = self.qa(x)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, C)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(dim=1)
        x = self.head(x)
        return x
