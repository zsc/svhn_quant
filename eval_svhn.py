from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.svhn_mat import SVHNMatDataset, SVHNTransformConfig
from models.svhn_cnn import QuantConfig, SVHNCNN
from models.svhn_vit import SVHNViT
from utils.checkpoint import load_checkpoint
from utils.meter import AverageMeter, accuracy_top1


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    dev = torch.device(device_arg)
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return dev


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        acc = accuracy_top1(logits, y)
        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(acc, n=x.size(0))
    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate SVHN checkpoint.")
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--device", type=str, choices=["auto", "mps", "cuda", "cpu"], default="auto")
    p.add_argument("--model", type=str, choices=["auto", "cnn", "vit"], default="auto")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)

    # ViT arch (only used when evaluating a ViT checkpoint)
    p.add_argument("--vit_patch", type=int, default=8)
    p.add_argument("--vit_dim", type=int, default=192)
    p.add_argument("--vit_depth", type=int, default=6)
    p.add_argument("--vit_heads", type=int, default=3)
    p.add_argument("--vit_mlp_ratio", type=float, default=4.0)
    p.add_argument("--vit_patch_norm", action="store_true")
    p.add_argument("--vit_pool", type=str, choices=["cls", "mean"], default="cls")
    p.add_argument("--vit_drop", type=float, default=0.0)
    p.add_argument("--vit_attn_drop", type=float, default=0.0)
    args = p.parse_args()

    device = _select_device(args.device)
    print(
        "Device:",
        str(device),
        "| mps available:",
        torch.backends.mps.is_available(),
        "| cuda available:",
        torch.cuda.is_available(),
    )
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    cfg_dict = ckpt.get("config", {})
    cfg = QuantConfig(**cfg_dict) if cfg_dict else QuantConfig()

    model_name = args.model
    if model_name == "auto":
        model_name = str(ckpt_args.get("model", "cnn"))

    if model_name == "cnn":
        model = SVHNCNN(cfg)
    elif model_name == "vit":
        if args.model == "auto":
            vit_patch = int(ckpt_args.get("vit_patch", args.vit_patch))
            vit_dim = int(ckpt_args.get("vit_dim", args.vit_dim))
            vit_depth = int(ckpt_args.get("vit_depth", args.vit_depth))
            vit_heads = int(ckpt_args.get("vit_heads", args.vit_heads))
            vit_mlp_ratio = float(ckpt_args.get("vit_mlp_ratio", args.vit_mlp_ratio))
            vit_patch_norm = bool(ckpt_args.get("vit_patch_norm", args.vit_patch_norm))
            vit_pool = str(ckpt_args.get("vit_pool", args.vit_pool))
            vit_drop = float(ckpt_args.get("vit_drop", args.vit_drop))
            vit_attn_drop = float(ckpt_args.get("vit_attn_drop", args.vit_attn_drop))
        else:
            vit_patch = int(args.vit_patch)
            vit_dim = int(args.vit_dim)
            vit_depth = int(args.vit_depth)
            vit_heads = int(args.vit_heads)
            vit_mlp_ratio = float(args.vit_mlp_ratio)
            vit_patch_norm = bool(args.vit_patch_norm)
            vit_pool = str(args.vit_pool)
            vit_drop = float(args.vit_drop)
            vit_attn_drop = float(args.vit_attn_drop)

        model = SVHNViT(
            cfg,
            patch_size=vit_patch,
            embed_dim=vit_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            patch_norm=vit_patch_norm,
            pool=vit_pool,
            drop=vit_drop,
            attn_drop=vit_attn_drop,
        )
    else:
        raise ValueError(f"Unknown model {model_name!r}")

    model = model.to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Model: {model_name}")
    print("Model param device:", next(model.parameters()).device)
    if device.type == "mps":
        print(f"MPS allocated (after model.to): {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")

    tf = SVHNTransformConfig(normalize=True, random_crop=False, horizontal_flip=False)
    test_ds = SVHNMatDataset(args.data_dir, "test", transform=tf, train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    m = evaluate(model, test_loader, device)
    print(f"Test | loss {m['loss']:.4f} acc {m['acc']:.4f}")


if __name__ == "__main__":
    main()
