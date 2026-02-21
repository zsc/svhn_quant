from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.svhn_mat import SVHNMatDataset, SVHNTransformConfig
from models.svhn_cnn import QuantConfig, SVHNCNN
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
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
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
    cfg_dict = ckpt.get("config", {})
    cfg = QuantConfig(**cfg_dict) if cfg_dict else QuantConfig()

    model = SVHNCNN(cfg).to(device)
    model.load_state_dict(ckpt["model"])
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
