from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from datasets.svhn_mat import SVHNMatDataset, SVHNTransformConfig
from models.svhn_cnn import QuantConfig, SVHNCNN
from models.svhn_vit import SVHNViT
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.meter import AverageMeter, accuracy_top1
from utils.seed import seed_everything


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


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
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


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    device: torch.device,
    *,
    grad_clip: float,
    pbar_desc: str,
    use_tqdm: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    it = tqdm(loader, desc=pbar_desc, leave=False) if use_tqdm else loader
    for x, y in it:
        x = x.to(device)
        y = y.to(device)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        for opt in optimizers:
            opt.step()

        acc = accuracy_top1(logits.detach(), y)
        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(acc, n=x.size(0))
        if use_tqdm:
            it.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def _build_train_val(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if args.dataset == "svhn":
        eval_tf = SVHNTransformConfig(
            normalize=True,
            random_crop=False,
            horizontal_flip=False,
        )
        train_tf = SVHNTransformConfig(
            normalize=True,
            random_crop=not args.no_augment,
            crop_padding=int(args.crop_padding),
            horizontal_flip=bool(args.hflip),
            hflip_p=float(args.hflip_p),
        )

        # Load arrays once, then create two "views" (train with aug, val without aug)
        # to avoid duplicating big numpy buffers (especially important when --use_extra).
        train_base = SVHNMatDataset(args.data_dir, "train", transform=eval_tf, train=False)
        train_aug = SVHNMatDataset(
            args.data_dir,
            "train",
            images=train_base.images,
            labels=train_base.labels,
            transform=train_tf,
            train=True,
        )
        train_noaug = SVHNMatDataset(
            args.data_dir,
            "train",
            images=train_base.images,
            labels=train_base.labels,
            transform=eval_tf,
            train=False,
        )

        parts_aug: list[torch.utils.data.Dataset[tuple[torch.Tensor, int]]] = [train_aug]
        parts_noaug: list[torch.utils.data.Dataset[tuple[torch.Tensor, int]]] = [train_noaug]

        if args.use_extra:
            extra_base = SVHNMatDataset(args.data_dir, "extra", transform=eval_tf, train=False)
            extra_aug = SVHNMatDataset(
                args.data_dir,
                "extra",
                images=extra_base.images,
                labels=extra_base.labels,
                transform=train_tf,
                train=True,
            )
            extra_noaug = SVHNMatDataset(
                args.data_dir,
                "extra",
                images=extra_base.images,
                labels=extra_base.labels,
                transform=eval_tf,
                train=False,
            )
            parts_aug.append(extra_aug)
            parts_noaug.append(extra_noaug)

        full_aug = ConcatDataset(parts_aug)
        full_noaug = ConcatDataset(parts_noaug)

        n = len(full_aug)
        val_size = int(round(n * float(args.val_split)))
        val_size = max(1, min(val_size, n - 1))
        gen = torch.Generator().manual_seed(int(args.seed))
        perm = torch.randperm(n, generator=gen).tolist()
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        train_ds = Subset(full_aug, train_idx)
        val_ds = Subset(full_noaug, val_idx)
        test_ds = SVHNMatDataset(args.data_dir, "test", transform=eval_tf, train=False)
    elif args.dataset in {"cifar10", "cifar100"}:
        from torchvision import transforms as T
        from torchvision.datasets import CIFAR10, CIFAR100

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

        train_ops: list[torch.nn.Module] = []
        if not bool(args.no_augment):
            train_ops.append(T.RandomCrop(32, padding=int(args.crop_padding)))
        if bool(args.hflip):
            train_ops.append(T.RandomHorizontalFlip(p=float(args.hflip_p)))
        train_ops += [T.ToTensor(), T.Normalize(mean, std)]
        train_tf = T.Compose(train_ops)

        ds_cls = CIFAR10 if args.dataset == "cifar10" else CIFAR100
        full_aug = ds_cls(root=args.data_dir, train=True, download=True, transform=train_tf)
        full_noaug = ds_cls(root=args.data_dir, train=True, download=True, transform=eval_tf)

        n = len(full_aug)
        val_size = int(round(n * float(args.val_split)))
        val_size = max(1, min(val_size, n - 1))
        gen = torch.Generator().manual_seed(int(args.seed))
        perm = torch.randperm(n, generator=gen).tolist()
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        train_ds = Subset(full_aug, train_idx)
        val_ds = Subset(full_noaug, val_idx)
        test_ds = ds_cls(root=args.data_dir, train=False, download=True, transform=eval_tf)
    else:
        raise ValueError(f"Unknown dataset {args.dataset!r}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVHN model (CNN/ViT) with Balanced Quantization (MPS-ready).")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing *_32x32.mat")
    parser.add_argument("--dataset", type=str, choices=["svhn", "cifar10", "cifar100"], default="svhn")
    parser.add_argument("--model", type=str, choices=["cnn", "vit"], default="cnn")
    parser.add_argument(
        "--use_extra",
        action="store_true",
        default=True,
        help="Include extra_32x32.mat in training (default: enabled)",
    )
    parser.add_argument(
        "--no_extra",
        action="store_false",
        dest="use_extra",
        help="Do not include extra_32x32.mat in training",
    )
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--num_workers", type=int, default=0, help="macOS: keep 0 to avoid dataset duplication via spawn")

    # Augmentation toggles
    parser.add_argument("--no_augment", action="store_true", help="Disable RandomCrop augmentation")
    parser.add_argument("--crop_padding", type=int, default=4)
    parser.add_argument("--hflip", action="store_true", help="Enable RandomHorizontalFlip (off by default for SVHN)")
    parser.add_argument("--hflip_p", type=float, default=0.5)

    # Quantization toggles
    parser.add_argument("--quant", type=str, choices=["none", "balanced", "uniform"], default="none")
    parser.add_argument("--w_bits", type=int, choices=[1, 2, 3, 4, 8, 32], default=32)
    parser.add_argument("--a_bits", type=int, choices=[1, 2, 3, 4, 8, 32], default=32)
    parser.add_argument("--equalize", type=str, choices=["recursive_mean"], default="recursive_mean")
    parser.add_argument("--scale_mode", type=str, choices=["maxabs", "meanabs2.5"], default="maxabs")
    parser.add_argument(
        "--w_transform",
        type=str,
        choices=["none", "tanh"],
        default="none",
        help="Balanced quantization weight transform (e.g., tanh(W) + maxabs)",
    )
    parser.add_argument(
        "--w_bias_mode",
        type=str,
        choices=["none", "mean"],
        default="none",
        help="Balanced quantization weight bias/zero-point: quantize W-mean(W) then add mean(W) back",
    )
    parser.add_argument("--fp32_first_last", action="store_true", help="Keep first conv/patch and last head in fp32")
    parser.add_argument("--fp32_first", action="store_true", help="Keep first conv/patch in fp32")
    parser.add_argument("--fp32_last", action="store_true", help="Keep last head in fp32")

    # ViT arch (used when --model vit)
    parser.add_argument("--vit_patch", type=int, default=8)
    parser.add_argument("--vit_dim", type=int, default=192)
    parser.add_argument("--vit_depth", type=int, default=6)
    parser.add_argument("--vit_heads", type=int, default=3)
    parser.add_argument("--vit_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--vit_patch_norm", action="store_true", help="Apply LayerNorm after patch embedding")
    parser.add_argument("--vit_pool", type=str, choices=["cls", "mean"], default="cls", help="Pooling for classification")
    parser.add_argument("--vit_drop", type=float, default=0.0)
    parser.add_argument("--vit_attn_drop", type=float, default=0.0)

    # Optimization
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "muon"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Clip grad norm if > 0 (useful for ViT)")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--step_size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)

    # I/O
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        default=True,
        help="Save optimizer state into checkpoints (default: enabled; can be large)",
    )
    parser.add_argument(
        "--no_save_optimizer",
        action="store_false",
        dest="save_optimizer",
        help="Do not save optimizer state (smaller checkpoints; resume will not restore optimizer)",
    )
    parser.add_argument(
        "--save_last",
        action="store_true",
        default=True,
        help="Save last.pt each epoch (default: enabled)",
    )
    parser.add_argument(
        "--no_save_last",
        action="store_false",
        dest="save_last",
        help="Do not save last.pt (reduce disk usage during sweeps)",
    )
    parser.add_argument(
        "--save_best",
        action="store_true",
        default=True,
        help="Save best.pt when val improves (default: enabled)",
    )
    parser.add_argument(
        "--no_save_best",
        action="store_false",
        dest="save_best",
        help="Do not save best.pt (metrics-only sweeps; reduces disk usage)",
    )
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars (useful for sweeps/logging)")

    args = parser.parse_args()

    seed_everything(int(args.seed))
    device = _select_device(args.device)
    print(
        "Device:",
        str(device),
        "| mps available:",
        torch.backends.mps.is_available(),
        "| cuda available:",
        torch.cuda.is_available(),
    )
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = _build_train_val(args)
    print(
        f"Data({args.dataset}): train {len(train_loader.dataset)} | val {len(val_loader.dataset)}"
        f" | test {len(test_loader.dataset)}"
        f"{' | use_extra ' + str(bool(args.use_extra)) if args.dataset == 'svhn' else ''}"
    )

    cfg = QuantConfig(
        quant=args.quant,
        w_bits=int(args.w_bits),
        a_bits=int(args.a_bits),
        equalize=args.equalize,
        scale_mode=args.scale_mode,
        w_transform=str(args.w_transform),
        w_bias_mode=str(args.w_bias_mode),
        fp32_first_last=bool(args.fp32_first_last),
        fp32_first=bool(args.fp32_first),
        fp32_last=bool(args.fp32_last),
    )
    num_classes = 10 if args.dataset in {"svhn", "cifar10"} else 100
    if args.model == "cnn":
        model = SVHNCNN(cfg, num_classes=num_classes)
    elif args.model == "vit":
        model = SVHNViT(
            cfg,
            patch_size=int(args.vit_patch),
            embed_dim=int(args.vit_dim),
            depth=int(args.vit_depth),
            num_heads=int(args.vit_heads),
            mlp_ratio=float(args.vit_mlp_ratio),
            patch_norm=bool(args.vit_patch_norm),
            pool=str(args.vit_pool),
            drop=float(args.vit_drop),
            attn_drop=float(args.vit_attn_drop),
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown --model {args.model!r}")

    model = model.to(device)
    print(f"Model: {args.model}")
    print("Model param device:", next(model.parameters()).device)
    if device.type == "mps":
        print(f"MPS allocated (after model.to): {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizers: list[torch.optim.Optimizer]
    if args.optimizer == "sgd":
        optimizers = [
            torch.optim.SGD(
                model.parameters(),
                lr=float(args.lr),
                momentum=float(args.momentum),
                weight_decay=float(args.weight_decay),
            )
        ]
    elif args.optimizer == "adamw":
        decay: list[torch.Tensor] = []
        no_decay: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                p.ndim == 1
                or name.endswith(".bias")
                or ".bn" in name
                or ".norm" in name
                or "pos_embed" in name
                or "cls_token" in name
            ):
                no_decay.append(p)
            else:
                decay.append(p)
        optimizers = [
            torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": float(args.weight_decay)},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=float(args.lr),
            )
        ]
    elif args.optimizer == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch.optim.Muon is not available in this PyTorch build.")

        muon_2d: list[torch.Tensor] = []
        aux_decay: list[torch.Tensor] = []
        aux_no_decay: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2:
                muon_2d.append(p)
                continue
            if (
                p.ndim == 1
                or name.endswith(".bias")
                or ".bn" in name
                or ".norm" in name
                or "pos_embed" in name
                or "cls_token" in name
            ):
                aux_no_decay.append(p)
            else:
                aux_decay.append(p)

        # Muon only supports 2D parameters. Optimize the rest (bias/norm/embeddings/conv, etc.)
        # with a standard method (AdamW), as suggested in torch.optim.Muon docs.
        muon_opt = torch.optim.Muon(
            muon_2d,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            momentum=float(args.momentum),
            nesterov=True,
        )
        aux_opt = torch.optim.AdamW(
            [
                {"params": aux_decay, "weight_decay": float(args.weight_decay)},
                {"params": aux_no_decay, "weight_decay": 0.0},
            ],
            lr=float(args.lr),
        )
        optimizers = [muon_opt, aux_opt]
        print(
            f"Muon param split: 2D={len(muon_2d)} | aux(decay)={len(aux_decay)} | aux(no_decay)={len(aux_no_decay)}"
        )
    else:
        raise ValueError(f"Unknown --optimizer {args.optimizer!r}")

    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
    if args.scheduler == "cosine":
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs)) for opt in optimizers
        ]
    elif args.scheduler == "step":
        schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=int(args.step_size), gamma=float(args.gamma))
            for opt in optimizers
        ]
    else:
        schedulers = []

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt_state = ckpt.get("optimizer")
        if isinstance(opt_state, list):
            if len(opt_state) != len(optimizers):
                raise RuntimeError(
                    f"Checkpoint has {len(opt_state)} optimizer states but current run uses {len(optimizers)}."
                )
            for opt, state in zip(optimizers, opt_state, strict=True):
                opt.load_state_dict(state)
        elif isinstance(opt_state, dict):
            optimizers[0].load_state_dict(opt_state)
        elif opt_state is None:
            pass
        else:
            raise RuntimeError(f"Unsupported optimizer state type in checkpoint: {type(opt_state)}")
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))
        # `last.pt` previously stored a potentially stale `best_val_acc` (saved before
        # updating best for that epoch). Prefer the value from `best.pt` if present.
        best_path = os.path.join(args.output_dir, "best.pt")
        if os.path.exists(best_path):
            try:
                best_ckpt = load_checkpoint(best_path, map_location=device)
                best_val_acc = max(best_val_acc, float(best_ckpt.get("best_val_acc", 0.0)))
            except Exception:
                pass
        # Make the LR scheduler continue from the resumed epoch index.
        for sch in schedulers:
            sch.last_epoch = start_epoch

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as mf:
        for epoch in range(start_epoch, int(args.epochs)):
            lr = float(optimizers[0].param_groups[0]["lr"])
            _sync_device(device)
            t0 = time.perf_counter()
            train_m = _train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizers,
                device,
                grad_clip=float(args.grad_clip),
                pbar_desc=f"epoch {epoch+1}/{args.epochs} [train]",
                use_tqdm=not bool(args.no_tqdm),
            )
            _sync_device(device)
            t_train = time.perf_counter() - t0

            _sync_device(device)
            t0 = time.perf_counter()
            val_m = _evaluate(model, val_loader, criterion, device)
            _sync_device(device)
            t_val = time.perf_counter() - t0

            t_epoch = t_train + t_val
            for sch in schedulers:
                sch.step()

            row: dict[str, Any] = {
                "epoch": epoch + 1,
                "lr": lr,
                "train": train_m,
                "val": val_m,
                "time_sec": {"train": t_train, "val": t_val, "epoch": t_epoch},
                "config": asdict(cfg),
            }
            mf.write(json.dumps(row) + "\n")
            mf.flush()

            print(
                f"Epoch {epoch+1:03d} | lr {lr:.5f} | "
                f"train loss {train_m['loss']:.4f} acc {train_m['acc']:.4f} | "
                f"val loss {val_m['loss']:.4f} acc {val_m['acc']:.4f} | "
                f"time train {t_train:.1f}s val {t_val:.1f}s epoch {t_epoch:.1f}s"
            )

            is_best = bool(val_m["acc"] > best_val_acc)
            if is_best:
                best_val_acc = float(val_m["acc"])

            if is_best and bool(args.save_best):
                best_path = os.path.join(args.output_dir, "best.pt")
                state: dict[str, Any] = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                    "config": asdict(cfg),
                }
                if bool(args.save_optimizer):
                    state["optimizer"] = [opt.state_dict() for opt in optimizers]
                save_checkpoint(state, best_path)

            if bool(args.save_last):
                last_path = os.path.join(args.output_dir, "last.pt")
                state = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                    "config": asdict(cfg),
                }
                if bool(args.save_optimizer):
                    state["optimizer"] = [opt.state_dict() for opt in optimizers]
                save_checkpoint(state, last_path)

    # Final test with best checkpoint.
    best_path = os.path.join(args.output_dir, "best.pt")
    if os.path.exists(best_path):
        ckpt = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    _sync_device(device)
    t0 = time.perf_counter()
    test_m = _evaluate(model, test_loader, criterion, device)
    _sync_device(device)
    t_test = time.perf_counter() - t0
    print(f"Test | loss {test_m['loss']:.4f} acc {test_m['acc']:.4f}")
    with open(metrics_path, "a", encoding="utf-8") as mf:
        mf.write(json.dumps({"epoch": "test", "test": test_m, "time_sec": {"test": t_test}}) + "\n")


if __name__ == "__main__":
    main()
