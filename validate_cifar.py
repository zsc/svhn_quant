from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    name: str
    cmd: list[str]
    output_dir: Path


@dataclass(frozen=True)
class RunResult:
    name: str
    epochs: int
    best_val_acc: float
    test_acc: float
    output_dir: str


def _read_metrics(metrics_path: Path) -> tuple[int, float, float]:
    best_val = 0.0
    test_acc = float("nan")
    epochs = 0
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("epoch") == "test":
                test = row.get("test", {})
                test_acc = float(test.get("acc", float("nan")))
                continue
            epochs = max(epochs, int(row.get("epoch", 0)))
            val = row.get("val", {})
            if "acc" in val:
                best_val = max(best_val, float(val["acc"]))
    return epochs, best_val, test_acc


def _wait_any(running: list[tuple[subprocess.Popen[bytes], RunSpec]]) -> tuple[subprocess.Popen[bytes], RunSpec, int]:
    while True:
        for i, (p, spec) in enumerate(running):
            rc = p.poll()
            if rc is not None:
                running.pop(i)
                return p, spec, int(rc)
        time.sleep(0.2)


def _build_runs(args: argparse.Namespace) -> list[RunSpec]:
    date = time.strftime("%Y-%m-%d")
    root = Path(args.output_root)
    dataset = str(args.dataset)

    common = [
        sys.executable,
        "-u",
        "train_svhn.py",
        "--dataset",
        dataset,
        "--device",
        str(args.device),
        "--data_dir",
        str(args.data_dir),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--val_split",
        str(args.val_split),
        "--num_workers",
        str(args.num_workers),
        "--quant",
        "balanced",
        "--equalize",
        "recursive_mean",
        "--no_tqdm",
        "--hflip",
    ]

    runs: list[tuple[str, int, list[str]]] = []

    # (A) CNN 1-epoch: check scale_mode collapse vs meanabs2.5.
    runs += [
        ("cnn_w8a8_e1", 1, ["--model", "cnn", "--w_bits", "8", "--a_bits", "8", "--epochs", "1", "--optimizer", "sgd", "--lr", "0.01"]),
        (
            "cnn_w2a4_e1_maxabs",
            1,
            ["--model", "cnn", "--w_bits", "2", "--a_bits", "4", "--epochs", "1", "--optimizer", "sgd", "--lr", "0.01", "--scale_mode", "maxabs"],
        ),
        (
            "cnn_w2a4_e1_meanabs2.5",
            1,
            [
                "--model",
                "cnn",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--epochs",
                "1",
                "--optimizer",
                "sgd",
                "--lr",
                "0.01",
                "--scale_mode",
                "meanabs2.5",
            ],
        ),
    ]

    # (B) ViT: dropout hurts (same as SVHN finding).
    runs += [
        (
            "vit_sgd_drop0_w2a4_meanabs2.5_e10",
            10,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "10",
                "--optimizer",
                "sgd",
                "--lr",
                "0.001",
                "--vit_drop",
                "0.0",
                "--vit_attn_drop",
                "0.0",
            ],
        ),
        (
            "vit_sgd_drop0.1_w2a4_meanabs2.5_e10",
            10,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "10",
                "--optimizer",
                "sgd",
                "--lr",
                "0.001",
                "--vit_drop",
                "0.1",
                "--vit_attn_drop",
                "0.1",
            ],
        ),
    ]

    # (C) Best recipe: AdamW + wd + grad-clip + mean-pool + patch-norm.
    runs += [
        (
            "vit_bestrecipe_adamw_w2a4_meanabs2.5_e10",
            10,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "10",
                "--optimizer",
                "adamw",
                "--lr",
                "0.0003",
                "--weight_decay",
                "0.05",
                "--grad_clip",
                "1.0",
                "--vit_pool",
                "mean",
                "--vit_patch_norm",
                "--vit_drop",
                "0.0",
                "--vit_attn_drop",
                "0.0",
            ],
        )
    ]

    # (D) Ablation of the 4 knobs (5 epochs).
    runs += [
        (
            "vit_ablate_full_w2a4_meanabs2.5_adamw_e5",
            5,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "5",
                "--optimizer",
                "adamw",
                "--lr",
                "0.0003",
                "--weight_decay",
                "0.05",
                "--grad_clip",
                "1.0",
                "--vit_pool",
                "mean",
                "--vit_patch_norm",
            ],
        ),
        (
            "vit_ablate_wd0_w2a4_meanabs2.5_adamw_e5",
            5,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "5",
                "--optimizer",
                "adamw",
                "--lr",
                "0.0003",
                "--weight_decay",
                "0.0",
                "--grad_clip",
                "1.0",
                "--vit_pool",
                "mean",
                "--vit_patch_norm",
            ],
        ),
        (
            "vit_ablate_clip0_w2a4_meanabs2.5_adamw_e5",
            5,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "5",
                "--optimizer",
                "adamw",
                "--lr",
                "0.0003",
                "--weight_decay",
                "0.05",
                "--grad_clip",
                "0.0",
                "--vit_pool",
                "mean",
                "--vit_patch_norm",
            ],
        ),
        (
            "vit_ablate_poolcls_w2a4_meanabs2.5_adamw_e5",
            5,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "5",
                "--optimizer",
                "adamw",
                "--lr",
                "0.0003",
                "--weight_decay",
                "0.05",
                "--grad_clip",
                "1.0",
                "--vit_pool",
                "cls",
                "--vit_patch_norm",
            ],
        ),
        (
            "vit_ablate_nopnorm_w2a4_meanabs2.5_adamw_e5",
            5,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "5",
                "--optimizer",
                "adamw",
                "--lr",
                "0.0003",
                "--weight_decay",
                "0.05",
                "--grad_clip",
                "1.0",
                "--vit_pool",
                "mean",
            ],
        ),
        (
            "vit_ablate_poolcls_nopnorm_w2a4_meanabs2.5_adamw_e5",
            5,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "5",
                "--optimizer",
                "adamw",
                "--lr",
                "0.0003",
                "--weight_decay",
                "0.05",
                "--grad_clip",
                "1.0",
                "--vit_pool",
                "cls",
            ],
        ),
    ]

    # (E) Optimizer ablation: patch-norm-only (10 epochs), compare AdamW vs SGD vs Muon.
    runs += [
        (
            "vit_patchnorm_only_adamw_w2a4_meanabs2.5_e10",
            10,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "10",
                "--optimizer",
                "adamw",
                "--lr",
                "0.0003",
                "--weight_decay",
                "0.0",
                "--grad_clip",
                "0.0",
                "--vit_pool",
                "cls",
                "--vit_patch_norm",
            ],
        ),
        (
            "vit_patchnorm_only_sgd_w2a4_meanabs2.5_e10",
            10,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "10",
                "--optimizer",
                "sgd",
                "--lr",
                "0.001",
                "--weight_decay",
                "0.0",
                "--grad_clip",
                "0.0",
                "--vit_pool",
                "cls",
                "--vit_patch_norm",
            ],
        ),
        (
            "vit_patchnorm_only_muon_w2a4_meanabs2.5_e10",
            10,
            [
                "--model",
                "vit",
                "--w_bits",
                "2",
                "--a_bits",
                "4",
                "--scale_mode",
                "meanabs2.5",
                "--epochs",
                "10",
                "--optimizer",
                "muon",
                "--lr",
                "0.001",
                "--momentum",
                "0.95",
                "--weight_decay",
                "0.0",
                "--grad_clip",
                "0.0",
                "--vit_pool",
                "cls",
                "--vit_patch_norm",
            ],
        ),
    ]

    specs: list[RunSpec] = []
    for name, _epochs, extra in runs:
        out_dir = root / f"{date}_{dataset}_{name}"
        cmd = common + ["--output_dir", str(out_dir)] + extra
        specs.append(RunSpec(name=name, cmd=cmd, output_dir=out_dir))
    return specs


def _write_report(results: list[RunResult], *, dataset: str, output_path: Path) -> None:
    by_name = {r.name: r for r in results}

    def get(name: str) -> RunResult:
        if name not in by_name:
            raise KeyError(name)
        return by_name[name]

    def fmt(x: float) -> str:
        return f"{x:.4f}"

    date = time.strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append(f"# CIFAR 验证报告（{dataset}）")
    lines.append("")
    lines.append(f"日期：{date}")
    lines.append("")
    lines.append("统一设定：")
    lines.append("")
    lines.append("- `--device mps --quant balanced --equalize recursive_mean`")
    lines.append("- `--batch_size 256 --seed 42 --val_split 0.1 --no_tqdm`")
    lines.append("- 数据增强：`RandomCrop(32,padding=4)` + `RandomHorizontalFlip(p=0.5)`（通过 `--hflip` 显式开启）")
    lines.append("")

    lines.append("## 1. CNN：scale_mode 稳定性（1 epoch）")
    lines.append("")
    lines.append("| 配置 | Best Val acc | Test acc | 输出目录 |")
    lines.append("|---|---:|---:|---|")
    for name, label in [
        ("cnn_w8a8_e1", "W8A8"),
        ("cnn_w2a4_e1_maxabs", "W2A4 (maxabs)"),
        ("cnn_w2a4_e1_meanabs2.5", "W2A4 (meanabs2.5)"),
    ]:
        r = get(name)
        lines.append(f"| {label} | {fmt(r.best_val_acc)} | {fmt(r.test_acc)} | `{r.output_dir}` |")
    lines.append("")

    lines.append("## 2. ViT：Dropout 是否有帮助？（SGD，10 epochs）")
    lines.append("")
    r0 = get("vit_sgd_drop0_w2a4_meanabs2.5_e10")
    r1 = get("vit_sgd_drop0.1_w2a4_meanabs2.5_e10")
    lines.append("| 配置 | Best Val acc | Test acc | ΔTest vs drop0 | 输出目录 |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(f"| drop=0.0 | {fmt(r0.best_val_acc)} | {fmt(r0.test_acc)} | +0.0000 | `{r0.output_dir}` |")
    lines.append(
        f"| drop=0.1 | {fmt(r1.best_val_acc)} | {fmt(r1.test_acc)} | {r1.test_acc - r0.test_acc:+.4f} | `{r1.output_dir}` |"
    )
    lines.append("")

    lines.append("## 3. ViT：best recipe（AdamW + wd/clip/mean-pool/patch-norm，10 epochs）")
    lines.append("")
    rb = get("vit_bestrecipe_adamw_w2a4_meanabs2.5_e10")
    lines.append("| 配置 | Best Val acc | Test acc | ΔTest vs SGD(drop0) | 输出目录 |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(
        f"| best recipe | {fmt(rb.best_val_acc)} | {fmt(rb.test_acc)} | {rb.test_acc - r0.test_acc:+.4f} | `{rb.output_dir}` |"
    )
    lines.append("")

    lines.append("## 4. ViT：四个 trick 的 ablation（AdamW，5 epochs）")
    lines.append("")
    base = get("vit_ablate_full_w2a4_meanabs2.5_adamw_e5")
    lines.append("| 变更（相对 baseline） | Best Val acc | Test acc | ΔTest | 输出目录 |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(f"| baseline（full） | {fmt(base.best_val_acc)} | {fmt(base.test_acc)} | +0.0000 | `{base.output_dir}` |")
    for name, label in [
        ("vit_ablate_wd0_w2a4_meanabs2.5_adamw_e5", "去掉 weight decay（wd=0）"),
        ("vit_ablate_clip0_w2a4_meanabs2.5_adamw_e5", "去掉 grad clip（clip=0）"),
        ("vit_ablate_poolcls_w2a4_meanabs2.5_adamw_e5", "pooling 改成 cls（pool=cls）"),
        ("vit_ablate_nopnorm_w2a4_meanabs2.5_adamw_e5", "去掉 patch-norm（无 `--vit_patch_norm`）"),
        ("vit_ablate_poolcls_nopnorm_w2a4_meanabs2.5_adamw_e5", "pool=cls + 无 patch-norm（交互）"),
    ]:
        r = get(name)
        lines.append(
            f"| {label} | {fmt(r.best_val_acc)} | {fmt(r.test_acc)} | {r.test_acc - base.test_acc:+.4f} | `{r.output_dir}` |"
        )
    lines.append("")

    lines.append("## 5. ViT：optimizer ablation（patch-norm-only，10 epochs）")
    lines.append("")
    ra = get("vit_patchnorm_only_adamw_w2a4_meanabs2.5_e10")
    rs = get("vit_patchnorm_only_sgd_w2a4_meanabs2.5_e10")
    rm = get("vit_patchnorm_only_muon_w2a4_meanabs2.5_e10")
    lines.append("| Optimizer | Best Val acc | Test acc | 输出目录 |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| AdamW | {fmt(ra.best_val_acc)} | {fmt(ra.test_acc)} | `{ra.output_dir}` |")
    lines.append(f"| SGD | {fmt(rs.best_val_acc)} | {fmt(rs.test_acc)} | `{rs.output_dir}` |")
    lines.append(f"| Muon | {fmt(rm.best_val_acc)} | {fmt(rm.test_acc)} | `{rm.output_dir}` |")
    lines.append("")

    lines.append("## 6. 简要结论")
    lines.append("")
    lines.append("- `meanabs2.5` 通常比 `maxabs` 更稳（尤其是低 bit）。")
    lines.append("- Dropout(0.1) 是否有益需要具体任务验证，但在 SVHN 上它明显变差；这里也给出对照结果。")
    lines.append("- ViT 上，`AdamW + wd + grad clip + mean-pool + patch-norm` 往往是最关键的一组稳定性/精度“开关”。")
    lines.append("- optimizer 对 ViT 影响很大：AdamW 通常远强于 SGD；Muon 需要按其约束（只优化 2D 参数）正确拆分。")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Validate SVHN findings on CIFAR-10/100 (runs many short experiments).")
    p.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--device", type=str, default="mps", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--output_root", type=str, default="sweeps")
    p.add_argument("--report_path", type=str, default="", help="Markdown report output path (default: report_cifar_<dataset>.md)")
    p.add_argument("--jobs", type=int, default=3)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    # Avoid multiple parallel processes downloading the same tarball concurrently.
    from torchvision.datasets import CIFAR10, CIFAR100

    ds_cls = CIFAR10 if args.dataset == "cifar10" else CIFAR100
    ds_cls(root=args.data_dir, train=True, download=True)
    ds_cls(root=args.data_dir, train=False, download=True)

    specs = _build_runs(args)
    for s in specs:
        print(" ".join(s.cmd))

    if args.dry_run:
        return

    os.makedirs(args.output_root, exist_ok=True)
    max_jobs = max(1, int(args.jobs))
    pending = specs[:]
    running: list[tuple[subprocess.Popen[bytes], RunSpec]] = []

    while pending or running:
        while pending and len(running) < max_jobs:
            spec = pending.pop(0)
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            log_path = spec.output_dir / "run.log"
            log_f = log_path.open("wb")
            p = subprocess.Popen(spec.cmd, stdout=log_f, stderr=subprocess.STDOUT)
            log_f.close()
            running.append((p, spec))
            print(f"[RUN] {spec.name} -> {spec.output_dir} (log: {log_path})")

        proc, spec, rc = _wait_any(running)
        if rc != 0:
            raise RuntimeError(f"Run {spec.name} failed with exit code {rc}. See: {spec.output_dir / 'run.log'}")
        print(f"[OK] {spec.name}")

    results: list[RunResult] = []
    for spec in specs:
        metrics_path = spec.output_dir / "metrics.jsonl"
        epochs, best_val, test_acc = _read_metrics(metrics_path)
        results.append(
            RunResult(
                name=spec.name,
                epochs=epochs,
                best_val_acc=best_val,
                test_acc=test_acc,
                output_dir=str(spec.output_dir),
            )
        )

    report_path = Path(args.report_path) if args.report_path else Path(f"report_cifar_{args.dataset}.md")
    _write_report(results, dataset=str(args.dataset), output_path=report_path)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
