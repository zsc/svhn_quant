from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepResult:
    quant: str
    w_bits: int
    a_bits: int
    epochs: int
    best_val_acc: float
    test_acc: float
    output_dir: str


def _pick_epochs(w_bits: int, a_bits: int, *, e8: int, e4: int, e2: int) -> int:
    m = min(int(w_bits), int(a_bits))
    if m <= 2:
        return int(e2)
    if m <= 4:
        return int(e4)
    return int(e8)


def _read_metrics(metrics_path: str) -> tuple[float, float]:
    best_val = 0.0
    test_acc = float("nan")
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("epoch") == "test":
                test = row.get("test", {})
                test_acc = float(test.get("acc", float("nan")))
                continue
            val = row.get("val", {})
            if "acc" in val:
                best_val = max(best_val, float(val["acc"]))
    return best_val, test_acc


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep (w_bits, a_bits) combinations for SVHN/CIFAR.")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--device", type=str, default="mps", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--dataset", type=str, default="svhn", choices=["svhn", "cifar10", "cifar100"])
    p.add_argument("--model", type=str, default="cnn", choices=["cnn", "vit"])
    p.add_argument("--quant", type=str, default="balanced", choices=["none", "balanced", "uniform"])
    p.add_argument("--w_bits", type=int, nargs="+", default=[8, 4, 2])
    p.add_argument("--a_bits", type=int, nargs="+", default=[8, 4, 2])
    p.add_argument("--scale_mode", type=str, default="maxabs", choices=["maxabs", "meanabs2.5"])
    p.add_argument("--equalize", type=str, default="recursive_mean", choices=["recursive_mean"])
    p.add_argument("--fp32_first_last", action="store_true")
    p.add_argument("--fp32_first", action="store_true")
    p.add_argument("--fp32_last", action="store_true")
    p.add_argument("--use_extra", action="store_true", default=True, help="Include extra_32x32.mat (default: enabled)")
    p.add_argument("--no_extra", action="store_false", dest="use_extra", help="Do not include extra_32x32.mat")
    p.add_argument("--hflip", action="store_true", help="Pass --hflip to train_svhn.py (recommended for CIFAR)")

    p.add_argument("--epochs_8", type=int, default=5, help="Epochs when min(w_bits,a_bits) >= 8")
    p.add_argument("--epochs_4", type=int, default=10, help="Epochs when min(w_bits,a_bits) == 4")
    p.add_argument("--epochs_2", type=int, default=20, help="Epochs when min(w_bits,a_bits) == 2")
    p.add_argument("--epochs", type=int, default=0, help="If >0, override and use the same epochs for all runs")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--output_root", type=str, default="sweeps")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--jobs", type=int, default=1, help="Max parallel jobs (use >1 to increase GPU utilization)")
    p.add_argument("--no_save_last", action="store_true", help="Pass --no_save_last to train_svhn.py")
    p.add_argument("--no_save_optimizer", action="store_true", help="Pass --no_save_optimizer to train_svhn.py")
    p.add_argument("--no_save_best", action="store_true", help="Pass --no_save_best to train_svhn.py (metrics-only)")
    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    results: list[SweepResult] = []

    combos = [(int(w), int(a)) for w in args.w_bits for a in args.a_bits]
    specs: list[tuple[int, int, int, str, list[str]]] = []
    for w_bits, a_bits in combos:
        if int(args.epochs) > 0:
            epochs = int(args.epochs)
        else:
            epochs = _pick_epochs(w_bits, a_bits, e8=args.epochs_8, e4=args.epochs_4, e2=args.epochs_2)
        out_dir = os.path.join(args.output_root, f"{args.quant}_w{w_bits}_a{a_bits}_e{epochs}")

        cmd = [
            sys.executable,
            "-u",
            "train_svhn.py",
            "--dataset",
            str(args.dataset),
            "--model",
            str(args.model),
            "--device",
            args.device,
            "--data_dir",
            args.data_dir,
            "--output_dir",
            out_dir,
            "--quant",
            args.quant,
            "--w_bits",
            str(w_bits),
            "--a_bits",
            str(a_bits),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--val_split",
            str(args.val_split),
            "--num_workers",
            str(args.num_workers),
            "--scale_mode",
            args.scale_mode,
            "--equalize",
            args.equalize,
            "--no_tqdm",
        ]
        if args.fp32_first_last:
            cmd.append("--fp32_first_last")
        if bool(args.fp32_first):
            cmd.append("--fp32_first")
        if bool(args.fp32_last):
            cmd.append("--fp32_last")
        if not args.use_extra:
            cmd.append("--no_extra")
        if bool(args.hflip):
            cmd.append("--hflip")
        if bool(args.no_save_last):
            cmd.append("--no_save_last")
        if bool(args.no_save_optimizer):
            cmd.append("--no_save_optimizer")
        if bool(args.no_save_best):
            cmd.append("--no_save_best")

        print(" ".join(cmd))
        specs.append((w_bits, a_bits, epochs, out_dir, cmd))

    if args.dry_run:
        return

    # Run with limited parallelism and log each run to <out_dir>/run.log.
    max_jobs = max(1, int(args.jobs))
    pending = specs[:]
    running: list[tuple[subprocess.Popen[bytes], tuple[int, int, int, str, list[str]]]] = []

    def wait_any() -> tuple[subprocess.Popen[bytes], tuple[int, int, int, str, list[str]], int]:
        while True:
            for i, (p, spec) in enumerate(running):
                rc = p.poll()
                if rc is not None:
                    running.pop(i)
                    return p, spec, int(rc)
            import time

            time.sleep(0.2)

    while pending or running:
        while pending and len(running) < max_jobs:
            w_bits, a_bits, epochs, out_dir, cmd = pending.pop(0)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(out_dir) / "run.log"
            log_f = log_path.open("wb")
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            log_f.close()
            running.append((p, (w_bits, a_bits, epochs, out_dir, cmd)))

        _, spec, rc = wait_any()
        w_bits, a_bits, epochs, out_dir, _ = spec
        if rc != 0:
            raise RuntimeError(f"Run failed: w_bits={w_bits} a_bits={a_bits} epochs={epochs}. See: {Path(out_dir)/'run.log'}")

    for w_bits, a_bits, epochs, out_dir, _cmd in specs:
        metrics_path = os.path.join(out_dir, "metrics.jsonl")
        best_val, test_acc = _read_metrics(metrics_path)
        results.append(
            SweepResult(
                quant=args.quant,
                w_bits=w_bits,
                a_bits=a_bits,
                epochs=epochs,
                best_val_acc=best_val,
                test_acc=test_acc,
                output_dir=out_dir,
            )
        )

    results_path = os.path.join(args.output_root, "results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quant", "w_bits", "a_bits", "epochs", "best_val_acc", "test_acc", "output_dir"])
        for r in sorted(results, key=lambda x: (x.w_bits, x.a_bits)):
            w.writerow([r.quant, r.w_bits, r.a_bits, r.epochs, f"{r.best_val_acc:.6f}", f"{r.test_acc:.6f}", r.output_dir])

    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
