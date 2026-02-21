# SVHN Quantization (PyTorch MPS)

This repo trains a small SVHN CNN on **Apple Silicon** (tested with PyTorch MPS) and supports **Balanced Quantization** (arXiv:1706.07145) for **weights** during training (STE).

## Data

Place the SVHN Format 2 `.mat` files in `--data_dir` (default `.`):

- `train_32x32.mat`
- `test_32x32.mat`
- `extra_32x32.mat` (optional via `--use_extra`)

SVHN uses label `10` to represent digit `0` — the dataset loader maps `10 -> 0` so labels are always `[0..9]`.

## Install

Required:

- `torch`, `torchvision`
- `numpy`, `scipy`
- `tqdm`

Optional:

- `pytest` (for unit tests)

## Train

Baseline (fp32):

```bash
python train_svhn.py --device mps --quant none --w_bits 32 --data_dir .
```

Balanced Quantization (paper Algorithm 1/2, recursive-mean equalization):

```bash
python train_svhn.py --device mps --epochs 1 --quant balanced --w_bits 4 --data_dir .
```

Uniform symmetric quantization baseline:

```bash
python train_svhn.py --device mps --epochs 1 --quant uniform --w_bits 4 --data_dir .
```

Common knobs:

- `--use_extra`: include `extra_32x32.mat` in training.
- `--scale_mode {maxabs,meanabs2.5}`: paper default is `maxabs`; `meanabs2.5` is a practical alternative (stop-gradient on statistics).
- `--fp32_first_last`: keep first conv + last fc in fp32.

Notes (macOS):

- Default `--num_workers 0` avoids duplicating large `.mat` arrays due to `spawn`.

## Evaluate

```bash
python eval_svhn.py --ckpt checkpoints/best.pt --device mps --data_dir .
```

## Outputs

Training writes:

- `checkpoints/best.pt` (best val acc)
- `checkpoints/last.pt`
- `checkpoints/metrics.jsonl` (train/val per epoch + final test row)

## Tests

```bash
pytest -q
```

