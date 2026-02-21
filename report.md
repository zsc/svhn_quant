# SVHN 量化实验报告（PyTorch + MPS）

日期：2026-02-21  
任务：在 Apple Silicon 上用 PyTorch(MPS) 训练 SVHN CNN，并测试不同 `W_bits/A_bits` 组合的 1-epoch 精度与耗时（默认使用 `extra_32x32.mat`）。

## 环境

- Python：3.12.12
- PyTorch：2.10.0
- 后端：`mps`（`torch.backends.mps.is_available()==True`）

## 数据与预处理

- 数据：SVHN Format 2（`train_32x32.mat`, `extra_32x32.mat`, `test_32x32.mat`）
- 训练集：`train + extra`（默认启用）
- 划分：从训练集合并集里按 `val_split=0.1` 抽验证集
  - train：543,949
  - val：60,439
  - test：26,032
- 预处理：
  - `uint8 -> float32`，除以 255
  - Normalize：`mean=(0.5,0.5,0.5)`, `std=(0.5,0.5,0.5)`
  - 增强：`RandomCrop(32, padding=4)`（默认开启），水平翻转默认关闭

## 实验设定（对齐）

所有实验统一使用：

- `--device mps`
- `--quant balanced`
- `--equalize recursive_mean`
- `--scale_mode maxabs`（除额外补充的 W2A4 meanabs2.5 变体外）
- `--epochs 1`
- `--batch_size 256`
- `--lr 0.01`（SGD，`momentum=0.9`, `weight_decay=5e-4`）
- Scheduler：cosine（默认）
- `--seed 42`
- `--no_tqdm`

耗时统计来自 `train_svhn.py` 内部 `time.perf_counter()`，并在计时前后对 MPS 做 `torch.mps.synchronize()`。

## 结果

| 配置 | Train acc | Val acc | Test acc | Train(s) | Val(s) | Epoch(s) | Test(s) | 输出目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W8A8 | 0.9231 | 0.9696 | 0.9469 | 345.3 | 106.0 | 451.2 | 44.8 | `sweeps/2026-02-21_balanced_w8a8_e1` |
| W4A8 | 0.9171 | 0.9686 | 0.9431 | 1121.9 | 84.3 | 1206.2 | 36.4 | `sweeps/2026-02-21_balanced_w4a8_e1` |
| W4A4 | 0.9170 | 0.9670 | 0.9400 | 1185.7 | 83.0 | 1268.7 | 36.3 | `sweeps/2026-02-21_balanced_w4a4_e1` |
| W2A4 | 0.1164 | 0.0831 | 0.0670 | 1201.6 | 13.5 | 1215.1 | 26.7 | `sweeps/2026-02-21_balanced_w2a4_e1` |
| W2A4 (meanabs2.5) | 0.8981 | 0.9669 | 0.9409 | 1073.5 | 73.3 | 1146.8 | 33.4 | `sweeps/2026-02-21_balanced_w2a4_e1_meanabs2.5` |

## 运行命令

```bash
python -u train_svhn.py --device mps --epochs 1 --quant balanced --w_bits 8 --a_bits 8 --batch_size 256 --data_dir . --output_dir sweeps/2026-02-21_balanced_w8a8_e1 --seed 42 --no_tqdm
python -u train_svhn.py --device mps --epochs 1 --quant balanced --w_bits 4 --a_bits 8 --batch_size 256 --data_dir . --output_dir sweeps/2026-02-21_balanced_w4a8_e1 --seed 42 --no_tqdm
python -u train_svhn.py --device mps --epochs 1 --quant balanced --w_bits 4 --a_bits 4 --batch_size 256 --data_dir . --output_dir sweeps/2026-02-21_balanced_w4a4_e1 --seed 42 --no_tqdm
python -u train_svhn.py --device mps --epochs 1 --quant balanced --w_bits 2 --a_bits 4 --batch_size 256 --data_dir . --output_dir sweeps/2026-02-21_balanced_w2a4_e1 --seed 42 --no_tqdm
python -u train_svhn.py --device mps --epochs 1 --quant balanced --w_bits 2 --a_bits 4 --scale_mode meanabs2.5 --batch_size 256 --data_dir . --output_dir sweeps/2026-02-21_balanced_w2a4_e1_meanabs2.5 --seed 42 --no_tqdm
```

## 备注与下一步

- W8A8 / W4A8 / W4A4 在 1 epoch 就能达到 ~0.94+ 的 test acc（使用 `train+extra`）。
- W2A4 在 `--scale_mode maxabs` 下出现明显失稳/坍塌（test acc 接近随机），但切换到 `--scale_mode meanabs2.5` 后恢复到 `~0.94` 的 test acc。

---

## ViT（8x8 patch）实验（≤10 epoch）

模型：`--model vit`，默认 `--vit_patch 8`，把 `32x32` 切成 `4x4=16` 个 patch token（再加 `cls` token，总 token=17）。

除特别说明外统一使用：

- 数据：`train+extra`（默认启用）
- `--device mps --quant balanced --equalize recursive_mean`
- `--batch_size 256 --seed 42 --no_tqdm`
- 结构：默认 ViT（`vit_dim=192, vit_depth=6, vit_heads=3`）

### Baseline（SGD，drop=0）

设定：

- `--lr 0.001 --optimizer sgd --vit_drop 0 --vit_attn_drop 0`

| 配置 | Epochs | Best Val acc | Test acc | 输出目录 |
|---|---:|---:|---:|---|
| W8A8 | 10 | 0.9120 | 0.8601 | `sweeps/2026-02-21_vit_balanced_w8a8_e10_extra_lr1e-3_drop0.0` |
| W2A4 | 10 | 0.8768 | 0.8231 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_lr1e-3_drop0.0` |
| W2A4 (meanabs2.5) | 10 | 0.8834 | 0.8324 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_meanabs2.5_lr1e-3_drop0.0` |

### Dropout=0.1（SGD）——明显变差

设定：

- `--lr 0.001 --optimizer sgd --vit_drop 0.1 --vit_attn_drop 0.1`

| 配置 | Epochs | Best Val acc | Test acc | 输出目录 |
|---|---:|---:|---:|---|
| W8A8 | 10 | 0.8479 | 0.7935 | `sweeps/2026-02-21_vit_balanced_w8a8_e10_extra_lr1e-3_drop0.1` |
| W2A4 | 10 | 0.8154 | 0.7636 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_lr1e-3_drop0.1` |
| W2A4 (meanabs2.5) | 10 | 0.8228 | 0.7711 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_meanabs2.5_lr1e-3_drop0.1` |

### Trick：AdamW + mean pooling + patch-norm + grad clip（≤10 epoch 就能很高）

设定（对 ViT 很关键）：

- `--optimizer adamw --lr 0.0003 --weight_decay 0.05 --grad_clip 1.0`
- `--vit_pool mean --vit_patch_norm`

| 配置 | Epochs | Best Val acc | Test acc | 输出目录 |
|---|---:|---:|---:|---|
| W8A8 | 5 | 0.9772 | 0.9582 | `sweeps/2026-02-21_vit_balanced_w8a8_e5_extra_adamw_lr3e-4_wd0.05_clip1_poolmean_pnorm_v2` |
| W2A4 | 10 | 0.9777 | 0.9607 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_adamw_lr3e-4_wd0.05_clip1_poolmean_pnorm` |
| W2A4 (meanabs2.5) | 10 | 0.9823 | 0.9669 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_meanabs2.5_adamw_lr3e-4_wd0.05_clip1_poolmean_pnorm` |

结论（ViT 部分）：

- 单纯把 epoch 提高 + dropout，并不能解决精度低的问题；dropout 在这里反而明显变差。
- 影响最大的“招”是：**AdamW + 合适的 weight decay + grad clip + mean pooling + patch-norm**，在 ≤10 epoch 内即可把 test acc 拉到 `~0.96+`。
