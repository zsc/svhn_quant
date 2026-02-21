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
