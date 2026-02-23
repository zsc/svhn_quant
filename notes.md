# 笔记 — 低比特量化实验

这是一个持续更新的实验记录，用来在本仓库里探索极低精度设置（例如 W2A2、W1A2）。

约定：

- `WkAk` 表示 `--w_bits k --a_bits k`。
- 若无特殊说明：默认 `--quant balanced --equalize recursive_mean`。
- 磁盘空间紧张时，优先用 `--no_save_last --no_save_optimizer`，只保留 `best.pt + metrics.jsonl`（或只留 `metrics.jsonl`）。

## 2026-02-22 — 环境与数据

- Repo：`svhn_quant`
- Torch：`2.8.0+cu128`（本环境有 CUDA；MPS 不可用）
- SVHN `.mat` 文件已存在：`train_32x32.mat`、`extra_32x32.mat`、`test_32x32.mat`

## 2026-02-22 — 低比特基线（待持续补全）

计划网格（CNN，1 epoch，seed=42，batch=256）：

- SVHN / CIFAR-10 / CIFAR-100
- W2A2 + `{maxabs, meanabs2.5}`
- W1A2 + `{maxabs, meanabs2.5}`

### 结果（CNN，1 epoch，SGD lr=0.01）

下表所有实验均使用：`--quant balanced --equalize recursive_mean --no_tqdm --no_save_last --no_save_optimizer`。

| 数据集 | 位宽 | `scale_mode` | 训练acc | 验证acc | 测试acc | 备注 | 输出目录 |
|---|---:|---|---:|---:|---:|---|---|
| SVHN | W2A2 | `maxabs` | 0.1080 | 0.0831 | 0.0670 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/svhn_cnn_balanced_w2a2_e1_maxabs` |
| SVHN | W2A2 | `meanabs2.5` | 0.8821 | 0.9640 | 0.9373 | 稳定 | `runs/lowbits_2026-02-22/svhn_cnn_balanced_w2a2_e1_meanabs2.5` |
| SVHN | W1A2 | `maxabs` | 0.0841 | 0.0831 | 0.0670 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/svhn_cnn_balanced_w1a2_e1_maxabs` |
| SVHN | W1A2 | `meanabs2.5` | 0.1718 | 0.1721 | 0.1959 | 稳定但很差 | `runs/lowbits_2026-02-22/svhn_cnn_balanced_w1a2_e1_meanabs2.5` |
| CIFAR-10 | W2A2 | `maxabs` | 0.3059 | 0.2930 | 0.2899 | 稳定但偏弱 | `runs/lowbits_2026-02-22/cifar10_cnn_balanced_w2a2_e1_maxabs` |
| CIFAR-10 | W2A2 | `meanabs2.5` | 0.3831 | 0.5132 | 0.5137 | 稳定 | `runs/lowbits_2026-02-22/cifar10_cnn_balanced_w2a2_e1_meanabs2.5` |
| CIFAR-10 | W1A2 | `maxabs` | 0.1008 | 0.1060 | 0.1000 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/cifar10_cnn_balanced_w1a2_e1_maxabs` |
| CIFAR-10 | W1A2 | `meanabs2.5` | 0.1023 | 0.0958 | 0.1001 | 稳定但接近随机 | `runs/lowbits_2026-02-22/cifar10_cnn_balanced_w1a2_e1_meanabs2.5` |
| CIFAR-100 | W2A2 | `maxabs` | 0.0568 | 0.1020 | 0.0970 | 稳定但偏弱 | `runs/lowbits_2026-02-22/cifar100_cnn_balanced_w2a2_e1_maxabs` |
| CIFAR-100 | W2A2 | `meanabs2.5` | 0.0769 | 0.1638 | 0.1626 | 稳定 | `runs/lowbits_2026-02-22/cifar100_cnn_balanced_w2a2_e1_meanabs2.5` |
| CIFAR-100 | W1A2 | `maxabs` | 0.0128 | 0.0108 | 0.0100 | 稳定但接近随机 | `runs/lowbits_2026-02-22/cifar100_cnn_balanced_w1a2_e1_maxabs` |
| CIFAR-100 | W1A2 | `meanabs2.5` | 0.0415 | 0.0840 | 0.0805 | 稳定但偏弱 | `runs/lowbits_2026-02-22/cifar100_cnn_balanced_w1a2_e1_meanabs2.5` |

即时结论（仍需更多 seed/epoch 才能“定论”）：

- 对 **W2A2**：`meanabs2.5` 一直更好（并避免了 SVHN 上 `maxabs` 的 `NaN` 崩溃）。
- 对 **W1A2**：`maxabs` 经常崩溃成 `NaN`（SVHN/CIFAR-10）。`meanabs2.5` 虽能避免 `NaN`，但在该设置下 **CNN+SGD 基本不学习**（接近随机）。

## 2026-02-22 — 消融：让 W1A2 学得动（CNN）

目标：找到让 `W1A2` 可训练的**最小改动集合**，然后在 `svhn/cifar10/cifar100` 上验证一致性。

### CIFAR-10（1 epoch）— 关键消融

下表所有实验均使用：`--dataset cifar10 --model cnn --quant balanced --epochs 1 --batch_size 256 --seed 42 --hflip --no_save_last --no_save_optimizer`。

| 相对基线的改动（基线：`W1A2 + meanabs2.5`） | `scale_mode` | 训练acc | 验证acc | 测试acc | 输出目录 |
|---|---|---:|---:|---:|---|
| 基线 | `meanabs2.5` | 0.1023 | 0.0958 | 0.1001 | `runs/lowbits_2026-02-22/cifar10_cnn_balanced_w1a2_e1_meanabs2.5` |
| `+ fp32_first_last` | `meanabs2.5` | 0.3922 | 0.5294 | 0.5193 | `runs/lowbits_2026-02-22/ablate_w1a2/cifar10_cnn_balanced_w1a2_e1_meanabs2.5_fp32fl` |
| `+ fp32_first_last` | `maxabs` | 0.3858 | 0.4756 | 0.4683 | `runs/lowbits_2026-02-22/ablate_w1a2/cifar10_cnn_balanced_w1a2_e1_maxabs_fp32fl` |
| `+ fp32_first_last + w_transform=tanh` | `meanabs2.5` | 0.3846 | 0.5256 | 0.5218 | `runs/lowbits_2026-02-22/ablate_w1a2/cifar10_cnn_balanced_w1a2_e1_meanabs2.5_fp32fl_tanh` |
| `+ fp32_first_last + w_bias_mode=mean` | `meanabs2.5` | 0.3859 | 0.4730 | 0.4772 | `runs/lowbits_2026-02-22/ablate_w1a2/cifar10_cnn_balanced_w1a2_e1_meanabs2.5_fp32fl_biasmean` |
| 激活消融：`W1A32` | `meanabs2.5` | 0.1022 | 0.1042 | 0.1007 | `runs/lowbits_2026-02-22/ablate_w1a2/cifar10_cnn_balanced_w1a32_e1_meanabs2.5` |

当前解释（暂定）：

- 让 W1A2 最关键的稳定器是 **`--fp32_first_last`**（没有它，即使激活变成 fp32，W=1 仍不学习）。
- 在 `--fp32_first_last` 下，`meanabs2.5` 依然优于 `maxabs`（但差距变小）。
- `w_transform=tanh` 影响很小；`w_bias_mode=mean` 反而变差。

### 跨数据集 sanity check（W1A2 + fp32_first_last）

| 数据集 | `scale_mode` | 训练acc | 验证acc | 测试acc | 输出目录 |
|---|---|---:|---:|---:|---|
| SVHN | `meanabs2.5` | 0.8968 | 0.9632 | 0.9342 | `runs/lowbits_2026-02-22/ablate_w1a2/svhn_cnn_balanced_w1a2_e1_meanabs2.5_fp32fl` |
| SVHN | `maxabs` | 0.8933 | 0.9619 | 0.9304 | `runs/lowbits_2026-02-22/ablate_w1a2/svhn_cnn_balanced_w1a2_e1_maxabs_fp32fl` |
| CIFAR-10 | `meanabs2.5` | 0.3922 | 0.5294 | 0.5193 | `runs/lowbits_2026-02-22/ablate_w1a2/cifar10_cnn_balanced_w1a2_e1_meanabs2.5_fp32fl` |
| CIFAR-100 | `meanabs2.5` | 0.0862 | 0.1672 | 0.1758 | `runs/lowbits_2026-02-22/ablate_w1a2/cifar100_cnn_balanced_w1a2_e1_meanabs2.5_fp32fl` |

### 最小性：`fp32_last` vs `fp32_first`（W1A2，meanabs2.5）

新增开关：`--fp32_first` 与 `--fp32_last`（见 `README.md`）。

所有实验：1 epoch，与上面相同设置，`scale_mode=meanabs2.5`。

| 数据集 | `fp32_first` | `fp32_last` | 训练acc | 验证acc | 测试acc | 备注 | 输出目录 |
|---|---:|---:|---:|---:|---:|---|---|
| SVHN | ✅ | ❌ | 0.1717 | 0.1721 | 0.1959 | **不学习** | `runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/svhn_cnn_w1a2_meanabs2.5_fp32first_e1_seed42` |
| SVHN | ❌ | ✅ | 0.8888 | 0.9556 | 0.9242 | 学得动 | `runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/svhn_cnn_w1a2_meanabs2.5_fp32last_e1_seed42` |
| CIFAR-10 | ✅ | ❌ | 0.1048 | 0.1038 | 0.1002 | **不学习** | `runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/cifar10_cnn_w1a2_meanabs2.5_fp32first_e1_seed42` |
| CIFAR-10 | ❌ | ✅ | 0.3773 | 0.4944 | 0.4920 | 学得动 | `runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/cifar10_cnn_w1a2_meanabs2.5_fp32last_e1_seed42` |
| CIFAR-100 | ✅ | ❌ | 0.0375 | 0.0682 | 0.0667 | **不学习** | `runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/cifar100_cnn_w1a2_meanabs2.5_fp32first_e1_seed42` |
| CIFAR-100 | ❌ | ✅ | 0.0845 | 0.1568 | 0.1586 | 学得动（1 epoch 仍偏弱） | `runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/cifar100_cnn_w1a2_meanabs2.5_fp32last_e1_seed42` |

小结（暂定）：在该设置下，W1A2 里 **`--fp32_last` 是最小有效开关**；仅 `--fp32_first` 不够。

### W1A2：`scale_mode=maxabs` + fp32 分离

快速检查（1 epoch，seed=42）：

| 数据集 | `fp32_first` | `fp32_last` | `scale_mode` | 测试acc | 备注 | 输出目录 |
|---|---:|---:|---|---:|---|---|
| SVHN | ✅ | ❌ | `maxabs` | 0.0670 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/ablate_scale_maxabs_w1a2/svhn_w1a2_maxabs_fp32first_e1_s42` |
| SVHN | ❌ | ✅ | `maxabs` | 0.9121 | 稳定 | `runs/lowbits_2026-02-22/ablate_scale_maxabs_w1a2/svhn_w1a2_maxabs_fp32last_e1_s42` |
| CIFAR-10 | ✅ | ❌ | `maxabs` | 0.1000 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/ablate_scale_maxabs_w1a2/cifar10_w1a2_maxabs_fp32first_e1_s42` |
| CIFAR-10 | ❌ | ✅ | `maxabs` | 0.4656 | 稳定 | `runs/lowbits_2026-02-22/ablate_scale_maxabs_w1a2/cifar10_w1a2_maxabs_fp32last_e1_s42` |

## 2026-02-22 — 重复实验：W2A2（更多 seed）

目标：检查 W2A2 结论是否对 seed 鲁棒（并进一步证明 `maxabs` 的崩溃现象）。

### SVHN W2A2（seed 0/1）

| `scale_mode` | Seed | 训练acc | 验证acc | 测试acc | 备注 | 输出目录 |
|---|---:|---:|---:|---:|---|---|
| `meanabs2.5` | 0 | 0.8878 | 0.9642 | 0.9375 | 稳定 | `runs/lowbits_2026-02-22/repeats_w2a2/svhn_cnn_balanced_w2a2_e1_meanabs2.5_seed0` |
| `meanabs2.5` | 1 | 0.8236 | 0.9559 | 0.9252 | 稳定 | `runs/lowbits_2026-02-22/repeats_w2a2/svhn_cnn_balanced_w2a2_e1_meanabs2.5_seed1` |
| `maxabs` | 0 | 0.0999 | 0.0834 | 0.0670 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/repeats_w2a2/svhn_cnn_balanced_w2a2_e1_maxabs_seed0` |
| `maxabs` | 1 | 0.1222 | 0.0826 | 0.0670 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/repeats_w2a2/svhn_cnn_balanced_w2a2_e1_maxabs_seed1` |

### CIFAR：W2A2（seed 0）

| 数据集 | `scale_mode` | Seed | 训练acc | 验证acc | 测试acc | 备注 | 输出目录 |
|---|---|---:|---:|---:|---:|---|---|
| CIFAR-10 | `meanabs2.5` | 0 | 0.3792 | 0.4488 | 0.4638 | 稳定 | `runs/lowbits_2026-02-22/repeats_w2a2/cifar10_cnn_balanced_w2a2_e1_meanabs2.5_seed0` |
| CIFAR-10 | `maxabs` | 0 | 0.2248 | 0.1022 | 0.1000 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/repeats_w2a2/cifar10_cnn_balanced_w2a2_e1_maxabs_seed0` |
| CIFAR-100 | `meanabs2.5` | 0 | 0.0756 | 0.1396 | 0.1356 | 稳定 | `runs/lowbits_2026-02-22/repeats_w2a2/cifar100_cnn_balanced_w2a2_e1_meanabs2.5_seed0` |
| CIFAR-100 | `maxabs` | 0 | 0.0464 | 0.0594 | 0.0644 | 稳定但更差 | `runs/lowbits_2026-02-22/repeats_w2a2/cifar100_cnn_balanced_w2a2_e1_maxabs_seed0` |

## 2026-02-22 — 消融：`fp32_first_last` 能否挽救 W2A2+maxabs？

动机：`W2A2 + maxabs` 在 SVHN/CIFAR-10 上会崩溃成 `NaN`。由于 `fp32_first_last` 对 W1A2 很关键，这里测试它是否也能稳定 W2A2（`maxabs`）。

所有实验：CNN + Balanced，1 epoch，`--fp32_first_last`，seed 如表所示。

| 数据集 | Seed | `scale_mode` | 训练acc | 验证acc | 测试acc | 备注 | 输出目录 |
|---|---:|---|---:|---:|---:|---|---|
| SVHN | 0 | `maxabs` | 0.9116 | 0.9676 | 0.9416 | **无 NaN** | `runs/lowbits_2026-02-22/ablate_w2a2/svhn_cnn_balanced_w2a2_e1_maxabs_fp32fl_seed0` |
| SVHN | 1 | `maxabs` | 0.9106 | 0.9626 | 0.9350 | **无 NaN** | `runs/lowbits_2026-02-22/ablate_w2a2/svhn_cnn_balanced_w2a2_e1_maxabs_fp32fl_seed1` |
| CIFAR-10 | 0 | `maxabs` | 0.4150 | 0.5140 | 0.5222 | **无 NaN** | `runs/lowbits_2026-02-22/ablate_w2a2/cifar10_cnn_balanced_w2a2_e1_maxabs_fp32fl_seed0` |
| CIFAR-100 | 0 | `maxabs` | 0.0877 | 0.1658 | 0.1792 | 相对 no-fp32fl 有提升 | `runs/lowbits_2026-02-22/ablate_w2a2/cifar100_cnn_balanced_w2a2_e1_maxabs_fp32fl_seed0` |

### 最小性：`fp32_last` vs `fp32_first`（W2A2，maxabs）

所有实验：1 epoch，`W2A2 + scale_mode=maxabs`。

| 数据集 | `fp32_first` | `fp32_last` | 训练acc | 验证acc | 测试acc | 备注 | 输出目录 |
|---|---:|---:|---:|---:|---:|---|---|
| SVHN | ✅ | ❌ | 0.1058 | 0.0831 | 0.0670 | `loss=NaN` 崩溃 | `runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/svhn_cnn_w2a2_maxabs_fp32first_e1_seed42` |
| SVHN | ❌ | ✅ | 0.9047 | 0.9691 | 0.9422 | 稳定 | `runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/svhn_cnn_w2a2_maxabs_fp32last_e1_seed42` |
| CIFAR-10 | ✅ | ❌ | 0.3250 | 0.3292 | 0.3286 | 稳定但偏弱 | `runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/cifar10_cnn_w2a2_maxabs_fp32first_e1_seed42` |
| CIFAR-10 | ❌ | ✅ | 0.4018 | 0.5418 | 0.5411 | 稳定 | `runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/cifar10_cnn_w2a2_maxabs_fp32last_e1_seed42` |
| CIFAR-100 | ✅ | ❌ | 0.0619 | 0.0996 | 0.0982 | 与基线接近 | `runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/cifar100_cnn_w2a2_maxabs_fp32first_e1_seed42` |
| CIFAR-100 | ❌ | ✅ | 0.0910 | 0.1846 | 0.1925 | 有提升 | `runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/cifar100_cnn_w2a2_maxabs_fp32last_e1_seed42` |

结论：对 `W2A2 + maxabs`，**`--fp32_last` 足够**防止崩溃并恢复精度；仅 `--fp32_first` 不够。

## 2026-02-22 — 消融：当 W=1 时，激活位宽是否重要（CIFAR）

问题：在 `W=1`（BQ）且采用最小稳定开关 `--fp32_last` 的前提下，提高激活精度是否能明显提升？

设置（所有实验）：

- CNN + `--quant balanced --scale_mode meanabs2.5 --fp32_last`
- 1 epoch，seed=42，batch=256
- CIFAR 开启 `--hflip`

### CIFAR-10（W1，A∈{2,4,8,32}）

| `a_bits` | 训练acc | 验证acc | 测试acc | 输出目录 |
|---:|---:|---:|---:|---|
| 2 | 0.3801 | 0.4910 | 0.4839 | `runs/lowbits_2026-02-22/ablate_a_bits_w1/cifar10/w1_a2_meanabs2.5_fp32last_e1_s42` |
| 4 | 0.3848 | 0.4906 | 0.4888 | `runs/lowbits_2026-02-22/ablate_a_bits_w1/cifar10/w1_a4_meanabs2.5_fp32last_e1_s42` |
| 8 | 0.3848 | 0.5010 | 0.4916 | `runs/lowbits_2026-02-22/ablate_a_bits_w1/cifar10/w1_a8_meanabs2.5_fp32last_e1_s42` |
| 32 | 0.3832 | 0.5180 | 0.5059 | `runs/lowbits_2026-02-22/ablate_a_bits_w1/cifar10/w1_a32_meanabs2.5_fp32last_e1_s42` |

### CIFAR-100（W1，A∈{2,4,8,32}）

| `a_bits` | 训练acc | 验证acc | 测试acc | 输出目录 |
|---:|---:|---:|---:|---|
| 2 | 0.0848 | 0.1670 | 0.1704 | `runs/lowbits_2026-02-22/ablate_a_bits_w1/cifar100/w1_a2_meanabs2.5_fp32last_e1_s42` |
| 4 | 0.0858 | 0.1686 | 0.1710 | `runs/lowbits_2026-02-22/ablate_a_bits_w1/cifar100/w1_a4_meanabs2.5_fp32last_e1_s42` |
| 8 | 0.0875 | 0.1638 | 0.1689 | `runs/lowbits_2026-02-22/ablate_a_bits_w1/cifar100/w1_a8_meanabs2.5_fp32last_e1_s42` |
| 32 | 0.0847 | 0.1600 | 0.1598 | `runs/lowbits_2026-02-22/ablate_a_bits_w1/cifar100/w1_a32_meanabs2.5_fp32last_e1_s42` |

暂时结论：激活位宽的影响属于**次要因素**，远弱于（a）权重位宽（W=1/2）与（b）`fp32_last` / `scale_mode` 的稳定性。CIFAR-10 上略有正向趋势；CIFAR-100 在 1 epoch 下较混乱。

## 2026-02-22 — 重复实验：W1A2 + fp32_last（CIFAR）

目标：检查“最小配方”（`W1A2 + meanabs2.5 + fp32_last`）对 seed 的敏感性。

所有实验：1 epoch，CNN + Balanced，`scale_mode=meanabs2.5`，`--fp32_last`，CIFAR 开启 `--hflip`。

| 数据集 | Seed | 训练acc | 验证acc | 测试acc | 输出目录 |
|---|---:|---:|---:|---:|---|
| CIFAR-10 | 0 | 0.3856 | 0.4916 | 0.4959 | `runs/lowbits_2026-02-22/repeats_w1a2_fp32last/cifar10_w1a2_meanabs2.5_fp32last_e1_seed0` |
| CIFAR-10 | 1 | 0.3805 | 0.4668 | 0.4643 | `runs/lowbits_2026-02-22/repeats_w1a2_fp32last/cifar10_w1a2_meanabs2.5_fp32last_e1_seed1` |
| CIFAR-100 | 0 | 0.0789 | 0.1504 | 0.1536 | `runs/lowbits_2026-02-22/repeats_w1a2_fp32last/cifar100_w1a2_meanabs2.5_fp32last_e1_seed0` |
| CIFAR-100 | 1 | 0.0795 | 0.1614 | 0.1591 | `runs/lowbits_2026-02-22/repeats_w1a2_fp32last/cifar100_w1a2_meanabs2.5_fp32last_e1_seed1` |

## 2026-02-22 — 更长一点：CIFAR-10（5 epochs）

目标：摆脱 1 epoch 的噪声，对比几种“已经稳定”的低比特配方在 CIFAR-10 上的表现。

设置（所有实验）：

- `--dataset cifar10 --model cnn --quant balanced --epochs 5 --seed 42 --batch_size 256 --hflip`
- `--no_save_last --no_save_optimizer --no_save_best`

| 配置 | 最佳验证acc | 测试acc | 输出目录 |
|---|---:|---:|---|
| W2A2 + `meanabs2.5` | 0.7172 | 0.7122 | `runs/lowbits_2026-02-22/cifar10_e5_compare/w2a2_meanabs2.5_e5_s42` |
| W2A2 + `maxabs` + `--fp32_last` | 0.7576 | 0.7447 | `runs/lowbits_2026-02-22/cifar10_e5_compare/w2a2_maxabs_fp32last_e5_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.7060 | 0.6929 | `runs/lowbits_2026-02-22/cifar10_e5_compare/w1a2_meanabs2.5_fp32last_e5_s42` |
| W1A2 + `maxabs` + `--fp32_last` | 0.6856 | 0.6845 | `runs/lowbits_2026-02-22/cifar10_e5_compare/w1a2_maxabs_fp32last_e5_s42` |

观察（暂定）：`meanabs2.5` 对稳定性很关键；但一旦通过 `--fp32_last` 等手段稳定住，`maxabs` 也可能做到不差甚至更好（此处示例：CIFAR-10 的 W2A2）。

## 2026-02-22 — 更长一点：SVHN（5 epochs，CNN，use_extra=True）

目标：在 SVHN 上也跑一个 5-epoch 对比，看看 CIFAR-10 上的趋势是否一致（尤其是 “稳定之后 maxabs 可能不差”）。

设置（所有实验）：

- `--dataset svhn --model cnn --quant balanced --epochs 5 --seed 42 --batch_size 256`
- `--no_tqdm --no_save_last --no_save_optimizer --no_save_best`
- 注意：SVHN 默认 `--use_extra` 启用（此处为 `use_extra=True`）

| 配置 | 最佳验证acc | 测试acc | 输出目录 |
|---|---:|---:|---|
| W2A2 + `meanabs2.5` | 0.9812 | 0.9667 | `runs/lowbits_2026-02-22/svhn_e5_compare/w2a2_meanabs2.5_e5_s42` |
| W2A2 + `maxabs` + `--fp32_last` | 0.9818 | 0.9671 | `runs/lowbits_2026-02-22/svhn_e5_compare/w2a2_maxabs_fp32last_e5_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.9793 | 0.9596 | `runs/lowbits_2026-02-22/svhn_e5_compare/w1a2_meanabs2.5_fp32last_e5_s42` |
| W1A2 + `maxabs` + `--fp32_last` | 0.9781 | 0.9587 | `runs/lowbits_2026-02-22/svhn_e5_compare/w1a2_maxabs_fp32last_e5_s42` |

观察（暂定）：在 SVHN 上，稳定住之后（W2A2 用 `meanabs2.5` 或 `maxabs+fp32_last`；W1A2 用 `fp32_last`），`maxabs` 与 `meanabs2.5` 的精度差距很小。

## 2026-02-22 — ViT：低比特快速扫一遍（1 epoch）

目标：验证在 ViT 上，低比特（W2A2/W1A2）的稳定性与 “`--fp32_last` 是否仍是关键最小改动” 的结论是否一致。

### CIFAR-10（ViT，1 epoch，seed=42）

设置（所有实验）：

- `--dataset cifar10 --model vit --epochs 1 --seed 42 --batch_size 256 --hflip`
- `--no_tqdm --no_save_last --no_save_optimizer --no_save_best`

| 配置 | 训练acc | 验证acc | 测试acc | 输出目录 |
|---|---:|---:|---:|---|
| fp32（`--quant none`） | 0.2993 | 0.2984 | 0.3060 | `runs/lowbits_2026-02-22/vit_cifar10_e1/fp32_sgd_e1_s42` |
| W2A2 + `meanabs2.5` | 0.2931 | 0.3324 | 0.3414 | `runs/lowbits_2026-02-22/vit_cifar10_e1/w2a2_meanabs2.5_e1_s42` |
| W2A2 + `maxabs` | 0.2927 | 0.3372 | 0.3417 | `runs/lowbits_2026-02-22/vit_cifar10_e1/w2a2_maxabs_e1_s42` |
| W2A2 + `maxabs` + `--fp32_last` | 0.2999 | 0.3204 | 0.3343 | `runs/lowbits_2026-02-22/vit_cifar10_e1/w2a2_maxabs_fp32last_e1_s42` |
| W1A2 + `meanabs2.5` | 0.2588 | 0.2968 | 0.2993 | `runs/lowbits_2026-02-22/vit_cifar10_e1/w1a2_meanabs2.5_e1_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.2998 | 0.3276 | 0.3268 | `runs/lowbits_2026-02-22/vit_cifar10_e1/w1a2_meanabs2.5_fp32last_e1_s42` |

观察（暂定）：

- ViT 上 **`maxabs` 没有出现 NaN 崩溃**（与 CNN 不同）。
- ViT 的 W1A2 在该 1-epoch 设置下**不至于完全随机**；但 `--fp32_last` 仍能带来可见收益。

### CIFAR-100（ViT，1 epoch，seed=42）

设置同上（换 `--dataset cifar100`）。

| 配置 | 训练acc | 验证acc | 测试acc | 输出目录 |
|---|---:|---:|---:|---|
| fp32（`--quant none`） | 0.0700 | 0.0998 | 0.0933 | `runs/lowbits_2026-02-22/vit_cifar100_e1/fp32_sgd_e1_s42` |
| W2A2 + `meanabs2.5` | 0.0697 | 0.0940 | 0.0995 | `runs/lowbits_2026-02-22/vit_cifar100_e1/w2a2_meanabs2.5_e1_s42` |
| W2A2 + `maxabs` | 0.0698 | 0.0974 | 0.0957 | `runs/lowbits_2026-02-22/vit_cifar100_e1/w2a2_maxabs_e1_s42` |
| W2A2 + `maxabs` + `--fp32_last` | 0.0698 | 0.1024 | 0.0961 | `runs/lowbits_2026-02-22/vit_cifar100_e1/w2a2_maxabs_fp32last_e1_s42` |
| W1A2 + `meanabs2.5` | 0.0602 | 0.0900 | 0.0941 | `runs/lowbits_2026-02-22/vit_cifar100_e1/w1a2_meanabs2.5_e1_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.0687 | 0.0942 | 0.0960 | `runs/lowbits_2026-02-22/vit_cifar100_e1/w1a2_meanabs2.5_fp32last_e1_s42` |

备注：CIFAR-100 的 1 epoch 仍处于“刚起步”，不同配方差异信号较弱；需要更长训练再下结论。

### SVHN（ViT，no_extra，1 epoch，seed=42）

设置（所有实验）：

- `--dataset svhn --model vit --epochs 1 --seed 42 --batch_size 256 --no_extra`
- `--no_tqdm --no_save_last --no_save_optimizer --no_save_best`

| 配置 | 训练acc | 验证acc | 测试acc | 输出目录 |
|---|---:|---:|---:|---|
| fp32（`--quant none`） | 0.1906 | 0.2162 | 0.2184 | `runs/lowbits_2026-02-22/vit_svhn_noextra_e1/fp32_sgd_e1_s42` |
| W2A2 + `meanabs2.5` | 0.1868 | 0.2184 | 0.2256 | `runs/lowbits_2026-02-22/vit_svhn_noextra_e1/w2a2_meanabs2.5_e1_s42` |
| W2A2 + `maxabs` | 0.1931 | 0.2434 | 0.2447 | `runs/lowbits_2026-02-22/vit_svhn_noextra_e1/w2a2_maxabs_e1_s42` |
| W1A2 + `meanabs2.5` | 0.1609 | 0.1912 | 0.1961 | `runs/lowbits_2026-02-22/vit_svhn_noextra_e1/w1a2_meanabs2.5_e1_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.2207 | 0.2654 | 0.2660 | `runs/lowbits_2026-02-22/vit_svhn_noextra_e1/w1a2_meanabs2.5_fp32last_e1_s42` |

备注：这里用 `--no_extra` 主要是为了加快迭代；后续如要与 CNN/SVHN(use_extra=True) 对齐，需要再补 full-data 跑。

### SVHN（ViT，use_extra=True，1 epoch，seed=42）

设置（所有实验）：

- `--dataset svhn --model vit --epochs 1 --seed 42 --batch_size 256`（默认 `use_extra=True`）
- `--no_tqdm --no_save_last --no_save_optimizer --no_save_best`

| 配置 | 训练acc | 验证acc | 测试acc | 输出目录 |
|---|---:|---:|---:|---|
| fp32（`--quant none`） | 0.5607 | 0.7951 | 0.7420 | `runs/lowbits_2026-02-22/vit_svhn_useextra_e1/fp32_sgd_e1_s42` |
| W2A2 + `meanabs2.5` | 0.5197 | 0.7604 | 0.6988 | `runs/lowbits_2026-02-22/vit_svhn_useextra_e1/w2a2_meanabs2.5_e1_s42` |
| W2A2 + `maxabs` | 0.5187 | 0.7536 | 0.6935 | `runs/lowbits_2026-02-22/vit_svhn_useextra_e1/w2a2_maxabs_e1_s42` |
| W1A2 + `meanabs2.5` | 0.4475 | 0.7065 | 0.6472 | `runs/lowbits_2026-02-22/vit_svhn_useextra_e1/w1a2_meanabs2.5_e1_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.5057 | 0.7667 | 0.7081 | `runs/lowbits_2026-02-22/vit_svhn_useextra_e1/w1a2_meanabs2.5_fp32last_e1_s42` |

观察（暂定）：在 SVHN（full-data）上，ViT 的 W1A2 明显受益于 `--fp32_last`；W2A2 下 `meanabs2.5` 与 `maxabs` 差异不大且都稳定。

## 2026-02-22 — ViT：CIFAR-10（5 epochs）

目标：把 ViT 的 1-epoch 观察延长到 5 epochs，看低比特差异是否仍存在。

设置（所有实验）：

- `--dataset cifar10 --model vit --epochs 5 --seed 42 --batch_size 256 --hflip`
- `--no_tqdm --no_save_last --no_save_optimizer --no_save_best`

| 配置 | 最佳验证acc | 测试acc | 输出目录 |
|---|---:|---:|---|
| fp32（`--quant none`） | 0.4396 | 0.4323 | `runs/lowbits_2026-02-22/vit_cifar10_e5_compare/fp32_sgd_e5_s42` |
| W2A2 + `meanabs2.5` | 0.4310 | 0.4290 | `runs/lowbits_2026-02-22/vit_cifar10_e5_compare/w2a2_meanabs2.5_e5_s42` |
| W2A2 + `maxabs` | 0.4268 | 0.4245 | `runs/lowbits_2026-02-22/vit_cifar10_e5_compare/w2a2_maxabs_e5_s42` |
| W1A2 + `meanabs2.5` | 0.4094 | 0.4092 | `runs/lowbits_2026-02-22/vit_cifar10_e5_compare/w1a2_meanabs2.5_e5_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.4226 | 0.4266 | `runs/lowbits_2026-02-22/vit_cifar10_e5_compare/w1a2_meanabs2.5_fp32last_e5_s42` |

观察（暂定）：

- W2A2 下 `maxabs` / `meanabs2.5` 差异很小，且都稳定。
- W1A2 下 `--fp32_last` 仍有帮助（测试从 ~0.409 提升到 ~0.427），接近 fp32 baseline。

## 2026-02-22 — ViT：CIFAR-100（5 epochs）

目标：把 CIFAR-100 也延长到 5 epochs，看看 W1A2 是否仍需要 `--fp32_last` 才能接近 fp32。

设置（所有实验）：

- `--dataset cifar100 --model vit --epochs 5 --seed 42 --batch_size 256 --hflip`
- `--no_tqdm --no_save_last --no_save_optimizer --no_save_best`

| 配置 | 最佳验证acc | 测试acc | 输出目录 |
|---|---:|---:|---|
| fp32（`--quant none`） | 0.1758 | 0.1708 | `runs/lowbits_2026-02-22/vit_cifar100_e5_compare/fp32_sgd_e5_s42` |
| W2A2 + `meanabs2.5` | 0.1758 | 0.1739 | `runs/lowbits_2026-02-22/vit_cifar100_e5_compare/w2a2_meanabs2.5_e5_s42` |
| W2A2 + `maxabs` | 0.1750 | 0.1623 | `runs/lowbits_2026-02-22/vit_cifar100_e5_compare/w2a2_maxabs_e5_s42` |
| W1A2 + `meanabs2.5` | 0.1714 | 0.1647 | `runs/lowbits_2026-02-22/vit_cifar100_e5_compare/w1a2_meanabs2.5_e5_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.1722 | 0.1719 | `runs/lowbits_2026-02-22/vit_cifar100_e5_compare/w1a2_meanabs2.5_fp32last_e5_s42` |

观察（暂定）：对 W1A2，`--fp32_last` 仍能把测试精度从 ~0.165 拉回到 ~0.172，接近 fp32 baseline。

## 2026-02-23 — ViT：CIFAR-10 超参调优（20 epochs，fp32）

动机：此前 ViT（默认超参 + SGD）精度明显落后于 CNN；先把 ViT 的 fp32 训练“拉起来”，再谈低比特结论是否一致。

设置（所有实验）：

- `--dataset cifar10 --model vit --epochs 20 --seed 42 --batch_size 256 --hflip`
- `--no_tqdm --no_save_last --no_save_optimizer --no_save_best`
- fp32：`--quant none --w_bits 32 --a_bits 32`

| 配置 | 最佳验证acc | 测试acc | 输出目录 |
|---|---:|---:|---|
| SGD `lr=1e-2 wd=5e-4 drop=0`（baseline） | 0.5952 | 0.5755 | `runs/lowbits_2026-02-22/vit_tune_cifar10_e20/sgd_lr1e-2_wd5e-4_drop0_e20` |
| AdamW `lr=3e-4 wd=0.05` + `patch_norm` + `pool=mean` + `drop=0.1` | 0.6484 | 0.6334 | `runs/lowbits_2026-02-22/vit_tune_cifar10_e20/adamw_lr3e-4_wd5e-2_drop0.1_mean_e20` |
| AdamW `lr=5e-4 wd=0.05` + `patch_norm` + `pool=cls` + `drop=0.1` | 0.6592 | 0.6433 | `runs/lowbits_2026-02-22/vit_tune_cifar10_e20/adamw_lr5e-4_wd5e-2_drop0.1_cls_e20` |
| AdamW `lr=5e-4 wd=0.05` + `patch_norm` + `pool=mean` + `drop=0.1` | 0.6796 | 0.6693 | `runs/lowbits_2026-02-22/vit_tune_cifar10_e20/adamw_lr5e-4_wd5e-2_drop0.1_mean_e20` |
| AdamW `lr=1e-3 wd=0.05` + `patch_norm` + `pool=mean` + `drop=0.1` | 0.6912 | 0.6842 | `runs/lowbits_2026-02-22/vit_tune_cifar10_e20/adamw_lr1e-3_wd5e-2_drop0.1_mean_e20` |
| **AdamW `lr=5e-4 wd=0.05` + `patch=4` + `patch_norm` + `pool=mean` + `drop=0.1`** | **0.7524** | **0.7414** | `runs/lowbits_2026-02-22/vit_tune_cifar10_e20/adamw_lr5e-4_wd5e-2_drop0.1_p4_mean_e20` |

小结（暂定）：

- ViT 的瓶颈很大一部分来自“训练配方”，而不是“天生比 CNN 差/只收敛慢一点”。换 AdamW + 合理 wd/drop 后提升显著。
- `patch=4`（更多 token）对 CIFAR-10 提升很大：测试从 ~0.67 提升到 ~0.74。

### ViT：CIFAR-10 更长跑（100 epochs，fp32）

采用 20-epoch 最优配方继续拉长到 100 epochs（fp32 + 低比特对照），用于判断：
（1）ViT 是否能继续追上/超过 CNN；（2）低比特 gap 是否会随训练变长而缩小/稳定。

设置（所有实验）：

- `--dataset cifar10 --model vit --epochs 100 --seed 42 --batch_size 256 --hflip`
- `--optimizer adamw --lr 5e-4 --weight_decay 0.05 --scheduler cosine`
- `--label_smoothing 0.1 --grad_clip 1.0`
- `--vit_patch 4 --vit_patch_norm --vit_pool mean --vit_drop 0.1 --vit_attn_drop 0.1`
- `--no_tqdm --no_save_last --no_save_optimizer --no_save_best`

结果（Best Val / Test）：

| 配置 | 最佳验证acc | 测试acc | 训练耗时（sum epoch time） | 输出目录 |
|---|---:|---:|---:|---|
| fp32（`--quant none`） | 0.8364 | 0.8278 | ~20.9 min | `runs/lowbits_2026-02-22/vit_tune_cifar10_e100/adamw_lr5e-4_wd5e-2_drop0.1_p4_mean_e100_s42` |
| W2A2（`maxabs`） | 0.8206 | 0.8107 | ~31.6 min | `runs/lowbits_2026-02-22/vit_tuned_lowbit_cifar10_e100/w2a2_maxabs_e100_s42` |
| W1A2（`meanabs2.5` + `--fp32_last`） | 0.8152 | 0.8104 | ~31.3 min | `runs/lowbits_2026-02-22/vit_tuned_lowbit_cifar10_e100/w1a2_meanabs2.5_fp32last_e100_s42` |

观察（暂定）：

- 这个“ViT 训练配方”确实能把 fp32 拉到 `~0.83`（test），比默认 SGD 的 5-epoch `~0.43` 高很多。
- 在该配方下，低比特（W2A2 / W1A2+fp32_last）与 fp32 的 gap 约 `~1.7%`（test 绝对值）。
