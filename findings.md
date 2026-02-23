# Findings（稳定结论）

本文件只收录**已经验证、可复现**的结论（理想情况下跨 `svhn/cifar10/cifar100` 一致）。
探索过程与不确定/未充分验证的结果请看 `notes.md`。

## 低比特（W≤2）— 待持续补强

### Balanced Quantization：`scale_mode` 对 W≤2 极其关键

在 `svhn/cifar10/cifar100` 上用 CNN + SGD（`lr=0.01`，训练 1 epoch）时，`--scale_mode maxabs` 在极低比特下可能**直接 NaN 崩溃**（多次观察到：SVHN 的 W2A2，以及 CIFAR-10 的 W2A2 / SVHN 的 W1A2）。相比之下，`--scale_mode meanabs2.5` **稳定性显著更好**，且在 W2A2 上精度也更高。

参考 runs（完整表格见 `notes.md`）：

- SVHN W2A2：`runs/lowbits_2026-02-22/svhn_cnn_balanced_w2a2_e1_maxabs`（NaN） vs `runs/lowbits_2026-02-22/svhn_cnn_balanced_w2a2_e1_meanabs2.5`（稳定/高精度）
- SVHN W2A2（更多 seed）：`runs/lowbits_2026-02-22/repeats_w2a2/svhn_cnn_balanced_w2a2_e1_maxabs_seed0` / `runs/lowbits_2026-02-22/repeats_w2a2/svhn_cnn_balanced_w2a2_e1_maxabs_seed1`（NaN） vs `runs/lowbits_2026-02-22/repeats_w2a2/svhn_cnn_balanced_w2a2_e1_meanabs2.5_seed0` / `runs/lowbits_2026-02-22/repeats_w2a2/svhn_cnn_balanced_w2a2_e1_meanabs2.5_seed1`
- CIFAR-10 W2A2：`runs/lowbits_2026-02-22/cifar10_cnn_balanced_w2a2_e1_maxabs` vs `runs/lowbits_2026-02-22/cifar10_cnn_balanced_w2a2_e1_meanabs2.5`
- CIFAR-10 W2A2（seed=0）：`runs/lowbits_2026-02-22/repeats_w2a2/cifar10_cnn_balanced_w2a2_e1_maxabs_seed0`（NaN） vs `runs/lowbits_2026-02-22/repeats_w2a2/cifar10_cnn_balanced_w2a2_e1_meanabs2.5_seed0`
- CIFAR-100 W2A2：`runs/lowbits_2026-02-22/cifar100_cnn_balanced_w2a2_e1_maxabs` vs `runs/lowbits_2026-02-22/cifar100_cnn_balanced_w2a2_e1_meanabs2.5`
  - 备注：在这些短跑里 CIFAR-100 的 `maxabs` 不一定 NaN，但整体仍明显弱于 `meanabs2.5`。

### `fp32_first_last` / `--fp32_last` 可挽救低比特不稳定（W2A2 + maxabs）

即使在 W2A2 下 `scale_mode=maxabs` 不稳定，只要把**最后一层分类头/线性层**保持 fp32（`--fp32_last`）就足以让训练稳定下来：

- SVHN：`runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/svhn_cnn_w2a2_maxabs_fp32last_e1_seed42`（稳定） vs `runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/svhn_cnn_w2a2_maxabs_fp32first_e1_seed42`（仍 NaN）
- CIFAR-10：`runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/cifar10_cnn_w2a2_maxabs_fp32last_e1_seed42`
- CIFAR-100：`runs/lowbits_2026-02-22/ablate_fp32_split_w2a2/cifar100_cnn_w2a2_maxabs_fp32last_e1_seed42`

### W1A2 基线（CNN+SGD）目前仍不学习（需额外稳定手段）

在“纯基线”（不加额外技巧）下，W1A2 要么崩溃（`maxabs`），要么在 `meanabs2.5` 下接近随机精度。示例：`runs/lowbits_2026-02-22/cifar10_cnn_balanced_w1a2_e1_maxabs` 与 `runs/lowbits_2026-02-22/cifar10_cnn_balanced_w1a2_e1_meanabs2.5`。

### W1A2 在 `--fp32_last` 下变得可训练（最小有效改动）

对 W1A2（Balanced Quantization）而言，把**最后一层分类头/线性层**保持 fp32（`--fp32_last`）是目前验证过的“最小有效改动”，并且在三个数据集上都成立：

- SVHN：`runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/svhn_cnn_w1a2_meanabs2.5_fp32last_e1_seed42`
- CIFAR-10：`runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/cifar10_cnn_w1a2_meanabs2.5_fp32last_e1_seed42`
- CIFAR-100：`runs/lowbits_2026-02-22/ablate_fp32_split_w1a2/cifar100_cnn_w1a2_meanabs2.5_fp32last_e1_seed42`

它也能防止 W1A2 + `scale_mode=maxabs` 在基线中出现的 NaN 崩溃（例：`runs/lowbits_2026-02-22/ablate_scale_maxabs_w1a2/svhn_w1a2_maxabs_fp32last_e1_s42`、`runs/lowbits_2026-02-22/ablate_scale_maxabs_w1a2/cifar10_w1a2_maxabs_fp32last_e1_s42`）。

## 速度/耗时（工程层面，和硬件强相关）

- 本仓库的量化实现是 **纯 PyTorch**（没有 bitwise/fused kernel），因此低 bit 主要用于验证量化算法与训练可行性，**不保证训练/推理更快**；在一些设置下甚至会更慢（量化本身引入额外张量算子开销）。如果目标是加速，需要专用 kernel/算子融合（不在本项目范围）。
