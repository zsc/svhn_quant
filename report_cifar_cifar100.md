# CIFAR 验证报告（cifar100）

日期：2026-02-22

统一设定：

- `--device mps --quant balanced --equalize recursive_mean`
- `--batch_size 256 --seed 42 --val_split 0.1 --no_tqdm`
- 数据增强：`RandomCrop(32,padding=4)` + `RandomHorizontalFlip(p=0.5)`（通过 `--hflip` 显式开启）

## 1. CNN：scale_mode 稳定性（1 epoch）

| 配置 | Best Val acc | Test acc | 输出目录 |
|---|---:|---:|---|
| W8A8 | 0.1816 | 0.1751 | `sweeps/2026-02-22_cifar100_cnn_w8a8_e1` |
| W2A4 (maxabs) | 0.0806 | 0.0729 | `sweeps/2026-02-22_cifar100_cnn_w2a4_e1_maxabs` |
| W2A4 (meanabs2.5) | 0.1378 | 0.1323 | `sweeps/2026-02-22_cifar100_cnn_w2a4_e1_meanabs2.5` |

## 2. ViT：Dropout 是否有帮助？（SGD，10 epochs）

| 配置 | Best Val acc | Test acc | ΔTest vs drop0 | 输出目录 |
|---|---:|---:|---:|---|
| drop=0.0 | 0.1494 | 0.1386 | +0.0000 | `sweeps/2026-02-22_cifar100_vit_sgd_drop0_w2a4_meanabs2.5_e10` |
| drop=0.1 | 0.1116 | 0.1068 | -0.0318 | `sweeps/2026-02-22_cifar100_vit_sgd_drop0.1_w2a4_meanabs2.5_e10` |

## 3. ViT：best recipe（AdamW + wd/clip/mean-pool/patch-norm，10 epochs）

| 配置 | Best Val acc | Test acc | ΔTest vs SGD(drop0) | 输出目录 |
|---|---:|---:|---:|---|
| best recipe | 0.2800 | 0.2764 | +0.1378 | `sweeps/2026-02-22_cifar100_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10` |

## 4. ViT：四个 trick 的 ablation（AdamW，5 epochs）

| 变更（相对 baseline） | Best Val acc | Test acc | ΔTest | 输出目录 |
|---|---:|---:|---:|---|
| baseline（full） | 0.1920 | 0.1910 | +0.0000 | `sweeps/2026-02-22_cifar100_vit_ablate_full_w2a4_meanabs2.5_adamw_e5` |
| 去掉 weight decay（wd=0） | 0.1950 | 0.1908 | -0.0002 | `sweeps/2026-02-22_cifar100_vit_ablate_wd0_w2a4_meanabs2.5_adamw_e5` |
| 去掉 grad clip（clip=0） | 0.2030 | 0.1995 | +0.0085 | `sweeps/2026-02-22_cifar100_vit_ablate_clip0_w2a4_meanabs2.5_adamw_e5` |
| pooling 改成 cls（pool=cls） | 0.2062 | 0.1949 | +0.0039 | `sweeps/2026-02-22_cifar100_vit_ablate_poolcls_w2a4_meanabs2.5_adamw_e5` |
| 去掉 patch-norm（无 `--vit_patch_norm`） | 0.2168 | 0.2136 | +0.0226 | `sweeps/2026-02-22_cifar100_vit_ablate_nopnorm_w2a4_meanabs2.5_adamw_e5` |
| pool=cls + 无 patch-norm（交互） | 0.2272 | 0.2183 | +0.0273 | `sweeps/2026-02-22_cifar100_vit_ablate_poolcls_nopnorm_w2a4_meanabs2.5_adamw_e5` |

## 5. ViT：optimizer ablation（patch-norm-only，10 epochs）

| Optimizer | Best Val acc | Test acc | 输出目录 |
|---|---:|---:|---|
| AdamW | 0.2824 | 0.2747 | `sweeps/2026-02-22_cifar100_vit_patchnorm_only_adamw_w2a4_meanabs2.5_e10` |
| SGD | 0.1300 | 0.1252 | `sweeps/2026-02-22_cifar100_vit_patchnorm_only_sgd_w2a4_meanabs2.5_e10` |
| Muon | 0.3024 | 0.2993 | `sweeps/2026-02-22_cifar100_vit_patchnorm_only_muon_w2a4_meanabs2.5_e10` |

## 6. 简要结论

- `meanabs2.5` 通常比 `maxabs` 更稳（尤其是低 bit）。
- Dropout(0.1) 是否有益需要具体任务验证，但在 SVHN 上它明显变差；这里也给出对照结果。
- ViT 上，`AdamW + wd + grad clip + mean-pool + patch-norm` 往往是最关键的一组稳定性/精度“开关”。
- optimizer 对 ViT 影响很大：AdamW 通常远强于 SGD；Muon 需要按其约束（只优化 2D 参数）正确拆分。

