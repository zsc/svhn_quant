# SVHN 量化训练（PyTorch + Apple Silicon MPS）

本仓库在 **Apple Silicon（MPS）** 上用 PyTorch 训练一个 SVHN CNN（10 类数字分类），并支持：

- **Balanced Quantization（arXiv:1706.07145）**：对 **权重** 做论文算法（Algorithm 1/2）量化，并用 **STE** 训练。
- 可选 **激活量化**：通过 `--a_bits` 开启（用于 W/A bitwidth 组合实验）。
- 参考对齐：提供与 `bit-rnn/bit_utils.py` 思路一致的 `scale=mean(abs(x))*2.5 + stop_gradient(detach)`、TF rounding 规则等工程实现。
- 复用同一套训练脚本在 **CIFAR-10/100** 上做验证：`--dataset {cifar10,cifar100}`（torchvision 自动下载）。

---

## 1. 数据集（SVHN Format 2 `.mat`）

把以下文件放在 `--data_dir`（默认当前目录 `.`）：

- `train_32x32.mat`
- `extra_32x32.mat`
- `test_32x32.mat`

注意：SVHN 的标签里用 **10 表示数字 0**，loader 已在 `datasets/svhn_mat.py` 里统一做了 `10 -> 0` 映射，保证 label 范围为 `[0..9]`。

默认行为：训练集会 **自动使用 `train + extra`**。如需关闭 extra，用 `--no_extra`。

补充：CIFAR-10/100

- 通过 `--dataset cifar10` / `--dataset cifar100` 使用 torchvision 数据集（会下载并解压到 `--data_dir`）。
- 仓库已在 `.gitignore` 中忽略 `cifar-10-batches-py/`、`cifar-100-python/` 等目录，避免误提交大文件。

---

## 2. 安装依赖

必需：

- `torch`（需要支持 `mps`）
- `torchvision`
- `numpy`
- `scipy`（读 `.mat`）
- `tqdm`

可选：

- `pytest`（跑单元测试）

---

## 3. 快速开始

### 3.1 FP32 baseline（不量化）

```bash
python train_svhn.py --device mps --quant none --w_bits 32 --a_bits 32 --data_dir .
```

### 3.2 Balanced Quantization（权重量化 + 可选激活量化）

例如 W4A4：

```bash
python train_svhn.py --device mps --quant balanced --w_bits 4 --a_bits 4 --data_dir .
```

### 3.3 ViT（8x8 patch）

默认把 32x32 图像切成 `8x8` patch（共 16 个 token）：

```bash
python train_svhn.py --device mps --model vit --quant balanced --w_bits 4 --a_bits 4 --data_dir .
```

在 CIFAR-10 上跑同样的模型/量化（示例）：

```bash
python train_svhn.py --dataset cifar10 --device mps --model vit --quant balanced --w_bits 4 --a_bits 4 --data_dir .
```

关闭 extra：

```bash
python train_svhn.py --device mps --quant balanced --w_bits 4 --a_bits 4 --no_extra --data_dir .
```

---

## 4. 量化实现细节（重点）

代码位置：

- 权重量化（Balanced Quantization）：`quantization/balanced.py`
- rounding/STE/bit-rnn 对齐工具：`quantization/ops.py`
- 量化层封装（Conv/Linear/Activation）：`models/svhn_cnn.py`

### 4.1 Balanced Quantization（论文 Algorithm 1）

对每个需要量化的权重张量 `W`（Conv2d / Linear 的 weight），计算：

1. **Scale（停止梯度）**
   - `scale = max(|W|)`（`--scale_mode maxabs`，论文默认）
   - 或 `scale = mean(|W|) * 2.5`（`--scale_mode meanabs2.5`，参考 bit-rnn 的工程技巧）
   - `scale` 均使用 `detach()`，避免梯度穿过统计量导致不稳定。

2. **Histogram Equalization（Algorithm 2）得到 `W_e ∈ [0,1]`**
   - `W_e = equalize_k(W)`，实现见下节。

3. **round-to-zero 再量化并映射回原值域**
   - 论文定义的 round-to-zero（tie-breaking “half towards zero”）：
     - `rtz(x) = sign(x) * ceil(|x| - 1/2)`
     - 注意它 **不同于** `torch.round`（ties-to-even），因此在 `quantization/ops.py` 里显式实现。
   - Balanced Quantization 的核心映射（`k` bits）：
     - `W_f = rtz(2^k * W_e - 1/2) / (2^k - 1) - 1/2`
     - `W_q = 2 * scale * W_f`（最终 `W_q ∈ [-scale, +scale]`）

4. **STE（Straight-Through Estimator）**
   - 前向：用量化后的 `W_q` 做 Conv/Linear。
   - 反向：对 rounding（`rtz`）使用 STE（梯度直接透传）。
   - 训练中始终保留 FP32 master weights（`nn.Parameter`），只在 forward 使用量化值。

### 4.2 Equalization（论文 Algorithm 2：mean-recursive）

论文 Algorithm 2 用递归 + mask 的方式，把权重集合按阈值 `T = mean(S_W)` 反复二分（用 mean 近似 median 来提速），最终得到均衡后的 `W_e ∈ [0,1]`。

在 `quantization/balanced.py` 中，为了避免递归实现导致的 `O(2^k)` 次 full-tensor pass（k=8 时会很慢），这里实现了等价的 **向量化 O(k) 版本**：

- 每个元素在 k 次 split 中累积一个 `k-bit` 前缀编码（left=0 / right=1），得到 leaf group id；
- 在 leaf 内做 `min/max` 归一化得到 residual；
- 最终：`W_e = (group_id + residual) / 2^k`，并 clamp 到 `[0,1]`；
- 统计量（mean/min/max 等）全部 `detach()`（对齐 TF 里 `stop_gradient` 的思路）。

### 4.3 bit-rnn（TensorFlow）对齐点（`bit-rnn/bit_utils.py`）

`quantization/ops.py` 里提供了对齐实现：

- TF rounding 规则：`round_half_away_from_zero`（`tf.round(0.5)=1`, `tf.round(-0.5)=-1`）
- `round_bit(x, bit) = round(x * (2^bit-1)) / (2^bit-1)`（带 STE 版本）
- `quantize_w_bitutils(x, bit)`：严格翻译 `bit_utils.quantize_w`：
  - `scale = stop_gradient(mean(abs(x))*2.5)`
  - `clip(x/scale, -0.5, 0.5)` 的梯度用直通（STE）
  - 再做 `round_bit`（TF rounding + STE）

当使用 `--quant uniform --scale_mode meanabs2.5` 时，模型会按 bit-rnn 的习惯对权重先做 `tanh` 再 `quantize_w_bitutils(tanh(W))`（见 `models/svhn_cnn.py`）。

---

## 5. 训练脚本与常用参数

入口：`train_svhn.py`

关键参数：

- `--device {auto,mps,cuda,cpu}`：建议 `--device mps`（Apple Silicon）
- `--dataset {svhn,cifar10,cifar100}`：数据集选择（默认 `svhn`）
- `--model {cnn,vit}`：选择模型（默认 `cnn`）
- `--use_extra / --no_extra`：仅对 SVHN 生效，是否把 `extra_32x32.mat` 拼进训练集（默认开启）
- `--quant {none,balanced,uniform}`
- `--w_bits {2,3,4,8,32}`：权重量化 bitwidth（32 表示不量化）
- `--a_bits {2,3,4,8,32}`：激活量化 bitwidth（32 表示不量化）
- `--scale_mode {maxabs,meanabs2.5}`
- `--w_transform {none,tanh}`：仅对 `--quant balanced` 生效，对权重先做变换（例如 `tanh(W)`）再走 BQ
- `--w_bias_mode {none,mean}`：仅对 `--quant balanced` 生效，引入简单的 bias/zero-point（量化 `W-mean(W)` 后再把 mean 加回去）
- `--fp32_first_last`：首层 conv + 末层 fc 保持 FP32（低 bit 可作为稳定性选项）
- `--no_tqdm`：关进度条，方便做 sweep/跑日志

ViT 结构参数（仅 `--model vit` 生效）：

- `--vit_patch`（默认 8）
- `--vit_dim`（默认 192）
- `--vit_depth`（默认 6）
- `--vit_heads`（默认 3）
- `--vit_mlp_ratio`（默认 4.0）
- `--vit_patch_norm`：在 patch embedding 后加 LayerNorm（可作为稳定性/精度小技巧）
- `--vit_pool {cls,mean}`：分类 pooling（默认 `cls`）
- `--vit_drop`（默认 0.0）
- `--vit_attn_drop`（默认 0.0）

macOS 建议：

- `--num_workers 0` 默认就是 0，避免 `spawn` 导致 `.mat` 大数组在 worker 间重复占内存。

优化相关（可选 trick）：

- `--optimizer {sgd,adamw,muon}`：ViT 常用 `adamw`；`muon` 需要注意它只支持 **2D 参数**，脚本会自动拆分：2D 用 Muon，其余参数用 AdamW。
- `--grad_clip`：Transformer 常见的 grad-norm 裁剪
- `--label_smoothing`

---

## 6. 输出文件

`--output_dir`（默认 `checkpoints/`）会写入：

- `best.pt`：按 val acc 最优
- `last.pt`：最后一个 epoch
- `metrics.jsonl`：每个 epoch 的 `train/val` 指标 + `time_sec`（train/val/epoch），以及最终 `test` 行

---

## 7. 评估

```bash
python eval_svhn.py --ckpt checkpoints/best.pt --device mps --data_dir .
```

---

## 8. 单元测试

```bash
pytest -q
```

---

## 9. Sweep（可选）

`sweep_bits.py` 可以跑一组 `w_bits/a_bits` 组合，并把结果汇总到 `sweeps/results.csv`：

```bash
python sweep_bits.py --device mps --quant balanced --w_bits 8 4 2 --a_bits 8 4 2
```

---

## 10. 实验结果

本节把目前跑过的核心实验结果集中整理（先列结果，再做总结），并尽量保证对齐可比。

除特别说明外统一设定：

- 设备：Apple M4 Pro（`--device mps`）
- 量化：`--quant balanced --equalize recursive_mean`
- 训练：`--batch_size 256 --seed 42 --val_split 0.1 --no_tqdm`
- SVHN：默认用 `train+extra`（`--use_extra` 默认开启），增强仅 `RandomCrop(32,padding=4)`；不启用水平翻转
- CIFAR-10/100：通过 `--dataset {cifar10,cifar100}`，增强为 `RandomCrop(32,padding=4)+RandomHorizontalFlip(0.5)`（`--hflip`）

指标含义：

- `Best Val acc`：训练过程中 val acc 的最大值
- `Test acc`：训练结束后加载 `best.pt` 在 test 上评估

### 10.1 ViT（8x8 patch，W2A4，≤10 epoch）结果一览

统一：`--model vit --w_bits 2 --a_bits 4`（除特别说明外使用 `--scale_mode meanabs2.5` 以便跨数据集对齐）

| Dataset | Recipe | patch-norm | Optimizer | lr | Epochs | Best Val acc | Test acc | 输出目录 |
|---|---|---:|---|---:|---:|---:|---:|---|
| SVHN | wd+clip+mean-pool | ✅ | AdamW | 3e-4 | 10 | 0.9823 | 0.9669 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_meanabs2.5_adamw_lr3e-4_wd0.05_clip1_poolmean_pnorm` |
| SVHN | wd+clip+mean-pool | ✅ | Muon（2D）+ AdamW（其余） | 1e-3 | 10 | 0.9843 | 0.9726 | `sweeps/2026-02-22_svhn_vit_bestrecipe_muon_w2a4_meanabs2.5_e10_lr1e-3_wd0.05_clip1_poolmean_pnorm` |
| CIFAR-10 | wd+clip+mean-pool | ✅ | AdamW | 3e-4 | 10 | 0.5810 | 0.5560 | `sweeps/2026-02-22_cifar10_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10` |
| CIFAR-10 | wd+clip+mean-pool | ✅ | AdamW | 1e-3 | 10 | 0.5722 | 0.5664 | `sweeps/2026-02-22_cifar10_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10_lr1e-3` |
| CIFAR-10 | wd+clip+mean-pool | ❌ | AdamW | 3e-4 | 10 | 0.5892 | 0.5715 | `sweeps/2026-02-22_cifar10_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10_nopnorm` |
| CIFAR-10 | wd+clip+mean-pool | ❌ | AdamW | 1e-3 | 10 | 0.5968 | 0.5785 | `sweeps/2026-02-22_cifar10_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10_nopnorm_lr1e-3` |
| CIFAR-10 | wd+clip+mean-pool | ❌ | Muon（2D）+ AdamW（其余） | 1e-3 | 10 | 0.6318 | 0.6042 | `sweeps/2026-02-22_cifar10_vit_bestrecipe_muon_w2a4_meanabs2.5_e10_nopnorm_lr1e-3` |
| CIFAR-100 | wd+clip+mean-pool | ✅ | AdamW | 3e-4 | 10 | 0.2800 | 0.2764 | `sweeps/2026-02-22_cifar100_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10` |
| CIFAR-100 | wd+clip+mean-pool | ✅ | AdamW | 1e-3 | 10 | 0.2874 | 0.2845 | `sweeps/2026-02-22_cifar100_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10_lr1e-3` |
| CIFAR-100 | wd+clip+mean-pool | ❌ | AdamW | 3e-4 | 10 | 0.3076 | 0.2945 | `sweeps/2026-02-22_cifar100_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10_nopnorm` |
| CIFAR-100 | wd+clip+mean-pool | ❌ | AdamW | 1e-3 | 10 | 0.3080 | 0.3018 | `sweeps/2026-02-22_cifar100_vit_bestrecipe_adamw_w2a4_meanabs2.5_e10_nopnorm_lr1e-3` |
| CIFAR-100 | wd+clip+mean-pool | ❌ | Muon（2D）+ AdamW（其余） | 1e-3 | 10 | 0.3252 | 0.3205 | `sweeps/2026-02-22_cifar100_vit_bestrecipe_muon_w2a4_meanabs2.5_e10_nopnorm_lr1e-3` |

备注：

- ViT 输入为 `32x32`，patch=`8`，因此 token 数为 `4x4 + 1(cls) = 17`。
- `Muon` 在本仓库里按 PyTorch 限制做了参数拆分：仅 2D 参数用 `torch.optim.Muon`，其余参数用 AdamW（见 `train_svhn.py`）。
- `wd+clip+mean-pool` 指：`--weight_decay 0.05 --grad_clip 1.0 --vit_pool mean`。
- Muon 对 `lr` 更敏感：同一 recipe 下 `lr=3e-4` 往往明显更差（例如 SVHN `0.9504`：`sweeps/2026-02-22_svhn_vit_bestrecipe_muon_w2a4_meanabs2.5_e10_lr3e-4_wd0.05_clip1_poolmean_pnorm`；CIFAR-10 `0.5059`：`sweeps/2026-02-22_cifar10_vit_bestrecipe_muon_w2a4_meanabs2.5_e10_nopnorm_lr3e-4`；CIFAR-100 `0.2074`：`sweeps/2026-02-22_cifar100_vit_bestrecipe_muon_w2a4_meanabs2.5_e10_nopnorm_lr3e-4`）。
- **ViT 并不“必须”使用 `meanabs2.5`**：在 SVHN 上同一套 best recipe 用论文默认的 `--scale_mode maxabs` 也能正常收敛到 `Test acc=0.9607`（`sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_adamw_lr3e-4_wd0.05_clip1_poolmean_pnorm`），但 `meanabs2.5` 仍有小幅提升到 `0.9669`。

### 10.2 CNN：1 epoch 低 bit 稳定性（SVHN）

以下为 SVHN（`train+extra`）上 1 epoch 的 W/A bitwidth 与耗时（除最后一行外 `--scale_mode maxabs`）：

| 配置 | Train acc | Val acc | Test acc | Train(s) | Val(s) | Epoch(s) | Test(s) | 输出目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W8A8 | 0.9231 | 0.9696 | 0.9469 | 345.3 | 106.0 | 451.2 | 44.8 | `sweeps/2026-02-21_balanced_w8a8_e1` |
| W4A8 | 0.9171 | 0.9686 | 0.9431 | 1121.9 | 84.3 | 1206.2 | 36.4 | `sweeps/2026-02-21_balanced_w4a8_e1` |
| W4A4 | 0.9170 | 0.9670 | 0.9400 | 1185.7 | 83.0 | 1268.7 | 36.3 | `sweeps/2026-02-21_balanced_w4a4_e1` |
| W2A4 | 0.1164 | 0.0831 | 0.0670 | 1201.6 | 13.5 | 1215.1 | 26.7 | `sweeps/2026-02-21_balanced_w2a4_e1` |
| W2A4 (meanabs2.5) | 0.8981 | 0.9669 | 0.9409 | 1073.5 | 73.3 | 1146.8 | 33.4 | `sweeps/2026-02-21_balanced_w2a4_e1_meanabs2.5` |

### 10.3 W2A4 的 scale / transform / bias 对照（CNN，1 epoch）

统一：`--model cnn --w_bits 2 --a_bits 4 --epochs 1`（其余同上）

额外开关（仅对 `--quant balanced` 生效）：

- `tanh(W)+maxabs`：`--scale_mode maxabs --w_transform tanh`
- `maxabs+bias(mean)`：`--scale_mode maxabs --w_bias_mode mean`（先量化 `W-mean(W)`，再把 mean 加回去）

| Dataset | maxabs（Test acc） | tanh(W)+maxabs（Test acc） | maxabs+bias(mean)（Test acc） | meanabs2.5（Test acc） | 输出目录 |
|---|---:|---:|---:|---:|---|
| SVHN | 0.0670 | 0.1959 | 0.0670 | 0.9409 | maxabs=`sweeps/2026-02-21_balanced_w2a4_e1`<br>tanh=`sweeps/2026-02-22_balanced_w2a4_e1_maxabs_tanh`<br>bias=`sweeps/2026-02-22_balanced_w2a4_e1_maxabs_biasmean`<br>meanabs=`sweeps/2026-02-21_balanced_w2a4_e1_meanabs2.5` |
| CIFAR-10 | 0.3862 | 0.0874 | 0.3412 | 0.4839 | maxabs=`sweeps/2026-02-22_cifar10_cnn_w2a4_e1_maxabs`<br>tanh=`sweeps/2026-02-22_cifar10_cnn_w2a4_e1_maxabs_tanh`<br>bias=`sweeps/2026-02-22_cifar10_cnn_w2a4_e1_maxabs_biasmean`<br>meanabs=`sweeps/2026-02-22_cifar10_cnn_w2a4_e1_meanabs2.5` |
| CIFAR-100 | 0.0729 | 0.0947 | 0.0706 | 0.1323 | maxabs=`sweeps/2026-02-22_cifar100_cnn_w2a4_e1_maxabs`<br>tanh=`sweeps/2026-02-22_cifar100_cnn_w2a4_e1_maxabs_tanh`<br>bias=`sweeps/2026-02-22_cifar100_cnn_w2a4_e1_maxabs_biasmean`<br>meanabs=`sweeps/2026-02-22_cifar100_cnn_w2a4_e1_meanabs2.5` |

### 10.4 ViT：Ablation（W2A4 meanabs2.5）

#### 10.4.1 Dropout（SGD，10 epoch）

统一：`--model vit --optimizer sgd --lr 1e-3 --vit_pool cls`，仅比较 `--vit_drop/--vit_attn_drop`

| Dataset | Drop | Best Val acc | Test acc | 输出目录 |
|---|---:|---:|---:|---|
| SVHN | 0.0 | 0.8834 | 0.8324 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_meanabs2.5_lr1e-3_drop0.0` |
| SVHN | 0.1 | 0.8228 | 0.7711 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_meanabs2.5_lr1e-3_drop0.1` |
| CIFAR-10 | 0.0 | 0.4238 | 0.4178 | `sweeps/2026-02-22_cifar10_vit_sgd_drop0_w2a4_meanabs2.5_e10` |
| CIFAR-10 | 0.1 | 0.3562 | 0.3596 | `sweeps/2026-02-22_cifar10_vit_sgd_drop0.1_w2a4_meanabs2.5_e10` |
| CIFAR-100 | 0.0 | 0.1494 | 0.1386 | `sweeps/2026-02-22_cifar100_vit_sgd_drop0_w2a4_meanabs2.5_e10` |
| CIFAR-100 | 0.1 | 0.1116 | 0.1068 | `sweeps/2026-02-22_cifar100_vit_sgd_drop0.1_w2a4_meanabs2.5_e10` |

#### 10.4.2 patch-norm 的方向在 SVHN vs CIFAR 相反（AdamW，5 epoch）

统一：`--model vit --optimizer adamw --lr 3e-4 --epochs 5 --weight_decay 0.05 --grad_clip 1.0 --vit_pool mean`，对比是否开启 `--vit_patch_norm`

| Dataset | 配置 | Best Val acc | Test acc | 输出目录 |
|---|---|---:|---:|---|
| SVHN | full（含 patch-norm） | 0.9740 | 0.9513 | `sweeps/2026-02-21_vit_ablate_full_w2a4_meanabs2.5_adamw_e5` |
| SVHN | no patch-norm | 0.9561 | 0.9313 | `sweeps/2026-02-21_vit_ablate_nopnorm_w2a4_meanabs2.5_adamw_e5` |
| CIFAR-10 | full（含 patch-norm） | 0.4966 | 0.4802 | `sweeps/2026-02-22_cifar10_vit_ablate_full_w2a4_meanabs2.5_adamw_e5` |
| CIFAR-10 | no patch-norm | 0.5082 | 0.4999 | `sweeps/2026-02-22_cifar10_vit_ablate_nopnorm_w2a4_meanabs2.5_adamw_e5` |
| CIFAR-100 | full（含 patch-norm） | 0.1920 | 0.1910 | `sweeps/2026-02-22_cifar100_vit_ablate_full_w2a4_meanabs2.5_adamw_e5` |
| CIFAR-100 | no patch-norm | 0.2168 | 0.2136 | `sweeps/2026-02-22_cifar100_vit_ablate_nopnorm_w2a4_meanabs2.5_adamw_e5` |

#### 10.4.3 Optimizer ablation（patch-norm-only，10 epoch）

为了尽量把变量收敛到 “优化器本身”，这里用最小设定：

- `--vit_pool cls --vit_patch_norm`
- `--weight_decay 0 --grad_clip 0`
- AdamW lr=3e-4；SGD lr=1e-3；Muon lr=1e-3, momentum=0.95

| Dataset | Optimizer | Best Val acc | Test acc | 输出目录 |
|---|---|---:|---:|---|
| SVHN | AdamW | 0.9798 | 0.9629 | `sweeps/2026-02-22_vit_verify_patchnorm_only_adamw_w2a4_meanabs2.5_e10_v2` |
| SVHN | SGD | 0.8998 | 0.8549 | `sweeps/2026-02-22_vit_verify_patchnorm_only_sgd_w2a4_meanabs2.5_e10` |
| SVHN | Muon（2D）+ AdamW（其余） | 0.9820 | 0.9664 | `sweeps/2026-02-22_vit_patchnorm_only_muon_w2a4_meanabs2.5_e10` |
| CIFAR-10 | AdamW | 0.5596 | 0.5428 | `sweeps/2026-02-22_cifar10_vit_patchnorm_only_adamw_w2a4_meanabs2.5_e10` |
| CIFAR-10 | SGD | 0.4150 | 0.4149 | `sweeps/2026-02-22_cifar10_vit_patchnorm_only_sgd_w2a4_meanabs2.5_e10` |
| CIFAR-10 | Muon（2D）+ AdamW（其余） | 0.5982 | 0.5777 | `sweeps/2026-02-22_cifar10_vit_patchnorm_only_muon_w2a4_meanabs2.5_e10` |
| CIFAR-100 | AdamW | 0.2824 | 0.2747 | `sweeps/2026-02-22_cifar100_vit_patchnorm_only_adamw_w2a4_meanabs2.5_e10` |
| CIFAR-100 | SGD | 0.1300 | 0.1252 | `sweeps/2026-02-22_cifar100_vit_patchnorm_only_sgd_w2a4_meanabs2.5_e10` |
| CIFAR-100 | Muon（2D）+ AdamW（其余） | 0.3024 | 0.2993 | `sweeps/2026-02-22_cifar100_vit_patchnorm_only_muon_w2a4_meanabs2.5_e10` |

### 10.5 总结

- **CNN（W2A4）**：`scale_mode=maxabs` 风险很高（SVHN 上会坍塌到接近随机），`meanabs2.5` 更稳且跨数据集更一致；`tanh(W)+maxabs` 与 `maxabs+bias(mean)` 在当前设定下都**不**是可靠替代（SVHN/CIFAR-10 上明显更差）。
- **ViT（W2A4）**：在 SVHN 上 `maxabs` 也能正常训练，但 `meanabs2.5` 仍有小幅优势；CIFAR 上目前统一用 `meanabs2.5` 做对齐对照，尚未系统验证 ViT+`maxabs`。
- ViT 在 ≤10 epoch 内的瓶颈主要不是 “多训一点/加 dropout”，而是优化器与 recipe：AdamW（或 Muon）明显强于 SGD；dropout(0.1) 在 SVHN/CIFAR-10/CIFAR-100 上都显著变差。
- `--vit_patch_norm` 不是通用必开：SVHN 正向，但 CIFAR-10/100 反向（5 epoch 与 10 epoch 均复核）。
- Muon 对超参（尤其 `lr`）更敏感：同 recipe 下需要单独调参；在本轮 `wd+clip+mean-pool` 设定里，Muon 用 `lr=1e-3` 在 SVHN / CIFAR-10 / CIFAR-100 都优于 AdamW，而 `lr=3e-4` 往往明显更差。
- 这些 CIFAR 指标主要用于验证“相对结论”（≤10 epoch + 小 ViT + 低 bit），不代表 SOTA；若追求绝对精度需更久训练与更大模型/更强增强。
