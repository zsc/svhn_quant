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

## 10. 实验结果示例

以下是在 Apple M4 Pro (MPS) 上运行 1 epoch 的实验结果（使用 `train + extra` 数据，`--quant balanced --equalize recursive_mean --scale_mode maxabs`）：

| 配置 | Train acc | Val acc | Test acc | Train(s) | Val(s) | Epoch(s) | Test(s) | 输出目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W8A8 | 0.9231 | 0.9696 | 0.9469 | 345.3 | 106.0 | 451.2 | 44.8 | `sweeps/2026-02-21_balanced_w8a8_e1` |
| W4A8 | 0.9171 | 0.9686 | 0.9431 | 1121.9 | 84.3 | 1206.2 | 36.4 | `sweeps/2026-02-21_balanced_w4a8_e1` |
| W4A4 | 0.9170 | 0.9670 | 0.9400 | 1185.7 | 83.0 | 1268.7 | 36.3 | `sweeps/2026-02-21_balanced_w4a4_e1` |
| W2A4 | 0.1164 | 0.0831 | 0.0670 | 1201.6 | 13.5 | 1215.1 | 26.7 | `sweeps/2026-02-21_balanced_w2a4_e1` |
| W2A4 (meanabs2.5) | 0.8981 | 0.9669 | 0.9409 | 1073.5 | 73.3 | 1146.8 | 33.4 | `sweeps/2026-02-21_balanced_w2a4_e1_meanabs2.5` |

**说明**：
- W8A8 / W4A8 / W4A4 在 1 epoch 即可达到 ~0.94+ 的 test acc
- W2A4 在 `--scale_mode maxabs` 下出现失稳（test acc 接近随机），但切换到 `--scale_mode meanabs2.5` 后恢复到 ~0.94

### 10.1 ViT（8x8 patch，≤10 epoch）

ViT 默认把 `32x32` 切成 `8x8` patch（`4x4=16` 个 patch token，再加 `cls` token）。

实测在 SVHN 上，ViT 更吃优化器/训练 recipe。下面这组设置在 ≤10 epoch 内就能把精度拉起来：

```bash
python train_svhn.py --device mps --model vit --quant balanced --w_bits 2 --a_bits 4 \
  --optimizer adamw --lr 0.0003 --weight_decay 0.05 --grad_clip 1.0 \
  --vit_pool mean --vit_patch_norm --epochs 10 --batch_size 256 --data_dir .
```

同样支持 `--optimizer muon`（需要 PyTorch 提供 `torch.optim.Muon`），可用 `--momentum/--weight_decay` 调参。

| 配置 | Epochs | Best Val acc | Test acc | 输出目录 |
|---|---:|---:|---:|---|
| W8A8 | 5 | 0.9772 | 0.9582 | `sweeps/2026-02-21_vit_balanced_w8a8_e5_extra_adamw_lr3e-4_wd0.05_clip1_poolmean_pnorm_v2` |
| W2A4 | 10 | 0.9777 | 0.9607 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_adamw_lr3e-4_wd0.05_clip1_poolmean_pnorm` |
| W2A4 (meanabs2.5) | 10 | 0.9823 | 0.9669 | `sweeps/2026-02-21_vit_balanced_w2a4_e10_extra_meanabs2.5_adamw_lr3e-4_wd0.05_clip1_poolmean_pnorm` |
