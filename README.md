# SVHN 量化训练（PyTorch + Apple Silicon MPS）

本仓库在 **Apple Silicon（MPS）** 上用 PyTorch 训练一个 SVHN CNN（10 类数字分类），并支持：

- **Balanced Quantization（arXiv:1706.07145）**：对 **权重** 做论文算法（Algorithm 1/2）量化，并用 **STE** 训练。
- 可选 **激活量化**：通过 `--a_bits` 开启（用于 W/A bitwidth 组合实验）。
- 参考对齐：提供与 `bit-rnn/bit_utils.py` 思路一致的 `scale=mean(abs(x))*2.5 + stop_gradient(detach)`、TF rounding 规则等工程实现。

---

## 1. 数据集（SVHN Format 2 `.mat`）

把以下文件放在 `--data_dir`（默认当前目录 `.`）：

- `train_32x32.mat`
- `extra_32x32.mat`
- `test_32x32.mat`

注意：SVHN 的标签里用 **10 表示数字 0**，loader 已在 `datasets/svhn_mat.py` 里统一做了 `10 -> 0` 映射，保证 label 范围为 `[0..9]`。

默认行为：训练集会 **自动使用 `train + extra`**。如需关闭 extra，用 `--no_extra`。

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
- `--use_extra / --no_extra`：是否把 `extra_32x32.mat` 拼进训练集（默认开启）
- `--quant {none,balanced,uniform}`
- `--w_bits {2,3,4,8,32}`：权重量化 bitwidth（32 表示不量化）
- `--a_bits {2,3,4,8,32}`：激活量化 bitwidth（32 表示不量化）
- `--scale_mode {maxabs,meanabs2.5}`
- `--fp32_first_last`：首层 conv + 末层 fc 保持 FP32（低 bit 可作为稳定性选项）
- `--no_tqdm`：关进度条，方便做 sweep/跑日志

macOS 建议：

- `--num_workers 0` 默认就是 0，避免 `spawn` 导致 `.mat` 大数组在 worker 间重复占内存。

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
