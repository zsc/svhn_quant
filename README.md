# SVHN/CIFAR 量化训练（PyTorch · Balanced Quantization · MPS/CUDA）

本仓库用 PyTorch 训练 SVHN（`.mat`）与 CIFAR-10/100 的 CNN/ViT（支持 `mps/cuda/cpu`），并实现：

- **Balanced Quantization（arXiv:1706.07145）**：对 **权重** 做论文算法（Algorithm 1/2）量化，并用 **STE** 训练。
- 可选 **激活量化**：通过 `--a_bits` 开启（用于 W/A bitwidth 组合实验）。
- 参考对齐：提供与 `bit-rnn/bit_utils.py` 思路一致的 `scale=mean(abs(x))*2.5 + stop_gradient(detach)`、TF rounding 规则等工程实现。
- 复用同一套训练脚本在 **CIFAR-10/100** 上做验证：`--dataset {cifar10,cifar100}`（torchvision 自动下载）。

> 备注：为加速 sweep，`notes.md` / `findings.md` 中的大部分实验记录是在 **CUDA GPU** 环境上完成的；MPS 路径主要做过可跑通/功能验证。

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

- `torch`（支持 `mps` 或 `cuda`；CPU 也能跑但更慢）
- `torchvision`
- `numpy`
- `scipy`（读 `.mat`）
- `tqdm`

可选：

- `pytest`（跑单元测试）

---

## 3. 快速开始

设备建议：默认用 `--device auto`（自动选择 `mps > cuda > cpu`）；也可以显式指定 `--device mps` / `--device cuda`。

### 3.1 FP32 baseline（不量化）

```bash
python train_svhn.py --device auto --quant none --w_bits 32 --a_bits 32 --data_dir .
```

### 3.2 Balanced Quantization（权重量化 + 可选激活量化）

例如 W4A4：

```bash
python train_svhn.py --device auto --quant balanced --w_bits 4 --a_bits 4 --data_dir .
```

### 3.3 ViT（8x8 patch）

默认把 32x32 图像切成 `8x8` patch（共 16 个 token）：

```bash
python train_svhn.py --device auto --model vit --quant balanced --w_bits 4 --a_bits 4 --data_dir .
```

在 CIFAR-10 上跑同样的模型/量化（示例）：

```bash
python train_svhn.py --dataset cifar10 --device auto --model vit --quant balanced --w_bits 4 --a_bits 4 --data_dir .
```

关闭 extra：

```bash
python train_svhn.py --device auto --quant balanced --w_bits 4 --a_bits 4 --no_extra --data_dir .
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

- `--device {auto,mps,cuda,cpu}`：建议 `--device auto`（有 NVIDIA GPU 用 `cuda`；Apple Silicon 用 `mps`）
- `--dataset {svhn,cifar10,cifar100}`：数据集选择（默认 `svhn`）
- `--model {cnn,vit}`：选择模型（默认 `cnn`）
- `--use_extra / --no_extra`：仅对 SVHN 生效，是否把 `extra_32x32.mat` 拼进训练集（默认开启）
- `--quant {none,balanced,uniform}`
- `--w_bits {1,2,3,4,8,32}`：权重量化 bitwidth（32 表示不量化）
- `--a_bits {1,2,3,4,8,32}`：激活量化 bitwidth（32 表示不量化）
- `--scale_mode {maxabs,meanabs2.5}`
- `--w_transform {none,tanh}`：仅对 `--quant balanced` 生效，对权重先做变换（例如 `tanh(W)`）再走 BQ
- `--w_bias_mode {none,mean}`：仅对 `--quant balanced` 生效，引入简单的 bias/zero-point（量化 `W-mean(W)` 后再把 mean 加回去）
- `--fp32_first_last`：首层 + 末层保持 FP32（低 bit 稳定性常用开关）
- `--fp32_first`：仅首层保持 FP32
- `--fp32_last`：仅末层保持 FP32
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

磁盘空间紧张时可用：

- `--no_save_last`：不写 `last.pt`
- `--no_save_optimizer`：checkpoint 不保存 optimizer state（体积显著变小，但 resume 不会恢复 optimizer）
- `--no_save_best`：不写 `best.pt`（只保留 `metrics.jsonl`；适合大规模 sweep）

---

## 7. 评估

```bash
python eval_svhn.py --ckpt checkpoints/best.pt --device auto --data_dir .
```

---

## 8. 单元测试

```bash
pytest -q
```

如果环境里有 pytest 插件冲突（例如 hydra/omegaconf 导致 import error），可用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

---

## 9. Sweep（可选）

`sweep_bits.py` 可以跑一组 `w_bits/a_bits` 组合，并把结果汇总到 `sweeps/results.csv`：

```bash
python sweep_bits.py --device auto --quant balanced --w_bits 8 4 2 --a_bits 8 4 2
```

支持并行与 CIFAR（示例：CIFAR-10 + 4-way 并行 + 少落盘）：

```bash
python sweep_bits.py --dataset cifar10 --device cuda --quant balanced --w_bits 2 1 --a_bits 2 --scale_mode meanabs2.5 --epochs 1 --jobs 4 --hflip --fp32_last --no_save_last --no_save_optimizer --no_save_best
```

---

## 10. 实验结论（findings.md / notes.md）

> 说明：代码支持 `mps/cuda/cpu`。为加速 sweep，`notes.md` / `findings.md` 中多数实验记录是在 CUDA 环境（`torch 2.8.0+cu128`）上完成；绝对耗时与最终精度会因硬件/实现而不同。完整实验日志请看 `notes.md`，稳定结论请看 `findings.md`，工程踩坑点请看 `skill.md`。

### 10.1 CNN：W≤2 的稳定性与最小改动（Balanced Quantization）

核心结论（摘自 `findings.md`，已在 `svhn/cifar10/cifar100` 上做过复核）：

- `--scale_mode maxabs` 在 W≤2 时可能 **NaN 崩溃**；`--scale_mode meanabs2.5` 更稳（尤其是先跑通基线/做大 sweep 时）。
- W1A2 想学得动，`--fp32_last` 是目前验证过的 **最小有效开关**（仅 `--fp32_first` 不够）。
- 如果坚持用论文默认 `maxabs`，建议优先 `+ --fp32_last` 先把训练稳定住，再做别的 ablation。
- 低 bit **不保证更快**：当前实现是纯 PyTorch（没有 fused/bitwise kernel），量化会引入额外张量算子开销。

代表性对照（CNN，5 epochs，seed=42，SGD lr=0.01）：

SVHN（`train+extra`）：

| 配置 | Best Val acc | Test acc | 输出目录（本地 runs/，已 gitignore） |
|---|---:|---:|---|
| W2A2 + `meanabs2.5` | 0.9812 | 0.9667 | `runs/lowbits_2026-02-22/svhn_e5_compare/w2a2_meanabs2.5_e5_s42` |
| W2A2 + `maxabs` + `--fp32_last` | 0.9818 | 0.9671 | `runs/lowbits_2026-02-22/svhn_e5_compare/w2a2_maxabs_fp32last_e5_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.9793 | 0.9596 | `runs/lowbits_2026-02-22/svhn_e5_compare/w1a2_meanabs2.5_fp32last_e5_s42` |
| W1A2 + `maxabs` + `--fp32_last` | 0.9781 | 0.9587 | `runs/lowbits_2026-02-22/svhn_e5_compare/w1a2_maxabs_fp32last_e5_s42` |

CIFAR-10（含 `--hflip`）：

| 配置 | Best Val acc | Test acc | 输出目录（本地 runs/，已 gitignore） |
|---|---:|---:|---|
| W2A2 + `meanabs2.5` | 0.7172 | 0.7122 | `runs/lowbits_2026-02-22/cifar10_e5_compare/w2a2_meanabs2.5_e5_s42` |
| W2A2 + `maxabs` + `--fp32_last` | 0.7576 | 0.7447 | `runs/lowbits_2026-02-22/cifar10_e5_compare/w2a2_maxabs_fp32last_e5_s42` |
| W1A2 + `meanabs2.5` + `--fp32_last` | 0.7060 | 0.6929 | `runs/lowbits_2026-02-22/cifar10_e5_compare/w1a2_meanabs2.5_fp32last_e5_s42` |
| W1A2 + `maxabs` + `--fp32_last` | 0.6856 | 0.6845 | `runs/lowbits_2026-02-22/cifar10_e5_compare/w1a2_maxabs_fp32last_e5_s42` |

补充解释（和消融策略有关）：`meanabs2.5` 更像“先救稳定性”的默认选项；一旦用 `--fp32_last` 稳住，`maxabs` 在部分组合上可能追平甚至更好（例如上面的 CIFAR-10 W2A2）。

### 10.2 ViT：acc 差距往往是 recipe，不是“收敛慢/结构问题”

- 本仓库的 ViT block 是 **pre-norm**（`x + Attn(LN(x))` / `x + MLP(LN(x))`，见 `models/svhn_vit.py`）。
- 如果 ViT acc 明显落后 CNN，优先怀疑 **优化器/正则/patch size** 等 recipe：默认 SGD + 短训会非常吃亏。
- 在 CIFAR-10 上，先把 fp32 的 ViT 训练 recipe 调到合理水平后，再做 W2A2/W1A2 对照，低比特 gap 可以很小。

推荐起手式（CIFAR 系列，先跑 fp32 拉起基线，再做量化对照）：

```bash
python train_svhn.py --dataset cifar10 --model vit --optimizer adamw --lr 5e-4 --weight_decay 0.05 --scheduler cosine --label_smoothing 0.1 --grad_clip 1.0 --vit_patch 4 --vit_patch_norm --vit_pool mean --vit_drop 0.1 --vit_attn_drop 0.1 --epochs 100 --quant none --w_bits 32 --a_bits 32 --device auto
```

代表性结果（CIFAR-10 ViT，100 epochs，seed=42）：

| 配置 | Best Val acc | Test acc | 训练耗时（sum epoch time） | 输出目录（本地 runs/，已 gitignore） |
|---|---:|---:|---:|---|
| fp32（`--quant none`） | 0.8364 | 0.8278 | ~20.9 min | `runs/lowbits_2026-02-22/vit_tune_cifar10_e100/adamw_lr5e-4_wd5e-2_drop0.1_p4_mean_e100_s42` |
| W2A2（`maxabs`） | 0.8206 | 0.8107 | ~31.6 min | `runs/lowbits_2026-02-22/vit_tuned_lowbit_cifar10_e100/w2a2_maxabs_e100_s42` |
| W1A2（`meanabs2.5` + `--fp32_last`） | 0.8152 | 0.8104 | ~31.3 min | `runs/lowbits_2026-02-22/vit_tuned_lowbit_cifar10_e100/w1a2_meanabs2.5_fp32last_e100_s42` |

注意：本仓库的量化实现是纯 PyTorch（没有 bitwise/fused kernel），因此低 bit 训练/推理 **不一定更快，甚至可能更慢**（上表就是一个例子）。如果目标是速度，需要另外做算子融合/专用 kernel（不在本项目范围）。
