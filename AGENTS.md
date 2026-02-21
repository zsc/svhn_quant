# SPEC — PyTorch (MPS) 在 Apple M4 Pro 上训练 SVHN CNN，并实现 Balanced Quantization（arXiv:1706.07145）

> 数据已下载到**当前目录**：`train_32x32.mat`, `test_32x32.mat`, `extra_32x32.mat`（SVHN Format 2: Cropped Digits，**非商业用途**）。

本 SPEC 用于驱动 `gemini-cli` / `codex` 生成可运行代码：在 Apple Silicon（M4 Pro）上用 PyTorch 训练一个 SVHN CNN，并在训练中对权重（可选激活）应用论文 **Balanced Quantization** 的量化方法；实现细节对齐论文算法 + 参考 `bit_utils.py` 的若干工程技巧（尤其是 `scale` 的估计与 `stop_gradient`/`detach` 思路）。:contentReference[oaicite:0]{index=0}

---

## 1. 目标与范围

### 1.1 目标（必须实现）
1. **可复现训练脚本**：在 macOS + Apple M4 Pro 上使用 PyTorch **MPS** 后端训练 CNN 做 SVHN（10 类数字）分类。
2. **Balanced Quantization（BQ）权重量化**：实现并在训练中使用 arXiv:1706.07145 的 Balanced Quantization（核心：**直方图均衡 + 再量化**）。
3. **可配置**：支持命令行切换
   - 权重量化 bitwidth：`--w_bits {2,3,4,8,32}`（32 表示不量化）
   - 是否使用 `extra_32x32.mat` 参与训练：`--use_extra`
   - BQ 的 equalization 方法（至少实现论文 Algorithm 2 的 mean/recursive 版本）
   - `scale` 估计策略：论文默认 `max(|W|)`，另提供可选 `reduce_mean(abs(W))*2.5`（来自参考实现思想）:contentReference[oaicite:1]{index=1}
4. **训练输出**：打印并保存训练/验证/测试指标（loss、acc），保存 best checkpoint（按 val acc）。

### 1.2 非目标（不要求）
- 不要求实现 bitwise 推理加速内核（论文里提到的按位运算加速不在本任务范围）。
- 不要求达到论文在 ImageNet 的指标；这里只做 SVHN 训练可跑通 + BQ 正确实现与可控实验。

---

## 2. 运行环境与依赖

### 2.1 平台
- macOS（Apple Silicon），重点是 **Apple M4 Pro**。
- PyTorch 需支持 `mps`。

### 2.2 Python 依赖
- `torch`, `torchvision`
- `numpy`
- `scipy`（用于读取 `.mat`）
- `tqdm`
- （可选）`tensorboard`

> torchvision 的 SVHN loader 同样依赖 `scipy` 来读 `.mat`。:contentReference[oaicite:2]{index=2}

---

## 3. 数据集（SVHN Format 2: Cropped Digits）

### 3.1 文件位置与格式
当前目录存在：
- `train_32x32.mat`
- `test_32x32.mat`
- `extra_32x32.mat`

`.mat` 文件关键字段：
- `X`：形状通常为 `(32, 32, 3, N)`（H, W, C, N）
- `y`：形状通常为 `(N, 1)`:contentReference[oaicite:3]{index=3}

### 3.2 标签规则
SVHN 原始标签用 `10` 表示数字 `0`。训练时需将 `10 -> 0`（保证类别是 `[0..9]`）。:contentReference[oaicite:4]{index=4}

### 3.3 数据划分建议
- 训练集：
  - 默认仅用 `train_32x32.mat`
  - `--use_extra` 时将 `extra_32x32.mat` 拼接进训练集（常见做法）
- 验证集：从训练集中按 `--val_split`（默认 0.05 或 0.1）划分
- 测试集：`test_32x32.mat`

### 3.4 预处理与增强（可配置）
- `X` 从 `(H,W,C,N)` 转为 PyTorch `float32` 张量 `(N,C,H,W)`。
- 像素缩放到 `[0,1]`。
- Normalize：
  - 默认：`mean=0.5, std=0.5`（每通道）
  - 可选：计算训练集统计（但会慢）
- 增强（训练集可选）：
  - `RandomCrop(32, padding=4)`
  - `RandomHorizontalFlip(p=0.5)`（SVHN 水平翻转可能改变数字语义，默认**关闭**）
  - 轻微 `ColorJitter`（默认关闭）

---

## 4. 模型：SVHN CNN（建议结构）

> 重点不在模型创新，而是量化算法与可训练性。模型应足够稳定（BatchNorm、合理 dropout）。

### 4.1 建议网络（baseline）
- 输入：`(3,32,32)`
- Block1: `Conv(3→64,3x3)` + `BN` + `ReLU` + `Conv(64→64,3x3)` + `BN` + `ReLU` + `MaxPool(2)`
- Block2: `Conv(64→128,3x3)` + `BN` + `ReLU` + `Conv(128→128,3x3)` + `BN` + `ReLU` + `MaxPool(2)`
- Block3: `Conv(128→256,3x3)` + `BN` + `ReLU` + `Conv(256→256,3x3)` + `BN` + `ReLU` + `MaxPool(2)`
- Head: `Flatten` + `Linear(256*4*4→512)` + `ReLU` + `Dropout(0.5)` + `Linear(512→10)`

### 4.2 量化插入点（必须可配置）
- 默认：量化所有 `Conv2d` 与 `Linear` 的 **weights**（bias 保持 fp32）。
- 可选策略（推荐提供开关）：
  - `--fp32_first_last`：第一层 conv 与最后一层 fc 保持 fp32（QNN 常见工程策略：首层/输入可更高 bitwidth）。:contentReference[oaicite:5]{index=5}

---

## 5. Balanced Quantization（核心实现要求）

### 5.1 论文关键点（必须对齐）
Balanced Quantization 的核心流程（Algorithm 1）：
1. `scale ← max(|W|)`（默认）:contentReference[oaicite:6]{index=6}  
2. `We ← equalize_k(W)`：直方图均衡（输出在 `[0,1]`）:contentReference[oaicite:7]{index=7}  
3. 将 `We` 量化为离散值并恢复到原值域：`Wq ← 2*scale*Wf`（`Wf ∈ [-1/2, 1/2]`）:contentReference[oaicite:8]{index=8}  
论文给出高效 equalization 的递归实现（Algorithm 2）：用 **均值 mean 近似 median**，递归二分集合以避免每步排序/分位数。:contentReference[oaicite:9]{index=9}

### 5.2 round-to-zero（必须实现）
实现论文定义的 round-to-zero（“round half towards zero”）：
- 规则：正数的 `.5` 向下（朝 0），负数的 `.5` 向上（朝 0）
- 公式见论文 Notations 部分。:contentReference[oaicite:10]{index=10}

**PyTorch 参考实现（语义要求）**：
- 不能直接用 `torch.round`（其 tie-breaking 规则可能不同）
- 需显式实现 tie-to-zero 行为（建议使用 `sign` + `ceil(abs(x)-0.5)`）

### 5.3 Qk（必须实现，用于对照/单元测试）
实现论文的 k-bit uniform quantizer `Qk`，输入必须在 `[0,1]`：
- `Qk(W) = round-to-zero((2^k - 1) * W) / (2^k - 1)`  
输出取值集合：`{0, 1/(2^k-1), ..., 1}`。:contentReference[oaicite:11]{index=11}

> 注意：Balanced Quantization 的“equalize 后再量化”的那一步与 `Qk` 的映射略有差异，论文明确指出“mapping … is different from Qk”。:contentReference[oaicite:12]{index=12}

### 5.4 equalize_k(W)：直方图均衡（必须实现 Algorithm 2 / mean-recursive）
实现论文 Algorithm 2（递归分割 + mask）版本，要求：
- 输入：`W`（任意 shape tensor），初始 mask `M = ones_like(W)`，`level = k`（要得到 `2^k` 个均衡 bins）:contentReference[oaicite:13]{index=13}
- 每层递归：
  - `SW = { w | M>0 }`
  - 若 `level == 0`：将 `SW` affine 映射到 `[0,1]`，并乘 `M`
  - 否则：
    - 阈值 `T = mean(SW) = sum(W*M)/sum(M)`（这里的 mean 是关键点，论文强调用 mean 近似 median 提速）:contentReference[oaicite:14]{index=14}
    - `Ml = M * (W < T)`，`Mg = M * (W >= T)`
    - 递归：`Wl = HE(W, Ml, level-1)`, `Wg = HE(W, Mg, level-1)`
    - 合并：`We = 0.5*Wl + (0.5*Wg + 0.5) * Mg`（与论文一致）:contentReference[oaicite:15]{index=15}

**工程要求（重要）**：
- **禁止 Python for-loop 遍历元素**（必须向量化），否则训练会极慢。
- 需要处理数值/边界：
  - `sum(M)==0`：返回全 0
  - `max(SW)==min(SW)`：返回全 0（或全 0.5，但需固定策略并写清）
- `T`、`min`、`max` 等统计量建议 `detach()`（stop gradient），避免不稳定梯度（对齐 TF 里 stop_gradient 的实践）。:contentReference[oaicite:16]{index=16}

### 5.5 BalancedQuantize(W, k)：量化主函数（必须实现）
提供函数/模块：`balanced_quantize_weight(W, k, scale_mode, equalize_mode)`：

**默认（论文）**：
- `scale = max(abs(W))`（detach）
- `We = equalize_k(W)`（输出 `[0,1]`）
- 执行论文 Algorithm 1 的 rounding + restore-range 步骤，得到 `Wq`（输出在 `[-scale, +scale]`）:contentReference[oaicite:17]{index=17}

**可选 scale_mode（来自参考工程实现思想）**：
- `scale = mean(abs(W)) * 2.5`（detach），并在必要时对 `W/scale` 做裁剪（clip）以抑制 outlier。该 scale 估计方式来自参考实现（TensorFlow）中的 `reduce_mean(abs(x))*2.5`。:contentReference[oaicite:18]{index=18}

> 备注：即使采用论文的 `scale=max(|W|)`，也允许额外提供 `--scale_mode meanabs2.5` 用于实验与稳定性对照。

### 5.6 STE（Straight-Through Estimator）梯度策略（必须实现）
- 训练时保留 fp32 master weights（`nn.Parameter`）。
- 前向：用 `Wq`（量化权重）参与 `Conv/Linear` 计算。
- 反向：对 quantize 算子采用 STE（最少要求：`grad_input = grad_output`）。
- 推荐实现方式：
  - `torch.autograd.Function` 自定义 `forward` 返回量化值，`backward` 直接透传梯度。
- 需要写单元测试验证 autograd 可用：`loss.backward()` 不报错，梯度非 None。

---

## 6. 代码结构（建议）

```

.
├── train_svhn.py
├── eval_svhn.py                  # 可选：单独评估脚本
├── datasets/
│   └── svhn_mat.py               # 读取 train/test/extra_32x32.mat
├── models/
│   └── svhn_cnn.py               # CNN + 量化层包装
├── quantization/
│   ├── ops.py                    # round_to_zero, Qk, STE helper
│   └── balanced.py               # equalize_k (Algorithm2), balanced_quantize_weight
├── utils/
│   ├── meter.py
│   ├── seed.py
│   └── checkpoint.py
└── README.md

```

---

## 7. 训练流程与默认超参

### 7.1 Device 选择
- 优先：`mps`
- fallback：`cuda`（如存在）
- fallback：`cpu`

### 7.2 优化器与调度
默认建议：
- Optimizer：`SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)`
- Scheduler：`CosineAnnealingLR` 或 `StepLR`
- Epoch：`50~100`
- Batch size：`128`（M4 Pro 可按显存调整）

### 7.3 Logging / 保存
- 每 epoch 输出：train loss/acc、val loss/acc、lr
- 保存：
  - `checkpoints/best.pt`（按 val acc）
  - `checkpoints/last.pt`
  - `metrics.jsonl` 或 `metrics.csv`

---

## 8. CLI（必须支持的参数）

`python train_svhn.py [args...]`

必备参数（至少这些）：
- `--data_dir`（默认 `.`）
- `--use_extra`（bool）
- `--val_split`（float，默认 0.1）
- `--epochs`
- `--batch_size`
- `--lr`
- `--seed`
- `--device {auto,mps,cuda,cpu}`

量化相关：
- `--quant {none,balanced,uniform}`  
  - `balanced`：Balanced Quantization（本任务重点）
  - `uniform`：按论文 Definition 1 的 uniform quant 或参考 bit_utils 的 clip+scale 版本（二选一或都做）
- `--w_bits {2,3,4,8,32}`
- `--a_bits {2,3,4,8,32}`（可选：做激活量化）
- `--equalize {recursive_mean}`（必须实现 recursive_mean；可选增加 exact_quantile）
- `--scale_mode {maxabs,meanabs2.5}`
- `--fp32_first_last`（bool）

---

## 9. 单元测试与验收标准（必须提供）

### 9.1 单元测试（pytest 或脚本自测均可）
1. **SVHN mat loader**
   - 能读取 `train_32x32.mat`
   - 输出样本 shape 为 `(3,32,32)`，label ∈ `[0..9]`
   - 检查 `10 -> 0` 已正确映射:contentReference[oaicite:19]{index=19}

2. **round_to_zero**
   - `round_to_zero(1.5)==1`
   - `round_to_zero(-1.5)==-1`
   - 与论文定义一致:contentReference[oaicite:20]{index=20}

3. **equalize_k 输出范围**
   - `We.min() >= 0` 且 `We.max() <= 1`
   - level=k 时能运行且无 NaN/Inf:contentReference[oaicite:21]{index=21}

4. **balanced_quantize_weight 输出范围与离散性**
   - `Wq` 全部在 `[-scale, +scale]` 内（允许极小浮点误差）
   - `unique(Wq)` 数量 ≤ `2^k`（或 ≤ `2^k + 少量误差`，但需解释原因）
   - backward 可用（STE）不报错

### 9.2 集成验收（必须可跑通）
- 命令：
  - `python train_svhn.py --device mps --epochs 1 --quant balanced --w_bits 4 --data_dir .`
- 结果：
  - 能在 MPS 上完成 1 epoch
  - 打印 train/val 指标
  - 保存 checkpoint 到 `checkpoints/`

---

## 10. 关键实现注意事项（必须在代码注释中解释）

1. **mean 代替 median/percentile 的原因**：论文说明均值可近似 median 并显著提速，Algorithm 2 用 `mean(SW)` 作为阈值。:contentReference[oaicite:22]{index=22}
2. **scale 的 detach/stop-gradient**：
   - 论文流程有 `scale ← max(|W|)`（作为范围匹配），建议 detach。
   - 参考工程实现里 `scale = reduce_mean(abs(x))*2.5` 并 stop_gradient（detach），避免梯度穿过统计量导致不稳。:contentReference[oaicite:23]{index=23}
3. **SVHN 标签 10->0**：必须在 dataset 层统一处理。:contentReference[oaicite:24]{index=24}

---

## 11. 交付清单（最终产物）

- 可运行训练脚本：`train_svhn.py`
- 数据集读取模块：`datasets/svhn_mat.py`
- Balanced Quantization 实现：`quantization/balanced.py` + `quantization/ops.py`
- CNN 模型：`models/svhn_cnn.py`（支持量化层包装）
- README：说明如何安装依赖、如何在 MPS 上运行、如何切换量化模式/bitwidth
- （可选）`eval_svhn.py`、tensorboard 支持、导出 ONNX（非必须）

---

## 12. 参考（实现必须对齐的资料）
- Balanced Quantization paper（Algorithm 1/2、round-to-zero、Qk 定义）。:contentReference[oaicite:25]{index=25}
- 参考工程实现：`qinyao-he/bit-rnn` 的 `bit_utils.py`（重点：`scale = reduce_mean(abs(x))*2.5` + stop_gradient 思路）。:contentReference[oaicite:26]{index=26}
- SVHN `.mat` 数据形状与 label=10 表示 0（torchvision 文档/源码）。:contentReference[oaicite:27]{index=27}

