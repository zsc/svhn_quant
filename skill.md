# Lessons Learned（SVHN + 量化 + Apple Silicon MPS）

本文件记录从零搭建到跑完 W/A bitwidth 实验过程中，最关键的工程经验与踩坑点，方便后续复现实验/继续扩展。

## 1) Apple Silicon / MPS 相关

- **设备选择**：优先 `mps`（`torch.backends.mps.is_available()`），并打印 `next(model.parameters()).device` 确认真的在 MPS 上跑。
- **计时要同步**：MPS/CUDA 都是异步执行，做 wall-time 计时前后必须 `synchronize()`，否则会严重低估时间。
  - MPS：`torch.mps.synchronize()`
  - CUDA：`torch.cuda.synchronize()`
- **macOS DataLoader 的 workers**：`num_workers>0` 会走 `spawn`，如果 Dataset 内部持有大 numpy buffer，容易在 worker 里被复制导致内存爆炸/速度变慢。
  - 最简单稳妥：`--num_workers 0`。
  - 如果要加速数据增强，再考虑把增强放到 GPU 或用更轻量的 pipeline。
- **性能对比要重复**：同一台机器上连续跑多个长任务会受温度/后台进程/电源策略影响，单次 1-epoch 时间波动可能很大。做严谨对比建议：
  - 同一配置重复 2~3 次取均值/方差；
  - 或随机顺序跑，降低 “越跑越慢” 的系统性偏差。

## 2) SVHN `.mat` 数据（Format 2）加载

- **形状**：`.mat` 里 `X` 通常是 `(32, 32, 3, N)`，训练需要转为 `(N, 3, 32, 32)`。
- **标签**：SVHN 用 `10` 表示数字 `0`，Dataset 层统一做 `10 -> 0`，保证 label 永远是 `[0..9]`。
- **内存**：把图片缓存在 RAM 里时，建议维持 `uint8` 存储、getitem 再转 `float32/255`，否则 `train+extra` 会非常占内存。
- **用 extra 默认更合理**：`train+extra`（~60 万样本）能显著提高 1 epoch 早期精度；同时 epoch 时间会明显变长，做实验要预估成本。
- **避免复制大数组**：如果需要 train/val 不同 transform，使用共享底层 numpy buffer 的 “view dataset” 方案（同一份 images/labels，不同 transform），避免双倍占内存。

## 3) Balanced Quantization（BQ）实现要点（正确性优先）

- **round-to-zero 不是 `torch.round`**：
  - 论文要求 “round half towards zero”：`sgn(x) * ceil(|x| - 1/2)`；
  - `torch.round` 是 ties-to-even，不满足论文定义。
- **统计量 stop-gradient（detach）很重要**：
  - `scale=max(|W|)` 或 `scale=mean(|W|)*2.5` 等统计量建议全部 `detach()`；
  - equalize 里的阈值（mean）、min/max 等也建议 `detach()`，否则梯度穿过统计量可能导致不稳定/训练发散。
- **STE 的边界**：训练时保存 FP32 master weights，只在 forward 用量化后的 `Wq`；backward 对 rounding 直接透传梯度（STE），保证 `loss.backward()` 可用。

## 4) Algorithm 2（equalize_k）的工程化：不要写指数复杂度

- **直接递归+mask 的字面实现会变成 `O(2^k)` 次 full-tensor pass**，k=8 训练会非常慢甚至不可用。
- 更可用的做法：实现等价的 **向量化 O(k)**：
  - 每次 split 用 `T = mean(S_W)`（论文用 mean 近似 median）；
  - 每个元素累积 `k-bit prefix code` 得到 leaf group id；
  - leaf 内做 min/max 归一化得到 residual；
  - `We = (group_id + residual) / 2^k` 并 clamp 到 `[0,1]`。
- 统计量与边界：
  - empty group / `max==min` 的 leaf 需要定义固定行为（例如 residual=0）；
  - 防 NaN/Inf：分母 clamp、结果 clamp。

## 5) 与 bit-rnn（TensorFlow）对齐时要区分两套 rounding

- BQ（论文）用 **round-to-zero**（half towards zero）。
- bit-rnn 的 `tf.round` 用 **round half away from zero**（`0.5->1, -0.5->-1`），并配合 gradient override（STE）。
- 如果要 “严格对齐 bit_utils.py”，需要：
  - `scale = stop_gradient(mean(abs(x))*2.5)`
  - `clip_by_value` 的梯度走直通（STE）
  - `round_bit(x, bit) = round(x*k)/k` 用 TF rounding 规则 + STE
  - 某些路径会对权重先做 `tanh(W)` 再量化（bit-rnn 常见做法）

## 6) 低 bit（W≤2：W2A2/W1A2 等）稳定性：`scale_mode` + `--fp32_last` 是关键

- **CNN + Balanced Quantization** 下，`--scale_mode maxabs` 在 W≤2 时很容易出现 **`NaN` 崩溃**（SVHN / CIFAR-10 多次复现）；`--scale_mode meanabs2.5` 是最稳妥的起手式。
- 如果想坚持用论文默认的 `maxabs`（例如追求更好的上限/更接近论文定义），建议先加 **`--fp32_last`**（最后一层 head 保持 fp32）再谈别的：
  - 对 W2A2：`maxabs + --fp32_last` 可以把 “NaN 崩溃” 拉回到正常训练；
  - 对 W1A2：即使 `meanabs2.5` 不 NaN，**不加 `--fp32_last` 往往也学不动**；`--fp32_last` 是目前验证过的最小有效改动（跨 SVHN/CIFAR-10/CIFAR-100）。
- `--fp32_first` 单独开通常不够（甚至仍会 NaN/不学习）；要么 `--fp32_last`，要么 `--fp32_first_last`。
- 激活位宽（`a_bits`）在 W=1 的场景里更像次要因素：先把 `scale_mode`/`fp32_last` 稳住，再做 A bit 消融。
- 经验：`meanabs2.5` 更像“稳定性技巧”。一旦用 `--fp32_last` 稳住，`maxabs` 在一些组合上可能不差甚至更好（例如 CIFAR-10 CNN 的 W2A2，5 epochs）。

## 7) ViT：别用 CNN 的默认超参直接比

- 本仓库的 ViT block 是 **pre-norm**（`x + Attn(LN(x))` / `x + MLP(LN(x))`），但训练仍对 optimizer/正则/patch size 很敏感。
- 如果看到 ViT acc 明显落后 CNN，优先怀疑是 **训练 recipe**，不是“ViT 天生更差/只收敛慢”：
  - CIFAR-10 上，默认 SGD（5 epochs）test 约 `~0.43`；
  - 换 AdamW + 合理正则 + `--vit_patch 4` 后，20 epochs test 可到 `~0.74`，100 epochs（fp32）test 可到 `~0.83`。
- CIFAR 系列的推荐起手式（fp32 / 低 bit 都可先用这个 recipe 再做对照）：
  - `--optimizer adamw --lr 5e-4 --weight_decay 0.05 --scheduler cosine`
  - `--label_smoothing 0.1 --grad_clip 1.0`
  - `--vit_patch 4 --vit_patch_norm --vit_pool mean --vit_drop 0.1 --vit_attn_drop 0.1`
- 低比特 ViT：在“调好 recipe”后 gap 可以很小。CIFAR-10 上 100 epochs 的例子里，W2A2 / W1A2(+`--fp32_last`) 的 test 约 `~0.81`，fp32 test 约 `~0.83`（绝对差 `~1.7%`）。
- `scale_mode`：在 ViT 上我们没有复现到 `maxabs` 的 NaN 崩溃（和 CNN 不同），但低 bit 仍建议做一次 sanity check（特别是换 optimizer / patch size / 正则之后）。

## 8) 复现与实验管理

- **对齐设定**：对比不同 W/A bits 时，统一 seed、batch_size、val_split、优化器、scheduler、数据增强开关。
- **记录命令/配置/时间**：把 config + 指标 + time 都落到 `metrics.jsonl`，再生成 report（表格最直观）。
- **不要覆盖/删除输出目录**：每次实验写到独立 `output_dir`，便于回溯与复现（也避免误删）。
- **把结论分层记录**：`notes.md` 作为实验日志（允许“暂定结论”）；`findings.md` 只放跨数据集/多次复核的稳定结论。
- **磁盘空间优先**：大 sweep 时建议默认 `--no_save_last --no_save_optimizer --no_save_best`（只留 `metrics.jsonl/run.log`），避免 checkpoint 爆盘。
- **文档别写“幽灵路径”**：README 里的表格/结论尽量引用真实存在的 `runs/...` 输出目录；如果目录结构调整过（例如历史上叫 `sweeps/...`），要同步更新，否则后续复现会被卡住。
- **并行跑提高吞吐**：GPU util 低时可以并行启动多个训练进程（例如用 `sweep_bits.py --jobs N`），但要接受单个 job 变慢的代价（资源竞争）。
- **下载/解压不要并行**：CIFAR/其他自动下载的数据，建议先单进程下载好再并行跑，避免多进程同时写同一个 tarball。

## 9) 推荐的下一步（如果继续扩展）

- 把 **W≤2** 的稳定性做成 sweep：`scale_mode × fp32_last × optimizer × epochs`（再按数据集复核）。
- 复杂实验一律做 ablation：优先找 “最小改动集合”（例如先测 `--fp32_last` 是否足够，再决定是否加 `--fp32_first` / `--w_transform` / 改优化器等）。
- ViT 先把 fp32 baseline 调到合理水平，再做低 bit 对照（否则很难分清是“量化问题”还是“训练 recipe 问题”）。
- 做多次重复跑（至少 3 次）报告均值/方差，降低系统噪声对结论的影响。

## 10) CIFAR-10 / CIFAR-100 复核（对照 SVHN 的“结论”是否可迁移）

用 `validate_cifar.py` 在 CIFAR-10/100 上按同一套量化/ViT 设定做了对照，结论更接近“哪些经验是稳健的、哪些是数据集依赖的”：

- **稳健结论（更容易迁移）**
  - 对 CNN + Balanced：W≤2 下 `scale_mode=meanabs2.5` 更稳；若用 `maxabs`，优先尝试 `--fp32_last` 来避免崩溃并让模型学得动。
  - 对 CNN + Balanced：W1A2 下 `--fp32_last` 是目前观测到的最小有效改动（跨 SVHN/CIFAR-10/CIFAR-100 均复核过）。
  - 对 ViT：optimizer/recipe 的影响往往大于量化本身；AdamW 通常明显强于 SGD（尤其是短 epoch）。

- **不稳健结论（强数据集/设定依赖）**
  - `--vit_patch_norm` / `--vit_pool` / dropout 等 trick 不是“必开”，它们会和 patch size / optimizer / epoch / 正则产生耦合：某些短跑设定下是负向，但换配方/训更久可能会反转；因此建议每次只改一个变量做 ablation。
  - `meanabs2.5` vs `maxabs` 的“精度上限”也依赖于是否已经稳定住：`meanabs2.5` 往往先救稳定性，但稳定后 `maxabs` 可能追平甚至更好（需要在固定 recipe 下跑到足够 epochs 才能判断）。

- **Muon（torch.optim.Muon）**
  - Muon **只支持 2D 参数**，需要把 1D（bias/norm）和 4D（conv weight）等参数交给另一个优化器（这里用 AdamW）。
  - 在 CIFAR-10/100 的 patch-norm-only 对照里，Muon（2D）+ AdamW（其余）比 AdamW/SGD 更好，值得作为 ViT 的额外候选优化器。

- **工程坑**
  - CIFAR 数据下载不要并行启动多个进程，否则会出现多个进程同时下载同一个 tarball 的风险；建议先单进程下载好，再并行跑实验。

## 11) 速度与耗时：量化 ≠ 加速（本仓库实现是纯 PyTorch）

- **不要默认以为低 bit 会更快**：本仓库的量化在 forward 里会做额外的 equalize/round/clamp 等张量操作；在没有 bitwise 推理内核/融合算子时，训练常见现象是 **低 bit 反而更慢**（尤其是 ViT）。
- **用“前几 epoch × epochs”估时**：先跑 3~5 个 epoch（计时需 `synchronize()`），再乘以总 epoch 数估算总耗时；注意前几轮有 warmup/缓存效应，最好再留 10%~20% buffer。
- **一个具体参照（仅供量级感知，强依赖硬件/实现）**：在本环境（CUDA）里 CIFAR-10 ViT 的 100 epochs，fp32 约 ~21 min，而 W2A2/W1A2（含量化）约 ~31 min（见 `notes.md` 对应记录）。在 Apple MPS 上的绝对时间会不同，但“量化未必更快”的结论通常仍成立。
