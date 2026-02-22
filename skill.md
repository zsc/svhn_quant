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

## 6) 低 bit（尤其 W2A4）稳定性：scale_mode 很关键

- 实测：`W2A4 + scale_mode=maxabs` 可能出现训练坍塌（loss 巨大、acc 接近随机）。
- 同样配置下改用 `--scale_mode meanabs2.5` 可显著稳定并恢复到正常精度（1 epoch 即可达到较高 test acc）。
- 进一步的稳定性开关（建议优先尝试）：
  - `--fp32_first_last`
  - 更小学习率（例如 `--lr 0.001`）
  - 适当增加 epoch（低 bit 通常需要更多 epoch）

## 7) ViT 的关键训练 recipe（≤10 epoch 也能很高）

- ViT（`--model vit --vit_patch 8`）在 SVHN 上对训练 recipe 非常敏感：
  - `SGD + lr=1e-3` 的 10-epoch test acc 只有 `~0.86`（W8A8）。
  - 以为“不够训/加 dropout”能救：实测 `drop=0.1` 反而让 test acc 掉到 `~0.79`（W8A8），W2A4 也明显更差。
- 影响最大的组合（实测能把 ViT 的 test acc 拉到 `~0.96+`）：
  - `--optimizer adamw --lr 3e-4 --weight_decay 0.05 --grad_clip 1.0`
  - `--vit_pool mean --vit_patch_norm`
- 简单 ablation（W2A4 meanabs2.5，5 epochs）显示重要性排序大致是：
  - `--vit_patch_norm`（最关键，去掉约 `-2.0%` test acc）
  - `--grad_clip 1.0`（有帮助，去掉约 `-0.4%`）
  - `--vit_pool mean`（小幅提升，换成 `cls` 约 `-0.2%`）
  - `--weight_decay 0.05`（小幅提升，设成 0 约 `-0.2%`）
- AdamW vs SGD（同样只开 `--vit_patch_norm`，10 epochs）差距非常大：AdamW test `~0.963`，SGD test `~0.855`。这说明在本设置下 **AdamW 本身也是关键因素**。
- “只开 patch-norm” 是否足够：AdamW + patch-norm-only 的 10-epoch test `~0.963`，距离最佳 recipe（`~0.967`）只差 `~0.4%`，但仍略落后；mean pooling / grad clip / 合适的 weight decay 贡献的是最后这点增益。
- AdamW 的 weight decay 分组也很关键：bias / norm / `pos_embed` / `cls_token` 建议 **不做** weight decay（这里已在 `train_svhn.py` 实现）。

## 8) 复现与实验管理

- **对齐设定**：对比不同 W/A bits 时，统一 seed、batch_size、val_split、优化器、scheduler、数据增强开关。
- **记录命令/配置/时间**：把 config + 指标 + time 都落到 `metrics.jsonl`，再生成 report（表格最直观）。
- **不要覆盖/删除输出目录**：每次实验写到独立 `output_dir`，便于回溯与复现（也避免误删）。

## 9) 推荐的下一步（如果继续扩展）

- 把 `W2A4` 的稳定性做成 sweep：`lr × scale_mode × fp32_first_last × epochs`。
- 做多次重复跑（至少 3 次）报告均值/方差，降低系统噪声对结论的影响。
- 如要更快：考虑把数据增强搬到 GPU、或使用 `num_workers>0` 但配合共享内存/内存映射避免复制（复杂度会上升）。
