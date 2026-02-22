# CIFAR-10 / CIFAR-100 量化复核报告（PyTorch + MPS）

日期：2026-02-22

目的：把 SVHN 上的关键结论（`meanabs2.5`、dropout、ViT 训练 recipe、optimizer ablation、Muon）在 CIFAR-10 / CIFAR-100 上按对齐设定复核一遍。

详细结果分别见：

- `report_cifar_cifar10.md`
- `report_cifar_cifar100.md`

## 统一设定（两套 CIFAR 都一致）

- 入口脚本：`train_svhn.py`（通过 `--dataset {cifar10,cifar100}` 复用训练/量化代码）
- 量化：`--quant balanced --equalize recursive_mean`
- `--device mps --batch_size 256 --seed 42 --val_split 0.1 --no_tqdm`
- 增强：`RandomCrop(32,padding=4)` + `RandomHorizontalFlip(p=0.5)`（通过 `--hflip` 开启）
- 复现实验：`python validate_cifar.py --dataset cifar10` / `python validate_cifar.py --dataset cifar100`

## 关键观察（对比 SVHN）

- `scale_mode=meanabs2.5` 在低 bit（W2A4）下依然更稳/更好，但在 CIFAR 上不一定表现为 “maxabs 直接坍塌”，更多是精度差距。
- Dropout(0.1) 在 CIFAR-10/100 这套设定下同样明显变差（与 SVHN 一致）。
- ViT 上 optimizer 依然是主导因素：AdamW 明显强于 SGD（与 SVHN 一致）。
- `--vit_patch_norm` 在 SVHN 上是强正向 trick，但在本轮 CIFAR-10/100 的 ablation 里 **去掉 patch-norm 反而更好**，提示该 trick 存在明显的任务/设定耦合，不应作为“必开”选项。
- Muon（`torch.optim.Muon`）由于只支持 2D 参数，需要拆分参数组；在 CIFAR-10/100 的 patch-norm-only 对照里，Muon（2D）+ AdamW（其余）优于 AdamW/SGD，值得加入候选 optimizer 列表。

