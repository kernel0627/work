# work

- 论文：[Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174)
- 官方 Repo：[WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)

这是一个通用伪造图像检测的精简复现版本，当前保留两条训练主线：

- `CLIP ViT-L/14 + linear head`
- `ResNet50 baseline`

训练使用 `ProGAN`，默认评估子集包括 `stylegan`、`biggan`、`ldm_200` 和 `dalle`。

README 当前分为两部分：

- 上半段：安装、训练、评估命令
- 下半段：当前最佳结果和展示

## 使用说明

### 数据与目录概览

- 训练数据：`datasets/train/progan`
- 验证数据：`datasets/val/progan`
- 测试数据：`datasets/test/*`
- diffusion 数据：`datasets/diffusion_datasets/*`

### 环境配置

```bash
bash setup_conda.sh
conda activate research_env
```

### 数据下载

```bash
bash downloaddata.sh --root ./datasets --parts train val test diffusion
```

### 训练

训练 CLIP 线性探针：

```bash
bash scripts/train_linear.sh
```

训练 ResNet50 基线：

```bash
bash scripts/train_resnet.sh
```

训练支持 `EMA`。启用后，训练内验证使用 EMA 权重；checkpoint 里的 `model` 仍然是原始模型权重，EMA 权重单独保存在 `ema` 字段。

训练输出目录：

- `runs/clip_vitl14_progan/`
- `runs/resnet50_progan/`

主要文件：

- `logs/console.log`
- `logs/train_history.csv`
- `ckpts/last.pt`
- `ckpts/best.pt`
- `ckpts/epoch_*.pt`
- `plots/train_loss.png`
- `plots/val_metrics.png`
- `reports/args.json`
- `reports/env.json`
- `reports/meta.json`
- `reports/best_metrics.json`
- `reports/final_report.json`
- `tensorboard/`

### 评估

```bash
bash scripts/eval_subset.sh
```

默认评估 source：

- `stylegan`
- `biggan`
- `ldm_200`
- `dalle`

评估输出目录：

- `eval_results/clip_subset/`
- `eval_results/resnet_subset/`

主要文件：

- `per_source_results.csv`
- `summary.csv`
- `summary.json`
- `plots/ap_by_source.png`
- `plots/acc_by_source.png`
- `plots/best_acc_by_source.png`
- `plots/pr_<source>.png`
- `plots/roc_<source>.png`
- `tensorboard/`

TensorBoard 启动命令：

```bash
tensorboard --logdir runs --port 6006 --bind_all
```

## 当前最佳结果与展示

### Eval 展示（截至 2026-04-10）

展示口径（按模型引用最新可用评估）：

- 日志口径：
  - `clip_linear`、`resnet50`：`logs/eval_subset.log` 最新一次综合评估（2026-04-08 16:07~16:09）
  - `clip_1nn`：`logs/eval_linear_nn.log` 最新一次评估（2026-04-10 17:10:47~17:13:38）
- 模型范围：`clip_linear`、`clip_1nn` 与 `resnet50`
- 数据集范围：`stylegan`、`biggan`、`ldm_200`、`dalle`
- 去重规则：
  - `eval_results/clip_subset/per_source_results.csv` 中历史重复行不纳入展示，仅保留最新一轮口径
  - `eval_results/clip_1nn_subset/per_source_results.csv` 中历史重复行不纳入展示，仅保留最后 4 行作为最新一轮口径
  - `eval_results/clip_1nn_subset/summary.csv` 仅保留最后 1 行作为 `AVG` 展示口径

### 结果总表（按 model 与 source 展开）

下表按 `model -> source` 展示，`AVG` 表示 `stylegan`、`biggan`、`ldm_200`、`dalle` 4 个 source 的平均值。

| model | source | ap (%) | acc (%) | real_acc (%) | fake_acc (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| clip_linear | stylegan | 96.21 | 78.00 | 99.45 | 56.55 |
| clip_linear | biggan | 99.32 | 96.13 | 96.90 | 95.35 |
| clip_linear | ldm_200 | 99.47 | 96.25 | 97.90 | 94.60 |
| clip_linear | dalle | 98.06 | 90.85 | 97.90 | 83.80 |
| clip_linear | AVG | 98.27 | 90.31 | 98.04 | 82.58 |
| clip_1nn | stylegan | 94.65 | 83.38 | 95.43 | 71.32 |
| clip_1nn | biggan | 94.70 | 85.95 | 86.55 | 85.35 |
| clip_1nn | ldm_200 | 98.38 | 93.00 | 90.20 | 95.80 |
| clip_1nn | dalle | 93.12 | 85.00 | 90.20 | 79.80 |
| clip_1nn | AVG | 95.21 | 86.83 | 90.59 | 83.07 |
| resnet50 | stylegan | 98.29 | 76.26 | 99.90 | 52.63 |
| resnet50 | biggan | 91.39 | 74.30 | 96.20 | 52.40 |
| resnet50 | ldm_200 | 73.96 | 58.10 | 96.50 | 19.70 |
| resnet50 | dalle | 76.11 | 62.20 | 96.50 | 27.90 |
| resnet50 | AVG | 84.94 | 67.72 | 97.27 | 38.16 |

展示结论（简版）：

- `clip_linear` 仍然是 4 个 source 上整体最强的平均表现。
- `clip_1nn` 明显强于 `resnet50`，且在 `stylegan` 上的 `acc` 与 `fake_acc` 优于 `clip_linear`。
- `clip_1nn` 的平均 `fake_acc` 略高于 `clip_linear`，但 `real_acc` 更低，所以整体 `AVG acc` 仍落后于 `clip_linear`。
