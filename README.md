# work

- 论文：[Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174)
- 官方 Repo：[WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)

这是一个通用伪造图像检测的精简复现版本，当前保留两条训练主线：

- `CLIP ViT-L/14 + linear head`
- `ResNet50 baseline`

训练使用 `ProGAN`，默认评估子集包括 `stylegan`、`biggan`、`ldm_200` 和 `dalle`。

## 概览

- 训练数据：`datasets/train/progan`
- 验证数据：`datasets/val/progan`
- 测试数据：`datasets/test/*`
- diffusion 数据：`datasets/diffusion_datasets/*`

## 环境配置

```bash
bash setup_conda.sh
conda activate research_env
```

## 数据下载

```bash
bash downloaddata.sh --root ./datasets --parts train val test diffusion
```

## 训练

训练 CLIP 线性探针：

```bash
bash scripts/train_linear.sh
```

训练 ResNet50 基线：

```bash
bash scripts/train_resnet.sh
```

训练支持 `EMA`。启用后，训练内验证使用 EMA 权重；checkpoint 里的 `model` 仍然是原始模型权重，EMA 权重单独保存在 `ema` 字段。

## 评估

```bash
bash scripts/eval_subset.sh
```

默认评估 source：

- `stylegan`
- `biggan`
- `ldm_200`
- `dalle`

## Eval 展示（2026-04-08）

展示口径（固定）：

- 日志口径：`logs/eval_subset.log` 最新一次综合评估（2026-04-08 16:07~16:09）
- 模型范围：`clip_linear` 与 `resnet50`
- 数据集范围：`stylegan`、`biggan`、`ldm_200`、`dalle`
- 去重规则：`clip_subset/per_source_results.csv` 中历史重复行不纳入展示，仅保留最新一轮口径

### 平均指标总览（4 个 source 平均）

| Model | Checkpoint | Weights | mean_ap | mean_roc_auc | mean_acc | mean_best_acc | mean_real_acc | mean_fake_acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_linear | `runs/clip_vitl14_progan/ckpts/best.pt` | ema | 0.9826585777146504 | 0.9811578869849208 | 0.9030633345852112 | 0.9336438407611418 | 0.9803729344016024 | 0.82575373476882 |
| resnet50 | `runs/resnet50_progan/ckpts/best.pt` | raw | 0.8493586272719814 | 0.8514771428157693 | 0.6771609914872307 | 0.7857408195626774 | 0.9727496244366549 | 0.3815723585378067 |

### clip_linear 分数据集核心结果

| source | n_samples | ap | roc_auc | acc | best_acc | real_acc | fake_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stylegan | 11982 | 0.9621 | 0.9597 | 0.7800 | 0.8836 | 0.9945 | 0.5655 |
| biggan | 4000 | 0.9932 | 0.9930 | 0.9613 | 0.9630 | 0.9690 | 0.9535 |
| ldm_200 | 2000 | 0.9947 | 0.9941 | 0.9625 | 0.9645 | 0.9790 | 0.9460 |
| dalle | 2000 | 0.9806 | 0.9778 | 0.9085 | 0.9235 | 0.9790 | 0.8380 |

### resnet50 分数据集核心结果

| source | n_samples | ap | roc_auc | acc | best_acc | real_acc | fake_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stylegan | 11982 | 0.9829 | 0.9823 | 0.7626 | 0.9200 | 0.9990 | 0.5263 |
| biggan | 4000 | 0.9139 | 0.9264 | 0.7430 | 0.8505 | 0.9620 | 0.5240 |
| ldm_200 | 2000 | 0.7396 | 0.7556 | 0.5810 | 0.6855 | 0.9650 | 0.1970 |
| dalle | 2000 | 0.7611 | 0.7415 | 0.6220 | 0.6870 | 0.9650 | 0.2790 |

展示结论（简版）：

- `clip_linear` 在 4 个数据集上的整体表现更均衡，平均指标更高。
- `resnet50` 在 `real_acc` 上保持较高，但 `fake_acc` 偏低，跨生成器场景下对伪造样本召回不足。

## 结果

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
