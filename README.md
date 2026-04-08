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
