# CLIP ViT 1-NN 使用说明

这是新增的一套独立流程，不会影响原来的 `clip_linear` 和 `resnet50`。

## 训练

```bash
bash scripts/train_linear_nn.sh
```

默认会：

- 读取 `./datasets/train/progan`
- 验证 `./datasets/val/progan`
- 输出到 `./runs/clip_vitl14_1nn_progan`

训练后的主要文件：

- `./runs/clip_vitl14_1nn_progan/ckpts/best.pt`
- `./runs/clip_vitl14_1nn_progan/reports/best_metrics.json`
- `./runs/clip_vitl14_1nn_progan/logs/console.log`
- `./runs/clip_vitl14_1nn_progan/tensorboard/`

## 评估

```bash
bash scripts/eval_linear_nn.sh
```

默认会：

- 读取 `./runs/clip_vitl14_1nn_progan/ckpts/best.pt`
- 评估 `stylegan biggan ldm_200 dalle`
- 输出到 `./eval_results/clip_1nn_subset`

评估后的主要文件：

- `./eval_results/clip_1nn_subset/per_source_results.csv`
- `./eval_results/clip_1nn_subset/summary.csv`
- `./eval_results/clip_1nn_subset/summary.json`
- `./eval_results/clip_1nn_subset/tensorboard/`

## 直接运行 Python

训练：

```bash
python -m src.train_clip_1nn \
  --train-root ./datasets/train/progan \
  --val-root ./datasets/val/progan \
  --output-dir ./runs/clip_vitl14_1nn_progan
```

评估：

```bash
python -m src.eval_clip_1nn \
  --data-root ./datasets \
  --sources stylegan biggan ldm_200 dalle \
  --checkpoint ./runs/clip_vitl14_1nn_progan/ckpts/best.pt \
  --output-dir ./eval_results/clip_1nn_subset
```

## 说明

- 这套实现是 `CLIP ViT-L/14 + 1-NN`
- 使用固定阈值 `0.5` 做主评估，同时会额外统计 best-threshold 指标
- 默认会写 TensorBoard 日志；如不需要，可加 `--no-tensorboard`
- 运行前需要准备好 `research_env` 环境和对应数据集
