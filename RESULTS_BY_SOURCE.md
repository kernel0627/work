# Evaluation Results by Source and Model

本文件专门汇总当前仓库里的评测数据，按 `source -> model` 展示全部结果，并单独突出 `resnet50` 的 `best_threshold / best_acc / best_real_acc / best_fake_acc`。

数据口径：
- `clip_linear`：取 `eval_results/clip_subset/per_source_results.csv` 的最新一轮 4 行。
- `clip_1nn`：取 `eval_results/clip_1nn_subset/per_source_results.csv` 的最新一轮 4 行。
- `resnet50`：取 `eval_results/resnet_subset/per_source_results.csv` 的 4 行。
- `acc / real_acc / fake_acc` 为固定阈值口径；`clip_linear` 与 `resnet50` 固定阈值为 `0.5`，`clip_1nn` 的固定阈值也为 `0.5`。
- 所有百分比均已转换为 `%` 并保留 2 位小数；`best_threshold` 保留 3 位小数。

## stylegan

| model | n_samples | ap (%) | roc_auc (%) | acc@0.5 (%) | real_acc@0.5 (%) | fake_acc@0.5 (%) | precision (%) | recall (%) | best_threshold | best_acc (%) | best_real_acc (%) | best_fake_acc (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_linear | 11982 | 96.21 | 95.97 | 78.00 | 99.45 | 56.55 | 99.04 | 56.55 | 0.030 | 88.36 | 87.61 | 89.10 |
| clip_1nn | 11982 | 94.65 | 94.22 | 83.38 | 95.43 | 71.32 | 93.97 | 71.32 | 0.496 | 85.89 | 84.74 | 87.03 |
| resnet50 | 11982 | 98.29 | 98.23 | 76.26 | 99.90 | 52.63 | 99.81 | 52.63 | 0.001 | 92.00 | 97.26 | 86.73 |

## biggan

| model | n_samples | ap (%) | roc_auc (%) | acc@0.5 (%) | real_acc@0.5 (%) | fake_acc@0.5 (%) | precision (%) | recall (%) | best_threshold | best_acc (%) | best_real_acc (%) | best_fake_acc (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_linear | 4000 | 99.32 | 99.30 | 96.13 | 96.90 | 95.35 | 96.85 | 95.35 | 0.434 | 96.30 | 96.30 | 96.30 |
| clip_1nn | 4000 | 94.70 | 93.64 | 85.95 | 86.55 | 85.35 | 86.39 | 85.35 | 0.503 | 86.80 | 94.00 | 79.60 |
| resnet50 | 4000 | 91.39 | 92.64 | 74.30 | 96.20 | 52.40 | 93.24 | 52.40 | 0.002 | 85.05 | 84.90 | 85.20 |

## ldm_200

| model | n_samples | ap (%) | roc_auc (%) | acc@0.5 (%) | real_acc@0.5 (%) | fake_acc@0.5 (%) | precision (%) | recall (%) | best_threshold | best_acc (%) | best_real_acc (%) | best_fake_acc (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_linear | 2000 | 99.47 | 99.41 | 96.25 | 97.90 | 94.60 | 97.83 | 94.60 | 0.446 | 96.45 | 97.50 | 95.40 |
| clip_1nn | 2000 | 98.38 | 98.17 | 93.00 | 90.20 | 95.80 | 90.72 | 95.80 | 0.503 | 94.10 | 96.20 | 92.00 |
| resnet50 | 2000 | 73.96 | 75.56 | 58.10 | 96.50 | 19.70 | 84.91 | 19.70 | 0.001 | 68.55 | 79.00 | 58.10 |

## dalle

| model | n_samples | ap (%) | roc_auc (%) | acc@0.5 (%) | real_acc@0.5 (%) | fake_acc@0.5 (%) | precision (%) | recall (%) | best_threshold | best_acc (%) | best_real_acc (%) | best_fake_acc (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_linear | 2000 | 98.06 | 97.78 | 90.85 | 97.90 | 83.80 | 97.56 | 83.80 | 0.256 | 92.35 | 93.30 | 91.40 |
| clip_1nn | 2000 | 93.12 | 91.88 | 85.00 | 90.20 | 79.80 | 89.06 | 79.80 | 0.500 | 85.00 | 90.20 | 79.80 |
| resnet50 | 2000 | 76.11 | 74.15 | 62.20 | 96.50 | 27.90 | 88.85 | 27.90 | 0.002 | 68.70 | 81.70 | 55.70 |

## Model AVG

| model | mean_ap (%) | mean_roc_auc (%) | mean_acc@0.5 (%) | mean_best_acc (%) | mean_real_acc@0.5 (%) | mean_fake_acc@0.5 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_linear | 98.27 | 98.12 | 90.31 | 93.36 | 98.04 | 82.58 |
| clip_1nn | 95.21 | 94.48 | 86.83 | 87.95 | 90.59 | 83.07 |
| resnet50 | 84.94 | 85.15 | 67.72 | 78.57 | 97.27 | 38.16 |

## ResNet50 Best-Threshold 重点

| source | best_threshold | best_acc (%) | best_real_acc (%) | best_fake_acc (%) |
| --- | ---: | ---: | ---: | ---: |
| stylegan | 0.001 | 92.00 | 97.26 | 86.73 |
| biggan | 0.002 | 85.05 | 84.90 | 85.20 |
| ldm_200 | 0.001 | 68.55 | 79.00 | 58.10 |
| dalle | 0.002 | 68.70 | 81.70 | 55.70 |

## Source Files

- `eval_results/clip_subset/per_source_results.csv`
- `eval_results/resnet_subset/per_source_results.csv`
- `eval_results/clip_1nn_subset/per_source_results.csv`
- `eval_results/clip_subset/summary.csv`
- `eval_results/resnet_subset/summary.csv`
- `eval_results/clip_1nn_subset/summary.csv`
