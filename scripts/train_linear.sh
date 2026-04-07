#!/usr/bin/env bash
set -euo pipefail

# --- 环境准备区 ---
# 强制指定 Conda 环境，防止用到那个破 venv
# 如果你的环境名叫 research_env，按下面写
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research_env

# 检查一下到底有没有 GPU，防止白跑
python -c "import torch; print('CUDA可用性:', torch.cuda.is_available())"

# --- 训练配置区 ---
mkdir -p logs runs
export PYTHONUNBUFFERED=1

# 运行训练
python -m src.train \
  --train-root ./datasets/train/progan \
  --val-root ./datasets/val/progan \
  --arch clip_linear \
  --epochs 10 \
  --batch-size 64 \
  --lr 5e-5 \
  --num-workers 4 \
  --amp \
  --save-every 1 \
  --patience 5 \
  --output-dir ./runs/clip_vitl14_progan 2>&1 | tee ./logs/train_linear.log