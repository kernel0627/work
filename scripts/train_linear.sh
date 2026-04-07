#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research_env

python -c "import torch; print('CUDA可用性:', torch.cuda.is_available())"

mkdir -p logs runs
export PYTHONUNBUFFERED=1

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
  --output-dir ./runs/clip_vitl14_progan \
  --ema \
  --ema-decay 0.9998 \
  2>&1 | tee ./logs/train_linear.log