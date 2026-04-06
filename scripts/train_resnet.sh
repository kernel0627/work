#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs runs
export PYTHONUNBUFFERED=1
python -m src.train \
  --train-root ./datasets/train/progan \
  --val-root ./datasets/val/progan \
  --arch resnet50 \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-4 \
  --num-workers 4 \
  --amp \
  --save-every 1 \
  --patience 5 \
  --output-dir ./runs/resnet50_progan | tee ./logs/train_resnet.log
