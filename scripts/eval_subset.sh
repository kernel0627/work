#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs eval_results
export PYTHONUNBUFFERED=1
python -m src.eval \
  --data-root ./datasets \
  --sources stylegan biggan ldm_200 dalle \
  --arch clip_linear \
  --checkpoint ./runs/clip_vitl14_progan/ckpts/best.pt \
  --batch-size 64 \
  --num-workers 4 \
  --amp \
  --output-dir ./eval_results/clip_subset | tee ./logs/eval_clip_subset.log
