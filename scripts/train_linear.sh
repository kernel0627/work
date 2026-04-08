#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="research_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs runs
export PYTHONUNBUFFERED=1

{
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found. Please create ${ENV_NAME} with bash fix.sh first."
    exit 1
  fi

  if ! CONDA_BASE="$(conda info --base 2>/dev/null)"; then
    echo "ERROR: failed to query conda base. Please ensure conda is installed correctly."
    exit 1
  fi

  set +u
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u

  if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "ERROR: conda environment '${ENV_NAME}' does not exist. Run bash fix.sh first."
    exit 1
  fi

  conda activate "$ENV_NAME"

  echo "Repo root: $REPO_ROOT"
  echo "Python: $(command -v python)"
  python -c "import torch; print('CUDA可用性:', torch.cuda.is_available())"

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
    --ema-decay 0.9998
} 2>&1 | tee ./logs/train_linear.log
