#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="research_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs eval_results eval_results/clip_subset eval_results/resnet_subset
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

  echo "===== Eval: clip_linear ====="
  python -m src.eval \
    --data-root ./datasets \
    --sources stylegan biggan ldm_200 dalle \
    --arch clip_linear \
    --checkpoint ./runs/clip_vitl14_progan/ckpts/best.pt \
    --batch-size 64 \
    --num-workers 4 \
    --amp \
    --output-dir ./eval_results/clip_subset

  echo "===== Eval: resnet50 ====="
  python -m src.eval \
    --data-root ./datasets \
    --sources stylegan biggan ldm_200 dalle \
    --arch resnet50 \
    --checkpoint ./runs/resnet50_progan/ckpts/best.pt \
    --batch-size 64 \
    --num-workers 4 \
    --amp \
    --output-dir ./eval_results/resnet_subset
} 2>&1 | tee ./logs/eval_subset.log
