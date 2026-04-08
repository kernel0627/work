#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="research_env"

conda create -y -n "$ENV_NAME" python=3.12

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
    open_clip_torch>=2.24.0 \
    scikit-learn>=1.4 \
    pillow>=10.0 \
    numpy>=1.26 \
    matplotlib>=3.8 \
    pandas>=2.2 \
    tqdm>=4.66 \
    gdown>=5.2.0 \
    pyyaml>=6.0.2

echo "--------------------------------"
echo "Environment Check:"
python - <<'PY'
import torch

print('Torch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Device:', torch.cuda.get_device_name(0))
    print('CUDA Version from Torch:', torch.version.cuda)
PY
