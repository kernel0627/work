#!/usr/bin/env bash
set -euo pipefail
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
PY
