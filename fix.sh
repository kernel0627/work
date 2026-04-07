#!/usr/bin/env bash
set -euo pipefail

# 1. 定义环境名称
ENV_NAME="research_env"

# 2. 如果存在旧的 venv 或同名 conda 环境，先清理（可选）
# rm -rf .venv

# 3. 创建 Conda 环境 (Python 3.12 匹配你的镜像)
conda create -y -n $ENV_NAME python=3.12

# 4. 激活环境 (在脚本中使用 source 激活)
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
set -u

# 5. 显式安装适配 CUDA 12.4 的 PyTorch
# 这是最关键的一步，必须指向 pytorch 的官方 whl 源
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 6. 安装 requirements 中的其他依赖
# 建议先安装 torch 再装其他的，防止版本覆盖
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

# 7. 验证 GPU 是否可用
echo "--------------------------------"
echo "验证结果："
python - <<'PY'
import torch
print('Torch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Device:', torch.cuda.get_device_name(0))
    print('CUDA Version from Torch:', torch.version.cuda)
else:
    print('--- 警告：CUDA 依然不可用 ---')
PY