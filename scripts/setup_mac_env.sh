#!/usr/bin/env bash
set -euo pipefail

echo "============================================="
echo " Setting up Local AI Environment on macOS    "
echo "============================================="

# 1. Check for Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "⚠️ Warning: Non-Apple Silicon architecture detected ($ARCH). Metal optimizations require arm64."
fi

# 2. Check for Homebrew
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 3. Install core CLI dependencies
echo "Installing core tools..."
brew install cmake git python@3.11 ffmpeg

# 4. Create virtual environment
echo "Setting up Python virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# 5. Install PyTorch with Metal (MPS) support, MLX, and transformers
echo "Installing Python packages with Apple Silicon acceleration..."
pip install --upgrade pip
pip install \
    mlx \
    mlx-lm \
    torch \
    torchvision \
    transformers \
    accelerate \
    peft \
    datasets \
    scipy \
    pillow

echo "✅ Environment setup complete! Activate with: source .venv/bin/activate"
