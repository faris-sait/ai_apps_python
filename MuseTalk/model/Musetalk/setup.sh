#!/bin/bash

# MuseTalk Setup Script for RTX 3090
# This script downloads model weights and sets up the environment

echo "=========================================="
echo "MuseTalk Setup for RTX 3090 GPU"
echo "=========================================="

# Go to parent directory (MuseTalk root)
cd "$(dirname "$0")/../../" || exit 1

echo "[1] Checking Python installation..."
python3 --version

echo -e "\n[2] Checking CUDA and GPU..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

echo -e "\n[3] Installing required packages..."
pip install -r requirements.txt -q

echo -e "\n[4] Downloading model weights..."
bash download_weights.sh

echo -e "\n[5] Verifying model weights..."
if [ -d "models/musetalk" ] || [ -d "models/musetalkV15" ]; then
    echo "✓ Model weights downloaded successfully!"
else
    echo "✗ Warning: Model weights not found. Please check download_weights.sh output."
fi

echo -e "\n=========================================="
echo "Setup completed!"
echo "=========================================="
echo "Next steps:"
echo "1. Prepare your input video and audio files"
echo "2. Update the config file at model/Musetalk/configs/your_config.yaml"
echo "3. Run: bash model/Musetalk/generate_video.sh"
echo "=========================================="
