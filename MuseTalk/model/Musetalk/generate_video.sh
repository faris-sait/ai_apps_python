#!/bin/bash

# MuseTalk Video Generation Script - RTX 3090 Optimized
# Usage: bash generate_video.sh [version] [mode] [config_file]
# Example: bash generate_video.sh v1.5 normal custom_config.yaml

# Default values
VERSION=${1:-"v1.5"}
MODE=${2:-"normal"}
CONFIG_FILE=${3:-"../../configs/inference/test.yaml"}

# Go to MuseTalk root directory
cd "$(dirname "$0")/../../" || exit 1

OUTPUT_DIR="model/Musetalk/output"

echo "=========================================="
echo "MuseTalk Video Generation - RTX 3090"
echo "=========================================="
echo "Version: $VERSION"
echo "Mode: $MODE"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Validate version
if [ "$VERSION" != "v1.0" ] && [ "$VERSION" != "v1.5" ]; then
    echo "Error: Invalid version. Use 'v1.0' or 'v1.5'"
    exit 1
fi

# Validate mode
if [ "$MODE" != "normal" ] && [ "$MODE" != "realtime" ]; then
    echo "Error: Invalid mode. Use 'normal' or 'realtime'"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Set model paths based on version
if [ "$VERSION" = "v1.0" ]; then
    MODEL_DIR="./models/musetalk"
    UNET_MODEL="$MODEL_DIR/pytorch_model.bin"
    UNET_CONFIG="$MODEL_DIR/musetalk.json"
    VERSION_ARG="v1"
else
    MODEL_DIR="./models/musetalkV15"
    UNET_MODEL="$MODEL_DIR/unet.pth"
    UNET_CONFIG="$MODEL_DIR/musetalk.json"
    VERSION_ARG="v15"
fi

# Check if models exist
if [ ! -f "$UNET_MODEL" ]; then
    echo "Error: Model file not found: $UNET_MODEL"
    echo "Please run 'bash model/Musetalk/setup.sh' first to download weights"
    exit 1
fi

if [ ! -f "$UNET_CONFIG" ]; then
    echo "Error: Model config not found: $UNET_CONFIG"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# RTX 3090 Optimization - Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_HOME="$MODEL_DIR"

echo -e "\n[GPU Info]"
python3 -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print(f'CUDA Capability: {torch.cuda.get_device_capability(0)}')
"

# Run inference based on mode
if [ "$MODE" = "normal" ]; then
    echo -e "\n[Running Inference - Normal Mode]"
    python3 -m scripts.inference \
        --inference_config "$CONFIG_FILE" \
        --result_dir "$OUTPUT_DIR" \
        --unet_model_path "$UNET_MODEL" \
        --unet_config "$UNET_CONFIG" \
        --version "$VERSION_ARG"
else
    echo -e "\n[Running Inference - Realtime Mode]"
    python3 -m scripts.realtime_inference \
        --inference_config "$CONFIG_FILE" \
        --result_dir "$OUTPUT_DIR" \
        --unet_model_path "$UNET_MODEL" \
        --unet_config "$UNET_CONFIG" \
        --version "$VERSION_ARG" \
        --fps 25
fi

echo -e "\n=========================================="
if [ $? -eq 0 ]; then
    echo "✓ Video generation completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
else
    echo "✗ Error during inference. Check logs above."
    exit 1
fi
echo "=========================================="
