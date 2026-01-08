#!/bin/bash

# LatentSync Video Generation Script
# This script generates a lip-synced video using LatentSync

# Set up paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LATENTSYNC_DIR="${SCRIPT_DIR}/LatentSync-main"
OUTPUT_DIR="/home/vineeth/ai_apps_python/latentsyncoutput"

# Activate conda environment
source /home/vineeth/miniconda3/etc/profile.d/conda.sh
conda activate latentsync

# Change to LatentSync directory
cd "${LATENTSYNC_DIR}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Starting LatentSync Video Generation"
echo "============================================"
echo "Input Video: assets/demo1_video.mp4"
echo "Input Audio: assets/demo1_audio.wav"
echo "Output: ${OUTPUT_DIR}/output_video.mp4"
echo "============================================"

# Run LatentSync inference
python -m scripts.inference \
    --unet_config_path "configs/unet/stage2_512.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --enable_deepcache \
    --video_path "assets/demo1_video.mp4" \
    --audio_path "assets/demo1_audio.wav" \
    --video_out_path "${OUTPUT_DIR}/output_video.mp4"

echo "============================================"
echo "Video generation complete!"
echo "Output saved to: ${OUTPUT_DIR}/output_video.mp4"
echo "============================================"
