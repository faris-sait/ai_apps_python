#!/bin/bash
# Simple script to run hallo inference with specific files

# Initialize conda
source /home/vineeth/miniconda3/etc/profile.d/conda.sh

# Navigate to hallo directory
cd /home/vineeth/Documents/projects/avatar_faris/ai_apps_python/hallo

# Activate hallo environment
conda activate hallo

# Create output directory
mkdir -p /home/vineeth/Documents/projects/avatar_faris/ai_apps_python/halloouput

# Run inference
python scripts/inference.py \
    --source_image /home/vineeth/Documents/projects/avatar_faris/ai_apps_python/hallo/examples/reference_images/5.jpg \
    --driving_audio /home/vineeth/Documents/projects/avatar_faris/ai_apps_python/hallo/examples/driving_audios/3.wav \
    --output /home/vineeth/Documents/projects/avatar_faris/ai_apps_python/halloouput/output.mp4
