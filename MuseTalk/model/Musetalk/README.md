# MuseTalk Video Generation Scripts - RTX 3090 Optimized

This folder contains scripts and tools for generating lip-synced videos using MuseTalk with RTX 3090 GPU optimization.

## Folder Structure

```
model/Musetalk/
├── setup.sh                  # Setup script to download models and dependencies
├── generate_video.sh         # Bash script to generate videos
├── generate_video.py         # Python script for video generation
├── README.md                 # This file
└── output/                   # Directory where generated videos are saved
```

## Quick Start

### 1. Initial Setup

Run the setup script to download model weights and install dependencies:

```bash
bash model/Musetalk/setup.sh
```

This script will:
- Check Python and CUDA installation
- Install required packages
- Download MuseTalk model weights (v1.0 and v1.5)
- Verify GPU availability

### 2. Generate Videos

#### Using Bash Script (Recommended)

```bash
# Using v1.5 model (default)
bash model/Musetalk/generate_video.sh

# Specify version and mode
bash model/Musetalk/generate_video.sh v1.5 normal

# Use realtime mode
bash model/Musetalk/generate_video.sh v1.5 realtime

# Use custom config file
bash model/Musetalk/generate_video.sh v1.5 normal custom_config.yaml
```

#### Using Python Script

```bash
# Basic usage (uses defaults)
python3 model/Musetalk/generate_video.py

# With custom options
python3 model/Musetalk/generate_video.py --version v1.5 --mode normal --fps 25

# Custom config and output
python3 model/Musetalk/generate_video.py \
    --version v1.5 \
    --mode normal \
    --config configs/inference/test.yaml \
    --output-dir model/Musetalk/output
```

## Configuration

### Video Configuration Files

Edit the YAML config file to specify input video and audio:

**Default config**: `configs/inference/test.yaml`

```yaml
task_0:
  video_path: "data/video/your_video.mp4"
  audio_path: "data/audio/your_audio.wav"
  bbox_shift: 0  # Adjust face detection box (optional)
```

**Example with multiple tasks**:

```yaml
task_0:
  video_path: "data/video/video1.mp4"
  audio_path: "data/audio/audio1.wav"
  bbox_shift: 0

task_1:
  video_path: "data/video/video1.mp4"
  audio_path: "data/audio/audio2.wav"
  bbox_shift: -5  # Shift face region up
```

### bbox_shift Parameter

- **Positive values**: Shift face region down
- **Negative values**: Shift face region up
- **0**: No shift (default)

This can significantly affect the quality of results.

## GPU Optimization for RTX 3090

The scripts are optimized for RTX 3090 with the following settings:

```bash
# Set in scripts automatically
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**RTX 3090 Specs**:
- VRAM: 24 GB
- CUDA Cores: 10,496
- Memory Bandwidth: 936 GB/s
- Supports mixed precision training/inference

## Inference Modes

### Normal Mode
- Slower but higher quality
- Processes all frames carefully
- Best for final video production

### Realtime Mode
- Faster processing
- Real-time capable (30+ fps on V100, faster on 3090)
- Suitable for live applications
- Optimized for speed

## Supported Models

### v1.0
- Original MuseTalk model
- Good balance of quality and speed
- Model size: ~500 MB

### v1.5
- **Recommended** - Enhanced version
- Better lip-sync accuracy
- Improved visual quality
- Perceptual loss + GAN loss + Sync loss
- Model size: ~1 GB

## Output

Generated videos are saved to `model/Musetalk/output/`

Each run creates:
- Lip-synced video files (`.mp4`)
- Intermediate results (if enabled in config)

## Troubleshooting

### CUDA Out of Memory

If you get CUDA OOM errors:

```bash
# Use memory optimization (already set in scripts)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Or try clearing cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

### Model Weights Not Found

```bash
# Re-run setup
bash model/Musetalk/setup.sh
```

### GPU Not Detected

```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check installed GPUs
nvidia-smi
```

### Video/Audio Format Issues

Supported formats:
- **Video**: MP4, AVI, MOV, FLV
- **Audio**: WAV, MP3, AAC

Convert if needed:
```bash
# Convert video
ffmpeg -i input.mp4 -c:v libx264 -preset fast output.mp4

# Convert audio
ffmpeg -i input.mp3 output.wav
```

## Performance Benchmarks

### RTX 3090 Expected Performance

| Mode | Version | Resolution | FPS | VRAM Used |
|------|---------|-----------|-----|-----------|
| Normal | v1.5 | 256x256 | 2-5 | ~18 GB |
| Realtime | v1.5 | 256x256 | 25+ | ~12 GB |
| Normal | v1.0 | 256x256 | 3-8 | ~16 GB |

## Advanced Usage

### Custom Python Integration

```python
import sys
from pathlib import Path

# Add MuseTalk to path
musetalk_root = Path(__file__).parent / "../.."
sys.path.insert(0, str(musetalk_root))

# Import and use
from scripts.inference import main as inference_main

inference_main(
    inference_config="configs/inference/test.yaml",
    result_dir="model/Musetalk/output",
    unet_model_path="models/musetalkV15/unet.pth",
    unet_config="models/musetalkV15/musetalk.json",
    version="v15"
)
```

## Requirements

- Python 3.8+
- CUDA 11.8+
- cuDNN 8.x
- PyTorch 2.0+
- FFmpeg 4.2+

See `requirements.txt` in the MuseTalk root directory.

## Links & Resources

- **GitHub**: https://github.com/TMElyralab/MuseTalk
- **HuggingFace**: https://huggingface.co/TMElyralab/MuseTalk
- **Paper**: https://arxiv.org/abs/2410.10122
- **Demo**: https://huggingface.co/spaces/TMElyralab/MuseTalk

## License

This project uses MuseTalk which is licensed under the MIT License. See LICENSE file in the root directory.

## Support

For issues with:
- MuseTalk core: See official GitHub repo
- These scripts: Check the troubleshooting section above
- GPU/CUDA: Check NVIDIA documentation and driver versions

---

Generated for RTX 3090 GPU optimization.
