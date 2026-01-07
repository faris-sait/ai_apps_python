"""
SadTalker Test Script
Generates a talking face video from an image and audio file.
"""

import sys
import os
from pathlib import Path

# Add paths
backend_path = Path(__file__).parent / "backend"
sadtalker_path = Path(__file__).parent / "models" / "sadtalker"
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(sadtalker_path))

# Set working directory
os.chdir(str(Path(__file__).parent / "backend"))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Test files
IMAGE_PATH = str(sadtalker_path / "examples" / "source_image" / "full3.png")
AUDIO_PATH = str(sadtalker_path / "examples" / "driven_audio" / "RD_Radio31_000.wav")
OUTPUT_DIR = str(Path(__file__).parent / "output")

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print(f"\n=== SadTalker Test ===")
print(f"Image: {IMAGE_PATH}")
print(f"Audio: {AUDIO_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

# Check if files exist
if not Path(IMAGE_PATH).exists():
    print(f"ERROR: Image file not found: {IMAGE_PATH}")
    sys.exit(1)
if not Path(AUDIO_PATH).exists():
    print(f"ERROR: Audio file not found: {AUDIO_PATH}")
    sys.exit(1)

print("\nLoading SadTalker models...")

from app.services.lipsync import SadTalkerProcessor, LipSyncConfig

# Create processor
processor = SadTalkerProcessor()

# Create config
config = LipSyncConfig(
    model="sadtalker",
    quality="medium",  # Use "high" for 512x512 + GFPGAN enhancement
    fps=25
)

output_path = str(Path(OUTPUT_DIR) / "sadtalker_output.mp4")

print(f"\nGenerating video...")
print(f"Output path: {output_path}")

try:
    success = processor.process(
        image_path=IMAGE_PATH,
        audio_path=AUDIO_PATH,
        output_path=output_path,
        config=config
    )

    if success:
        print(f"\n SUCCESS! Video generated at: {output_path}")
    else:
        print("\n FAILED: Video generation returned False")

except Exception as e:
    print(f"\n ERROR: {e}")
    import traceback
    traceback.print_exc()
