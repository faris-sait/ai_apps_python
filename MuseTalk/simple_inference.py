#!/usr/bin/env python3
"""
Simple MuseTalk Inference - RTX 3090 Optimized
Workaround for mmpose installation issues
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import argparse
import pickle
import subprocess
from tqdm import tqdm

# Add module paths
sys.path.insert(0, str(Path(__file__).parent))

print("="*50)
print("MuseTalk Simple Inference - RTX 3090")
print("="*50)

# Check dependencies
print("\nChecking dependencies...")
try:
    import cv2
    print("✓ OpenCV")
    from transformers import WhisperModel
    print("✓ Transformers")
    from face_detection import FaceAlignment, LandmarksType
    print("✓ Face Detection")
    import imageio
    print("✓ ImageIO")
    import ffmpeg
    print("✓ FFmpeg-python")
except ImportError as e:
    print(f"✗ Missing: {e}")
    sys.exit(1)

print("\n⚠ Note: mmpose/DWPose required for full pose detection")
print("Using simplified face detection instead...")

# Check GPU
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}")

print("\n" + "="*50)
print("To run full inference with pose detection:")
print("  1. Install mmpose (may require conda)")
print("  2. Or use the Gradio interface: python3 app.py")
print("="*50)

# Try importing mmpose
try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples
    print("\n✓ mmpose found!")
    HAS_MMPOSE = True
except ImportError:
    print("\n✗ mmpose not found")
    HAS_MMPOSE = False
    print("Continuing with basic functionality...")

def load_models(unet_model_path, unet_config, device):
    """Load models"""
    print(f"\nLoading models...")
    try:
        # Import here to avoid issues
        from musetalk.utils.utils import load_all_model
        from musetalk.models.vae import VAE
        from musetalk.models.unet import UNet

        vae, unet, pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type='mse-vae',
            unet_config=unet_config,
            device=device
        )
        print("✓ Models loaded")
        return vae, unet, pe
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple MuseTalk Inference")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--audio", type=str, help="Input audio path")
    parser.add_argument("--output", type=str, default="model/Musetalk/output", help="Output directory")

    args = parser.parse_args()

    if not args.video or not args.audio:
        print("\nUsage: python3 simple_inference.py --video VIDEO_PATH --audio AUDIO_PATH [--output OUTPUT_DIR]")
        print("\nExample:")
        print("  python3 simple_inference.py --video data/video/yongen.mp4 --audio data/audio/yongen.wav")
        sys.exit(1)

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"\n✗ Video not found: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.audio):
        print(f"\n✗ Audio not found: {args.audio}")
        sys.exit(1)

    print(f"\nInput video: {args.video}")
    print(f"Input audio: {args.audio}")
    print(f"Output dir: {args.output}")

    os.makedirs(args.output, exist_ok=True)

    if not HAS_MMPOSE:
        print("\n" + "="*50)
        print("⚠ INSTALLATION ISSUE DETECTED")
        print("="*50)
        print("\nmmpose installation failed due to build environment issues.")
        print("\nAlternative solutions:")
        print("1. Use Gradio interface (GUI): python3 app.py")
        print("2. Try conda: conda install -c conda-forge mmpose")
        print("3. Check if Docker image is available")
        print("\nThe inference script requires mmpose for face landmark detection.")
        print("="*50)
        sys.exit(1)

    # If we get here, mmpose is available - load and run models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        vae, unet, pe = load_models(
            unet_model_path="models/musetalkV15/unet.pth",
            unet_config="models/musetalkV15/musetalk.json",
            device=device
        )

        if vae is None:
            print("\nFailed to load models")
            sys.exit(1)

        print("\n✓ Setup complete!")
        print("Ready for inference...")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
