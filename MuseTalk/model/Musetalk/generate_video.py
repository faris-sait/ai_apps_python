#!/usr/bin/env python3

"""
MuseTalk Video Generation Script - RTX 3090 Optimized
This script provides a Python interface for generating videos using MuseTalk
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def setup_gpu():
    """Configure GPU for RTX 3090"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available. Falling back to CPU (will be slow)")
        return False

    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    total_memory = device_props.total_memory / 1e9

    print(f"\n{'='*50}")
    print("GPU Configuration")
    print(f"{'='*50}")
    print(f"GPU Name: {device_name}")
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"CUDA Capability: {device_props.major}.{device_props.minor}")
    print(f"{'='*50}\n")

    return True


def load_config(config_path):
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_model_weights(version, model_dir):
    """Validate that model weights exist"""
    if version == "v1.0":
        required_files = ['pytorch_model.bin', 'musetalk.json']
    else:  # v1.5
        required_files = ['unet.pth', 'musetalk.json']

    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}\n"
                                   f"Please run 'bash model/Musetalk/setup.sh' to download weights")

    return True


def run_inference(args):
    """Run the inference pipeline"""
    # Get MuseTalk root directory
    musetalk_root = Path(__file__).parent.parent.parent
    os.chdir(musetalk_root)

    # Setup GPU
    gpu_available = setup_gpu()

    # Validate model weights
    model_dir = f"./models/musetalk{'V15' if args.version == 'v1.5' else ''}"
    validate_model_weights(args.version, model_dir)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = load_config(args.config)

    # Set model paths
    if args.version == "v1.0":
        unet_model = os.path.join(model_dir, "pytorch_model.bin")
        version_arg = "v1"
    else:
        unet_model = os.path.join(model_dir, "unet.pth")
        version_arg = "v15"

    unet_config = os.path.join(model_dir, "musetalk.json")

    print(f"\n{'='*50}")
    print("Inference Configuration")
    print(f"{'='*50}")
    print(f"Version: {args.version}")
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {unet_model}")
    print(f"{'='*50}\n")

    # Prepare command
    if args.mode == "normal":
        from scripts import inference
        inference.run(
            inference_config=args.config,
            result_dir=args.output_dir,
            unet_model_path=unet_model,
            unet_config=unet_config,
            version=version_arg
        )
    else:
        from scripts import realtime_inference
        realtime_inference.run(
            inference_config=args.config,
            result_dir=args.output_dir,
            unet_model_path=unet_model,
            unet_config=unet_config,
            version=version_arg,
            fps=args.fps
        )

    print(f"\n{'='*50}")
    print("✓ Video generation completed successfully!")
    print(f"Output saved to: {args.output_dir}")

    # List output files
    if os.path.exists(args.output_dir):
        files = os.listdir(args.output_dir)
        for file in files:
            file_path = os.path.join(args.output_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="MuseTalk Video Generation - RTX 3090 Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_video.py --version v1.5 --mode normal
  python3 generate_video.py --version v1.0 --mode realtime --fps 25
  python3 generate_video.py --config custom_config.yaml --output my_output
        """
    )

    parser.add_argument(
        '--version',
        choices=['v1.0', 'v1.5'],
        default='v1.5',
        help='MuseTalk version to use (default: v1.5)'
    )

    parser.add_argument(
        '--mode',
        choices=['normal', 'realtime'],
        default='normal',
        help='Inference mode (default: normal)'
    )

    parser.add_argument(
        '--config',
        default='configs/inference/test.yaml',
        help='Path to inference config file'
    )

    parser.add_argument(
        '--output-dir',
        default='model/Musetalk/output',
        help='Output directory for generated videos'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='FPS for realtime mode (default: 25)'
    )

    args = parser.parse_args()

    try:
        run_inference(args)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
