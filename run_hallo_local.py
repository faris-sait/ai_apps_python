#!/usr/bin/env python3
"""
Script to run hallo lip sync inference directly on the current machine.
Finds input files and generates video in the halloouput folder.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("/home/vineeth/Documents/projects/avatar_faris/ai_apps_python/halloouput")
CONDA_ENV = "hallo"


def find_input_files(input_path):
    """
    Find source image and audio files in the specified path.
    Returns tuple (image_path, audio_path) or (None, None) if not found.
    """
    input_path = Path(input_path)
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    # Common audio extensions
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac']
    
    image_files = []
    audio_files = []
    
    if input_path.is_file():
        # Single file - determine type
        ext = input_path.suffix.lower()
        if ext in image_extensions:
            image_files.append(input_path)
        elif ext in audio_extensions:
            audio_files.append(input_path)
    else:
        # Directory - search for files
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
        for ext in audio_extensions:
            audio_files.extend(input_path.rglob(f"*{ext}"))
    
    # Select first found files
    source_image = image_files[0] if image_files else None
    driving_audio = audio_files[0] if audio_files else None
    
    return source_image, driving_audio


def run_hallo_inference(source_image, driving_audio, output_path, hallo_dir):
    """
    Run hallo inference with the provided source image and driving audio.
    
    Args:
        source_image: Path to source image
        driving_audio: Path to driving audio file
        output_path: Path to save output video
        hallo_dir: Path to hallo directory
    """
    print(f"\n{'='*60}")
    print("Running Hallo Lip Sync Inference")
    print(f"{'='*60}")
    print(f"Source Image: {source_image}")
    print(f"Driving Audio: {driving_audio}")
    print(f"Output Video: {output_path}")
    print(f"Hallo Directory: {hallo_dir}")
    print(f"{'='*60}\n")
    
    # Change to hallo directory
    original_dir = os.getcwd()
    os.chdir(hallo_dir)
    
    try:
        # Build inference command
        inference_cmd = [
            "conda", "run", "-n", CONDA_ENV, "python", "scripts/inference.py",
            "--source_image", str(source_image),
            "--driving_audio", str(driving_audio),
            "--output", str(output_path)
        ]
        
        print(f"Running command: {' '.join(inference_cmd)}")
        result = subprocess.run(inference_cmd, check=True)
        
        print(f"\n✓ Successfully generated video: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running inference: {e}", file=sys.stderr)
        return False
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run hallo lip sync inference on the current machine"
    )
    
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to input files (directory or file containing image/audio)")
    parser.add_argument("--hallo-dir", type=str, 
                        default="/home/vineeth/Documents/projects/avatar_faris/ai_apps_python/hallo",
                        help="Path to hallo directory (default: ./hallo)")
    
    # Input files (optional if auto-detection works)
    parser.add_argument("--source-image", type=str, default=None,
                        help="Path to source image (if not auto-detected)")
    parser.add_argument("--driving-audio", type=str, default=None,
                        help="Path to driving audio (if not auto-detected)")
    
    # Output settings
    parser.add_argument("--output-name", type=str, default="output.mp4",
                        help="Output video filename (default: output.mp4)")
    
    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Find input files
    print("\n[Step 1/3] Locating input files...")
    source_image, driving_audio = find_input_files(args.input_path)
    
    # Use provided paths if auto-detection failed
    if args.source_image:
        source_image = Path(args.source_image)
    if args.driving_audio:
        driving_audio = Path(args.driving_audio)
    
    if not source_image or not source_image.exists():
        print(f"Error: Source image not found. Searched in: {args.input_path}", file=sys.stderr)
        print("Please provide --source-image argument.")
        sys.exit(1)
    
    if not driving_audio or not driving_audio.exists():
        print(f"Error: Driving audio not found. Searched in: {args.input_path}", file=sys.stderr)
        print("Please provide --driving-audio argument.")
        sys.exit(1)
    
    print(f"Found source image: {source_image}")
    print(f"Found driving audio: {driving_audio}")
    
    # Convert to absolute paths
    source_image = source_image.resolve()
    driving_audio = driving_audio.resolve()
    hallo_dir = Path(args.hallo_dir).resolve()
    
    # Verify hallo directory exists
    if not hallo_dir.exists():
        print(f"Error: Hallo directory not found at {hallo_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Step 2: Setup output path
    print("\n[Step 2/3] Setting up output path...")
    output_path = OUTPUT_DIR / args.output_name
    print(f"Output will be saved to: {output_path}")
    
    # Step 3: Run inference
    print("\n[Step 3/3] Running inference...")
    if run_hallo_inference(str(source_image), str(driving_audio), str(output_path), str(hallo_dir)):
        print(f"\n{'='*60}")
        print("✓ SUCCESS!")
        print(f"{'='*60}")
        print(f"Output video saved to: {output_path}")
        print(f"{'='*60}\n")
    else:
        print("\n✗ Inference failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
