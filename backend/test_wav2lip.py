"""
Quick test script for Wav2Lip processor.

Usage:
    python test_wav2lip.py <video_path> <audio_path>

Example:
    python test_wav2lip.py face.mp4 audio.wav
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  PyTorch: FAILED - {e}")
        return False

    try:
        import cv2
        print(f"  OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"  OpenCV: FAILED - {e}")
        return False

    try:
        import librosa
        print(f"  Librosa: {librosa.__version__}")
    except ImportError as e:
        print(f"  Librosa: FAILED - {e}")
        return False

    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
    except ImportError as e:
        print(f"  NumPy: FAILED - {e}")
        return False

    print("All imports successful!")
    return True


def test_model_loading():
    """Test that the Wav2Lip model can be loaded"""
    print("\nTesting model loading...")

    from app.services.lipsync import Wav2LipProcessor, WAV2LIP_DIR

    checkpoint_path = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        return False

    print(f"  Checkpoint found: {checkpoint_path}")

    processor = Wav2LipProcessor()
    print(f"  Device: {processor.device}")

    try:
        model = processor._load_wav2lip_model(str(checkpoint_path))
        print("  Model loaded successfully!")
        return True
    except Exception as e:
        print(f"  Model loading failed: {e}")
        return False


def test_full_pipeline(video_path: str, audio_path: str):
    """Test the full lip sync pipeline"""
    print(f"\nTesting full pipeline...")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")

    from app.services.lipsync import Wav2LipProcessor, LipSyncConfig

    output_path = "test_output.mp4"

    processor = Wav2LipProcessor()
    config = LipSyncConfig(
        model="wav2lip",
        quality="medium",
        resize_factor=2  # Reduce resolution for faster testing
    )

    try:
        success = processor.process(video_path, audio_path, output_path, config)
        if success:
            print(f"  Success! Output saved to: {output_path}")
            return True
        else:
            print("  Processing returned False")
            return False
    except Exception as e:
        print(f"  Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not test_imports():
        sys.exit(1)

    if not test_model_loading():
        sys.exit(1)

    if len(sys.argv) >= 3:
        video_path = sys.argv[1]
        audio_path = sys.argv[2]

        if not Path(video_path).exists():
            print(f"Video file not found: {video_path}")
            sys.exit(1)
        if not Path(audio_path).exists():
            print(f"Audio file not found: {audio_path}")
            sys.exit(1)

        test_full_pipeline(video_path, audio_path)
    else:
        print("\nTo test the full pipeline, run:")
        print("  python test_wav2lip.py <video_path> <audio_path>")
        print("\nExample:")
        print("  python test_wav2lip.py face.mp4 audio.wav")
