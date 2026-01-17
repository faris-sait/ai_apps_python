#!/usr/bin/env python3
"""
MuseTalk Inference Wrapper for API
Direct inference without subprocess complications
"""
import sys
import os
from pathlib import Path

# Add MuseTalk to path
MUSETALK_DIR = Path(__file__).parent
sys.path.insert(0, str(MUSETALK_DIR))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="v15")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--job_id", default=None, help="Job ID for output filename")
    
    args = parser.parse_args()
    
    print(f"MuseTalk API Wrapper")
    print(f"Video: {args.video}")
    print(f"Audio: {args.audio}")
    print(f"Output: {args.output}")
    print(f"Version: {args.version}")
    print(f"Job ID: {args.job_id}")
    
    # Import and run inference
    try:
        # Create args object for inference with all required attributes
        class InferenceArgs:
            def __init__(self):
                self.video_path = args.video
                self.audio_path = args.audio
                self.result_dir = args.output
                self.version = args.version
                self.fps = args.fps
                self.use_float16 = True
                self.use_saved_coord = False
                self.saved_coord = False
                self.bbox_shift = 5
                self.gpu_id = 0
                
                # Model paths
                self.vae_type = "sd-vae"
                self.unet_config = "./models/musetalkV15/musetalk.json" if args.version == "v15" else "./models/musetalk/config.json"
                self.unet_model_path = "./models/musetalkV15/unet.pth" if args.version == "v15" else "./models/musetalk/pytorch_model.bin"
                self.whisper_dir = "./models/whisper"
                # Use an existing config file (test.yaml exists in repo)
                self.inference_config = "configs/inference/test.yaml"
                
                # Other params
                
                # Preflight: verify critical model files exist
                sd_vae_path = Path("models/sd-vae/diffusion_pytorch_model.safetensors")
                if not sd_vae_path.exists():
                    print(f"⚠ Warning: expected safetensors not found at {sd_vae_path}. Found files: {list(Path('models/sd-vae').iterdir())}")
                    print("If you have .bin model files, MuseTalk may still work but you may need to convert or provide safetensors.")
                self.ffmpeg_path = "./ffmpeg-4.4-amd64-static/"
                self.extra_margin = 10
                self.audio_padding_length_left = 2
                self.audio_padding_length_right = 2
                self.batch_size = 8
                # Use job_id for output filename if provided
                self.output_vid_name = f"{args.job_id}.mp4" if args.job_id else None
                self.parsing_mode = "jaw"
                self.left_cheek_width = 90
                self.right_cheek_width = 90
        
        inference_args = InferenceArgs()
        
        # Import inference function
        import sys
        import yaml
        import tempfile
        sys.path.insert(0, str(MUSETALK_DIR / "scripts"))
        from inference import main as inference_main
        
        # Create a temporary config file with our single task
        temp_config = {
            "task_api": {
                "video_path": args.video,
                "audio_path": args.audio
            }
        }
        
        # Write temp config file
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(temp_config, temp_config_file)
        temp_config_file.close()
        
        # Update inference args to use our temp config
        inference_args.inference_config = temp_config_file.name
        
        try:
            inference_main(inference_args)
            print("✓ Inference completed")
            
            # Construct expected output path
            if args.job_id:
                expected_output = Path(args.output) / args.version / f"{args.job_id}.mp4"
            else:
                video_name = Path(args.video).stem
                audio_name = Path(args.audio).stem
                output_basename = f"{video_name}_{audio_name}"
                expected_output = Path(args.output) / args.version / f"{output_basename}.mp4"
            print(f"Expected output: {expected_output}")
        finally:
            # Clean up temp config
            try:
                os.unlink(temp_config_file.name)
            except:
                pass
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
