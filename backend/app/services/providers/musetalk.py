"""
MuseTalk Lip Sync Provider

This module provides the MuseTalk lip-sync implementation.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.providers.base import BaseLipSyncProvider

logger = logging.getLogger(__name__)

# MuseTalk directory relative to backend
MUSETALK_DIR = Path(__file__).parent.parent.parent.parent.parent / "MuseTalk"


class MuseTalkProvider(BaseLipSyncProvider):
    """
    MuseTalk lip-sync provider.
    
    Uses the MuseTalk library for high-quality lip-sync generation.
    """
    
    name = "musetalk"
    
    def __init__(self):
        self.musetalk_dir = MUSETALK_DIR
        self.inference_script = self.musetalk_dir / "scripts" / "inference.py"
        self.config_path = self.musetalk_dir / "configs" / "inference" / "test.yaml"
        
    def run(
        self,
        video_path: str,
        audio_path: str,
        output_path: str | None = None,
        options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Run MuseTalk lip-sync generation.
        
        Args:
            video_path: Path to the input video or image
            audio_path: Path to the input audio file
            output_path: Optional output path
            options: Optional dict with keys:
                - version: "v1" or "v15" (default: "v15")
                - use_float16: bool (default: True)
                - fps: int (default: 25)
                - gpu_id: int (default: 0)
                
        Returns:
            dict with success status and output path
        """
        # Validate inputs
        is_valid, error_msg = self.validate_inputs(video_path, audio_path)
        if not is_valid:
            return {
                "success": False,
                "output_path": None,
                "message": error_msg,
                "metadata": None
            }
        
        # Check MuseTalk installation
        if not self.musetalk_dir.exists():
            return {
                "success": False,
                "output_path": None,
                "message": f"MuseTalk not found at {self.musetalk_dir}",
                "metadata": None
            }
        
        # Parse options
        opts = options or {}
        version = opts.get("version", "v15")
        use_float16 = opts.get("use_float16", True)
        fps = opts.get("fps", 25)
        gpu_id = opts.get("gpu_id", 0)
        
        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(video_path).stem
            audio_name = Path(audio_path).stem
            output_dir = Path("/home/vineeth/Documents/projects/avatar_faris/ai_apps_python/MuseTalk_inputs_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{video_name}_{audio_name}_{timestamp}.mp4")
        
        try:
            # Create a temporary config for this inference
            result = self._run_inference(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                version=version,
                use_float16=use_float16,
                fps=fps,
                gpu_id=gpu_id
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"MuseTalk inference failed: {e}")
            return {
                "success": False,
                "output_path": None,
                "message": f"MuseTalk inference failed: {str(e)}",
                "metadata": {"error_type": type(e).__name__}
            }
    
    def _run_inference(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        version: str,
        use_float16: bool,
        fps: int,
        gpu_id: int
    ) -> dict[str, Any]:
        """
        Run the actual MuseTalk inference.
        """
        # Build command
        cmd = [
            sys.executable,
            str(self.inference_script),
            "--inference_config", str(self.config_path),
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--result_dir", str(Path(output_path).parent),
            "--version", version,
            "--fps", str(fps),
            "--gpu_id", str(gpu_id),
        ]
        
        if use_float16:
            cmd.append("--use_float16")
        
        logger.info(f"Running MuseTalk: {' '.join(cmd)}")
        
        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Run subprocess
        result = subprocess.run(
            cmd,
            cwd=str(self.musetalk_dir),
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"MuseTalk stderr: {result.stderr}")
            return {
                "success": False,
                "output_path": None,
                "message": f"MuseTalk failed: {result.stderr[:500]}",
                "metadata": {
                    "return_code": result.returncode,
                    "stderr": result.stderr[:1000]
                }
            }
        
        # Check if output exists
        if Path(output_path).exists():
            return {
                "success": True,
                "output_path": output_path,
                "message": "MuseTalk inference completed successfully",
                "metadata": {
                    "version": version,
                    "fps": fps,
                    "stdout": result.stdout[:500] if result.stdout else None
                }
            }
        else:
            # Try to find the output in the result directory
            result_dir = Path(output_path).parent / version
            output_files = list(result_dir.glob("*.mp4"))
            
            if output_files:
                # Move the first output to the expected path
                actual_output = output_files[0]
                import shutil
                shutil.move(str(actual_output), output_path)
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "message": "MuseTalk inference completed successfully",
                    "metadata": {
                        "version": version,
                        "fps": fps,
                        "original_output": str(actual_output)
                    }
                }
            
            return {
                "success": False,
                "output_path": None,
                "message": "MuseTalk completed but output file not found",
                "metadata": {
                    "expected_path": output_path,
                    "result_dir": str(result_dir)
                }
            }
