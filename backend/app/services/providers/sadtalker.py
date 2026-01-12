"""
SadTalker Lip Sync Provider

This module provides the SadTalker lip-sync implementation.
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

# SadTalker directory relative to backend
SADTALKER_DIR = Path(__file__).parent.parent.parent.parent.parent / "models" / "sadtalker"


class SadTalkerProvider(BaseLipSyncProvider):
    """
    SadTalker lip-sync provider.
    
    Uses the SadTalker library for stylized audio-driven talking face generation.
    Note: SadTalker only supports image input (not video).
    """
    
    name = "sadtalker"
    
    def __init__(self):
        self.sadtalker_dir = SADTALKER_DIR
        self.inference_script = self.sadtalker_dir / "inference.py"
        self.checkpoint_dir = self.sadtalker_dir / "checkpoints"
        
    def run(
        self,
        video_path: str,
        audio_path: str,
        output_path: str | None = None,
        options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Run SadTalker lip-sync generation.
        
        Args:
            video_path: Path to the input image (SadTalker only supports images)
            audio_path: Path to the input audio file
            output_path: Optional output path
            options: Optional dict with keys:
                - pose_style: int (0-45, default: 0)
                - expression_scale: float (default: 1.0)
                - enhancer: str ("gfpgan" or "RestoreFormer", default: None)
                - preprocess: str ("crop", "resize", "full", default: "crop")
                - still: bool (default: False)
                - size: int (256 or 512, default: 256)
                
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
        
        # Check SadTalker installation
        if not self.sadtalker_dir.exists():
            return {
                "success": False,
                "output_path": None,
                "message": f"SadTalker not found at {self.sadtalker_dir}",
                "metadata": None
            }
        
        # Check if input is image (SadTalker only supports images)
        opts = options or {}
        input_type = opts.get("input_type", "image")
        if input_type == "video":
            return {
                "success": False,
                "output_path": None,
                "message": "SadTalker only supports image input. Use mode='image_audio' instead.",
                "metadata": None
            }
        
        # Parse options
        pose_style = opts.get("pose_style", 0)
        expression_scale = opts.get("expression_scale", 1.0)
        enhancer = opts.get("enhancer", None)
        preprocess = opts.get("preprocess", "crop")
        still = opts.get("still", False)
        size = opts.get("size", 256)
        
        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = Path(video_path).stem
            audio_name = Path(audio_path).stem
            output_dir = Path(__file__).parent.parent.parent.parent.parent / "Sadtalker_inputs_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{image_name}_{audio_name}_{timestamp}.mp4")
        
        try:
            result = self._run_inference(
                image_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                pose_style=pose_style,
                expression_scale=expression_scale,
                enhancer=enhancer,
                preprocess=preprocess,
                still=still,
                size=size
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"SadTalker inference failed: {e}")
            return {
                "success": False,
                "output_path": None,
                "message": f"SadTalker inference failed: {str(e)}",
                "metadata": {"error_type": type(e).__name__}
            }
    
    def _run_inference(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        pose_style: int,
        expression_scale: float,
        enhancer: str | None,
        preprocess: str,
        still: bool,
        size: int
    ) -> dict[str, Any]:
        """
        Run the actual SadTalker inference.
        """
        output_dir = Path(output_path).parent
        
        # Build command
        cmd = [
            sys.executable,
            str(self.inference_script),
            "--source_image", image_path,
            "--driven_audio", audio_path,
            "--result_dir", str(output_dir),
            "--checkpoint_dir", str(self.checkpoint_dir),
            "--pose_style", str(pose_style),
            "--expression_scale", str(expression_scale),
            "--preprocess", preprocess,
            "--size", str(size),
        ]
        
        if enhancer:
            cmd.extend(["--enhancer", enhancer])
        
        if still:
            cmd.append("--still")
        
        logger.info(f"Running SadTalker: {' '.join(cmd)}")
        
        # Set environment
        env = os.environ.copy()
        
        # Run subprocess
        result = subprocess.run(
            cmd,
            cwd=str(self.sadtalker_dir),
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"SadTalker stderr: {result.stderr}")
            return {
                "success": False,
                "output_path": None,
                "message": f"SadTalker failed: {result.stderr[:500]}",
                "metadata": {
                    "return_code": result.returncode,
                    "stderr": result.stderr[:1000]
                }
            }
        
        # Find the output file (SadTalker creates timestamped output)
        # Look for the most recent .mp4 file in the output directory
        output_files = sorted(output_dir.glob("*.mp4"), key=os.path.getmtime, reverse=True)
        
        if output_files:
            actual_output = output_files[0]
            
            # If the output path is different, rename it
            if str(actual_output) != output_path:
                import shutil
                shutil.move(str(actual_output), output_path)
            
            return {
                "success": True,
                "output_path": output_path,
                "message": "SadTalker inference completed successfully",
                "metadata": {
                    "pose_style": pose_style,
                    "expression_scale": expression_scale,
                    "preprocess": preprocess,
                    "size": size,
                    "enhancer": enhancer
                }
            }
        
        return {
            "success": False,
            "output_path": None,
            "message": "SadTalker completed but output file not found",
            "metadata": {
                "expected_dir": str(output_dir),
                "stdout": result.stdout[:500] if result.stdout else None
            }
        }
