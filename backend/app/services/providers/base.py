"""
Base Lip Sync Provider

Abstract base class for all lip-sync providers.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLipSyncProvider(ABC):
    """
    Abstract base class for lip-sync providers.
    
    All providers must implement the `run` method with the following signature:
        run(video_path: str, audio_path: str, output_path: str | None, options: dict | None) -> dict
    """
    
    name: str = "base"
    
    @abstractmethod
    def run(
        self,
        video_path: str,
        audio_path: str,
        output_path: str | None = None,
        options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Run lip-sync generation.
        
        Args:
            video_path: Path to the input video or image file
            audio_path: Path to the input audio file
            output_path: Optional path for the output file
            options: Provider-specific options
            
        Returns:
            dict with keys:
                - success: bool
                - output_path: str (path to generated video)
                - message: str (optional message)
                - metadata: dict (optional additional data)
        """
        pass
    
    def validate_inputs(self, video_path: str, audio_path: str) -> tuple[bool, str]:
        """
        Validate input files exist.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        from pathlib import Path
        
        video = Path(video_path)
        audio = Path(audio_path)
        
        if not video.exists():
            return False, f"Video file not found: {video_path}"
        
        if not audio.exists():
            return False, f"Audio file not found: {audio_path}"
        
        # Check video extensions
        valid_video_ext = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png"}
        if video.suffix.lower() not in valid_video_ext:
            return False, f"Invalid video format: {video.suffix}. Supported: {valid_video_ext}"
        
        # Check audio extensions
        valid_audio_ext = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        if audio.suffix.lower() not in valid_audio_ext:
            return False, f"Invalid audio format: {audio.suffix}. Supported: {valid_audio_ext}"
        
        return True, ""
