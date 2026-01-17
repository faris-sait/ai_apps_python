"""
Cloudflare R2 Storage Service

Provides utilities for downloading and uploading files to/from R2.
Files are downloaded to temporary directories and cleaned up after processing.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import boto3
from botocore.config import Config

# R2 Configuration from environment
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_ACCESS_KEY_ID = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
CLOUDFLARE_SECRET_ACCESS_KEY = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

# R2 endpoint URL
R2_ENDPOINT_URL = f"https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com" if CLOUDFLARE_ACCOUNT_ID else None


def get_r2_client():
    """Create and return an S3 client configured for Cloudflare R2."""
    if not all([CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_ACCESS_KEY_ID, CLOUDFLARE_SECRET_ACCESS_KEY]):
        raise ValueError("R2 credentials not configured in environment")
    
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=CLOUDFLARE_ACCESS_KEY_ID,
        aws_secret_access_key=CLOUDFLARE_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def download_from_r2(r2_key: str, local_path: str) -> str:
    """
    Download a file from R2 to a local path.
    
    Args:
        r2_key: The key (path) of the file in R2 bucket
        local_path: Local file path to save the file
        
    Returns:
        The local file path
    """
    client = get_r2_client()
    
    # Ensure parent directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Download file
    client.download_file(R2_BUCKET_NAME, r2_key, local_path)
    
    return local_path


def upload_to_r2(local_path: str, r2_key: str) -> str:
    """
    Upload a local file to R2.
    
    Args:
        local_path: Local file path
        r2_key: The key (path) to store the file in R2
        
    Returns:
        The R2 key
    """
    client = get_r2_client()
    
    # Determine content type
    content_type = get_content_type(Path(local_path))
    
    # Upload file
    client.upload_file(
        local_path,
        R2_BUCKET_NAME,
        r2_key,
        ExtraArgs={"ContentType": content_type}
    )
    
    return r2_key


def get_content_type(file_path: Path) -> str:
    """Get content type based on file extension."""
    content_types = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return content_types.get(file_path.suffix.lower(), "application/octet-stream")


def is_r2_path(path: str) -> bool:
    """
    Check if a path is an R2 path (starts with r2:// or AI_video/).
    
    R2 paths can be:
    - r2://bucket/key
    - r2://key (uses default bucket)
    - AI_video/filename (relative R2 key)
    """
    if not path:
        return False
    return path.startswith("r2://") or path.startswith("AI_video/")


def parse_r2_path(path: str) -> str:
    """
    Parse an R2 path and return the key.
    
    Args:
        path: R2 path (r2://key or AI_video/filename)
        
    Returns:
        The R2 key
    """
    if path.startswith("r2://"):
        # Remove r2:// prefix
        return path[5:]
    return path


class TempR2Files:
    """
    Context manager for downloading R2 files to temp directory.
    Automatically cleans up temp files when done.
    
    Usage:
        with TempR2Files() as temp:
            video_path = temp.download("AI_video/video.mp4")
            audio_path = temp.download("AI_video/audio.wav")
            # Process files...
        # Temp files automatically deleted
    """
    
    def __init__(self, prefix: str = "lipsync_"):
        self.prefix = prefix
        self.temp_dir: Optional[str] = None
        self.downloaded_files: list[str] = []
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix=self.prefix)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def download(self, r2_key: str) -> str:
        """
        Download a file from R2 to the temp directory.
        
        Args:
            r2_key: The R2 key (e.g., "AI_video/video.mp4")
            
        Returns:
            Local path to the downloaded file
        """
        if not self.temp_dir:
            raise RuntimeError("TempR2Files not initialized. Use 'with' statement.")
        
        # Parse the key
        key = parse_r2_path(r2_key)
        
        # Get filename from key
        filename = Path(key).name
        local_path = os.path.join(self.temp_dir, filename)
        
        # Download
        download_from_r2(key, local_path)
        self.downloaded_files.append(local_path)
        
        return local_path
    
    def get_local_path(self, path: str) -> str:
        """
        Get local path for a file - downloads from R2 if needed.
        
        If path is an R2 path, downloads it and returns local path.
        If path is already local, returns it unchanged.
        
        Args:
            path: Either an R2 path or local file path
            
        Returns:
            Local file path
        """
        if is_r2_path(path):
            return self.download(path)
        return path
    
    def cleanup(self):
        """Remove all temp files and directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
            self.downloaded_files = []


@contextmanager
def temp_r2_files(prefix: str = "lipsync_"):
    """
    Context manager for working with R2 files locally.
    
    Usage:
        with temp_r2_files() as temp:
            video = temp.get_local_path("AI_video/video.mp4")  # Downloads from R2
            audio = temp.get_local_path("/local/audio.wav")     # Returns as-is
            # Process...
        # Temp files cleaned up automatically
    """
    temp = TempR2Files(prefix=prefix)
    try:
        temp.temp_dir = tempfile.mkdtemp(prefix=prefix)
        yield temp
    finally:
        temp.cleanup()


def list_r2_files(prefix: str = "AI_video/") -> list[dict]:
    """
    List files in R2 bucket with given prefix.
    
    Args:
        prefix: Key prefix to filter by
        
    Returns:
        List of file info dicts
    """
    client = get_r2_client()
    
    response = client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix)
    files = []
    
    for obj in response.get("Contents", []):
        files.append({
            "key": obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"].isoformat(),
            "filename": Path(obj["Key"]).name
        })
    
    return files
