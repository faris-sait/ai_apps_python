#!/usr/bin/env python3
"""
Upload files to Cloudflare R2 bucket under AI_video folder.

Usage:
    python scripts/upload_to_r2.py <file_path1> <file_path2> ...
    
Example:
    python scripts/upload_to_r2.py /path/to/video.mp4 /path/to/audio.wav
"""

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

# Load environment variables from root .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# R2 Configuration
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_ACCESS_KEY_ID = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
CLOUDFLARE_SECRET_ACCESS_KEY = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

# R2 endpoint URL
R2_ENDPOINT_URL = f"https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Folder prefix in R2
R2_FOLDER = "AI_video"


def get_r2_client():
    """Create and return an S3 client configured for Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=CLOUDFLARE_ACCESS_KEY_ID,
        aws_secret_access_key=CLOUDFLARE_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def upload_file(client, file_path: str) -> dict:
    """
    Upload a single file to R2 bucket under AI_video folder.
    
    Args:
        client: boto3 S3 client
        file_path: Local path to the file
        
    Returns:
        dict with upload result
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"success": False, "file": str(file_path), "error": "File not found"}
    
    # Key is the path in R2: AI_video/filename
    key = f"{R2_FOLDER}/{file_path.name}"
    
    try:
        # Upload the file
        client.upload_file(
            str(file_path),
            R2_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": get_content_type(file_path)}
        )
        
        # Generate the public URL (if bucket is public) or the R2 path
        r2_url = f"https://{R2_BUCKET_NAME}.{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com/{key}"
        
        return {
            "success": True,
            "file": str(file_path),
            "key": key,
            "bucket": R2_BUCKET_NAME,
            "url": r2_url
        }
    except Exception as e:
        return {"success": False, "file": str(file_path), "error": str(e)}


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


def list_files_in_folder(client, prefix: str = R2_FOLDER) -> list:
    """List all files in the AI_video folder."""
    try:
        response = client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix)
        files = []
        for obj in response.get("Contents", []):
            files.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat()
            })
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def main():
    # Validate configuration
    if not all([CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_ACCESS_KEY_ID, CLOUDFLARE_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        print("Error: Missing R2 configuration in .env file")
        print("Required: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_ACCESS_KEY_ID, CLOUDFLARE_SECRET_ACCESS_KEY, R2_BUCKET_NAME")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCurrently configured:")
        print(f"  Bucket: {R2_BUCKET_NAME}")
        print(f"  Folder: {R2_FOLDER}/")
        print(f"  Endpoint: {R2_ENDPOINT_URL}")
        
        # List existing files
        print("\n--- Files in AI_video folder ---")
        client = get_r2_client()
        files = list_files_in_folder(client)
        if files:
            for f in files:
                print(f"  {f['key']} ({f['size']} bytes)")
        else:
            print("  (empty)")
        sys.exit(0)
    
    # Get file paths from command line
    file_paths = sys.argv[1:]
    
    print(f"Uploading {len(file_paths)} file(s) to R2 bucket '{R2_BUCKET_NAME}' in folder '{R2_FOLDER}/'")
    print(f"Endpoint: {R2_ENDPOINT_URL}\n")
    
    # Create R2 client
    client = get_r2_client()
    
    # Upload each file
    results = []
    for file_path in file_paths:
        print(f"Uploading: {file_path}...")
        result = upload_file(client, file_path)
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ Uploaded to: {result['key']}")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n--- Summary ---")
    print(f"Uploaded: {successful}/{len(results)} files")
    
    if successful > 0:
        print(f"\nFiles are stored in: {R2_BUCKET_NAME}/{R2_FOLDER}/")


if __name__ == "__main__":
    main()
