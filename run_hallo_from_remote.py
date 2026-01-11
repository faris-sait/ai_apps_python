#!/usr/bin/env python3
"""
Script to SSH into a remote server and run hallo lip sync inference directly on the server.
The generated video will be copied back to the local halloouput folder.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("/home/vineeth/Documents/projects/avatar_faris/ai_apps_python/halloouput")
CONDA_ENV = "hallo"


def run_command(cmd, check=True, shell=False):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def build_ssh_cmd(remote_host, remote_user, ssh_key=None, port=22):
    """Build SSH command prefix."""
    if ssh_key:
        return f"ssh -i {ssh_key} -p {port} {remote_user}@{remote_host}"
    else:
        return f"ssh -p {port} {remote_user}@{remote_host}"


def build_scp_cmd(remote_host, remote_user, remote_path, local_path, ssh_key=None, port=22):
    """Build SCP command."""
    if ssh_key:
        return f"scp -i {ssh_key} -P {port} {remote_user}@{remote_host}:{remote_path} {local_path}"
    else:
        return f"scp -P {port} {remote_user}@{remote_host}:{remote_path} {local_path}"


def find_input_files_remote(remote_host, remote_user, remote_path, ssh_key=None, port=22):
    """
    Find source image and audio files on remote server.
    Returns tuple (image_path, audio_path) or (None, None) if not found.
    """
    ssh_cmd = build_ssh_cmd(remote_host, remote_user, ssh_key, port)
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    # Common audio extensions
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac']
    
    # Build find command
    find_cmd = f'{ssh_cmd} "'
    find_cmd += f'if [ -f "{remote_path}" ]; then '
    find_cmd += f'  echo "FILE:{remote_path}"; '
    find_cmd += f'else '
    find_cmd += f'  find "{remote_path}" -type f \\( '
    find_cmd += ' '.join([f'-iname "*{ext}"' for ext in image_extensions + audio_extensions])
    find_cmd += ' \\) 2>/dev/null | head -20; '
    find_cmd += 'fi"'
    
    try:
        result = run_command(find_cmd, shell=True, check=False)
        files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        
        image_files = []
        audio_files = []
        
        for file in files:
            if file.startswith('FILE:'):
                file = file.replace('FILE:', '')
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in image_extensions):
                image_files.append(file)
            elif any(file_lower.endswith(ext) for ext in audio_extensions):
                audio_files.append(file)
        
        source_image = image_files[0] if image_files else None
        driving_audio = audio_files[0] if audio_files else None
        
        return source_image, driving_audio
    except Exception as e:
        print(f"Error finding files on remote server: {e}", file=sys.stderr)
        return None, None


def run_hallo_inference_remote(remote_host, remote_user, remote_hallo_dir, 
                               source_image, driving_audio, remote_output_path,
                               ssh_key=None, port=22):
    """
    Run hallo inference directly on the remote server via SSH.
    
    Args:
        remote_host: Remote server hostname or IP
        remote_user: Remote server username
        remote_hallo_dir: Path to hallo directory on remote server
        source_image: Path to source image on remote server
        driving_audio: Path to driving audio on remote server
        remote_output_path: Path to save output video on remote server
        ssh_key: Path to SSH private key (optional)
        port: SSH port (default: 22)
    """
    print(f"\n{'='*60}")
    print("Running Hallo Lip Sync Inference on Remote Server")
    print(f"{'='*60}")
    print(f"Remote Server: {remote_user}@{remote_host}")
    print(f"Source Image: {source_image}")
    print(f"Driving Audio: {driving_audio}")
    print(f"Output Video: {remote_output_path}")
    print(f"{'='*60}\n")
    
    ssh_cmd = build_ssh_cmd(remote_host, remote_user, ssh_key, port)
    
    # Create output directory on remote server
    remote_output_dir = os.path.dirname(remote_output_path)
    mkdir_cmd = f'{ssh_cmd} "mkdir -p {remote_output_dir}"'
    run_command(mkdir_cmd, shell=True)
    
    # Build inference command
    inference_cmd = f'{ssh_cmd} "cd {remote_hallo_dir} && '
    inference_cmd += f'conda run -n {CONDA_ENV} python scripts/inference.py '
    inference_cmd += f'--source_image "{source_image}" '
    inference_cmd += f'--driving_audio "{driving_audio}" '
    inference_cmd += f'--output "{remote_output_path}""'
    
    try:
        run_command(inference_cmd, shell=True)
        print(f"\n✓ Successfully generated video on remote server: {remote_output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running inference: {e}", file=sys.stderr)
        return False


def copy_output_from_remote(remote_host, remote_user, remote_output_path, 
                            local_output_path, ssh_key=None, port=22):
    """
    Copy the output video from remote server to local machine.
    """
    print(f"\nCopying output video from remote server...")
    print(f"From: {remote_user}@{remote_host}:{remote_output_path}")
    print(f"To: {local_output_path}")
    
    scp_cmd = build_scp_cmd(remote_host, remote_user, remote_output_path, 
                           str(local_output_path), ssh_key, port)
    
    try:
        run_command(scp_cmd, shell=True)
        print(f"✓ Successfully copied video to {local_output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error copying output video: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SSH to remote server and run hallo lip sync inference directly on the server"
    )
    
    # SSH connection details
    parser.add_argument("--remote-host", type=str, required=True,
                        help="Remote server hostname or IP address")
    parser.add_argument("--remote-user", type=str, required=True,
                        help="Remote server username")
    parser.add_argument("--remote-path", type=str, required=True,
                        help="Path to hallo input files on remote server (file or directory)")
    parser.add_argument("--remote-hallo-dir", type=str, required=True,
                        help="Path to hallo directory on remote server")
    parser.add_argument("--ssh-key", type=str, default=None,
                        help="Path to SSH private key (optional)")
    parser.add_argument("--ssh-port", type=int, default=22,
                        help="SSH port (default: 22)")
    
    # Input files (optional if auto-detection fails)
    parser.add_argument("--source-image", type=str, default=None,
                        help="Path to source image on remote server (if not auto-detected)")
    parser.add_argument("--driving-audio", type=str, default=None,
                        help="Path to driving audio on remote server (if not auto-detected)")
    
    # Output settings
    parser.add_argument("--output-name", type=str, default="output.mp4",
                        help="Output video filename (default: output.mp4)")
    parser.add_argument("--remote-output-dir", type=str, default=None,
                        help="Remote output directory (default: same as remote-hallo-dir/halloouput)")
    
    args = parser.parse_args()
    
    # Create local output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Local output directory: {OUTPUT_DIR}")
    
    # Step 1: Find input files on remote server
    print("\n[Step 1/4] Locating input files on remote server...")
    source_image, driving_audio = find_input_files_remote(
        args.remote_host,
        args.remote_user,
        args.remote_path,
        args.ssh_key,
        args.ssh_port
    )
    
    # Use provided paths if auto-detection failed
    if args.source_image:
        source_image = args.source_image
    if args.driving_audio:
        driving_audio = args.driving_audio
    
    if not source_image:
        print(f"Error: Source image not found. Searched in: {args.remote_path}", file=sys.stderr)
        print("Please provide --source-image argument.")
        sys.exit(1)
    
    if not driving_audio:
        print(f"Error: Driving audio not found. Searched in: {args.remote_path}", file=sys.stderr)
        print("Please provide --driving-audio argument.")
        sys.exit(1)
    
    print(f"Found source image: {source_image}")
    print(f"Found driving audio: {driving_audio}")
    
    # Step 2: Setup remote output path
    print("\n[Step 2/4] Setting up output paths...")
    if args.remote_output_dir:
        remote_output_dir = args.remote_output_dir
    else:
        remote_output_dir = os.path.join(args.remote_hallo_dir, "halloouput")
    
    remote_output_path = os.path.join(remote_output_dir, args.output_name)
    local_output_path = OUTPUT_DIR / args.output_name
    
    print(f"Remote output path: {remote_output_path}")
    print(f"Local output path: {local_output_path}")
    
    # Step 3: Run inference on remote server
    print("\n[Step 3/4] Running inference on remote server...")
    if not run_hallo_inference_remote(
        args.remote_host,
        args.remote_user,
        args.remote_hallo_dir,
        source_image,
        driving_audio,
        remote_output_path,
        args.ssh_key,
        args.ssh_port
    ):
        print("\n✗ Inference failed. Check error messages above.")
        sys.exit(1)
    
    # Step 4: Copy output video from remote server
    print("\n[Step 4/4] Copying output video from remote server...")
    if not copy_output_from_remote(
        args.remote_host,
        args.remote_user,
        remote_output_path,
        local_output_path,
        args.ssh_key,
        args.ssh_port
    ):
        print("\n✗ Failed to copy output video. Check error messages above.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✓ SUCCESS!")
    print(f"{'='*60}")
    print(f"Output video saved to: {local_output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
