"""
Test script for LipSync API
"""

import requests
import time
import os

API_BASE = "http://localhost:8001/api/v1"

# Test files
IMAGE_PATH = "models/sadtalker/examples/source_image/full3.png"
AUDIO_PATH = "models/sadtalker/examples/driven_audio/RD_Radio31_000.wav"

def test_generate(model="sadtalker"):
    """Generate lip-synced video"""
    print(f"\n{'='*50}")
    print(f"Testing {model.upper()} generation")
    print(f"{'='*50}")

    # Check if files exist
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        return None
    if not os.path.exists(AUDIO_PATH):
        print(f"ERROR: Audio not found: {AUDIO_PATH}")
        return None

    print(f"Image: {IMAGE_PATH}")
    print(f"Audio: {AUDIO_PATH}")
    print(f"Model: {model}")

    # Submit job
    print("\nSubmitting job...")
    with open(IMAGE_PATH, "rb") as img_file, open(AUDIO_PATH, "rb") as audio_file:
        files = {
            "video": ("image.png", img_file, "image/png"),
            "audio": ("audio.wav", audio_file, "audio/wav")
        }
        data = {
            "model": model,
            "quality": "medium",
            "resize_factor": 1
        }

        response = requests.post(f"{API_BASE}/lipsync/generate", files=files, data=data)

    if response.status_code != 200:
        print(f"ERROR: {response.status_code} - {response.text}")
        return None

    result = response.json()
    job_id = result["job_id"]
    print(f"Job submitted: {job_id}")
    print(f"Status: {result['status']}")

    # Poll for status
    print("\nWaiting for processing...")
    max_wait = 600  # 10 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait:
        status_response = requests.get(f"{API_BASE}/lipsync/status/{job_id}")
        status = status_response.json()

        print(f"  Status: {status['status']}, Progress: {status['progress']*100:.1f}%")

        if status["status"] == "completed":
            print(f"\nSUCCESS! Output URL: {status['output_url']}")

            # Download the result
            output_path = f"output_{model}_{job_id[:8]}.mp4"
            download_response = requests.get(f"{API_BASE}/lipsync/download/{job_id}")

            if download_response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(download_response.content)
                print(f"Downloaded to: {output_path}")

            return job_id

        elif status["status"] == "failed":
            print(f"\nFAILED: {status.get('error', 'Unknown error')}")
            return None

        time.sleep(5)

    print("\nTIMEOUT: Job took too long")
    return None

if __name__ == "__main__":
    print("LipSync API Test")
    print(f"API Base: {API_BASE}")

    # Test available models
    print("\nAvailable models:")
    models_response = requests.get(f"{API_BASE}/lipsync/models")
    models = models_response.json()["models"]
    for m in models:
        print(f"  - {m['name']}: {m['description']}")

    # Test with Wav2Lip (SadTalker requires more memory)
    test_generate("wav2lip")
