# Lip Sync API

This FastAPI backend provides endpoints for generating lip-synced videos using open-source libraries.

## Supported Models

| Model | Description | Input | Repository |
|-------|-------------|-------|------------|
| **Wav2Lip** | Accurate lip sync with pre-trained models | Video/Image + Audio | [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip) |
| **SadTalker** | Stylized audio-driven talking face generation | Image + Audio | [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker) |
| **VideoReTalking** | Audio-based lip sync for videos | Video + Audio | [OpenTalker/video-retalking](https://github.com/OpenTalker/video-retalking) |

## API Endpoints

### Generate Lip Sync
```
POST /api/v1/lipsync/generate
```
Upload video/image and audio to generate lip-synced output.

### Check Job Status
```
GET /api/v1/lipsync/status/{job_id}
```

### Download Result
```
GET /api/v1/lipsync/download/{job_id}
```

### List Available Models
```
GET /api/v1/lipsync/models
```

## Setup Instructions

### 1. Install Base Dependencies
```bash
cd backend
uv sync
```

### 2. Install Lip Sync Model (Choose One)

#### Wav2Lip
```bash
git clone https://github.com/Rudrabha/Wav2Lip.git models/wav2lip
cd models/wav2lip
pip install -r requirements.txt
# Download pretrained models from the repo
```

#### SadTalker
```bash
git clone https://github.com/OpenTalker/SadTalker.git models/sadtalker
cd models/sadtalker
pip install -r requirements.txt
# Download checkpoints
```

#### VideoReTalking
```bash
git clone https://github.com/OpenTalker/video-retalking.git models/video_retalking
cd models/video_retalking
pip install -r requirements.txt
```

### 3. Configure Environment
Add to your `.env`:
```
UPLOAD_DIR=./uploads
LIPSYNC_MODEL_DIR=./models
LIPSYNC_DEFAULT_MODEL=wav2lip
MAX_UPLOAD_SIZE_MB=100
```

### 4. Run the Server
```bash
fastapi dev app/main.py
```

## Usage Example

```python
import requests

# Upload files and start job
files = {
    'video': open('face.mp4', 'rb'),
    'audio': open('speech.wav', 'rb')
}
data = {'model': 'wav2lip', 'quality': 'high'}

response = requests.post(
    'http://localhost:8000/api/v1/lipsync/generate',
    files=files,
    data=data
)
job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/api/v1/lipsync/status/{job_id}')
print(status.json())

# Download when complete
if status.json()['status'] == 'completed':
    result = requests.get(f'http://localhost:8000/api/v1/lipsync/download/{job_id}')
    with open('output.mp4', 'wb') as f:
        f.write(result.content)
```

## Implementing Custom Models

To add a new lip sync model, create a processor class in `app/services/lipsync.py`:

```python
class MyCustomProcessor:
    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        config: LipSyncConfig
    ) -> bool:
        # Your implementation here
        return True
```

Then register it in `LipSyncService._processors`.
