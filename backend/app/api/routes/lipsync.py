"""
Lip Sync API Routes

This module provides endpoints for lip sync generation using open-source libraries.
Supported libraries can include: Wav2Lip, SadTalker, VideoReTalking, MuseTalk, etc.
"""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.schemas.lipsync import LipSyncProvider, LipSyncRequest, LipSyncResponse
from app.services.factory import run_lipsync, UnsupportedProviderError
from app.services.lipsync import LipSyncService, LipSyncStatus

router = APIRouter(prefix="/lipsync", tags=["lipsync"])


# =============================================================================
# NEW UNIFIED ENDPOINT - Provider-based API
# =============================================================================

@router.post("/", response_model=LipSyncResponse)
async def lip_sync(request: LipSyncRequest) -> LipSyncResponse:
    """
    Unified lip-sync endpoint supporting multiple providers.
    
    Send a request with the provider name and file paths to generate lip-synced video.
    
    **Supported Providers:**
    - `musetalk` - MuseTalk for high-quality lip sync
    - `wav2lip` - (coming soon)
    - `sadtalker` - (coming soon)
    - `video_retalking` - (coming soon)
    - `latentsync` - (coming soon)
    - `hallo` - (coming soon)
    
    **Example Request:**
    ```json
    {
        "provider": "musetalk",
        "video_path": "/path/to/video.mp4",
        "audio_path": "/path/to/audio.wav",
        "output_path": "/path/to/output.mp4",
        "options": {
            "version": "v15",
            "use_float16": true,
            "fps": 25
        }
    }
    ```
    
    **Returns:**
    - `success`: Whether the operation completed successfully
    - `provider`: The provider that was used
    - `output_path`: Path to the generated video
    - `message`: Status message or error details
    - `metadata`: Additional provider-specific metadata
    """
    return run_lipsync(request)


@router.get("/providers")
async def list_providers() -> dict:
    """
    List all available lip-sync providers and their status.
    """
    return {
        "providers": [
            {
                "name": "musetalk",
                "status": "available",
                "description": "MuseTalk - High-quality lip sync with Whisper audio features",
                "supports_video": True,
                "supports_image": True,
                "options": {
                    "version": "v1 or v15 (default: v15)",
                    "use_float16": "boolean (default: true)",
                    "fps": "integer (default: 25)",
                    "gpu_id": "integer (default: 0)"
                }
            },
            {
                "name": "wav2lip",
                "status": "coming_soon",
                "description": "Wav2Lip - Accurate lip sync with pre-trained models"
            },
            {
                "name": "sadtalker",
                "status": "coming_soon",
                "description": "SadTalker - Stylized audio-driven talking face generation"
            },
            {
                "name": "video_retalking",
                "status": "coming_soon",
                "description": "VideoReTalking - Audio-based lip sync for videos"
            },
            {
                "name": "latentsync",
                "status": "coming_soon",
                "description": "LatentSync - Latent space lip sync"
            },
            {
                "name": "hallo",
                "status": "coming_soon",
                "description": "Hallo - Audio-driven portrait animation"
            }
        ]
    }


# =============================================================================
# LEGACY ENDPOINTS - Keep for backward compatibility
# =============================================================================


class LipSyncJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class LipSyncStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    output_url: str | None = None
    error: str | None = None


class LipSyncConfig(BaseModel):
    """Configuration options for lip sync generation"""
    model: str = "wav2lip"  # wav2lip, sadtalker, video_retalking
    quality: str = "medium"  # low, medium, high
    fps: int = 25
    resize_factor: int = 1


# In-memory job storage (replace with database in production)
jobs: dict[str, LipSyncStatus] = {}


@router.post("/generate", response_model=LipSyncJobResponse)
async def generate_lipsync(
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File(description="Source video or image file")],
    audio: Annotated[UploadFile, File(description="Audio file for lip sync")],
    model: Annotated[str, Form()] = "wav2lip",
    quality: Annotated[str, Form()] = "medium",
    resize_factor: Annotated[int, Form()] = 2,
) -> LipSyncJobResponse:
    """
    Generate lip-synced video from source video/image and audio.

    - **video**: Source video or image file (mp4, avi, jpg, png)
    - **audio**: Audio file to sync lips to (wav, mp3)
    - **model**: Lip sync model to use (wav2lip, sadtalker, video_retalking)
    - **quality**: Output quality (low, medium, high)
    - **resize_factor**: Reduce resolution by this factor (1=original, 2=half, 4=quarter). Use 2-4 for large videos.

    Returns a job ID to track the processing status.
    """
    # Validate file types
    allowed_video_types = ["video/mp4", "video/avi", "image/jpeg", "image/png"]
    allowed_audio_types = ["audio/wav", "audio/mpeg", "audio/mp3"]
    
    if video.content_type not in allowed_video_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video type. Allowed: {allowed_video_types}"
        )
    
    if audio.content_type not in allowed_audio_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio type. Allowed: {allowed_audio_types}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create upload directory
    upload_dir = Path(settings.UPLOAD_DIR) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    video_path = upload_dir / f"input_video{Path(video.filename or 'video').suffix}"
    audio_path = upload_dir / f"input_audio{Path(audio.filename or 'audio').suffix}"
    
    video_content = await video.read()
    audio_content = await audio.read()
    
    video_path.write_bytes(video_content)
    audio_path.write_bytes(audio_content)
    
    # Initialize job status
    jobs[job_id] = LipSyncStatus(
        job_id=job_id,
        status="queued",
        progress=0.0
    )
    
    # Start background processing
    config = LipSyncConfig(model=model, quality=quality, resize_factor=resize_factor)
    background_tasks.add_task(
        LipSyncService.process_lipsync,
        job_id=job_id,
        video_path=str(video_path),
        audio_path=str(audio_path),
        config=config,
        jobs=jobs
    )
    
    return LipSyncJobResponse(
        job_id=job_id,
        status="queued",
        message="Lip sync job queued for processing"
    )


@router.get("/status/{job_id}", response_model=LipSyncStatusResponse)
async def get_job_status(job_id: str) -> LipSyncStatusResponse:
    """
    Get the status of a lip sync job.
    
    - **job_id**: The job ID returned from /generate endpoint
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return LipSyncStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        output_url=job.output_url,
        error=job.error
    )


@router.get("/models")
async def list_available_models() -> dict:
    """
    List available lip sync models and their capabilities.
    """
    return {
        "models": [
            {
                "name": "wav2lip",
                "description": "Wav2Lip - Accurate lip sync with pre-trained models",
                "supports_video": True,
                "supports_image": True,
                "quality_options": ["low", "medium", "high"],
                "repo": "https://github.com/Rudrabha/Wav2Lip"
            },
            {
                "name": "sadtalker",
                "description": "SadTalker - Stylized audio-driven talking face generation",
                "supports_video": False,
                "supports_image": True,
                "quality_options": ["medium", "high"],
                "repo": "https://github.com/OpenTalker/SadTalker"
            },
            {
                "name": "video_retalking",
                "description": "VideoReTalking - Audio-based lip sync for videos",
                "supports_video": True,
                "supports_image": False,
                "quality_options": ["medium", "high"],
                "repo": "https://github.com/OpenTalker/video-retalking"
            }
        ]
    }


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """
    Cancel a pending or running lip sync job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    job.status = "cancelled"
    return {"message": f"Job {job_id} cancelled"}


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """
    Download the completed lip sync video.
    """
    from fastapi.responses import FileResponse
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not ready. Current status: {job.status}"
        )
    
    output_path = Path(settings.UPLOAD_DIR) / job_id / "output.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"lipsync_{job_id}.mp4"
    )
