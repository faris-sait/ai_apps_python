"""
Lip Sync API Routes

This module provides endpoints for lip sync generation using open-source libraries.
Supported libraries can include: Wav2Lip, SadTalker, VideoReTalking, MuseTalk, etc.
"""

import asyncio
import json
import random
import string
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.lipsync import LipSyncProvider, LipSyncRequest, LipSyncResponse
from app.services.factory import run_lipsync, UnsupportedProviderError
from app.services.lipsync import LipSyncService, LipSyncStatus

router = APIRouter(prefix="/lipsync", tags=["lipsync"])


# =============================================================================
# JOB MANAGEMENT - Background processing with DB persistence
# =============================================================================

class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobInfo(BaseModel):
    """Job information model"""
    id: str
    status: JobStatus
    provider: LipSyncProvider
    mode: str
    input_path: str | None
    audio_path: str
    output_path: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str | None = None
    log: str | None = None
    log_path: str | None = None
    metadata: dict | None = None


# In-memory job cache (also persisted to DB)
jobs_cache: dict[str, JobInfo] = {}


# =============================================================================
# NEW UNIFIED ENDPOINT - Provider-based API
# =============================================================================

@router.post("/", response_model=LipSyncResponse)
async def lip_sync(request: LipSyncRequest) -> LipSyncResponse:
    """
    Unified lip-sync endpoint supporting multiple providers.
    
    Send a request with the provider name, mode, and file paths to generate lip-synced video.
    
    **Modes:**
    - `image_audio` - Generate lip-synced video from still image + audio
    - `video_audio` - Generate lip-synced video from video + audio
    
    **Supported Providers:**
    - `musetalk` - MuseTalk for high-quality lip sync (supports image & video)
    - `sadtalker` - SadTalker for stylized talking face generation (image only)
    
    **Example Request (Video + Audio):**
    ```json
    {
        "provider": "musetalk",
        "mode": "video_audio",
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
    
    **Example Request (Image + Audio):**
    ```json
    {
        "provider": "musetalk",
        "mode": "image_audio",
        "image_path": "/path/to/portrait.jpg",
        "audio_path": "/path/to/audio.wav",
        "output_path": "/path/to/output.mp4",
        "options": {
            "version": "v15",
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
                "name": "sadtalker",
                "status": "available",
                "description": "SadTalker - Stylized audio-driven talking face generation",
                "supports_video": False,
                "supports_image": True,
                "options": {
                    "pose_style": "integer 0-45 (default: 0)",
                    "expression_scale": "float (default: 1.0)",
                    "enhancer": "'gfpgan' or 'RestoreFormer' (default: none)",
                    "preprocess": "'crop', 'resize', or 'full' (default: 'crop')",
                    "still": "boolean (default: false)",
                    "size": "256 or 512 (default: 256)"
                }
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


@router.get("/r2/files")
async def list_r2_files(prefix: str = "AI_video/") -> dict:
    """
    List available files in R2 storage.
    
    - **prefix**: Folder prefix to filter by (default: AI_video/)
    
    Returns list of files with their keys and metadata.
    Files can be used as input paths with r2:// prefix or AI_video/ prefix.
    """
    try:
        from app.services.r2_storage import list_r2_files as r2_list_files
        files = r2_list_files(prefix)
        return {
            "files": files,
            "count": len(files),
            "prefix": prefix,
            "usage_hint": "Use file keys as input paths, e.g., 'AI_video/video.mp4' or 'r2://AI_video/video.mp4'"
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="R2 storage not configured")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list R2 files: {str(e)}")


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


# =============================================================================
# BACKGROUND JOBS API - Database-backed job tracking
# =============================================================================

# Import DB functions (create a stub if db.py doesn't exist in backend)
try:
    import sys
    from pathlib import Path as P
    # Add parent directory to path to import root-level db.py
    root_dir = P(__file__).parent.parent.parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from db import (
        create_job_record, 
        update_job_record, 
        get_job_record, 
        list_job_records,
        append_job_log,
        get_job_logs as db_get_job_logs,
        init_db
    )
    # Initialize DB on module load
    init_db()
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    # Stub functions if DB not available
    def create_job_record(data): pass
    def update_job_record(job_id, updates): pass
    def get_job_record(job_id): return None
    def list_job_records(): return []
    def append_job_log(job_id, content): pass
    def db_get_job_logs(job_id): return []


# Import R2 storage service
try:
    from app.services.r2_storage import temp_r2_files, is_r2_path, upload_to_r2
    R2_AVAILABLE = True
except ImportError:
    R2_AVAILABLE = False
    def is_r2_path(path): return False


# Import provider runners (from root-level lipsync_api.py)
try:
    from lipsync_api import run_musetalk, run_sadtalker, LipSyncMode
except ImportError:
    # Fallback stubs
    class LipSyncMode(str, Enum):
        IMAGE_AUDIO = "image_audio"
        VIDEO_AUDIO = "video_audio"
    
    def run_musetalk(input_path, audio_path, output_path, options):
        return {"success": False, "message": "MuseTalk not configured"}
    
    def run_sadtalker(input_path, audio_path, output_path, options):
        return {"success": False, "message": "SadTalker not configured"}


def _run_lipsync_with_r2(job_id: str, request: LipSyncRequest, input_path: str, audio_path: str, options: dict) -> dict:
    """
    Run lipsync processing, downloading R2 files to temp if needed.
    Temp files are automatically cleaned up after processing.
    """
    # Check if any paths are R2 paths
    input_is_r2 = is_r2_path(input_path)
    audio_is_r2 = is_r2_path(audio_path)
    
    if R2_AVAILABLE and (input_is_r2 or audio_is_r2):
        # Use temp directory for R2 files
        with temp_r2_files(prefix=f"lipsync_{job_id}_") as temp:
            # Download R2 files to temp, or use local paths as-is
            local_input = temp.get_local_path(input_path)
            local_audio = temp.get_local_path(audio_path)
            
            # Run the provider with local paths
            if request.provider == LipSyncProvider.MUSETALK:
                result = run_musetalk(local_input, local_audio, request.output_path, options)
            elif request.provider == LipSyncProvider.SADTALKER:
                result = run_sadtalker(local_input, local_audio, request.output_path, options)
            else:
                result = {"success": False, "message": f"Unknown provider: {request.provider}"}
            
            # If successful and output should go to R2, upload it
            if result.get("success") and result.get("output_path"):
                output_path = result["output_path"]
                # Optionally upload output to R2
                if request.output_path and is_r2_path(request.output_path):
                    r2_key = request.output_path.replace("r2://", "")
                    upload_to_r2(output_path, r2_key)
                    result["r2_output_key"] = r2_key
            
            return result
            # Temp files automatically cleaned up here
    else:
        # No R2 paths, run directly with local files
        if request.provider == LipSyncProvider.MUSETALK:
            return run_musetalk(input_path, audio_path, request.output_path, options)
        elif request.provider == LipSyncProvider.SADTALKER:
            return run_sadtalker(input_path, audio_path, request.output_path, options)
        else:
            return {"success": False, "message": f"Unknown provider: {request.provider}"}


async def _run_background_job(job_id: str, request: LipSyncRequest) -> None:
    """Internal: run a job in background and update its status."""
    job = jobs_cache[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()
    
    # Update DB
    update_job_record(job_id, {
        "status": job.status.value,
        "started_at": job.started_at
    })
    
    # Determine input path
    input_path = request.image_path if request.mode == "image_audio" else request.video_path
    options = request.options or {}
    options["input_type"] = "image" if request.mode == "image_audio" else "video"
    options["job_id"] = job_id  # Add job_id for unique output paths
    
    try:
        # Run provider (with R2 support)
        result = await asyncio.to_thread(
            _run_lipsync_with_r2, 
            job_id, 
            request, 
            input_path, 
            request.audio_path, 
            options
        )
        
        job.finished_at = datetime.utcnow()
        job.message = result.get("message")
        job.metadata = result.get("metadata")
        job.output_path = result.get("output_path")
        
        # Write logs to file
        logs_dir = Path(settings.UPLOAD_DIR).parent / "job_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{job.id}.log"
        raw_stdout = result.get("raw_stdout", "")
        raw_stderr = result.get("raw_stderr", "")
        
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"Job {job.id}\n")
            fh.write(f"Provider: {job.provider}\n")
            fh.write(f"Mode: {job.mode}\n")
            fh.write(f"Input: {job.input_path}\n")
            fh.write(f"Audio: {job.audio_path}\n")
            fh.write(f"Output: {job.output_path}\n")
            fh.write(f"Created: {job.created_at}\n")
            fh.write(f"Started: {job.started_at}\n")
            fh.write(f"Finished: {job.finished_at}\n")
            fh.write(f"\n{'='*60}\nSTDOUT:\n{'='*60}\n")
            fh.write(raw_stdout)
            fh.write(f"\n{'='*60}\nSTDERR:\n{'='*60}\n")
            fh.write(raw_stderr)
        
        job.log_path = str(log_path)
        job.log = f"Logs saved to {log_path}"
        
        # Persist logs to DB
        if raw_stdout:
            append_job_log(job.id, f"STDOUT:\n{raw_stdout}")
        if raw_stderr:
            append_job_log(job.id, f"STDERR:\n{raw_stderr}")
        
        # Update final status
        if result.get("success"):
            job.status = JobStatus.SUCCEEDED
            if not job.log:
                job.log = "Job completed successfully"
        else:
            job.status = JobStatus.FAILED
            if not job.log:
                job.log = job.message or "No error message"
        
    except Exception as e:
        job.finished_at = datetime.utcnow()
        job.status = JobStatus.FAILED
        job.message = str(e)
        job.log = str(e)
        
        # Write error to log file
        logs_dir = Path(settings.UPLOAD_DIR).parent / "job_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{job.id}.log"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("\nException:\n")
            fh.write(str(e))
        job.log_path = str(log_path)
        
        append_job_log(job.id, f"Exception: {e}")
    
    # Final DB update
    update_job_record(job_id, {
        "status": job.status.value,
        "finished_at": job.finished_at,
        "message": job.message,
        "log_path": job.log_path,
        "output_path": job.output_path,
        "metadata_json": json.dumps(job.metadata) if job.metadata else None
    })


@router.post("/jobs", status_code=202)
async def create_lipsync_job(request: LipSyncRequest):
    """
    Create a background lip-sync job and return job ID.
    
    This endpoint accepts a lip-sync request and processes it asynchronously.
    Use the returned job_id to poll for status and retrieve results.
    """
    # Validate input
    if request.mode == "image_audio":
        if not request.image_path:
            raise HTTPException(status_code=400, detail="image_path required for image_audio mode")
        input_path = request.image_path
    else:
        if not request.video_path:
            raise HTTPException(status_code=400, detail="video_path required for video_audio mode")
        input_path = request.video_path
    
    # Generate full UUID job ID
    job_id = str(uuid.uuid4())
    
    job = JobInfo(
        id=job_id,
        status=JobStatus.PENDING,
        provider=request.provider,
        mode=request.mode,
        input_path=input_path,
        audio_path=request.audio_path,
        output_path=request.output_path,
        created_at=datetime.utcnow()
    )
    
    jobs_cache[job_id] = job
    
    # Persist to DB
    create_job_record({
        "id": job.id,
        "provider": job.provider.value,
        "mode": job.mode,
        "input_path": job.input_path,
        "audio_path": job.audio_path,
        "output_path": job.output_path,
        "status": job.status.value,
        "created_at": job.created_at,
        "message": job.message,
        "log_path": job.log_path,
        "metadata_json": json.dumps(job.metadata) if job.metadata else None
    })
    
    # Schedule background run
    asyncio.create_task(_run_background_job(job_id, request))
    
    return {"job_id": job_id, "status": job.status}


@router.get("/jobs/{job_id}")
async def get_lipsync_job(job_id: str):
    """Get job status and details."""
    job = jobs_cache.get(job_id)
    if not job:
        # Try fetching from DB
        db_job = get_job_record(job_id)
        if not db_job:
            raise HTTPException(status_code=404, detail="Job not found")
        return db_job
    return job


@router.get("/jobs")
async def list_lipsync_jobs():
    """List all jobs (from cache and DB)."""
    all_jobs = list(jobs_cache.values())
    # Optionally merge with DB jobs
    if DB_AVAILABLE:
        db_jobs = list_job_records()
        # Add DB jobs not in cache
        cache_ids = {j.id for j in all_jobs}
        for db_job in db_jobs:
            if db_job.id not in cache_ids:
                all_jobs.append(db_job)
    return {"jobs": all_jobs}


@router.get("/jobs/{job_id}/logs")
async def get_lipsync_job_logs(job_id: str):
    """Return job logs as plain text (tries filesystem then DB)."""
    job = jobs_cache.get(job_id)
    if not job:
        job = get_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Try log file on disk
    if getattr(job, "log_path", None):
        log_file = Path(job.log_path)
        if log_file.exists():
            return FileResponse(log_file, media_type="text/plain", filename=f"{job_id}.log")
    
    # Fallback to DB logs
    logs = []
    try:
        db_logs = db_get_job_logs(job_id)
        logs = [l.content for l in db_logs]
    except Exception:
        logs = ["No logs available"]
    
    return PlainTextResponse("\n\n".join(logs))
