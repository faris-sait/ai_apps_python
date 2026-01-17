"""
Lip-Sync API - Standalone FastAPI Application

Run with: uvicorn lipsync_api:app --reload --host 0.0.0.0 --port 8000
"""

from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =============================================================================
# SCHEMAS
# =============================================================================

class LipSyncProvider(str, Enum):
    """Supported lip-sync providers"""
    MUSETALK = "musetalk"
    SADTALKER = "sadtalker"


class LipSyncMode(str, Enum):
    """Input mode for lip-sync generation"""
    IMAGE_AUDIO = "image_audio"
    VIDEO_AUDIO = "video_audio"


class LipSyncRequest(BaseModel):
    """Request schema for lip-sync generation"""
    provider: LipSyncProvider = Field(..., description="The lip-sync provider to use")
    mode: LipSyncMode = Field(..., description="Input mode: 'image_audio' or 'video_audio'")
    video_path: str | None = Field(default=None, description="Path to input video (for video_audio mode)")
    image_path: str | None = Field(default=None, description="Path to input image (for image_audio mode)")
    audio_path: str = Field(..., description="Path to input audio file")
    output_path: str | None = Field(default=None, description="Optional output path")
    options: dict[str, Any] | None = Field(default=None, description="Provider-specific options")


class LipSyncResponse(BaseModel):
    """Response schema for lip-sync generation"""
    success: bool
    provider: str
    output_path: str | None = None
    message: str | None = None
    metadata: dict[str, Any] | None = None


# =============================================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================================

import logging
import os
import subprocess
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = Path(__file__).parent
MUSETALK_DIR = BASE_DIR / "MuseTalk"
SADTALKER_DIR = BASE_DIR / "models" / "sadtalker"

# Add MuseTalk to Python path for imports
if str(MUSETALK_DIR) not in sys.path:
    sys.path.insert(0, str(MUSETALK_DIR))


def validate_inputs(video_path: str, audio_path: str) -> tuple[bool, str]:
    """Validate input files exist."""
    if not Path(video_path).exists():
        return False, f"Input file not found: {video_path}"
    if not Path(audio_path).exists():
        return False, f"Audio file not found: {audio_path}"
    return True, ""


def run_musetalk(
    input_path: str,
    audio_path: str,
    output_path: str | None,
    options: dict[str, Any] | None
) -> dict[str, Any]:
    """Run MuseTalk inference."""
    is_valid, error_msg = validate_inputs(input_path, audio_path)
    if not is_valid:
        return {"success": False, "output_path": None, "message": error_msg}
    
    if not MUSETALK_DIR.exists():
        return {"success": False, "output_path": None, "message": f"MuseTalk not found at {MUSETALK_DIR}"}
    
    opts = options or {}
    version = opts.get("version", "v15")
    fps = opts.get("fps", 25)
    job_id = opts.get("job_id", None)
    
    # Generate output path - save directly to MuseTalk_inputs_outputs folder
    if not output_path:
        output_dir = Path("/home/vineeth/Documents/projects/avatar_faris/ai_apps_python/MuseTalk_inputs_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if job_id:
            # Use job_id in filename for uniqueness
            output_path = str(output_dir / f"{job_id}.mp4")
        else:
            # Fallback to timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"musetalk_output_{timestamp}.mp4")
    
    try:
        # Use our custom API wrapper
        cmd = [
            sys.executable,
            str(MUSETALK_DIR / "api_wrapper.py"),
            "--video", input_path,
            "--audio", audio_path,
            "--output", str(Path(output_path).parent),
            "--version", version,
            "--fps", str(fps),
        ]
        if job_id:
            cmd.extend(["--job_id", job_id])
        
        try:
            # increase timeout to 1 hour (may take long to load models)
            result = subprocess.run(cmd, cwd=str(MUSETALK_DIR), capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired as te:
            stdout = te.stdout.decode() if isinstance(te.stdout, (bytes, bytearray)) else (te.stdout or "")
            stderr = te.stderr.decode() if isinstance(te.stderr, (bytes, bytearray)) else (te.stderr or "")
            return {"success": False, "output_path": None, "message": f"MuseTalk timed out after 1 hour. stdout: {stdout[:1000]} stderr: {stderr[:1000]}"}
        
        # Capture raw stdout/stderr for logs
        raw_stdout = result.stdout or ""
        raw_stderr = result.stderr or ""

        if result.returncode != 0:
            error_msg = raw_stderr if raw_stderr else raw_stdout
            return {"success": False, "output_path": None, "message": f"MuseTalk failed: {error_msg[:1000]}", "raw_stdout": raw_stdout, "raw_stderr": raw_stderr}
        
        # MuseTalk generates output as: output_dir/version/{job_id}.mp4 or {videoname}_{audioname}.mp4
        # We'll move it to the final output path
        if job_id:
            temp_output = Path(output_path).parent / version / f"{job_id}.mp4"
            final_output = Path(output_path).parent / f"{job_id}.mp4"
        else:
            video_name = Path(input_path).stem
            audio_name = Path(audio_path).stem
            temp_output = Path(output_path).parent / version / f"{video_name}_{audio_name}.mp4"
            final_output = Path(output_path).parent / f"{video_name}_{audio_name}.mp4"
        
        # Check if output exists in version subfolder
        if not temp_output.exists():
            return {"success": False, "output_path": None, "message": f"MuseTalk completed but output not found at {temp_output}", "raw_stdout": raw_stdout, "raw_stderr": raw_stderr}
        
        # Move file to final location
        import shutil
        shutil.move(str(temp_output), str(final_output))
        
        return {"success": True, "output_path": str(final_output), "message": "MuseTalk completed", "metadata": {"version": version}, "raw_stdout": raw_stdout, "raw_stderr": raw_stderr}
    
    except Exception as e:
        return {"success": False, "output_path": None, "message": str(e)}


def run_sadtalker(
    input_path: str,
    audio_path: str,
    output_path: str | None,
    options: dict[str, Any] | None
) -> dict[str, Any]:
    """Run SadTalker inference."""
    is_valid, error_msg = validate_inputs(input_path, audio_path)
    if not is_valid:
        return {"success": False, "output_path": None, "message": error_msg}
    
    if not SADTALKER_DIR.exists():
        return {"success": False, "output_path": None, "message": f"SadTalker not found at {SADTALKER_DIR}"}
    
    opts = options or {}
    job_id = opts.get("job_id", None)
    
    # SadTalker only supports images
    if opts.get("input_type") == "video":
        return {"success": False, "output_path": None, "message": "SadTalker only supports image input. Use mode='image_audio'"}
    
    pose_style = opts.get("pose_style", 0)
    preprocess = opts.get("preprocess", "crop")
    size = opts.get("size", 256)
    
    # Generate output path - save directly to Sadtalker_inputs_outputs folder
    if not output_path:
        output_dir = BASE_DIR / "Sadtalker_inputs_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if job_id:
            # Use job_id in filename for uniqueness
            output_path = str(output_dir / f"{job_id}.mp4")
        else:
            # Fallback to timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"sadtalker_output_{timestamp}.mp4")
    
    try:
        cmd = [
            sys.executable,
            str(SADTALKER_DIR / "inference.py"),
            "--source_image", input_path,
            "--driven_audio", audio_path,
            "--result_dir", str(Path(output_path).parent),
            "--checkpoint_dir", str(SADTALKER_DIR / "checkpoints"),
            "--pose_style", str(pose_style),
            "--preprocess", preprocess,
            "--size", str(size),
        ]
        
        if opts.get("enhancer"):
            cmd.extend(["--enhancer", opts["enhancer"]])
        
        result = subprocess.run(cmd, cwd=str(SADTALKER_DIR), capture_output=True, text=True)
        
        raw_stdout = result.stdout or ""
        raw_stderr = result.stderr or ""

        if result.returncode != 0:
            return {"success": False, "output_path": None, "message": f"SadTalker failed: {raw_stderr[:500]}", "raw_stdout": raw_stdout, "raw_stderr": raw_stderr}
        
        # SadTalker generates its own filename with timestamp
        # Find the most recent .mp4 file in the output directory and rename it to job_id
        output_dir = Path(output_path).parent
        mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not mp4_files:
            return {"success": False, "output_path": None, "message": "SadTalker completed but no output file found", "raw_stdout": raw_stdout, "raw_stderr": raw_stderr}
        
        # Get the most recent file (the one just created)
        generated_file = mp4_files[0]
        
        # If job_id is provided, rename the file to use job_id
        if job_id:
            final_output = output_dir / f"{job_id}.mp4"
            import shutil
            shutil.move(str(generated_file), str(final_output))
            actual_output_path = str(final_output)
        else:
            actual_output_path = str(generated_file)
        
        return {"success": True, "output_path": actual_output_path, "message": "SadTalker completed", "metadata": {"pose_style": pose_style}, "raw_stdout": raw_stdout, "raw_stderr": raw_stderr}
    
    except Exception as e:
        return {"success": False, "output_path": None, "message": str(e)}


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Lip-Sync API",
    description="Unified API for lip-sync generation using MuseTalk and SadTalker",
    version="1.0.0"
)

# Background job support
import asyncio
from uuid import uuid4
from typing import Dict, Optional

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobInfo(BaseModel):
    id: str
    status: JobStatus
    provider: LipSyncProvider
    mode: LipSyncMode
    input_path: str
    audio_path: str
    output_path: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    message: Optional[str] = None
    log: Optional[str] = None
    log_path: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


# In-memory job store (kept for quick read cache)
jobs: Dict[str, JobInfo] = {}

# Initialize DB (creates tables if necessary)
from db import init_db, create_job_record, update_job_record, get_job_record, append_job_log, list_job_records, test_connection
init_db()
# Test DB connection at startup and print a short notice
try:
    ok = test_connection()
    if ok:
        logger.info("DB connection successful")
    else:
        logger.warning("DB connection failed - jobs will use local SQLite until configured")
except Exception as e:
    logger.warning("DB connection test raised an exception: %s", e)


async def _run_job(job_id: str, request: LipSyncRequest) -> None:
    """Internal: run a job in a background task and update its status."""
    job = jobs[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()

    # Determine input path
    if request.mode == LipSyncMode.IMAGE_AUDIO:
        input_path = request.image_path
    else:
        input_path = request.video_path

    options = request.options or {}
    options["input_type"] = "image" if request.mode == LipSyncMode.IMAGE_AUDIO else "video"

    try:
        if request.provider == LipSyncProvider.MUSETALK:
            result = await asyncio.to_thread(run_musetalk, input_path, request.audio_path, request.output_path, options)
        elif request.provider == LipSyncProvider.SADTALKER:
            result = await asyncio.to_thread(run_sadtalker, input_path, request.audio_path, request.output_path, options)
        else:
            result = {"success": False, "message": f"Unknown provider: {request.provider}"}

        job.finished_at = datetime.utcnow()
        job.message = result.get("message")
        job.metadata = result.get("metadata")
        job.output_path = result.get("output_path")

        # write logs to file
        logs_dir = BASE_DIR / "job_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{job.id}.log"
        raw_stdout = result.get("raw_stdout", "")
        raw_stderr = result.get("raw_stderr", "")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"Job {job.id}\n")
            fh.write(f"Provider: {job.provider}\n")
            fh.write(f"Mode: {job.mode}\n")
            fh.write(f"Input: {job.input_path}\n")
            fh.write(f"Started: {job.started_at}\n")
            fh.write("\n=== STDOUT ===\n")
            fh.write(raw_stdout)
            fh.write("\n=== STDERR ===\n")
            fh.write(raw_stderr)

        job.log = (raw_stdout + "\n" + raw_stderr)[:2000]
        job.log_path = str(log_path)

        # Persist updates to DB (non-blocking via to_thread)
        try:
            update_job_record(job.id, {
                "status": JobStatus.SUCCEEDED.value if result.get("success") else JobStatus.FAILED.value,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "message": job.message,
                "output_path": job.output_path,
                "log_path": job.log_path,
                "metadata_json": json.dumps(job.metadata) if job.metadata else None,
            })

            # append logs to DB
            append_job_log(job.id, "=== STDOUT ===\n" + raw_stdout + "\n=== STDERR ===\n" + raw_stderr)
        except Exception as e:
            # keep going; DB is best-effort
            print("DB update failed:", e)

        if result.get("success"):
            job.status = JobStatus.SUCCEEDED
        else:
            job.status = JobStatus.FAILED
            if not job.log:
                job.log = job.message or "No error message"
    except Exception as e:
        job.finished_at = datetime.utcnow()
        job.status = JobStatus.FAILED
        job.message = str(e)
        job.log = str(e)
        # ensure log file exists
        logs_dir = BASE_DIR / "job_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{job.id}.log"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("\nException:\n")
            fh.write(str(e))
        job.log_path = str(log_path)



@app.post("/api/v1/lipsync/", response_model=LipSyncResponse)
async def lip_sync(request: LipSyncRequest) -> LipSyncResponse:
    """
    Generate lip-synced video.
    
    **Modes:**
    - `image_audio` - Image + audio → lip-synced video
    - `video_audio` - Video + audio → lip-synced video
    
    **Providers:**
    - `musetalk` - Supports both image and video input
    - `sadtalker` - Supports only image input
    """
    # Validate mode and get input path
    if request.mode == LipSyncMode.IMAGE_AUDIO:
        if not request.image_path:
            raise HTTPException(status_code=400, detail="image_path required for image_audio mode")
        input_path = request.image_path
        input_type = "image"
    else:
        if not request.video_path:
            raise HTTPException(status_code=400, detail="video_path required for video_audio mode")
        input_path = request.video_path
        input_type = "video"
    
    # Add input_type to options
    options = request.options or {}
    options["input_type"] = input_type
    
    # Route to provider
    if request.provider == LipSyncProvider.MUSETALK:
        result = run_musetalk(input_path, request.audio_path, request.output_path, options)
    elif request.provider == LipSyncProvider.SADTALKER:
        result = run_sadtalker(input_path, request.audio_path, request.output_path, options)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
    
    return LipSyncResponse(
        success=result.get("success", False),
        provider=request.provider.value,
        output_path=result.get("output_path"),
        message=result.get("message"),
        metadata=result.get("metadata")
    )


# -------------------- Background jobs endpoints --------------------

@app.post("/api/v1/lipsync/jobs", status_code=202)
async def create_job(request: LipSyncRequest):
    """Create a background lip-sync job and return job id."""
    # Validate input same as sync endpoint
    if request.mode == LipSyncMode.IMAGE_AUDIO:
        if not request.image_path:
            raise HTTPException(status_code=400, detail="image_path required for image_audio mode")
    else:
        if not request.video_path:
            raise HTTPException(status_code=400, detail="video_path required for video_audio mode")

    job_id = str(uuid4())
    input_path = request.image_path if request.mode == LipSyncMode.IMAGE_AUDIO else request.video_path

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

    jobs[job_id] = job

    # Persist job to DB (non-blocking): create record
    create_job_record({
        "id": job.id,
        "provider": job.provider.value,
        "mode": job.mode.value,
        "input_path": job.input_path,
        "audio_path": job.audio_path,
        "output_path": job.output_path,
        "status": job.status.value,
        "created_at": job.created_at,
        "message": job.message,
        "log_path": job.log_path,
        "metadata_json": None
    })

    # schedule background run
    asyncio.create_task(_run_job(job_id, request))

    return {"job_id": job_id, "status": job.status}


@app.get("/api/v1/lipsync/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and details."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/v1/lipsync/jobs")
async def list_jobs():
    """List all jobs."""
    return {"jobs": list(jobs.values())}


@app.get("/api/v1/lipsync/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Return job logs as plain text file (tries filesystem then DB)."""
    job = jobs.get(job_id) or get_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # If log file exists on disk, return it
    if getattr(job, "log_path", None):
        from fastapi.responses import FileResponse
        if Path(job.log_path).exists():
            return FileResponse(job.log_path, media_type="text/plain", filename=f"{job_id}.log")
    # Fall back: return DB logs concatenated
    logs = []
    try:
        db_logs = get_job_logs(job_id)
        logs = [l.content for l in db_logs]
    except Exception:
        logs = ["No logs available"]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n\n".join(logs))



@app.get("/api/v1/lipsync/providers")
async def list_providers():
    """List available lip-sync providers."""
    return {
        "providers": [
            {
                "name": "musetalk",
                "status": "available",
                "supports_video": True,
                "supports_image": True,
                "options": {"version": "v15", "use_float16": True, "fps": 25}
            },
            {
                "name": "sadtalker",
                "status": "available",
                "supports_video": False,
                "supports_image": True,
                "options": {"pose_style": 0, "preprocess": "crop", "size": 256, "enhancer": None}
            }
        ]
    }


@app.get("/")
async def root():
    return {"message": "Lip-Sync API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
