"""
Lip Sync Request/Response Schemas

This module contains Pydantic models for the lip-sync API.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LipSyncProvider(str, Enum):
    """Supported lip-sync providers"""
    MUSETALK = "musetalk"
    WAV2LIP = "wav2lip"
    SADTALKER = "sadtalker"
    VIDEO_RETALKING = "video_retalking"
    LATENTSYNC = "latentsync"
    HALLO = "hallo"


class LipSyncRequest(BaseModel):
    """Request schema for lip-sync generation"""
    provider: LipSyncProvider = Field(
        ...,
        description="The lip-sync provider to use",
        examples=["musetalk"]
    )
    video_path: str = Field(
        ...,
        description="Path to the input video or image file",
        examples=["/path/to/video.mp4"]
    )
    audio_path: str = Field(
        ...,
        description="Path to the input audio file",
        examples=["/path/to/audio.wav"]
    )
    output_path: str | None = Field(
        default=None,
        description="Optional output path. If not provided, will be auto-generated",
        examples=["/path/to/output.mp4"]
    )
    options: dict[str, Any] | None = Field(
        default=None,
        description="Provider-specific options",
        examples=[{"quality": "high", "fps": 25}]
    )


class LipSyncResponse(BaseModel):
    """Response schema for lip-sync generation"""
    success: bool = Field(
        ...,
        description="Whether the operation was successful"
    )
    provider: str = Field(
        ...,
        description="The provider that was used"
    )
    output_path: str | None = Field(
        default=None,
        description="Path to the generated output video"
    )
    message: str | None = Field(
        default=None,
        description="Additional message or error details"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata from the provider"
    )


class LipSyncErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str = Field(
        ...,
        description="Error message"
    )
    provider: str | None = Field(
        default=None,
        description="The provider that was requested"
    )
