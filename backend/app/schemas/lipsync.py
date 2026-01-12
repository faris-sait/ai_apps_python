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
    SADTALKER = "sadtalker"


class LipSyncMode(str, Enum):
    """Input mode for lip-sync generation"""
    IMAGE_AUDIO = "image_audio"  # Still image + audio -> lip-synced talking head video
    VIDEO_AUDIO = "video_audio"  # Video + audio -> lip-synced video


class LipSyncRequest(BaseModel):
    """Request schema for lip-sync generation"""
    provider: LipSyncProvider = Field(
        ...,
        description="The lip-sync provider to use",
        examples=["musetalk"]
    )
    mode: LipSyncMode = Field(
        ...,
        description="Input mode: 'image_audio' for image+audio or 'video_audio' for video+audio",
        examples=["video_audio"]
    )
    video_path: str | None = Field(
        default=None,
        description="Path to the input video file (required for video_audio mode)",
        examples=["/path/to/video.mp4"]
    )
    image_path: str | None = Field(
        default=None,
        description="Path to the input image file (required for image_audio mode)",
        examples=["/path/to/image.jpg"]
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
