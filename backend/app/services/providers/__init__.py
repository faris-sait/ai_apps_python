"""
Lip Sync Providers Package

This package contains individual lip-sync provider implementations.
Each provider must implement the `run(video_path, audio_path) -> dict` interface.
"""

from app.services.providers.base import BaseLipSyncProvider
from app.services.providers.musetalk import MuseTalkProvider

__all__ = [
    "BaseLipSyncProvider",
    "MuseTalkProvider",
]
