"""
Lip Sync Provider Factory

This module provides the factory pattern for selecting lip-sync providers.
"""

from typing import Any

from app.schemas.lipsync import LipSyncProvider, LipSyncRequest, LipSyncResponse
from app.services.providers.base import BaseLipSyncProvider
from app.services.providers.musetalk import MuseTalkProvider


class UnsupportedProviderError(Exception):
    """Raised when an unsupported provider is requested."""
    pass


def get_provider(provider: LipSyncProvider) -> BaseLipSyncProvider:
    """
    Factory function to get the appropriate lip-sync provider.
    
    Args:
        provider: The provider enum value
        
    Returns:
        An instance of the requested provider
        
    Raises:
        UnsupportedProviderError: If the provider is not supported
    """
    match provider:
        case LipSyncProvider.MUSETALK:
            return MuseTalkProvider()
        
        case LipSyncProvider.WAV2LIP:
            raise UnsupportedProviderError(
                f"Provider '{provider.value}' is not yet implemented. "
                "Available providers: musetalk"
            )
        
        case LipSyncProvider.SADTALKER:
            raise UnsupportedProviderError(
                f"Provider '{provider.value}' is not yet implemented. "
                "Available providers: musetalk"
            )
        
        case LipSyncProvider.VIDEO_RETALKING:
            raise UnsupportedProviderError(
                f"Provider '{provider.value}' is not yet implemented. "
                "Available providers: musetalk"
            )
        
        case LipSyncProvider.LATENTSYNC:
            raise UnsupportedProviderError(
                f"Provider '{provider.value}' is not yet implemented. "
                "Available providers: musetalk"
            )
        
        case LipSyncProvider.HALLO:
            raise UnsupportedProviderError(
                f"Provider '{provider.value}' is not yet implemented. "
                "Available providers: musetalk"
            )
        
        case _:
            raise UnsupportedProviderError(
                f"Unknown provider: '{provider}'. "
                "Available providers: musetalk, wav2lip, sadtalker, video_retalking, latentsync, hallo"
            )


def run_lipsync(request: LipSyncRequest) -> LipSyncResponse:
    """
    Run lip-sync generation using the specified provider.
    
    Args:
        request: The lip-sync request containing provider and file paths
        
    Returns:
        LipSyncResponse with the result
    """
    try:
        # Get the provider instance
        provider = get_provider(request.provider)
        
        # Run the lip-sync
        result = provider.run(
            video_path=request.video_path,
            audio_path=request.audio_path,
            output_path=request.output_path,
            options=request.options
        )
        
        return LipSyncResponse(
            success=result.get("success", False),
            provider=request.provider.value,
            output_path=result.get("output_path"),
            message=result.get("message"),
            metadata=result.get("metadata")
        )
        
    except UnsupportedProviderError as e:
        return LipSyncResponse(
            success=False,
            provider=request.provider.value,
            output_path=None,
            message=str(e),
            metadata=None
        )
    except Exception as e:
        return LipSyncResponse(
            success=False,
            provider=request.provider.value,
            output_path=None,
            message=f"Unexpected error: {str(e)}",
            metadata={"error_type": type(e).__name__}
        )
