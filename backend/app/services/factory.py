"""
Lip Sync Provider Factory

This module provides the factory pattern for selecting lip-sync providers.
"""

from typing import Any

from app.schemas.lipsync import LipSyncMode, LipSyncProvider, LipSyncRequest, LipSyncResponse
from app.services.providers.base import BaseLipSyncProvider
from app.services.providers.musetalk import MuseTalkProvider
from app.services.providers.sadtalker import SadTalkerProvider


class UnsupportedProviderError(Exception):
    """Raised when an unsupported provider is requested."""
    pass


class InvalidInputError(Exception):
    """Raised when input validation fails."""
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
        
        case LipSyncProvider.SADTALKER:
            return SadTalkerProvider()
        
        case _:
            raise UnsupportedProviderError(
                f"Unknown provider: '{provider}'. "
                "Available providers: musetalk, sadtalker"
            )


def validate_request(request: LipSyncRequest) -> tuple[str, str]:
    """
    Validate the request and return the input path based on mode.
    
    Args:
        request: The lip-sync request
        
    Returns:
        Tuple of (input_path, input_type) where input_type is 'image' or 'video'
        
    Raises:
        InvalidInputError: If required inputs are missing
    """
    if request.mode == LipSyncMode.IMAGE_AUDIO:
        if not request.image_path:
            raise InvalidInputError(
                "image_path is required when mode is 'image_audio'"
            )
        return request.image_path, "image"
    
    elif request.mode == LipSyncMode.VIDEO_AUDIO:
        if not request.video_path:
            raise InvalidInputError(
                "video_path is required when mode is 'video_audio'"
            )
        return request.video_path, "video"
    
    else:
        raise InvalidInputError(f"Unknown mode: {request.mode}")


def run_lipsync(request: LipSyncRequest) -> LipSyncResponse:
    """
    Run lip-sync generation using the specified provider.
    
    Args:
        request: The lip-sync request containing provider and file paths
        
    Returns:
        LipSyncResponse with the result
    """
    try:
        # Validate request and get input path
        input_path, input_type = validate_request(request)
        
        # Get the provider instance
        provider = get_provider(request.provider)
        
        # Add input_type to options for provider to handle appropriately
        options = request.options or {}
        options["input_type"] = input_type
        options["mode"] = request.mode.value
        
        # Run the lip-sync
        result = provider.run(
            video_path=input_path,  # Can be image or video based on mode
            audio_path=request.audio_path,
            output_path=request.output_path,
            options=options
        )
        
        return LipSyncResponse(
            success=result.get("success", False),
            provider=request.provider.value,
            output_path=result.get("output_path"),
            message=result.get("message"),
            metadata=result.get("metadata")
        )
    
    except InvalidInputError as e:
        return LipSyncResponse(
            success=False,
            provider=request.provider.value,
            output_path=None,
            message=str(e),
            metadata={"error_type": "validation_error"}
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
