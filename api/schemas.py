"""
Pydantic schemas and enums for the video swap pipeline API.

Defines all request models, response models, configuration objects,
and status enumerations for jobs, chunks, and references.
"""

from __future__ import annotations

from enum import Enum
import json
from typing import Any, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle status of a video swap job."""
    PENDING = "PENDING"
    CHUNKING = "CHUNKING"
    PROCESSING = "PROCESSING"
    STITCHING = "STITCHING"
    LIP_SYNCING = "LIP_SYNCING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ChunkStatus(str, Enum):
    """Processing status of an individual video chunk."""
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class VideoUploadResponse(BaseModel):
    """Response returned upon successful video upload and job initiation."""
    job_id: str = Field(..., description="Unique UUID identifier for the created job")
    message: str = Field(
        default="Video uploaded successfully and queued for processing",
        description="Informational status message",
    )
    total_chunks: int = Field(..., description="Total number of chunks the video was split into")
    estimated_duration_seconds: float = Field(
        ..., description="Total estimated video duration in seconds"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "message": "Video uploaded successfully and queued for processing",
                "total_chunks": 6,
                "estimated_duration_seconds": 36.5,
            }
        }
    }


class ChunkUploadResponse(BaseModel):
    """Response returned when an individual video chunk is uploaded."""
    job_id: str = Field(..., description="Job identifier this chunk belongs to")
    chunk_index: int = Field(..., description="0-based sequential index of the chunk")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    status: ChunkStatus = Field(default=ChunkStatus.UPLOADED, description="Current chunk status")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "chunk_index": 0,
                "chunk_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890_chunk_0000",
                "status": "UPLOADED",
            }
        }
    }


class ChunkProgress(BaseModel):
    """Progress report for a single chunk within a processing job."""
    chunk_index: int = Field(..., description="0-based sequential index of the chunk")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    status: ChunkStatus = Field(..., description="Current status of the chunk")
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Chunk processing completion percentage"
    )
    error_message: Optional[str] = Field(
        default=None, description="Detailed error description if chunk processing failed"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "chunk_index": 0,
                "chunk_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890_chunk_0000",
                "status": "COMPLETED",
                "progress_percent": 100.0,
                "error_message": None,
            }
        }
    }


class JobStatusResponse(BaseModel):
    """Comprehensive status response for a video swap job including per-chunk states."""
    job_id: str = Field(..., description="Unique UUID identifier for the job")
    status: JobStatus = Field(..., description="Current overarching lifecycle status of the job")
    total_chunks: int = Field(..., description="Total number of chunks in this job")
    completed_chunks: int = Field(default=0, description="Number of successfully processed chunks")
    failed_chunks: int = Field(default=0, description="Number of chunks that failed processing")
    chunks: list[ChunkProgress] = Field(
        default_factory=list, description="List of progress objects for all chunks"
    )
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall job progress percentage (0-100)"
    )
    output_url: Optional[str] = Field(
        default=None, description="Download URL or file path for final stitched output video"
    )
    error_message: Optional[str] = Field(
        default=None, description="Overall error message if job failed"
    )
    created_at: str = Field(..., description="ISO 8601 UTC timestamp of job creation")
    updated_at: str = Field(..., description="ISO 8601 UTC timestamp of last status update")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "status": "PROCESSING",
                "total_chunks": 4,
                "completed_chunks": 2,
                "failed_chunks": 0,
                "chunks": [
                    {
                        "chunk_index": 0,
                        "chunk_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890_chunk_0000",
                        "status": "COMPLETED",
                        "progress_percent": 100.0,
                        "error_message": None,
                    },
                    {
                        "chunk_index": 1,
                        "chunk_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890_chunk_0001",
                        "status": "PROCESSING",
                        "progress_percent": 45.0,
                        "error_message": None,
                    },
                ],
                "progress_percent": 36.25,
                "output_url": None,
                "error_message": None,
                "created_at": "2026-11-09T18:00:00Z",
                "updated_at": "2026-11-09T18:05:30Z",
            }
        }
    }


class ReferenceUploadResponse(BaseModel):
    """Response returned upon uploading character reference image(s)."""
    job_id: str = Field(..., description="Job identifier")
    reference_id: str = Field(..., description="Identifier or saved filename for the reference image")
    message: str = Field(
        default="Reference character image uploaded successfully",
        description="Status message",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "reference_id": "ref_0000.png",
                "message": "Reference character image uploaded successfully",
            }
        }
    }


class ProcessingConfig(BaseModel):
    """Configuration options for the neural video swap pipeline."""
    inference_steps: int = Field(
        default=30, ge=1, le=150, description="Number of diffusion denoising inference steps"
    )
    guidance_scale: float = Field(
        default=7.5, ge=1.0, le=30.0, description="Classifier-Free Guidance (CFG) scale"
    )
    controlnet_scale: float = Field(
        default=0.8, ge=0.0, le=2.0, description="Conditioning conditioning strength for ControlNet"
    )
    ip_adapter_scale: float = Field(
        default=0.6, ge=0.0, le=2.0, description="Conditioning strength for IP-Adapter identity"
    )
    enable_lip_sync: bool = Field(
        default=True, description="Whether to apply audio-driven lip synchronization post-swap"
    )
    enable_temporal_consistency: bool = Field(
        default=True,
        description="Whether to enforce cross-frame optical flow / temporal consistency smoothing",
    )
    target_resolution: Optional[int] = Field(
        default=None,
        ge=128,
        le=2160,
        description="Target vertical/horizontal resolution (e.g. 512, 720, 1080). None preserves original.",
    )

    @classmethod
    def parse_from_payload(cls, data: Any) -> "ProcessingConfig":
        """
        Safely instantiate ProcessingConfig from either a dict, JSON string, or instance.
        """
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            trimmed = data.strip()
            if not trimmed:
                return cls()
            parsed = json.loads(trimmed)
            if hasattr(cls, "model_validate"):
                return cls.model_validate(parsed)
            return cls.parse_obj(parsed)
        if isinstance(data, dict):
            if hasattr(cls, "model_validate"):
                return cls.model_validate(data)
            return cls.parse_obj(data)
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary across Pydantic v1/v2."""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    def to_json(self) -> str:
        """Convert config to JSON string across Pydantic v1/v2."""
        if hasattr(self, "model_dump_json"):
            return self.model_dump_json()
        return self.json()

    model_config = {
        "json_schema_extra": {
            "example": {
                "inference_steps": 30,
                "guidance_scale": 7.5,
                "controlnet_scale": 0.8,
                "ip_adapter_scale": 0.6,
                "enable_lip_sync": True,
                "enable_temporal_consistency": True,
                "target_resolution": None,
            }
        }
    }
