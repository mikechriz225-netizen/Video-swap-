"""
FastAPI route handlers for the video swap pipeline.

Exposes endpoints for creating video swap jobs, uploading chunks, uploading
reference character identities, tracking job progress, downloading processed
chunks and stitched outputs, and cancelling active jobs.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from api.schemas import (
    ChunkProgress,
    ChunkStatus,
    ChunkUploadResponse,
    JobStatus,
    JobStatusResponse,
    ProcessingConfig,
    ReferenceUploadResponse,
    VideoUploadResponse,
)
from core.chunk_manager import ChunkManager
from core.job_manager import JobManager
from core.storage import StorageManager

# Optional imports for Celery task dispatch and FFmpeg utility helpers
try:
    from workers.tasks import process_chunk_task
except ImportError:
    process_chunk_task = None  # type: ignore[assignment]

try:
    from celery import current_app as celery_app
except ImportError:
    celery_app = None  # type: ignore[assignment]

try:
    import utils.ffmpeg as ffmpeg_utils
except ImportError:
    ffmpeg_utils = None  # type: ignore[assignment]

# Maximum allowed source video duration in seconds (10 minutes default)
MAX_VIDEO_DURATION: float = float(os.getenv("MAX_VIDEO_DURATION", "600.0"))

router = APIRouter(tags=["Video Swap"])

# -------------------------------------------------------------------------
# Dependency Injection Singletons / Factories
# -------------------------------------------------------------------------
_storage_manager: Optional[StorageManager] = None
_job_manager: Optional[JobManager] = None


def get_storage_manager() -> StorageManager:
    """Provide a singleton instance of StorageManager."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager


def get_job_manager() -> JobManager:
    """Provide a singleton instance of JobManager."""
    global _job_manager
    if _job_manager is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _job_manager = JobManager(redis_url=redis_url)
    return _job_manager


def get_chunk_manager(
    storage: StorageManager = Depends(get_storage_manager),
    job_manager: JobManager = Depends(get_job_manager),
) -> ChunkManager:
    """Provide a ChunkManager instance wired with storage and job manager dependencies."""
    return ChunkManager(storage=storage, job_manager=job_manager)


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@router.post(
    "/jobs/create",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new video swap job",
    description="Upload source video and reference character image to create and launch a new swap job.",
)
@router.post("/create", response_model=VideoUploadResponse, include_in_schema=False)
async def create_job(
    background_tasks: BackgroundTasks,
    source_video: UploadFile = File(..., description="Original video file to be swapped"),
    reference_image: UploadFile = File(..., description="Target character identity reference image"),
    config: Optional[str] = Form(None, description="JSON string representation of ProcessingConfig"),
    storage: StorageManager = Depends(get_storage_manager),
    job_manager: JobManager = Depends(get_job_manager),
    chunk_manager: ChunkManager = Depends(get_chunk_manager),
) -> VideoUploadResponse:
    """
    Handle initial video and reference upload, validate duration, chunk video,
    register state in Redis, and dispatch Celery tasks.
    """
    job_id = str(uuid.uuid4())
    logger.info(
        "Received job creation request: job_id={} source_file={} ref_file={}",
        job_id,
        source_video.filename,
        reference_image.filename,
    )

    # 1. Parse ProcessingConfig
    try:
        proc_config = ProcessingConfig.parse_from_payload(config or "{}")
    except Exception as exc:
        logger.error("Failed to parse ProcessingConfig JSON for job {}: {}", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration payload: {exc}",
        )

    # 2. Save source video
    source_ext = Path(source_video.filename or "video.mp4").suffix or ".mp4"
    source_dest = storage.get_job_dir(job_id) / f"source{source_ext}"
    try:
        await storage.save_upload(source_video, source_dest)
        logger.debug("Source video saved to {}", source_dest)
    except Exception as exc:
        logger.error("Failed saving source video for job {}: {}", job_id, exc)
        storage.cleanup_job(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded video file.",
        )

    # 3. Probe video duration and reject if > MAX_VIDEO_DURATION
    try:
        if ffmpeg_utils and hasattr(ffmpeg_utils, "probe_duration"):
            duration = await ffmpeg_utils.probe_duration(source_dest)
        else:
            duration = await chunk_manager.probe_duration(source_dest)
    except Exception as exc:
        logger.error("Duration probe failed for job {}: {}", job_id, exc)
        storage.cleanup_job(job_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to probe video file duration: {exc}",
        )

    if duration > MAX_VIDEO_DURATION:
        logger.warning(
            "Video duration {:.1f}s exceeds maximum limit {:.1f}s for job {}",
            duration,
            MAX_VIDEO_DURATION,
            job_id,
        )
        storage.cleanup_job(job_id)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video duration ({duration:.1f}s) exceeds maximum allowed duration ({MAX_VIDEO_DURATION:.1f}s).",
        )

    # 4. Save reference character image
    ref_ext = Path(reference_image.filename or "reference.png").suffix or ".png"
    ref_dest = storage.get_reference_path(job_id, ref_index=0, suffix=ref_ext)
    try:
        await storage.save_upload(reference_image, ref_dest)
        logger.debug("Reference image saved to {}", ref_dest)
    except Exception as exc:
        logger.error("Failed saving reference image for job {}: {}", job_id, exc)
        storage.cleanup_job(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded reference character image.",
        )

    # 5. Chunk the video at keyframes
    try:
        if ffmpeg_utils and hasattr(ffmpeg_utils, "chunk_video_at_keyframes"):
            logger.info("Using utils.ffmpeg.chunk_video_at_keyframes for job {}", job_id)
            chunk_paths = await ffmpeg_utils.chunk_video_at_keyframes(
                video_path=source_dest,
                output_dir=storage.get_job_dir(job_id),
            )
        else:
            logger.info("Using chunk_manager.split_video for job {}", job_id)
            chunk_paths = await chunk_manager.split_video(
                job_id=job_id,
                video_path=source_dest,
                min_chunk_sec=4.0,
                max_chunk_sec=10.0,
            )
    except Exception as exc:
        logger.exception("Video chunking failed for job {}: {}", job_id, exc)
        storage.cleanup_job(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video chunking failed: {exc}",
        )

    total_chunks = len(chunk_paths)
    if total_chunks == 0:
        logger.error("No chunks were generated for job {}", job_id)
        storage.cleanup_job(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Video chunking produced 0 chunks.",
        )

    # 6. Initialize Job State in Redis
    try:
        await job_manager.create_job(job_id=job_id, total_chunks=total_chunks, config=proc_config)
        await job_manager.update_job_status(job_id, JobStatus.PROCESSING)
    except Exception as exc:
        logger.error("Failed to initialize job state in Redis for {}: {}", job_id, exc)
        storage.cleanup_job(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize job state tracking.",
        )

    # 7. Dispatch Celery tasks for each chunk
    optimal_order = await chunk_manager.get_processing_order(job_id)
    config_dict = proc_config.to_dict()

    for chunk_idx in optimal_order:
        if process_chunk_task is not None:
            try:
                task = process_chunk_task.delay(
                    job_id=job_id,
                    chunk_index=chunk_idx,
                    config=config_dict,
                )
                await job_manager.add_celery_task_id(job_id, task.id)
                logger.debug(
                    "Dispatched Celery task {} for job {} chunk {}",
                    task.id,
                    job_id,
                    chunk_idx,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to dispatch Celery task for job {} chunk {}: {}",
                    job_id,
                    chunk_idx,
                    exc,
                )
        else:
            logger.debug(
                "Celery task module not loaded; chunk {} recorded as queued",
                chunk_idx,
            )

    logger.info(
        "Job {} created successfully with {} chunks ({:.2f}s duration)",
        job_id,
        total_chunks,
        duration,
    )

    return VideoUploadResponse(
        job_id=job_id,
        message="Video uploaded, chunked at keyframes, and queued for processing",
        total_chunks=total_chunks,
        estimated_duration_seconds=round(duration, 2),
    )


@router.post(
    "/jobs/{job_id}/chunks/{chunk_index}",
    response_model=ChunkUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload an individual video chunk",
    description="Enables client-side chunked uploading. Saves chunk, updates state, and dispatches processing.",
)
@router.post("/chunks/{chunk_index}", response_model=ChunkUploadResponse, include_in_schema=False)
async def upload_individual_chunk(
    job_id: str,
    chunk_index: int,
    chunk_file: UploadFile = File(..., description="Individual chunk video file"),
    storage: StorageManager = Depends(get_storage_manager),
    job_manager: JobManager = Depends(get_job_manager),
) -> ChunkUploadResponse:
    """
    Accept an individual client-uploaded chunk, save it to storage, and queue processing.
    """
    logger.info("Received chunk upload: job_id={} chunk_index={}", job_id, chunk_index)

    job = await job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    suffix = Path(chunk_file.filename or ".mp4").suffix or ".mp4"
    dest_path = storage.get_chunk_path(job_id, chunk_index, suffix=suffix)

    try:
        await storage.save_upload(chunk_file, dest_path)
    except Exception as exc:
        logger.error("Failed saving chunk {} for job {}: {}", chunk_index, job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save chunk file: {exc}",
        )

    # Update chunk status
    chunk_id = f"{job_id}_chunk_{chunk_index:04d}"
    await job_manager.update_chunk_status(
        job_id=job_id,
        chunk_index=chunk_index,
        status=ChunkStatus.UPLOADED,
        progress=0.0,
    )

    # Dispatch Celery processing task if available
    if process_chunk_task is not None:
        try:
            task = process_chunk_task.delay(
                job_id=job_id,
                chunk_index=chunk_index,
            )
            await job_manager.add_celery_task_id(job_id, task.id)
            await job_manager.update_chunk_status(
                job_id=job_id,
                chunk_index=chunk_index,
                status=ChunkStatus.QUEUED,
                progress=0.0,
            )
            logger.info("Queued processing task {} for chunk {}", task.id, chunk_id)
        except Exception as exc:
            logger.warning("Error dispatching task for chunk {}: {}", chunk_id, exc)

    return ChunkUploadResponse(
        job_id=job_id,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        status=ChunkStatus.QUEUED,
    )


@router.post(
    "/jobs/{job_id}/reference",
    response_model=ReferenceUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload character reference image(s)",
    description="Upload one or more reference character images for identity capture and facial representation.",
)
@router.post("/reference", response_model=ReferenceUploadResponse, include_in_schema=False)
async def upload_reference_images(
    job_id: str,
    files: list[UploadFile] = File(..., description="One or more target character images"),
    storage: StorageManager = Depends(get_storage_manager),
    job_manager: JobManager = Depends(get_job_manager),
) -> ReferenceUploadResponse:
    """
    Store additional or updated reference character images for an active or pending job.
    """
    logger.info("Uploading {} reference images for job_id={}", len(files), job_id)

    job = await job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No reference files provided.",
        )

    # Count existing reference images
    ref_dir = storage.get_job_dir(job_id) / "references"
    existing_count = len(list(ref_dir.glob("ref_*.*"))) if ref_dir.exists() else 0

    saved_names: list[str] = []
    for offset, file_item in enumerate(files):
        idx = existing_count + offset
        suffix = Path(file_item.filename or ".png").suffix or ".png"
        ref_path = storage.get_reference_path(job_id, ref_index=idx, suffix=suffix)

        try:
            await storage.save_upload(file_item, ref_path)
            saved_names.append(ref_path.name)
            logger.info("Saved reference image {} to {}", file_item.filename, ref_path)
        except Exception as exc:
            logger.error("Failed saving reference image {}: {}", file_item.filename, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed writing reference file {file_item.filename}: {exc}",
            )

    return ReferenceUploadResponse(
        job_id=job_id,
        reference_id=",".join(saved_names),
        message=f"Successfully uploaded {len(saved_names)} reference character image(s).",
    )


@router.get(
    "/jobs/{job_id}/status",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get job processing status",
    description="Retrieve full job status, overall progress percentage, and per-chunk details.",
)
@router.get("/{job_id}/status", response_model=JobStatusResponse, include_in_schema=False)
async def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobStatusResponse:
    """
    Fetch comprehensive progress information for a job from Redis.
    """
    logger.debug("Querying status for job_id={}", job_id)
    job = await job_manager.get_job(job_id)
    if job is None:
        logger.warning("Job {} not found in status query", job_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    return job


@router.get(
    "/jobs/{job_id}/download",
    response_class=FileResponse,
    summary="Download final stitched video",
    description="Download the final processed and stitched output video.",
)
@router.get("/{job_id}/download", response_class=FileResponse, include_in_schema=False)
async def download_output_video(
    job_id: str,
    storage: StorageManager = Depends(get_storage_manager),
    job_manager: JobManager = Depends(get_job_manager),
) -> FileResponse:
    """
    Stream the completed stitched output video file back to the client.
    """
    logger.info("Requested final video download for job_id={}", job_id)

    job = await job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    out_path = storage.get_output_path(job_id)

    # Check for any valid output video file format
    candidates = list(storage.output_dir.glob(f"{job_id}.*"))
    valid_file = None
    if out_path.exists() and out_path.is_file() and out_path.stat().st_size > 0:
        valid_file = out_path
    elif candidates:
        for c in candidates:
            if c.is_file() and c.stat().st_size > 0:
                valid_file = c
                break

    if valid_file is None:
        if job.status in {JobStatus.PENDING, JobStatus.CHUNKING, JobStatus.PROCESSING, JobStatus.STITCHING, JobStatus.LIP_SYNCING}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is still {job.status.value}. Output video is not ready yet.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Output video for job {job_id} was not found on disk.",
        )

    logger.info("Serving output video download: {}", valid_file)
    return FileResponse(
        path=str(valid_file),
        media_type="video/mp4",
        filename=f"video_swap_{job_id}{valid_file.suffix}",
    )


@router.get(
    "/jobs/{job_id}/chunks/{chunk_index}/download",
    response_class=FileResponse,
    summary="Download individual processed chunk",
    description="Download a specific processed chunk video file by index.",
)
@router.get("/chunks/{chunk_index}/download", response_class=FileResponse, include_in_schema=False)
async def download_processed_chunk(
    job_id: str,
    chunk_index: int,
    storage: StorageManager = Depends(get_storage_manager),
    job_manager: JobManager = Depends(get_job_manager),
) -> FileResponse:
    """
    Stream an individual processed chunk file.
    """
    logger.info("Requested chunk download: job_id={} chunk_index={}", job_id, chunk_index)

    chunk_path = storage.get_processed_chunk_path(job_id, chunk_index)
    if not chunk_path.exists() or chunk_path.stat().st_size == 0:
        # Check if job exists at all
        job = await job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processed chunk {chunk_index} for job {job_id} not found or still processing.",
        )

    return FileResponse(
        path=str(chunk_path),
        media_type="video/mp4",
        filename=f"chunk_{chunk_index:04d}_{job_id}{chunk_path.suffix}",
    )


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel job and clean up resources",
    description="Revokes active Celery tasks, deletes file artifacts from all storage tiers, and purges Redis state.",
)
@router.delete("/{job_id}", status_code=status.HTTP_200_OK, include_in_schema=False)
async def cancel_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    storage: StorageManager = Depends(get_storage_manager),
    job_manager: JobManager = Depends(get_job_manager),
) -> dict[str, Any]:
    """
    Cancel an in-progress or completed job, terminating tasks and cleaning disk & Redis.
    """
    logger.info("Received cancellation request for job_id={}", job_id)

    # 1. Revoke Celery tasks
    try:
        task_ids = await job_manager.get_celery_task_ids(job_id)
        if task_ids:
            logger.info("Revoking {} Celery tasks for job {}", len(task_ids), job_id)
            if celery_app is not None:
                for tid in task_ids:
                    try:
                        celery_app.control.revoke(tid, terminate=True, signal="SIGTERM")
                        logger.debug("Revoked task {}", tid)
                    except Exception as err:
                        logger.warning("Error revoking Celery task {}: {}", tid, err)
    except Exception as exc:
        logger.warning("Failed retrieving or revoking Celery tasks for {}: {}", job_id, exc)

    # 2. Mark status as FAILED in case queries arrive before deletion completes
    try:
        await job_manager.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message="Job cancelled by user request.",
        )
    except Exception as exc:
        logger.debug("Could not update status to FAILED for cancelled job {}: {}", job_id, exc)

    # 3. Schedule or execute storage cleanup
    background_tasks.add_task(storage.cleanup_job, job_id)

    # 4. Delete Redis state keys
    try:
        await job_manager.delete_job(job_id)
    except Exception as exc:
        logger.warning("Failed deleting Redis keys for job {}: {}", job_id, exc)

    logger.info("Job {} successfully cancelled and scheduled for cleanup", job_id)
    return {
        "job_id": job_id,
        "message": "Job cancelled, background tasks revoked, and storage cleanup initiated.",
    }
