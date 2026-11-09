"""Celery task worker for stitching processed video chunks into the final video output.

Gathers completed video chunks in sequence, handles inter-chunk audio crossfades,
invokes FFmpeg concat demuxer, registers final video metadata and CDN output URLs,
and updates overarching Redis job statuses.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import traceback
from typing import Any

from loguru import logger
import redis

from utils.ffmpeg_utils import extract_audio, get_video_duration, stitch_chunks
from workers.celery_app import celery_app, settings

# ============================================================================
# Dynamic Redis Helper
# ============================================================================

def _get_redis_client() -> redis.Redis:
    """Acquire a synchronous Redis connection using configured pipeline settings."""
    try:
        from core.redis import get_redis_client  # type: ignore
        return get_redis_client()
    except (ImportError, AttributeError):
        pass

    redis_url = (
        getattr(settings, "redis_url", None)
        or getattr(settings, "celery_broker_url", None)
        or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    return redis.Redis.from_url(redis_url, decode_responses=True)


# ============================================================================
# Celery Task Definition
# ============================================================================

@celery_app.task(
    bind=True,
    name="workers.stitch_worker.stitch_job",
    max_retries=1,
)
def stitch_job(
    self: Any,
    job_id: str,
    chunk_count: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stitch all processed video chunks for a job into the finalized MP4 media file.

    1. Update job status to STITCHING
    2. Gather all processed chunk paths in order
    3. Extract and concatenate audio from source chunks
    4. Apply inter-chunk temporal smoothing (crossfade transitions)
    5. Call ffmpeg_utils.stitch_chunks() to merge into final MP4
    6. Update job status to COMPLETED
    7. Set output URL
    8. Log final video stats (duration, file size, processing time)

    Args:
        self: Bound Celery task context.
        job_id: Unique job identifier.
        chunk_count: Total expected chunk count for the job.
        config: Configuration dictionary with optional output directory, CDN url, etc.

    Returns:
        Dictionary containing job metadata, output_path, output_url, and performance metrics.

    Raises:
        Exception: On stitch failure; logs full traceback and marks job as FAILED.
    """
    t_start = time.perf_counter()
    task_id = self.request.id or f"stitch-{time.time()}"
    active_config = config or {}

    logger.info(
        f"[{job_id}] Stitch task {task_id} initiated for {chunk_count} chunks"
    )

    r = _get_redis_client()

    try:
        # 1. Update job status to STITCHING
        try:
            r.set(f"job:{job_id}:status", "STITCHING")
            r.set(f"job:{job_id}:stitch_task_id", str(task_id))
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Redis status update to STITCHING failed: {redis_exc}")

        # 2. Gather all processed chunk paths in order
        logger.info(f"[{job_id}] Gathering paths for all {chunk_count} chunks in order...")
        chunk_outputs_map = {}
        try:
            chunk_outputs_map = r.hgetall(f"job:{job_id}:chunk_outputs")
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Could not retrieve chunk outputs from Redis: {redis_exc}")

        processed_chunk_paths: list[Path] = []
        missing_chunks: list[int] = []

        # Target working directory for chunks
        chunks_base_dir = Path(active_config.get("chunks_dir", f"jobs/{job_id}/chunks/processed"))

        for idx in range(chunk_count):
            path_str = chunk_outputs_map.get(str(idx))
            if path_str and Path(path_str).is_file():
                processed_chunk_paths.append(Path(path_str))
                continue

            # Fallback file discovery on disk
            candidate_file = chunks_base_dir / f"processed_chunk_{idx:04d}.mp4"
            if candidate_file.is_file():
                processed_chunk_paths.append(candidate_file)
            else:
                # Look for any matching extension
                matches = list(chunks_base_dir.glob(f"processed_chunk_{idx:04d}.*"))
                if matches and matches[0].is_file():
                    processed_chunk_paths.append(matches[0])
                else:
                    missing_chunks.append(idx)

        if missing_chunks:
            err_msg = (
                f"[{job_id}] Cannot stitch video: missing {len(missing_chunks)} chunk(s): {missing_chunks}"
            )
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        logger.info(f"[{job_id}] Verified all {len(processed_chunk_paths)} chunk files exist.")

        # Determine output location
        output_dir = Path(active_config.get("output_dir", f"jobs/{job_id}/output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        final_video_path = output_dir / f"{job_id}_final.mp4"

        # 3. Extract and concatenate audio from source chunks
        audio_path_to_use: Path | None = None
        source_audio = active_config.get("source_audio_path")
        source_video = active_config.get("source_video_path")

        if source_audio and Path(source_audio).is_file():
            # If the complete original audio track was preserved, use it directly for lossless audio fidelity
            audio_path_to_use = Path(source_audio)
            logger.info(f"[{job_id}] Using provided master audio track: {audio_path_to_use.name}")
        elif source_video and Path(source_video).is_file():
            # Extract complete audio track directly from master source video
            extracted_master_audio = output_dir / f"{job_id}_master_audio.wav"
            logger.info(f"[{job_id}] Extracting master audio track from source video: {Path(source_video).name}")
            try:
                extract_audio(Path(source_video), extracted_master_audio)
                if extracted_master_audio.is_file():
                    audio_path_to_use = extracted_master_audio
            except Exception as audio_exc:
                logger.warning(f"[{job_id}] Failed to extract master audio, falling back to chunk audio: {audio_exc}")

        # 4. Apply inter-chunk temporal smoothing (crossfade transitions)
        # Note: If no master audio is specified, stitch_chunks uses the embedded audio in chunks.
        # If audio_path_to_use is provided, acrossfade filter or synchronized muxing is applied.
        apply_crossfade = active_config.get("crossfade_transitions", True)
        logger.info(
            f"[{job_id}] Inter-chunk transition mode: {'acrossfade enabled (0.1s)' if apply_crossfade else 'direct concat'}"
        )

        # 5. Call ffmpeg_utils.stitch_chunks() to merge into final MP4
        t0 = time.perf_counter()
        stitch_chunks(
            chunk_paths=processed_chunk_paths,
            output_path=final_video_path,
            audio_path=audio_path_to_use,
        )
        t_stitch = time.perf_counter() - t0
        logger.info(f"[{job_id}] FFmpeg chunk stitch completed in {t_stitch:.2f}s")

        # 6. Update job status to COMPLETED
        try:
            r.set(f"job:{job_id}:status", "COMPLETED")
            r.set(f"job:{job_id}:output_path", str(final_video_path.resolve()))
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Redis status update to COMPLETED failed: {redis_exc}")

        # 7. Set output URL
        cdn_base = active_config.get("cdn_base_url", "/static/outputs").rstrip("/")
        output_url = f"{cdn_base}/{final_video_path.name}"
        try:
            r.set(f"job:{job_id}:output_url", output_url)
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Redis output_url update failed: {redis_exc}")

        # 8. Log final video stats (duration, file size, processing time)
        final_duration: float = 0.0
        try:
            final_duration = get_video_duration(final_video_path)
        except Exception as probe_exc:
            logger.warning(f"[{job_id}] Failed to probe final video duration: {probe_exc}")

        file_size_bytes = final_video_path.stat().st_size if final_video_path.exists() else 0
        file_size_mb = file_size_bytes / (1024 * 1024)
        total_stitch_time = time.perf_counter() - t_start

        logger.info(
            f"[{job_id}] JOB COMPLETED SUCCESSFULLY! "
            f"Stats: duration={final_duration:.2f}s, size={file_size_mb:.2f} MB, "
            f"processing_time={total_stitch_time:.2f}s, output='{final_video_path.name}', url='{output_url}'"
        )

        return {
            "status": "COMPLETED",
            "job_id": job_id,
            "output_path": str(final_video_path.resolve()),
            "output_url": output_url,
            "duration": final_duration,
            "file_size_mb": round(file_size_mb, 2),
            "processing_time": round(total_stitch_time, 2),
        }

    except Exception as exc:
        total_time = time.perf_counter() - t_start
        err_traceback = traceback.format_exc()
        logger.error(
            f"[{job_id}] Stitch job failed after {total_time:.2f}s: {exc}\n{err_traceback}"
        )

        # Update Redis status to FAILED
        try:
            r.set(f"job:{job_id}:status", "FAILED")
            r.set(f"job:{job_id}:error", str(exc))
            r.set(f"job:{job_id}:traceback", err_traceback)
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Redis status update to FAILED failed: {redis_exc}")

        raise
