"""Celery task worker for processing individual video chunks in parallel.

Handles task lifecycle, Redis job status tracking, dynamic GPU VRAM allocation,
neural pipeline orchestration, failure retries, and stitch task dispatching.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from loguru import logger
import redis

from pipeline.orchestrator import ChunkPipelineOrchestrator
from workers.celery_app import celery_app, settings

# ============================================================================
# Dynamic Helper Imports & Fallbacks
# ============================================================================

def _select_optimal_gpu() -> int:
    """Determine the optimal GPU device based on current free VRAM."""
    try:
        from core.vram import select_optimal_device  # type: ignore
        return select_optimal_device()
    except (ImportError, AttributeError):
        pass

    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count <= 1:
                return 0
            best_device = 0
            max_free_bytes = 0
            for idx in range(device_count):
                free_bytes, _ = torch.cuda.mem_get_info(idx)
                if free_bytes > max_free_bytes:
                    max_free_bytes = free_bytes
                    best_device = idx
            return best_device
    except Exception as exc:
        logger.warning(f"Failed to query torch CUDA memory: {exc}")

    return 0


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
    name="workers.chunk_worker.process_chunk",
    max_retries=2,
    default_retry_delay=30,
)
def process_chunk(
    self: Any,
    job_id: str,
    chunk_index: int,
    chunk_path: str,
    reference_paths: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute neural face swap and lip sync on a single video chunk.

    1. Log task start with self.request.id, job_id, chunk_index
    2. Update chunk status to PROCESSING via sync Redis call
    3. Determine optimal GPU device via vram.select_optimal_device()
    4. Create ChunkPipelineOrchestrator with selected device
    5. Compute output path
    6. Call orchestrator.process_chunk()
    7. Update chunk status to COMPLETED
    8. Increment completed count
    9. Check if all chunks complete -> if so, dispatch stitch_job task
    10. Return result dict with output_path, duration, etc.

    Args:
        self: Bound Celery task context.
        job_id: ID of the overarching video swap job.
        chunk_index: Sequential index of the chunk.
        chunk_path: Path to the input video chunk file.
        reference_paths: Paths to target identity reference images.
        config: Optional runtime hyperparameter configuration.

    Returns:
        Result summary dict including output_path, chunk_index, and processing duration.

    Raises:
        Exception: Re-raises on unrecoverable failure or triggers Celery retry.
    """
    task_id = self.request.id or f"direct-{time.time()}"
    active_config = config or {}
    t_start = time.perf_counter()

    # 1. Log task start with self.request.id, job_id, chunk_index
    logger.info(
        f"[{job_id}] Task {task_id} started for Chunk {chunk_index:04d} (file={Path(chunk_path).name})"
    )

    r = _get_redis_client()

    try:
        # 2. Update chunk status to PROCESSING via sync Redis call
        try:
            r.set(f"job:{job_id}:chunk:{chunk_index}:status", "PROCESSING")
            r.hset(f"job:{job_id}:chunk_status", str(chunk_index), "PROCESSING")
            r.hset(f"job:{job_id}:chunk_tasks", str(chunk_index), str(task_id))
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Redis status update to PROCESSING failed: {redis_exc}")

        # 3. Determine optimal GPU device via vram.select_optimal_device()
        device_id = _select_optimal_gpu()
        logger.info(f"[{job_id}] Assigned Chunk {chunk_index} to GPU device {device_id}")

        # 4. Create ChunkPipelineOrchestrator with selected device
        orchestrator = ChunkPipelineOrchestrator(
            device_id=device_id,
            config=active_config,
        )

        # 5. Compute output path
        chunk_file = Path(chunk_path)
        output_dir = chunk_file.parent / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"processed_{chunk_file.name}"

        # 6. Call orchestrator.process_chunk()
        ref_image_paths = [Path(p) for p in reference_paths]
        final_chunk_path = orchestrator.process_chunk(
            chunk_path=chunk_file,
            reference_images=ref_image_paths,
            output_path=output_path,
            job_id=job_id,
            chunk_index=chunk_index,
            config=active_config,
        )

        # 7. Update chunk status to COMPLETED
        try:
            r.set(f"job:{job_id}:chunk:{chunk_index}:status", "COMPLETED")
            r.hset(f"job:{job_id}:chunk_status", str(chunk_index), "COMPLETED")
            r.hset(f"job:{job_id}:chunk_outputs", str(chunk_index), str(final_chunk_path.resolve()))
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Redis status update to COMPLETED failed: {redis_exc}")

        # 8. Increment completed count
        completed_count = 0
        try:
            completed_count = int(r.incr(f"job:{job_id}:completed_count"))
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Redis increment completed_count failed: {redis_exc}")

        # 9. Check if all chunks complete -> if so, dispatch stitch_job task
        total_chunks = active_config.get("total_chunks")
        if total_chunks is None:
            try:
                val = r.get(f"job:{job_id}:total_chunks")
                total_chunks = int(val) if val is not None else None
            except Exception:
                pass

        if total_chunks is not None and completed_count >= int(total_chunks):
            logger.info(
                f"[{job_id}] All {total_chunks} chunks completed! Dispatching stitch_job task..."
            )
            celery_app.send_task(
                "workers.stitch_worker.stitch_job",
                kwargs={
                    "job_id": job_id,
                    "chunk_count": int(total_chunks),
                    "config": active_config,
                },
                queue="stitch",
            )

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[{job_id}] Chunk {chunk_index} processing finished successfully in {elapsed:.2f}s"
        )

        # 10. Return result dict with output_path, duration, etc.
        return {
            "status": "COMPLETED",
            "job_id": job_id,
            "chunk_index": chunk_index,
            "output_path": str(final_chunk_path.resolve()),
            "duration": elapsed,
            "device_id": device_id,
            "task_id": task_id,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        logger.error(f"[{job_id}] Error processing Chunk {chunk_index} after {elapsed:.2f}s: {exc}")

        # Update chunk status to FAILED in Redis
        try:
            r.set(f"job:{job_id}:chunk:{chunk_index}:status", "FAILED")
            r.hset(f"job:{job_id}:chunk_status", str(chunk_index), "FAILED")
            r.incr(f"job:{job_id}:failed_count")
        except Exception as redis_exc:
            logger.warning(f"[{job_id}] Redis status update to FAILED failed: {redis_exc}")

        # Check retry limit
        if self.request.retries < self.max_retries:
            logger.warning(
                f"[{job_id}] Retrying Chunk {chunk_index} (attempt {self.request.retries + 1}/{self.max_retries})..."
            )
            raise self.retry(exc=exc)
        else:
            logger.critical(
                f"[{job_id}] Chunk {chunk_index} permanently failed after {self.max_retries} retries."
            )
            raise
