"""
Redis-backed job and chunk state manager for the video swap pipeline.

Tracks full job lifecycle, chunk progress, Celery task associations, and output URLs
using async Redis with configurable key expiration (24h default TTL).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from api.schemas import (
    ChunkProgress,
    ChunkStatus,
    JobStatus,
    JobStatusResponse,
    ProcessingConfig,
)


class JobManager:
    """
    Asynchronous state store for video processing jobs and chunk lifecycles.

    Uses Redis hashes to maintain job metadata, per-chunk progress,
    and associated Celery task IDs with automatic 24-hour TTL expiration.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 86400,  # 24 hours
    ) -> None:
        """
        Initialize JobManager.

        Args:
            redis_url: Redis connection URL string.
            ttl_seconds: Time-To-Live in seconds for job keys (default: 86400s / 24h).
        """
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._redis: Optional[aioredis.Redis] = None
        logger.info("JobManager configured with redis_url={} ttl={}s", self.redis_url, self.ttl_seconds)

    async def _get_redis(self) -> aioredis.Redis:
        """
        Get or lazily initialize the async Redis connection client.

        Returns:
            aioredis.Redis client instance.
        """
        if self._redis is None:
            logger.debug("Initializing async Redis client connection to {}", self.redis_url)
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def close(self) -> None:
        """Close the underlying Redis connection pool gracefully."""
        if self._redis is not None:
            logger.debug("Closing Redis connection pool")
            await self._redis.aclose()
            self._redis = None

    def _job_key(self, job_id: str) -> str:
        """Key for overarching job metadata hash."""
        return f"job:{job_id}"

    def _chunks_key(self, job_id: str) -> str:
        """Key for per-chunk progress hash."""
        return f"job:{job_id}:chunks"

    def _get_utc_now(self) -> str:
        """Return current ISO 8601 UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    async def create_job(
        self,
        job_id: str,
        total_chunks: int,
        config: ProcessingConfig,
    ) -> None:
        """
        Create a new job state entry in Redis and pre-seed chunk records.

        Args:
            job_id: Unique job identifier.
            total_chunks: Total number of video chunks.
            config: ProcessingConfig model instance.
        """
        logger.info("Creating job state in Redis for job_id={} total_chunks={}", job_id, total_chunks)
        try:
            r = await self._get_redis()
            now = self._get_utc_now()
            config_json = config.to_json() if hasattr(config, "to_json") else json.dumps(config.dict())

            job_data = {
                "job_id": job_id,
                "status": JobStatus.PENDING.value,
                "total_chunks": str(total_chunks),
                "completed_chunks": "0",
                "failed_chunks": "0",
                "config": config_json,
                "created_at": now,
                "updated_at": now,
                "celery_task_ids": json.dumps([]),
                "output_url": "",
                "error_message": "",
            }

            job_key = self._job_key(job_id)
            chunks_key = self._chunks_key(job_id)

            async with r.pipeline(transaction=True) as pipe:
                pipe.hset(job_key, mapping=job_data)
                pipe.expire(job_key, self.ttl_seconds)

                # Pre-populate chunk records
                if total_chunks > 0:
                    chunk_map: dict[str, str] = {}
                    for i in range(total_chunks):
                        chunk_progress = {
                            "chunk_index": i,
                            "chunk_id": f"{job_id}_chunk_{i:04d}",
                            "status": ChunkStatus.QUEUED.value,
                            "progress_percent": 0.0,
                            "error_message": None,
                        }
                        chunk_map[str(i)] = json.dumps(chunk_progress)
                    pipe.hset(chunks_key, mapping=chunk_map)
                    pipe.expire(chunks_key, self.ttl_seconds)

                await pipe.execute()

            logger.info("Job {} successfully created in Redis with 24h TTL", job_id)

        except RedisError as exc:
            logger.error("Redis error while creating job {}: {}", job_id, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error while creating job {}: {}", job_id, exc)
            raise

    async def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Fetch overarching job status and per-chunk progress from Redis.

        Args:
            job_id: Unique job identifier.

        Returns:
            JobStatusResponse if found, None otherwise.
        """
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            job_data = await r.hgetall(job_key)

            if not job_data:
                logger.debug("Job {} not found in Redis", job_id)
                return None

            chunks = await self.get_chunk_statuses(job_id)
            total_chunks = int(job_data.get("total_chunks", 0))
            completed_chunks = int(job_data.get("completed_chunks", 0))
            failed_chunks = int(job_data.get("failed_chunks", 0))

            # Compute overall progress
            if chunks:
                avg_chunk_progress = sum(c.progress_percent for c in chunks) / len(chunks)
                progress_percent = round(min(100.0, max(0.0, avg_chunk_progress)), 2)
            elif total_chunks > 0:
                progress_percent = round((completed_chunks / total_chunks) * 100.0, 2)
            else:
                progress_percent = 0.0

            raw_status = job_data.get("status", JobStatus.PENDING.value)
            try:
                status = JobStatus(raw_status)
            except ValueError:
                status = JobStatus.PENDING

            output_url = job_data.get("output_url") or None
            error_message = job_data.get("error_message") or None

            return JobStatusResponse(
                job_id=job_id,
                status=status,
                total_chunks=total_chunks,
                completed_chunks=completed_chunks,
                failed_chunks=failed_chunks,
                chunks=chunks,
                progress_percent=progress_percent,
                output_url=output_url,
                error_message=error_message,
                created_at=job_data.get("created_at", self._get_utc_now()),
                updated_at=job_data.get("updated_at", self._get_utc_now()),
            )

        except RedisError as exc:
            logger.error("Redis error fetching job {}: {}", job_id, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error fetching job {}: {}", job_id, exc)
            raise

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update the overarching lifecycle status of a job.

        Args:
            job_id: Unique job identifier.
            status: New JobStatus enum.
            error_message: Optional failure explanation.
        """
        logger.info("Updating job_id={} status to {} (error={})", job_id, status.value, error_message)
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            now = self._get_utc_now()

            status_val = status.value if isinstance(status, JobStatus) else str(status)
            mapping: dict[str, str] = {
                "status": status_val,
                "updated_at": now,
            }
            if error_message is not None:
                mapping["error_message"] = error_message

            async with r.pipeline(transaction=True) as pipe:
                pipe.hset(job_key, mapping=mapping)
                pipe.expire(job_key, self.ttl_seconds)
                await pipe.execute()

        except RedisError as exc:
            logger.error("Redis error updating status for job {}: {}", job_id, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error updating status for job {}: {}", job_id, exc)
            raise

    async def update_chunk_status(
        self,
        job_id: str,
        chunk_index: int,
        status: ChunkStatus,
        progress: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """
        Update the status, progress percentage, and optional error for a single chunk.

        Args:
            job_id: Unique job identifier.
            chunk_index: 0-based chunk sequence index.
            status: New ChunkStatus enum.
            progress: Progress completion percentage (0.0 - 100.0).
            error: Optional error description string.
        """
        logger.debug(
            "Updating job_id={} chunk={} status={} progress={}% error={}",
            job_id,
            chunk_index,
            status.value,
            progress,
            error,
        )
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            chunks_key = self._chunks_key(job_id)

            status_val = status.value if isinstance(status, ChunkStatus) else str(status)
            clamped_progress = round(min(100.0, max(0.0, float(progress))), 2)

            chunk_payload = {
                "chunk_index": chunk_index,
                "chunk_id": f"{job_id}_chunk_{chunk_index:04d}",
                "status": status_val,
                "progress_percent": clamped_progress,
                "error_message": error,
            }

            now = self._get_utc_now()

            async with r.pipeline(transaction=True) as pipe:
                pipe.hset(chunks_key, str(chunk_index), json.dumps(chunk_payload))
                pipe.expire(chunks_key, self.ttl_seconds)
                pipe.hset(job_key, "updated_at", now)
                pipe.expire(job_key, self.ttl_seconds)
                await pipe.execute()

        except RedisError as exc:
            logger.error(
                "Redis error updating chunk {} status for job {}: {}",
                chunk_index,
                job_id,
                exc,
            )
            raise
        except Exception as exc:
            logger.exception(
                "Unexpected error updating chunk {} status for job {}: {}",
                chunk_index,
                job_id,
                exc,
            )
            raise

    async def increment_completed(self, job_id: str) -> int:
        """
        Atomically increment the completed chunk counter for a job.

        Args:
            job_id: Unique job identifier.

        Returns:
            New completed chunks count.
        """
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            now = self._get_utc_now()

            async with r.pipeline(transaction=True) as pipe:
                pipe.hincrby(job_key, "completed_chunks", 1)
                pipe.hset(job_key, "updated_at", now)
                pipe.expire(job_key, self.ttl_seconds)
                results = await pipe.execute()

            new_val = int(results[0])
            logger.info("Job {} completed_chunks incremented to {}", job_id, new_val)
            return new_val

        except RedisError as exc:
            logger.error("Redis error incrementing completed chunks for {}: {}", job_id, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error incrementing completed chunks for {}: {}", job_id, exc)
            raise

    async def increment_failed(self, job_id: str) -> int:
        """
        Atomically increment the failed chunk counter for a job.

        Args:
            job_id: Unique job identifier.

        Returns:
            New failed chunks count.
        """
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            now = self._get_utc_now()

            async with r.pipeline(transaction=True) as pipe:
                pipe.hincrby(job_key, "failed_chunks", 1)
                pipe.hset(job_key, "updated_at", now)
                pipe.expire(job_key, self.ttl_seconds)
                results = await pipe.execute()

            new_val = int(results[0])
            logger.warning("Job {} failed_chunks incremented to {}", job_id, new_val)
            return new_val

        except RedisError as exc:
            logger.error("Redis error incrementing failed chunks for {}: {}", job_id, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error incrementing failed chunks for {}: {}", job_id, exc)
            raise

    async def get_chunk_statuses(self, job_id: str) -> list[ChunkProgress]:
        """
        Retrieve all chunk progress entries for a job, sorted by chunk index.

        Args:
            job_id: Unique job identifier.

        Returns:
            Ordered list of ChunkProgress models.
        """
        try:
            r = await self._get_redis()
            chunks_key = self._chunks_key(job_id)
            raw_chunks = await r.hgetall(chunks_key)

            if not raw_chunks:
                return []

            chunk_list: list[ChunkProgress] = []
            for raw_val in raw_chunks.values():
                try:
                    data = json.loads(raw_val)
                    status_enum = ChunkStatus(data["status"])
                    chunk_list.append(
                        ChunkProgress(
                            chunk_index=int(data["chunk_index"]),
                            chunk_id=str(data["chunk_id"]),
                            status=status_enum,
                            progress_percent=float(data.get("progress_percent", 0.0)),
                            error_message=data.get("error_message"),
                        )
                    )
                except Exception as parse_err:
                    logger.warning("Failed parsing chunk data for job {}: {}", job_id, parse_err)

            chunk_list.sort(key=lambda c: c.chunk_index)
            return chunk_list

        except RedisError as exc:
            logger.error("Redis error retrieving chunk statuses for {}: {}", job_id, exc)
            return []
        except Exception as exc:
            logger.exception("Unexpected error retrieving chunk statuses for {}: {}", job_id, exc)
            return []

    async def add_celery_task_id(self, job_id: str, task_id: str) -> None:
        """
        Associate an asynchronous Celery task ID with a job for cancellation tracking.

        Args:
            job_id: Unique job identifier.
            task_id: Celery async task UUID string.
        """
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            raw_task_ids = await r.hget(job_key, "celery_task_ids")

            task_ids: list[str] = []
            if raw_task_ids:
                try:
                    task_ids = json.loads(raw_task_ids)
                except Exception:
                    task_ids = []

            if task_id not in task_ids:
                task_ids.append(task_id)

            now = self._get_utc_now()
            async with r.pipeline(transaction=True) as pipe:
                pipe.hset(job_key, "celery_task_ids", json.dumps(task_ids))
                pipe.hset(job_key, "updated_at", now)
                pipe.expire(job_key, self.ttl_seconds)
                await pipe.execute()

            logger.debug("Associated Celery task {} with job {}", task_id, job_id)

        except RedisError as exc:
            logger.error("Redis error adding task ID to job {}: {}", job_id, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error adding task ID to job {}: {}", job_id, exc)
            raise

    async def get_celery_task_ids(self, job_id: str) -> list[str]:
        """
        Retrieve all Celery task IDs registered for a job.

        Args:
            job_id: Unique job identifier.

        Returns:
            List of Celery task UUID strings.
        """
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            raw_task_ids = await r.hget(job_key, "celery_task_ids")

            if not raw_task_ids:
                return []

            try:
                task_ids = json.loads(raw_task_ids)
                return task_ids if isinstance(task_ids, list) else []
            except Exception as parse_err:
                logger.warning("Error parsing task IDs for job {}: {}", job_id, parse_err)
                return []

        except RedisError as exc:
            logger.error("Redis error retrieving task IDs for job {}: {}", job_id, exc)
            return []
        except Exception as exc:
            logger.exception("Unexpected error retrieving task IDs for job {}: {}", job_id, exc)
            return []

    async def set_output_url(self, job_id: str, url: str) -> None:
        """
        Set the final output video URL or download path for a completed job.

        Args:
            job_id: Unique job identifier.
            url: Accessible URL or endpoint path for video download.
        """
        logger.info("Setting output URL for job_id={}: {}", job_id, url)
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            now = self._get_utc_now()

            async with r.pipeline(transaction=True) as pipe:
                pipe.hset(job_key, "output_url", url)
                pipe.hset(job_key, "updated_at", now)
                pipe.expire(job_key, self.ttl_seconds)
                await pipe.execute()

        except RedisError as exc:
            logger.error("Redis error setting output URL for job {}: {}", job_id, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error setting output URL for job {}: {}", job_id, exc)
            raise

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete all Redis state keys associated with a job.

        Args:
            job_id: Unique job identifier.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        logger.info("Deleting state from Redis for job_id={}", job_id)
        try:
            r = await self._get_redis()
            job_key = self._job_key(job_id)
            chunks_key = self._chunks_key(job_id)

            deleted_count = await r.delete(job_key, chunks_key)
            logger.debug("Deleted {} Redis keys for job {}", deleted_count, job_id)
            return bool(deleted_count > 0)

        except RedisError as exc:
            logger.error("Redis error deleting job {}: {}", job_id, exc)
            return False
        except Exception as exc:
            logger.exception("Unexpected error deleting job {}: {}", job_id, exc)
            return False
