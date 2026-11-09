"""Celery application configuration for distributed video swap pipeline workers.

Configures queues, task serialization, GPU worker prefetch restrictions,
and automatic task discovery for chunk processing and video stitching workflows.
"""

from __future__ import annotations

import os
from celery import Celery
from loguru import logger

# Import configuration settings with graceful fallback
try:
    from core.config import get_settings
except ImportError:
    try:
        from config import get_settings  # type: ignore
    except ImportError:
        def get_settings():  # type: ignore
            class _FallbackSettings:
                celery_broker_url: str = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
                celery_result_backend: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
                redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            return _FallbackSettings()

settings = get_settings()

# Extract broker and backend from settings
broker_url = (
    getattr(settings, "celery_broker_url", None)
    or getattr(settings, "broker_url", None)
    or getattr(settings, "redis_url", None)
    or os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
)

result_backend = (
    getattr(settings, "celery_result_backend", None)
    or getattr(settings, "result_backend", None)
    or os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
)

logger.info(f"Initializing Celery app 'video_swap_pipeline' (broker={broker_url}, backend={result_backend})")

celery_app = Celery(
    "video_swap_pipeline",
    broker=broker_url,
    backend=result_backend,
)

# Apply Celery configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Strict single-task prefetch for GPU memory management
    task_routes={
        "workers.chunk_worker.process_chunk": {"queue": "chunks"},
        "workers.stitch_worker.stitch_job": {"queue": "stitch"},
    },
    task_default_queue="chunks",
    result_expires=86400,  # 24 hours
    worker_max_tasks_per_child=50,  # Periodic worker recycling to prevent GPU memory fragmentation
)

# Autodiscover tasks inside the workers package
celery_app.autodiscover_tasks(["workers"])
