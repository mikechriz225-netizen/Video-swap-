"""Structured logging configuration for the video swap pipeline.

Configures Loguru logging with formatted console outputs, rotating file handlers,
and structured JSON log streams for distributed observability and job tracing.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from loguru import logger

# Default log format for human-readable console and text file logs
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure system-wide loguru handlers.

    Removes existing default handlers, attaches a formatted stderr stream,
    registers a structured JSON lines rotating file sink at DEBUG level under 'logs/pipeline.jsonl',
    and optionally attaches a specified rotating log file.

    Args:
        log_level: Log level threshold for stderr (and optional log_file). Defaults to 'INFO'.
        log_file: Optional path to an additional rotating text log file.
    """
    # Remove existing default loguru handlers
    logger.remove()

    # 1. Console stderr handler
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format=LOG_FORMAT,
        colorize=True,
        enqueue=True,
    )

    # 2. Structured JSON log handler for production observability
    logs_dir = Path("logs")
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        json_log_path = logs_dir / "pipeline.jsonl"
        logger.add(
            str(json_log_path),
            level="DEBUG",
            serialize=True,
            rotation="100 MB",
            retention="7 days",
            enqueue=True,
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning(f"Could not create structured json log handler under 'logs/': {exc}")

    # 3. Optional user-specified rotating file handler
    if log_file:
        file_path = Path(log_file)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                str(file_path),
                level=log_level.upper(),
                format=LOG_FORMAT,
                rotation="100 MB",
                retention="7 days",
                enqueue=True,
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error(f"Failed to initialize custom log file at {log_file}: {exc}")


def get_job_logger(job_id: str) -> Any:
    """Return a contextual logger bound with a job_id for distributed tracing.

    Args:
        job_id: Unique identifier for the pipeline job.

    Returns:
        Loguru logger instance bound with context `job_id`.
    """
    return logger.bind(job_id=job_id)


# Automatically configure logging on module import
_default_level = os.environ.get("LOG_LEVEL", "INFO")
_default_file = os.environ.get("LOG_FILE", None)
setup_logging(log_level=_default_level, log_file=_default_file)
