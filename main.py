"""FastAPI application entry point for the Video Swap Pipeline.

Provides RESTful API routing, CORS handling for mobile and web clients,
system telemetry logging on startup, health checks, and graceful shutdown handlers.
"""

from datetime import datetime, timezone
import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    logger = logging.getLogger(__name__)  # type: ignore

from config import settings

# Import API routes router
try:
    from api.routes import router
except (ImportError, ModuleNotFoundError):
    # Graceful fallback in case api.routes is initialized dynamically or in progress
    from fastapi import APIRouter
    router = APIRouter()

# Initialize FastAPI application
app = FastAPI(
    title="Video Swap Pipeline API",
    version="1.0.0",
    description="High-throughput video face-swapping and latent diffusion pipeline API",
)

# Configure CORS middleware to permit requests from mobile and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API endpoints under /api/v1 prefix
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event() -> None:
    """Log system runtime specifications and GPU memory availability on startup."""
    logger.info("Starting Video Swap Pipeline API v1.0.0...")
    logger.info(
        f"Configuration initialized | Log Level: {settings.LOG_LEVEL} | "
        f"Upload Dir: {settings.UPLOAD_DIR} | Processed Dir: {settings.PROCESSED_DIR}"
    )

    if torch is not None and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        cuda_version = getattr(torch.version, "cuda", "N/A")
        logger.info(f"CUDA initialized: version={cuda_version}, device_count={gpu_count}")
        for dev_idx in range(gpu_count):
            props = torch.cuda.get_device_properties(dev_idx)
            total_vram_gb = props.total_memory / (1024 ** 3)
            allocated_vram_gb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
            reserved_vram_gb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
            available_vram_gb = total_vram_gb - reserved_vram_gb
            logger.info(
                f"GPU [{dev_idx}]: {props.name} | "
                f"Total VRAM: {total_vram_gb:.2f} GB | "
                f"Allocated VRAM: {allocated_vram_gb:.2f} GB | "
                f"Available VRAM: {available_vram_gb:.2f} GB | "
                f"Compute Capability: {props.major}.{props.minor}"
            )
    else:
        logger.warning(
            "CUDA is not available or torch was compiled without CUDA support. "
            "Operating in CPU-only / fallback mode."
        )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Perform resource cleanup and release hardware allocations on application shutdown."""
    logger.info("Shutting down Video Swap Pipeline API...")
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.info("Flushed CUDA memory cache successfully.")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Error flushing CUDA memory cache during shutdown: {exc}")


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint providing status and GPU availability.

    Returns:
        dict: Operational status, active GPU device count, and current UTC timestamp.
    """
    gpu_count = 0
    if torch is not None:
        try:
            gpu_count = torch.cuda.device_count()
        except Exception as exc:
            logger.warning(f"Error querying CUDA device count: {exc}")
            gpu_count = 0

    return {
        "status": "ok",
        "gpu_count": gpu_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )
