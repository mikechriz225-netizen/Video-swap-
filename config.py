"""Centralized application configuration using Pydantic Settings.

Manages pipeline hyperparameters, model identifiers, Redis/Celery connection
strings, filesystem directories, and hardware options.
"""

from functools import lru_cache
import logging
from pathlib import Path
from typing import List

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    logger = logging.getLogger(__name__)  # type: ignore


class Settings(BaseSettings):
    """Central configuration schema for the video swap pipeline."""

    # Redis and Celery message broker / backend configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Filesystem directories
    UPLOAD_DIR: Path = Path("/tmp/vswap/uploads")
    PROCESSED_DIR: Path = Path("/tmp/vswap/processed")
    OUTPUT_DIR: Path = Path("/tmp/vswap/output")

    # Video constraints and chunking parameters
    MAX_VIDEO_DURATION_SECONDS: int = 600  # 10 minutes max
    CHUNK_MIN_SECONDS: float = 5.0
    CHUNK_MAX_SECONDS: float = 10.0
    MAX_PARALLEL_WORKERS: int = 4

    # Pretrained model checkpoints and repository IDs
    DIFFUSION_MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_OPENPOSE_MODEL: str = "lllyasviel/control_v11p_sd15_openpose"
    CONTROLNET_DEPTH_MODEL: str = "lllyasviel/control_v11f1p_sd15_depth"
    IP_ADAPTER_MODEL: str = "h94/IP-Adapter-FaceID-Plus"
    ANIMATEDIFF_MODEL: str = "guoyww/animatediff-motion-adapter-v1-5-3"
    WAV2LIP_CHECKPOINT: str = "checkpoints/wav2lip_gan.pth"

    # Diffusion inference and resolution hyperparameters
    LATENT_RESOLUTION: int = 512
    MAX_LATENT_RESOLUTION: int = 1024
    INFERENCE_STEPS: int = 30
    GUIDANCE_SCALE: float = 7.5
    CONTROLNET_CONDITIONING_SCALE: float = 0.8
    IP_ADAPTER_SCALE: float = 0.6

    # Face blending and post-processing settings
    FACE_MASK_DILATION_PX: int = 15
    BILATERAL_FILTER_D: int = 9
    BILATERAL_SIGMA_COLOR: float = 75.0
    BILATERAL_SIGMA_SPACE: float = 75.0

    # Hardware & Logging
    GPU_DEVICES: List[int] = [0]
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="VSWAP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Retrieve and cache the singleton application settings instance.

    Returns:
        Settings: The validated pipeline configuration object.
    """
    return Settings()


def _ensure_directories(cfg: Settings) -> None:
    """Create essential working directories if they do not exist.

    Args:
        cfg: Configuration settings instance holding directory paths.
    """
    for dir_path in (cfg.UPLOAD_DIR, cfg.PROCESSED_DIR, cfg.OUTPUT_DIR):
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to create directory {dir_path}: {exc}")


# Initialize settings and ensure required runtime directories exist on module load
settings: Settings = get_settings()
_ensure_directories(settings)
