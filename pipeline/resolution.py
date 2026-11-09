"""Aspect ratio preservation and latent resolution mapping utilities.

Provides mathematical and image processing functions to map source video frame
resolutions to latent diffusion dimensions (multiples of 8) without distorting
aspect ratios, as well as high-quality resampling with OpenCV.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
from loguru import logger

__all__ = [
    "AspectRatio",
    "detect_aspect_ratio",
    "compute_latent_dimensions",
    "get_resize_dimensions",
    "resize_frame",
    "resize_to_original",
]


class AspectRatio(Enum):
    """Standard video and image aspect ratio categories."""

    PORTRAIT_9_16 = "9:16"
    LANDSCAPE_16_9 = "16:9"
    SQUARE_1_1 = "1:1"
    OTHER = "other"


def detect_aspect_ratio(width: int, height: int, tolerance: float = 0.08) -> AspectRatio:
    """Detect the aspect ratio category of given image dimensions.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        tolerance: Allowed deviation threshold from exact ratios (default: 0.08).

    Returns:
        AspectRatio enum member matching the detected category.
    """
    if width <= 0 or height <= 0:
        logger.error("Invalid image dimensions provided: width={}, height={}", width, height)
        raise ValueError(f"Dimensions must be positive non-zero integers: ({width}, {height})")

    ratio = float(width) / float(height)
    target_9_16 = 9.0 / 16.0  # 0.5625
    target_16_9 = 16.0 / 9.0  # 1.7778
    target_1_1 = 1.0

    detected: AspectRatio
    if abs(ratio - target_9_16) <= tolerance:
        detected = AspectRatio.PORTRAIT_9_16
    elif abs(ratio - target_16_9) <= tolerance:
        detected = AspectRatio.LANDSCAPE_16_9
    elif abs(ratio - target_1_1) <= tolerance:
        detected = AspectRatio.SQUARE_1_1
    else:
        detected = AspectRatio.OTHER

    logger.debug(
        "Detected aspect ratio for {}x{} (ratio={:.4f}): {}",
        width,
        height,
        ratio,
        detected.value,
    )
    return detected


def compute_latent_dimensions(
    source_width: int,
    source_height: int,
    max_latent: int = 512,
) -> Tuple[int, int]:
    """Map source resolution to nearest multiple-of-8 latent dimensions preserving aspect ratio.

    Ensures that both width and height are strictly divisible by 8 (required by
    Variational Autoencoders in latent diffusion architectures) and that neither
    dimension exceeds `max_latent`.

    Example:
        1080x1920 (9:16) -> (288, 512)
        1920x1080 (16:9) -> (512, 288)
        1080x1080 (1:1)  -> (512, 512)

    Args:
        source_width: Original width in pixels.
        source_height: Original height in pixels.
        max_latent: Maximum dimension bound for the latent space (default: 512).

    Returns:
        Tuple of (latent_width, latent_height), both multiples of 8.
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Source dimensions must be positive: ({source_width}, {source_height})")

    # Ensure max_latent is itself a valid multiple of 8
    max_bound = max(8, (max_latent // 8) * 8)

    aspect = float(source_width) / float(source_height)

    if source_width >= source_height:
        # Landscape or square
        latent_width = max_bound
        raw_height = max_bound / aspect
        latent_height = int(round(raw_height / 8.0) * 8)
    else:
        # Portrait
        latent_height = max_bound
        raw_width = max_bound * aspect
        latent_width = int(round(raw_width / 8.0) * 8)

    # Enforce minimum dimension of 8 and bound by max_bound
    latent_width = max(8, min(max_bound, latent_width))
    latent_height = max(8, min(max_bound, latent_height))

    # Re-verify multiple of 8
    latent_width = (latent_width // 8) * 8
    latent_height = (latent_height // 8) * 8

    logger.debug(
        "Mapped source resolution {}x{} to latent resolution {}x{} (max_latent={})",
        source_width,
        source_height,
        latent_width,
        latent_height,
        max_bound,
    )
    return (latent_width, latent_height)


def get_resize_dimensions(
    source_width: int,
    source_height: int,
    target_max: int = 1024,
) -> Tuple[int, int]:
    """Calculate downscaled/adjusted dimensions constrained by target_max and divisible by 8.

    Args:
        source_width: Original width in pixels.
        source_height: Original height in pixels.
        target_max: Maximum allowed length for any dimension (default: 1024).

    Returns:
        Tuple of (target_width, target_height), both multiples of 8.
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Source dimensions must be positive: ({source_width}, {source_height})")

    longest_side = max(source_width, source_height)
    if longest_side > target_max:
        scale = float(target_max) / float(longest_side)
    else:
        scale = 1.0

    target_w = max(8, int(round((source_width * scale) / 8.0) * 8))
    target_h = max(8, int(round((source_height * scale) / 8.0) * 8))

    logger.debug(
        "Calculated resize dimensions for {}x{} -> {}x{} (target_max={})",
        source_width,
        source_height,
        target_w,
        target_h,
        target_max,
    )
    return (target_w, target_h)


def resize_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize an image frame to specified target dimensions using Lanczos resampling.

    Args:
        frame: Source frame as numpy array (H, W, C) or (H, W).
        target_width: Desired target width in pixels.
        target_height: Desired target height in pixels.

    Returns:
        Resized numpy array of shape (target_height, target_width, ...).
    """
    if frame is None or not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected frame to be a numpy ndarray, got {type(frame)}")

    h, w = frame.shape[:2]
    if (w, h) == (target_width, target_height):
        return frame.copy()

    start_time = time.perf_counter()
    resized = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    logger.debug(
        "Resized frame from {}x{} to {}x{} (Lanczos4, {:.2f} ms)",
        w,
        h,
        target_width,
        target_height,
        elapsed_ms,
    )
    return resized


def resize_to_original(
    frame: np.ndarray,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """Scale a processed frame back to its original video resolution.

    Args:
        frame: Processed frame numpy array.
        original_width: Target original width.
        original_height: Target original height.

    Returns:
        Upscaled/restored numpy array matching (original_height, original_width).
    """
    if frame is None or not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected frame to be a numpy ndarray, got {type(frame)}")

    h, w = frame.shape[:2]
    if (w, h) == (original_width, original_height):
        return frame.copy()

    logger.info(
        "Scaling frame back up from {}x{} to original {}x{}",
        w,
        h,
        original_width,
        original_height,
    )
    return resize_frame(frame, original_width, original_height)
