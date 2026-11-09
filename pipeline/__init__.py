"""Video character identity swap and neural generation pipeline package.

Exposes submodules for VRAM management, aspect-ratio and resolution handling,
face detection and segmentation, ControlNet conditioning, IP-Adapter identity
transfer, temporal consistency stabilization, and artifact removal.
"""

from __future__ import annotations

from pipeline.vram import (
    VRAMGuard,
    clear_vram,
    check_vram_threshold,
    get_vram_info,
    move_to_device,
    offload_to_cpu,
    select_optimal_device,
)
from pipeline.resolution import (
    AspectRatio,
    compute_latent_dimensions,
    detect_aspect_ratio,
    get_resize_dimensions,
    resize_frame,
    resize_to_original,
)
from pipeline.face_detection import FaceDetector
from pipeline.controlnet import ControlNetProcessor
from pipeline.identity_transfer import IdentityTransfer
from pipeline.temporal import TemporalConsistencyProcessor
from pipeline.artifact_removal import ArtifactRemover

__all__ = [
    # VRAM
    "VRAMGuard",
    "clear_vram",
    "check_vram_threshold",
    "get_vram_info",
    "move_to_device",
    "offload_to_cpu",
    "select_optimal_device",
    # Resolution
    "AspectRatio",
    "compute_latent_dimensions",
    "detect_aspect_ratio",
    "get_resize_dimensions",
    "resize_frame",
    "resize_to_original",
    # Face Detection
    "FaceDetector",
    # ControlNet
    "ControlNetProcessor",
    # Identity Transfer
    "IdentityTransfer",
    # Temporal
    "TemporalConsistencyProcessor",
    # Artifact Removal
    "ArtifactRemover",
]
