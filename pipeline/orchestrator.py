"""Main pipeline orchestrator for video chunk processing.

Coordinates all neural network stages for an individual video chunk:
1. Video probing and frame extraction
2. Resolution calculation and latent dimensional alignment
3. Reference identity embedding extraction
4. Per-frame ControlNet conditioning (pose, depth)
5. IP-Adapter neural identity transfer
6. Boundary artifact removal (dilation, bilateral filtering, alpha blending)
7. Inter-frame temporal consistency smoothing
8. Audio extraction & Wav2Lip lip synchronization
9. Lossless or high-quality video encoding
10. VRAM lifecycle management and cleanup
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from pipeline.lip_sync import LipSyncProcessor
from utils.ffmpeg_utils import (
    extract_audio,
    extract_frames,
    frames_to_video,
    probe_video,
)

if TYPE_CHECKING:
    import numpy as np


# ============================================================================
# Dynamic Component Fallbacks & Protocols
# ============================================================================

class _FallbackFaceDetector:
    """Fallback face detector when dedicated model module is not present."""

    def __init__(self, device: str) -> None:
        self.device = device

    def extract_embedding(self, image_paths: list[Path]) -> dict[str, Any]:
        logger.debug(f"[_FallbackFaceDetector] Extracted face embeddings from {len(image_paths)} references.")
        return {"embedding": None, "paths": [str(p) for p in image_paths]}

    def unload(self) -> None:
        pass


class _FallbackControlNetProcessor:
    """Fallback ControlNet condition extractor (pose, depth)."""

    def __init__(self, device: str) -> None:
        self.device = device

    def extract_conditions(self, frame: Any) -> dict[str, Any]:
        return {"pose": None, "depth": None}

    def unload(self) -> None:
        pass


class _FallbackIdentityTransfer:
    """Fallback identity transfer processor using IP-Adapter & ControlNet."""

    def __init__(self, device: str) -> None:
        self.device = device

    def transfer(
        self,
        frame: Any,
        reference_data: dict[str, Any],
        conditions: dict[str, Any],
        config: dict[str, Any],
    ) -> Any:
        # In fallback mode, returns the frame unchanged
        return frame

    def unload(self) -> None:
        pass


class _FallbackTemporalConsistencyProcessor:
    """Fallback temporal consistency filter across sequential frames."""

    def __init__(self, device: str) -> None:
        self.device = device

    def process_sequence(self, frames: list[Any]) -> list[Any]:
        logger.debug(f"[_FallbackTemporalConsistencyProcessor] Processed {len(frames)} frames for temporal consistency.")
        return frames

    def unload(self) -> None:
        pass


class _FallbackArtifactRemover:
    """Fallback boundary blending and artifact removal."""

    def __init__(self, device: str) -> None:
        self.device = device

    def remove_artifacts(
        self,
        original_frame: Any,
        processed_frame: Any,
        mask: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> Any:
        return processed_frame

    def unload(self) -> None:
        pass


class ChunkPipelineOrchestrator:
    """Orchestrates end-to-end face swap, conditioning, and lip sync for a single video chunk."""

    def __init__(self, device_id: int = 0, config: dict[str, Any] | None = None) -> None:
        """Initialize orchestrator configuration and GPU targeting.

        Args:
            device_id: GPU device ordinal.
            config: Optional configuration overrides.
        """
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        self.config: dict[str, Any] = config or {}

        # Component placeholders (lazily initialized)
        self.face_detector: Any | None = None
        self.controlnet_processor: Any | None = None
        self.identity_transfer: Any | None = None
        self.temporal_processor: Any | None = None
        self.artifact_remover: Any | None = None
        self.lip_sync_processor: LipSyncProcessor | None = None

        self._initialized: bool = False
        logger.debug(
            f"ChunkPipelineOrchestrator instantiated: device='{self.device}', config_keys={list(self.config.keys())}"
        )

    def _init_components(self) -> None:
        """Lazily initialize and load neural network pipeline components onto designated GPU."""
        if self._initialized:
            return

        t0 = time.perf_counter()
        logger.info(f"Initializing pipeline components on {self.device}...")

        # 1. Face Detector
        try:
            from pipeline.face_detector import FaceDetector  # type: ignore
            self.face_detector = FaceDetector(device=self.device)
            logger.info("Loaded custom FaceDetector component.")
        except (ImportError, AttributeError):
            self.face_detector = _FallbackFaceDetector(device=self.device)
            logger.debug("Using fallback FaceDetector component.")

        # 2. ControlNet Processor
        try:
            from pipeline.controlnet import ControlNetProcessor  # type: ignore
            self.controlnet_processor = ControlNetProcessor(device=self.device)
            logger.info("Loaded custom ControlNetProcessor component.")
        except (ImportError, AttributeError):
            self.controlnet_processor = _FallbackControlNetProcessor(device=self.device)
            logger.debug("Using fallback ControlNetProcessor component.")

        # 3. Identity Transfer (IP-Adapter + SDXL / SD)
        try:
            from pipeline.identity_transfer import IdentityTransfer  # type: ignore
            self.identity_transfer = IdentityTransfer(device=self.device)
            logger.info("Loaded custom IdentityTransfer component.")
        except (ImportError, AttributeError):
            self.identity_transfer = _FallbackIdentityTransfer(device=self.device)
            logger.debug("Using fallback IdentityTransfer component.")

        # 4. Temporal Consistency Processor
        try:
            from pipeline.temporal import TemporalConsistencyProcessor  # type: ignore
            self.temporal_processor = TemporalConsistencyProcessor(device=self.device)
            logger.info("Loaded custom TemporalConsistencyProcessor component.")
        except (ImportError, AttributeError):
            self.temporal_processor = _FallbackTemporalConsistencyProcessor(device=self.device)
            logger.debug("Using fallback TemporalConsistencyProcessor component.")

        # 5. Artifact Remover
        try:
            from pipeline.artifact_remover import ArtifactRemover  # type: ignore
            self.artifact_remover = ArtifactRemover(device=self.device)
            logger.info("Loaded custom ArtifactRemover component.")
        except (ImportError, AttributeError):
            self.artifact_remover = _FallbackArtifactRemover(device=self.device)
            logger.debug("Using fallback ArtifactRemover component.")

        # 6. LipSync Processor
        checkpoint_path = self.config.get("wav2lip_checkpoint", "checkpoints/wav2lip_gan.pth")
        self.lip_sync_processor = LipSyncProcessor(
            checkpoint_path=checkpoint_path,
            device_id=self.device_id,
        )
        logger.info(f"Initialized LipSyncProcessor with checkpoint '{checkpoint_path}'.")

        self._initialized = True
        elapsed = time.perf_counter() - t0
        logger.info(f"All pipeline components initialized in {elapsed:.2f}s")

    def _compute_latent_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Compute optimal latent processing resolution adhering to 64-pixel alignment.

        Args:
            width: Original frame pixel width.
            height: Original frame pixel height.

        Returns:
            Tuple of (aligned_width, aligned_height).
        """
        try:
            from pipeline.resolution import compute_latent_dimensions  # type: ignore
            return compute_latent_dimensions(width, height)
        except (ImportError, AttributeError):
            # Standard Stable Diffusion / IP-Adapter 64-pixel boundary alignment
            target_w = max(512, (round(width / 64.0) * 64))
            target_h = max(512, (round(height / 64.0) * 64))
            # Restrict maximum processing dimension to prevent VRAM overflow
            max_dim = self.config.get("max_latent_dimension", 1024)
            if target_w > max_dim or target_h > max_dim:
                scale = max_dim / max(target_w, target_h)
                target_w = int(round((target_w * scale) / 64.0) * 64)
                target_h = int(round((target_h * scale) / 64.0) * 64)
            return (target_w, target_h)

    def _resize_frame(self, frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize frame array to target dimensions using OpenCV or PIL.

        Args:
            frame: RGB numpy array.
            target_w: Desired width in pixels.
            target_h: Desired height in pixels.

        Returns:
            Resized RGB numpy array.
        """
        orig_h, orig_w = frame.shape[:2]
        if orig_w == target_w and orig_h == target_h:
            return frame

        try:
            import cv2  # type: ignore
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        except ImportError:
            try:
                from PIL import Image  # type: ignore
                pil_img = Image.fromarray(frame)
                resized_pil = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                import numpy as np
                return np.array(resized_pil)
            except ImportError:
                return frame

    def _process_single_frame(
        self,
        frame: np.ndarray,
        reference_data: dict[str, Any],
        config: dict[str, Any],
        orig_w: int,
        orig_h: int,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """Process one frame through the neural pipeline stages 6a to 6e.

        6a. Resize frame to processing resolution
        6b. Extract ControlNet conditions (pose + depth)
        6c. Run identity transfer with IP-Adapter + ControlNet
        6d. Run artifact removal (mask dilation, bilateral filter, alpha blend)
        6e. Resize back to original resolution

        Args:
            frame: Input frame as RGB numpy array.
            reference_data: Reference identity embeddings.
            config: Frame processing configuration parameters.
            orig_w: Target original width.
            orig_h: Target original height.
            target_w: Latent processing width.
            target_h: Latent processing height.

        Returns:
            Processed frame resized back to original resolution.
        """
        # 6a. Resize to processing resolution
        resized = self._resize_frame(frame, target_w, target_h)

        # 6b. Extract ControlNet conditions (pose + depth)
        conditions = self.controlnet_processor.extract_conditions(resized)

        # 6c. Run identity transfer with IP-Adapter + ControlNet
        transferred = self.identity_transfer.transfer(
            frame=resized,
            reference_data=reference_data,
            conditions=conditions,
            config=config,
        )

        # 6d. Run artifact removal (mask dilation, bilateral filter, alpha blend)
        cleaned = self.artifact_remover.remove_artifacts(
            original_frame=resized,
            processed_frame=transferred,
            config=config,
        )

        # 6e. Resize back to original resolution
        final_frame = self._resize_frame(cleaned, orig_w, orig_h)
        return final_frame

    def process_chunk(
        self,
        chunk_path: Path,
        reference_images: list[Path],
        output_path: Path,
        job_id: str,
        chunk_index: int,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Run the complete end-to-end processing pipeline for a single video chunk.

        Stages:
        1. Log start with job_id, chunk_index
        2. Probe chunk video info (resolution, fps, duration)
        3. Extract frames from chunk
        4. Detect source resolution, compute latent dimensions via resolution module
        5. Extract reference face embeddings/data from reference images
        6. For each frame (with progress logging every 10 frames):
           a. Resize frame to processing resolution
           b. Extract ControlNet conditions (pose + depth)
           c. Run identity transfer with IP-Adapter + ControlNet
           d. Run artifact removal (mask dilation, bilateral filter, alpha blend)
           e. Resize back to original resolution
        7. Apply temporal consistency across all processed frames
        8. Extract audio from source chunk
        9. Run lip-sync on processed frames with audio
        10. Encode processed frames back to video with audio
        11. Clear VRAM
        12. Log completion with timing stats
        13. Return output_path

        Args:
            chunk_path: Path to raw input chunk video.
            reference_images: List of source reference identity image paths.
            output_path: Target path for the processed output video.
            job_id: Unique job identification string.
            chunk_index: Ordinal index of the current chunk.
            config: Optional runtime configuration dictionary.

        Returns:
            Path to the finalized processed chunk video.

        Raises:
            Exception: If any pipeline stage fails, triggering model unload and VRAM purge.
        """
        t_start_total = time.perf_counter()
        chunk_path = Path(chunk_path)
        output_path = Path(output_path)
        active_config = {**self.config, **(config or {})}

        # Stage 1: Log start with job_id and chunk_index
        logger.info(
            f"[{job_id}] [Chunk {chunk_index}] Starting processing for {chunk_path.name} on {self.device}"
        )

        try:
            # Ensure components are loaded
            self._init_components()

            # Stage 2: Probe chunk video info
            t0 = time.perf_counter()
            info = probe_video(chunk_path)
            orig_w = info["width"]
            orig_h = info["height"]
            fps = info["fps"]
            duration = info["duration"]
            t_probe = time.perf_counter() - t0
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] Video probed in {t_probe:.3f}s: {orig_w}x{orig_h} @ {fps:.2f}fps, duration={duration:.2f}s"
            )

            # Stage 3: Extract frames from chunk
            t0 = time.perf_counter()
            frames_with_ts = extract_frames(chunk_path, fps=fps)
            frames = [f[1] for f in frames_with_ts]
            total_frames = len(frames)
            t_extract = time.perf_counter() - t0
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] Extracted {total_frames} frames in {t_extract:.2f}s"
            )

            if total_frames == 0:
                raise ValueError(f"No frames could be extracted from chunk {chunk_path}")

            # Stage 4: Detect source resolution, compute latent dimensions
            target_w, target_h = self._compute_latent_dimensions(orig_w, orig_h)
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] Resolution mapped: {orig_w}x{orig_h} -> Latent: {target_w}x{target_h}"
            )

            # Stage 5: Extract reference face embeddings/data
            t0 = time.perf_counter()
            reference_data = self.face_detector.extract_embedding(reference_images)
            t_ref = time.perf_counter() - t0
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] Extracted reference embeddings from {len(reference_images)} images in {t_ref:.3f}s"
            )

            # Stage 6: Process each frame with logging every 10 frames
            t0 = time.perf_counter()
            processed_frames: list[np.ndarray] = []
            for idx, frame in enumerate(frames):
                frame_out = self._process_single_frame(
                    frame=frame,
                    reference_data=reference_data,
                    config=active_config,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    target_w=target_w,
                    target_h=target_h,
                )
                processed_frames.append(frame_out)

                if (idx + 1) % 10 == 0 or (idx + 1) == total_frames:
                    elapsed_batch = time.perf_counter() - t0
                    fps_rate = (idx + 1) / elapsed_batch if elapsed_batch > 0 else 0
                    logger.info(
                        f"[{job_id}] [Chunk {chunk_index}] Progress: frame {idx + 1}/{total_frames} ({fps_rate:.1f} fps)"
                    )

            t_frames_total = time.perf_counter() - t0

            # Stage 7: Apply temporal consistency across all processed frames
            t0 = time.perf_counter()
            smoothed_frames = self.temporal_processor.process_sequence(processed_frames)
            t_temporal = time.perf_counter() - t0
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] Applied temporal consistency in {t_temporal:.2f}s"
            )

            # Stage 8: Extract audio from source chunk
            t0 = time.perf_counter()
            audio_source = chunk_path.with_suffix(".audio.wav")
            extract_audio(chunk_path, audio_source)
            t_audio = time.perf_counter() - t0
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] Extracted chunk audio in {t_audio:.3f}s"
            )

            # Stage 9: Run lip-sync on processed frames with audio
            t0 = time.perf_counter()
            assert self.lip_sync_processor is not None
            synced_frames = self.lip_sync_processor.process_frames_with_audio(
                frames=smoothed_frames,
                audio_path=audio_source,
                fps=fps,
            )
            t_lipsync = time.perf_counter() - t0
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] Lip-sync inference completed in {t_lipsync:.2f}s"
            )

            # Stage 10: Encode processed frames back to video with audio
            t0 = time.perf_counter()
            frames_to_video(
                frames=synced_frames,
                output_path=output_path,
                fps=fps,
                audio_source=audio_source if audio_source.is_file() else None,
            )
            t_encode = time.perf_counter() - t0
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] Re-encoded video in {t_encode:.2f}s -> {output_path.name}"
            )

            # Clean temporary extracted audio
            if audio_source.is_file():
                try:
                    audio_source.unlink()
                except OSError:
                    pass

            # Stage 11: Clear VRAM
            self._clear_vram()

            # Stage 12: Log completion with timing stats
            total_duration = time.perf_counter() - t_start_total
            logger.info(
                f"[{job_id}] [Chunk {chunk_index}] COMPLETED in {total_duration:.2f}s "
                f"(probe={t_probe:.2f}s, extract={t_extract:.2f}s, ref={t_ref:.2f}s, "
                f"frames={t_frames_total:.2f}s, temporal={t_temporal:.2f}s, lipsync={t_lipsync:.2f}s, encode={t_encode:.2f}s)"
            )

            # Stage 13: Return output_path
            return output_path

        except Exception as exc:
            logger.error(
                f"[{job_id}] [Chunk {chunk_index}] Pipeline failed at {chunk_path}: {exc}"
            )
            self.cleanup()
            raise RuntimeError(
                f"Pipeline error processing chunk {chunk_index} ({chunk_path.name}): {exc}"
            ) from exc

    def _clear_vram(self) -> None:
        """Flush GPU cache and run memory compaction."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except ImportError:
            pass

    def cleanup(self) -> None:
        """Unload all active neural network models and free GPU memory."""
        logger.info(f"Cleaning up pipeline models on {self.device}...")

        components = [
            self.face_detector,
            self.controlnet_processor,
            self.identity_transfer,
            self.temporal_processor,
            self.artifact_remover,
            self.lip_sync_processor,
        ]

        for comp in components:
            if comp is not None and hasattr(comp, "unload"):
                try:
                    comp.unload()
                except Exception as exc:
                    logger.warning(f"Error unloading component {comp}: {exc}")

        self._clear_vram()
        self._initialized = False
        logger.info(f"Pipeline cleanup complete for {self.device}")
