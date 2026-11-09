"""Temporal consistency and motion coherence enforcement module.

Implements optical-flow texture propagation, AnimateDiff motion adapter integration,
Gaussian temporal window smoothing, and chunk-boundary cross-fades to eliminate
flicker and seam artifacts across video sequences.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Optional

import cv2
import numpy as np
import torch
from loguru import logger

from pipeline.vram import VRAMGuard, clear_vram, offload_to_cpu

if TYPE_CHECKING:
    from diffusers import MotionAdapter

__all__ = ["TemporalConsistencyProcessor"]


class TemporalConsistencyProcessor:
    """Manages temporal coherence across generated video frames and chunks."""

    def __init__(
        self,
        device_id: int = 0,
        motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
    ) -> None:
        """Initialize the TemporalConsistencyProcessor.

        Args:
            device_id: CUDA device index.
            motion_adapter_id: HuggingFace model ID for AnimateDiff motion adapter.
        """
        self.device_id = device_id
        self.motion_adapter_id = motion_adapter_id
        self._motion_adapter: Optional[Any] = None

        self._device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu"
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _load_motion_adapter(self) -> Any:
        """Load and cache the AnimateDiff motion adapter model."""
        if self._motion_adapter is not None:
            return self._motion_adapter

        with VRAMGuard(self.device_id, tag="LoadAnimateDiffMotionAdapter"):
            start = time.perf_counter()
            logger.info("Loading AnimateDiff motion adapter from {}", self.motion_adapter_id)
            from diffusers import MotionAdapter

            self._motion_adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_id,
                torch_dtype=self._dtype,
            ).to(self._device)
            logger.info("AnimateDiff motion adapter loaded in {:.2f}s", time.perf_counter() - start)

        return self._motion_adapter

    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute dense optical flow from frame1 to frame2 using Farneback's algorithm.

        Args:
            frame1: First frame (H, W, C) or (H, W).
            frame2: Second frame (H, W, C) or (H, W).

        Returns:
            Flow field array of shape (H, W, 2) where flow[..., 0] is dx and flow[..., 1] is dy.
        """
        if frame1 is None or frame2 is None:
            raise ValueError("Frames for optical flow computation cannot be None.")

        # Convert to grayscale for gradient calculations
        if frame1.ndim == 3 and frame1.shape[2] >= 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1

        if frame2.ndim == 3 and frame2.shape[2] >= 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2

        start = time.perf_counter()
        flow = cv2.calcOpticalFlowFarneback(
            prev=gray1,
            next=gray2,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "Computed dense optical flow (shape={}, {:.1f}ms)",
            flow.shape,
            elapsed_ms,
        )
        return flow

    def warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp an image frame using a 2D dense optical flow vector field.

        Args:
            frame: Image array to warp (H, W, C).
            flow: Dense flow field (H, W, 2) specifying displacement vectors (dx, dy).

        Returns:
            Warped frame array (H, W, C) with Lanczos interpolation.
        """
        if frame is None or flow is None:
            raise ValueError("Frame and flow cannot be None.")

        h, w = frame.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (grid_x.astype(np.float32) + flow[..., 0]).astype(np.float32)
        map_y = (grid_y.astype(np.float32) + flow[..., 1]).astype(np.float32)

        warped = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT,
        )
        return warped

    def apply_temporal_smoothing(
        self,
        frames: List[np.ndarray],
        window_size: int = 5,
    ) -> List[np.ndarray]:
        """Apply weighted temporal averaging across rolling frame windows to remove high-frequency flicker.

        Uses normalized Gaussian weights centered on the current frame.

        Args:
            frames: Sequence of video frames (each H, W, C uint8).
            window_size: Odd integer window length (default: 5).

        Returns:
            List of smoothed video frames matching input length and dimensions.
        """
        if not frames:
            return []

        if len(frames) == 1:
            return [frames[0].copy()]

        start_time = time.perf_counter()
        # Guarantee odd window size
        if window_size % 2 == 0:
            window_size += 1

        radius = window_size // 2
        sigma = max(0.5, float(radius) / 2.0)
        kernel_indices = np.arange(-radius, radius + 1)
        raw_weights = np.exp(-0.5 * (kernel_indices / sigma) ** 2)
        norm_weights = raw_weights / np.sum(raw_weights)

        num_frames = len(frames)
        smoothed_frames: List[np.ndarray] = []

        for i in range(num_frames):
            acc = np.zeros_like(frames[i], dtype=np.float32)
            total_weight = 0.0

            for offset_idx, k in enumerate(kernel_indices):
                neighbor_idx = np.clip(i + k, 0, num_frames - 1)
                w = norm_weights[offset_idx]
                acc += frames[neighbor_idx].astype(np.float32) * w
                total_weight += w

            blended = acc / total_weight
            smoothed_frames.append(np.clip(blended, 0, 255).astype(np.uint8))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Applied Gaussian temporal smoothing across {} frames (window={}, {:.1f}ms)",
            num_frames,
            window_size,
            elapsed_ms,
        )
        return smoothed_frames

    def enforce_consistency(
        self,
        frames: List[np.ndarray],
        reference_frame_idx: int = 0,
    ) -> List[np.ndarray]:
        """Propagate stable textures across a sequence using bidirectional optical flow warping.

        This is the primary anti-flicker method, warping preceding consistent frames
        along motion trajectories with occlusion-aware adaptive alpha blending.

        Args:
            frames: List of frames to stabilize.
            reference_frame_idx: Index of anchor frame with desired identity/texture (default: 0).

        Returns:
            List of temporally consistent stabilized frames.
        """
        if not frames:
            return []
        if len(frames) <= 1:
            return [frames[0].copy()]

        with VRAMGuard(self.device_id, tag="EnforceTemporalConsistency"):
            start_time = time.perf_counter()
            num_frames = len(frames)
            ref_idx = max(0, min(num_frames - 1, reference_frame_idx))

            consistent_frames: List[np.ndarray] = [f.copy() for f in frames]
            logger.info(
                "Enforcing temporal consistency across {} frames (anchor index: {})...",
                num_frames,
                ref_idx,
            )

            # 1. Forward propagation from ref_idx to end
            for i in range(ref_idx + 1, num_frames):
                curr = frames[i]
                prev_consistent = consistent_frames[i - 1]

                # Compute backward flow from curr -> prev
                flow = self.compute_optical_flow(curr, prev_consistent)
                warped_prev = self.warp_frame(prev_consistent, flow)

                # Compute photometric difference to handle occlusions and sudden scene changes
                diff = np.mean(
                    np.abs(warped_prev.astype(np.float32) - curr.astype(np.float32)),
                    axis=2,
                    keepdims=True,
                ) / 255.0  # normalized [0, 1]

                # If difference is low, rely heavily on warped texture (anti-flicker).
                # If difference is high (occlusion/disocclusion), rely on current frame.
                adaptive_alpha = np.clip(0.65 * (1.0 - diff * 2.5), 0.05, 0.70)
                blended = (
                    adaptive_alpha * warped_prev.astype(np.float32)
                    + (1.0 - adaptive_alpha) * curr.astype(np.float32)
                )
                consistent_frames[i] = np.clip(blended, 0, 255).astype(np.uint8)

            # 2. Backward propagation from ref_idx down to 0
            for i in range(ref_idx - 1, -1, -1):
                curr = frames[i]
                next_consistent = consistent_frames[i + 1]

                flow = self.compute_optical_flow(curr, next_consistent)
                warped_next = self.warp_frame(next_consistent, flow)

                diff = np.mean(
                    np.abs(warped_next.astype(np.float32) - curr.astype(np.float32)),
                    axis=2,
                    keepdims=True,
                ) / 255.0

                adaptive_alpha = np.clip(0.65 * (1.0 - diff * 2.5), 0.05, 0.70)
                blended = (
                    adaptive_alpha * warped_next.astype(np.float32)
                    + (1.0 - adaptive_alpha) * curr.astype(np.float32)
                )
                consistent_frames[i] = np.clip(blended, 0, 255).astype(np.uint8)

            elapsed_s = time.perf_counter() - start_time
            logger.info("Temporal consistency enforced across {} frames in {:.2f}s", num_frames, elapsed_s)
            return consistent_frames

    def smooth_transitions(
        self,
        chunk_end_frames: List[np.ndarray],
        chunk_start_frames: List[np.ndarray],
        blend_frames: int = 5,
    ) -> List[np.ndarray]:
        """Create smooth cross-fade transitions between video chunks to eliminate boundary seams.

        Args:
            chunk_end_frames: Tail frames from the preceding video chunk.
            chunk_start_frames: Head frames from the succeeding video chunk.
            blend_frames: Number of transition frames to cross-fade (default: 5).

        Returns:
            List of smoothly cross-faded transition frames.
        """
        n_end = len(chunk_end_frames)
        n_start = len(chunk_start_frames)
        actual_blend = min(blend_frames, n_end, n_start)

        if actual_blend <= 0:
            logger.warning("No overlapping frames provided for transition smoothing.")
            return []

        logger.info("Smoothing transition between chunks over {} boundary frames...", actual_blend)
        start_time = time.perf_counter()
        transitions: List[np.ndarray] = []

        end_subset = chunk_end_frames[-actual_blend:]
        start_subset = chunk_start_frames[:actual_blend]

        for i in range(actual_blend):
            # Smooth cosine ramp: s-curve from 0.0 to 1.0
            t = float(i + 1) / float(actual_blend + 1)
            alpha = 0.5 * (1.0 - np.cos(np.pi * t))

            frame_a = end_subset[i].astype(np.float32)
            frame_b = start_subset[i].astype(np.float32)

            blended = (1.0 - alpha) * frame_a + alpha * frame_b
            transitions.append(np.clip(blended, 0, 255).astype(np.uint8))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug("Generated {} transition frames in {:.1f}ms", len(transitions), elapsed_ms)
        return transitions

    def unload(self) -> None:
        """Unload motion models to CPU and reclaim GPU memory."""
        logger.info("Unloading TemporalConsistencyProcessor...")
        if self._motion_adapter is not None:
            offload_to_cpu(self._motion_adapter)
            self._motion_adapter = None

        clear_vram(self.device_id)
        logger.info("TemporalConsistencyProcessor completely unloaded.")
