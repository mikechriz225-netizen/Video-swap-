"""ControlNet conditioning processor for structural and pose preservation.

Extracts OpenPose body/hand/face keypoints and MiDaS monocular depth maps
from video frames, and loads ControlNet models for guided latent diffusion.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from pipeline.vram import VRAMGuard, clear_vram, offload_to_cpu

if TYPE_CHECKING:
    from diffusers import ControlNetModel

__all__ = ["ControlNetProcessor"]


class ControlNetProcessor:
    """Manages structural feature extraction (pose and depth) and ControlNet model lifecycle."""

    def __init__(
        self,
        device_id: int = 0,
        openpose_model_id: str = "lllyasviel/control_v11p_sd15_openpose",
        depth_model_id: str = "lllyasviel/control_v11f1p_sd15_depth",
    ) -> None:
        """Initialize the ControlNet processor.

        Args:
            device_id: CUDA device index.
            openpose_model_id: HuggingFace hub repository ID for OpenPose ControlNet.
            depth_model_id: HuggingFace hub repository ID for Depth ControlNet.
        """
        self.device_id = device_id
        self.openpose_model_id = openpose_model_id
        self.depth_model_id = depth_model_id

        self._openpose_detector: Optional[Any] = None
        self._depth_detector: Optional[Any] = None
        self._openpose_controlnet: Optional[Any] = None
        self._depth_controlnet: Optional[Any] = None

        self._device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu"
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _load_openpose(self) -> Any:
        """Load and cache the OpenPose body and face/hands estimator."""
        if self._openpose_detector is not None:
            return self._openpose_detector

        with VRAMGuard(self.device_id, tag="LoadOpenPoseDetector"):
            start = time.perf_counter()
            logger.info("Loading OpenPose detector...")
            from controlnet_aux import OpenposeDetector

            self._openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            if hasattr(self._openpose_detector, "to") and torch.cuda.is_available():
                self._openpose_detector.to(self._device)
            logger.info("OpenPose detector loaded in {:.2f}s", time.perf_counter() - start)

        return self._openpose_detector

    def _load_depth(self) -> Any:
        """Load and cache the MiDaS depth estimator."""
        if self._depth_detector is not None:
            return self._depth_detector

        with VRAMGuard(self.device_id, tag="LoadMidasDetector"):
            start = time.perf_counter()
            logger.info("Loading MiDaS depth detector...")
            from controlnet_aux import MidasDetector

            self._depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
            if hasattr(self._depth_detector, "to") and torch.cuda.is_available():
                self._depth_detector.to(self._device)
            logger.info("MiDaS depth detector loaded in {:.2f}s", time.perf_counter() - start)

        return self._depth_detector

    def _load_controlnet_models(self) -> List[Any]:
        """Load and cache the OpenPose and Depth ControlNet models for Stable Diffusion.

        Returns:
            List containing [openpose_controlnet, depth_controlnet].
        """
        if self._openpose_controlnet is not None and self._depth_controlnet is not None:
            return [self._openpose_controlnet, self._depth_controlnet]

        with VRAMGuard(self.device_id, tag="LoadControlNetModels"):
            from diffusers import ControlNetModel

            if self._openpose_controlnet is None:
                start = time.perf_counter()
                logger.info("Loading ControlNet OpenPose from {}", self.openpose_model_id)
                self._openpose_controlnet = ControlNetModel.from_pretrained(
                    self.openpose_model_id,
                    torch_dtype=self._dtype,
                ).to(self._device)
                logger.info(
                    "ControlNet OpenPose loaded in {:.2f}s",
                    time.perf_counter() - start,
                )

            if self._depth_controlnet is None:
                start = time.perf_counter()
                logger.info("Loading ControlNet Depth from {}", self.depth_model_id)
                self._depth_controlnet = ControlNetModel.from_pretrained(
                    self.depth_model_id,
                    torch_dtype=self._dtype,
                ).to(self._device)
                logger.info(
                    "ControlNet Depth loaded in {:.2f}s",
                    time.perf_counter() - start,
                )

        return [self._openpose_controlnet, self._depth_controlnet]

    def _to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert a BGR or RGB numpy array to RGB PIL Image."""
        if isinstance(frame, Image.Image):
            return frame
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected frame to be numpy ndarray or PIL Image, got {type(frame)}")

        # Assume standard OpenCV BGR if 3-channel
        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        elif frame.ndim == 2:
            return Image.fromarray(frame).convert("RGB")
        return Image.fromarray(frame)

    def extract_pose(self, frame: np.ndarray) -> Image.Image:
        """Extract body and hand/face OpenPose skeleton conditioning image.

        Args:
            frame: Input video frame as numpy array (BGR).

        Returns:
            PIL Image containing the extracted pose skeleton.
        """
        with VRAMGuard(self.device_id, tag="ExtractPose"):
            start = time.perf_counter()
            detector = self._load_openpose()
            pil_image = self._to_pil(frame)

            # Detect pose including hands and face
            try:
                pose_map = detector(
                    pil_image,
                    hand_and_face=True,
                    image_resolution=max(pil_image.size),
                )
            except TypeError:
                # Fallback if specific kwargs differ in installed version
                pose_map = detector(pil_image)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.debug(
                "Extracted pose map (size={}, {:.1f}ms)",
                pose_map.size,
                elapsed_ms,
            )
            return pose_map

    def extract_depth(self, frame: np.ndarray) -> Image.Image:
        """Extract MiDaS monocular depth map conditioning image.

        Args:
            frame: Input video frame as numpy array (BGR).

        Returns:
            PIL Image containing grayscale/normalized depth map.
        """
        with VRAMGuard(self.device_id, tag="ExtractDepth"):
            start = time.perf_counter()
            detector = self._load_depth()
            pil_image = self._to_pil(frame)

            try:
                depth_map = detector(
                    pil_image,
                    image_resolution=max(pil_image.size),
                )
            except TypeError:
                depth_map = detector(pil_image)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.debug(
                "Extracted depth map (size={}, {:.1f}ms)",
                depth_map.size,
                elapsed_ms,
            )
            return depth_map

    def process_frame_conditions(self, frame: np.ndarray) -> Dict[str, Image.Image]:
        """Extract both pose and depth conditioning images for a single frame.

        Args:
            frame: Input video frame as numpy array (BGR).

        Returns:
            Dict containing {'pose': Image, 'depth': Image}.
        """
        with VRAMGuard(self.device_id, tag="ProcessFrameConditions"):
            start = time.perf_counter()
            pose_img = self.extract_pose(frame)
            depth_img = self.extract_depth(frame)
            logger.debug(
                "Processed all frame conditions in {:.1f}ms",
                (time.perf_counter() - start) * 1000.0,
            )
            return {"pose": pose_img, "depth": depth_img}

    def process_video_conditions(self, frames: List[np.ndarray]) -> List[Dict[str, Image.Image]]:
        """Batch process condition extraction across multiple video frames.

        Args:
            frames: List of video frame arrays.

        Returns:
            List of dictionaries each containing {'pose': Image, 'depth': Image}.
        """
        total = len(frames)
        logger.info("Extracting ControlNet conditions for {} frames...", total)
        start_total = time.perf_counter()
        results: List[Dict[str, Image.Image]] = []

        log_interval = max(1, total // 10)
        for idx, frame in enumerate(frames):
            cond = self.process_frame_conditions(frame)
            results.append(cond)

            if (idx + 1) % log_interval == 0 or (idx + 1) == total:
                logger.info(
                    "Condition extraction progress: {}/{} frames ({:.1f}%)",
                    idx + 1,
                    total,
                    ((idx + 1) / total) * 100.0,
                )

        total_elapsed = time.perf_counter() - start_total
        fps = (total / total_elapsed) if total_elapsed > 0 else 0.0
        logger.info(
            "Completed condition extraction for {} frames in {:.2f}s ({:.2f} fps)",
            total,
            total_elapsed,
            fps,
        )
        return results

    def get_controlnet_models(self) -> List[Any]:
        """Retrieve the loaded OpenPose and Depth ControlNet models.

        Returns:
            List of ControlNetModel instances: [openpose_model, depth_model].
        """
        return self._load_controlnet_models()

    def unload(self) -> None:
        """Offload all ControlNet neural models and detectors to CPU and reclaim VRAM."""
        logger.info("Unloading ControlNet processor models and clearing VRAM...")
        if self._openpose_controlnet is not None:
            offload_to_cpu(self._openpose_controlnet)
            self._openpose_controlnet = None

        if self._depth_controlnet is not None:
            offload_to_cpu(self._depth_controlnet)
            self._depth_controlnet = None

        if self._openpose_detector is not None:
            if hasattr(self._openpose_detector, "to"):
                try:
                    self._openpose_detector.to("cpu")
                except Exception as exc:
                    logger.debug("Could not move openpose detector to cpu: {}", exc)
            self._openpose_detector = None

        if self._depth_detector is not None:
            if hasattr(self._depth_detector, "to"):
                try:
                    self._depth_detector.to("cpu")
                except Exception as exc:
                    logger.debug("Could not move depth detector to cpu: {}", exc)
            self._depth_detector = None

        clear_vram(self.device_id)
        logger.info("ControlNet processor completely unloaded.")
