"""Edge blending, mask dilation, and bilateral filtering for seamless compositing.

This module removes seam artifacts, matches skin/lighting color histograms in LAB
color space, and performs edge-feathered bilateral alpha blending to achieve
invisible compositing of transferred character faces onto background video frames.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

__all__ = ["ArtifactRemover"]


class ArtifactRemover:
    """Compositing and artifact cleanup engine for face and character swaps."""

    def __init__(
        self,
        dilation_px: int = 15,
        bilateral_d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> None:
        """Initialize the ArtifactRemover.

        Args:
            dilation_px: Default radius in pixels for mask dilation (default: 15).
            bilateral_d: Diameter of each pixel neighborhood used during bilateral filtering (default: 9).
            sigma_color: Filter sigma in the color space (default: 75.0).
            sigma_space: Filter sigma in the coordinate space (default: 75.0).
        """
        self.dilation_px = dilation_px
        self.bilateral_d = bilateral_d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def dilate_mask(self, mask: np.ndarray, pixels: Optional[int] = None) -> np.ndarray:
        """Apply morphological dilation to expand a binary mask.

        Args:
            mask: Binary mask array (H, W) uint8.
            pixels: Expansion distance in pixels (defaults to self.dilation_px).

        Returns:
            Dilated binary mask of identical dimensions and uint8 type.
        """
        px = self.dilation_px if pixels is None else pixels
        if px <= 0:
            return mask.copy()

        ksize = 2 * px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return dilated

    def create_feathered_mask(self, mask: np.ndarray, feather_radius: int = 21) -> np.ndarray:
        """Smooth mask boundaries via Gaussian blur to create a soft blending alpha channel.

        Args:
            mask: Input mask (H, W) uint8 with values in [0, 255].
            feather_radius: Kernel size for Gaussian blur (must be positive odd integer).

        Returns:
            Feathered alpha mask array of shape (H, W) with float32 values in [0.0, 1.0].
        """
        if feather_radius % 2 == 0:
            feather_radius += 1
        feather_radius = max(3, feather_radius)

        # Normalize mask to float32 [0.0, 1.0]
        if mask.dtype == np.uint8:
            mask_float = mask.astype(np.float32) / 255.0
        else:
            mask_float = mask.astype(np.float32)
            if mask_float.max() > 1.0:
                mask_float = mask_float / 255.0

        blurred = cv2.GaussianBlur(mask_float, (feather_radius, feather_radius), 0)
        return np.clip(blurred, 0.0, 1.0).astype(np.float32)

    def bilateral_smooth_edges(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering exclusively to the transition boundary of the mask.

        Calculates the mask perimeter band (dilated minus eroded mask) and blends
        bilateral-filtered pixels into that boundary zone while keeping inner and outer
        areas unaffected.

        Args:
            image: Input image (BGR uint8).
            mask: Binary mask defining the subject region (H, W).

        Returns:
            Image with smoothed perimeter boundaries (BGR uint8).
        """
        # Determine the transition band: (dilated - eroded)
        band_radius = max(3, self.dilation_px // 2)
        if band_radius % 2 == 0:
            band_radius += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_radius * 2 + 1, band_radius * 2 + 1))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary_band = cv2.subtract(dilated, eroded)

        if np.count_nonzero(boundary_band) == 0:
            return image.copy()

        # Apply bilateral filter to the whole frame
        filtered = cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space,
        )

        # Softly feather the boundary band to blend back seamlessly
        band_feather = cv2.GaussianBlur(boundary_band.astype(np.float32) / 255.0, (15, 15), 0)
        band_3c = np.expand_dims(band_feather, axis=2)

        # Composite filtered boundary into the image
        smoothed = (band_3c * filtered.astype(np.float32) + (1.0 - band_3c) * image.astype(np.float32))
        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def alpha_blend(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Composite foreground over background using a feathered alpha mask.

        Args:
            foreground: Swapped subject image (H, W, 3) uint8.
            background: Original source image (H, W, 3) uint8.
            mask: Alpha mask (H, W) or (H, W, 1) either float32 [0.0, 1.0] or uint8 [0, 255].

        Returns:
            Composited image array (H, W, 3) uint8.
        """
        fg_float = foreground.astype(np.float32)
        bg_float = background.astype(np.float32)

        # Standardize mask to (H, W, 1) float32 in [0.0, 1.0]
        if mask.dtype == np.uint8:
            alpha = mask.astype(np.float32) / 255.0
        else:
            alpha = mask.astype(np.float32)
            if alpha.max() > 1.0:
                alpha = alpha / 255.0

        if alpha.ndim == 2:
            alpha = np.expand_dims(alpha, axis=2)
        elif alpha.ndim == 3 and alpha.shape[2] == 3:
            alpha = alpha[:, :, :1]

        blended = fg_float * alpha + bg_float * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def color_match(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Match the color distribution of source to target within the masked region in LAB space.

        Uses Reinhard's statistical color transfer:
            matched_channel = (channel - mean_src) * (std_target / std_src) + mean_target

        Args:
            source: Image to be color-adjusted (swapped face/body) BGR uint8.
            target: Reference color image (original video frame) BGR uint8.
            mask: Binary mask defining the region of interest.

        Returns:
            Color-matched source image BGR uint8.
        """
        # Convert both images to LAB space
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Extract mask boolean array
        mask_bool = (mask > 30) if mask.ndim == 2 else (mask[:, :, 0] > 30)
        pixel_count = np.count_nonzero(mask_bool)

        if pixel_count < 100:
            logger.debug("Mask contains only {} pixels; skipping color matching.", pixel_count)
            return source.copy()

        matched_lab = src_lab.copy()

        # Process L, A, B channels independently
        for c in range(3):
            src_pixels = src_lab[:, :, c][mask_bool]
            tgt_pixels = tgt_lab[:, :, c][mask_bool]

            src_mean = float(np.mean(src_pixels))
            src_std = float(np.std(src_pixels)) + 1e-6

            tgt_mean = float(np.mean(tgt_pixels))
            tgt_std = float(np.std(tgt_pixels)) + 1e-6

            # Scale and shift distribution
            channel_data = src_lab[:, :, c]
            adjusted = ((channel_data - src_mean) * (tgt_std / src_std)) + tgt_mean
            matched_lab[:, :, c] = np.clip(adjusted, 0.0, 255.0)

        matched_bgr = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return matched_bgr

    def remove_artifacts(
        self,
        swapped_frame: np.ndarray,
        original_frame: np.ndarray,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        """Execute full post-processing pipeline for seamless compositing.

        Steps:
            1. Dilate initial mask to encompass face perimeter and transition areas.
            2. Color match the swapped face to original skin/lighting tones in LAB space.
            3. Apply bilateral smoothing across mask boundaries to prevent hard seam edges.
            4. Generate Gaussian-feathered alpha channel.
            5. Alpha blend composited face onto the original background frame.

        Args:
            swapped_frame: Output from diffusion identity transfer (BGR uint8).
            original_frame: Original unmodified video frame (BGR uint8).
            face_mask: Face detection mask (uint8, 0 or 255).

        Returns:
            Final composited frame with invisible seams and matched lighting.
        """
        start_time = time.perf_counter()

        # Step 1: Dilate mask
        t0 = time.perf_counter()
        dilated_mask = self.dilate_mask(face_mask, self.dilation_px)
        t_dilate = (time.perf_counter() - t0) * 1000.0

        # Step 2: Color matching in LAB color space
        t0 = time.perf_counter()
        color_matched = self.color_match(swapped_frame, original_frame, dilated_mask)
        t_color = (time.perf_counter() - t0) * 1000.0

        # Step 3: Bilateral filtering along the boundary band
        t0 = time.perf_counter()
        edge_smoothed = self.bilateral_smooth_edges(color_matched, dilated_mask)
        t_bilateral = (time.perf_counter() - t0) * 1000.0

        # Step 4: Feathered alpha mask creation
        t0 = time.perf_counter()
        feathered_mask = self.create_feathered_mask(dilated_mask, feather_radius=21)
        t_feather = (time.perf_counter() - t0) * 1000.0

        # Step 5: Alpha blending onto original background
        t0 = time.perf_counter()
        composite = self.alpha_blend(edge_smoothed, original_frame, feathered_mask)
        t_blend = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(
            "Artifact removal completed in {:.2f}ms (dilate={:.1f}ms, color={:.1f}ms, "
            "bilateral={:.1f}ms, feather={:.1f}ms, blend={:.1f}ms)",
            total_ms,
            t_dilate,
            t_color,
            t_bilateral,
            t_feather,
            t_blend,
        )
        return composite

    def process_batch(
        self,
        swapped_frames: List[np.ndarray],
        original_frames: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Process a batch of frames through the artifact removal and compositing pipeline.

        Args:
            swapped_frames: List of generated swapped frames.
            original_frames: List of original video frames.
            masks: List of corresponding face/body masks.

        Returns:
            List of seamlessly composited frames.
        """
        count = len(swapped_frames)
        if count != len(original_frames) or count != len(masks):
            raise ValueError(
                f"Batch lengths mismatch: {count} swapped, {len(original_frames)} original, {len(masks)} masks."
            )

        start_batch = time.perf_counter()
        logger.info("Processing artifact removal on batch of {} frames...", count)
        results: List[np.ndarray] = []

        for i in range(count):
            comp = self.remove_artifacts(swapped_frames[i], original_frames[i], masks[i])
            results.append(comp)

        total_elapsed = time.perf_counter() - start_batch
        fps = (count / total_elapsed) if total_elapsed > 0 else 0.0
        logger.info(
            "Completed artifact removal for {} frames in {:.2f}s ({:.1f} fps)",
            count,
            total_elapsed,
            fps,
        )
        return results
