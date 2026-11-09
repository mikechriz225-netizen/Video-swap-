"""Face and character detection and segmentation module using InsightFace.

This module provides detection, 512-d ArcFace embedding extraction,
landmark-based facial mask generation, and person/body segmentation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

__all__ = ["FaceDetector"]


class FaceDetector:
    """InsightFace-based face analysis and mask generation system.

    Models are loaded lazily upon first invocation to preserve memory until needed.
    """

    def __init__(
        self,
        device_id: int = 0,
        device: Optional[Union[str, int]] = None,
    ) -> None:
        """Initialize the FaceDetector.

        Args:
            device_id: CUDA device index for inference execution (default: 0).
            device: Optional device string (e.g. 'cuda:0', 'cpu') or int override.
        """
        if device is not None:
            if isinstance(device, int):
                device_id = device
            elif isinstance(device, str):
                if ":" in device:
                    try:
                        device_id = int(device.split(":")[-1])
                    except ValueError:
                        device_id = 0
                elif device.isdigit():
                    device_id = int(device)
                elif device.lower() == "cpu":
                    device_id = -1

        self.device_id = device_id
        self.device = f"cuda:{device_id}" if device_id >= 0 else "cpu"
        self._app: Optional[Any] = None
        self._initialized = False

    def _init_model(self) -> None:
        """Initialize and prepare the InsightFace FaceAnalysis application ('buffalo_l')."""
        if self._initialized and self._app is not None:
            return

        start_time = time.perf_counter()
        logger.info("Initializing InsightFace 'buffalo_l' on device {}...", self.device)

        try:
            from insightface.app import FaceAnalysis
            import torch

            if torch.cuda.is_available() and self.device_id >= 0:
                providers = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": self.device_id,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True,
                        },
                    ),
                    "CPUExecutionProvider",
                ]
            else:
                logger.warning("CUDA unavailable for InsightFace; falling back to CPUExecutionProvider.")
                providers = ["CPUExecutionProvider"]

            app = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=["detection", "recognition", "genderage", "landmark_2d_106"],
            )
            app.prepare(ctx_id=self.device_id if self.device_id >= 0 else -1, det_size=(640, 640))
            self._app = app
            self._initialized = True
            elapsed_s = time.perf_counter() - start_time
            logger.info("InsightFace successfully initialized in {:.2f}s", elapsed_s)
        except Exception as exc:
            logger.exception("Failed to initialize InsightFace FaceAnalysis: {}", exc)
            self._app = None
            self._initialized = False
            raise

    def _get_model(self) -> Any:
        """Retrieve model instance with lazy loading guarantee."""
        if not self._initialized or self._app is None:
            self._init_model()
        return self._app

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect all faces in a frame and extract biometrics and landmarks.

        Args:
            frame: Input video frame as BGR uint8 numpy array of shape (H, W, 3).

        Returns:
            List of dictionaries, each containing:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates.
                - 'kps': (5, 2) facial keypoints (eyes, nose, mouth corners).
                - 'landmarks': (N, 2) dense facial landmarks (e.g. 106-point mesh).
                - 'embedding': 512-dimensional normalized face embedding.
                - 'age': Estimated age (int).
                - 'gender': Estimated gender (0 for female, 1 for male).
                - 'det_score': Confidence score of detection.
                - 'area': Pixel area of the face bounding box.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected numpy ndarray for frame, got {type(frame)}")

        start_time = time.perf_counter()
        app = self._get_model()

        try:
            raw_faces = app.get(frame)
        except Exception as exc:
            logger.exception("Error during InsightFace inference: {}", exc)
            return []

        results: List[Dict[str, Any]] = []
        for face in raw_faces:
            bbox = face.bbox.astype(float).tolist()
            x1, y1, x2, y2 = bbox
            area = max(0.0, (x2 - x1) * (y2 - y1))

            landmarks = getattr(face, "landmark_2d_106", None)
            if landmarks is None:
                landmarks = getattr(face, "landmark_3d_68", None)
            if landmarks is None:
                landmarks = face.kps

            embedding = getattr(face, "embedding", None)
            if embedding is not None and isinstance(embedding, np.ndarray):
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    embedding = embedding / norm

            age = int(face.age) if hasattr(face, "age") and face.age is not None else None
            gender = int(face.gender) if hasattr(face, "gender") and face.gender is not None else None
            score = float(face.det_score) if hasattr(face, "det_score") else 1.0

            results.append({
                "bbox": bbox,
                "kps": face.kps,
                "landmarks": np.asarray(landmarks, dtype=np.float32),
                "embedding": embedding,
                "age": age,
                "gender": gender,
                "det_score": score,
                "area": area,
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug("Detected {} face(s) in {:.2f} ms", len(results), elapsed_ms)
        return results

    def get_primary_face(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Identify and return the primary (largest area) face in the frame.

        Args:
            frame: Input image array.

        Returns:
            Dictionary with face metadata and embeddings, or None if no faces detected.
        """
        faces = self.detect_faces(frame)
        if not faces:
            logger.debug("No faces found in frame.")
            return None

        primary = max(faces, key=lambda f: f["area"])
        logger.debug(
            "Primary face selected (area={:.1f}px, score={:.2f}, bbox={})",
            primary["area"],
            primary["det_score"],
            [round(v, 1) for v in primary["bbox"]],
        )
        return primary

    def extract_face_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512-dimensional normalized face embedding for the primary subject.

        Args:
            frame: Input image containing the subject.

        Returns:
            512-d unit-norm numpy array or None if no face detected.
        """
        primary = self.get_primary_face(frame)
        if primary is None or primary.get("embedding") is None:
            logger.warning("Could not extract face embedding: no primary face found.")
            return None
        return primary["embedding"]

    def extract_embedding(
        self,
        reference_images: Union[List[Union[str, Path, np.ndarray]], np.ndarray, str, Path],
    ) -> Dict[str, Any]:
        """Extract reference embeddings from a list of image paths or image arrays.

        Compatible with pipeline orchestrator calling patterns.

        Args:
            reference_images: List of image paths (str or Path) or numpy arrays.

        Returns:
            Dict containing 'embedding', 'paths', and extracted reference features.
        """
        if not isinstance(reference_images, list):
            reference_images = [reference_images]

        embeddings: List[np.ndarray] = []
        paths: List[str] = []

        for ref in reference_images:
            if isinstance(ref, (str, Path)):
                paths.append(str(ref))
                frame = cv2.imread(str(ref))
                if frame is None:
                    logger.warning("Failed to load reference image from {}", ref)
                    continue
            elif isinstance(ref, np.ndarray):
                paths.append("<ndarray>")
                frame = ref
            else:
                continue

            emb = self.extract_face_embedding(frame)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.warning("No face embeddings extracted from reference images.")
            return {"embedding": None, "paths": paths}

        stacked = np.stack(embeddings, axis=0)
        mean_emb = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            mean_emb /= norm

        return {
            "embedding": mean_emb,
            "embeddings": embeddings,
            "paths": paths,
        }

    def generate_face_mask(
        self,
        frame: np.ndarray,
        face_info: Dict[str, Any],
        dilation_px: int = 15,
    ) -> np.ndarray:
        """Create a smooth binary mask covering the face region with convex hull and dilation.

        Args:
            frame: Source video frame (used for shape reference).
            face_info: Face dictionary from `detect_faces`.
            dilation_px: Number of pixels to expand mask boundaries (default: 15).

        Returns:
            Binary mask as uint8 numpy array with values 0 or 255 of shape (H, W).
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        landmarks = face_info.get("landmarks")
        bbox = face_info.get("bbox")

        if landmarks is not None and len(landmarks) >= 3:
            pts = np.asarray(landmarks, dtype=np.int32).reshape((-1, 2))
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)

            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                face_height = max(1, y2 - y1)
                forehead_y1 = max(0, y1 - int(face_height * 0.1))
                forehead_y2 = y1 + int(face_height * 0.35)
                forehead_x1 = max(0, x1 + int((x2 - x1) * 0.1))
                forehead_x2 = min(w, x2 - int((x2 - x1) * 0.1))

                if forehead_y2 > forehead_y1 and forehead_x2 > forehead_x1:
                    forehead_ellipse_center = (
                        (forehead_x1 + forehead_x2) // 2,
                        (forehead_y1 + forehead_y2) // 2,
                    )
                    forehead_axes = (
                        (forehead_x2 - forehead_x1) // 2,
                        (forehead_y2 - forehead_y1) // 2,
                    )
                    cv2.ellipse(
                        mask,
                        forehead_ellipse_center,
                        forehead_axes,
                        0,
                        0,
                        360,
                        255,
                        -1,
                    )
        elif bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        else:
            logger.warning("No landmarks or bbox in face_info to generate face mask.")
            return mask

        if dilation_px > 0:
            ksize = 2 * dilation_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask = cv2.dilate(mask, kernel, iterations=1)

        logger.debug(
            "Generated face mask (active_pixels={}, dilation_px={})",
            int(np.count_nonzero(mask)),
            dilation_px,
        )
        return mask

    def generate_body_mask(self, frame: np.ndarray) -> np.ndarray:
        """Generate a rough body / foreground mask for character compositing.

        Uses face location priors combined with Otsu thresholding and GrabCut
        segmentation to delineate the person from the background.

        Args:
            frame: Input video frame (H, W, 3).

        Returns:
            Binary mask (uint8, 0 or 255) of shape (H, W).
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        primary = self.get_primary_face(frame)
        if primary is not None and primary.get("bbox") is not None:
            x1, y1, x2, y2 = [int(v) for v in primary["bbox"]]
            face_w = max(10, x2 - x1)
            face_h = max(10, y2 - y1)

            roi_x1 = max(0, x1 - int(face_w * 1.2))
            roi_y1 = max(0, y1 - int(face_h * 0.3))
            roi_x2 = min(w, x2 + int(face_w * 1.2))
            roi_y2 = h
            rect = (roi_x1, roi_y1, max(1, roi_x2 - roi_x1), max(1, roi_y2 - roi_y1))
        else:
            rect = (int(w * 0.15), int(h * 0.1), int(w * 0.7), int(h * 0.9))

        try:
            grabcut_mask = np.zeros((h, w), dtype=np.uint8)
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)

            cv2.grabCut(
                frame,
                grabcut_mask,
                rect,
                bgd_model,
                fgd_model,
                iterCount=3,
                mode=cv2.GC_INIT_WITH_RECT,
            )

            mask = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)

            ksize = 9
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            logger.debug(
                "Generated body mask via GrabCut (foreground_pixels={})",
                int(np.count_nonzero(mask)),
            )
        except Exception as exc:
            logger.warning("GrabCut segmentation failed: {}. Falling back to thresholding.", exc)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            rx, ry, rw, rh = rect
            roi_mask[ry : ry + rh, rx : rx + rw] = 255
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.bitwise_and(thresh, roi_mask)

        return mask

    def unload(self) -> None:
        """Release InsightFace model resources from memory."""
        logger.info("Unloading FaceDetector on {}...", self.device)
        self._app = None
        self._initialized = False
