"""Character identity transfer module using IP-Adapter-FaceID Plus and ControlNet.

Performs reference face identity embedding extraction, multi-modal conditioning,
and latent diffusion generation with pose and depth structure preservation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from pipeline.controlnet import ControlNetProcessor
from pipeline.face_detection import FaceDetector
from pipeline.resolution import compute_latent_dimensions, resize_to_original
from pipeline.vram import VRAMGuard, clear_vram, offload_to_cpu

if TYPE_CHECKING:
    from diffusers import StableDiffusionControlNetPipeline

__all__ = ["IdentityTransfer"]


class IdentityTransfer:
    """Manages identity injection into diffusion pipelines via IP-Adapter-FaceID Plus."""

    DEFAULT_PROMPT = (
        "photorealistic portrait, high quality, 8k uhd, sharp focus, natural skin texture, masterpiece"
    )
    DEFAULT_NEGATIVE_PROMPT = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, deformed eyes, extra limbs"
    )

    def __init__(
        self,
        device_id: int = 0,
        ip_adapter_scale: float = 0.6,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        ip_adapter_repo: str = "h94/IP-Adapter-FaceID",
        ip_adapter_weight: str = "ip-adapter-faceid-plusv2_sd15.bin",
        face_detector: Optional[FaceDetector] = None,
        controlnet_processor: Optional[ControlNetProcessor] = None,
    ) -> None:
        """Initialize the IdentityTransfer module.

        Args:
            device_id: Target CUDA device ID (default: 0).
            ip_adapter_scale: Conditioning strength of IP-Adapter [0.0, 1.0].
            base_model_id: HuggingFace model repo or path for base SD checkpoint.
            ip_adapter_repo: Repository ID holding IP-Adapter weights.
            ip_adapter_weight: Weight file name for IP-Adapter FaceID.
            face_detector: Optional existing FaceDetector instance to reuse.
            controlnet_processor: Optional existing ControlNetProcessor instance to reuse.
        """
        self.device_id = device_id
        self.ip_adapter_scale = ip_adapter_scale
        self.base_model_id = base_model_id
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_weight = ip_adapter_weight

        self.face_detector = face_detector or FaceDetector(device_id=device_id)
        self.controlnet_processor = controlnet_processor or ControlNetProcessor(device_id=device_id)

        self.pipeline: Optional[Any] = None
        self._image_encoder: Optional[Any] = None

        self._device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu"
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _load_models(self) -> Any:
        """Load and configure the Stable Diffusion ControlNet pipeline with IP-Adapter."""
        if self.pipeline is not None:
            return self.pipeline

        with VRAMGuard(self.device_id, tag="LoadIdentityTransferPipeline"):
            start_time = time.perf_counter()
            logger.info("Setting up StableDiffusionControlNetPipeline with base '{}'", self.base_model_id)

            from diffusers import DDIMScheduler, StableDiffusionControlNetPipeline
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

            controlnets = self.controlnet_processor.get_controlnet_models()

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.base_model_id,
                controlnet=controlnets,
                torch_dtype=self._dtype,
                safety_checker=None,
            )

            # Use DDIM scheduler for faster, deterministic sampling
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            # Load CLIP image encoder for FaceID Plus multi-modal conditioning
            logger.info("Loading CLIP Vision model for FaceID Plus...")
            try:
                self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    torch_dtype=self._dtype,
                ).to(self._device)
            except Exception as exc:
                logger.warning(
                    "Could not load ViT-H-14; attempting fallback to standard CLIP vision: {}",
                    exc,
                )
                self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    torch_dtype=self._dtype,
                ).to(self._device)

            # Attach IP-Adapter FaceID Plus weights
            logger.info(
                "Loading IP-Adapter weights from {} ({})",
                self.ip_adapter_repo,
                self.ip_adapter_weight,
            )
            try:
                pipe.load_ip_adapter(
                    self.ip_adapter_repo,
                    subfolder="",
                    weight_name=self.ip_adapter_weight,
                    image_encoder_folder=None,
                )
                pipe.set_ip_adapter_scale(self.ip_adapter_scale)
            except Exception as exc:
                logger.exception("Failed to load IP-Adapter weights into pipeline: {}", exc)
                raise

            pipe = pipe.to(self._device)

            # Memory optimization hooks
            if torch.cuda.is_available():
                try:
                    pipe.enable_vae_slicing()
                    pipe.enable_vae_tiling()
                except Exception as exc:
                    logger.debug("Could not enable VAE slicing/tiling: {}", exc)

            self.pipeline = pipe
            elapsed_s = time.perf_counter() - start_time
            logger.info("IdentityTransfer pipeline assembled successfully in {:.2f}s", elapsed_s)

        return self.pipeline

    def prepare_reference(self, reference_images: List[np.ndarray]) -> Dict[str, Any]:
        """Process reference images to extract face identity embeddings and CLIP visual features.

        Args:
            reference_images: List of source images containing the target character (BGR uint8).

        Returns:
            Dict containing:
                - 'face_embeddings': PyTorch tensor of shape (1, num_faces, 512) or (1, 512).
                - 'face_crops': List of PIL images cropped closely to the face.
                - 'clip_images': List of PIL images formatted for CLIP encoder.
                - 'num_valid': Number of valid face references detected.
        """
        if not reference_images:
            raise ValueError("Reference images list cannot be empty.")

        start_time = time.perf_counter()
        logger.info("Extracting identity reference embeddings from {} image(s)...", len(reference_images))

        valid_embeddings: List[np.ndarray] = []
        face_crops: List[Image.Image] = []

        for idx, img in enumerate(reference_images):
            face = self.face_detector.get_primary_face(img)
            if face is None or face.get("embedding") is None:
                logger.warning("No face detected in reference image #{}", idx)
                continue

            emb = face["embedding"]
            valid_embeddings.append(emb)

            # Crop face bounding box with moderate safety margin
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
            h, w = img.shape[:2]
            pad_w = int((x2 - x1) * 0.25)
            pad_h = int((y2 - y1) * 0.25)

            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(w, x2 + pad_w)
            crop_y2 = min(h, y2 + pad_h)

            crop_bgr = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_bgr.size > 0:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                face_crops.append(Image.fromarray(crop_rgb))

        if not valid_embeddings:
            raise RuntimeError("Failed to detect a valid face in any provided reference image.")

        # Compute centroid embedding across references
        stacked = np.stack(valid_embeddings, axis=0)
        mean_embedding = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 1e-6:
            mean_embedding /= norm

        # Format as torch tensor: (1, 1, 512)
        face_tensor = torch.from_numpy(mean_embedding).float().unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available() and self.device_id >= 0:
            face_tensor = face_tensor.to(self._device)

        reference_data = {
            "face_embeddings": face_tensor,
            "face_crops": face_crops,
            "clip_images": face_crops if face_crops else [Image.new("RGB", (224, 224), (128, 128, 128))],
            "num_valid": len(valid_embeddings),
        }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Prepared reference data from {}/{} images in {:.1f} ms",
            len(valid_embeddings),
            len(reference_images),
            elapsed_ms,
        )
        return reference_data

    def apply_identity(
        self,
        pipeline: Any,
        reference_data: Dict[str, Any],
        controlnet_conditions: Dict[str, Image.Image],
        prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        latent_width: int = 512,
        latent_height: int = 512,
    ) -> Image.Image:
        """Run latent diffusion with IP-Adapter identity injection and ControlNet guidance.

        Args:
            pipeline: Loaded StableDiffusionControlNetPipeline instance.
            reference_data: Output dict from `prepare_reference`.
            controlnet_conditions: Dict with 'pose' and 'depth' PIL Images.
            prompt: Text prompt string.
            negative_prompt: Negative text prompt string.
            num_inference_steps: Denoising steps (default: 30).
            guidance_scale: Classifier-free guidance scale (default: 7.5).
            latent_width: Generated latent width (must be multiple of 8).
            latent_height: Generated latent height (must be multiple of 8).

        Returns:
            Generated PIL Image containing the synthesized character.
        """
        with VRAMGuard(self.device_id, tag="ApplyIdentityInference"):
            start_time = time.perf_counter()

            prompt_str = prompt.strip() if prompt else self.DEFAULT_PROMPT
            neg_prompt_str = negative_prompt.strip() if negative_prompt else self.DEFAULT_NEGATIVE_PROMPT

            pose_img = controlnet_conditions.get("pose")
            depth_img = controlnet_conditions.get("depth")

            if pose_img is None or depth_img is None:
                raise ValueError("Both 'pose' and 'depth' conditions are required in controlnet_conditions.")

            # Resize conditioning images to target latent resolution
            pose_resized = pose_img.resize((latent_width, latent_height), Image.Resampling.LANCZOS)
            depth_resized = depth_img.resize((latent_width, latent_height), Image.Resampling.LANCZOS)
            cond_images = [pose_resized, depth_resized]

            face_embeddings = reference_data["face_embeddings"]
            clip_images = reference_data.get("clip_images")

            logger.info(
                "Running diffusion: steps={}, guidance={:.1f}, resolution={}x{}",
                num_inference_steps,
                guidance_scale,
                latent_width,
                latent_height,
            )

            # Invoke pipeline with ip_adapter conditioning arguments
            inference_kwargs: Dict[str, Any] = {
                "prompt": prompt_str,
                "negative_prompt": neg_prompt_str,
                "image": cond_images,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": latent_width,
                "height": latent_height,
            }

            # Pass face identity embeddings to IP-Adapter
            if hasattr(pipeline, "set_ip_adapter_scale"):
                pipeline.set_ip_adapter_scale(self.ip_adapter_scale)

            # Different versions of diffusers handle ip_adapter_image vs ip_adapter_image_embeds
            if "ip_adapter_image_embeds" in pipeline.__call__.__code__.co_varnames:
                inference_kwargs["ip_adapter_image_embeds"] = face_embeddings
            elif "ip_adapter_image" in pipeline.__call__.__code__.co_varnames:
                inference_kwargs["ip_adapter_image"] = clip_images[0] if clip_images else pose_resized
            else:
                # Standard fallback for FaceID Plus
                inference_kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": None}
                inference_kwargs["ip_adapter_image"] = clip_images[0] if clip_images else pose_resized

            output = pipeline(**inference_kwargs)
            generated_image: Image.Image = output.images[0]

            elapsed_s = time.perf_counter() - start_time
            logger.info("Frame generation completed in {:.2f}s", elapsed_s)
            return generated_image

    def transfer_frame(
        self,
        source_frame: np.ndarray,
        reference_data: Dict[str, Any],
        controlnet_conditions: Dict[str, Image.Image],
        config: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Execute end-to-end identity transfer for a single video frame.

        Args:
            source_frame: Original video frame (BGR uint8).
            reference_data: Character identity embeddings and crops.
            controlnet_conditions: Structural conditioning images ('pose', 'depth').
            config: Optional configuration dictionary with runtime options:
                - 'prompt': Custom text prompt.
                - 'negative_prompt': Custom negative prompt.
                - 'num_inference_steps': Number of denoising steps (default: 30).
                - 'guidance_scale': CFG scale (default: 7.5).
                - 'max_latent': Maximum latent dimension (default: 512).

        Returns:
            Transferred frame as BGR uint8 numpy array matching source_frame resolution.
        """
        cfg = config or {}
        orig_h, orig_w = source_frame.shape[:2]
        max_latent = int(cfg.get("max_latent", 512))

        latent_w, latent_h = compute_latent_dimensions(orig_w, orig_h, max_latent=max_latent)
        pipe = self._load_models()

        generated_pil = self.apply_identity(
            pipeline=pipe,
            reference_data=reference_data,
            controlnet_conditions=controlnet_conditions,
            prompt=cfg.get("prompt", ""),
            negative_prompt=cfg.get("negative_prompt", ""),
            num_inference_steps=int(cfg.get("num_inference_steps", 30)),
            guidance_scale=float(cfg.get("guidance_scale", 7.5)),
            latent_width=latent_w,
            latent_height=latent_h,
        )

        # Convert generated PIL Image (RGB) back to OpenCV BGR array
        rgb_array = np.array(generated_pil, dtype=np.uint8)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        # Upscale back to source video resolution
        restored_frame = resize_to_original(bgr_array, orig_w, orig_h)
        return restored_frame

    def unload(self) -> None:
        """Release pipeline models from GPU and free VRAM."""
        logger.info("Unloading IdentityTransfer models...")
        if self.pipeline is not None:
            offload_to_cpu(self.pipeline.unet)
            offload_to_cpu(self.pipeline.vae)
            offload_to_cpu(self.pipeline.text_encoder)
            self.pipeline = None

        if self._image_encoder is not None:
            offload_to_cpu(self._image_encoder)
            self._image_encoder = None

        self.controlnet_processor.unload()
        clear_vram(self.device_id)
        logger.info("IdentityTransfer completely unloaded.")
