"""VRAM management and Out-Of-Memory (OOM) protection utilities.

This module provides GPU memory monitoring, automatic cache cleanup,
context-managed VRAM guarding, device selection, and model offloading
for deep learning inference pipelines.
"""

from __future__ import annotations

import gc
import sys
from typing import Any, Dict, Optional

import torch
from loguru import logger

__all__ = [
    "get_vram_info",
    "clear_vram",
    "check_vram_threshold",
    "VRAMGuard",
    "move_to_device",
    "offload_to_cpu",
    "select_optimal_device",
]


def get_vram_info(device_id: int = 0) -> Dict[str, float]:
    """Retrieve current VRAM allocation statistics for a CUDA device.

    Args:
        device_id: CUDA device index (default: 0).

    Returns:
        Dict containing:
            - 'total_mb': Total physical VRAM in megabytes.
            - 'used_mb': Currently allocated/reserved VRAM in megabytes.
            - 'free_mb': Available unallocated VRAM in megabytes.
            - 'utilization_percent': Percentage of total VRAM currently used.
    """
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available on this system. Returning zeroed VRAM stats for device {}",
            device_id,
        )
        return {
            "total_mb": 0.0,
            "used_mb": 0.0,
            "free_mb": 0.0,
            "utilization_percent": 0.0,
        }

    device_count = torch.cuda.device_count()
    if device_id < 0 or device_id >= device_count:
        logger.error(
            "Invalid device_id {}. System has {} CUDA device(s).",
            device_id,
            device_count,
        )
        raise ValueError(
            f"Device ID {device_id} is out of range [0, {device_count - 1}]."
        )

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        total_mb = float(total_bytes) / (1024.0 * 1024.0)
        free_mb = float(free_bytes) / (1024.0 * 1024.0)
        used_mb = total_mb - free_mb
        utilization_percent = (used_mb / total_mb * 100.0) if total_mb > 0.0 else 0.0

        stats = {
            "total_mb": round(total_mb, 2),
            "used_mb": round(used_mb, 2),
            "free_mb": round(free_mb, 2),
            "utilization_percent": round(utilization_percent, 2),
        }
        logger.debug(
            "Device cuda:{}: Total={:.1f}MB, Used={:.1f}MB, Free={:.1f}MB ({:.1f}%)",
            device_id,
            stats["total_mb"],
            stats["used_mb"],
            stats["free_mb"],
            stats["utilization_percent"],
        )
        return stats
    except Exception as exc:
        logger.exception("Failed to query VRAM info for device cuda:{}: {}", device_id, exc)
        return {
            "total_mb": 0.0,
            "used_mb": 0.0,
            "free_mb": 0.0,
            "utilization_percent": 0.0,
        }


def clear_vram(device_id: int = 0) -> float:
    """Clear cached GPU memory and trigger garbage collection.

    Calls `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, and Python's
    `gc.collect()`. Calculates and logs the amount of memory freed.

    Args:
        device_id: CUDA device index to clear (default: 0).

    Returns:
        Amount of VRAM freed in megabytes (float).
    """
    if not torch.cuda.is_available():
        collected = gc.collect()
        logger.debug("CUDA unavailable; ran CPU gc.collect() (reclaimed {} objects)", collected)
        return 0.0

    before_info = get_vram_info(device_id)
    free_before = before_info["free_mb"]

    gc.collect()
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception as exc:
        logger.warning("Error during torch.cuda.empty_cache on cuda:{}: {}", device_id, exc)

    after_info = get_vram_info(device_id)
    free_after = after_info["free_mb"]
    freed_mb = max(0.0, free_after - free_before)

    logger.info(
        "VRAM cleanup on cuda:{}: Freed {:.2f} MB (Free before: {:.1f} MB -> after: {:.1f} MB)",
        device_id,
        freed_mb,
        free_before,
        free_after,
    )
    return freed_mb


def check_vram_threshold(required_mb: float, device_id: int = 0) -> bool:
    """Check if the specified device has sufficient unallocated VRAM.

    Args:
        required_mb: Amount of VRAM required in megabytes.
        device_id: CUDA device index.

    Returns:
        True if free VRAM meets or exceeds required_mb, False otherwise.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. VRAM check for {:.1f} MB returning False.", required_mb)
        return False

    vram_info = get_vram_info(device_id)
    has_enough = vram_info["free_mb"] >= required_mb
    if not has_enough:
        logger.warning(
            "VRAM threshold check failed on cuda:{}: required {:.1f} MB, only {:.1f} MB free.",
            device_id,
            required_mb,
            vram_info["free_mb"],
        )
    else:
        logger.debug(
            "VRAM threshold satisfied on cuda:{}: {:.1f} MB available (required {:.1f} MB).",
            device_id,
            vram_info["free_mb"],
            required_mb,
        )
    return has_enough


class VRAMGuard:
    """Context manager for tracking memory deltas and guaranteeing VRAM cleanup.

    On enter, records baseline memory. On exit, logs memory delta, invokes
    `clear_vram()`, and reports net change.
    """

    def __init__(self, device_id: int = 0, tag: str = "Block") -> None:
        """Initialize the VRAMGuard context manager.

        Args:
            device_id: CUDA device index to monitor.
            tag: Descriptive label for the guarded code block.
        """
        self.device_id = device_id
        self.tag = tag
        self.initial_info: Dict[str, float] = {}

    def __enter__(self) -> "VRAMGuard":
        """Record initial VRAM state."""
        self.initial_info = get_vram_info(self.device_id)
        logger.info(
            "[{}] Entering VRAMGuard (cuda:{}): Used={:.1f}MB, Free={:.1f}MB",
            self.tag,
            self.device_id,
            self.initial_info["used_mb"],
            self.initial_info["free_mb"],
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Execute cleanup on exit and log memory delta."""
        if exc_type is not None:
            logger.error(
                "[{}] Exception occurred within VRAMGuard: {} ({})",
                self.tag,
                exc_type.__name__,
                exc_val,
            )

        mid_info = get_vram_info(self.device_id)
        delta_mb = mid_info["used_mb"] - self.initial_info["used_mb"]
        logger.debug(
            "[{}] Peak delta during block execution: {:+.1f} MB",
            self.tag,
            delta_mb,
        )

        freed_mb = clear_vram(self.device_id)
        final_info = get_vram_info(self.device_id)
        net_delta = final_info["used_mb"] - self.initial_info["used_mb"]

        logger.info(
            "[{}] Exited VRAMGuard (cuda:{}): Freed={:.1f}MB, Net delta={:+.1f}MB (Final Free={:.1f}MB)",
            self.tag,
            self.device_id,
            freed_mb,
            net_delta,
            final_info["free_mb"],
        )
        # Do not suppress exceptions
        return False


def move_to_device(
    model: torch.nn.Module,
    device_id: int = 0,
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """Move a PyTorch model to GPU with specified dtype and log memory consumption.

    Args:
        model: PyTorch module/model to move.
        device_id: Target CUDA device index.
        dtype: Desired torch floating-point precision (default: torch.float16).

    Returns:
        The moved model instance.
    """
    if not isinstance(model, torch.nn.Module):
        logger.warning(
            "Passed object is not an instance of torch.nn.Module (got {}). Attempting .to() anyway.",
            type(model),
        )

    if torch.cuda.is_available():
        target_device = f"cuda:{device_id}"
        before_info = get_vram_info(device_id)
        logger.info(
            "Moving model {} to {} with dtype {}",
            model.__class__.__name__,
            target_device,
            dtype,
        )
        model = model.to(device=target_device, dtype=dtype)
        after_info = get_vram_info(device_id)
        vram_increase = after_info["used_mb"] - before_info["used_mb"]
        logger.info(
            "Model {} loaded on {}. VRAM consumption increased by {:.1f} MB (Used: {:.1f} MB)",
            model.__class__.__name__,
            target_device,
            vram_increase,
            after_info["used_mb"],
        )
    else:
        logger.warning(
            "CUDA not available. Keeping model {} on CPU (dtype set to float32).",
            model.__class__.__name__,
        )
        model = model.to(device="cpu", dtype=torch.float32)

    return model


def offload_to_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """Move a PyTorch model back to CPU and clear GPU memory caches.

    Args:
        model: PyTorch module/model to offload.

    Returns:
        The CPU-offloaded model.
    """
    model_name = getattr(model, "__class__", type(model)).__name__
    logger.info("Offloading model {} to CPU", model_name)
    try:
        model = model.to("cpu")
    except Exception as exc:
        logger.warning("Failed to move model {} to CPU: {}", model_name, exc)

    if torch.cuda.is_available():
        clear_vram()

    return model


def select_optimal_device(required_mb: float = 4000.0) -> int:
    """Scan all available CUDA devices and select the one with most free VRAM.

    Args:
        required_mb: Minimum required free VRAM in megabytes.

    Returns:
        Selected device ID integer (0 if CUDA is not available or fallback).
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; default device 0 selected (CPU fallback mode).")
        return 0

    device_count = torch.cuda.device_count()
    if device_count == 0:
        logger.warning("No CUDA devices detected. Defaulting to 0.")
        return 0

    best_device = 0
    max_free_mb = -1.0
    qualifying_devices = []

    for dev_idx in range(device_count):
        info = get_vram_info(dev_idx)
        free_mb = info["free_mb"]
        logger.debug(
            "Scanning cuda:{}: Free={:.1f} MB, Total={:.1f} MB",
            dev_idx,
            free_mb,
            info["total_mb"],
        )
        if free_mb > max_free_mb:
            max_free_mb = free_mb
            best_device = dev_idx
        if free_mb >= required_mb:
            qualifying_devices.append((dev_idx, free_mb))

    if qualifying_devices:
        qualifying_devices.sort(key=lambda item: item[1], reverse=True)
        chosen_id, chosen_free = qualifying_devices[0]
        logger.info(
            "Selected optimal device cuda:{} ({:.1f} MB free >= {:.1f} MB requested)",
            chosen_id,
            chosen_free,
            required_mb,
        )
        return chosen_id

    logger.warning(
        "No CUDA device meets required threshold of {:.1f} MB. Selecting device with most free VRAM (cuda:{} with {:.1f} MB free).",
        required_mb,
        best_device,
        max_free_mb,
    )
    return best_device
