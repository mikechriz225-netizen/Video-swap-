"""
Storage management module for the video swap pipeline.

Provides the StorageManager abstraction layer responsible for managing file paths,
streaming file uploads in 1MB chunks, tracking job storage usage, and cleaning up
job artifacts across upload, processed, and output directories.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import UploadFile
from loguru import logger

# Try importing Settings from core.config, fallback to a local dataclass if not found.
try:
    from core.config import Settings
except ImportError:
    class Settings:  # type: ignore[no-redef]
        """Fallback settings class for path management."""
        def __init__(
            self,
            upload_dir: Optional[Union[str, Path]] = None,
            processed_dir: Optional[Union[str, Path]] = None,
            output_dir: Optional[Union[str, Path]] = None,
            base_dir: Optional[Union[str, Path]] = None,
        ):
            root = Path(base_dir or Path(__file__).resolve().parent.parent)
            data_dir = root / "data"
            self.UPLOAD_DIR = Path(upload_dir or data_dir / "uploads")
            self.PROCESSED_DIR = Path(processed_dir or data_dir / "processed")
            self.OUTPUT_DIR = Path(output_dir or data_dir / "outputs")


class StorageManager:
    """
    File storage abstraction layer for job artifacts.

    Handles creation, resolution, streaming writes, indexing, and cleanup
    of video chunks, reference character images, and final stitched outputs.
    """

    CHUNK_READ_SIZE: int = 1024 * 1024  # 1 MB stream buffer

    def __init__(self, settings: Optional[Any] = None) -> None:
        """
        Initialize StorageManager with configured storage paths.

        Args:
            settings: Optional Settings object containing UPLOAD_DIR,
                      PROCESSED_DIR, and OUTPUT_DIR attributes.
        """
        if settings is None:
            settings = Settings()

        # Resolve paths from attributes (uppercase or lowercase) with safe fallbacks
        base_dir = Path(__file__).resolve().parent.parent / "data"

        upload_path = getattr(
            settings,
            "UPLOAD_DIR",
            getattr(settings, "upload_dir", base_dir / "uploads"),
        )
        processed_path = getattr(
            settings,
            "PROCESSED_DIR",
            getattr(settings, "processed_dir", base_dir / "processed"),
        )
        output_path = getattr(
            settings,
            "OUTPUT_DIR",
            getattr(settings, "output_dir", base_dir / "outputs"),
        )

        self.upload_dir: Path = Path(upload_path).resolve()
        self.processed_dir: Path = Path(processed_path).resolve()
        self.output_dir: Path = Path(output_path).resolve()

        # Ensure base directories exist
        for directory in (self.upload_dir, self.processed_dir, self.output_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.error(f"Failed to create directory {directory}: {exc}")
                raise

        logger.info(
            "StorageManager initialized [upload={}, processed={}, output={}]",
            self.upload_dir,
            self.processed_dir,
            self.output_dir,
        )

    def get_job_dir(self, job_id: str) -> Path:
        """
        Get and ensure existence of the upload directory for a given job.

        Args:
            job_id: Unique job identifier string.

        Returns:
            Path to the job's directory in the upload storage.
        """
        job_dir = self.upload_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def get_chunk_path(self, job_id: str, chunk_index: int, suffix: str = ".mp4") -> Path:
        """
        Get the file path for an uploaded/split source chunk.

        Args:
            job_id: Unique job identifier.
            chunk_index: 0-based chunk sequence index.
            suffix: File extension including dot (default: .mp4).

        Returns:
            Path for the source chunk file.
        """
        job_dir = self.get_job_dir(job_id)
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return job_dir / f"chunk_{chunk_index:04d}{suffix}"

    def get_reference_path(self, job_id: str, ref_index: int = 0, suffix: str = ".png") -> Path:
        """
        Get the file path for a reference character image.

        Args:
            job_id: Unique job identifier.
            ref_index: 0-based reference index (for multiple character angles).
            suffix: Image file extension including dot (default: .png).

        Returns:
            Path for the reference image file.
        """
        ref_dir = self.get_job_dir(job_id) / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return ref_dir / f"ref_{ref_index:04d}{suffix}"

    def get_processed_chunk_path(
        self, job_id: str, chunk_index: int, suffix: str = ".mp4"
    ) -> Path:
        """
        Get the destination file path for a processed/swapped video chunk.

        Args:
            job_id: Unique job identifier.
            chunk_index: 0-based chunk sequence index.
            suffix: File extension including dot (default: .mp4).

        Returns:
            Path for the processed chunk file.
        """
        job_processed_dir = self.processed_dir / job_id
        job_processed_dir.mkdir(parents=True, exist_ok=True)
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return job_processed_dir / f"chunk_{chunk_index:04d}{suffix}"

    def get_output_path(self, job_id: str, suffix: str = ".mp4") -> Path:
        """
        Get the path for the final stitched output video of a completed job.

        Args:
            job_id: Unique job identifier.
            suffix: File extension including dot (default: .mp4).

        Returns:
            Path for the final output video.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return self.output_dir / f"{job_id}{suffix}"

    async def save_upload(self, file: UploadFile, dest: Path) -> Path:
        """
        Stream an uploaded file to disk asynchronously in 1MB chunks.

        Args:
            file: FastAPI UploadFile instance.
            dest: Target destination Path.

        Returns:
            The destination Path upon successful write.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = 0

        logger.debug("Beginning upload stream write to dest={}", dest)
        try:
            with open(dest, "wb") as buffer:
                while True:
                    chunk = await file.read(self.CHUNK_READ_SIZE)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    total_bytes += len(chunk)

            # Reset file pointer in case the caller needs to re-read
            await file.seek(0)
            logger.info("Successfully wrote {} bytes to {}", total_bytes, dest)
            return dest
        except Exception as exc:
            logger.error("Failed writing upload file to {}: {}", dest, exc)
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            raise

    def cleanup_job(self, job_id: str) -> None:
        """
        Remove all directories and files associated with a job across storage tiers.

        Args:
            job_id: Unique job identifier.
        """
        logger.info("Cleaning up storage for job_id={}", job_id)

        # 1. Upload directory
        job_upload_dir = self.upload_dir / job_id
        if job_upload_dir.exists():
            try:
                shutil.rmtree(job_upload_dir, ignore_errors=True)
                logger.debug("Deleted upload dir {}", job_upload_dir)
            except Exception as exc:
                logger.warning("Error deleting upload dir {}: {}", job_upload_dir, exc)

        # 2. Processed chunks directory
        job_processed_dir = self.processed_dir / job_id
        if job_processed_dir.exists():
            try:
                shutil.rmtree(job_processed_dir, ignore_errors=True)
                logger.debug("Deleted processed dir {}", job_processed_dir)
            except Exception as exc:
                logger.warning("Error deleting processed dir {}: {}", job_processed_dir, exc)

        # 3. Final output files matching job_id
        try:
            for item in self.output_dir.glob(f"{job_id}*"):
                if item.is_file():
                    item.unlink(missing_ok=True)
                    logger.debug("Deleted output file {}", item)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    logger.debug("Deleted output dir {}", item)
        except Exception as exc:
            logger.warning("Error deleting output files for {}: {}", job_id, exc)

        logger.info("Storage cleanup finished for job_id={}", job_id)

    def get_job_size_bytes(self, job_id: str) -> int:
        """
        Calculate total disk space in bytes consumed by a job across all storage tiers.

        Args:
            job_id: Unique job identifier.

        Returns:
            Total size in bytes.
        """
        total_size = 0

        def _dir_size(path: Path) -> int:
            size = 0
            if not path.exists():
                return 0
            if path.is_file():
                return path.stat().st_size
            for root, _, files in os.walk(path):
                for f in files:
                    file_path = os.path.join(root, f)
                    try:
                        if not os.path.islink(file_path):
                            size += os.path.getsize(file_path)
                    except OSError:
                        pass
            return size

        # Upload dir
        total_size += _dir_size(self.upload_dir / job_id)
        # Processed dir
        total_size += _dir_size(self.processed_dir / job_id)
        # Output files
        for output_item in self.output_dir.glob(f"{job_id}*"):
            total_size += _dir_size(output_item)

        return total_size

    def list_chunks(self, job_id: str, directory: str = "upload") -> list[Path]:
        """
        List all chunk video files for a given job, ordered by chunk index.

        Args:
            job_id: Unique job identifier.
            directory: Tier to list from ('upload' or 'processed').

        Returns:
            Ordered list of chunk file paths.
        """
        if directory == "upload":
            target_dir = self.upload_dir / job_id
        elif directory == "processed":
            target_dir = self.processed_dir / job_id
        else:
            raise ValueError(f"Invalid directory '{directory}', must be 'upload' or 'processed'")

        if not target_dir.exists() or not target_dir.is_dir():
            return []

        pattern = re.compile(r"chunk_(\d+)")
        chunks_with_index: list[tuple[int, Path]] = []

        for item in target_dir.iterdir():
            if item.is_file() and item.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".ts"}:
                match = pattern.search(item.name)
                if match:
                    idx = int(match.group(1))
                    chunks_with_index.append((idx, item))
                else:
                    # Fallback for chunk files without regex match
                    chunks_with_index.append((999999, item))

        chunks_with_index.sort(key=lambda x: x[0])
        return [path for _, path in chunks_with_index]
