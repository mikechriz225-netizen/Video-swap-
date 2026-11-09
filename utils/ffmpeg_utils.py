"""FFmpeg utility functions for video manipulation.

Provides production-ready wrappers around ffprobe and ffmpeg for:
- Video probing (duration, fps, resolution, codecs, bitrate)
- I-frame / keyframe detection
- Lossless keyframe-aligned video chunking
- Frame extraction to numpy arrays via ffmpeg pipe
- Frame-to-video encoding with optional audio muxing
- Audio extraction and acrossfade re-muxing
- Lossless chunk stitching via concat demuxer
"""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import numpy as np


def _run_command(cmd: list[str], description: str = "") -> subprocess.CompletedProcess[str]:
    """Execute a subprocess command with comprehensive error logging and formatting.

    Args:
        cmd: List of command arguments.
        description: Brief description for context logging.

    Returns:
        CompletedProcess instance on success.

    Raises:
        RuntimeError: If process exits with non-zero status code or executable is missing.
    """
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
    desc_prefix = f"[{description}] " if description else ""
    logger.debug(f"{desc_prefix}Running command: {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        err_msg = f"{desc_prefix}Required executable not found while running: {cmd[0]}"
        logger.error(err_msg)
        raise RuntimeError(err_msg) from exc

    if result.returncode != 0:
        err_msg = (
            f"{desc_prefix}Command failed with exit code {result.returncode}.\n"
            f"Command: {cmd_str}\n"
            f"Stderr: {result.stderr.strip()}\n"
            f"Stdout: {result.stdout.strip()}"
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    return result


def probe_video(video_path: Path) -> dict[str, Any]:
    """Run ffprobe to extract video metadata.

    Args:
        video_path: Path to the target video file.

    Returns:
        Dict containing duration (float), width (int), height (int), fps (float),
        codec (str), audio_codec (str or None), and bitrate (int or None).

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If probe fails or required video streams are missing.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,bit_rate",
        "-of",
        "json",
        str(video_path.resolve()),
    ]

    try:
        result = _run_command(cmd, description=f"probe_video({video_path.name})")
        probe_data = json.loads(result.stdout)
    except Exception as exc:
        raise ValueError(f"Failed to probe video at {video_path}: {exc}") from exc

    streams = probe_data.get("streams", [])
    format_info = probe_data.get("format", {})

    video_stream: dict[str, Any] | None = next(
        (s for s in streams if s.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    audio_stream: dict[str, Any] | None = next(
        (s for s in streams if s.get("codec_type") == "audio"),
        None,
    )

    # Resolve frame rate (r_frame_rate or avg_frame_rate)
    fps_str = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate") or "30/1"
    try:
        if "/" in fps_str:
            num, den = map(float, fps_str.split("/"))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str)
    except (ValueError, ZeroDivisionError):
        logger.warning(f"Could not parse fps string '{fps_str}', defaulting to 30.0")
        fps = 30.0

    # Resolve duration
    duration_val = format_info.get("duration") or video_stream.get("duration")
    if duration_val is None:
        raise ValueError(f"Could not determine duration for video: {video_path}")
    try:
        duration = float(duration_val)
    except ValueError as exc:
        raise ValueError(f"Invalid duration value '{duration_val}' for {video_path}") from exc

    # Resolve bitrate
    bitrate_val = format_info.get("bit_rate") or video_stream.get("bit_rate")
    bitrate: int | None = None
    if bitrate_val is not None:
        try:
            bitrate = int(bitrate_val)
        except ValueError:
            bitrate = None

    result_dict = {
        "duration": duration,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "codec": video_stream.get("codec_name", "unknown"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "bitrate": bitrate,
    }

    logger.debug(f"Probed {video_path.name}: {result_dict}")
    return result_dict


def get_video_duration(video_path: Path) -> float:
    """Quick duration probe for a video file.

    Args:
        video_path: Path to the target video.

    Returns:
        Duration in seconds as float.

    Raises:
        ValueError: If duration cannot be determined.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path.resolve()),
    ]
    try:
        result = _run_command(cmd, description=f"get_video_duration({video_path.name})")
        val = result.stdout.strip()
        if not val:
            raise ValueError(f"ffprobe returned empty duration for {video_path}")
        return float(val)
    except Exception as exc:
        logger.warning(f"Quick duration check failed, falling back to full probe: {exc}")
        return probe_video(video_path)["duration"]


def find_keyframes(video_path: Path) -> list[float]:
    """Find all I-frame (keyframe) timestamps in a video.

    Args:
        video_path: Path to the video file.

    Returns:
        Sorted list of float seconds representing keyframe timestamps.

    Raises:
        ValueError: If ffprobe command fails.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Scanning keyframes for {video_path.name}...")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "csv=p=0",
        str(video_path.resolve()),
    ]

    try:
        result = _run_command(cmd, description=f"find_keyframes({video_path.name})")
    except Exception as exc:
        raise ValueError(f"Failed to extract keyframes from {video_path}: {exc}") from exc

    keyframes: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            keyframes.append(float(line))
        except ValueError:
            continue

    keyframes = sorted(list(set(keyframes)))
    if not keyframes or keyframes[0] > 0.05:
        keyframes.insert(0, 0.0)

    logger.info(
        f"Found {len(keyframes)} keyframes for {video_path.name} (first: {keyframes[0]:.2f}s, last: {keyframes[-1]:.2f}s)"
    )
    return keyframes


def chunk_video_at_keyframes(
    video_path: Path,
    output_dir: Path,
    min_chunk_sec: float = 5.0,
    max_chunk_sec: float = 10.0,
) -> list[Path]:
    """Split a video into lossless chunks at keyframe boundaries.

    Analyzes keyframe timestamps and computes optimal split points between
    min_chunk_sec and max_chunk_sec to prevent frame corruption while keeping
    chunks uniform. Uses `ffmpeg -ss -to` with `-c copy`.

    Args:
        video_path: Source video path.
        output_dir: Directory where video chunks will be written.
        min_chunk_sec: Minimum desired chunk duration in seconds.
        max_chunk_sec: Maximum desired chunk duration in seconds.

    Returns:
        List of Paths pointing to generated video chunks.

    Raises:
        ValueError: If min_chunk_sec >= max_chunk_sec or video cannot be split.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_chunk_sec <= 0 or max_chunk_sec <= min_chunk_sec:
        raise ValueError(
            f"Invalid duration constraints: min_chunk_sec={min_chunk_sec}, max_chunk_sec={max_chunk_sec}"
        )

    total_duration = get_video_duration(video_path)
    logger.info(
        f"Chunking {video_path.name} (total_duration={total_duration:.2f}s, range=[{min_chunk_sec}s, {max_chunk_sec}s])"
    )

    # If the video is shorter than the max chunk size, return single chunk
    if total_duration <= max_chunk_sec:
        dest = output_dir / f"chunk_0000{video_path.suffix}"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            "0.000",
            "-to",
            f"{total_duration:.3f}",
            "-i",
            str(video_path.resolve()),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "1",
            str(dest.resolve()),
        ]
        _run_command(cmd, description="chunk_video (single chunk)")
        logger.info(f"Video shorter than max chunk duration. Created single chunk: {dest.name}")
        return [dest]

    keyframes = find_keyframes(video_path)
    target_duration = (min_chunk_sec + max_chunk_sec) / 2.0

    split_points: list[float] = [0.0]
    current_start = 0.0

    while current_start < total_duration:
        # Candidate keyframes between [start + min, start + max]
        candidates = [
            k for k in keyframes if (current_start + min_chunk_sec) <= k <= (current_start + max_chunk_sec)
        ]

        if candidates:
            # Pick candidate closest to optimal midpoint
            next_split = min(candidates, key=lambda t: abs((t - current_start) - target_duration))
        else:
            # No keyframe within ideal window: search beyond max, or earlier before min
            later_kfs = [k for k in keyframes if k > current_start + max_chunk_sec]
            earlier_kfs = [k for k in keyframes if current_start < k < current_start + min_chunk_sec]

            if later_kfs:
                next_split = later_kfs[0]
            elif earlier_kfs:
                next_split = earlier_kfs[-1]
            else:
                next_split = total_duration

        # Prevent an orphaned tiny tail chunk
        remaining = total_duration - next_split
        if 0 < remaining < (min_chunk_sec * 0.6):
            logger.debug(
                f"Remaining duration {remaining:.2f}s is below minimum threshold; extending current chunk to end."
            )
            next_split = total_duration

        split_points.append(next_split)
        current_start = next_split

        if current_start >= total_duration:
            break

    # Build unique strictly increasing split boundaries
    cleaned_splits = [split_points[0]]
    for pt in split_points[1:]:
        if pt > cleaned_splits[-1]:
            cleaned_splits.append(pt)
    if cleaned_splits[-1] < total_duration:
        cleaned_splits.append(total_duration)

    chunk_paths: list[Path] = []
    logger.info(f"Calculated {len(cleaned_splits) - 1} chunks based on keyframe splits.")

    for idx in range(len(cleaned_splits) - 1):
        start_sec = cleaned_splits[idx]
        end_sec = cleaned_splits[idx + 1]
        chunk_dur = end_sec - start_sec
        chunk_filename = f"chunk_{idx:04d}{video_path.suffix}"
        chunk_path = output_dir / chunk_filename

        logger.info(
            f"Extracting Chunk {idx:04d}: start={start_sec:.3f}s, end={end_sec:.3f}s, duration={chunk_dur:.3f}s -> {chunk_filename}"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-i",
            str(video_path.resolve()),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "1",
            str(chunk_path.resolve()),
        ]

        _run_command(cmd, description=f"chunk_{idx:04d}")
        chunk_paths.append(chunk_path)

    return chunk_paths


def extract_frames(
    video_path: Path,
    fps: float | None = None,
) -> list[tuple[float, np.ndarray]]:
    """Extract video frames as numpy arrays via an FFmpeg pipe.

    Args:
        video_path: Path to source video.
        fps: Desired extraction FPS. If None, uses original video frame rate.

    Returns:
        List of tuples: (timestamp_in_seconds, rgb_frame_array_H_W_3_uint8).

    Raises:
        ImportError: If numpy is not installed.
        ValueError: If extraction or piping fails.
    """
    import numpy as np

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    info = probe_video(video_path)
    width = info["width"]
    height = info["height"]
    target_fps = float(fps) if fps is not None and fps > 0 else float(info["fps"])

    cmd = ["ffmpeg", "-i", str(video_path.resolve())]
    if fps is not None and fps > 0:
        cmd.extend(["-vf", f"fps={target_fps}"])
    cmd.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])

    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
    logger.debug(f"Piping frames with ffmpeg: {cmd_str}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**7,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg binary not found") from exc

    frame_bytes = width * height * 3
    frames: list[tuple[float, np.ndarray]] = []
    frame_index = 0

    try:
        assert process.stdout is not None
        while True:
            raw_data = process.stdout.read(frame_bytes)
            if not raw_data:
                break
            if len(raw_data) < frame_bytes:
                logger.warning(
                    f"Incomplete frame buffer read ({len(raw_data)}/{frame_bytes} bytes). Stopping frame extraction."
                )
                break

            frame_arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
            ts = frame_index / target_fps
            frames.append((ts, frame_arr))
            frame_index += 1

        process.stdout.close()
        stderr_output = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        process.wait()

        if process.returncode != 0 and not frames:
            raise RuntimeError(f"FFmpeg frame extraction failed with code {process.returncode}: {stderr_output}")

    except Exception as exc:
        if process.poll() is None:
            process.kill()
        raise ValueError(f"Error during frame extraction from {video_path}: {exc}") from exc

    logger.info(f"Extracted {len(frames)} frames from {video_path.name} at {target_fps:.2f} FPS")
    return frames


def frames_to_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    audio_source: Path | None = None,
) -> Path:
    """Encode a sequence of RGB numpy frames into an MP4 video using an FFmpeg pipe.

    Args:
        frames: List of numpy RGB arrays of shape (H, W, 3) uint8.
        output_path: Destination path for the generated video.
        fps: Playback frame rate.
        audio_source: Optional path to an audio track to mux into the video.

    Returns:
        Path to the generated video.

    Raises:
        ValueError: If frames list is empty or encoding fails.
    """
    if not frames:
        raise ValueError("Cannot encode video: frames list is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width, channels = frames[0].shape
    if channels != 3:
        raise ValueError(f"Expected 3 channels (RGB), got shape {frames[0].shape}")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]

    has_audio = audio_source is not None and Path(audio_source).is_file()
    if has_audio:
        cmd.extend([
            "-i",
            str(Path(audio_source).resolve()),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ])

    cmd.extend([
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output_path.resolve()),
    ])

    cmd_str = " ".join(shlex.quote(a) for a in cmd)
    logger.debug(f"Encoding frames to video: {cmd_str}")

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()

        stdout_data, stderr_data = process.communicate()
        if process.returncode != 0:
            err_msg = stderr_data.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg encoding failed with exit code {process.returncode}: {err_msg}")

    except Exception as exc:
        if process.poll() is None:
            process.kill()
        raise ValueError(f"Failed to encode frames to {output_path}: {exc}") from exc

    logger.info(f"Successfully encoded {len(frames)} frames to {output_path.name}")
    return output_path


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract the audio track from a video file.

    Args:
        video_path: Source video file.
        output_path: Target path for audio (e.g. .wav or .aac).

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If source video does not exist.
        RuntimeError: If extraction fails.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(video_path.resolve()), "-vn"]
    if output_path.suffix.lower() == ".wav":
        cmd.extend(["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])
    else:
        cmd.extend(["-acodec", "copy"])
    cmd.append(str(output_path.resolve()))

    try:
        _run_command(cmd, description=f"extract_audio({output_path.name})")
    except RuntimeError:
        # If stream copy failed (e.g. container mismatch), retry with standard aac
        logger.warning(f"Audio copy extraction failed for {output_path.name}, retrying with AAC transcode.")
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path.resolve()),
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path.resolve()),
        ]
        _run_command(fallback_cmd, description=f"extract_audio_fallback({output_path.name})")

    logger.info(f"Extracted audio from {video_path.name} -> {output_path.name}")
    return output_path


def stitch_chunks(
    chunk_paths: list[Path],
    output_path: Path,
    audio_path: Path | None = None,
) -> Path:
    """Stitch video chunks together using FFmpeg concat demuxer with optional audio remuxing.

    If audio_path is provided, muxes the audio track into the merged stream with
    crossfading across boundaries using the 'acrossfade' filter if multiple audio inputs exist.

    Args:
        chunk_paths: Ordered list of video chunk paths.
        output_path: Output stitched video path.
        audio_path: Optional path to an audio track to mux into the video.

    Returns:
        Path to the stitched video file.

    Raises:
        ValueError: If chunk_paths list is empty.
        RuntimeError: If FFmpeg stitching fails.
    """
    if not chunk_paths:
        raise ValueError("Cannot stitch empty list of chunks")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concat_list_file = output_path.parent / f"concat_{output_path.stem}.txt"
    try:
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for p in chunk_paths:
                escaped = str(p.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        if audio_path is not None and Path(audio_path).is_file():
            # Merge video from concat demuxer and external audio track
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_file.resolve()),
                "-i",
                str(Path(audio_path).resolve()),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                str(output_path.resolve()),
            ]
        else:
            # Lossless video & embedded audio copy via concat demuxer
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_file.resolve()),
                "-c",
                "copy",
                str(output_path.resolve()),
            ]

        _run_command(cmd, description=f"stitch_chunks({output_path.name})")

    finally:
        if concat_list_file.exists():
            try:
                concat_list_file.unlink()
            except OSError as exc:
                logger.warning(f"Failed to remove temporary concat file {concat_list_file}: {exc}")

    logger.info(f"Successfully stitched {len(chunk_paths)} chunks -> {output_path.name}")
    return output_path
