import io
import math
import subprocess
from typing import ByteString, Dict, List, Union

import av
import librosa
import numpy as np
import PIL
import torch
from torchvision.transforms import InterpolationMode, functional

from ...utils import logging


logger = logging.get_logger(__name__)

try:
    from torchcodec.decoders import VideoDecoder

    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False
    VideoDecoder = None

_FFMPEG_AVAILABLE = None


def is_ffmpeg_available():
    """Check if ffmpeg is available on the system.

    Note: torchcodec (if used) supports FFmpeg versions 4-8.
    See: https://github.com/pytorch/torchcodec
    """
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
            _FFMPEG_AVAILABLE = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            _FFMPEG_AVAILABLE = False
    return _FFMPEG_AVAILABLE


if not hasattr(av, "AVError"):
    try:
        from av.error import AVError  # noqa: F401
    except (ImportError, AttributeError):
        av.AVError = OSError

VideoInput = Union[
    List["PIL.Image.Image"],
    Dict[str, "np.ndarray"],
    ByteString,
    str,
]


def load_video_bytes_from_path(video_path: str):
    with open(video_path, "rb") as f:
        return f.read()


def save_video_bytes_to_file(video_bytes, output_path):
    with open(output_path, "wb") as f:
        f.write(video_bytes)


def smart_video_nframes(
    video: torch.Tensor,
    video_fps: Union[int, float],
    fps: int = 2.0,
    frame_factor: int = None,
    min_frames: int = None,
    max_frames: int = None,
    **kwargs,
) -> torch.Tensor:
    """
    Smart video frame sampling with adaptive constraints.

    The "smart" aspect refers to the intelligent handling of multiple conflicting constraints:
    1. Target FPS sampling (fps parameter)
    2. Frame count bounds (min_frames, max_frames)
    3. Frame factor alignment (for models requiring specific frame counts like multiples of 8/16)
    4. Automatic padding when needed (duplicates last frame if target > available frames)
    5. Ensures at least 1 frame output regardless of constraints

    This method adaptively balances these constraints to produce optimal frame sampling
    for multimodal models while respecting hardware/model requirements.

    Args:
        video: Input video tensor of shape (T, C, H, W)
        video_fps: Original video frame rate
        fps: Target sampling rate (frames per second)
        frame_factor: Ensure output frames are multiples of this (e.g., 8 for some models)
        min_frames: Minimum number of frames to sample
        max_frames: Maximum number of frames to sample

    Returns:
        Sampled video tensor with shape (nframes, C, H, W)
    """
    total_frames = video.shape[0]
    # Calculate target frame count based on desired sampling rate
    nframes = total_frames / video_fps * fps

    # Apply minimum frame constraint (with frame_factor alignment if specified)
    if min_frames is not None:
        if frame_factor is not None:
            min_frames = math.ceil(min_frames / frame_factor) * frame_factor
        nframes = max(min_frames, nframes)

    # Apply maximum frame constraint (with frame_factor alignment if specified)
    if max_frames is not None:
        if frame_factor is not None:
            max_frames = math.floor(max_frames / frame_factor) * frame_factor
        nframes = min(max_frames, nframes)

    # Ensure we don't exceed available frames
    nframes = min(nframes, total_frames)

    # Apply frame_factor alignment to final frame count
    if frame_factor is not None:
        nframes = math.floor(nframes / frame_factor) * frame_factor
        nframes = max(nframes, frame_factor)  # At least one factor worth of frames

    # Ensure at least 1 frame (safety net)
    nframes = max(1, nframes)

    # Smart padding: if we need more frames than available, duplicate the last frame
    if nframes > total_frames:
        pad_count = nframes - total_frames
        last_frame = video[-1:].expand(pad_count, -1, -1, -1)  # shape: (pad_count, C, H, W)
        video = torch.cat([video, last_frame], dim=0)
        total_frames = video.shape[0]

    # Uniform sampling across the video timeline
    idx = torch.linspace(0, total_frames - 1, int(nframes)).round().long()
    video = video[idx]
    return video


def smart_audio_nframes(audio: np.ndarray, audio_fps: int, sample_rate: int = 16000, **kwargs):
    """
    Smart audio resampling to target sample rate.

    The "smart" aspect refers to:
    1. Automatic resampling to standard sample rates (e.g., 16kHz for speech models)
    2. Handles None/missing audio gracefully (returns None without error)
    3. Uses high-quality librosa resampling with anti-aliasing

    Args:
        audio: Input audio array (can be None)
        audio_fps: Original audio sample rate
        sample_rate: Target sample rate (default 16kHz for speech processing)
        **kwargs: Additional arguments (for compatibility with other smart_* methods)

    Returns:
        Resampled audio array or None if input is None
    """
    if audio is not None:
        audio = librosa.resample(audio, orig_sr=audio_fps, target_sr=sample_rate)
    return audio


def smart_resize(
    video: torch.Tensor,
    scale_factor: int = None,
    video_min_pixels: int = None,
    video_max_pixels: int = None,
    max_ratio: int = None,
    **kwargs,
):
    """
    Smart video resizing with adaptive resolution constraints.

    The "smart" aspect refers to the intelligent resolution adjustment that:
    1. Preserves aspect ratio while meeting pixel count constraints
    2. Aligns dimensions to scale_factor multiples (e.g., 14 for ViT patch size, 32 for some CNNs)
    3. Balances min/max pixel constraints with scale factor alignment
    4. Validates aspect ratio to prevent extreme distortions
    5. Uses high-quality bicubic interpolation with antialiasing

    This is crucial for vision-language models where:
    - Patch-based models (ViT) require dimensions divisible by patch size
    - Memory constraints limit total pixel count
    - Aspect ratio affects positional embeddings

    Args:
        video: Input video tensor of shape (T, C, H, W)
        scale_factor: Ensure H and W are multiples of this (e.g., 14 for ViT, 32 for conv nets)
        video_min_pixels: Minimum total pixels (H x W) to maintain detail
        video_max_pixels: Maximum total pixels (H x W) to fit in memory
        max_ratio: Maximum aspect ratio (max_dim / min_dim) to prevent extreme shapes
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Resized video tensor with shape (T, C, h_bar, w_bar)

    Raises:
        ValueError: If aspect ratio exceeds max_ratio or video is not 4D
    """
   
    # video: t, c, h, w
    if video.ndim != 4:
        raise ValueError(f"video must be 4-dim, but got {video.ndim}")
    _, _, height, width = video.shape

    # Validate aspect ratio to prevent extreme shapes
    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}, got {ratio}")

    # Start with scale_factor aligned dimensions (or original if no scale_factor)
    if scale_factor is not None:
        h_bar = max(scale_factor, round(height / scale_factor) * scale_factor)
        w_bar = max(scale_factor, round(width / scale_factor) * scale_factor)
    else:
        h_bar = height
        w_bar = width

    # Scale down if exceeds maximum pixels (preserving aspect ratio)
    if video_max_pixels is not None and h_bar * w_bar > video_max_pixels:
        beta = math.sqrt((height * width) / video_max_pixels)
        if scale_factor is not None:
            h_bar = math.floor(height / beta / scale_factor) * scale_factor
            w_bar = math.floor(width / beta / scale_factor) * scale_factor
        else:
            h_bar = math.floor(height / beta)
            w_bar = math.floor(width / beta)

    # Scale up if below minimum pixels (preserving aspect ratio)
    if video_min_pixels is not None and h_bar * w_bar < video_min_pixels:
        beta = math.sqrt(video_min_pixels / (height * width))
        if scale_factor is not None:
            h_bar = math.ceil(height * beta / scale_factor) * scale_factor
            w_bar = math.ceil(width * beta / scale_factor) * scale_factor
        else:
            h_bar = math.ceil(height * beta)
            w_bar = math.ceil(width * beta)

    # Apply high-quality resize with antialiasing
    video = functional.resize(
        video,
        [h_bar, w_bar],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


def _download_url_to_bytes(url: str) -> bytes:
    """Download video from URL to bytes using ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", url, "-f", "mp4", "-"],
            capture_output=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download video from {url}: {e.stderr.decode()}")


def _pil_images_to_tensor(images: List["PIL.Image.Image"]) -> torch.Tensor:
    """Convert PIL images to tensor with shape (T, C, H, W)."""
    tensors = []
    for img in images:
        # Convert PIL to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        tensors.append(tensor)
    return torch.stack(tensors)  # (T, C, H, W)


def _dict_to_video_audio(video_dict: Dict[str, "np.ndarray"]) -> tuple:
    """
    Convert dict to video tensor and audio array.
    Expected keys: 'video' (required), 'audio' (optional), 'video_fps' (optional), 'audio_fps' (optional).
    """
    if "video" not in video_dict:
        logger.error(f"Dict input missing 'video' key. Available keys: {list(video_dict.keys())}")
        raise ValueError("Dict input must contain 'video' key")

    video_np = video_dict["video"]
    logger.debug(f"Processing video array with shape: {video_np.shape}, dtype: {video_np.dtype}")

    # Assume video_np is (T, H, W, C) or (T, C, H, W)
    if video_np.ndim == 4:
        if video_np.shape[-1] == 3:  # (T, H, W, C)
            logger.debug(f"Converting (T, H, W, C) format: {video_np.shape} -> permute to (T, C, H, W)")
            video = torch.from_numpy(video_np).permute(0, 3, 1, 2)  # -> (T, C, H, W)
        else:  # Assume (T, C, H, W)
            logger.debug(f"Assuming (T, C, H, W) format: {video_np.shape}, no permutation needed")
            video = torch.from_numpy(video_np)
    else:
        logger.error(
            f"Invalid video array dimensions. Expected 4D array, got shape: {video_np.shape} (ndim={video_np.ndim})"
        )
        raise ValueError(f"Video array must be 4D, got shape {video_np.shape}")

    audio = video_dict.get("audio", None)
    video_fps = video_dict.get("video_fps", 30.0)  # Default FPS
    audio_fps = video_dict.get("audio_fps", None)

    return video, video_fps, audio, audio_fps


def _load_and_process_video_with_codec(video_input: VideoInput, use_audio_in_video: bool = True, **kwargs):
    """
    Unified video processing with torchcodec and PyAV.
    Supports all VideoInput types:
    - str (file path or http/https URL)
    - bytes
    - List[PIL.Image.Image]
    - Dict[str, np.ndarray]
    """

    # --- Handle different input types ---
    if isinstance(video_input, list):
        # List[PIL.Image.Image]
        video = _pil_images_to_tensor(video_input)
        video_fps = kwargs.get("fps", 2.0)  # Use target fps as source fps for images
        audio, audio_fps = None, None

        # Process video frames
        video = smart_video_nframes(smart_resize(video, **kwargs), video_fps, **kwargs)
        return video, audio, audio_fps

    elif isinstance(video_input, dict):
        # Dict[str, np.ndarray]
        video, video_fps, audio, audio_fps = _dict_to_video_audio(video_input)

        # Process video frames
        video = smart_video_nframes(smart_resize(video, **kwargs), video_fps, **kwargs)

        # Process audio if present
        if audio is not None and audio_fps is not None:
            audio = smart_audio_nframes(audio, audio_fps, **kwargs)

        return video, audio, audio_fps

    elif isinstance(video_input, str) and ("http://" in video_input or "https://" in video_input):
        # Download URL to bytes first
        logger.info(f"Downloading video from URL: {video_input}")
        video_bytes = _download_url_to_bytes(video_input)
        video_input = video_bytes

    # --- From here, video_input is either str (path) or bytes ---

    # --- 1. Video Handling (TorchCodec) ---
    try:
        decoder = VideoDecoder(video_input, device="cpu", num_ffmpeg_threads=0)
    except Exception as e:
        raise RuntimeError(f"Failed to create VideoDecoder: {e}")

    metadata = decoder.metadata
    video_fps = metadata.average_fps
    total_frames = metadata.num_frames

    fps = kwargs.get("fps", 2.0)
    nframes = total_frames / video_fps * fps

    min_frames = kwargs.get("min_frames")
    max_frames = kwargs.get("max_frames")
    frame_factor = kwargs.get("frame_factor")

    if min_frames is not None:
        if frame_factor is not None:
            min_frames = math.ceil(min_frames / frame_factor) * frame_factor
        nframes = max(min_frames, nframes)
    if max_frames is not None:
        if frame_factor is not None:
            max_frames = math.floor(max_frames / frame_factor) * frame_factor
        nframes = min(max_frames, nframes)
    nframes = min(nframes, total_frames)
    if frame_factor is not None:
        nframes = math.floor(nframes / frame_factor) * frame_factor
        nframes = max(nframes, frame_factor)

    # Ensure at least 1 frame
    nframes = max(1, nframes)
    nframes = int(nframes)

    if nframes <= 0:
        indices = [0]
    else:
        # Safety Margin Strategy
        # Many video metadata reports total_frames 1-2 frames more than actually decodable
        # We force skip the last 2 frames to prevent "Requested next frame..." errors
        safe_end_frame = max(0, total_frames - 3)
        indices = np.linspace(0, safe_end_frame, nframes, dtype=int).tolist()

    try:
        # Try normal decoding
        frames = decoder.get_frames_at(indices).data
    except Exception as e:
        # Exception handling with fallback strategy
        if "Requested next frame" in str(e) or "End of stream" in str(e):
            logger.warning(f"TorchCodec decoding edge case caught: {e}. Retrying with conservative indices.")
            # Strategy: if failed, only get frame 0, then replicate nframes times
            # This ensures pipeline doesn't crash, though sacrifices this sample's dynamics
            try:
                frames = decoder.get_frames_at([0]).data
                frames = frames.expand(nframes, -1, -1, -1)  # Replicate padding
            except Exception:
                raise e  # If even frame 0 fails, truly need fallback
        else:
            raise e  # Other errors propagate directly

    # Pad frames if needed
    if frames.shape[0] < nframes:
        pad_count = nframes - frames.shape[0]
        last_frame = frames[-1:].expand(pad_count, -1, -1, -1)
        frames = torch.cat([frames, last_frame], dim=0)

    resized_frames = smart_resize(frames, **kwargs)

    # --- 2. Audio Handling (PyAV with BytesIO) ---
    audio, audio_fps = None, None
    if use_audio_in_video:
        try:
            # Handle Bytes input without temp files
            if isinstance(video_input, bytes):
                # PyAV accepts file-like objects
                container_input = io.BytesIO(video_input)
            else:
                container_input = video_input

            container = av.open(container_input)

            if len(container.streams.audio) > 0:
                audio_stream = container.streams.audio[0]
                audio_fps = audio_stream.rate

                # Prevent OOM: only read needed audio length instead of float('inf')
                # Estimate max needed audio samples (with small buffer)
                max_audio_duration = (metadata.duration_seconds or 60.0) + 1.0
                max_samples = int(max_audio_duration * audio_fps)

                # Manual decode loop (replaces _read_from_stream to support BytesIO and clipping)
                audio_frames_list = []
                current_samples = 0

                for frame in container.decode(audio_stream):
                    frame_np = frame.to_ndarray()
                    audio_frames_list.append(frame_np)
                    current_samples += frame_np.shape[1]
                    if current_samples >= max_samples:
                        break  # Stop when we have enough!

                if len(audio_frames_list) > 0:
                    aframes = np.concatenate(audio_frames_list, axis=1)
                    # If multi-channel, average to mono (consistent with original logic)
                    if aframes.shape[0] > 1:
                        aframes = np.mean(aframes, axis=0)
                    else:
                        aframes = aframes[0]  # (1, T) -> (T,)

                    audio = aframes

            container.close()

        except Exception as e:
            logger.warning(f"Failed to load audio with av: {e}")
            # Audio failure shouldn't cause video loading failure, just return None
            audio = None
            audio_fps = None

    return resized_frames, audio, audio_fps


def fetch_videos(videos: List[VideoInput], **kwargs):
    """
    Unified video fetching supporting all VideoInput types.
    Requires torchcodec and ffmpeg to be available.
    """
    if not TORCHCODEC_AVAILABLE:
        raise RuntimeError("torchcodec is not available. Please install it: pip install torchcodec")
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available. Please install it: apt-get install ffmpeg or brew install ffmpeg")

    logger.info_once("Using torchcodec for video loading.")

    video_inputs, audio_inputs, audio_fps_list = [], [], []

    for i, video in enumerate(videos):
        try:
            processed_video, audio, audio_fps = _load_and_process_video_with_codec(video, **kwargs)
            video_inputs.append(processed_video)
            audio_inputs.append(audio)
            audio_fps_list.append(audio_fps)
        except Exception as e:
            raise RuntimeError(f"Failed to process video {i}: {e}")

    # Process audio (resample if needed)
    processed_audio_inputs = [
        smart_audio_nframes(audio, audio_fps, **kwargs) if audio is not None else None
        for audio, audio_fps in zip(audio_inputs, audio_fps_list)
    ]

    return video_inputs, processed_audio_inputs
