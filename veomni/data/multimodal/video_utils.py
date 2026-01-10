import math
import subprocess
from typing import ByteString, Dict, List, Union

import librosa
import numpy as np
import PIL
import torch
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import InterpolationMode, functional

from ...utils import logging
from ...utils.import_utils import is_ffmpeg_available
from .audio_utils import extract_audio_from_video


logger = logging.get_logger(__name__)

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


def calculate_frame_indices(
    total_frames: int,
    video_fps: Union[int, float],
    fps: float = 2.0,
    frame_factor: int = None,
    min_frames: int = None,
    max_frames: int = None,
    **kwargs,
) -> tuple[List[int], int]:
    """Calculate frame indices to sample and padding count.

    Args:
        total_frames: Total frames in video
        video_fps: Original video FPS
        fps: Target sampling FPS
        frame_factor: Align output frame count to multiples of this
        min_frames: Minimum frames to output
        max_frames: Maximum frames to output
        **kwargs: Extra arguments (ignored)

    Returns:
        (indices, pad_count): Frame indices to sample and padding count
    """
    # Calculate target frame count
    nframes = total_frames / video_fps * fps

    # Apply min/max limits
    if min_frames is not None:
        if frame_factor is not None:
            min_frames = math.ceil(min_frames / frame_factor) * frame_factor
        nframes = max(min_frames, nframes)

    if max_frames is not None:
        if frame_factor is not None:
            max_frames = math.floor(max_frames / frame_factor) * frame_factor
        nframes = min(max_frames, nframes)

    # Align to frame_factor
    if frame_factor is not None:
        nframes = math.floor(nframes / frame_factor) * frame_factor
        nframes = max(nframes, frame_factor)

    nframes = int(max(1, nframes))

    # Calculate padding
    pad_count = 0
    if nframes > total_frames:
        pad_count = nframes - total_frames
        sample_count = total_frames
    else:
        sample_count = nframes

    # Uniform sampling
    if sample_count > 0:
        indices = np.linspace(0, total_frames - 1, sample_count).round().astype(int).tolist()
    else:
        indices = []

    return indices, pad_count


def smart_video_nframes(
    video: torch.Tensor,
    video_fps: Union[int, float],
    fps: int = 2.0,
    frame_factor: int = None,
    min_frames: int = None,
    max_frames: int = None,
    **kwargs,
) -> tuple[torch.Tensor, Dict[str, Union[float, int]]]:
    """Sample video frames with fps and alignment constraints.

    Args:
        video: Video tensor (T, C, H, W)
        video_fps: Original video FPS
        fps: Target sampling FPS
        frame_factor: Align output frame count to multiples of this
        min_frames: Minimum frames to output
        max_frames: Maximum frames to output
        **kwargs: Additional arguments (e.g., 'frames' for explicit count)

    Returns:
        (video, metadata): Processed video and metadata dict with 'fps', 'total_num_frames'
    """
    total_frames = video.shape[0]

    # Support explicit frame count override
    if "frames" in kwargs:
        target_frames = kwargs["frames"]
        indices, pad_count = calculate_frame_indices(
            total_frames=total_frames,
            video_fps=video_fps,
            fps=fps,
            frame_factor=frame_factor,
            min_frames=target_frames,
            max_frames=target_frames,
        )
    else:
        indices, pad_count = calculate_frame_indices(
            total_frames=total_frames,
            video_fps=video_fps,
            fps=fps,
            frame_factor=frame_factor,
            min_frames=min_frames,
            max_frames=max_frames,
        )

    video = video[indices]

    # Pad with last frame if needed
    if pad_count > 0:
        last_frame = video[-1:].expand(pad_count, -1, -1, -1)
        video = torch.cat([video, last_frame], dim=0)

    nframes = video.shape[0]
    fps_out = video_fps * nframes / total_frames if total_frames > 0 else fps

    return video, {"fps": fps_out, "total_num_frames": nframes}


def smart_audio_nframes(
    audio: np.ndarray, audio_fps: int, sample_rate: int = 16000, **kwargs
) -> tuple[np.ndarray, Dict[str, Union[float, int]]]:
    """Resample audio to target sample rate.

    Args:
        audio: Input audio array (can be None)
        audio_fps: Original sample rate
        sample_rate: Target sample rate (default: 16kHz)
        **kwargs: Additional arguments (ignored)

    Returns:
        (audio, metadata): Resampled audio and metadata dict with 'fps', 'total_num_frames'
    """
    if audio is not None and audio_fps != sample_rate:
        audio = librosa.resample(audio, orig_sr=audio_fps, target_sr=sample_rate)
    num_frames = len(audio) if audio is not None else 0
    return audio, {"fps": sample_rate, "total_num_frames": num_frames}


def smart_resize(
    video: torch.Tensor,
    scale_factor: int = None,
    video_min_pixels: int = None,
    video_max_pixels: int = None,
    max_ratio: int = None,
    **kwargs,
):
    """Resize video preserving aspect ratio with pixel and alignment constraints.

    Args:
        video: Video tensor (T, C, H, W)
        scale_factor: Align H and W to multiples of this (e.g., 14 for ViT)
        video_min_pixels: Minimum total pixels (H × W)
        video_max_pixels: Maximum total pixels (H × W)
        max_ratio: Maximum aspect ratio (max_dim / min_dim)
        **kwargs: Additional arguments (ignored)

    Returns:
        Resized video tensor (T, C, h_bar, w_bar)

    Raises:
        ValueError: If aspect ratio exceeds max_ratio or video is not 4D
    """
    if video.ndim != 4:
        raise ValueError(f"video must be 4-dim, but got {video.ndim}")
    _, _, height, width = video.shape

    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}, got {ratio}")

    if scale_factor is not None:
        h_bar = max(scale_factor, round(height / scale_factor) * scale_factor)
        w_bar = max(scale_factor, round(width / scale_factor) * scale_factor)
    else:
        h_bar = height
        w_bar = width

    if video_max_pixels is not None and h_bar * w_bar > video_max_pixels:
        beta = math.sqrt((height * width) / video_max_pixels)
        if scale_factor is not None:
            h_bar = math.floor(height / beta / scale_factor) * scale_factor
            w_bar = math.floor(width / beta / scale_factor) * scale_factor
        else:
            h_bar = math.floor(height / beta)
            w_bar = math.floor(width / beta)

    if video_min_pixels is not None and h_bar * w_bar < video_min_pixels:
        beta = math.sqrt(video_min_pixels / (height * width))
        if scale_factor is not None:
            h_bar = math.ceil(height * beta / scale_factor) * scale_factor
            w_bar = math.ceil(width * beta / scale_factor) * scale_factor
        else:
            h_bar = math.ceil(height * beta)
            w_bar = math.ceil(width * beta)

    video = functional.resize(
        video,
        [h_bar, w_bar],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


def _download_url_to_bytes(url: str) -> bytes:
    """Download video from URL using ffmpeg."""
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
    """Convert PIL images to tensor (T, C, H, W)."""
    tensors = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        tensors.append(tensor)
    return torch.stack(tensors)


def _dict_to_video_audio(video_dict: Dict[str, "np.ndarray"]) -> tuple:
    """Convert dict to video tensor and audio array.

    Expected keys: 'video' (required), 'audio', 'video_fps', 'audio_fps' (optional).
    """
    if "video" not in video_dict:
        logger.error(f"Dict input missing 'video' key. Available keys: {list(video_dict.keys())}")
        raise ValueError("Dict input must contain 'video' key")

    video_np = video_dict["video"]
    logger.debug(f"Processing video array with shape: {video_np.shape}, dtype: {video_np.dtype}")

    if video_np.ndim == 4:
        if video_np.shape[-1] == 3:
            logger.debug(f"Converting (T, H, W, C) format: {video_np.shape} -> permute to (T, C, H, W)")
            video = torch.from_numpy(video_np).permute(0, 3, 1, 2)
        else:
            logger.debug(f"Assuming (T, C, H, W) format: {video_np.shape}, no permutation needed")
            video = torch.from_numpy(video_np)
    else:
        logger.error(
            f"Invalid video array dimensions. Expected 4D array, got shape: {video_np.shape} (ndim={video_np.ndim})"
        )
        raise ValueError(f"Video array must be 4D, got shape {video_np.shape}")

    audio = video_dict.get("audio", None)
    video_fps = video_dict.get("video_fps", 30.0)
    audio_fps = video_dict.get("audio_fps", None)

    return video, video_fps, audio, audio_fps


def _load_and_process_video_with_codec(video_input: VideoInput, use_audio_in_video: bool = True, **kwargs):
    """Load and process video using torchcodec (video) and PyAV (audio).

    Supports: str (path/URL), bytes, List[PIL.Image], List[bytes], Dict[str, np.ndarray].

    Returns:
        (video, audio, audio_fps, frames_indices): Processed video, audio array, audio FPS,
        and sampled frame indices from original video
    """
    if isinstance(video_input, list):
        if len(video_input) > 0 and isinstance(video_input[0], bytes):
            from io import BytesIO

            pil_images = []
            for frame_bytes in video_input:
                with PIL.Image.open(BytesIO(frame_bytes)) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    pil_images.append(img.copy())
            video = _pil_images_to_tensor(pil_images)
        else:
            video = _pil_images_to_tensor(video_input)

        video_fps = kwargs.get("fps", 2.0)
        audio, audio_fps = None, None

        video, _ = smart_video_nframes(smart_resize(video, **kwargs), video_fps, **kwargs)
        frames_indices = torch.arange(video.shape[0])
        return video, audio, audio_fps, frames_indices

    elif isinstance(video_input, dict):
        video, video_fps, audio, audio_fps = _dict_to_video_audio(video_input)
        video, _ = smart_video_nframes(smart_resize(video, **kwargs), video_fps, **kwargs)
        frames_indices = torch.arange(video.shape[0])
        return video, audio, audio_fps, frames_indices

    # video_input is str (path/URL) or bytes
    try:
        decoder = VideoDecoder(video_input, device="cpu", num_ffmpeg_threads=0)
    except Exception as e:
        if isinstance(video_input, str) and ("http://" in video_input or "https://" in video_input):
            logger.warning(f"Direct URL decoding failed: {e}. Downloading with ffmpeg...")
            try:
                video_bytes = _download_url_to_bytes(video_input)
                decoder = VideoDecoder(video_bytes, device="cpu", num_ffmpeg_threads=0)
            except Exception as download_error:
                raise RuntimeError(f"Failed to decode video from URL {video_input}: {download_error}")
        else:
            raise RuntimeError(f"Failed to create VideoDecoder: {e}")

    metadata = decoder.metadata
    video_fps = metadata.average_fps
    total_frames = metadata.num_frames

    # Safety margin for inaccurate frame counts
    effective_total_frames = max(1, total_frames - 1)

    indices, pad_count = calculate_frame_indices(total_frames=effective_total_frames, video_fps=video_fps, **kwargs)

    try:
        frames = decoder.get_frames_at(indices).data
        sampled_indices = indices
    except Exception as e:
        if "Requested next frame" in str(e) or "End of stream" in str(e):
            logger.warning(f"Decoding failed: {e}. Retrying with first frame only.")
            try:
                frames = decoder.get_frames_at([0]).data
                sampled_indices = [0]
                _, pad_count = calculate_frame_indices(total_frames=1, video_fps=video_fps, **kwargs)
            except Exception:
                raise RuntimeError(f"Failed to decode even the first frame: {e}")
        else:
            raise e

    resized_frames = smart_resize(frames, **kwargs)

    if pad_count > 0:
        last_frame = resized_frames[-1:].expand(pad_count, -1, -1, -1)
        final_frames = torch.cat([resized_frames, last_frame], dim=0)
        padded_indices = sampled_indices + [sampled_indices[-1]] * pad_count
    else:
        final_frames = resized_frames
        padded_indices = sampled_indices

    # Extract audio with PyAV
    audio, audio_fps = None, None
    if use_audio_in_video:
        max_audio_duration = (metadata.duration_seconds or 60.0) + 1.0
        audio, audio_fps = extract_audio_from_video(video_input, max_duration_seconds=max_audio_duration)

    frames_indices = torch.tensor(padded_indices, dtype=torch.long)

    return final_frames, audio, audio_fps, frames_indices


def fetch_videos(videos: List[VideoInput], **kwargs):
    """Fetch and process videos.

    Note: Does NOT return frames_indices. Use fetch_videos_metadata() for temporal modeling
    (e.g., Qwen3-VL timestamp calculation).
    """
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available. Please install it: apt-get install ffmpeg or brew install ffmpeg")

    logger.info_once("Using torchcodec for video loading.")

    video_inputs, audio_inputs, audio_fps_list = [], [], []

    for i, video in enumerate(videos):
        try:
            processed_video, audio, audio_fps, _ = _load_and_process_video_with_codec(video, **kwargs)
            video_inputs.append(processed_video)
            audio_inputs.append(audio)
            audio_fps_list.append(audio_fps)
        except Exception as e:
            raise RuntimeError(f"Failed to process video {i}: {e}")

    processed_audio_inputs = [
        smart_audio_nframes(audio, audio_fps, **kwargs)[0] if audio is not None and audio_fps is not None else None
        for audio, audio_fps in zip(audio_inputs, audio_fps_list)
    ]

    return video_inputs, processed_audio_inputs


def fetch_videos_metadata(videos: List[VideoInput], **kwargs):
    """Fetch and process videos with full metadata.

    IMPORTANT: For Qwen3-VL, this returns frames_indices needed for timestamp calculation.

    Args:
        videos: List of video inputs (paths, bytes, PIL images, or dicts)
        **kwargs: Processing parameters (fps, min_frames, max_frames, etc.)

    Returns:
        (videos, video_metadata, audios, audio_metadata): Processed videos and audios with metadata
    """
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available. Please install it: apt-get install ffmpeg or brew install ffmpeg")

    logger.info_once("Using torchcodec for video loading with metadata.")

    video_inputs, video_metadata_list = [], []
    audio_inputs, audio_metadata_list = [], []

    for i, video in enumerate(videos):
        try:
            processed_video, audio, audio_fps, frames_indices = _load_and_process_video_with_codec(video, **kwargs)

            video_meta = {
                "fps": kwargs.get("fps", 2.0),
                "total_num_frames": processed_video.shape[0],
                "frames_indices": frames_indices,
            }

            video_inputs.append(processed_video)
            video_metadata_list.append(video_meta)

            if audio is not None and audio_fps is not None:
                processed_audio, audio_meta = smart_audio_nframes(audio, audio_fps, **kwargs)
                audio_inputs.append(processed_audio)
                audio_metadata_list.append(audio_meta)
            else:
                audio_inputs.append(None)
                audio_metadata_list.append(None)

        except Exception as e:
            raise RuntimeError(f"Failed to process video {i}: {e}")

    return video_inputs, video_metadata_list, audio_inputs, audio_metadata_list


# Backward compatibility with torchvision-based API
def load_video_from_path(video_path: str, use_audio_in_video: bool = True, **kwargs):
    """Load video from file path (compatibility wrapper).

    Args:
        video_path: Path to video file or URL
        use_audio_in_video: Whether to extract audio
        **kwargs: Additional processing parameters

    Returns:
        (video, video_metadata, audio, audio_metadata)
    """
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available. Please install it: apt-get install ffmpeg or brew install ffmpeg")

    videos, video_meta, audios, audio_meta = fetch_videos_metadata(
        [video_path], use_audio_in_video=use_audio_in_video, **kwargs
    )
    return videos[0], video_meta[0], audios[0], audio_meta[0]


def load_video_from_bytes(video_bytes: bytes, use_audio_in_video: bool = True, **kwargs):
    """Load video from bytes (compatibility wrapper).

    Args:
        video_bytes: Video data as bytes
        use_audio_in_video: Whether to extract audio
        **kwargs: Additional processing parameters

    Returns:
        (video, video_metadata, audio, audio_metadata)
    """
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available. Please install it: apt-get install ffmpeg or brew install ffmpeg")

    videos, video_meta, audios, audio_meta = fetch_videos_metadata(
        [video_bytes], use_audio_in_video=use_audio_in_video, **kwargs
    )
    return videos[0], video_meta[0], audios[0], audio_meta[0]


def load_video_from_bytes_list(video_frames: Union[List[bytes], np.ndarray], **kwargs):
    """Load video from list of image bytes (compatibility wrapper).

    Args:
        video_frames: List of image bytes or numpy array
        **kwargs: Additional processing parameters (must include 'fps')

    Returns:
        (video, video_metadata, audio, audio_metadata)

    Raises:
        ValueError: If video_frames is empty
    """
    if isinstance(video_frames, np.ndarray):
        video_frames = video_frames.tolist()
    if not video_frames:
        raise ValueError("Input video frame list is empty")

    from io import BytesIO

    import PIL.Image

    pil_images = []
    for frame_bytes in video_frames:
        with PIL.Image.open(BytesIO(frame_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            pil_images.append(img.copy())

    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available. Please install it: apt-get install ffmpeg or brew install ffmpeg")

    videos, video_meta, audios, audio_meta = fetch_videos_metadata([pil_images], **kwargs)
    return videos[0], video_meta[0], audios[0], audio_meta[0]


def load_video(video: VideoInput, **kwargs):
    """Unified video loading interface (compatibility wrapper).

    Args:
        video: Video input (str path, bytes, list of PIL images, or dict)
        **kwargs: Additional processing parameters

    Returns:
        (video, video_metadata, audio, audio_metadata)
    """
    if isinstance(video, str):
        return load_video_from_path(video, **kwargs)
    elif isinstance(video, bytes):
        return load_video_from_bytes(video, **kwargs)
    elif isinstance(video, (list, np.ndarray)):
        if len(video) > 0:
            if isinstance(video[0], bytes):
                return load_video_from_bytes_list(video, **kwargs)
            if not is_ffmpeg_available():
                raise RuntimeError(
                    "ffmpeg is not available. Please install it: apt-get install ffmpeg or brew install ffmpeg"
                )
            videos, video_meta, audios, audio_meta = fetch_videos_metadata([video], **kwargs)
            return videos[0], video_meta[0], audios[0], audio_meta[0]
    elif isinstance(video, dict):
        if not is_ffmpeg_available():
            raise RuntimeError(
                "ffmpeg is not available. Please install it: apt-get install ffmpeg or brew install ffmpeg"
            )
        videos, video_meta, audios, audio_meta = fetch_videos_metadata([video], **kwargs)
        return videos[0], video_meta[0], audios[0], audio_meta[0]

    raise NotImplementedError(f"Unsupported video input type: {type(video)}")
