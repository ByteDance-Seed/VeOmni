import math
from io import BytesIO
from typing import ByteString, Dict, List, Optional, Tuple, Union

import av
import librosa
import numpy as np
import PIL.Image
import torch
import torchvision
from torchvision.io.video import _read_from_stream
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from ...utils import logging


logger = logging.get_logger(__name__)

if not hasattr(av, "AVError"):
    try:
        from av.error import AVError  # noqa: F401
    except (ImportError, AttributeError):
        av.AVError = OSError

VideoInput = Union[
    List["PIL.Image.Image"],
    Dict[str, "np.ndarray"],
    List[bytes],
    ByteString,
    str,
]


def load_video_bytes_from_path(video_path: str):
    with open(video_path, "rb") as f:
        return f.read()


def save_video_bytes_to_file(video_bytes, output_path):
    with open(output_path, "wb") as f:
        f.write(video_bytes)


def smart_resize(
    video: torch.Tensor,
    scale_factor: int = None,
    video_min_pixels: int = None,
    video_max_pixels: int = None,
    max_ratio: int = None,
    **kwargs,
) -> torch.Tensor:
    """
    Resizes a video tensor based on scaling factor or pixel limits.

    Args:
        video: Input video tensor of shape (T, C, H, W).
        scale_factor: Factor to scale dimensions to multiples of this value.
        video_min_pixels: Minimum total pixels constraint.
        video_max_pixels: Maximum total pixels constraint.
        max_ratio: Maximum allowed aspect ratio (longer side / shorter side).

    Returns:
        Resized video tensor.
    """
    height, width = video.shape[2], video.shape[3]

    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"Absolute aspect ratio must be smaller than {max_ratio}, got {ratio:.2f}")

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

    video = F.resize(
        video,
        [h_bar, w_bar],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


def smart_video_nframes(
    video: torch.Tensor,
    video_fps: Union[int, float],
    fps: int = 2.0,
    frame_factor: int = None,
    min_frames: int = None,
    max_frames: int = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Union[float, int]]]:
    """
    Samples frames from a video tensor intelligently.

    Args:
        video: Video tensor of shape (T, C, H, W).
        video_fps: Original FPS of the video.
        fps: Target FPS for sampling.
        frame_factor: Frame count must be a multiple of this factor.
        min_frames: Minimum number of frames to keep.
        max_frames: Maximum number of frames to keep.

    Returns:
        Tuple containing processed video tensor and metadata dictionary.
    """
    total_frames = video.shape[0]

    if "frames" in kwargs:
        nframes = kwargs["frames"]
    else:
        nframes = total_frames / video_fps * fps

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

    # Padding if necessary (though current logic prevents nframes > total_frames above,
    # original logic retained for robustness or edge cases)
    if nframes > total_frames:
        pad_count = int(nframes - total_frames)
        last_frame = video[-1:].expand(pad_count, -1, -1, -1)
        video = torch.cat([video, last_frame], dim=0)
        total_frames = video.shape[0]

    fps_out = video_fps * nframes / total_frames
    idx = torch.linspace(0, total_frames - 1, int(nframes)).round().long()
    video = video[idx]

    return video, {"fps": fps_out, "total_num_frames": len(idx)}


def smart_audio_nframes(audio: np.ndarray, audio_fps: int, sample_rate: int = 16000, **kwargs):
    if audio is not None:
        if audio_fps != sample_rate:
            audio = librosa.resample(y=audio, orig_sr=audio_fps, target_sr=sample_rate)
    num_frames = len(audio) if audio is not None else 0
    return audio, {"fps": sample_rate, "total_num_frames": num_frames}


def load_video_from_path(video: str, use_audio_in_video: bool = True, **kwargs) -> Tuple:
    if "http://" in video or "https://" in video:
        from packaging import version

        if version.parse(torchvision.__version__) < version.parse("0.19.0"):
            logger.warning_once(
                "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0."
            )
    video, _audio, info = torchvision.io.read_video(
        video,
        0.0,
        None,
        pts_unit="sec",
        output_format="TCHW",
    )
    video_fps = info["video_fps"]
    video_metadata = {"fps": video_fps, "total_num_frames": video.shape[0]}
    audio, audio_metadata = None, None
    if use_audio_in_video and _audio.numel() > 0:
        # Average across channels if multi-channel
        audio = torch.mean(_audio, dim=0).numpy()
        audio_fps = info["audio_fps"]
        audio_metadata = {"fps": audio_fps, "total_num_frames": _audio.shape[0]}

    return video, video_metadata, audio, audio_metadata


def load_video_from_bytes_list(video: Union[List[bytes], np.ndarray], **kwargs) -> Tuple:
    """
    Loads video frames from a list of bytes with memory optimization.
    Expects 'fps' in kwargs for metadata.
    """
    if isinstance(video, np.ndarray):
        video = video.tolist()
    if not video:
        raise ValueError("Input video frame list is empty")

    fps_val = kwargs.get("fps", 2.0)
    nframes = len(video)

    # Decode first frame to get dimensions
    with PIL.Image.open(BytesIO(video[0])) as img:
        img = img.convert("RGB")
        w, h = img.size

    T, C = nframes, 3
    # Memory optimization: Allocate uint8 tensor
    video_tensor = torch.empty((T, C, h, w), dtype=torch.uint8)

    for i, frame_bytes in enumerate(video):
        with PIL.Image.open(BytesIO(frame_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            frame_arr = np.array(img)
            # Convert to Tensor (C, H, W)
            frame_t = torch.from_numpy(frame_arr).permute(2, 0, 1)
            video_tensor[i] = frame_t

    video_metadata = {"fps": fps_val, "total_num_frames": nframes}
    return video_tensor, video_metadata, None, None


def load_video_from_bytes(video: bytes, use_audio_in_video: bool = True, **kwargs):
    container = av.open(BytesIO(video))
    video_frames = _read_from_stream(
        container,
        0.0,
        float("inf"),
        "sec",
        container.streams.video[0],
        {"video": 0},
    )
    video_fps = container.streams.video[0].average_rate
    video_metadata = {"fps": video_fps, "total_num_frames": len(video_frames)}
    vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
    video = torch.as_tensor(np.stack(vframes_list)).permute(0, 3, 1, 2)  # t,c,h,w

    audio, audio_metadata = None, None
    if use_audio_in_video and len(container.streams.audio) > 0:
        audio_frames = _read_from_stream(
            container,
            0.0,
            float("inf"),
            "sec",
            container.streams.audio[0],
            {"audio": 0},
        )

        aframes_list = [frame.to_ndarray() for frame in audio_frames]
        if len(aframes_list) > 0:
            aframes = np.concatenate(aframes_list, 1)
            aframes = np.mean(aframes, axis=0)
            audio_fps = container.streams.audio[0].rate
            audio_metadata = {"fps": audio_fps, "total_num_frames": len(aframes_list)}
            audio = aframes

    return video, video_metadata, audio, audio_metadata


def load_video(video: VideoInput, **kwargs):
    if isinstance(video, str):
        return load_video_from_path(video, **kwargs)
    elif isinstance(video, bytes):
        return load_video_from_bytes(video, **kwargs)
    elif isinstance(video, (list, np.ndarray)):
        return load_video_from_bytes_list(video, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported video input type: {type(video)}")


def fetch_videos(videos: List[VideoInput], **kwargs) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """
    Loads, resizes, and samples frames for a list of videos.
    Returns processed video tensors and audio arrays.
    """
    video_inputs, video_metadata_list = [], []
    audio_inputs, audio_metadata_list = [], []

    # Load all videos first
    for video in videos:
        v, v_meta, a, a_meta = load_video(video, **kwargs)
        video_inputs.append(v)
        video_metadata_list.append(v_meta)
        audio_inputs.append(a)
        audio_metadata_list.append(a_meta)

    # Process videos
    processed_videos = [
        smart_video_nframes(smart_resize(v, **kwargs), v_meta["fps"], **kwargs)[0]
        for v, v_meta in zip(video_inputs, video_metadata_list)
    ]

    # Process audio
    processed_audios = [
        smart_audio_nframes(a, a_meta["fps"], **kwargs)[0] if a_meta else None
        for a, a_meta in zip(audio_inputs, audio_metadata_list)
    ]

    # Filter out None audios if necessary or keep structure depending on downstream needs
    # Here we return raw list which might contain None
    return processed_videos, processed_audios


def fetch_videos_metadata(
    videos: List[VideoInput], **kwargs
) -> Tuple[List[torch.Tensor], List[Dict], List[Optional[np.ndarray]], List[Optional[Dict]]]:
    """
    Loads and processes videos and audio, returning full metadata.
    Supports returning raw data if 'fps' is provided in kwargs (direct return mode).
    """
    video_inputs_raw, video_metadata_raw = [], []
    audio_inputs_raw, audio_metadata_raw = [], []

    # Determine if we are in 'direct return' mode (bypassing smart processing)
    direct_return = False
    if "fps" in kwargs:
        direct_return = True
        fps = kwargs.pop("fps")
        if not isinstance(fps, List):
            fps = [fps] * len(videos)
        fps_list = fps
    else:
        fps_list = [None] * len(videos)

    # Load Loop
    for video_item, fps in zip(videos, fps_list):
        # Pass fps kwarg specifically for bytes_list loader
        load_kwargs = kwargs.copy()
        if fps is not None:
            load_kwargs["fps"] = fps

        v, v_meta, a, a_meta = load_video(video_item, **load_kwargs)
        video_inputs_raw.append(v)
        video_metadata_raw.append(v_meta)
        audio_inputs_raw.append(a)
        audio_metadata_raw.append(a_meta)

    if direct_return:
        return video_inputs_raw, video_metadata_raw, audio_inputs_raw, audio_metadata_raw

    # Smart Processing Loop
    video_inputs_final, video_metadata_final = [], []
    audio_inputs_final, audio_metadata_final = [], []

    for v_raw, v_meta in zip(video_inputs_raw, video_metadata_raw):
        processed_v, processed_v_meta = smart_video_nframes(smart_resize(v_raw, **kwargs), v_meta["fps"], **kwargs)
        video_inputs_final.append(processed_v)
        video_metadata_final.append(processed_v_meta)

    for a_raw, a_meta in zip(audio_inputs_raw, audio_metadata_raw):
        # Handle cases where audio might be None
        if a_raw is not None and a_meta is not None:
            # Use 'sample_rate' if available, fallback to 'fps'
            orig_sr = a_meta["fps"]
            processed_a, processed_a_meta = smart_audio_nframes(a_raw, orig_sr, **kwargs)
        else:
            processed_a, processed_a_meta = None, None

        audio_inputs_final.append(processed_a)
        audio_metadata_final.append(processed_a_meta)

    return video_inputs_final, video_metadata_final, audio_inputs_final, audio_metadata_final
