import math
from io import BytesIO
from typing import ByteString, Dict, List, Union

import av
import librosa
import numpy as np
import PIL
import torch
import torchvision
from torchvision.io.video import _read_from_stream
from torchvision.transforms import InterpolationMode, functional

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
    total_frames = video.shape[0]
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

    if nframes > total_frames:
        pad_count = nframes - total_frames
        last_frame = video[-1:].expand(pad_count, -1, -1, -1)  # shape: (pad_count, C, H, W)
        video = torch.cat([video, last_frame], dim=0)
        total_frames = video.shape[0]

    idx = torch.linspace(0, total_frames - 1, int(nframes)).round().long()
    video = video[idx]
    return video


def smart_audio_nframes(audio: np.ndarray, audio_fps: int, sample_rate: int = 16000, **kwargs):
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
    height, width = video.shape[2], video.shape[3]
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


def load_video_from_path(video: str, use_audio_in_video: bool = True, **kwargs):
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
    audio, audio_fps = None, None
    if use_audio_in_video and _audio.numel() > 0:
        audio = torch.mean(_audio, dim=0).numpy()
        audio_fps = info["audio_fps"]
    return video, video_fps, audio, audio_fps


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
    vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
    video = torch.as_tensor(np.stack(vframes_list)).permute(0, 3, 1, 2)  # t,c,h,w

    audio, audio_fps = None, None
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
            audio = aframes

    return video, video_fps, audio, audio_fps


def load_video(video: VideoInput, **kwargs):
    if isinstance(video, str):
        return load_video_from_path(video, **kwargs)
    elif isinstance(video, bytes):
        return load_video_from_bytes(video, **kwargs)
    else:
        raise NotImplementedError


def fetch_videos(videos: List[VideoInput], **kwargs):
    video_inputs, video_fps_list, audio_inputs, audio_fps_list = [], [], [], []
    for video in videos:
        video, video_fps, audio, audio_fps = load_video(video, **kwargs)
        video_inputs.append(video)
        video_fps_list.append(video_fps)
        audio_inputs.append(audio)
        audio_fps_list.append(audio_fps)

    video_inputs = [
        smart_video_nframes(smart_resize(video, **kwargs), video_fps, **kwargs)
        for video, video_fps in zip(video_inputs, video_fps_list)
    ]

    audio_inputs = [
        smart_audio_nframes(audio, audio_fps, **kwargs) for audio, audio_fps in zip(audio_inputs, audio_fps_list)
    ]
    return video_inputs, audio_inputs
