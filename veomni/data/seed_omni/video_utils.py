# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal video IO for the SeedOmni V2 data layer.

Intentionally tiny — just enough to avoid OOM when a clip is large:

1. **decode** only the frames we keep (sample ~``fps`` frames, capped at
   ``max_frames`` by uniform sub-sampling) — we never materialise the whole clip;
2. **aspect-preserving downscale** each frame to ``video_max_pixels`` (OOM guard);
3. optionally extract the **audio** track (off by default — Qwen3-VL has no audio
   modality, so ``use_audio_in_video=False``).

It deliberately does **not** do ``smart_resize`` / temporal frame-factor
alignment / patchify — that model-specific work is owned by the video module's
processor (e.g. ``Qwen3VLVideoProcessor``). See ``docs/seed_omni/design.md``
"Layer 2".
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils import logging
from ...utils.import_utils import is_ffmpeg_available
from ..multimodal.audio_utils import extract_audio_from_video


logger = logging.get_logger(__name__)

VideoInput = Union[List["Image.Image"], List[bytes], bytes, str]


@dataclass
class VideoInputs:
    """Decoded video carried as a SeedOmni V2 conversation item ``value``.

    A single clip is one media item whose ``value`` bundles both streams:

    * ``video`` — sampled-frame tensor ``(T, C, H, W)`` uint8.
    * ``video_fps`` — the fps these sampled frames represent (the data layer's
      memory-bound pre-trim rate). Forwarded as the HF processor's
      ``video_metadata`` fps so it can sub-sample to the model's authoritative
      ``self.fps``; also the basis for the temporal grid timeline
      (``video_second_per_grid = temporal_patch_size / video_fps``).
    * ``audio`` — optional in-video waveform (``None`` when there is no audio
      track or extraction was disabled). When present, downstream modules wrap it
      with ``<|vision_bos|><|audio_bos|> … <|audio_eos|><|vision_eos|>`` and
      time-interleave the streams (see ``design.md`` § av-video).
    * ``audio_fps`` — audio sample rate, kept for the backbone's TMRoPE timeline.
    """

    video: torch.Tensor
    video_fps: float | None = None
    audio: np.ndarray | None = None
    audio_fps: float | None = None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


def _sample_frame_indices(
    total_frames: int,
    video_fps: float | None,
    fps: float,
    max_frames: int | None,
) -> list[int]:
    """Uniform frame indices: keep ~``fps`` frames/sec, then cap at ``max_frames``."""
    if total_frames <= 0:
        return [0]
    stride = max(1, round(video_fps / fps)) if (video_fps and fps) else 1
    indices = list(range(0, total_frames, stride))
    if max_frames and len(indices) > max_frames:
        sel = torch.linspace(0, len(indices) - 1, steps=max_frames).round().long().tolist()
        indices = [indices[i] for i in sel]
    return indices or [0]


def _resize_frames_to_max_pixels(video: torch.Tensor, max_pixels: int | None) -> torch.Tensor:
    """Aspect-preserving downscale of ``(T, C, H, W)`` frames (OOM guard only)."""
    if not max_pixels:
        return video
    _, _, h, w = video.shape
    if h * w <= max_pixels:
        return video
    scale = (max_pixels / (h * w)) ** 0.5
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = F.interpolate(video.float(), size=(new_h, new_w), mode="bilinear", align_corners=False)
    return resized.round().clamp(0, 255).to(torch.uint8)


def _frames_from_list(video: list) -> torch.Tensor:
    """Pre-decoded frames (list of PIL / encoded bytes) → ``(T, C, H, W)`` uint8."""
    pil = []
    for frame in video:
        if isinstance(frame, (bytes, bytearray)):
            pil.append(Image.open(BytesIO(frame)).convert("RGB"))
        else:
            pil.append(frame.convert("RGB") if frame.mode != "RGB" else frame)
    arr = np.stack([np.array(p, dtype=np.uint8) for p in pil])  # (T, H, W, C)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def load_video(
    video: VideoInput,
    fps: float = 2.0,
    max_frames: int | None = None,
    video_max_pixels: int | None = None,
    use_audio_in_video: bool = False,
    **kwargs,
) -> VideoInputs:
    """Decode + sample + OOM-cap one clip into a :class:`VideoInputs` bundle."""
    del kwargs
    audio, audio_fps = None, None

    if isinstance(video, list):
        # Frames already decoded upstream — no fps metadata, keep all then cap.
        frames = _frames_from_list(video)
        indices = _sample_frame_indices(frames.shape[0], None, fps, max_frames)
        frames = frames[indices]
    else:
        # str (path/URL) or bytes — needs the ffmpeg/torchcodec stack. Decode only
        # the sampled frames so a long clip never lands fully in memory.
        if not is_ffmpeg_available():
            raise RuntimeError(
                "ffmpeg is not available; required to decode str/bytes video. Install ffmpeg "
                "or feed pre-decoded frames (list of PIL / bytes)."
            )
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video, device="cpu", num_ffmpeg_threads=0)
        meta = decoder.metadata
        indices = _sample_frame_indices(max(1, meta.num_frames), meta.average_fps, fps, max_frames)
        frames = decoder.get_frames_at(indices).data  # (T, C, H, W) uint8
        if use_audio_in_video:
            max_audio_duration = (meta.duration_seconds or 60.0) + 1.0
            audio, audio_fps = extract_audio_from_video(video, max_duration_seconds=max_audio_duration)

    frames = _resize_frames_to_max_pixels(frames, video_max_pixels)
    return VideoInputs(video=frames, video_fps=fps, audio=audio, audio_fps=audio_fps)


def fetch_videos(
    videos: list[VideoInput],
    fps: float = 2.0,
    max_frames: int | None = None,
    video_max_pixels: int | None = None,
    use_audio_in_video: bool = False,
    **kwargs,
) -> list[VideoInputs]:
    """Decode + OOM-cap a list of clips into :class:`VideoInputs` bundles."""
    del kwargs
    return [
        load_video(
            v,
            fps=fps,
            max_frames=max_frames,
            video_max_pixels=video_max_pixels,
            use_audio_in_video=use_audio_in_video,
        )
        for v in videos
    ]


__all__ = ["VideoInput", "VideoInputs", "load_video", "fetch_videos"]
