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

"""Minimal video IO for the SeedOmni data layer.

Intentionally tiny — just enough to avoid OOM when a clip is large:

1. **decode** only the frames we keep (sample ~``fps`` frames, capped at
   ``max_frames`` by uniform sub-sampling) — we never materialise the whole clip;
2. **aspect-preserving downscale** each frame to ``video_max_pixels`` (OOM guard);
3. optionally extract the **audio** track (off by default — Qwen3-VL has no audio
   modality, so ``use_audio_in_video=False``).

It deliberately does **not** do ``smart_resize`` / temporal frame-factor
alignment / patchify — that model-specific work is owned by the video module's
processor (e.g. ``Qwen3VLVideoProcessor``).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ....utils import logging
from ....utils.import_utils import is_ffmpeg_available
from ...multimodal.audio_utils import extract_audio_from_video
from .media_metadata import AUDIO_METADATA_KEY, VIDEO_METADATA_KEY, AudioMetadata, VideoMetadata, is_frame_rate


logger = logging.get_logger(__name__)

VideoInput = Union[List["Image.Image"], List[bytes], bytes, str]


@dataclass
class VideoInputs:
    """Decoded video carried as a SeedOmni conversation item ``value``.

    Payload only — both streams of one clip, so a clip stays one item:

    * ``video`` — kept-frame tensor ``(T, C, H, W)`` uint8.
    * ``audio`` — optional in-video waveform (``None`` when there is no audio
      track or extraction was disabled). No module consumes it yet.

    The two timelines are *not* here. They ride on the item's
    ``ConversationItem.meta`` under :data:`VIDEO_METADATA_KEY` and
    :data:`AUDIO_METADATA_KEY`, because ``value`` does not survive the item's
    lifecycle — the vision encoder overwrites it with patches — while the
    backbone needs the timeline after that point. ``meta`` is also what lets a
    reader find an audio rate without knowing whether the clip arrived as an
    ``"audio"`` item or inside a video.
    """

    video: torch.Tensor
    audio: np.ndarray | None = None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


def _warn_if_audio_hit_the_cap(audio_metadata: AudioMetadata, max_duration_seconds: float | None) -> None:
    """Say so when the OOM cap, not the audio track, decided where sound ends.

    A sound track that simply stops early is ordinary and silent about it; a
    track cut by our own buffer limit is a clip whose two streams no longer
    span the same time, which is the failure mode worth a log line.
    """
    decoded = audio_metadata.duration_seconds
    if max_duration_seconds is None or decoded is None:
        return
    # The extractor stops on a whole frame once it has ``int(cap * rate)``
    # samples, so a capped clip can land just under the cap. One sample period
    # of slack, not a fixed epsilon.
    slack = 1.0 / audio_metadata.sampling_rate if audio_metadata.sampling_rate else 0.0
    if decoded >= max_duration_seconds - slack:
        # Message kept free of per-clip numbers: ``warning_once`` memoises on the
        # formatted string, and this runs per sample in a DataLoader worker.
        logger.warning_once(
            "In-video audio stopped at the OOM cap, so it may be truncated while the video frames are "
            f"not, and meta[{AUDIO_METADATA_KEY!r}].duration_seconds then understates the clip. Compare "
            f"it against meta[{VIDEO_METADATA_KEY!r}].duration before using either as the span."
        )
        logger.debug(f"audio capped at {max_duration_seconds:.3f}s, decoded {decoded:.3f}s")


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
    frames_fps: float | None = None,
    **kwargs,
) -> tuple[VideoInputs, dict[str, Any]]:
    """Decode + sample + OOM-cap one clip into ``(payload, meta)``.

    ``meta`` is the timeline entries to merge into the item's
    ``ConversationItem.meta`` — :data:`VIDEO_METADATA_KEY` always, and
    :data:`AUDIO_METADATA_KEY` when the clip has sound. Same shape
    :func:`~..audio.fetch_audios` returns, so the transform treats every
    modality the same way.

    ``frames_fps`` is the rate a pre-decoded frame list was taken at, and only a
    list reads it; a container states its own. It is the list's *source* rate,
    so ``fps`` still picks which of those frames are kept.
    """
    del kwargs
    audio, audio_metadata = None, None

    if isinstance(video, list):
        # Frames decoded upstream carry no timing of their own, so the caller
        # states the rate they were taken at. With it the list is a source clip
        # like any other, re-sampled to ``fps`` the same way a container is.
        if frames_fps is None:
            raise ValueError(
                "load_video: a pre-decoded frame list carries no frame rate, so its frames have no times. "
                "Declare the rate they were taken at with `frames_fps` (mm_configs.frames_fps), or pass the "
                "clip as a path / bytes so the container states it."
            )
        frames = _frames_from_list(video)
        source_fps = frames_fps
        source_num_frames = frames.shape[0]
        indices = _sample_frame_indices(source_num_frames, source_fps, fps, max_frames)
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
        source_meta = decoder.metadata
        source_fps = source_meta.average_fps
        if not source_fps:
            # torchcodec reports the rate as optional; without one the kept frames
            # have no times and the sound has no clock to be placed on.
            # ``VideoMetadata`` refuses it anyway — said here so the message
            # names the clip's container, not the dataclass.
            raise ValueError(
                "load_video: the container states no frame rate, so its frames have no times. Re-encode the "
                "clip with timing metadata, or decode it to frames and declare `frames_fps`."
            )
        source_num_frames = max(1, source_meta.num_frames)
        indices = _sample_frame_indices(source_num_frames, source_fps, fps, max_frames)
        frames = decoder.get_frames_at(indices).data  # (T, C, H, W) uint8
        if use_audio_in_video:
            # The cap is an OOM bound, so it has to sit above the clip's real
            # length or it silently shortens the sound track while every video
            # frame is kept — the two tracks then describe different spans of
            # the same clip and nothing says so. Fall back to the video's own
            # span (frames / rate) rather than a fixed guess when the container
            # does not state a duration.
            video_duration = source_meta.duration_seconds or source_num_frames / source_fps
            max_audio_duration = video_duration + 1.0
            audio, audio_rate = extract_audio_from_video(video, max_duration_seconds=max_audio_duration)
            if audio is not None:
                audio_metadata = AudioMetadata(sampling_rate=audio_rate, num_samples=len(audio))
                _warn_if_audio_hit_the_cap(audio_metadata, max_audio_duration)

    frames = _resize_frames_to_max_pixels(frames, video_max_pixels)
    meta: dict[str, Any] = {
        VIDEO_METADATA_KEY: VideoMetadata(
            fps=source_fps,
            total_num_frames=source_num_frames,
            frames_indices=indices,
            requested_fps=fps,
        )
    }
    if audio_metadata is not None:
        meta[AUDIO_METADATA_KEY] = audio_metadata
    return VideoInputs(video=frames, audio=audio), meta


def fetch_videos(
    videos: list[VideoInput],
    fps: float = 2.0,
    max_frames: int | None = None,
    video_max_pixels: int | None = None,
    use_audio_in_video: bool = False,
    frames_fps: float | None = None,
    **kwargs,
) -> list[tuple[VideoInputs, dict[str, Any]]]:
    """Decode + OOM-cap a list of clips into ``(payload, meta)`` pairs.

    A repeated ref (the same path / bytes object appearing more than once in the
    list) is decoded once and deep-copied on reuse, so duplicates don't pay a
    second decode and each item carries an independent bundle. The metadata is
    copied along with the payload: it lands on a per-item ``meta`` dict that
    modules write into, so sharing one instance would let the first writer
    reach the other item."""
    del kwargs
    cache: dict = {}
    out: list[tuple[VideoInputs, dict[str, Any]]] = []
    for v in videos:
        # Hashable refs (str / bytes) key by value; other refs key by identity.
        key = v if isinstance(v, (str, bytes)) else id(v)
        if key in cache:
            out.append(copy.deepcopy(cache[key]))
        else:
            cache[key] = load_video(
                v,
                fps=fps,
                max_frames=max_frames,
                video_max_pixels=video_max_pixels,
                use_audio_in_video=use_audio_in_video,
                frames_fps=frames_fps,
            )
            out.append(cache[key])
    return out


def save_video(
    path: str,
    video: VideoInputs | torch.Tensor,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Write one generated clip, muxing its sound track when it has one.

    ``meta`` is the item's whole ``ConversationItem.meta``, the same argument
    every saver takes. This one reads ``meta[VIDEO_METADATA_KEY].fps`` and, for a
    clip with sound, ``meta[AUDIO_METADATA_KEY].sampling_rate`` — both on the one
    item, because a clip with sound is one item (see ``VideoInputs``).

    The inverse of :func:`load_video`, and it keeps :func:`~..audio.save_audio`'s
    contract on the rates: both must come from the module that produced the clip,
    so both are refused rather than defaulted. ``write_video_audio`` would
    otherwise infer a missing audio rate from ``samples / duration``, and a wrong
    rate on either stream writes a file that plays at the wrong speed or drifts
    out of lip sync — neither distinguishable from a model that generated it
    badly.

    ``video`` is the same :class:`VideoInputs` carrier the data layer decodes
    into, so a generated clip with sound stays one item the way a loaded one
    does; a bare ``(T, C, H, W)`` uint8 tensor is accepted for a silent clip.
    ``audio`` may be a numpy array (as decoded) or a tensor (as a vocoder emits
    it), mono ``(L,)`` or ``(channels, L)``. ``fps`` has to be whole — see the
    check below for why a fractional rate is refused rather than truncated.
    """
    from ...multimodal.video_utils import write_video_audio

    meta = meta or {}
    fps = getattr(meta.get(VIDEO_METADATA_KEY), "fps", None)
    audio_sampling_rate = getattr(meta.get(AUDIO_METADATA_KEY), "sampling_rate", None)
    frames, audio = (video.video, video.audio) if isinstance(video, VideoInputs) else (video, None)
    if fps is None:
        raise ValueError(
            f"save_video: the video item for {path} carries no meta[{VIDEO_METADATA_KEY!r}] fps; the emitting "
            f"module must declare the rate it generated at. A generated clip is its own source, so its "
            f"VideoMetadata states the playback rate and keeps every frame."
        )
    # Validate the rate before doing any arithmetic on it. ``write_video_audio``
    # divides by it (``len(video) / fps``) and then stores the container rate as
    # ``int(fps)``, so every bad shape fails somewhere inside the muxer with a
    # message about avcodec or a ZeroDivisionError — naming none of them the rate.
    # ``VideoMetadata`` checks at construction, but it is a mutable dataclass and
    # a producer may set ``fps`` afterwards.
    if not is_frame_rate(fps):
        raise ValueError(
            f"save_video: the video item for {path} declares fps={fps!r}, which is not a positive frame rate. "
            f"The emitting module must state the rate it generated at."
        )
    if fps != int(fps):
        # A fractional rate is truncated by that ``int(fps)``: an NTSC 23.976 clip
        # would be written at 23. Refused rather than written, for the same reason
        # a missing rate is — the file plays at the wrong speed and nothing
        # downstream can tell that from a model that generated it badly. A producer
        # that genuinely generates at a fractional rate needs the writer taught to
        # carry one (av takes a Fraction) before this can be relaxed.
        raise ValueError(
            f"save_video: the video item for {path} declares fps={fps!r}, which the container writer can only "
            f"store truncated to {int(fps)}. Generate at a whole number of frames per second, or teach "
            f"write_video_audio to carry a fractional rate."
        )
    if not torch.is_tensor(frames):
        raise TypeError(
            f"save_video: the video item for {path} holds {type(frames).__name__} frames, expected a "
            f"(T, C, H, W) uint8 tensor — the shape load_video produces."
        )
    if frames.ndim != 4:
        raise ValueError(
            f"save_video: the video item for {path} has shape {tuple(frames.shape)}, which is not a single "
            f"(T, C, H, W) clip. Emit one item per clip."
        )
    if frames.shape[0] == 0:
        raise ValueError(
            f"save_video: the video item for {path} carries no frames. The emitting module produced an empty "
            f"clip — writing it would leave a playable-looking file with nothing in it."
        )
    if frames.dtype != torch.uint8:
        # The write casts with ``.to(torch.uint8)``, which truncates. A VAE
        # decoder — the usual producer here — emits floats in [0, 1], every one of
        # which truncates to 0: a playable, entirely black clip. Which scale and
        # clamp to use is the producer's call, not a guess this function can make,
        # and getting it wrong looks exactly like a bad generation.
        raise ValueError(
            f"save_video: the video item for {path} holds {frames.dtype} frames, expected uint8. Convert to "
            f"8-bit pixels first (a [0, 1] float clip needs scaling by 255, not a cast) — casting here would "
            f"truncate a float clip to black."
        )
    if audio is not None and audio_sampling_rate is None:
        raise ValueError(
            f"save_video: the video item for {path} has a sound track but no meta[{AUDIO_METADATA_KEY!r}] "
            f"rate. Left to the muxer the rate is inferred from the frame count, which silently resamples "
            f"the speech to whatever makes the two tracks the same length."
        )

    pil_frames = [Image.fromarray(frame.permute(1, 2, 0).to(torch.uint8).cpu().numpy()) for frame in frames]
    write_video_audio(
        video=pil_frames,
        audio=_audio_to_muxable(audio, path) if audio is not None else None,
        output_path=path,
        fps=fps,
        audio_sample_rate=audio_sampling_rate,
    )


def _audio_to_muxable(audio: np.ndarray | torch.Tensor, path: str) -> torch.Tensor:
    """Normalize a sound track to the ``(channels, samples)`` tensor the muxer takes.

    ``float()`` for the same reason :func:`~..audio.save_audio` needs it: under
    mixed precision a vocoder emits ``bfloat16``, which numpy has no dtype for.
    """
    waveform = torch.as_tensor(audio) if not torch.is_tensor(audio) else audio
    waveform = waveform.detach().float().cpu().squeeze()
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim != 2:
        raise ValueError(
            f"save_video: the sound track of the video item for {path} has shape {tuple(waveform.shape)}, "
            f"which is neither mono (L,) nor (channels, L)."
        )
    return waveform


__all__ = [
    "AUDIO_METADATA_KEY",
    "VIDEO_METADATA_KEY",
    "AudioMetadata",
    "VideoInput",
    "VideoInputs",
    "VideoMetadata",
    "load_video",
    "fetch_videos",
    "save_video",
]
