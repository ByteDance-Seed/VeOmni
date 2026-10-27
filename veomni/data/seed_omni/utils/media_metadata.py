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

"""What the data layer states about a decoded media item, on its ``meta``.

Its own module rather than a section of ``audio.py`` or ``video.py`` because
both fill it in: :class:`AudioMetadata` describes a standalone clip *and* the
track lifted out of a video, so putting it in either loader would make the
other import it sideways. Nothing here imports the loaders, so the model
modules that read these off ``meta`` can name the types too.

**Why ``meta`` and not the item's ``value``.** ``value`` does not survive the
item's lifecycle — the first encoder to touch the item overwrites it with
embeddings (the audio tower replaces a waveform with mel features, the vision
tower replaces frames with patches), and the backbone then overwrites it again
with hidden states. The timeline is needed *after* that: TMRoPE places a clip on
the shared clock, the codec checks the rate it was handed, and ``OmniInferencer``
writes a wav. ``meta`` is the only channel that lives that long.

One consequence worth stating: :class:`AudioMetadata` describes a standalone
``type="audio"`` item *and* the audio track inside a video, under the same
:data:`AUDIO_METADATA_KEY`. A reader that wants a sample rate spells it one way
and does not have to know which modality the sound arrived as.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from transformers.video_utils import VideoMetadata as HFVideoMetadata


# Keys these live under on ``ConversationItem.meta``.
VIDEO_METADATA_KEY = "video_metadata"
AUDIO_METADATA_KEY = "audio_metadata"


@dataclass
class VideoMetadata(HFVideoMetadata):
    """Timeline of the kept frames, expressed on the *source* clip's axis.

    ``transformers.video_utils.VideoMetadata`` plus what VeOmni needs on top:
    :attr:`requested_fps`, patch-aware :meth:`frame_timestamps`, and
    :meth:`seconds_per_patch` for TMRoPE. Being the HF type, it can be handed to
    an HF video processor as is — but hand it a copy, because the processor
    writes ``frames_indices`` back when it re-samples.

    ``fps`` and ``total_num_frames`` describe the clip as it was on disk, and
    ``frames_indices`` maps each kept frame back onto it. Keeping the source
    axis (rather than only the trimmed clip's own rate) is what makes
    timestamps recoverable: Qwen3-VL derives them as ``index / fps`` — see
    ``Qwen3VLChatTemplate._calculate_timestamps`` in ``veomni/data/chat_template.py``.

    **``fps`` is always a real rate.** Construction refuses a missing one, which
    is what makes inheriting safe: the HF type's fallbacks — ``sampled_fps``
    returning ``24`` and the Qwen3-VL processor writing ``metadata.fps = 24``
    back — only fire on ``fps is None``, and that value cannot exist here. A
    container states its rate; a pre-decoded frame list has none, so its caller
    declares one (``frames_fps``) or ``load_video`` refuses the list.

    A *generated* clip is its own source, and so needs no second type: nothing
    was sub-sampled, so ``fps`` is both the source rate and the rate it plays
    back at, and ``frames_indices`` is the full range (the default when left
    unset). That is also the rate :func:`~.video.save_video` writes the
    container at.

    ``duration`` is filled from ``total_num_frames / fps`` when not given, so it
    is always the *source* clip's length — the span to compare the sound lifted
    out of it against (:attr:`AudioMetadata.duration_seconds`). Both are
    measured from the container's zero.

    Consumers that want a rate for timing want a time, and ask for it:
    :meth:`frame_timestamps` for per-frame times, :meth:`seconds_per_patch` for
    the single scalar TMRoPE scales by. Both derive from the source axis, so
    neither can be paired with the wrong rate.
    """

    requested_fps: float | None = None
    """The rate ``load_video`` was asked to pre-trim to. Recorded because the
    achieved rate usually differs: the stride is integer-rounded and
    ``max_frames`` may thin the result further. Timing must never be derived
    from it — that is the bug in PR #1165."""

    def __post_init__(self) -> None:
        fps = self.fps
        if not is_frame_rate(fps):
            raise ValueError(
                f"VideoMetadata: fps={fps!r} is not a frame rate. Every clip states the rate its frames are on: "
                "a container carries one, a pre-decoded frame list declares one via `frames_fps`, and a "
                "generated clip is written at one. Do not substitute requested_fps — that is the #1165 bug."
            )
        if self.frames_indices is None:
            self.frames_indices = list(range(self.total_num_frames))
        if self.duration is None:
            self.duration = self.total_num_frames / float(fps)

    def frame_timestamps(self, temporal_patch_size: int = 1) -> list[float]:
        """When each kept frame happens, in seconds on the source clip.

        This lives here, rather than at each call site, because the pairing is
        the whole trap: ``frames_indices`` are source-frame numbers, so they
        divide by the *source* ``fps``. Dividing them by a target sampling rate
        is what PR #1165 fixed — frame 300 of a 30 fps clip is 10 s, but 300/2
        reads as 150 s. No caller has to know that if no caller does the
        division.

        ``temporal_patch_size`` folds frames into patches the way the vision
        tower does: pad to a whole number of patches by repeating the last
        frame, then take each patch's mid-time. Matches
        ``Qwen3VLChatTemplate._calculate_timestamps``.
        """
        _check_temporal_patch_size(temporal_patch_size)
        indices = list(self.frames_indices)
        if not indices:
            return []
        # Copied above: padding in place would append duplicate frames to the
        # caller's metadata. Repeating the last index mirrors how the processor
        # pads the frames themselves, so times stay attached to real frames.
        if len(indices) % temporal_patch_size:
            indices.extend([indices[-1]] * (temporal_patch_size - len(indices) % temporal_patch_size))
        seconds = [index / self.fps for index in indices]
        return [
            (seconds[i] + seconds[i + temporal_patch_size - 1]) / 2
            for i in range(0, len(seconds), temporal_patch_size)
        ]

    def seconds_per_patch(self, temporal_patch_size: int = 1) -> float:
        """Seconds one temporal patch spans on the source clock.

        The scalar TMRoPE scales its temporal row by (Qwen3-Omni:
        ``second_per_grid * position_id_per_seconds``), so it decides where
        every position id in and after the clip lands. Upstream computes it as
        ``temporal_patch_size / fps`` with ``fps`` being the rate the clip was
        *requested* at, which is the same axis error as #1165 wearing different
        clothes: a 25 fps source trimmed to "2 fps" keeps frames 0.48 s apart,
        not 0.5 s, and ``max_frames`` can miss by far more than that.

        Derived from the kept frames' real spacing instead. When ``max_frames``
        spaced them unevenly this is their mean — a single scalar cannot say
        more, and the mean keeps the clip's total span right even though
        individual frames drift inside it. Use :meth:`frame_timestamps` where
        per-frame accuracy matters.
        """
        _check_temporal_patch_size(temporal_patch_size)
        indices = self.frames_indices
        if len(indices) >= 2 and indices[-1] > indices[0]:
            frame_spacing = (indices[-1] - indices[0]) / (len(indices) - 1) / self.fps
        else:
            # A lone kept frame has no spacing to measure; it stands for one
            # source frame period, which is the only defensible reading.
            frame_spacing = 1.0 / self.fps
        return temporal_patch_size * frame_spacing


def is_frame_rate(value: object) -> bool:
    """A positive, finite number. ``bool`` is excluded because it is an ``int``:
    ``True`` would otherwise pass as 1 fps."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _check_temporal_patch_size(temporal_patch_size: int) -> None:
    if isinstance(temporal_patch_size, bool) or not isinstance(temporal_patch_size, int) or temporal_patch_size < 1:
        raise ValueError(f"temporal_patch_size must be an int >= 1, got {temporal_patch_size!r}")


@dataclass
class AudioMetadata:
    """Rate and length of one clip, standalone or lifted out of a video.

    * ``sampling_rate`` — the clip's own rate, never normalised. The data layer
      refuses to pick one because there is no single right one: a Qwen3-Omni
      thinker's tower wants 16 kHz and the codec that tokenises assistant speech
      wants 24 kHz. See ``veomni/data/seed_omni/utils/audio.py``.
    * ``num_samples`` — length of the waveform. Not redundant with the payload:
      by the time the backbone wants a duration, ``value`` holds mel features
      and the sample count is no longer recoverable from it.

    Both describe the clip *as decoded*, and a module that resamples for its own
    tower must not write its target rate back here. Rewriting only
    ``sampling_rate`` leaves ``num_samples`` on the old axis and
    :attr:`duration_seconds` then reports a length the clip never had; rewriting
    both discards the source axis every other module still reads. Resample a
    local copy and leave the timeline alone — the clip is the same number of
    seconds long either way, and seconds are what the shared timeline is in.
    """

    sampling_rate: int | None = None
    num_samples: int = 0

    @property
    def duration_seconds(self) -> float | None:
        """Seconds of wall clock, which is what puts the clip on TMRoPE's shared
        timeline. A rate read wrong drifts the clip against a video track rather
        than making it sound wrong — 22.05 kHz read as 16 kHz turns 4 s into
        5.54 s."""
        if not self.sampling_rate:
            return None
        return self.num_samples / float(self.sampling_rate)


__all__ = ["AUDIO_METADATA_KEY", "VIDEO_METADATA_KEY", "AudioMetadata", "VideoMetadata"]
