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

"""One decode path for every modality, shared by training and inference.

A modality is one entry in :data:`MEDIA_FETCHERS`. Everything downstream — the
conversation carrier, the per-module preprocessors, the metadata on
``ConversationItem.meta`` — is written against the pair shape every fetcher
returns, so adding a modality (action / 3d / camera) is that one entry plus its
fetcher, and touches neither the training transform nor the inference request
builder.

This module exists because those two used to decode differently: the training
transform loaded through the fetchers while request building had its own
PIL-only path that attached no metadata at all, so an audio item built for
inference carried no sampling rate — a fact its consumers require and cannot
recover. Sharing the table is what makes "the same clip" mean the same thing on
both sides.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import torch

from ....utils.import_utils import is_video_audio_available
from .audio import fetch_audios, save_audio
from .image import fetch_images, save_image


if is_video_audio_available():
    from .video import fetch_videos, save_video
else:

    def fetch_videos(*args, **kwargs):
        # Raising beats the empty list this used to return: a caller only gets
        # here having named a clip, so dropping it silently turns a missing
        # system dependency into a count mismatch reported much later against
        # the wrong cause (training) or a text-only answer to a question about a
        # video (inference).
        raise RuntimeError(
            "video refs were passed but the video/audio decode stack is unavailable. "
            "Install the optional dependencies (ffmpeg + torchcodec) to decode clips."
        )

    def save_video(path: str, *args, **kwargs):
        raise RuntimeError(
            f"save_video: a generated clip is to be written to {path}, but the video/audio stack is "
            "unavailable. Install the optional dependencies (ffmpeg + torchcodec) to write clips."
        )


def _fetch_image_items(refs: list, **kwargs) -> list[tuple[torch.Tensor, dict[str, Any]]]:
    """``fetch_images`` in the ``(payload, meta)`` shape the other fetchers use.

    An image states nothing the way a clip states its rate or a video its frame
    timeline — the tensor is the whole story — so the meta is empty rather than
    invented. Adapted here rather than in ``utils/image.py`` so that function
    stays usable on its own.
    """
    return [(tensor, {}) for tensor in fetch_images(refs, **kwargs)]


# Which fetcher decodes each modality's refs, keyed by the ``ConversationItem``
# type those refs pair with.
#
# Every fetcher returns ``[(payload, meta), ...]``: payload becomes the item's
# ``value``, meta is merged into the item's ``meta``. See ``media_metadata.py``
# for why the loaded facts go on ``meta``. Sound belonging to a video clip is
# not a fourth entry — it rides on the video payload's ``VideoInputs.audio``,
# so one clip stays one item however many tracks it has.
MEDIA_FETCHERS = {
    "image": _fetch_image_items,
    "video": fetch_videos,
    "audio": fetch_audios,
}


def fetch_media(
    media_refs: dict[str, list[Any]],
    context: str,
    knobs: Mapping[str, Any] | None = None,
) -> dict[str, list[tuple[Any, dict[str, Any]]]]:
    """Decode every modality's refs into ``{type: [(payload, meta), ...]}``.

    ``knobs`` is one flat bag of loading options (``image_max_pixels`` / ``fps`` /
    ``max_frames`` / ``video_max_pixels`` / ``audio_sampling_rate`` /
    ``use_audio_in_video``) offered to every fetcher; each drops the keywords it
    does not read, so a knob that one modality ignores is ignored rather than
    rejected. That tolerance is what lets one caller-side config serve all
    modalities. It is an explicit mapping rather than ``**kwargs`` because
    callers forward whole config bags in here, and a bag holding a key that
    collides with this signature would fail as a duplicate-argument TypeError.

    ``context`` names the caller in the unknown-modality error, which is
    otherwise hard to place: the same message can come from a dataset
    preprocessor declaring refs or from a hand-built inference request.
    """
    unknown = set(media_refs) - set(MEDIA_FETCHERS)
    if unknown:
        # Named rather than silently dropped: refs nothing can decode would
        # leave the media turns unpaired, and the failure would surface much
        # later as a count mismatch naming the wrong cause.
        raise ValueError(
            f"{context}: refs declared for {sorted(unknown)}, which has no fetcher. "
            f"Known modalities: {sorted(MEDIA_FETCHERS)}."
        )
    return {type_: MEDIA_FETCHERS[type_](refs, **(knobs or {})) for type_, refs in media_refs.items() if refs}


class MediaSaver(NamedTuple):
    suffix: str
    save: Callable[[str, Any, Mapping[str, Any] | None], None]


# The write side of :data:`MEDIA_FETCHERS`, keyed the same way. Every saver is
# ``save(path, value, meta)`` with ``meta`` the item's whole
# ``ConversationItem.meta``: the caller does not pick fields out of it, each
# saver reads what its modality needs. So a modality that gains a fact on meta
# (a colour space, a channel layout) is written correctly without the caller
# changing, and a new modality is one entry here.
#
# Sound belonging to a clip is not an entry for the reason it is not a fetcher:
# a generated clip with sound is one ``video`` item carrying ``VideoInputs``
# with both tracks, and ``save_video`` muxes them into one file.
MEDIA_SAVERS = {
    "image": MediaSaver(".png", save_image),
    "audio": MediaSaver(".wav", save_audio),
    "video": MediaSaver(".mp4", save_video),
}


__all__ = [
    "MEDIA_FETCHERS",
    "MEDIA_SAVERS",
    "MediaSaver",
    "fetch_media",
]
