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

"""SeedOmni multimodal data transform.

Reads a raw jsonl-style sample (already source-tagged conversations + media
paths/bytes) and returns ``[{"conversation_list": [...]}]`` where each
item is a :class:`~veomni.models.seed_omni.utils.conversation.ConversationItem`
with ``type`` / ``value`` / ``role`` (empty ``meta`` at the data boundary).

Understanding vs generation images both use ``type="image"``; ``role`` distinguishes
them (``user`` = SigLIP input, ``assistant`` = VQVAE target).

This transform is intentionally minimal — it does **only**:

1. Source-specific conversation normalization (delegated to
   ``veomni.data.seed_omni.preprocess``) so that downstream code sees a uniform
   ``[[role, (type, value), ...], ...]`` structure regardless of the upstream
   dataset.
2. Image IO + an aspect-preserving downscale to ``image_max_pixels`` (delegated
   to ``utils/image.fetch_images``), followed by PIL → uint8 ``torch.Tensor`` of
   shape ``(C, H, W)``.  That downscale is an OOM guard, not a resize to the
   model's grid: images are **not** ``smart_resize``-d, **not** normalized,
   **not** patchified, and **not** wrapped in any processor-specific feature
   dict — those steps are owned by the vision encoder module (e.g.
   ``JanusSiglip`` / ``JanusVqvae``) at forward time.
3. Conversation list assembly — pair each ``("image", None)`` tuple with the
   next image tensor in source order and attach ``role`` per item.

Anything else (chat-template formatting, tokenization, boundary marker
emission, ``input_ids`` / ``labels`` / ``attention_mask`` construction,
position id calculation, image normalization, image patchification) is
deliberately **not** done here — it belongs in model modules.

Audio turns (``("audio", _)``) are paired with the per-sample ``audios`` list and
loaded by ``utils/audio.fetch_audios`` **at the source's own rate**, which each
item then declares in ``meta["sampling_rate"]``. The data layer does not pick a
rate because there is no single right one: a Qwen3-Omni thinker's tower wants
16 kHz while the codec that turns assistant speech into talker targets wants
24 kHz, and the rate is also what places a clip on TMRoPE's shared clock.
``role`` separates the two uses of the same ``type="audio"`` row — ``user`` is an
input to an audio encoder, ``assistant`` is the speech a talker is trained to
say, i.e. a label.

Video turns (``("video", _)``) are paired with the per-sample ``videos`` list
and decoded via ``fetch_videos`` into a :class:`VideoInputs` bundle — the
sampled-frame tensor plus the optional in-video audio waveform.  Both streams
ride on one media item; the intended split across a video and an audio module is
recorded in ``docs/seed_omni/av_video_design.md`` (not implemented yet).

Registered as ``data_type: seedomni`` in
``veomni/data/data_transform.py``::

    args:
      data:
        data_type: seedomni
"""

from __future__ import annotations

import json
from typing import Any, List

import torch

from ...models.seed_omni.utils.conversation import ConversationItem
from ...utils.import_utils import is_video_audio_available
from ..data_transform import DATA_TRANSFORM_REGISTRY
from .preprocess import conv_preprocess
from .utils.audio import SAMPLING_RATE_KEY, fetch_audios
from .utils.image import fetch_images


if is_video_audio_available():
    from .utils.video import VideoInputs, fetch_videos
else:
    VideoInputs = None

    def fetch_videos(*args, **kwargs):
        return []


# Tuple-form turn entry used by ``conv_preprocess``: ``(type, value)`` or the
# optional 3-tuple ``(type, value, meta)`` where ``meta`` is merged into the
# resulting ``ConversationItem.meta`` (e.g. ``{_IMG_TAG_KEY: "und"}``). For
# ``type == "image"`` / ``"video"`` the inline ``value`` is always ``None`` —
# the actual tensor is pulled from the per-sample media list in source order.
_TupleTurn = List  # ``[role: str, (type, value) | (type, value, meta), ...]``


def _is_rate(declared: Any, rate: int) -> bool:
    """Whether a preprocessor's declared rate agrees with the decoded one.

    Non-numeric counts as disagreement so the caller's message names the field,
    rather than a bare ``int()`` failure naming nothing.
    """
    try:
        return int(declared) == rate
    except (TypeError, ValueError):
        return False


def _build_conversation_list(
    constructed: list[_TupleTurn],
    image_tensors: list[torch.Tensor],
    video_inputs: list[VideoInputs],
    audio_clips: list[tuple[Any, int]] | None = None,
) -> list[ConversationItem]:
    """Flatten ``[[role, (type, value) | (type, value, meta), ...], ...]`` into
    :class:`ConversationItem` rows and pair image / video turns with
    ``image_tensors`` / ``video_inputs`` in source order.

    A video turn's value is a :class:`VideoInputs` bundling the decoded frame
    tensor and the optional in-video audio waveform — the single carrier the
    downstream video / audio modules each read their own stream from.

    An optional 3-tuple ``(type, value, meta)`` carries per-item ``meta``
    (merged into ``ConversationItem.meta``), e.g. ``{_IMG_TAG_KEY: "und"}``.
    Pairing is pure sequential — every ``("image", _)`` turn consumes the next
    image tensor and every ``("video", _)`` turn consumes the next video bundle,
    so the caller's preprocessor is responsible for emitting a ref list whose
    length matches the flattened image/video entry count.

    Raises:
        ValueError: if the number of ``("image"|"video", _)`` turns doesn't
            match the number of supplied media — that means upstream
            preprocessor / dataset disagree on count and we refuse to
            silently misalign.
    """
    image_iter = iter(image_tensors)
    video_iter = iter(video_inputs)
    audio_iter = iter(audio_clips or [])
    image_consumed = 0
    video_consumed = 0
    audio_consumed = 0
    out: list[ConversationItem] = []
    for turn in constructed:
        if not turn:
            continue
        role = turn[0]
        assert role in ["user", "assistant"], f"role must be user or assistant, got {role}"
        for entry in turn[1:]:
            assert len(entry) in (2, 3), (
                f"turn entry must be a (type, value) or (type, value, meta) tuple, got {entry}"
            )
            type_ = entry[0]
            value = entry[1]

            # 2-tuples keep an empty meta.
            meta = dict(entry[2]) if len(entry) == 3 else {}
            if type_ == "image":
                value: torch.Tensor = next(image_iter)
                image_consumed += 1
            elif type_ == "video":
                value: VideoInputs = next(video_iter)
                video_consumed += 1
            elif type_ == "audio":
                # The rate rides along rather than being normalised away: the
                # modules that read this item disagree on what they want (16 kHz
                # tower, 24 kHz codec) and each resamples for itself, so the
                # data layer's job is to say what it loaded, not to choose.
                value, rate = next(audio_iter)
                declared = meta.get(SAMPLING_RATE_KEY)
                if declared is not None and not _is_rate(declared, rate):
                    # The decoded header wins any argument, so a preprocessor
                    # asserting a different rate is stating something false
                    # rather than expressing a preference — and the effect is a
                    # silent misplacement on TMRoPE's clock, not a bad-sounding
                    # clip, so it is worth stopping for.
                    raise ValueError(
                        f"audio item declares meta[{SAMPLING_RATE_KEY!r}]={declared} but the clip decodes at "
                        f"{rate} Hz. Drop the key and let the decoded rate stand."
                    )
                meta[SAMPLING_RATE_KEY] = rate
                audio_consumed += 1
            elif type_ == "text":
                assert value is not None, "text value must not be None"
            else:
                raise ValueError(f"modality type {type_!r} is not yet handled")
            out.append(ConversationItem(type=type_, value=value, role=role, meta=meta))
    leftover_images = list(image_iter)
    assert len(leftover_images) == 0, (
        f"sample has {len(leftover_images)} unused image(s) after consuming {image_consumed}"
    )
    leftover_videos = list(video_iter)
    assert len(leftover_videos) == 0, (
        f"sample has {len(leftover_videos)} unused video(s) after consuming {video_consumed}"
    )
    leftover_audios = list(audio_iter)
    assert len(leftover_audios) == 0, (
        f"sample has {len(leftover_audios)} unused audio clip(s) after consuming {audio_consumed}"
    )
    return out


@DATA_TRANSFORM_REGISTRY.register("seedomni")
def process_seedomni_example(
    example: dict[str, Any],
    **kwargs,
) -> list[dict[str, Any]]:
    """SeedOmni transform — emit a single-key sample ``{"conversation_list": [...]}``.

    Args:
        example: a dataset sample dict.  Required keys:
            - ``"source_name"`` (or pass ``source_name=...`` via ``kwargs``):
              key into ``SEED_OMNI_PREPROCESSOR_REGISTRY`` from
              ``veomni/data/seed_omni/preprocess.py``.
            - ``"conversations"``: list of message dicts in the source's
              native schema (``conv_preprocess`` translates it).  May be
              JSON-encoded ``bytes`` for parquet/arrow formats.
            - ``"images"`` (optional): list of image refs (paths / bytes /
              URLs / PIL).  The preprocessor reorganizes these into the ref
              list actually decoded, according to the data layout it emits —
              the decoded length matches the flattened ``("image", _)`` turns
              so ``_build_conversation_list`` can pair them in order.
            - ``"videos"`` (optional): list of video refs.  The preprocessor
              returns the ref list paired in order with the flattened
              ``("video", _)`` turns.
            - ``"audios"`` (optional): list of audio refs (paths / wav-or-flac
              bytes).  Paired in order with the flattened ``("audio", _)``
              turns and decoded at each clip's own rate, which the resulting
              item declares in ``meta["sampling_rate"]``.
        **kwargs: forwarded to both ``conv_preprocess`` (e.g.
            ``generation_ratio``) and ``fetch_images`` / ``fetch_videos``
            (``image_max_pixels`` / ``video_max_pixels`` / ``fps`` /
            ``max_frames`` — the OOM caps; see
            ``utils/image.resize_to_max_pixels``).  Both fetchers drop every
            other keyword, so a knob named here that they do not read is
            ignored rather than rejected.
            ``OmniTrainer`` injects ``tokenizer`` / ``max_seq_len`` /
            ``text_keys`` here (legacy contract); they are silently
            ignored — SeedOmni modules own their own tokenizer.

    Returns:
        A single-element list ``[{"conversation_list": items}]`` to match
        the ``MappingDataset`` contract (one source sample → one or more
        training samples).  We do not split by length here.
    """
    # Non-destructive read — datasets often share dict references and a
    # subsequent ``__getitem__`` would otherwise see the key gone.
    source = example.get("source_name", kwargs.get("source_name"))
    if source is None:
        raise KeyError(
            "process_seedomni_example: sample is missing 'source_name' (and no fallback "
            "was passed via kwargs); without it ``conv_preprocess`` cannot dispatch to "
            "the right dataset preprocessor."
        )

    conversations = example["conversations"] if ("conversations" in example and example["conversations"]) else example
    if isinstance(conversations, (bytes, bytearray)):
        conversations = json.loads(conversations.decode("utf-8"))

    constructed, image_refs, video_refs, audio_refs = conv_preprocess(source, conversations, example, **kwargs)

    image_tensors = fetch_images(image_refs, **kwargs)
    video_inputs = fetch_videos(video_refs, **kwargs)
    audio_clips = fetch_audios(audio_refs, **kwargs)

    conversation_list = _build_conversation_list(constructed, image_tensors, video_inputs, audio_clips)
    return [{"conversation_list": conversation_list}]
