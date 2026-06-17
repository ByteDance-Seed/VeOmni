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

"""SeedOmni V2 multimodal data transform — Feature D1.

Reads a raw jsonl-style sample (already source-tagged conversations + media
paths/bytes) and returns ``[{"conversation_list": [...]}]`` where each
item is a :class:`~veomni.models.seed_omni.conversation.ConversationItem`
with ``type`` / ``value`` / ``role`` (empty ``meta`` at the data boundary).

Understanding vs generation images both use ``type="image"``; ``role`` distinguishes
them (``user`` = SigLIP input, ``assistant`` = VQVAE target).

This transform is intentionally minimal — it does **only**:

1. Source-specific conversation normalization (delegated to
   ``veomni.data.seed_omni.preprocess``) so that downstream code sees a uniform
   ``[[role, (type, value), ...], ...]`` structure regardless of the upstream
   dataset.
2. Image IO + ``smart_resize`` (delegated to ``image_utils.fetch_images``),
   followed by PIL → uint8 ``torch.Tensor`` of shape ``(C, H, W)``.  Images are
   **not** normalized, **not** patchified, and **not** wrapped in any
   processor-specific feature dict — those steps are owned by the vision
   encoder module (e.g. ``JanusSiglip`` / ``JanusVqvae``) at forward time.
3. Conversation list assembly — pair each ``("image", None)`` tuple with the
   next image tensor in source order and attach ``role`` per item.

Anything else (chat-template formatting, tokenization, boundary marker
emission, ``input_ids`` / ``labels`` / ``attention_mask`` construction,
position id calculation, image normalization, image patchification) is
deliberately **not** done here — it belongs in model modules per the V2
design contract (see ``docs/seed_omni/design.md`` § "数据路由" and the per-module
responsibility table).

Video turns (``("video", _)``) are paired with the per-sample ``videos`` list
and decoded via ``fetch_videos`` into a :class:`VideoInputs` bundle — the
sampled-frame tensor plus the optional in-video audio waveform.  Both streams
ride on one media item; the downstream video / audio modules each read their
own stream (see ``design.md`` § av-video).

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

from ...models.seed_omni.conversation import ConversationItem
from ...utils.import_utils import is_video_audio_available
from ..data_transform import DATA_TRANSFORM_REGISTRY
from .image_utils import fetch_images
from .preprocess import conv_preprocess


if is_video_audio_available():
    from .video_utils import VideoInputs, fetch_videos
else:
    VideoInputs = None

    def fetch_videos(*args, **kwargs):
        return []


# Tuple-form turn entry used by ``conv_preprocess``: ``(type, value)``.
# For ``type == "image"`` the inline ``value`` is always ``None`` — the actual
# tensor is pulled from the per-sample image list in source order.
_TupleTurn = List  # ``[role: str, (type, value), ...]``


def _build_conversation_list(
    constructed: list[_TupleTurn],
    image_tensors: list[torch.Tensor],
    video_inputs: list[VideoInputs],
) -> list[ConversationItem]:
    """Flatten ``[[role, (type, value), ...], ...]`` into
    :class:`ConversationItem` rows and pair image / video turns with
    ``image_tensors`` / ``video_inputs`` in source order.

    A video turn's value is a :class:`VideoInputs` bundling the decoded frame
    tensor and the optional in-video audio waveform — the single carrier the
    downstream video / audio modules each read their own stream from.

    Raises:
        ValueError: if the number of ``("image"|"video", _)`` turns doesn't
            match the number of supplied media — that means upstream
            preprocessor / dataset disagree on count and we refuse to
            silently misalign.
    """
    image_iter = iter(image_tensors)
    video_iter = iter(video_inputs)
    image_consumed = 0
    video_consumed = 0
    out: list[ConversationItem] = []
    for turn in constructed:
        if not turn:
            continue
        role = turn[0]
        assert role in ["user", "assistant"], f"role must be user or assistant, got {role}"
        for entry in turn[1:]:
            assert len(entry) == 2, f"turn entry must be a (type, value) pair, got {entry}"
            type_, value = entry
            if type_ == "image":
                value: torch.Tensor = next(image_iter)
                image_consumed += 1
            elif type_ == "video":
                value: VideoInputs = next(video_iter)
                video_consumed += 1
            elif type_ == "text":
                assert value is not None, "text value must not be None"
            else:
                raise ValueError(f"modality type {type_!r} is not yet handled")
            out.append(ConversationItem(type=type_, value=value, role=role))
    leftover_images = list(image_iter)
    assert len(leftover_images) == 0, (
        f"sample has {len(leftover_images)} unused image(s) after consuming {image_consumed}"
    )
    leftover_videos = list(video_iter)
    assert len(leftover_videos) == 0, (
        f"sample has {len(leftover_videos)} unused video(s) after consuming {video_consumed}"
    )
    return out


@DATA_TRANSFORM_REGISTRY.register("seedomni")
def process_seedomni_example(
    example: dict[str, Any],
    **kwargs,
) -> list[dict[str, Any]]:
    """SeedOmni V2 transform — emit a single-key sample ``{"conversation_list": [...]}``.

    Args:
        example: a dataset sample dict.  Required keys:
            - ``"source_name"`` (or pass ``source_name=...`` via ``kwargs``):
              key into ``SEED_OMNI_PREPROCESSOR_REGISTRY`` from
              ``veomni/data/seed_omni/preprocess.py``.
            - ``"conversations"``: list of message dicts in the source's
              native schema (``conv_preprocess`` translates it).  May be
              JSON-encoded ``bytes`` for parquet/arrow formats.
            - ``"images"`` (optional): list of image refs (paths / bytes /
              URLs / PIL).  Length must match the number of ``("image", _)``
              turns produced by the preprocessor.
        **kwargs: forwarded to both ``conv_preprocess`` (e.g.
            ``generation_ratio``) and ``fetch_images``
            (e.g. ``image_min_pixels`` / ``image_max_pixels`` /
            ``scale_factor`` / ``max_ratio`` — see ``image_utils.smart_resize``).
            ``OmniTrainer`` injects ``tokenizer`` / ``max_seq_len`` /
            ``text_keys`` here (legacy contract); they are silently
            ignored — V2 modules own their own tokenizer.

    Returns:
        A single-element list ``[{"conversation_list": items}]`` to match
        the ``MappingDataset`` contract (one source sample → one or more
        training samples).  We do not split by length here — packing is a
        downstream collator concern (Feature D2).
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

    constructed = conv_preprocess(source, conversations, **kwargs)

    image_tensors = fetch_images(example.get("images", []) or [], **kwargs)
    video_inputs = fetch_videos(example.get("videos", []) or [], **kwargs)

    conversation_list = _build_conversation_list(constructed, image_tensors, video_inputs)
    return [{"conversation_list": conversation_list}]
