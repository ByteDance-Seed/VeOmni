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
design contract (see ``design.md`` § "数据路由" and the per-module
responsibility table).

Video / audio modalities are not yet handled — Feature D1 covers the
text + image minimum needed by the Janus / Bagel SeedOmni stacks.  When
those modalities are added back in (alongside their respective encoder
modules), they should follow the same "IO + resize → tensor in
``item['value']``" pattern.

Registered as ``data_type: seedomni`` in
``veomni/data/data_transform.py``::

    args:
      data:
        data_type: seedomni
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from PIL import Image

from ...models.seed_omni.conversation import ConversationItem
from ..data_transform import DATA_TRANSFORM_REGISTRY
from ..multimodal.image_utils import fetch_images
from .preprocess import conv_preprocess


# Tuple-form turn entry used by ``conv_preprocess``: ``(type, value)``.
# For ``type == "image"`` the inline ``value`` is always ``None`` — the actual
# tensor is pulled from the per-sample image list in source order.
_TupleTurn = List  # ``[role: str, (type, value), ...]``


def _pil_to_uint8_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a ``(C, H, W) uint8`` torch tensor.

    No normalization, no float conversion, no channel-mean subtraction —
    those are vision-encoder-specific decisions and live in the encoder
    module's ``pre_forward``/forward.  Keeping pixels as uint8 makes the
    dataloader-worker → main-process IPC roughly 4x cheaper than float32
    and preserves all original pixel information.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    # ``np.array`` (vs ``np.asarray``) forces a writable copy so torch
    # doesn't warn about non-writable tensors when we later .permute().
    arr = np.array(image, dtype=np.uint8)  # (H, W, C)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C, H, W)
    return tensor


def _build_conversation_list(
    constructed: list[_TupleTurn],
    image_tensors: list[torch.Tensor],
) -> list[ConversationItem]:
    """Flatten ``[[role, (type, value), ...], ...]`` into
    :class:`ConversationItem` rows and pair image turns with
    ``image_tensors`` in source order.

    Raises:
        ValueError: if the number of ``("image", _)`` turns doesn't match
            the number of supplied image tensors — that means upstream
            preprocessor / dataset disagree on count and we refuse to
            silently misalign.
    """
    image_iter = iter(image_tensors)
    image_consumed = 0
    out: list[ConversationItem] = []
    for turn in constructed:
        if not turn:
            continue
        role = turn[0]
        if not isinstance(role, str):
            raise TypeError(f"first element of a turn must be a role string, got {type(role).__name__}: {turn!r}")
        for entry in turn[1:]:
            if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
                raise TypeError(f"turn entry must be a (type, value) pair, got {entry!r}")
            type_, value = entry
            if type_ == "image":
                try:
                    value = next(image_iter)
                except StopIteration as e:
                    raise ValueError(
                        f"conversation has more image turns than supplied images "
                        f"({image_consumed} consumed before this one)"
                    ) from e
                image_consumed += 1
            elif type_ == "text":
                if value is None:
                    value = ""
            else:
                # Other modalities (video / audio / ...) are not handled in
                # D1; raise so the caller knows to extend this transform
                # alongside the matching encoder module.
                raise NotImplementedError(
                    f"modality type {type_!r} is not yet handled by the SeedOmni V2 "
                    f"transform; extend ``seedomni_transform.py`` and ensure a matching "
                    f"encoder module exists."
                )
            out.append(ConversationItem(type=type_, value=value, role=role))
    # Validate that we consumed exactly the supplied images — leftover
    # images mean the dataset has unreferenced media which is almost
    # always a bug in upstream data prep.
    leftover = list(image_iter)
    if leftover:
        raise ValueError(
            f"sample has {len(leftover)} unused image(s) after consuming "
            f"{image_consumed} — preprocessor / dataset mismatch."
        )
    return out


def _modalities_in(constructed: list[_TupleTurn]) -> tuple[bool, bool]:
    """Quick scan: does this sample reference any image turns?

    Returns ``(has_image, has_video)`` — kept simple, used only to skip
    media IO when the modality is absent.  ``has_video`` is a stub for
    when video support lands; today the function never returns True for it.
    """
    has_image = False
    for turn in constructed:
        for entry in turn[1:]:
            if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
                continue
            if entry[0] == "image":
                has_image = True
                break
    return has_image, False


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
    import json

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

    has_image, _ = _modalities_in(constructed)
    if has_image:
        raw_images = example.get("images", []) or []
        pil_images = fetch_images(raw_images, **kwargs)
        image_tensors = [_pil_to_uint8_tensor(img) for img in pil_images]
    else:
        image_tensors = []

    conversation_list = _build_conversation_list(constructed, image_tensors)

    return [{"conversation_list": conversation_list}]
