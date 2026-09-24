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
item then states in ``meta["audio_metadata"]``. The data layer does not pick a
rate because there is no single right one: a Qwen3-Omni thinker's tower wants
16 kHz while the codec that turns assistant speech into talker targets wants
24 kHz, and the rate is also what places a clip on TMRoPE's shared clock.
``role`` separates the two uses of the same ``type="audio"`` row — ``user`` is an
input to an audio encoder, ``assistant`` is the speech a talker is trained to
say, i.e. a label.

Video turns (``("video", _)``) are paired with the per-sample ``videos`` list
and decoded via ``fetch_videos`` into a :class:`VideoInputs` payload — the
sampled-frame tensor plus the optional in-video audio waveform.  Both streams
ride on one media item; splitting them across a video and an audio module is
not implemented yet.

Both fetchers hand back ``(payload, meta)``. What the data layer *loaded* —
frame timeline, sample rate — goes on ``ConversationItem.meta``, not into the
payload, because ``value`` is overwritten by the first encoder that touches the
item while the backbone reads the timeline after that. It also means a rate
reads the same whether the sound came in as its own item or inside a clip: both
spell it ``meta["audio_metadata"].sampling_rate``.

Registered as ``data_type: seedomni`` in
``veomni/data/data_transform.py``::

    args:
      data:
        data_type: seedomni
"""

from __future__ import annotations

import json
from typing import Any, List

from ...models.seed_omni.utils.conversation import ConversationItem
from ..data_transform import DATA_TRANSFORM_REGISTRY
from .preprocess import conv_preprocess
from .utils.media import MEDIA_FETCHERS, fetch_media


# Tuple-form turn entry used by ``conv_preprocess``: ``(type, value)`` or the
# optional 3-tuple ``(type, value, meta)`` where ``meta`` is merged into the
# resulting ``ConversationItem.meta`` (e.g. ``{_IMG_TAG_KEY: "und"}``). For
# ``type == "image"`` / ``"video"`` the inline ``value`` is always ``None`` —
# the actual tensor is pulled from the per-sample media list in source order.
_TupleTurn = List  # ``[role: str, (type, value) | (type, value, meta), ...]``


def _build_conversation_list(
    constructed: list[_TupleTurn],
    media: dict[str, list[tuple[Any, dict[str, Any]]]] | None = None,
) -> list[ConversationItem]:
    """Flatten ``[[role, (type, value) | (type, value, meta), ...], ...]`` into
    :class:`ConversationItem` rows, pairing each media turn with ``media``.

    ``media`` is the decoded output of :data:`~veomni.data.seed_omni.utils.media.MEDIA_FETCHERS`, keyed by item
    type: ``{"image": [(tensor, meta), ...], "video": [...], ...}``. Every
    modality is handled by the same branch — take the next ``(payload, meta)``
    for this type, make the payload the item's ``value`` and merge the meta into
    the item's ``meta`` — so a new modality needs no code here.

    Pairing is purely positional: the *n*-th ``("video", _)`` turn takes the
    *n*-th entry of ``media["video"]``. The preprocessor owns that alignment,
    and a mismatch in either direction stops the sample rather than shifting
    every later turn onto the wrong media.

    An optional 3-tuple ``(type, value, meta)`` carries per-item ``meta``, e.g.
    ``{_IMG_TAG_KEY: "und"}``. It is copied, not aliased, so a preprocessor's
    dict cannot be mutated by a module downstream. Keys a fetcher writes (the
    decoded timeline, e.g. ``audio_metadata`` / ``video_metadata``) belong to
    the decoder: the file's header is the only authority on them, so a
    preprocessor that declares one is refused rather than silently overwritten.

    Raises:
        ValueError: if a turn names a modality ``media`` has no entries left
            for, or if entries are left over once every turn is built — either
            way the preprocessor and the dataset disagree on a count. Also if a
            preprocessor's per-item meta declares a key the fetcher writes.
    """
    iters = {type_: iter(items) for type_, items in (media or {}).items()}
    consumed = dict.fromkeys(iters, 0)
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
            if type_ in MEDIA_FETCHERS:
                # ``next`` with a sentinel rather than letting StopIteration
                # escape: this runs inside dataset iteration, where a leaked
                # StopIteration ends the epoch silently instead of naming a bad
                # sample.
                item = next(iters.get(type_, iter(())), None)
                if item is None:
                    raise ValueError(
                        f"sample has more {type_} turn(s) than decoded {type_} media "
                        f"({consumed.get(type_, 0)} supplied); they are paired by position, so the missing "
                        f"one would shift every later turn onto the wrong media."
                    )
                value, media_meta = item
                declared = sorted(meta.keys() & media_meta.keys())
                if declared:
                    raise ValueError(
                        f"{type_} item declares meta key(s) {declared} that the {type_} fetcher writes from the "
                        f"decoded file. Drop them from the preprocessor; the decoded values are the only ones "
                        f"that describe the payload."
                    )
                meta.update(media_meta)
                consumed[type_] += 1
            elif type_ == "text":
                assert value is not None, "text value must not be None"
            else:
                raise ValueError(f"modality type {type_!r} is not yet handled")
            out.append(ConversationItem(type=type_, value=value, role=role, meta=meta))
    for type_, it in iters.items():
        leftover = len(list(it))
        if leftover:
            raise ValueError(
                f"sample has {leftover} unused {type_} media after consuming {consumed[type_]}; they are "
                f"paired by position with the {type_} turns, so an unequal count would misattach them."
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
              URLs / PIL).
            - ``"videos"`` (optional): list of video refs.
            - ``"audios"`` (optional): list of audio refs (paths / wav-or-flac
              bytes), decoded at each clip's own rate, which the resulting item
              states in ``meta["audio_metadata"]``.

            The preprocessor reorganizes these into the refs actually decoded,
            keyed by item type (``conv_preprocess`` returns
            ``{"image": [...], "video": [...], ...}``), so each list's length
            matches that modality's flattened turn count and
            ``_build_conversation_list`` can pair them in order.
        **kwargs: forwarded to ``conv_preprocess`` (e.g. ``generation_ratio``)
            and to every fetcher in :data:`~veomni.data.seed_omni.utils.media.MEDIA_FETCHERS`
            (``image_max_pixels`` / ``video_max_pixels`` / ``fps`` /
            ``max_frames`` — the OOM caps; see
            ``utils/image.resize_to_max_pixels``).  The fetchers drop every
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

    constructed, media_refs = conv_preprocess(source, conversations, example, **kwargs)

    media = fetch_media(media_refs, f"process_seedomni_example: preprocessor for source {source!r}", kwargs)

    conversation_list = _build_conversation_list(constructed, media)
    return [{"conversation_list": conversation_list}]
