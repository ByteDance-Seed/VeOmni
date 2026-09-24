"""``ConversationItem`` — the single carrier object of SeedOmni V2.

The whole pipeline (training and inference) operates on one batched
``conversation_list`` (``list[list[ConversationItem]]`` for training, a flat
``list[ConversationItem]`` for one inference request).  Modules read and
**mutate ``item.value`` in place** — there are no per-field edge channels;
``OmniModel`` writes the (possibly grown) list back into the shared batch /
``ctx`` after every node.

Item shape
----------
Each item is ``{type, value, role, meta}``:

* ``type``  — ``"text"`` | ``"image"`` | ``"video"`` | ``"audio"`` | ``"output"``.
  ``"output"`` is the transient row a backbone appends while generating;
  :func:`seal_outputs` renames it to its final modality once the span closes.
  Janus interleaved generation also accepts the legacy ``"image_output"`` on
  input, which ``JanusSiglip.generate`` re-encodes and seals to ``"image"``.
  ``"audio"`` is a standalone sound item (speech in, or speech out once sealed).
  Sound carried *inside* a video clip is not an ``"audio"`` item — it rides on
  the video item's own ``value`` (``VideoInputs.audio``) so that one clip stays
  one item and the backbone can interleave both streams on a shared timeline.
  Either way its rate is stated the same way, in ``meta["audio_metadata"]``.
* ``value`` — polymorphic: raw content (``str`` / PIL image / pixel tensor /
  ``(samples,)`` waveform) before encoding, an ``(L, D)`` / ``(1, L, D)``
  embedding tensor after.
* ``role``  — ``"user"`` | ``"assistant"`` | ``"dummy"`` (``"dummy"`` rows are
  zero-tensor FSDP placeholders appended by encoders on text-only
  micro-batches; the backbone skips them and folds a zero-grad anchor).
* ``meta``  — the item's only durable channel. What the data layer loaded
  (``video_metadata`` / ``audio_metadata``, see
  ``veomni/data/seed_omni/utils/media_metadata.py``)
  plus per-module baggage written during forward (``labels`` /
  ``attention_mask`` / ``janus_vqvae_labels`` / ``source`` / …). Anything the
  backbone still needs after an encoder has overwritten ``value`` has to be
  here.

Lifecycle
---------
1. An item is born with a raw ``value`` (text string, PIL image, pixels).
2. An encoder (SigLIP / VQVAE / text wte) overwrites ``value`` with its
   embedding tensor.
3. The backbone overwrites ``value`` again with the per-segment hidden state.
4. During inference, backbone steps append ``type="output"`` rows; when a
   modality span finishes, :func:`seal_outputs` renames the trailing
   ``output`` row to ``"text"`` / ``"image"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Iterator, Union

import torch
from PIL import Image


ItemType = str  # "text" | "image" | "video" | "audio" | "output" (+ legacy "image_output")
ItemRole = str  # "user" | "assistant" | "dummy"
ItemValue = Union[str, torch.Tensor, Image.Image]

# Sentinel so ``__value_repr__`` can distinguish "repr self.value" (no arg) from
# "repr this explicit meta value" (which may legitimately be ``None``).
_UNSET = object()

# Per-image data-tag key written on ``meta`` by the data layer (preprocessors).
# Image-only: text items do NOT carry it. Values: ``"und"`` | ``"gen"`` | ``"edit"``
# — a model-agnostic declaration of the image's role in the sample. Model-side
# routing reads it via ``iter_desired_items(..., meta={_IMG_TAG_KEY: [...]})`` or
# ``item.meta.get(_IMG_TAG_KEY)``; this module only owns the key.
_IMG_TAG_KEY = "_img_tag"


@dataclass
class ConversationItem:
    """One element of a conversation list — ``{type, value, role, meta}``."""

    type: ItemType
    value: ItemValue
    role: ItemRole = "user"
    source: str | None = None
    meta: dict = field(default_factory=dict)

    def __value_repr__(self, value: Any = _UNSET) -> str:
        # Repr ``self.value`` by default; ``__meta_repr__`` passes meta values.
        if value is _UNSET:
            value = self.value
        if isinstance(value, str):
            return f"[str]{repr(value)}"
        elif isinstance(value, torch.Tensor):
            return f"[torch.Tensor]{tuple(value.shape)}"
        elif isinstance(value, Image.Image):
            return f"[PIL.Image]{value.size}"
        elif hasattr(value, "video") and hasattr(value, "has_audio"):
            # VideoInputs payload — duck-typed so core conversation.py doesn't
            # import the optional video/audio (ffmpeg/torchcodec/librosa) stack.
            # Only the two streams are here; the timelines are on ``meta`` (see
            # ``data/seed_omni/utils/media_metadata.py``) and print with the
            # rest of it.
            v = value
            shape = tuple(v.video.shape) if isinstance(v.video, torch.Tensor) else type(v.video).__name__
            parts = [f"video.shape={shape}"]
            audio = getattr(v, "audio", None)
            if audio is not None:
                audio_shape = tuple(audio.shape) if isinstance(audio, torch.Tensor) else type(audio).__name__
                parts.append(f"audio.shape={audio_shape}")
            return f"[VideoInputs | {', '.join(parts)}]"
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            # Raw audio waveforms arrive as numpy arrays. Duck-typed for the same
            # reason as VideoInputs above: keep this core module import-free.
            return f"[{type(value).__name__}]{tuple(value.shape)}"
        else:
            return f"[UnknownType]{type(value).__name__}"

    def __meta_repr__(self) -> str:
        meta_items = [f"{key}={self.__value_repr__(value)}" for key, value in self.meta.items()]
        return f"{{{','.join(meta_items)}}}"

    def __repr__(self) -> str:
        return f"ConversationItem(type={self.type}, value={self.__value_repr__()}, role={self.role}, source={self.source}, meta={self.__meta_repr__()})"


def is_dummy(item: ConversationItem) -> bool:
    return item.role == "dummy"


def maybe_merge_outputs(parts: list[ConversationItem]) -> bool:
    """Merge the last two ``output`` rows in the same AR phase (concat on seq dim)."""
    if len(parts) < 2:
        return False
    a, b = parts[-2], parts[-1]
    if a.type != "output" or b.type != "output":
        return False
    emb_a, emb_b = a.value, b.value
    emb_b = emb_b.to(device=emb_a.device, dtype=emb_a.dtype)
    a.value = torch.cat([emb_a, emb_b], dim=-2)
    parts.pop()
    return True


def seal_outputs(parts: list[ConversationItem], new_type: ItemType) -> None:
    """Rename completed ``output`` spans to a sealed type (``text`` / ``image``)."""
    assert parts[-1].type == "output"
    parts[-1].type = new_type


def _split_entry(entry: Any) -> tuple[Any, Mapping]:
    """Read a media entry as ``(payload, meta)``, defaulting the meta to empty.

    Pair-ness is decided on the second element being a mapping rather than on
    tuple-ness alone, so a payload that happens to be a 2-tuple is not mistaken
    for a pair.
    """
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], Mapping):
        return entry[0], entry[1]
    return entry, {}


def build_conversation(
    *,
    prompt: str,
    media: Mapping[str, Sequence[Any]] | None = None,
) -> list[ConversationItem]:
    """Build the canonical conversation list for a single inference request.

    ``media`` is ``{item_type: [payload | (payload, meta), ...]}`` — the shape
    ``veomni.data.seed_omni.utils.media.fetch_media`` returns. Passing the
    fetchers' output straight through is what makes a request carry the same
    metadata a training sample does: a clip's sampling rate and a video's frame
    timeline are facts only the decode knew, and a consumer cannot recover them
    from the payload. Bare payloads are accepted for callers that already hold
    decoded media and have nothing to state about it.

    Items come out in the mapping's own order, prompt last. Adding a modality
    needs no change here — any type keyed in that table flows through untouched.

    Sound belonging to a video clip is not a separate ``"audio"`` entry: it rides
    on the video payload's ``VideoInputs.audio``, so one clip stays one item
    however many tracks it has.
    """
    parts: list[ConversationItem] = []
    for item_type, entries in (media or {}).items():
        for entry in entries:
            value, meta = _split_entry(entry)
            parts.append(ConversationItem(type=item_type, value=value, role="user", meta=dict(meta)))
    parts.append(ConversationItem(type="text", value=prompt, role="user"))
    return parts


# Training batch helpers, unified with the inference ConversationItem.


def iter_desired_items(
    conversation_list: list[list[ConversationItem]],
    types: list[str] | None = None,
    roles: list[str] | None = None,
    sources: list[str] | None = None,
    reverse_item: bool = False,
    *,
    meta_keys: list[str] | None = None,
    meta: dict[str, list[Any]] | None = None,
) -> Iterator[ConversationItem]:
    """Yield matching items in micro-batch order (sample 0, then sample 1, …).

    ``meta`` requires every configured key to exist and its value to match one
    of the corresponding allowed values.
    """

    for sample in conversation_list:
        items = reversed(sample) if reverse_item else sample
        for item in items:
            if types is not None and item.type not in types:
                continue
            if roles is not None and item.role not in roles:
                continue
            if sources is not None and item.source not in sources:
                continue
            if meta_keys is not None and any(key not in item.meta for key in meta_keys):
                continue
            if meta is not None and any(
                key not in item.meta or item.meta[key] not in allowed_values for key, allowed_values in meta.items()
            ):
                continue
            yield item


def get_tail_output_item(
    conversation_list: list[ConversationItem],
    *,
    sources: list[str] | None = None,
    roles: list[str] | None = None,
    meta_keys: list[str] | None = None,
) -> ConversationItem | None:
    """Return the latest matching output item in a single conversation."""
    return next(
        iter_desired_items(
            [conversation_list],
            types=["output"],
            roles=roles,
            sources=sources,
            reverse_item=True,
            meta_keys=meta_keys,
        ),
        None,
    )


def collect_desired_values(
    conversation_list: list[list[ConversationItem]],
    types: list[str] | None = None,
    roles: list[str] | None = None,
    sources: list[str] | None = None,
    *,
    meta_keys: list[str] | None = None,
) -> list[Any]:
    """Flat ``item.value`` list for matching items in micro-batch order."""
    return [
        item.value
        for item in iter_desired_items(
            conversation_list,
            types,
            roles,
            sources,
            meta_keys=meta_keys,
        )
    ]


__all__ = [
    "ConversationItem",
    "build_conversation",
    "is_dummy",
    "maybe_merge_outputs",
    "seal_outputs",
    "get_tail_output_item",
    "iter_desired_items",
    "collect_desired_values",
    "_IMG_TAG_KEY",
]
