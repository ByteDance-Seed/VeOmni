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

* ``type``  — ``"text"`` | ``"image"`` | ``"output"`` (and the legacy
* ``value`` — polymorphic: raw content (``str`` / PIL image / pixel tensor)
  before encoding, an ``(L, D)`` / ``(1, L, D)`` embedding tensor after.
* ``role``  — ``"user"`` | ``"assistant"`` | ``"dummy"`` (``"dummy"`` rows are
  zero-tensor FSDP placeholders appended by encoders on text-only
  micro-batches; the backbone skips them and folds a zero-grad anchor).
* ``meta``  — per-module baggage written during forward (``labels`` /
  ``attention_mask`` / ``janus_vqvae_labels`` / ``source`` / …).

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

from dataclasses import dataclass, field
from typing import Any, Iterator, Union

import torch
from PIL import Image


ItemType = str  # "text" | "image" | "output"
ItemRole = str  # "user" | "assistant" | "dummy"
ItemValue = Union[str, torch.Tensor, Image.Image]

# Sentinel so ``__value_repr__`` can distinguish "repr self.value" (no arg) from
# "repr this explicit meta value" (which may legitimately be ``None``).
_UNSET = object()


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
        elif hasattr(value, "video") and hasattr(value, "video_fps"):
            # VideoInputs bundle — duck-typed so core conversation.py doesn't
            # import the optional video/audio (ffmpeg/torchcodec/librosa) stack.
            v = value
            shape = tuple(v.video.shape) if isinstance(v.video, torch.Tensor) else type(v.video).__name__
            parts = [f"video.shape={shape}", f"fps={v.video_fps}"]
            audio = getattr(v, "audio", None)
            if audio is not None:
                audio_shape = tuple(audio.shape) if isinstance(audio, torch.Tensor) else type(audio).__name__
                parts.append(f"audio.shape={audio_shape}")
                parts.append(f"audio_fps={getattr(v, 'audio_fps', None)}")
            return f"[VideoInputs | {', '.join(parts)}]"
        else:
            return f"[UnknownType]{type(value).__name__}"

    def __meta_repr__(self) -> str:
        meta_items = [f"{key}={self.__value_repr__(value)}" for key, value in self.meta.items()]
        return f"{{{','.join(meta_items)}}}"

    def __repr__(self) -> str:
        return f"ConversationItem(type={self.type}, value={self.__value_repr__()}, role={self.role}, source={self.source}, meta={self.__meta_repr__()}"


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


def seal_outputs(parts: list[ConversationItem], new_type: ItemType) -> int:
    """Rename completed ``output`` spans to a sealed type (``text`` / ``image``)."""
    assert parts[-1].type == "output"
    parts[-1].type = new_type


def build_conversation(
    *,
    prompt: str,
    images: list[Any] | None = None,
) -> list[ConversationItem]:
    """Build the canonical conversation list for a single inference request."""
    parts: list[ConversationItem] = []
    for img in images or []:
        parts.append(ConversationItem(type="image", value=img, role="user"))
    parts.append(ConversationItem(type="text", value=prompt, role="user"))
    return parts


# ── Training batch helpers (unified with inference ConversationItem) ────────


def iter_desired_items(
    conversation_list: list[list[ConversationItem]],
    types: list[str] | None = None,
    roles: list[str] | None = None,
    sources: list[str] | None = None,
    reverse_item: bool = False,
    *,
    meta_keys: list[str] | None = None,
) -> Iterator[ConversationItem]:
    """Yield matching items in micro-batch order (sample 0, then sample 1, …)."""

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
            yield item


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
    "iter_desired_items",
    "collect_desired_values",
]
