"""Janus packed-sequence helpers (CPU pack + GPU mask scatter).

The conversation carrier path (``accelerated.py``) still walks
``conversation_list`` between modules. The packed path builds tokens / masks
once on the CPU preprocessor, then each module only:

1. gathers the embeddings it owns via a boolean mask, and
2. ``masked_scatter``s them back onto the packed sequence.

Dummy FSDP-anchor images stay *out* of the packed sequence (same as llama's
conversation packer skipping ``role='dummy'``) and are folded in with
``mean() * 0`` so the codec still participates in FSDP collectives.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from veomni.utils.constants import IGNORE_INDEX

from ...utils.conversation import ConversationItem, is_dummy


JANUS_NUM_IMAGE_TOKENS = 576
SIGLIP_SOURCE = "janus_siglip"
VQVAE_SOURCE = "janus_vqvae"

# Keys written onto the training batch by :class:`JanusTextEncoderPreprocessor`
# when ``packed_preprocess`` is set.
PACKED_INPUT_IDS = "packed_input_ids"
PACKED_LABELS = "packed_labels"
PACKED_ATTENTION_MASK = "packed_attention_mask"
PACKED_POSITION_IDS = "packed_position_ids"
PACKED_FEATURES = "packed_features"
PACKED_HIDDEN = "packed_hidden"
UND_IMAGE_MASK = "und_image_mask"
GEN_IMAGE_MASK = "gen_image_mask"
PIXEL_VALUES_UND = "pixel_values_und"
PIXEL_VALUES_GEN = "pixel_values_gen"
UND_NUM_REAL = "und_num_real"
GEN_NUM_REAL = "gen_num_real"
VQ_TOKEN_IDS = "vq_token_ids"


def _as_1d_long(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(-1).to(dtype=torch.long)


def pack_janus_conversations(
    conversation_list: Iterable[list[ConversationItem]],
    *,
    pad_token_id: int,
    num_image_tokens: int = JANUS_NUM_IMAGE_TOKENS,
) -> dict[str, Any]:
    """Build packed ids / masks / stacked pixels from a tokenized conversation.

    Image rows (non-dummy) expand to ``num_image_tokens`` pad ids with the
    matching und/gen mask True; dummy rows are collected only as FSDP-anchor
    pixels. Per-sample ``position_ids`` restart at 0 (FlashAttention varlen).

    ``conversation_list`` is consumed exactly once, so the caller may pass a
    generator that tokenizes each sample on demand instead of materialising
    every tokenized sample up front.
    """
    ids_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    position_chunks: list[torch.Tensor] = []
    und_mask_chunks: list[torch.Tensor] = []
    gen_mask_chunks: list[torch.Tensor] = []
    und_pixels: list[torch.Tensor] = []
    gen_pixels: list[torch.Tensor] = []
    und_dummy_pixels: list[torch.Tensor] = []
    gen_dummy_pixels: list[torch.Tensor] = []

    pad_ids = torch.full((num_image_tokens,), int(pad_token_id), dtype=torch.long)
    ignore = torch.full((num_image_tokens,), IGNORE_INDEX, dtype=torch.long)
    ones_img = torch.ones(num_image_tokens, dtype=torch.bool)
    zeros_img = torch.zeros(num_image_tokens, dtype=torch.bool)

    for sample in conversation_list:
        sample_ids: list[torch.Tensor] = []
        sample_labels: list[torch.Tensor] = []
        sample_und: list[torch.Tensor] = []
        sample_gen: list[torch.Tensor] = []
        for part in sample:
            if is_dummy(part):
                if not isinstance(part.value, torch.Tensor):
                    continue
                if part.source == SIGLIP_SOURCE:
                    und_dummy_pixels.append(part.value)
                elif part.source == VQVAE_SOURCE:
                    gen_dummy_pixels.append(part.value)
                continue
            if part.type == "text":
                token_ids = _as_1d_long(part.value)
                labels = _as_1d_long(part.meta.get("labels", torch.full_like(token_ids, IGNORE_INDEX)))
                sample_ids.append(token_ids)
                sample_labels.append(labels)
                sample_und.append(torch.zeros(token_ids.numel(), dtype=torch.bool))
                sample_gen.append(torch.zeros(token_ids.numel(), dtype=torch.bool))
                continue
            if part.type != "image":
                continue
            if not isinstance(part.value, torch.Tensor):
                raise TypeError(
                    "pack_janus_conversations: image item.value must be a pixel tensor "
                    f"(run SigLIP/VQVAE preprocessors first); got {type(part.value).__name__}."
                )
            sample_ids.append(pad_ids.clone())
            sample_labels.append(ignore.clone())
            if part.source == SIGLIP_SOURCE or (part.source is None and part.role == "user"):
                sample_und.append(ones_img)
                sample_gen.append(zeros_img)
                und_pixels.append(part.value)
            else:
                sample_und.append(zeros_img)
                sample_gen.append(ones_img)
                gen_pixels.append(part.value)

        if not sample_ids:
            continue
        cat_ids = torch.cat(sample_ids, dim=0)
        cat_labels = torch.cat(sample_labels, dim=0)
        # ``shift_packed_labels`` shifts globally, so sample k's first label would
        # become the target of sample k-1's last position -- a token that sits in a
        # different packed span. Mask it, exactly as ``PackingCollator`` does on the
        # single-model path, instead of relying on the chat template happening to
        # emit a masked leading marker. ``torch.cat`` copied, so the conversation
        # carrier's labels stay intact.
        if ids_chunks:
            cat_labels[0] = IGNORE_INDEX
        ids_chunks.append(cat_ids)
        label_chunks.append(cat_labels)
        mask_chunks.append(torch.ones(cat_ids.numel(), dtype=torch.long))
        position_chunks.append(torch.arange(cat_ids.numel(), dtype=torch.long))
        und_mask_chunks.append(torch.cat(sample_und, dim=0))
        gen_mask_chunks.append(torch.cat(sample_gen, dim=0))

    if not ids_chunks:
        raise ValueError("pack_janus_conversations: no non-dummy tokens to pack.")

    packed_ids = torch.cat(ids_chunks, dim=0).unsqueeze(0)
    packed_labels = torch.cat(label_chunks, dim=0).unsqueeze(0)
    packed_attention = torch.cat(mask_chunks, dim=0).unsqueeze(0)
    packed_position = torch.cat(position_chunks, dim=0).unsqueeze(0)
    und_mask = torch.cat(und_mask_chunks, dim=0).unsqueeze(0)
    gen_mask = torch.cat(gen_mask_chunks, dim=0).unsqueeze(0)

    und_num_real = len(und_pixels)
    gen_num_real = len(gen_pixels)
    pixel_values_und = _stack_pixels(und_pixels, und_dummy_pixels)
    pixel_values_gen = _stack_pixels(gen_pixels, gen_dummy_pixels)

    return {
        PACKED_INPUT_IDS: packed_ids,
        PACKED_LABELS: packed_labels,
        PACKED_ATTENTION_MASK: packed_attention,
        PACKED_POSITION_IDS: packed_position,
        UND_IMAGE_MASK: und_mask,
        GEN_IMAGE_MASK: gen_mask,
        PIXEL_VALUES_UND: pixel_values_und,
        PIXEL_VALUES_GEN: pixel_values_gen,
        UND_NUM_REAL: und_num_real,
        GEN_NUM_REAL: gen_num_real,
    }


def _stack_pixels(real: list[torch.Tensor], dummy: list[torch.Tensor]) -> torch.Tensor:
    images = real + dummy
    if not images:
        raise ValueError(
            "pack_janus_conversations: training requires at least one real or dummy image "
            "per tower so FSDP ranks stay symmetric."
        )
    stacked = [img if img.dim() == 3 else img.squeeze(0) for img in images]
    return torch.stack(stacked, dim=0)


def ensure_packed_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """``[T, D]`` → ``[1, T, D]``; leave ``[1, T, ...]`` unchanged."""
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    return tensor


def masked_scatter_embeds(
    packed_features: torch.Tensor,
    mask: torch.Tensor,
    embeds: torch.Tensor,
) -> torch.Tensor:
    """Write ``embeds`` onto ``packed_features`` where ``mask`` is True.

    ``embeds`` is ``[N, S, D]`` or ``[N * S, D]``. ``mask`` is ``[1, T]`` / ``[T]``.
    """
    packed = ensure_packed_batch_dim(packed_features)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    hidden = embeds
    if hidden.dim() == 3:
        hidden = hidden.reshape(-1, hidden.size(-1))
    mask_3d = mask.unsqueeze(-1).expand_as(packed)
    n_true = int(mask.sum().item()) if mask.device.type == "cpu" else hidden.size(0)
    if hidden.size(0) != n_true and mask.device.type == "cpu":
        raise ValueError(f"masked_scatter_embeds: mask selects {n_true} tokens but embeds has {hidden.size(0)} rows.")
    return packed.masked_scatter(mask_3d, hidden.to(device=packed.device, dtype=packed.dtype))


def fold_dummy_anchor(target: torch.Tensor, dummy: torch.Tensor | None) -> torch.Tensor:
    """Keep dummy activations on the autograd graph without changing values."""
    if dummy is None:
        return target
    anchor = dummy.mean().to(device=target.device, dtype=target.dtype) * 0
    return target + anchor


def teacher_force_vq_hidden(packed_hidden: torch.Tensor, gen_image_mask: torch.Tensor) -> torch.Tensor:
    """Hidden states that predict each gen-image token (token *before* the mask).

    ``packed_hidden[:, :-1][gen_image_mask[:, 1:]]`` — same teacher-forcing as
    concatenating ``prev_hidden[-1:]`` with ``image_hidden[:-1]`` per span.
    """
    hidden = ensure_packed_batch_dim(packed_hidden)
    mask = gen_image_mask if gen_image_mask.dim() == 2 else gen_image_mask.unsqueeze(0)
    selected = hidden[:, :-1][mask[:, 1:]]
    return selected.unsqueeze(0) if selected.dim() == 2 else selected


def shift_packed_labels(packed_labels: torch.Tensor) -> torch.Tensor:
    """Causal LM shift: drop token 0, pad IGNORE_INDEX at the end."""
    labels = packed_labels[..., 1:].contiguous()
    return F.pad(labels, (0, 1), "constant", IGNORE_INDEX)


__all__ = [
    "JANUS_NUM_IMAGE_TOKENS",
    "SIGLIP_SOURCE",
    "VQVAE_SOURCE",
    "PACKED_INPUT_IDS",
    "PACKED_LABELS",
    "PACKED_ATTENTION_MASK",
    "PACKED_POSITION_IDS",
    "PACKED_FEATURES",
    "PACKED_HIDDEN",
    "UND_IMAGE_MASK",
    "GEN_IMAGE_MASK",
    "PIXEL_VALUES_UND",
    "PIXEL_VALUES_GEN",
    "UND_NUM_REAL",
    "GEN_NUM_REAL",
    "VQ_TOKEN_IDS",
    "pack_janus_conversations",
    "ensure_packed_batch_dim",
    "masked_scatter_embeds",
    "fold_dummy_anchor",
    "teacher_force_vq_hidden",
    "shift_packed_labels",
]
