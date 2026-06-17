"""Per-item BAGEL training layout and final assembly helpers."""

from __future__ import annotations

from typing import Any

import torch

from ....conversation import ConversationItem
from .anchors import dummy_anchors_from_conversation
from .carrier import BAGEL_TRAIN_LABEL_IDS, conversation_samples
from .packing import _prepare_attention_mask_per_sample, _shifted_timesteps


BAGEL_TRAIN_LAYOUT_PLANNED = "bagel_train_layout_planned"
BAGEL_TRAIN_SEQUENCE_INDEXES = "bagel_train_sequence_indexes"
BAGEL_TRAIN_TEXT_INDEXES = "bagel_train_text_indexes"
BAGEL_TRAIN_VIT_INDEXES = "bagel_train_vit_indexes"
BAGEL_TRAIN_VAE_INDEXES = "bagel_train_vae_indexes"
BAGEL_TRAIN_SPLIT_LEN = "bagel_train_split_len"
BAGEL_TRAIN_ATTN_MODE = "bagel_train_attn_mode"
BAGEL_TRAIN_CE_INDEXES = "bagel_train_ce_indexes"
BAGEL_TRAIN_MSE_INDEXES = "bagel_train_mse_indexes"
BAGEL_TRAIN_TEXT_IDS = "bagel_train_text_ids"
BAGEL_TRAIN_TEXT_EMBEDS = "bagel_train_text_embeds"
BAGEL_TRAIN_MSE_TARGET = "bagel_train_mse_target"
BAGEL_TRAIN_SAMPLE_ATTN_MASK = "bagel_train_sample_attention_mask"
BAGEL_TRAIN_SAMPLE_LEN = "bagel_train_sample_len"
BAGEL_TRAIN_HIDDEN_STATES = "bagel_train_hidden_states"


def training_layout_planned(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> bool:
    for sample in conversation_samples(conversation_list):
        if sample and sample[0].meta.get(BAGEL_TRAIN_LAYOUT_PLANNED):
            return True
    return False


def ensure_training_layout_planned(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem],
    kwargs: dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
) -> None:
    if training_layout_planned(conversation_list):
        return
    kwargs = kwargs or {}
    resolved_device = device
    if resolved_device is None:
        for sample in conversation_samples(conversation_list):
            for item in sample:
                text_ids = item.meta.get(BAGEL_TRAIN_TEXT_IDS)
                if torch.is_tensor(text_ids):
                    resolved_device = text_ids.device
                    break
            if resolved_device is not None:
                break
    if resolved_device is None:
        raise ValueError("BAGEL training layout planning requires materialized text token metadata.")

    for sample in conversation_samples(conversation_list):
        sequence_cursor = 0
        sample_splits: list[int] = []
        sample_attn_modes: list[str] = []
        sample_position_cursor = 0
        sample_start = 0

        for item in sample:
            if item.type == "text":
                sequence_cursor, sample_position_cursor = _plan_text_item(
                    item,
                    sequence_cursor=sequence_cursor,
                    sample_position_cursor=sample_position_cursor,
                    device=resolved_device,
                    sample_splits=sample_splits,
                    sample_attn_modes=sample_attn_modes,
                )
                continue
            if item.type != "image":
                continue
            if item.role == "user":
                sequence_cursor, sample_position_cursor = _plan_image_understanding_item(
                    item,
                    sequence_cursor=sequence_cursor,
                    sample_position_cursor=sample_position_cursor,
                    device=resolved_device,
                    sample_splits=sample_splits,
                    sample_attn_modes=sample_attn_modes,
                )
                continue
            sequence_cursor, sample_position_cursor = _plan_image_generation_item(
                item,
                sequence_cursor=sequence_cursor,
                sample_position_cursor=sample_position_cursor,
                device=resolved_device,
                sample_splits=sample_splits,
                sample_attn_modes=sample_attn_modes,
            )

        sample_len = sequence_cursor - sample_start
        attn_mask = _prepare_attention_mask_per_sample(sample_splits, sample_attn_modes).to(resolved_device)
        if sample:
            sample[0].meta[BAGEL_TRAIN_SAMPLE_LEN] = sample_len
            sample[0].meta[BAGEL_TRAIN_SAMPLE_ATTN_MASK] = attn_mask
            sample[0].meta[BAGEL_TRAIN_LAYOUT_PLANNED] = True


def materialize_training_text_ids(
    text_encoder: Any,
    conversation_list: list[list[ConversationItem]] | list[ConversationItem],
) -> torch.Tensor | None:
    """Tokenize text and image-boundary spans without assigning global sequence layout."""

    device = text_encoder.device
    eos = text_encoder._resolve_eos_token_id()
    start = text_encoder._resolve_start_token_id()
    image_start, image_end = text_encoder._image_boundary_token_ids()
    parts: list[torch.Tensor] = []

    for sample in conversation_samples(conversation_list):
        for item in sample:
            if item.type == "text":
                text_ids = _raw_training_text_ids(text_encoder, item)
                token_ids = (
                    text_ids
                    if item.meta.get("bagel_train_exact_text_ids")
                    else torch.cat(
                        (
                            torch.tensor([start], device=device, dtype=torch.long),
                            text_ids,
                            torch.tensor([eos], device=device, dtype=torch.long),
                        )
                    )
                )
                item.meta[BAGEL_TRAIN_TEXT_IDS] = token_ids
                if item.role == "assistant":
                    item.meta[BAGEL_TRAIN_LABEL_IDS] = token_ids[1:].detach()
                parts.append(token_ids)
                continue
            if item.type != "image" or item.meta.get("bagel_train_exact_no_boundaries"):
                continue
            boundary_ids = torch.tensor([image_start, image_end], device=device, dtype=torch.long)
            item.meta[BAGEL_TRAIN_TEXT_IDS] = boundary_ids
            parts.append(boundary_ids)

    if not parts:
        return None
    return torch.cat(parts).to(device=device, dtype=torch.long)


def collect_training_text_ids(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem],
) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
    for sample in conversation_samples(conversation_list):
        for item in sample:
            text_ids = item.meta.get(BAGEL_TRAIN_TEXT_IDS)
            if torch.is_tensor(text_ids) and int(text_ids.numel()) > 0:
                parts.append(text_ids.detach().reshape(-1))
    if not parts:
        return None
    device = parts[0].device
    return torch.cat(parts).to(device=device, dtype=torch.long)


def scatter_training_text_embeds(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem],
    packed_text_embeds: torch.Tensor,
) -> None:
    offset = 0
    for sample in conversation_samples(conversation_list):
        for item in sample:
            text_ids = item.meta.get(BAGEL_TRAIN_TEXT_IDS)
            if not torch.is_tensor(text_ids):
                continue
            length = int(text_ids.numel())
            if length == 0:
                continue
            item.meta[BAGEL_TRAIN_TEXT_EMBEDS] = packed_text_embeds[offset : offset + length]
            offset += length


def assemble_training_forward(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem],
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
) -> dict[str, Any]:
    samples = conversation_samples(conversation_list)
    if not samples:
        raise ValueError("BAGEL training assembly requires non-empty conversation_list.")

    sample_lens: list[int] = []
    nested_attention_masks: list[torch.Tensor] = []
    packed_position_ids: list[torch.Tensor] = []
    und_indexes: list[torch.Tensor] = []
    gen_indexes: list[torch.Tensor] = []
    sequence_length = 0

    for sample in samples:
        sample_len = int(sample[0].meta.get(BAGEL_TRAIN_SAMPLE_LEN, 0))
        attn_mask = sample[0].meta.get(BAGEL_TRAIN_SAMPLE_ATTN_MASK)
        if sample_len <= 0 or not torch.is_tensor(attn_mask):
            raise ValueError("BAGEL training assembly requires planned sample layout metadata.")
        sample_lens.append(sample_len)
        nested_attention_masks.append(attn_mask.to(device=device))
        sequence_length += sample_len

        for item in sample:
            position_ids = item.meta.get("bagel_train_position_ids")
            if torch.is_tensor(position_ids):
                packed_position_ids.append(position_ids.detach().to(device=device, dtype=torch.long).reshape(-1))

            text_indexes = item.meta.get(BAGEL_TRAIN_TEXT_INDEXES)
            if torch.is_tensor(text_indexes) and int(text_indexes.numel()) > 0:
                und_indexes.append(text_indexes.detach().to(device=device, dtype=torch.long).reshape(-1))

            vit_indexes = item.meta.get(BAGEL_TRAIN_VIT_INDEXES)
            if torch.is_tensor(vit_indexes) and int(vit_indexes.numel()) > 0:
                und_indexes.append(vit_indexes.detach().to(device=device, dtype=torch.long).reshape(-1))

            vae_indexes = item.meta.get(BAGEL_TRAIN_VAE_INDEXES)
            if torch.is_tensor(vae_indexes) and int(vae_indexes.numel()) > 0:
                gen_indexes.append(vae_indexes.detach().to(device=device, dtype=torch.long).reshape(-1))

    packed_sequence = torch.zeros(sequence_length, hidden_size, device=device, dtype=dtype)
    for sample in samples:
        for item in sample:
            text_embeds = item.meta.get(BAGEL_TRAIN_TEXT_EMBEDS)
            text_indexes = item.meta.get(BAGEL_TRAIN_TEXT_INDEXES)
            if torch.is_tensor(text_embeds) and torch.is_tensor(text_indexes):
                packed_sequence[text_indexes.to(device=device)] = text_embeds.to(device=device, dtype=dtype)

            vit_embeds = item.meta.get("packed_vit_embeds")
            vit_indexes = item.meta.get(BAGEL_TRAIN_VIT_INDEXES)
            if torch.is_tensor(vit_embeds) and torch.is_tensor(vit_indexes):
                packed_sequence[vit_indexes.to(device=device)] = vit_embeds.to(device=device, dtype=dtype)

            latent_embeds = item.meta.get("packed_latent_embeds")
            vae_indexes = item.meta.get(BAGEL_TRAIN_VAE_INDEXES)
            if torch.is_tensor(latent_embeds) and torch.is_tensor(vae_indexes):
                packed_sequence[vae_indexes.to(device=device)] = latent_embeds.to(device=device, dtype=dtype)

    packed_und_token_indexes = (
        torch.cat(und_indexes) if und_indexes else torch.empty(0, device=device, dtype=torch.long)
    )
    packed_gen_token_indexes = (
        torch.cat(gen_indexes) if gen_indexes else torch.empty(0, device=device, dtype=torch.long)
    )
    packed_position_ids_tensor = (
        torch.cat(packed_position_ids).to(device=device, dtype=torch.long)
        if packed_position_ids
        else torch.arange(sequence_length, device=device, dtype=torch.long)
    )

    return {
        "sequence_length": sequence_length,
        "packed_sequence": packed_sequence,
        "sample_lens": sample_lens,
        "nested_attention_masks": nested_attention_masks,
        "packed_position_ids": packed_position_ids_tensor,
        "packed_und_token_indexes": packed_und_token_indexes,
        "packed_gen_token_indexes": packed_gen_token_indexes,
        "dummy_anchors": dummy_anchors_from_conversation(conversation_list),
    }


def scatter_training_hidden_states(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem],
    packed_hidden_states: torch.Tensor,
) -> None:
    for sample in conversation_samples(conversation_list):
        for item in sample:
            ce_indexes = item.meta.get(BAGEL_TRAIN_CE_INDEXES)
            if torch.is_tensor(ce_indexes) and int(ce_indexes.numel()) > 0:
                item.meta[BAGEL_TRAIN_HIDDEN_STATES] = packed_hidden_states[
                    ce_indexes.to(device=packed_hidden_states.device, dtype=torch.long)
                ]
                continue
            mse_indexes = item.meta.get(BAGEL_TRAIN_MSE_INDEXES)
            if torch.is_tensor(mse_indexes) and int(mse_indexes.numel()) > 0:
                item.meta[BAGEL_TRAIN_HIDDEN_STATES] = packed_hidden_states[
                    mse_indexes.to(device=packed_hidden_states.device, dtype=torch.long)
                ]


def collect_ce_loss_inputs(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    hidden_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for sample in conversation_samples(conversation_list):
        for item in sample:
            hidden = item.meta.get(BAGEL_TRAIN_HIDDEN_STATES)
            labels = item.meta.get(BAGEL_TRAIN_LABEL_IDS)
            if torch.is_tensor(hidden) and torch.is_tensor(labels) and int(labels.numel()) > 0:
                hidden_parts.append(hidden.reshape(-1, hidden.shape[-1]))
                label_parts.append(labels.reshape(-1))
    if not hidden_parts:
        return None
    return torch.cat(hidden_parts, dim=0), torch.cat(label_parts, dim=0)


def collect_mse_loss_inputs(
    conversation_list: list[list[ConversationItem]] | list[ConversationItem] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    hidden_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for sample in conversation_samples(conversation_list):
        for item in sample:
            hidden = item.meta.get(BAGEL_TRAIN_HIDDEN_STATES)
            target = item.meta.get(BAGEL_TRAIN_MSE_TARGET)
            if torch.is_tensor(hidden) and torch.is_tensor(target) and int(target.numel()) > 0:
                hidden_parts.append(hidden.reshape(-1, hidden.shape[-1]))
                target_parts.append(target.reshape(-1, target.shape[-1]))
    if not hidden_parts:
        return None
    return torch.cat(hidden_parts, dim=0), torch.cat(target_parts, dim=0)


def patchified_clean_latents(latent: torch.Tensor, h: int, w: int) -> torch.Tensor:
    patch_h = int(latent.shape[-2]) // h
    patch_w = int(latent.shape[-1]) // w
    return (
        latent.detach()
        .reshape(1, int(latent.shape[1]), h, patch_h, w, patch_w)
        .permute(0, 2, 4, 3, 5, 1)
        .flatten(0, 2)
        .flatten(1, 3)
    )


def prepare_flow_training_metadata(
    item: ConversationItem,
    *,
    device: torch.device,
    dtype: torch.dtype,
    timestep_shift: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    latent = item.meta.get("padded_latent")
    latent_shape = item.meta.get("patchified_vae_latent_shape")
    if not torch.is_tensor(latent) or not isinstance(latent_shape, tuple):
        return None
    h, w = int(latent_shape[0]), int(latent_shape[1])
    clean = patchified_clean_latents(latent, h, w).to(device=device, dtype=dtype)
    item_timesteps = item.meta.get("flow_timesteps")
    item_noise = item.meta.get("flow_noise")
    if torch.is_tensor(item_timesteps):
        raw_timesteps = item_timesteps.detach().to(device=device, dtype=torch.float32).reshape(-1)
    else:
        raw_timesteps = torch.randn(int(clean.shape[0]), device=device, dtype=torch.float32)
    if torch.is_tensor(item_noise):
        fixed_noise = item_noise.detach().to(device=device, dtype=dtype)
    else:
        fixed_noise = torch.randn_like(clean, device=device)
    shifted = _shifted_timesteps(raw_timesteps, float(item.meta.get("timestep_shift", timestep_shift)))
    timesteps = shifted.to(device=clean.device).reshape(-1, 1)
    noised = (1.0 - timesteps) * clean + timesteps * fixed_noise
    target = fixed_noise - clean
    return noised, target, shifted


def _plan_text_item(
    item: ConversationItem,
    *,
    sequence_cursor: int,
    sample_position_cursor: int,
    device: torch.device,
    sample_splits: list[int],
    sample_attn_modes: list[str],
) -> tuple[int, int]:
    token_ids = item.meta.get(BAGEL_TRAIN_TEXT_IDS)
    if not torch.is_tensor(token_ids):
        raise ValueError("BAGEL text item requires bagel_train_text_ids before layout planning.")
    token_ids = token_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
    length = int(token_ids.numel())
    indexes = torch.arange(sequence_cursor, sequence_cursor + length, device=device, dtype=torch.long)
    item.meta[BAGEL_TRAIN_TEXT_IDS] = token_ids
    item.meta[BAGEL_TRAIN_TEXT_INDEXES] = indexes
    item.meta[BAGEL_TRAIN_SEQUENCE_INDEXES] = indexes
    item.meta[BAGEL_TRAIN_SPLIT_LEN] = length
    item.meta[BAGEL_TRAIN_ATTN_MODE] = "causal"
    position_ids = _position_ids(item, sample_position_cursor, length, device=device)
    item.meta["bagel_train_position_ids"] = position_ids
    if item.role == "assistant":
        item.meta[BAGEL_TRAIN_CE_INDEXES] = indexes[:-1]
        item.meta[BAGEL_TRAIN_LABEL_IDS] = token_ids[1:].detach()
    sequence_cursor += length
    sample_position_cursor = max(
        sample_position_cursor + length,
        int(position_ids.max().item()) + 1,
    )
    sample_splits.append(length)
    sample_attn_modes.append("causal")
    return sequence_cursor, sample_position_cursor


def _plan_image_understanding_item(
    item: ConversationItem,
    *,
    sequence_cursor: int,
    sample_position_cursor: int,
    device: torch.device,
    sample_splits: list[int],
    sample_attn_modes: list[str],
) -> tuple[int, int]:
    vit_embeds = item.meta.get("packed_vit_embeds")
    vit_len = int(vit_embeds.shape[0]) if torch.is_tensor(vit_embeds) else _expected_vit_len(item)
    if item.meta.get("bagel_train_exact_no_boundaries"):
        length = vit_len
        vit_indexes = torch.arange(sequence_cursor, sequence_cursor + length, device=device, dtype=torch.long)
        position_ids = _position_ids(item, sample_position_cursor, length, device=device, full=True)
        item.meta[BAGEL_TRAIN_VIT_INDEXES] = vit_indexes
        item.meta[BAGEL_TRAIN_SEQUENCE_INDEXES] = vit_indexes
        item.meta[BAGEL_TRAIN_TEXT_INDEXES] = torch.empty(0, device=device, dtype=torch.long)
        item.meta[BAGEL_TRAIN_TEXT_IDS] = torch.empty(0, device=device, dtype=torch.long)
        item.meta[BAGEL_TRAIN_SPLIT_LEN] = length
        item.meta[BAGEL_TRAIN_ATTN_MODE] = "causal"
        item.meta["bagel_train_position_ids"] = position_ids
        sequence_cursor += length
        sample_splits.append(length)
        sample_attn_modes.append("causal")
        sample_position_cursor = max(sample_position_cursor + 1, int(position_ids.max().item()) + 1)
        return sequence_cursor, sample_position_cursor

    boundary_ids = item.meta.get(BAGEL_TRAIN_TEXT_IDS)
    if not torch.is_tensor(boundary_ids):
        raise ValueError("BAGEL image understanding item requires boundary text ids before layout planning.")
    boundary_ids = boundary_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
    length = vit_len + int(boundary_ids.numel())
    text_indexes = torch.tensor([sequence_cursor, sequence_cursor + length - 1], device=device, dtype=torch.long)
    vit_indexes = torch.arange(sequence_cursor + 1, sequence_cursor + 1 + vit_len, device=device, dtype=torch.long)
    position_ids = torch.full((length,), sample_position_cursor, device=device, dtype=torch.long)
    item.meta[BAGEL_TRAIN_TEXT_IDS] = boundary_ids
    item.meta[BAGEL_TRAIN_TEXT_INDEXES] = text_indexes
    item.meta[BAGEL_TRAIN_VIT_INDEXES] = vit_indexes
    item.meta[BAGEL_TRAIN_SEQUENCE_INDEXES] = torch.cat((text_indexes, vit_indexes)).sort().values
    item.meta[BAGEL_TRAIN_SPLIT_LEN] = length
    item.meta[BAGEL_TRAIN_ATTN_MODE] = "full"
    item.meta["bagel_train_position_ids"] = position_ids
    sequence_cursor += length
    sample_splits.append(length)
    sample_attn_modes.append("full")
    return sequence_cursor, sample_position_cursor + 1


def _plan_image_generation_item(
    item: ConversationItem,
    *,
    sequence_cursor: int,
    sample_position_cursor: int,
    device: torch.device,
    sample_splits: list[int],
    sample_attn_modes: list[str],
) -> tuple[int, int]:
    latent = item.meta.get("padded_latent")
    latent_shape = item.meta.get("patchified_vae_latent_shape")
    if not torch.is_tensor(latent) or not isinstance(latent_shape, tuple):
        raise ValueError("BAGEL image generation target was not prepared by VAE pre_forward.")
    h, w = int(latent_shape[0]), int(latent_shape[1])
    clean_len = int(patchified_clean_latents(latent, h, w).shape[0])
    if item.meta.get("bagel_train_exact_no_boundaries"):
        length = clean_len
        vae_indexes = torch.arange(sequence_cursor, sequence_cursor + length, device=device, dtype=torch.long)
        position_ids = _position_ids(item, sample_position_cursor, length, device=device, full=True)
        item.meta[BAGEL_TRAIN_VAE_INDEXES] = vae_indexes
        item.meta[BAGEL_TRAIN_MSE_INDEXES] = vae_indexes
        item.meta[BAGEL_TRAIN_SEQUENCE_INDEXES] = vae_indexes
        item.meta[BAGEL_TRAIN_TEXT_INDEXES] = torch.empty(0, device=device, dtype=torch.long)
        item.meta[BAGEL_TRAIN_TEXT_IDS] = torch.empty(0, device=device, dtype=torch.long)
        item.meta[BAGEL_TRAIN_SPLIT_LEN] = length
        item.meta[BAGEL_TRAIN_ATTN_MODE] = "causal"
        item.meta["bagel_train_position_ids"] = position_ids
        sequence_cursor += length
        sample_splits.append(length)
        sample_attn_modes.append("causal")
        sample_position_cursor = max(sample_position_cursor + length, int(position_ids.max().item()) + 1)
        return sequence_cursor, sample_position_cursor

    boundary_ids = item.meta.get(BAGEL_TRAIN_TEXT_IDS)
    if not torch.is_tensor(boundary_ids):
        raise ValueError("BAGEL image generation item requires boundary text ids before layout planning.")
    boundary_ids = boundary_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
    length = clean_len + int(boundary_ids.numel())
    text_indexes = torch.tensor([sequence_cursor, sequence_cursor + length - 1], device=device, dtype=torch.long)
    vae_indexes = torch.arange(sequence_cursor + 1, sequence_cursor + 1 + clean_len, device=device, dtype=torch.long)
    position_ids = torch.arange(sample_position_cursor, sample_position_cursor + length, device=device)
    item.meta[BAGEL_TRAIN_TEXT_IDS] = boundary_ids
    item.meta[BAGEL_TRAIN_TEXT_INDEXES] = text_indexes
    item.meta[BAGEL_TRAIN_VAE_INDEXES] = vae_indexes
    item.meta[BAGEL_TRAIN_MSE_INDEXES] = vae_indexes
    item.meta[BAGEL_TRAIN_SEQUENCE_INDEXES] = torch.cat((text_indexes, vae_indexes)).sort().values
    item.meta[BAGEL_TRAIN_SPLIT_LEN] = length
    item.meta[BAGEL_TRAIN_ATTN_MODE] = "noise"
    item.meta["bagel_train_position_ids"] = position_ids
    sequence_cursor += length
    sample_splits.append(length)
    sample_attn_modes.append("noise")
    return sequence_cursor, sample_position_cursor + length


def _raw_training_text_ids(text_encoder: Any, item: ConversationItem) -> torch.Tensor:
    value = item.value
    device = text_encoder.device
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.long).reshape(-1)
    if not isinstance(value, str):
        raise TypeError(f"BAGEL raw training text must be str or token tensor, got {type(value).__name__}.")
    tokenizer = getattr(text_encoder, "_tokenizer", None)
    if tokenizer is None:
        raise ValueError("BAGEL tokenizer is required for raw training text.")
    return torch.tensor(tokenizer.encode(value, add_special_tokens=False), device=device, dtype=torch.long)


def _position_ids(
    item: ConversationItem,
    start: int,
    length: int,
    *,
    device: torch.device,
    full: bool = False,
) -> torch.Tensor:
    exact_position_ids = item.meta.get("bagel_train_position_ids")
    if torch.is_tensor(exact_position_ids):
        return exact_position_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
    if full:
        return torch.full((length,), start, device=device, dtype=torch.long)
    return torch.arange(start, start + length, device=device, dtype=torch.long)


def _expected_vit_len(item: ConversationItem) -> int:
    vit_token_lens = item.meta.get("vit_token_lens")
    if torch.is_tensor(vit_token_lens):
        return int(vit_token_lens.reshape(-1).sum().item())
    vit_embeds = item.meta.get("packed_vit_embeds")
    if torch.is_tensor(vit_embeds):
        return int(vit_embeds.shape[0])
    raise ValueError("BAGEL image understanding item requires vit_token_lens or packed_vit_embeds metadata.")
