"""Latent patch and timestep helpers for BAGEL flow connector."""

from __future__ import annotations

from typing import Any

import torch

from ....conversation import ConversationItem, is_dummy


def autocast_enabled_for_device(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.is_autocast_enabled("cuda")
    if device.type == "cpu":
        return torch.is_autocast_enabled("cpu")
    return False


def single_inference_conversation(
    conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None,
) -> list[ConversationItem]:
    if conversation_list is None:
        raise ValueError("BAGEL flow inference requires conversation_list.")
    if not conversation_list:
        raise ValueError("BAGEL flow inference received an empty conversation_list.")
    first = conversation_list[0]
    if isinstance(first, ConversationItem):
        return conversation_list  # type: ignore[return-value]
    if isinstance(first, list):
        if len(conversation_list) != 1:
            raise ValueError("BAGEL flow inference currently supports one sample at a time.")
        return first
    raise TypeError("BAGEL flow inference conversation_list must contain ConversationItem objects.")


def active_output_item(conversation: list[ConversationItem]) -> ConversationItem | None:
    for item in reversed(conversation):
        if item.type == "output":
            return item
    return None


def latent_grid(value: torch.Tensor, *, z_channels: int) -> torch.Tensor:
    latent = value.detach()
    if latent.dim() == 4 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.dim() != 3:
        raise ValueError(f"BAGEL flow connector expects latent grid tensors, got shape {tuple(value.shape)}.")
    if int(latent.shape[0]) != int(z_channels):
        raise ValueError(
            f"BAGEL flow connector latent channel mismatch: got {int(latent.shape[0])}, expected {int(z_channels)}."
        )
    return latent


def is_latent_grid(value: object, *, z_channels: int) -> bool:
    if not torch.is_tensor(value):
        return False
    if value.dim() == 4 and value.shape[0] == 1:
        return value.shape[1] == z_channels
    return value.dim() == 3 and value.shape[0] == z_channels


def is_flow_latent_item(item: ConversationItem, *, z_channels: int) -> bool:
    if is_dummy(item) or item.type != "output":
        return False
    return is_latent_grid(item.value, z_channels=z_channels)


def is_flow_hidden_item(item: ConversationItem, *, hidden_size: int) -> bool:
    if is_dummy(item) or item.type != "output" or not torch.is_tensor(item.value):
        return False
    value = item.value
    if value.dim() == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    return value.dim() == 2 and int(value.shape[-1]) == int(hidden_size)


def patchify_latent_grid(
    value: torch.Tensor,
    *,
    z_channels: int,
    latent_patch_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    latent = latent_grid(value, z_channels=z_channels)
    patch = int(latent_patch_size)
    channels, height, width = latent.shape
    if height % patch != 0 or width % patch != 0:
        raise ValueError(
            "BAGEL flow connector latent grid is not divisible by latent_patch_size: "
            f"shape={tuple(latent.shape)}, latent_patch_size={patch}."
        )
    grid_h = height // patch
    grid_w = width // patch
    tokens = latent.reshape(channels, grid_h, patch, grid_w, patch)
    tokens = torch.einsum("chpwq->hwpqc", tokens)
    return tokens.reshape(-1, patch * patch * channels), (grid_h, grid_w)


def unpatchify_latent_tokens(
    tokens: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    z_channels: int,
    latent_patch_size: int,
) -> torch.Tensor:
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    patch = int(latent_patch_size)
    channels = int(z_channels)
    expected = grid_h * grid_w
    if tokens.dim() != 2:
        raise ValueError(f"BAGEL flow unpatchify expects rank-2 tokens, got shape {tuple(tokens.shape)}.")
    if int(tokens.shape[0]) != expected:
        raise ValueError(f"BAGEL flow token count mismatch: got {tokens.shape[0]}, expected {expected}.")
    if int(tokens.shape[1]) != patch * patch * channels:
        raise ValueError(
            f"BAGEL flow patch dimension mismatch: got {tokens.shape[1]}, expected {patch * patch * channels}."
        )
    latent = tokens.reshape(grid_h, grid_w, patch, patch, channels)
    latent = torch.einsum("hwpqc->chpwq", latent)
    return latent.reshape(channels, grid_h * patch, grid_w * patch)


def flattened_position_ids(
    grid_shape: tuple[int, int],
    *,
    max_latent_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    if grid_h > max_latent_size or grid_w > max_latent_size:
        raise ValueError(
            "BAGEL flow connector latent grid exceeds max_latent_size: "
            f"grid={grid_shape}, max_latent_size={max_latent_size}."
        )
    coords_h = torch.arange(grid_h, device=device, dtype=torch.long)
    coords_w = torch.arange(grid_w, device=device, dtype=torch.long)
    return (coords_h[:, None] * int(max_latent_size) + coords_w).flatten()


def shifted_timestep_logits(logits: torch.Tensor, *, timestep_shift: float) -> torch.Tensor:
    values = torch.sigmoid(logits)
    return float(timestep_shift) * values / (1.0 + (float(timestep_shift) - 1.0) * values)


def flow_noise_from_item_meta(
    meta: dict[str, object],
    clean: torch.Tensor,
) -> torch.Tensor:
    noise = meta.get("noise")
    if torch.is_tensor(noise):
        noise = noise.detach().to(device=clean.device, dtype=clean.dtype)
        if noise.dim() == 1 and noise.numel() == int(clean.shape[-1]):
            noise = noise.unsqueeze(0).expand_as(clean)
        if noise.dim() == 2 and noise.shape[0] == 1 and noise.shape[1] == clean.shape[-1]:
            noise = noise.expand_as(clean)
        if noise.shape != clean.shape:
            raise ValueError(
                f"BAGEL flow noise shape mismatch: got {tuple(noise.shape)}, expected {tuple(clean.shape)}."
            )
        return noise
    return torch.randn_like(clean)


def flow_timestep_from_item_meta(
    meta: dict[str, object],
    *,
    token_count: int,
    device: torch.device,
    timestep_shift: float,
) -> torch.Tensor:
    timestep = meta.get("timestep")
    if torch.is_tensor(timestep):
        timestep = timestep.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if timestep.numel() == 1:
            return timestep.expand(token_count)
        if timestep.numel() != token_count:
            raise ValueError("BAGEL flow timestep must be scalar or have one value per latent token.")
        return timestep
    raw = torch.randn(token_count, device=device, dtype=torch.float32)
    return shifted_timestep_logits(raw, timestep_shift=timestep_shift)


def flow_latent_items(
    conversation_list: list[list[ConversationItem]] | None,
    *,
    z_channels: int,
) -> list[ConversationItem]:
    return [
        item
        for sample in conversation_list or []
        for item in sample
        if is_flow_latent_item(item, z_channels=z_channels)
    ]


def prepare_embed_latent_inputs(
    embed_items: list[ConversationItem],
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    timestep_shift: float,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    latent_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    timestep_parts: list[torch.Tensor] = []
    embed_lengths: list[int] = []
    for item in embed_items:
        clean, grid_shape = patchify_latent_grid(
            item.value,
            z_channels=int(config.z_channels),
            latent_patch_size=int(config.latent_patch_size),
        )
        clean = clean.to(device=device, dtype=dtype)
        token_count = int(clean.shape[0])
        noise = flow_noise_from_item_meta(item.meta, clean)
        timestep = flow_timestep_from_item_meta(
            item.meta,
            token_count=token_count,
            device=device,
            timestep_shift=timestep_shift,
        )
        timestep_values = timestep.reshape(-1, 1).to(device=clean.device, dtype=torch.float32)
        noised = (1.0 - timestep_values) * clean + timestep_values * noise
        item.meta["timestep"] = timestep.detach()
        item.meta["noise"] = noise.detach()
        item.meta["flow_velocity_target"] = (noise - clean).detach()
        latent_parts.append(noised)
        position_parts.append(
            flattened_position_ids(
                grid_shape,
                max_latent_size=int(config.max_latent_size),
                device=device,
            )
        )
        timestep_parts.append(timestep)
        embed_lengths.append(token_count)

    return (
        {
            "latents": torch.cat(latent_parts, dim=0),
            "position_ids": torch.cat(position_parts, dim=0),
            "timesteps": torch.cat(timestep_parts, dim=0),
        },
        embed_lengths,
    )


def prepare_context_embed_latent_inputs(
    embed_items: list[ConversationItem],
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    latent_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    embed_lengths: list[int] = []
    for item in embed_items:
        clean, grid_shape = patchify_latent_grid(
            item.value,
            z_channels=int(config.z_channels),
            latent_patch_size=int(config.latent_patch_size),
        )
        clean = clean.to(device=device)
        token_count = int(clean.shape[0])
        latent_parts.append(clean)
        position_parts.append(
            flattened_position_ids(
                grid_shape,
                max_latent_size=int(config.max_latent_size),
                device=device,
            )
        )
        embed_lengths.append(token_count)

    return (
        {
            "latents": torch.cat(latent_parts, dim=0),
            "position_ids": torch.cat(position_parts, dim=0),
            "timesteps": torch.zeros(1, device=device, dtype=torch.float32),
        },
        embed_lengths,
    )


def scatter_latent_embeds(
    embed_items: list[ConversationItem],
    embed_lengths: list[int],
    latent_embeds: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    offset = 0
    for item, length in zip(embed_items, embed_lengths, strict=True):
        item.value = latent_embeds[offset : offset + length].to(device=device, dtype=dtype)
        offset += length
    if offset != int(latent_embeds.shape[0]):
        raise RuntimeError("BAGEL flow connector latent count mismatch during embed scatter.")


def flow_hidden_items(
    conversation_list: list[list[ConversationItem]] | None,
    *,
    hidden_size: int,
) -> list[ConversationItem]:
    return [
        item
        for sample in conversation_list or []
        for item in sample
        if is_flow_hidden_item(item, hidden_size=hidden_size)
    ]


def prepare_decode_velocity_inputs(
    decode_items: list[ConversationItem],
    *,
    hidden_size: int,
    patch_latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], list[int], torch.Tensor | None]:
    hidden_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    decode_lengths: list[int] = []
    for item in decode_items:
        hidden = item.value
        if hidden.dim() == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        if hidden.dim() != 2:
            raise ValueError(f"BAGEL flow velocity expects hidden states, got shape {tuple(item.value.shape)}.")
        if int(hidden.shape[-1]) != int(hidden_size):
            raise ValueError(
                f"BAGEL flow velocity hidden-size mismatch: got {hidden.shape[-1]}, expected {hidden_size}."
            )
        hidden = hidden.to(device=device)
        hidden_parts.append(hidden)
        decode_lengths.append(int(hidden.shape[0]))
        target = item.meta.get("flow_velocity_target")
        if torch.is_tensor(target):
            target = target.to(device=device, dtype=dtype)
            expected_shape = (hidden.shape[0], int(patch_latent_dim))
            if target.shape != expected_shape:
                raise ValueError(
                    f"BAGEL flow_velocity_target shape mismatch: got {tuple(target.shape)}, expected {expected_shape}."
                )
            target_parts.append(target)

    if target_parts and len(target_parts) != len(decode_items):
        raise ValueError("BAGEL flow velocity loss requires every decoded item to carry flow_velocity_target.")
    target = torch.cat(target_parts, dim=0) if target_parts else None
    return {"hidden_states": torch.cat(hidden_parts, dim=0)}, decode_lengths, target


def scatter_velocity(
    decode_items: list[ConversationItem],
    decode_lengths: list[int],
    velocity: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    offset = 0
    for item, length in zip(decode_items, decode_lengths, strict=True):
        item.value = velocity[offset : offset + length].to(device=device, dtype=dtype)
        offset += length
    if offset != int(velocity.shape[0]):
        raise RuntimeError("BAGEL flow connector token count mismatch during velocity scatter.")


__all__ = [
    "active_output_item",
    "autocast_enabled_for_device",
    "flattened_position_ids",
    "flow_hidden_items",
    "flow_latent_items",
    "flow_noise_from_item_meta",
    "flow_timestep_from_item_meta",
    "is_flow_hidden_item",
    "is_flow_latent_item",
    "is_latent_grid",
    "latent_grid",
    "patchify_latent_grid",
    "prepare_decode_velocity_inputs",
    "prepare_context_embed_latent_inputs",
    "prepare_embed_latent_inputs",
    "scatter_latent_embeds",
    "scatter_velocity",
    "shifted_timestep_logits",
    "single_inference_conversation",
    "unpatchify_latent_tokens",
]
