"""Latent patch and timestep helpers for BAGEL flow connector."""

from __future__ import annotations

from typing import Any

import torch

from ....conversation import ConversationItem


def preprocess_latent_embed(
    embed_items: list[ConversationItem],
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    timestep_shift: float,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Build noised latent embed inputs and training velocity targets."""
    latents: list[torch.Tensor] = []
    position_ids: list[torch.Tensor] = []
    timesteps: list[torch.Tensor] = []
    embed_lengths: list[int] = []
    for item in embed_items:
        clean, grid_shape = _patchify_latent_grid(
            item.value,
            z_channels=int(config.z_channels),
            latent_patch_size=int(config.latent_patch_size),
        )
        clean = clean.to(device=device, dtype=dtype)
        token_count = int(clean.shape[0])

        noise = torch.randn_like(clean)
        timestep = torch.sigmoid(torch.randn(token_count, device=device, dtype=torch.float32))
        timestep = float(timestep_shift) * timestep / (1.0 + (float(timestep_shift) - 1.0) * timestep)
        timestep_values = timestep.reshape(-1, 1).to(device=clean.device, dtype=torch.float32)
        noised = (1.0 - timestep_values) * clean + timestep_values * noise
        noised = noised.to(dtype=dtype)

        item.meta["timestep"] = timestep.detach()
        item.meta["noise"] = noise.detach()
        item.meta["flow_velocity_target"] = (noise - clean).detach()
        latents.append(noised)
        position_ids.append(
            flattened_position_ids(
                grid_shape,
                max_latent_size=int(config.max_latent_size),
                device=device,
            )
        )
        timesteps.append(timestep)
        embed_lengths.append(token_count)

    return (
        {
            "latents": torch.cat(latents, dim=0),
            "position_ids": torch.cat(position_ids, dim=0),
            "timesteps": torch.cat(timesteps, dim=0),
        },
        embed_lengths,
    )


def preprocess_context_latent_embed(
    embed_items: list[ConversationItem],
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Build clean latent embed inputs for inference context latents."""
    latents: list[torch.Tensor] = []
    position_ids: list[torch.Tensor] = []
    embed_lengths: list[int] = []
    for item in embed_items:
        clean, grid_shape = _patchify_latent_grid(
            item.value,
            z_channels=int(config.z_channels),
            latent_patch_size=int(config.latent_patch_size),
        )
        clean = clean.to(device=device)
        token_count = int(clean.shape[0])

        latents.append(clean)
        position_ids.append(
            flattened_position_ids(
                grid_shape,
                max_latent_size=int(config.max_latent_size),
                device=device,
            )
        )
        embed_lengths.append(token_count)

    return (
        {
            "latents": torch.cat(latents, dim=0),
            "position_ids": torch.cat(position_ids, dim=0),
            "timesteps": torch.zeros(1, device=device, dtype=torch.float32),
        },
        embed_lengths,
    )


def preprocess_decode_velocity(
    decode_items: list[ConversationItem],
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], list[int], torch.Tensor]:
    """Build packed hidden states and velocity targets for training decode."""
    hidden_states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    decode_lengths: list[int] = []
    for item in decode_items:
        hidden = item.value
        if not torch.is_tensor(hidden):
            raise ValueError("BAGEL flow decode_velocity expects tensor hidden states.")
        if hidden.dim() == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        if hidden.dim() != 2:
            raise ValueError(
                f"BAGEL flow decode_velocity expects rank-2 hidden states, got {tuple(item.value.shape)}."
            )
        if int(hidden.shape[-1]) != int(config.hidden_size):
            raise ValueError(
                "BAGEL flow decode_velocity hidden-size mismatch: "
                f"got {hidden.shape[-1]}, expected {config.hidden_size}."
            )
        hidden = hidden.to(device=device)
        hidden_states.append(hidden)
        decode_lengths.append(int(hidden.shape[0]))

        target = item.meta.get("flow_velocity_target").to(device=device, dtype=dtype)
        expected_shape = (hidden.shape[0], int(config.patch_latent_dim))
        if target.shape != expected_shape:
            raise ValueError(
                f"BAGEL flow_velocity_target shape mismatch: got {tuple(target.shape)}, expected {expected_shape}."
            )
        targets.append(target)

    return {"hidden_states": torch.cat(hidden_states, dim=0)}, decode_lengths, torch.cat(targets, dim=0)


def _patchify_latent_grid(
    value: torch.Tensor,
    *,
    z_channels: int,
    latent_patch_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Patchify a latent grid into a list of tokens."""
    latent = value.detach()
    if latent.dim() == 4 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.dim() != 3:
        raise ValueError(f"BAGEL flow connector expects latent grid tensors, got shape {tuple(value.shape)}.")
    if int(latent.shape[0]) != int(z_channels):
        raise ValueError(
            f"BAGEL flow connector latent channel mismatch: got {int(latent.shape[0])}, expected {int(z_channels)}."
        )
    channels, height, width = latent.shape

    patch_size = int(latent_patch_size)
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "BAGEL flow connector latent grid is not divisible by latent_patch_size: "
            f"shape={tuple(latent.shape)}, latent_patch_size={patch_size}."
        )

    grid_h, grid_w = height // patch_size, width // patch_size
    tokens = latent.reshape(channels, grid_h, patch_size, grid_w, patch_size)
    tokens = torch.einsum("chpwq->hwpqc", tokens)
    return tokens.reshape(-1, patch_size * patch_size * channels), (grid_h, grid_w)


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


__all__ = [
    "flattened_position_ids",
    "preprocess_context_latent_embed",
    "preprocess_decode_velocity",
    "preprocess_latent_embed",
    "unpatchify_latent_tokens",
]
