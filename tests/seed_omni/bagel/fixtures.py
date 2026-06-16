"""BAGEL-specific deterministic tensor fixtures for parity tests."""

from __future__ import annotations

from typing import Any

import torch


def latent_position_ids(
    height: int,
    width: int,
    *,
    max_latent_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.arange(height, device=device, dtype=torch.long)[:, None] * max_latent_size
    cols = torch.arange(width, device=device, dtype=torch.long)[None]
    return (rows + cols).flatten()


def synthetic_latent_fixture(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    latent_grid = (2, 2)
    latent_channels = 16
    latent_patch_size = 2
    h, w = latent_grid
    target_latent = torch.linspace(
        -0.75,
        0.75,
        steps=latent_channels * h * latent_patch_size * w * latent_patch_size,
        device=device,
        dtype=dtype,
    ).reshape(1, latent_channels, h * latent_patch_size, w * latent_patch_size)
    num_vae_tokens = h * w
    patch_dim = latent_channels * latent_patch_size * latent_patch_size
    return {
        "target_latent": target_latent,
        "latent_grid": latent_grid,
        "latent_patch_size": latent_patch_size,
        "max_latent_size": 64,
        "flow_timesteps": torch.linspace(-0.5, 0.5, steps=num_vae_tokens, device=device),
        "flow_noise": torch.linspace(
            -0.25,
            0.25,
            steps=num_vae_tokens * patch_dim,
            device=device,
            dtype=dtype,
        ).reshape(num_vae_tokens, patch_dim),
    }


def synthetic_vit_fixture(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    vit_tokens = 2
    vit_patch_dim = 3 * 14 * 14
    return {
        "vit_tokens": torch.linspace(
            -1.0,
            1.0,
            steps=vit_tokens * vit_patch_dim,
            device=device,
            dtype=dtype,
        ).reshape(vit_tokens, vit_patch_dim),
        "vit_position_ids": torch.arange(vit_tokens, device=device, dtype=torch.long),
        "vit_token_lens": torch.tensor([vit_tokens], device=device, dtype=torch.int32),
    }


__all__ = [
    "latent_position_ids",
    "synthetic_latent_fixture",
    "synthetic_vit_fixture",
]
