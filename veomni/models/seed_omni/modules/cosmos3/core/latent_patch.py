"""Latent <-> patch rows, in the reference's ``thw, (p, q, c)`` order."""

from __future__ import annotations

import torch


def patchify_latent(latent: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """``[C, F, H, W]`` -> ``[F * H/p * W/p, p*p*C]``."""
    channels, frames, height, width = latent.shape
    ph, pw = patch_size
    if height % ph or width % pw:
        raise ValueError(f"latent spatial size {(height, width)} is not divisible by the patch size {(ph, pw)}.")
    patches = latent.reshape(channels, frames, height // ph, ph, width // pw, pw)
    patches = torch.einsum("cthpwq->thwpqc", patches)
    return patches.reshape(-1, ph * pw * channels)


def unpatchify_latent(
    patches: torch.Tensor, latent_shape: tuple[int, int, int, int], patch_size: tuple[int, int]
) -> torch.Tensor:
    """Inverse of :func:`patchify_latent`."""
    channels, frames, height, width = latent_shape
    ph, pw = patch_size
    shaped = patches.reshape(frames, height // ph, width // pw, ph, pw, channels)
    latent = torch.einsum("thwpqc->cthpwq", shaped)
    return latent.reshape(channels, frames, height, width)


def patch_row_count(latent_shape: tuple[int, int, int, int], patch_size: tuple[int, int]) -> int:
    """Rows :func:`patchify_latent` produces for one latent."""
    _, frames, height, width = latent_shape
    return frames * (height // patch_size[0]) * (width // patch_size[1])
