"""The diffusers Wan2.2 VAE with Cosmos3's latent normalization."""

from __future__ import annotations

import os

import torch


class Wan22VAE(torch.nn.Module):
    """``[B, 3, T, H, W]`` in ``[-1, 1]`` -> normalized posterior-mode latents ``[B, 48, T_lat, H/16, W/16]``."""

    def __init__(self, vae: torch.nn.Module, *, component_name: str = "vae"):
        super().__init__()
        self.vae = vae
        self.component_name = component_name

    @classmethod
    def from_pretrained(cls, base_model_path: str, *, subfolder: str = "vae", torch_dtype=torch.float32):
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder=subfolder or None, torch_dtype=torch_dtype)
        vae.eval()
        vae.requires_grad_(False)
        return cls(vae, component_name=os.path.join(base_model_path, subfolder) if subfolder else base_model_path)

    @property
    def config(self):
        return self.vae.config

    @property
    def spatial_scale(self) -> int:
        return int(self.vae.config.scale_factor_spatial)

    @property
    def temporal_scale(self) -> int:
        return int(self.vae.config.scale_factor_temporal)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def _channel_stats(self, name: str, *, device, dtype) -> torch.Tensor:
        values = getattr(self.vae.config, name)
        return torch.as_tensor([float(value) for value in values], dtype=dtype, device=device).view(1, -1, 1, 1, 1)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self._channel_stats("latents_mean", device=latents.device, dtype=latents.dtype)
        std = self._channel_stats("latents_std", device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std

    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
        # ``mode()``: the reference trains against the posterior mean, not a sample.
        return self.normalize_latents(self.vae.encode(video).latent_dist.mode())
