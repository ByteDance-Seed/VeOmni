"""Cosmos3 VAE: the frozen Wan2.2 ``AutoencoderKLWan`` and Cosmos3's latent normalization."""

from typing import Any, Dict

import torch

from ....mixins.offline_encoding_mixin import OfflineEncodingMixin
from ...module_modeling_base import PretrainedOmniModule
from .configuration import Cosmos3VaeConfig
from .processing import Cosmos3VaePreprocessor


class Cosmos3Vae(OfflineEncodingMixin, PretrainedOmniModule):
    """``encode`` (online) == ``online_process(offline_encode(...))``.

    The reference trains against the posterior *mode*, not a sample, so the whole
    encoding is deterministic and caches as is: ``offline_encode`` is ``encode``,
    and ``online_process`` has nothing left to do.
    """

    config_class = Cosmos3VaeConfig
    preprocessor_class = Cosmos3VaePreprocessor
    main_input_name = "videos"

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # The diffusers VAE runs its own attention, not an HF attention backend.
        return attn_implementation

    def __init__(self, config: Cosmos3VaeConfig):
        super().__init__(config)
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_config(config.vae_config)
        self.post_init()

    def freeze_model(self) -> None:
        self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, videos: list[torch.Tensor], **kwargs: Any) -> Dict[str, list[torch.Tensor]]:
        """``(3, T, H, W)`` clips in ``[-1, 1]`` -> normalized ``(C, T_lat, H/16, W/16)`` latents."""
        del kwargs
        return {"latents": [self._encode_clip(video) for video in videos]}

    def offline_encode(self, videos: list[torch.Tensor], **kwargs: Any) -> Dict[str, list[torch.Tensor]]:
        return self.encode(videos, **kwargs)

    def online_process(self, **kwargs: Any) -> Dict[str, Any]:
        """The cached latent is already final; reads nothing, so it runs on meta."""
        del kwargs
        return {}

    def _encode_clip(self, video: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode(video.unsqueeze(0)).latent_dist.mode()
        mean = self._channel_stats("latents_mean", like=latent)
        std = self._channel_stats("latents_std", like=latent)
        return ((latent - mean) / std)[0]

    def _channel_stats(self, name: str, *, like: torch.Tensor) -> torch.Tensor:
        values = [float(value) for value in self.config.vae_config[name]]
        return torch.as_tensor(values, dtype=like.dtype, device=like.device).view(1, -1, 1, 1, 1)


__all__ = ["Cosmos3Vae"]
