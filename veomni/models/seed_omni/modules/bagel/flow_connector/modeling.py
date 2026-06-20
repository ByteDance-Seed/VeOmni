"""BAGEL flow connector.

``embed_latent`` patchifies VAE latent grids, applies rectified-flow timestep
conditioning, projects latent tokens to MoT hidden width, and writes embeddings
back to the carrier. ``decode_velocity`` projects MoT hidden states back to
patch-latent velocity tokens and optionally computes flow MSE.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration import BagelFlowConnectorConfig
from .modulemixin import BagelFlowConnectorModuleMixin
from .processing import autocast_enabled_for_device


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        first_weight = self.mlp[0].weight
        return self.mlp(t_freq.to(device=first_weight.device, dtype=first_weight.dtype))


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int) -> None:
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pos_embed.data.copy_(_get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side))

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        return self.pos_embed[position_ids.to(device=self.pos_embed.device)]


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    return torch.from_numpy(_get_2d_sincos_pos_embed_from_grid(embed_dim, grid)).float()


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class BagelFlowConnector(BagelFlowConnectorModuleMixin, PreTrainedModel):
    config_class = BagelFlowConnectorConfig
    base_model_prefix = "bagel_flow_connector"
    main_input_name = "hidden_states"
    _no_split_modules: list[str] = []
    supports_gradient_checkpointing = True

    def __init__(self, config: BagelFlowConnectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False
        self.time_embedder = TimestepEmbedder(config.hidden_size, config.timestep_frequency_embedding_size)
        self.vae2llm = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(config.max_latent_size, config.hidden_size)
        self.post_init()
        nn.init.constant_(self.llm2vae.weight, 0)
        nn.init.constant_(self.llm2vae.bias, 0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, PositionEmbedding):
            module.reset_parameters()
            return
        super()._init_weights(module)

    def embed_latent(
        self,
        latents: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        timesteps: torch.Tensor | None = None,
        conversation_list: Any | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del generation_kwargs, kwargs
        if latents is None and conversation_list is not None:
            return self._embed_context_latents(conversation_list)
        if latents is None:
            dummy = self.dummy_inputs(kind="embed_latent")
            outputs = self.embed_latent(**dummy)
            outputs["is_dummy"] = True
            return outputs
        if position_ids is None or timesteps is None:
            raise ValueError("BagelFlowConnector.embed_latent requires position_ids and timesteps.")

        weight = self.vae2llm.weight
        latent_dtype = latents.dtype if autocast_enabled_for_device(weight.device) else weight.dtype
        latents = latents.to(device=weight.device, dtype=latent_dtype)
        position_ids = position_ids.to(device=self.latent_pos_embed.pos_embed.device, dtype=torch.long).reshape(-1)
        timesteps = timesteps.to(device=self.time_embedder.mlp[0].weight.device, dtype=torch.float32).reshape(-1)
        if timesteps.numel() not in {1, latents.shape[0]}:
            raise ValueError("timesteps must be a scalar or have one value per latent token.")
        if position_ids.numel() != latents.shape[0]:
            raise ValueError("position_ids must have one value per latent token.")

        latent_embeds = self.vae2llm(latents)
        time_embeds = self.time_embedder(timesteps)
        pos_embeds = self.latent_pos_embed(position_ids)
        return {
            "latent_embeds": latent_embeds
            + time_embeds.to(device=latent_embeds.device, dtype=latent_embeds.dtype)
            + pos_embeds.to(device=latent_embeds.device, dtype=latent_embeds.dtype)
        }

    def decode_velocity(self, hidden_states: torch.Tensor | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if hidden_states is None:
            dummy = self.dummy_inputs(kind="decode_velocity")
            outputs = self.decode_velocity(**dummy)
            outputs["is_dummy"] = True
            return outputs
        weight = self.llm2vae.weight
        hidden_states = hidden_states.to(device=weight.device)
        return {"velocity": self.llm2vae(hidden_states)}


__all__ = [
    "BagelFlowConnector",
    "BagelFlowConnectorConfig",
    "PositionEmbedding",
    "TimestepEmbedder",
]
