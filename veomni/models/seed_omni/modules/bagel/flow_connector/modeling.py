"""BAGEL flow-generation connector layers between VAE tokens and Qwen2 hidden states."""

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration import BagelFlowConnectorConfig
from .modulemixin import BagelFlowConnectorModuleMixin


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
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
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


class BagelFlowConnector(BagelFlowConnectorModuleMixin, PreTrainedModel):
    config_class = BagelFlowConnectorConfig
    base_model_prefix = "bagel_flow_connector"
    main_input_name = "hidden_states"
    _no_split_modules: list[str] = []
    supports_gradient_checkpointing = True

    def __init__(self, config: BagelFlowConnectorConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.time_embedder = TimestepEmbedder(config.hidden_size, config.timestep_frequency_embedding_size)
        self.vae2llm = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(config.max_latent_size, config.hidden_size)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, PositionEmbedding):
            module._init_weights()
            return
        super()._init_weights(module)

    def embed_latent(
        self,
        latents: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if latents is None:
            return self._embed_latent_graph(**kwargs)
        if position_ids is None or timesteps is None:
            raise ValueError("BagelFlowConnector.embed_latent requires position_ids and timesteps.")
        latents = latents.to(device=self.device)
        if not torch.is_autocast_enabled(latents.device.type):
            latents = latents.to(dtype=self.dtype)
        position_ids = position_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        timesteps = timesteps.to(device=self.device).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(latents.shape[0])
        if timesteps.numel() != latents.shape[0]:
            raise ValueError("timesteps must be a scalar or have one value per latent token.")

        latent_embeds = self.vae2llm(latents) + self.time_embedder(timesteps) + self.latent_pos_embed(position_ids)
        return {"latent_embeds": latent_embeds.to(dtype=self.dtype)}

    def decode_velocity(self, hidden_states: Optional[torch.Tensor] = None, **kwargs: Any) -> Dict[str, Any]:
        if hidden_states is None:
            return self._decode_velocity_graph(**kwargs)
        return {"velocity": self.llm2vae(hidden_states.to(device=self.device, dtype=self.dtype))}

    def forward(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        return self.embed_latent(**kwargs)
