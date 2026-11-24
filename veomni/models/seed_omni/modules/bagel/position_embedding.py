"""Shared 2D sin-cos position embedding for BAGEL vision and latent patch grids."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor


class PositionEmbedding(nn.Module):
    """Fixed 2D sin-cos position embedding table indexed by flattened patch IDs."""

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
        """Populate ``pos_embed`` with deterministic 2D sin-cos values."""
        if self.pos_embed.is_meta:
            return
        new_data = _get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        if isinstance(self.pos_embed.data, DTensor):
            local_target = self.pos_embed.data.to_local()
            sharded = distribute_tensor(
                new_data.to(device=local_target.device, dtype=local_target.dtype),
                self.pos_embed.data.device_mesh,
                self.pos_embed.data.placements,
                src_data_rank=None,
            )
            local_target.copy_(sharded.to_local().to(device=local_target.device, dtype=local_target.dtype))
        elif hasattr(self.pos_embed.data, "to_local"):
            self.pos_embed.data.to_local().copy_(new_data)
        else:
            self.pos_embed.data.copy_(new_data)

    def _init_weights(self) -> None:
        self.reset_parameters()

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        return self.pos_embed[position_ids.to(device=self.pos_embed.device)]


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Build a ``[grid_size**2, embed_dim]`` 2D sin-cos position embedding tensor."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.from_numpy(np.concatenate([emb_h, emb_w], axis=1)).float()


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Build a 1D sin-cos position embedding array from flattened grid coordinates."""
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


__all__ = ["PositionEmbedding"]
