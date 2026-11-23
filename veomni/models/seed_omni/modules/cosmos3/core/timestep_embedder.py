"""Sinusoidal timestep conditioning for the Cosmos3 generation tower.

``linear_1`` / ``linear_2`` are the released checkpoint's names.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def _build_frequencies(dim: int, max_period: int) -> Tensor:
    half = dim // 2
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)


def timestep_embedding(t: Tensor, dim: int, *, max_period: int = 10000, frequencies: Tensor | None = None) -> Tensor:
    """``(N,)`` timesteps -> ``(N, dim)`` ``[cos, sin]`` features."""
    if frequencies is None:
        frequencies = _build_frequencies(dim, max_period)
    args = t.to(torch.float32)[:, None] * frequencies.to(device=t.device, dtype=torch.float32)[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """256-dim sinusoid -> Linear -> SiLU -> Linear."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        # Built on the CPU like the reference, so every device sees the same values.
        self.register_buffer(
            "_timestep_frequencies", _build_frequencies(frequency_embedding_size, 10000), persistent=False
        )

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.linear_1.bias)
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, t: Tensor) -> Tensor:
        # The MLP runs in fp32 like the reference's (never bf16-cast) embedder; a bf16
        # MLP is ~3e-3 off on the conditioning vector.
        features = timestep_embedding(t, self.frequency_embedding_size, frequencies=self._timestep_frequencies)
        hidden = nn.functional.linear(
            features, self.linear_1.weight.to(torch.float32), self.linear_1.bias.to(torch.float32)
        )
        hidden = self.act(hidden)
        out = nn.functional.linear(
            hidden, self.linear_2.weight.to(torch.float32), self.linear_2.bias.to(torch.float32)
        )
        return out.to(self.linear_1.weight.dtype)


def add_time_conditioning(
    hidden: Tensor,
    sigma: Tensor,
    embedder: TimestepEmbedder,
    timestep_scale: float,
    *,
    row_weight: Tensor | None = None,
) -> Tensor:
    """``hidden + embedder(sigma * timestep_scale) * row_weight``.

    ``row_weight`` is ``1 - condition_mask`` per row: clean conditioning frames get no
    timestep embedding, as in the reference.
    """
    sigma = torch.as_tensor(sigma).to(device=hidden.device, dtype=torch.float32).reshape(-1)
    if sigma.numel() == 1:
        # One timestep per sample: embed once and broadcast, like the reference; an
        # [N, F] GEMM over repeated rows accumulates differently.
        embedding = embedder(sigma * timestep_scale).to(hidden.dtype).expand(hidden.shape[0], -1)
    else:
        embedding = embedder(sigma * timestep_scale).to(hidden.dtype)
    if row_weight is not None:
        embedding = embedding * row_weight.to(device=hidden.device, dtype=hidden.dtype).reshape(-1, 1)
    return hidden + embedding
