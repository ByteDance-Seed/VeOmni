"""Native equivalents of the SeedVR2 LayerNorm/RMSNorm checkpoint modules."""

from typing import Callable

import torch
from torch import nn


norm_layer_type = Callable[[int, float, bool], nn.Module]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, hidden_states):
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        normalized = normalized.to(hidden_states.dtype)
        return normalized if self.weight is None else normalized * self.weight


def get_norm_layer(norm_type):
    def make(dim, eps, elementwise_affine):
        if norm_type is None:
            return nn.Identity()
        if norm_type in {"layer", "fusedln"}:
            return nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        if norm_type in {"rms", "fusedrms"}:
            return RMSNorm(dim, eps, elementwise_affine)
        raise ValueError(f"Unsupported SeedVR2 normalization: {norm_type}")

    return make
