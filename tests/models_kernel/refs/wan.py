# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""Adapted official Wan eager math for models_kernel toys.

Source:
- Kind: adapted
- From: ``veomni/models/transformers/wan/modeling_wan.py`` and ``config_wan.py``
- Upstream: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
- Not from: HuggingFace ``transformers.models`` (no Wan) or a live ``veomni.models`` import
- Changes: single-process eager only. Dropped Ulysses SP, FA3/sage, ``VeomniKernel``,
  and Diffusers ``save_pretrained``. ``rope_apply`` keeps VeOmni's packed ``[B, S, N*D]``
  face, not Wan2.1's per-sample ``grid_sizes`` loop.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange


class WanConfig:
    """Adapted from Alibaba Wan ``WanConfig``. Diffusers save helpers omitted."""

    def __init__(
        self,
        patch_size=None,
        dim=5120,
        eps=1e-06,
        ffn_dim=13824,
        freq_dim=256,
        in_dim=36,
        num_heads=40,
        num_layers=40,
        out_dim=16,
        text_dim=4096,
        text_len=512,
        has_image_input="false",
    ):
        if patch_size is None:
            patch_size = [1, 2, 2]
        self.patch_size = patch_size
        self.dim = dim
        self.eps = eps
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.text_dim = text_dim
        self.text_len = text_len
        self.has_image_input = has_image_input


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    **kwargs,
):
    head_dim = query.shape[-1]
    scaling = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def rope_apply(x, **kwargs):
    """VeOmni packed ``rope_apply`` from ``modeling_wan.py``.

    Math is the Wan2.1 complex multiply. Layout is VeOmni ``[B, S, N*D]``,
    not upstream ``rope_apply(x, grid_sizes, freqs)``.
    """
    freqs = kwargs.pop("freqs")
    head_dim = kwargs.pop("head_dim")
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.is_causal = False

    def forward(self, query_states, key_states, value_states, **kwargs):
        query_states = rearrange(query_states, "b s (n d) -> b n s d", d=self.head_dim)
        key_states = rearrange(key_states, "b s (n d) -> b n s d", d=self.head_dim)
        value_states = rearrange(value_states, "b s (n d) -> b n s d", d=self.head_dim)
        attention_mask = kwargs.pop("attention_mask", None)
        attn_output, _ = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            **kwargs,
        )
        return rearrange(attn_output, "b s n d -> b s (n d)")


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads, self.head_dim)

    def forward(self, x, freqs, cos, sin, last_loss, self_attn_mask=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs=freqs, cos=cos, sin=sin, head_dim=self.head_dim)
        k = rope_apply(k, freqs=freqs, cos=cos, sin=sin, head_dim=self.head_dim)
        x = self.attn(q, k, v, last_loss=last_loss, isSelfAttn=True, attention_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads, self.head_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v, **kwargs)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = self.attn(q, k_img, v_img, head_dim=self.head_dim, **kwargs)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, cos, sin, last_loss, self_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, cos, sin, last_loss, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        batch, dim = t_mod.shape
        t_mod = t_mod.view(batch, 1, dim)
        modulation = self.modulation.expand(batch, -1, -1).to(t_mod)
        combined = modulation + t_mod
        shift, scale = (chunk.squeeze(1) for chunk in combined.chunk(2, dim=1))
        normalized = self.norm(x)
        modulated = normalized * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.head(modulated)


class WanModel(nn.Module):
    def __init__(self, config: WanConfig):
        super().__init__()
        self.dim = config.dim
        self.freq_dim = config.freq_dim
        self.has_image_input = config.has_image_input == "true"
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv3d(
            config.in_dim,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, config.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.dim, config.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.has_image_input, config.dim, config.num_heads, config.ffn_dim, config.eps)
                for _ in range(config.num_layers)
            ]
        )
        self.head = Head(config.dim, config.out_dim, tuple(config.patch_size), config.eps)
        self.freqs = precompute_freqs_cis_3d(config.dim // config.num_heads)

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, last_loss=None, **kwargs):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        x, (f, h, w) = self.patchify(x)
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        cos = freqs.real.squeeze().contiguous()
        sin = freqs.imag.squeeze().contiguous()
        for block in self.blocks:
            x = block(x, context, t_mod, freqs, cos, sin, last_loss=last_loss)
        x = self.head(x, t)
        return self.unpatchify(x, (f, h, w))
