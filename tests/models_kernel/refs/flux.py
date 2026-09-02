# Copyright 2024-2025 The Black-forest-labs Authors. All rights reserved.
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

"""Adapted official Flux eager math for models_kernel toys.

Source:
- Kind: adapted
- From: ``veomni/models/transformers/flux/modeling_flux.py``
  (``RMSNorm``, ``RoPEEmbedding``, ``FluxJointAttention``, ``flash_attention``)
- Upstream: https://github.com/black-forest-labs/flux (Black Forest Labs copyright on the VeOmni file)
- Not from: HuggingFace ``transformers.models`` or a live ``veomni.models`` import
- Changes: eager RMSNorm only. Attention is SDPA. Dropped Liger class-swap, FA2/FA3,
  Ulysses, and ip-adapter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones((dim,)))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, attn_mask=None):
    """Adapted Flux attention: SDPA only so the toy can run on CPU."""
    if attn_mask is not None and causal:
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(~attn_mask, float("-inf"))
        causal_bias = torch.triu(torch.full_like(attn_mask, float("-inf")), diagonal=1)
        attn_mask = attn_mask + causal_bias
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal and attn_mask is None)


class RoPEEmbedding(torch.nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def rope(self, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0, "The dimension must be even."
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        batch_size, _seq_length = pos.shape
        out = torch.einsum("...n,d->...nd", pos, omega)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        return stacked_out.view(batch_size, -1, dim // 2, 2, 2).float()

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([self.rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


class FluxJointAttention(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim, only_out_a=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.only_out_a = only_out_a
        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.b_to_qkv = torch.nn.Linear(dim_b, dim_b * 3)
        self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_q_b = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_b = RMSNorm(head_dim, eps=1e-6)
        self.a_to_out = torch.nn.Linear(dim_a, dim_a)
        if not only_out_a:
            self.b_to_out = torch.nn.Linear(dim_b, dim_b)

    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def forward(self, hidden_states_a, hidden_states_b, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None):
        batch_size = hidden_states_a.shape[0]
        qkv_a = self.a_to_qkv(hidden_states_a)
        qkv_a = qkv_a.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_a, k_a, v_a = qkv_a.chunk(3, dim=1)
        q_a, k_a = self.norm_q_a(q_a), self.norm_k_a(k_a)

        qkv_b = self.b_to_qkv(hidden_states_b)
        qkv_b = qkv_b.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=1)
        q_b, k_b = self.norm_q_b(q_b), self.norm_k_b(k_b)

        q = torch.concat([q_b, q_a], dim=2)
        k = torch.concat([k_b, k_a], dim=2)
        v = torch.concat([v_b, v_a], dim=2)
        q, k = self.apply_rope(q, k, image_rotary_emb)
        hidden_states = flash_attention(q, k, v, causal=False, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states_b, hidden_states_a = (
            hidden_states[:, : hidden_states_b.shape[1]],
            hidden_states[:, hidden_states_b.shape[1] :],
        )
        hidden_states_a = self.a_to_out(hidden_states_a)
        if self.only_out_a:
            return hidden_states_a
        hidden_states_b = self.b_to_out(hidden_states_b)
        return hidden_states_a, hidden_states_b
