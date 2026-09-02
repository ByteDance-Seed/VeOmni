# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Wan-local FA3 fp8 path. Not a kernel row and not a global FA3 patch.

Self-attn with a finite ``last_loss`` quantizes QKV and calls FA3 with
descale tensors. Cross-attn, missing/NaN ``last_loss``, and non-FA3 impls
stay on the generic attention kernel.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange


try:
    import flash_attn_interface
except ModuleNotFoundError:
    flash_attn_interface = None


_FA3_IMPLS = frozenset({"flash_attention_3", "veomni_flash_attention_3"})


def stochastic_round_tensor(x: torch.Tensor) -> torch.Tensor:
    """Stochastically round ``x`` to the nearest integers."""
    floor_x = torch.floor(x)
    frac = x - floor_x
    rand_vals = torch.rand_like(x)
    round_up = rand_vals < frac
    return floor_x + round_up.to(x.dtype)


def symmetric_quantize(x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head symmetric quantize ``[B, S, H, D]`` to ``dtype``.

    Returns the quantized tensor and per-head scales ``[B, H]``.
    """
    x = x.to(torch.float32)
    max_vals = x.abs().amax(dim=(1, 3), keepdim=True)
    finfo = torch.finfo(dtype)
    eps = 1e-12
    scales = (max_vals + eps) / finfo.max
    scales = scales.clamp(min=eps)
    x_scaled = x / scales
    x_rounded = stochastic_round_tensor(x_scaled)
    x_clamped = x_rounded.clamp(min=finfo.min, max=finfo.max)
    x_quantized = x_clamped.to(dtype)
    scales = scales.squeeze((1, 3)).to(torch.float32)
    return x_quantized, scales


def should_use_fa3_fp8(
    impl: str,
    *,
    is_self_attn: bool,
    last_loss: float | None,
) -> bool:
    """True only for FA3 self-attn with a finite previous-step loss."""
    if impl not in _FA3_IMPLS or not is_self_attn or last_loss is None:
        return False
    return not math.isnan(last_loss)


def flash_attention_3_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Quantize ``[B, H, S, D]`` QKV and run FA3. Returns ``[B, S, H, D]``."""
    if flash_attn_interface is None:
        raise ImportError("Wan FA3 fp8 requires the flash_attn_interface package.")

    head_dim = query.shape[-1]
    q = rearrange(query, "b n s d -> b s n d", d=head_dim)
    k = rearrange(key, "b n s d -> b s n d", d=head_dim)
    v = rearrange(value, "b n s d -> b s n d", d=head_dim)
    original_q, original_k, original_v = q, k, v
    q, qscale = symmetric_quantize(q)
    k, kscale = symmetric_quantize(k)
    v, vscale = symmetric_quantize(v)
    return flash_attn_interface.flash_attn_func(
        q,
        k,
        v,
        q_descale=qscale,
        k_descale=kscale,
        v_descale=vscale,
        original_q=original_q,
        original_k=original_k,
        original_v=original_v,
    )
