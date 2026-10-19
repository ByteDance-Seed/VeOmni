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

"""QKV layout and compute dtype handling for the Flex adapter."""

import torch


def _normalize_flex_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Normalize THD or BHSD inputs to BHSD without guessing axis order."""
    if query.ndim not in (3, 4) or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("FlexAttention requires Q/K/V with the same ndim: 3D [T,H,D] or 4D [B,H,S,D].")
    if any(dim == 0 for tensor in (query, key, value) for dim in tensor.shape):
        raise ValueError("FlexAttention does not support query/key/value tensors with zero dimensions.")
    token_major = query.ndim == 3
    if token_major:
        query, key, value = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (query, key, value))
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("FlexAttention requires matching Q/K/V batch sizes.")
    if key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]:
        raise ValueError("FlexAttention requires matching K/V head counts and sequence lengths.")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("FlexAttention requires matching Q/K head dimensions.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            f"FlexAttention GQA requires query heads ({query.shape[1]}) to be divisible by "
            f"key/value heads ({key.shape[1]})."
        )
    return query, key, value, token_major


def _flex_fa4_compute_dtype(module: torch.nn.Module, query: torch.Tensor) -> torch.dtype:
    """Recover the compute dtype when LayerNorm left QKV in fp32.

    Matches the flash adapter: autocast, then a quantized config hint, then the
    first Linear weight. A fully fp32 module stays fp32.
    """
    if query.dtype != torch.float32:
        return query.dtype
    if torch.is_autocast_enabled():
        return torch.get_autocast_gpu_dtype()
    config = getattr(module, "config", None)
    pre_quant = getattr(config, "_pre_quantization_dtype", None)
    if pre_quant is not None:
        return pre_quant
    linear = next((layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)), None)
    if linear is not None:
        return linear.weight.dtype
    return query.dtype


def _maybe_recast_qkv_for_fa4(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast fp32 QKV back to a FA4-legal dtype when the module is low precision."""
    target_dtype = _flex_fa4_compute_dtype(module, query)
    if target_dtype == query.dtype:
        return query, key, value
    return query.to(target_dtype), key.to(target_dtype), value.to(target_dtype)
