# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MagiAttention mask APIs and CP1/SP-aware attention adapter."""

from typing import Callable, Optional

import torch

from .....distributed.parallel_state import get_parallel_state
from .....utils.device import IS_CUDA_AVAILABLE, get_device_type
from ..ulysses import prepare_ulysses_qkv, restore_ulysses_output
from .mask import (
    MagiAttentionMask,
    _require_all,
    create_magi_mask,
    register_veomni_magi_attention_mask_builder,
)


def _resolve_default_magi_backend(device: torch.device) -> Callable:
    """Resolve the platform backend while keeping device-specific code isolated."""
    active_device_type = get_device_type()
    if device.type != active_device_type:
        raise RuntimeError(
            f"VeOmni `magi_attention` received tensors on {device.type}, "
            f"but the active device type is {active_device_type}."
        )

    if IS_CUDA_AVAILABLE:
        from ._fa4_cuda import _fa4_cuda_attention_forward

        return _fa4_cuda_attention_forward

    raise RuntimeError(f"VeOmni `magi_attention` does not yet provide a {active_device_type.upper()} backend.")


def _default_magi_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    *,
    softmax_scale: float | None,
    softcap: float,
):
    """Dispatch MagiAttention to the backend for the active device platform."""
    attention_forward = _resolve_default_magi_backend(query.device)
    return attention_forward(
        query,
        key,
        value,
        q_ranges,
        k_ranges,
        attn_type_map,
        softmax_scale=softmax_scale,
        softcap=softcap,
    )


# Module-level patch slot for the default platform backend.
_magi_attention_forward: Callable = _default_magi_attention_forward


def magi_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: MagiAttentionMask | None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run MagiAttention FFA for CP1 with optional VeOmni Ulysses exchange."""
    del module, kwargs

    if not isinstance(attention_mask, MagiAttentionMask):
        raise TypeError(f"MagiAttention requires a MagiAttentionMask, got {type(attention_mask).__name__}.")
    if dropout != 0.0:
        raise ValueError(f"MagiAttention FFA does not support attention dropout, got dropout={dropout}.")
    if sliding_window is not None:
        raise ValueError(
            "MagiAttention does not accept standalone sliding_window metadata; encode visibility in MagiAttentionMask."
        )

    _validate_qkv(query, key, value)
    if attention_mask.q_ranges.device != query.device:
        raise ValueError(
            "MagiAttentionMask tensors must be on the same device as query/key/value, "
            f"got mask device {attention_mask.q_ranges.device} and query device {query.device}."
        )

    parallel_state = get_parallel_state()
    if parallel_state.cp_size != 1:
        raise ValueError(f"MagiAttention FFA currently supports cp_size == 1, got cp_size={parallel_state.cp_size}.")

    ulysses_enabled = parallel_state.ulysses_enabled and not skip_ulysses
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if ulysses_enabled:
        query, key, value, _ = prepare_ulysses_qkv(
            query,
            key,
            value,
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    _require_all(
        attention_mask.q_ranges[:, 1] <= query.shape[0],
        f"MagiAttention q_ranges must end within the post-exchange query length ({query.shape[0]}).",
    )
    _require_all(
        attention_mask.k_ranges[:, 1] <= key.shape[0],
        f"MagiAttention k_ranges must end within the post-exchange key length ({key.shape[0]}).",
    )

    output, meta = _magi_attention_forward(
        query,
        key,
        value,
        attention_mask.q_ranges,
        attention_mask.k_ranges,
        attention_mask.attn_type_map,
        softmax_scale=scaling,
        softcap=0.0 if softcap is None else softcap,
    )

    output = output.unsqueeze(0)
    lse = meta.lse
    if lse is not None:
        lse = lse.unsqueeze(0).unsqueeze(-1)

    if ulysses_enabled:
        output = restore_ulysses_output(output, group=parallel_state.ulysses_group)
        if lse is not None:
            lse = restore_ulysses_output(lse, group=parallel_state.ulysses_group)

    if lse is not None:
        lse = lse.squeeze(-1).transpose(1, 2).contiguous()

    return output, lse


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.ndim != 4:
            raise ValueError(
                f"MagiAttention requires {name} in [batch, heads, sequence, head_dim] layout, "
                f"got shape {tuple(tensor.shape)}."
            )
        if any(dim == 0 for dim in tensor.shape):
            raise ValueError(f"MagiAttention does not support {name} tensors with zero dimensions.")

    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError(
            "MagiAttention currently requires batch size 1 because FFA consumes packed three-dimensional tensors, "
            f"got query/key/value batch sizes {query.shape[0]}/{key.shape[0]}/{value.shape[0]}."
        )
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            f"MagiAttention GQA requires query heads ({query.shape[1]}) to be divisible by "
            f"key/value heads ({key.shape[1]})."
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            "MagiAttention requires query, key, and value on the same device, "
            f"got {query.device}, {key.device}, and {value.device}."
        )


__all__ = [
    "MagiAttentionMask",
    "create_magi_mask",
    "magi_attention_forward",
    "register_veomni_magi_attention_mask_builder",
]
