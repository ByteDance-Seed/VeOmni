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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flex adapter entry points, backend routing, and Ulysses coordination."""

from collections.abc import Callable
from typing import Optional

import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.integrations.flex_attention import flex_attention_forward as _flex_attention_triton

from ......distributed.parallel_state import get_parallel_state
from ......utils.device import get_gpu_compute_capability, is_nvidia_cuda_device
from ...mask.flex import _flex_block_size
from ...ulysses import (
    prepare_ulysses_qkv,
    restore_ulysses_output,
    should_apply_ulysses,
    slice_ulysses_head_auxiliary,
)
from . import _fa4
from ._fa4 import _flex_attention_fa4
from ._inputs import _maybe_recast_qkv_for_fa4, _normalize_flex_qkv


FLEX_BACKEND_FLASH = "FLASH"
FLEX_BACKEND_TRITON = "TRITON"
_HOPPER_MIN_CC = 90


def _normalized_block_size(block_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize scalar and rectangular BlockMask tile sizes for backend selection."""
    return (block_size, block_size) if isinstance(block_size, int) else block_size


def resolve_flex_attention(
    device: torch.device,
    kernel_options: dict,
    *,
    attention_sinks: bool = False,
    query_dtype: torch.dtype | None = None,
    head_dim: int | None = None,
    value_head_dim: int | None = None,
    attention_mask: BlockMask | None = None,
    trainable_position_bias: bool = False,
) -> Callable:
    """Select the FlexAttention implementation for this call.

    The registry row stays GPU-agnostic. NVIDIA SM90 and newer use the FLASH
    adapter (CuteDSL wrapping FA4). SM80, CPU, and other devices stay on the
    Transformers Triton adapter. An explicit ``kernel_options`` backend wins.
    Attention sinks need LSE renormalization, and FLASH backward rejects dLSE,
    so sinks fall back to Triton unless FLASH was forced. FA4 also rejects
    fp32, so a remaining fp32 query stays on Triton unless FLASH was forced.
    Unsupported sparse head dimensions, incomplete BlockMask tables,
    architecture-mismatched mask tiles, and trainable position bias also
    retain the Triton path.
    """
    backend = kernel_options.get("BACKEND")
    if backend is not None and str(backend).upper() == FLEX_BACKEND_FLASH and trainable_position_bias:
        raise ValueError("FlexAttention FLASH backend does not support trainable position_bias; use TRITON.")
    if backend is not None:
        backend = str(backend).upper()
        if backend == FLEX_BACKEND_FLASH and attention_sinks:
            raise ValueError(
                "FlexAttention FLASH backend does not support attention sinks "
                "(s_aux). FLASH backward rejects dLSE, and sinks need LSE "
                "renormalization. Omit BACKEND to fall back to Triton, or drop s_aux."
            )
    elif attention_sinks or trainable_position_bias or not is_nvidia_cuda_device(device):
        backend = FLEX_BACKEND_TRITON
    elif query_dtype is not None and not _fa4._dtype_supports_flex_fa4(query_dtype):
        backend = FLEX_BACKEND_TRITON
    elif head_dim is not None and not _fa4._head_dim_supports_flex_fa4(head_dim):
        backend = FLEX_BACKEND_TRITON
    elif attention_mask is not None and any(
        getattr(attention_mask, name) is None
        for name in ("full_kv_num_blocks", "full_kv_indices", "full_q_num_blocks", "full_q_indices")
    ):
        backend = FLEX_BACKEND_TRITON
    elif attention_mask is not None and _normalized_block_size(attention_mask.BLOCK_SIZE) != _normalized_block_size(
        _flex_block_size(device)
    ):
        backend = FLEX_BACKEND_TRITON
    elif not _fa4._sparse_head_dims_support_flex_fa4(get_gpu_compute_capability(device), head_dim, value_head_dim):
        backend = FLEX_BACKEND_TRITON
    elif get_gpu_compute_capability(device) >= _HOPPER_MIN_CC and _fa4._flash_attn_cute_available():
        backend = FLEX_BACKEND_FLASH
    else:
        backend = FLEX_BACKEND_TRITON

    kernel_options["BACKEND"] = backend
    if backend == FLEX_BACKEND_FLASH:
        return _flex_attention_fa4
    return _flex_attention_triton


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the Transformers FlexAttention adapter with optional Ulysses exchange.

    Inputs are BHSD (4D) or THD (3D); outputs are BSHD or THD, respectively.
    Token-major inputs describe a single token stream, with packing in the mask.
    ``skip_ulysses`` opts a call out of sync Ulysses when its tokens are not
    on the SP mesh. Async Ulysses stays outside attention.

    The second result is optional LSE, never attention weights. Requests for
    ``output_attentions=True`` fail before distributed work or backend dispatch.
    """
    if kwargs.get("output_attentions", False):
        raise ValueError("FlexAttention does not support `output_attentions=True`; use eager attention.")
    if not isinstance(attention_mask, BlockMask):
        raise TypeError(f"FlexAttention requires a BlockMask, got {type(attention_mask).__name__}.")

    query, key, value, token_major = _normalize_flex_qkv(query, key, value)

    # Transformers models may pass ``sliding_window`` metadata together with a
    # BlockMask that already encodes the window predicate. The BlockMask remains
    # the sole source of visibility semantics; do not reconstruct or modify it
    # from the integer metadata.
    del sliding_window

    kernel_options = dict(kwargs.pop("kernel_options", {}) or {})
    query, key, value = _maybe_recast_qkv_for_fa4(module, query, key, value)

    parallel_state = get_parallel_state()
    ulysses_enabled = should_apply_ulysses(skip_ulysses=skip_ulysses)
    if ulysses_enabled:
        # Local head indices restart at zero on every Ulysses rank, so head-specific
        # masks require rank-aware slicing and rebasing before they can be supported.
        if attention_mask.shape[1] != 1:
            raise ValueError("FlexAttention with Ulysses requires a head-broadcast BlockMask.")

        query, key, value, query_head_count = prepare_ulysses_qkv(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if "s_aux" in kwargs:
            kwargs["s_aux"] = slice_ulysses_head_auxiliary(
                kwargs["s_aux"],
                query_head_count=query_head_count,
                local_query_head_count=query.shape[1],
                group=parallel_state.ulysses_group,
            )

    flex_attention = resolve_flex_attention(
        query.device,
        kernel_options,
        attention_sinks=kwargs.get("s_aux") is not None,
        query_dtype=query.dtype,
        head_dim=query.shape[-1],
        value_head_dim=value.shape[-1],
        attention_mask=attention_mask,
        trainable_position_bias=(kwargs.get("position_bias") is not None and kwargs["position_bias"].requires_grad),
    )
    output, lse = flex_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        softcap=softcap,
        kernel_options=kernel_options,
        **kwargs,
    )

    if ulysses_enabled:
        output = restore_ulysses_output(output, group=parallel_state.ulysses_group)
        if lse is not None:
            lse = restore_ulysses_output(
                lse.transpose(1, 2).unsqueeze(-1),
                group=parallel_state.ulysses_group,
            ).squeeze(-1)
            lse = lse.transpose(1, 2).contiguous()

    if token_major:
        output = output.squeeze(0)
        if lse is not None:
            lse = lse.squeeze(0)
    return output, lse
