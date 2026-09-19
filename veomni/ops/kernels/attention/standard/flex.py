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

"""FlexAttention backend and SP-aware adapter implementation."""

from collections.abc import Callable
from functools import cache
from typing import Optional

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as raw_flex_attention
from transformers.integrations.flex_attention import (
    flex_attention_forward as _flex_attention_triton,
)
from transformers.integrations.flex_attention import get_flex_attention_lse_kwargs, repeat_kv
from transformers.utils.import_utils import is_torchdynamo_compiling

from .....distributed.parallel_state import get_parallel_state
from .....utils.device import get_gpu_compute_capability
from ..ulysses import (
    prepare_ulysses_qkv,
    restore_ulysses_output,
    should_apply_ulysses,
    slice_ulysses_head_auxiliary,
)


FLEX_BACKEND_FLASH = "FLASH"
FLEX_BACKEND_TRITON = "TRITON"
_HOPPER_MIN_CC = 90
_FA4_DTYPES = frozenset(
    dtype
    for dtype in (
        torch.float16,
        torch.bfloat16,
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
)


@cache
def _flash_attn_cute_available() -> bool:
    """Return whether PyTorch's FLASH Flex backend can import FA4."""
    try:
        import flash_attn.cute  # noqa: F401
    except ImportError:
        return False
    return True


@cache
def _compiled_flex_attention_fa4():
    """Compile FlexAttention once with static shapes for the FA4 path.

    Hugging Face's shared ``WrappedFlexAttention`` uses the default dynamic
    compile. FLASH cannot inline symbolic BlockMask scalars into CuteDSL.
    """
    return torch.compile(raw_flex_attention, dynamic=False)


def _flex_attention_fa4(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: BlockMask,
    *,
    dropout: float,
    scaling: Optional[float],
    softcap: Optional[float],
    kernel_options: dict,
    position_bias: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run compiled FlexAttention without requesting LSE.

    FLASH backward does not support dLSE. Keep this path free of
    ``return_lse`` / ``AuxRequest(lse=True)`` so SM90 training can use FA4.
    """
    del module, kwargs
    if dropout > 0:
        raise ValueError(
            "`flex_attention` does not support `dropout`. Please use it with inference"
            " only (`model.eval()`) or turn off the attention dropout in the respective config."
        )

    score_mod = None
    if softcap is not None or position_bias is not None:

        def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
            if softcap is not None:
                score = softcap * torch.tanh(score / softcap)
            if position_bias is not None:
                score = score + position_bias[batch_idx, head_idx, q_idx, kv_idx]
            return score

    enable_gqa = True
    num_local_query_heads = query.shape[1]
    if (num_local_query_heads & (num_local_query_heads - 1)) != 0:
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    flex_fn = raw_flex_attention if is_torchdynamo_compiling() else _compiled_flex_attention_fa4()
    attention_output = flex_fn(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=attention_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,
        **get_flex_attention_lse_kwargs(False),
    )
    return attention_output.transpose(1, 2).contiguous(), None


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


def _dtype_supports_flex_fa4(dtype: torch.dtype) -> bool:
    return dtype in _FA4_DTYPES


def _head_dim_supports_flex_fa4(head_dim: int) -> bool:
    # FA4 cute backward preprocess ICE when hd % 32 != 0
    # (Dao-AILab/flash-attention#2492). hd % 32 == 0 skips the OOB
    # predicate and can train. TODO: drop this gate after upgrading
    # past flash-attn 4.0.0b16 once #2518 / #2698 land in the wheel.
    return head_dim % 32 == 0


def resolve_flex_attention(
    device: torch.device,
    kernel_options: dict,
    *,
    attention_sinks: bool = False,
    query_dtype: torch.dtype | None = None,
    head_dim: int | None = None,
) -> Callable:
    """Select the FlexAttention implementation for this call.

    The registry row stays GPU-agnostic. NVIDIA SM90 and newer use the FLASH
    adapter (CuteDSL wrapping FA4). SM80, CPU, and other devices stay on the
    Transformers Triton adapter. An explicit ``kernel_options`` backend wins.
    Attention sinks need LSE renormalization, and FLASH backward rejects dLSE,
    so sinks fall back to Triton unless FLASH was forced. FA4 also rejects
    fp32, so a remaining fp32 query stays on Triton unless FLASH was forced.
    Head dims that are not a multiple of 32 stay on Triton for the same
    reason: see ``_head_dim_supports_flex_fa4``.
    """
    backend = kernel_options.get("BACKEND")
    if backend is not None:
        backend = str(backend).upper()
        if backend == FLEX_BACKEND_FLASH and attention_sinks:
            raise ValueError(
                "FlexAttention FLASH backend does not support attention sinks "
                "(s_aux). FLASH backward rejects dLSE, and sinks need LSE "
                "renormalization. Omit BACKEND to fall back to Triton, or drop s_aux."
            )
    elif attention_sinks or device.type != "cuda":
        backend = FLEX_BACKEND_TRITON
    elif query_dtype is not None and not _dtype_supports_flex_fa4(query_dtype):
        backend = FLEX_BACKEND_TRITON
    elif head_dim is not None and not _head_dim_supports_flex_fa4(head_dim):
        backend = FLEX_BACKEND_TRITON
    elif get_gpu_compute_capability(device) >= _HOPPER_MIN_CC and _flash_attn_cute_available():
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

    ``skip_ulysses`` opts a call out of sync Ulysses when its tokens are not
    on the SP mesh. Async Ulysses stays outside attention.
    """
    if not isinstance(attention_mask, BlockMask):
        raise TypeError(f"FlexAttention requires a BlockMask, got {type(attention_mask).__name__}.")

    if any(dim == 0 for tensor in (query, key, value) for dim in tensor.shape):
        raise ValueError("FlexAttention does not support query/key/value tensors with zero dimensions.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            f"FlexAttention GQA requires query heads ({query.shape[1]}) to be divisible by "
            f"key/value heads ({key.shape[1]})."
        )

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

    return output, lse
