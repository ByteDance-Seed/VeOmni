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

"""FA4 eligibility and compiled execution; inputs use BHSD layout."""

from functools import cache
from typing import Optional

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as raw_flex_attention
from transformers.integrations.flex_attention import get_flex_attention_lse_kwargs, repeat_kv
from transformers.utils.import_utils import is_torchdynamo_compiling

from ._fa4_sm90_compat import install_fa4_sm90_aux_bounds_fix


install_fa4_sm90_aux_bounds_fix()


_FA4_DTYPES = frozenset((torch.float16, torch.bfloat16))


@cache
def _flash_attn_cute_available() -> bool:
    """Return whether PyTorch's FLASH Flex backend can import FA4."""
    try:
        import flash_attn.cute  # noqa: F401
    except ImportError:
        return False
    return True


def _dtype_supports_flex_fa4(dtype: torch.dtype) -> bool:
    return dtype in _FA4_DTYPES


def _head_dim_supports_flex_fa4(head_dim: int) -> bool:
    # FA4 cute backward preprocess ICE when hd % 32 != 0
    # (Dao-AILab/flash-attention#2492). hd % 32 == 0 skips the OOB
    # predicate and can train. TODO: drop this gate after upgrading
    # past flash-attn 4.0.0b16 once #2518 / #2698 land in the wheel.
    return head_dim % 32 == 0


def _sparse_head_dims_support_flex_fa4(capability: int, head_dim: int | None, value_head_dim: int | None) -> bool:
    """Conservative training eligibility for the pinned FA4 sparse kernels."""
    if capability not in (90, 100, 110):
        return False
    value_head_dim = head_dim if value_head_dim is None else value_head_dim
    if head_dim is None or value_head_dim is None:
        return True
    # Larger dimensions change the native tile sizes or use dense-only kernels;
    # neither matches the general Flex sparse training contract here.
    return all(0 < dim <= 128 and dim % 32 == 0 for dim in (head_dim, value_head_dim))


@cache
def _compiled_flex_attention_fa4():
    """Compile fixed-shape FLASH calls; raise instead of falling back to dense eager attention."""
    return torch.compile(raw_flex_attention, dynamic=False, fullgraph=True)


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

    # b16 forward omits broadcast strides from its native compile key.
    # Materialize only broadcast views so different calls cannot alias
    # incompatible zero-stride specializations through the shared cache.
    query, key, value = (
        tensor.clone(memory_format=torch.contiguous_format) if 0 in tensor.stride() else tensor
        for tensor in (query, key, value)
    )

    if (query.shape[-2], key.shape[-2]) != attention_mask.seq_lengths:
        raise ValueError("Flex-FA4 requires BlockMask sequence lengths to match query/key lengths exactly.")
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
