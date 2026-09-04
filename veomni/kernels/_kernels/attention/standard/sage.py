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

"""SageAttention adapter for ``veomni_sage_attention``.

Official ``sageattn`` is an inference-only FlashAttention stand-in. It
quantizes attention for speed and does not register backward on any SM.
Training should use a FlashAttention impl, not this adapter.
"""

from typing import Optional

import torch

from .....distributed.parallel_state import get_parallel_state
from ..ulysses import (
    prepare_ulysses_qkv,
    restore_ulysses_output,
    should_apply_ulysses,
    slice_ulysses_head_auxiliary,
)


try:
    from sageattention import sageattn
except ModuleNotFoundError:
    sageattn = None


def _requires_attention_grad(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    """True when this call is on a training graph.

    Construction cannot decide this: the same module is built once and later
    used for both ``train()`` and ``eval()``. ``nn.Module`` also starts in
    training mode. Autograd state is only known at forward.
    """
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value))


def sage_attention_forward(
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
) -> tuple[torch.Tensor, None]:
    """Run SageAttention on ``[B, H, S, D]`` and return ``[B, S, H, D]``.

    Official ``sageattn`` is flash-like: causal visibility is ``is_causal``,
    not a dense mask tensor. The public dispatcher is inference-only on every
    SM; it does not register backward. This adapter therefore refuses a
    training graph instead of returning a detached tensor. Use
    ``flash_attention_2`` / ``3`` / ``4`` when gradients are needed.
    ``skip_ulysses`` opts a call out of sync Ulysses when its tokens are not
    on the SP mesh. Async Ulysses stays outside attention.
    """
    if sageattn is None:
        raise ImportError("veomni_sage_attention requires the sageattention package.")
    if _requires_attention_grad(query, key, value):
        raise RuntimeError(
            "veomni_sage_attention is inference-only; official sageattn has no backward. "
            "Use flash_attention_2, flash_attention_3, or flash_attention_4 for training."
        )
    if attention_mask is not None:
        raise ValueError(
            "veomni_sage_attention does not take a dense attention_mask. "
            "Pass is_causal for a causal pattern, matching official sageattn."
        )
    if sliding_window is not None:
        raise ValueError("veomni_sage_attention does not support sliding_window.")
    if softcap is not None:
        raise ValueError("veomni_sage_attention does not support softcap.")
    if dropout != 0.0:
        raise ValueError(f"veomni_sage_attention does not support attention dropout, got dropout={dropout}.")
    if any(dim == 0 for tensor in (query, key, value) for dim in tensor.shape):
        raise ValueError("SageAttention does not support query/key/value tensors with zero dimensions.")

    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = bool(getattr(module, "is_causal", False))

    parallel_state = get_parallel_state()
    ulysses_enabled = should_apply_ulysses(skip_ulysses=skip_ulysses)
    if ulysses_enabled:
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

    output = sageattn(
        query,
        key,
        value,
        tensor_layout="HND",
        is_causal=is_causal,
        sm_scale=scaling,
    )
    output = output.transpose(1, 2).contiguous()

    if ulysses_enabled:
        output = restore_ulysses_output(output, group=parallel_state.ulysses_group)

    return output, None
