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
)


sageattn = None

_UNSUPPORTED_ATTENTION_METADATA = (
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "cu_seqlens",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "max_length_q",
    "max_length_k",
    "indices",
    "s_aux",
)


def _load_sageattn():
    """Load the optional SageAttention callable only when this adapter runs."""
    if sageattn is not None:
        return sageattn
    try:
        from sageattention import sageattn as imported_sageattn
    except ModuleNotFoundError as exc:
        if exc.name != "sageattention":
            raise
        raise ImportError("veomni_sage_attention requires the sageattention package.") from exc
    return imported_sageattn


def _requires_attention_grad(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    """True when this call is on a training graph.

    Construction cannot decide this: the same module is built once and later
    used for both ``train()`` and ``eval()``. ``nn.Module`` also starts in
    training mode. Autograd state is only known at forward.
    """
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value))


def _reject_unsupported_attention_metadata(kwargs: dict) -> None:
    """Reject metadata whose semantics SageAttention cannot represent."""
    unsupported = tuple(name for name in _UNSUPPORTED_ATTENTION_METADATA if kwargs.get(name) is not None)
    if unsupported:
        names = ", ".join(f"`{name}`" for name in unsupported)
        raise ValueError(
            "veomni_sage_attention does not support packed, sparse, or auxiliary attention metadata: " + names
        )


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
    Packed, sparse, and auxiliary attention metadata is rejected because the
    SageAttention call below cannot represent those visibility semantics.
    ``skip_ulysses`` opts a call out of sync Ulysses when its tokens are not
    on the SP mesh. Async Ulysses stays outside attention.
    """
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

    _reject_unsupported_attention_metadata(kwargs)

    sageattn_fn = _load_sageattn()

    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = bool(getattr(module, "is_causal", False))

    parallel_state = get_parallel_state()
    ulysses_enabled = should_apply_ulysses(skip_ulysses=skip_ulysses)
    if ulysses_enabled:
        query, key, value, _ = prepare_ulysses_qkv(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

    output = sageattn_fn(
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
