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

"""SP-aware FlexAttention mask builder."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    and_masks,
    bidirectional_mask_function,
    causal_mask_function,
    sliding_window_overlay,
)

from veomni.utils.device import IS_NPU_AVAILABLE

from ..ulysses import effective_sequence_lengths, should_apply_ulysses
from .packed import packed_mask_function


def _flex_block_mask_compile_enabled() -> bool:
    """HF compiles ``create_block_mask`` on torch>=2.6.

    That inductor path needs Triton. ``torch_npu`` also intercepts inductor and
    still imports Triton, which NPU CI does not ship.
    """
    if IS_NPU_AVAILABLE:
        return False
    return importlib.util.find_spec("triton") is not None


@contextmanager
def _flex_create_block_mask(*, compile_block_mask: bool) -> Iterator[None]:
    if compile_block_mask:
        yield
        return

    import transformers.masking_utils as masking_utils

    original = masking_utils.create_block_mask

    def create_block_mask_eager(*args, **kwargs):
        kwargs["_compile"] = False
        return original(*args, **kwargs)

    masking_utils.create_block_mask = create_block_mask_eager
    try:
        yield
    finally:
        masking_utils.create_block_mask = original


def flex_attention_mask_builder(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    skip_ulysses: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    compile_block_mask: bool | None = None,
    **kwargs,
) -> BlockMask:
    """Build a Transformers FlexAttention mask.

    Expand local lengths to the Ulysses-global sequence only when the
    adapter would gather Q/K/V itself: ``ulysses_size > 1`` and not
    ``skip_ulysses``. Callers that already gathered pass
    ``skip_ulysses=True``, so the lengths already passed in are the
    effective/global ones and the mask is built at that scale. Explicit
    cumulative lengths are sufficient packed-sequence metadata and do not
    require an additional 2D mask.
    Canonical masks can be rebuilt from global lengths; custom predicates
    require global metadata.
    """
    sliding_window = kwargs.pop("sliding_window", None)
    if cu_seqlens_k is None:
        cu_seqlens_k = kwargs.pop("cu_seq_lens_k", None)
    device = kwargs.get("device", attention_mask.device if attention_mask is not None else "cpu")
    if should_apply_ulysses(skip_ulysses=skip_ulysses):
        if q_offset != 0 or kv_offset != 0:
            raise ValueError("FlexAttention with Ulysses does not support cached mask offsets.")
        if (
            attention_mask is None
            and cu_seqlens is None
            and mask_function not in (causal_mask_function, bidirectional_mask_function)
        ):
            raise ValueError(
                "FlexAttention with Ulysses requires full-sequence metadata for a custom mask function; "
                "pass a 2D attention mask or cu_seqlens."
            )
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("FlexAttention with Ulysses requires a full-sequence 2D attention mask.")

        full_q_length, full_kv_length = effective_sequence_lengths(
            q_length,
            kv_length,
            skip_ulysses=skip_ulysses,
        )
        if attention_mask is not None and attention_mask.shape[-1] != full_kv_length:
            raise ValueError(
                "FlexAttention with Ulysses requires the full attention-mask sequence length to equal "
                f"the post-Ulysses key length, got attention_mask.shape[-1]={attention_mask.shape[-1]} "
                f"and expected {full_kv_length}."
            )
        q_length, kv_length = full_q_length, full_kv_length
        q_offset = kv_offset = 0

    if sliding_window is not None:
        mask_function = and_masks(mask_function, sliding_window_overlay(sliding_window))
    if cu_seqlens is not None:
        mask_function = packed_mask_function(
            mask_function=mask_function,
            q_length=q_length,
            kv_length=kv_length,
            q_offset=q_offset,
            kv_offset=kv_offset,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            device=device,
        )

    if compile_block_mask is None:
        compile_block_mask = _flex_block_mask_compile_enabled()
    with _flex_create_block_mask(compile_block_mask=compile_block_mask):
        return ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"](
            batch_size=batch_size,
            q_length=q_length,
            kv_length=kv_length,
            q_offset=q_offset,
            kv_offset=kv_offset,
            mask_function=mask_function,
            attention_mask=attention_mask,
            **kwargs,
        )
