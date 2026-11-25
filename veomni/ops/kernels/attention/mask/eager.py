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

"""HF-signature additive mask builder for ``veomni_eager``."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from transformers.masking_utils import causal_mask_function

from ..helper import strip_sdpa_packed_metadata
from .sdpa import _dense_attention_mask_builder


def _to_eager_additive(mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Convert a boolean visibility mask to additive 0 / ``finfo.min`` form."""
    if mask is None:
        return None
    if mask.dtype.is_floating_point:
        return mask
    if mask.dtype != torch.bool:
        mask = mask != 0
    min_dtype = torch.finfo(dtype).min
    return torch.where(mask, torch.zeros((), device=mask.device, dtype=dtype), min_dtype)


def eager_attention_mask_builder(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    skip_ulysses: bool = False,
    **kwargs,
) -> Tensor | None:
    """HF-signature additive mask builder for ``veomni_eager``.

    Single-segment packed/varlen metadata is dropped. Several segments per row
    raise; build them with ``packed_causal_mask``. Causal skip is always
    disabled so eager receives an explicit additive mask, while bidirectional
    elision remains available when explicitly requested.
    """
    kwargs = strip_sdpa_packed_metadata(kwargs, batch_size=batch_size, dense_mask=None)
    kwargs.pop("allow_is_causal_skip", None)
    mask = _dense_attention_mask_builder(
        batch_size,
        q_length,
        kv_length,
        q_offset,
        kv_offset,
        mask_function,
        attention_mask,
        skip_ulysses=skip_ulysses,
        allow_is_causal_skip=False,
        **kwargs,
    )
    return _to_eager_additive(mask, dtype)
