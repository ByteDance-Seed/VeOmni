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

"""Attention family.

Triplet rows forward to ``ALL_ATTENTION_FUNCTIONS``. ``veomni_*`` adapters
are registered on that dict by ``install.apply_veomni_attention_patch``.
Mask builders live under ``mask/`` and are not kernel rows.
"""

from collections.abc import Callable
from typing import Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...registry import register_kernel
from .install import apply_veomni_attention_patch


def lookup(impl: str) -> Callable:
    """Return a wrapper that dispatches ``impl`` through the HF attention dict."""

    def wrapper(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        softcap: Optional[float] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return ALL_ATTENTION_FUNCTIONS[impl](
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            **kwargs,
        )

    wrapper.__name__ = f"attention_{impl}"
    wrapper.__qualname__ = wrapper.__name__
    return wrapper


_STANDARD_IMPLS = (
    "eager",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "flex_attention",
    "magi_attention",
    "native-sparse",
    "veomni_flash_attention_2",
    "veomni_flash_attention_3",
    "veomni_flash_attention_4",
    "veomni_flex_attention",
    "veomni_magi_attention",
    "veomni_sdpa",
)

for _impl in _STANDARD_IMPLS:
    register_kernel("attention", "standard", _impl, wrapper=lookup(_impl))

apply_veomni_attention_patch()
