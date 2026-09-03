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
are registered on that dict by ``apply_kernel_patch``.
Mask builders live under ``mask/`` and are not kernel rows.
"""

import sys
from collections.abc import Callable
from typing import Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...registry import register_kernel


def _module_eager_forward(module: torch.nn.Module) -> Callable | None:
    """HF does not register ``eager`` on ``ALL_ATTENTION_FUNCTIONS``.

    Each modeling file keeps a local ``eager_attention_forward`` (Gemma3
    softcap, GPT-OSS sinks, ...). Resolve that from the Attention class
    module so ``get_interface("eager", default)`` matches HF consume.
    """
    defining = sys.modules.get(type(module).__module__)
    if defining is None:
        return None
    eager = getattr(defining, "eager_attention_forward", None)
    return eager if callable(eager) else None


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
        eager_default = _module_eager_forward(module)
        forward = ALL_ATTENTION_FUNCTIONS.get_interface(impl, eager_default)
        if forward is None:
            raise KeyError(
                f"attention impl {impl!r} is not registered and "
                f"{type(module).__module__} has no eager_attention_forward"
            )

        return forward(
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
    "veomni_sage_attention",
    "veomni_sdpa",
)

for _impl in _STANDARD_IMPLS:
    register_kernel("attention", "standard", _impl, wrapper=lookup(_impl))
