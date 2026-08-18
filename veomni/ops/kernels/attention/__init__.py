# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Public attention APIs and Transformers attention registration."""

from typing import Callable, Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from . import flash, flex
from .flash import (
    flash_attention_forward,
    patch_transformers_hub_kernel_loader_for_veomni,
)
from .flex import flex_attention_forward, register_veomni_flex_attention_mask_builder


_ATTENTION_FORWARD_DISPATCH: dict[str, Callable] = {
    "veomni_flex_attention_with_sp": flex_attention_forward,
    "veomni_flash_attention_2_with_sp": flash_attention_forward,
    "veomni_flash_attention_3_with_sp": flash_attention_forward,
    "veomni_flash_attention_4_with_sp": flash_attention_forward,
}


def fused_attention_forward(
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
    """Dispatch the model's resolved fused-attention implementation to its backend adapter."""
    implementation = module.config._attn_implementation
    try:
        attention_forward = _ATTENTION_FORWARD_DISPATCH[implementation]
    except KeyError as error:
        supported = ", ".join(sorted(_ATTENTION_FORWARD_DISPATCH))
        raise ValueError(
            f"Unsupported VeOmni fused attention implementation: {implementation!r}. Supported: {supported}."
        ) from error

    return attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        skip_ulysses=skip_ulysses,
        **kwargs,
    )


def apply_veomni_attention_patch():
    """Register VeOmni's fused-attention implementations."""
    patch_transformers_hub_kernel_loader_for_veomni()
    register_veomni_flex_attention_mask_builder()
    ALL_ATTENTION_FUNCTIONS.register("veomni_flex_attention_with_sp", fused_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("veomni_flash_attention_2_with_sp", fused_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("veomni_flash_attention_3_with_sp", fused_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("veomni_flash_attention_4_with_sp", fused_attention_forward)
