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
are registered on that dict by ``apply_ops_patch``.
Mask builders live under ``mask/`` and are not kernel rows.
"""

import sys
from collections.abc import Callable
from typing import Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...platform import (
    NVIDIA_SM90_PLUS,
    ROCM_GPU,
    GpuKernelRequirement,
    KernelRequirement,
    MluKernelRequirement,
    NpuKernelRequirement,
    NvidiaGpuPlatform,
)
from ...registry import register_op


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
        """Dispatch one attention call through the selected Transformers interface."""
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


_NVIDIA_SM80_PLUS = NvidiaGpuPlatform(min_cc=80)
_FA2_ROWS: tuple[tuple[KernelRequirement, tuple[str, ...]], ...] = (
    (
        GpuKernelRequirement(platforms=(_NVIDIA_SM80_PLUS, ROCM_GPU)),
        ("flash_attn",),
    ),
    (NpuKernelRequirement(), ()),
    (MluKernelRequirement(), ("flash_attn",)),
)
_FA3_ROWS: tuple[tuple[KernelRequirement, tuple[str, ...]], ...] = (
    (GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,)), ("flash_attn_interface",)),
)
_FA4_ROWS: tuple[tuple[KernelRequirement, tuple[str, ...]], ...] = (
    (GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,)), ("flash_attn.cute",)),
)
_MAGI_ROWS: tuple[tuple[KernelRequirement, tuple[str, ...]], ...] = (
    (GpuKernelRequirement(platforms=(NVIDIA_SM90_PLUS,)), ("magi_attention",)),
)
_SAGE_ROWS: tuple[tuple[KernelRequirement, tuple[str, ...]], ...] = (
    (GpuKernelRequirement(platforms=(_NVIDIA_SM80_PLUS,)), ("sageattention",)),
)
_ANY_ROW: tuple[tuple[KernelRequirement | None, tuple[str, ...]], ...] = ((None, ()),)


def _register_attention(
    impl: str,
    description: str,
    *,
    rows: tuple[tuple[KernelRequirement | None, tuple[str, ...]], ...] = _ANY_ROW,
    interface: str | None = None,
) -> None:
    """Register one public attention name with its static availability rows."""
    wrapper = lookup(interface or impl)
    for requirement, requires in rows:
        register_op(
            "attention",
            "standard",
            impl,
            description=description,
            wrapper=wrapper,
            requirement=requirement,
            requires=requires,
        )


_register_attention("eager", "Transformers model-local eager attention")
_register_attention("sdpa", "PyTorch scaled dot-product attention through Transformers")
_register_attention("flash_attention_2", "FlashAttention 2 through Transformers", rows=_FA2_ROWS)
_register_attention("flash_attention_3", "FlashAttention 3 through Transformers", rows=_FA3_ROWS)
_register_attention("flash_attention_4", "FlashAttention 4 through Transformers", rows=_FA4_ROWS)
_register_attention("flex_attention", "PyTorch FlexAttention through Transformers")
_register_attention(
    "magi_attention",
    "MagiAttention short-name alias to the VeOmni adapter",
    rows=_MAGI_ROWS,
    interface="veomni_magi_attention",
)
_register_attention("native-sparse", "Transformers native sparse attention")
_register_attention(
    "veomni_flash_attention_2",
    "VeOmni FlashAttention 2 adapter through Transformers",
    rows=_FA2_ROWS,
)
_register_attention(
    "veomni_flash_attention_3",
    "VeOmni FlashAttention 3 adapter through Transformers",
    rows=_FA3_ROWS,
)
_register_attention(
    "veomni_flash_attention_4",
    "VeOmni FlashAttention 4 adapter through Transformers",
    rows=_FA4_ROWS,
)
_register_attention("veomni_flex_attention", "VeOmni FlexAttention adapter through Transformers")
_register_attention(
    "veomni_magi_attention",
    "VeOmni MagiAttention adapter through Transformers",
    rows=_MAGI_ROWS,
)
_register_attention(
    "veomni_sage_attention",
    "VeOmni SageAttention adapter through Transformers",
    rows=_SAGE_ROWS,
)
_register_attention("veomni_sdpa", "VeOmni scaled dot-product attention adapter through Transformers")
