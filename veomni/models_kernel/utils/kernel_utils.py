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
# See the License for the specific language governing limitations
# under the License.

"""Helpers for constructing ``VeomniKernel`` handles.

Read impl names from ``get_kernels_config`` at construct time. ``npu`` on
cross-entropy maps to ``chunk_loss``.
"""

from __future__ import annotations

from torch import Tensor, nn

from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config


def resolve_kernel_impl(field: str, *, npu_as: str | None = None) -> str:
    """Return the impl name on the installed kernel config, or ``eager``.

    ``npu_as`` remaps the ``npu`` CE name to ``chunk_loss``. Missing config
    is eager so unit tests can construct a module without ``set_kernels_config``.
    """
    cfg = get_kernels_config()
    impl = "eager" if cfg is None else getattr(cfg, field, "eager")
    if npu_as is not None and impl == "npu":
        return npu_as
    return impl


def attention_kernel() -> VeomniKernel:
    """Return the interned standard-attention kernel for the active impl.

    Missing kernels config resolves to ``eager``. Construct in ``__init__``
    when the attention module is already patched; otherwise call this in
    ``forward``. The handle is interned either way.
    """
    return VeomniKernel("attention", "standard", resolve_kernel_impl("attn_implementation"))


def resolve_moe_impl() -> str:
    """Map ``moe_implementation`` onto a ``moe_experts`` registry impl.

    Config may use ``fused_triton`` / ``fused_quack`` / ``fused_npu``.
    Registry rows are ``triton`` / ``quack`` / ``npu``.
    """
    impl = resolve_kernel_impl("moe_implementation")
    if impl.startswith("fused_"):
        return impl.removeprefix("fused_")
    return impl


def empty_bias(weight: Tensor) -> Tensor:
    """Empty unused-layout bias for a Linear that has ``bias=None``."""
    return weight.new_empty(0)


def linear_bias(linear: nn.Linear) -> Tensor:
    """Return the Linear bias, or the empty unused-layout sentinel."""
    return linear.bias if linear.bias is not None else empty_bias(linear.weight)
