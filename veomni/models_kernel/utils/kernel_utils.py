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

"""Helpers for modeling consume of ``VeomniKernel``.

Read impl names from ``get_kernels_config`` at construct time. Do not bind
``OpSlot``. ``npu`` on cross-entropy maps to ``chunk_loss``.
"""

from __future__ import annotations

from torch import Tensor, nn

from veomni.kernels.config import get_kernels_config


def resolve_kernel_impl(field: str, *, npu_as: str | None = None) -> str:
    """Return the impl name on the installed kernel config, or ``eager``.

    ``npu_as`` remaps the legacy ``npu`` CE alias. Missing config is eager so
    unit tests can construct a module without ``set_kernels_config``.
    """
    cfg = get_kernels_config()
    impl = "eager" if cfg is None else getattr(cfg, field)
    if npu_as is not None and impl == "npu":
        return npu_as
    return impl


def empty_bias(weight: Tensor) -> Tensor:
    """Empty unused-layout bias for a Linear that has ``bias=None``."""
    return weight.new_empty(0)


def linear_bias(linear: nn.Linear) -> Tensor:
    """Return the Linear bias, or the empty unused-layout sentinel."""
    return linear.bias if linear.bias is not None else empty_bias(linear.weight)
