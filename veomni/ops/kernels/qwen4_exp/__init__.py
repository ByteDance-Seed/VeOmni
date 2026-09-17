# Copyright 2026 ByteDance Ltd. and/or its affiliates
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

"""Qwen4-Exp model-specific SM90+ GPU kernels.

Imports stay inside the public wrappers because TileLang is an optional,
GPU-only dependency. Importing VeOmni on CPU or NPU must not load it.
"""

import torch

from ....utils.device import IS_CUDA_AVAILABLE


def qsa_attn_tilelang(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_indices: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        raise RuntimeError("Qwen4-Exp TileLang kernels require an NVIDIA CUDA GPU")

    from .tilelang_qsa import qsa_attn_tilelang as impl

    return impl(query, key, value, selected_indices, sm_scale)


__all__ = ["qsa_attn_tilelang"]
