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

"""First-party helpers for gated-delta-rule kernels."""

from __future__ import annotations

import torch
from torch import Tensor


def optional_tensor(tensor: Tensor | None) -> Tensor | None:
    """Return ``None`` when *tensor* is missing or the empty unused sentinel."""
    return None if tensor is None or tensor.numel() == 0 else tensor


def unused_like(tensor: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Empty unused-layout sentinel on *tensor*'s device."""
    return tensor.new_empty(0) if dtype is None else tensor.new_empty(0, dtype=dtype)
