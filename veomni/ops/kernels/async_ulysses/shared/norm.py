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

"""Normalized-shape coercion used by async Ulysses QKV metadata."""

from __future__ import annotations

import numbers
from typing import Any


def normalize_shape(normalized_shape: int | tuple[int, ...] | None) -> Any:
    """Coerce ``normalized_shape`` to ``torch.Size`` for QKV metadata."""
    if normalized_shape is None:
        return None
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    import torch

    return torch.Size(normalized_shape)
