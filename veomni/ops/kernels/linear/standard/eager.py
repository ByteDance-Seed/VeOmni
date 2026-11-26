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

"""Standard dense linear eager implementation."""

from __future__ import annotations

from torch import Tensor, nn


def wrapper(linear: nn.Module, x: Tensor, *, block_size: int = 128) -> Tensor:
    """Run ``linear`` in the model dtype; ``block_size`` is unused here.

    The family-wide keyword is part of the shared signature so a model can call any
    row the same way; this reference does not quantize.

    The module is *called* rather than driven through ``F.linear(x, weight)`` on
    purpose. Reading the weight directly would skip the module's own hooks, and a
    projection listed in ``basic_modules`` is ``fully_shard``-wrapped on its own --
    its unshard hook runs in ``forward``, so bypassing it hands this row a sharded
    ``DTensor`` weight. Calling the module also covers adapter-wrapped projections
    without a special case, which is why this row needs no ``resolve_projection``.
    """
    del block_size
    if getattr(linear, "is_quantized", False) or getattr(linear, "weight_scale", None) is not None:
        raise ValueError(
            f"linear_implementation='eager' cannot run on pre-quantized projection "
            f"{type(linear).__name__!r} (holds `weight_scale`). Clear `weight_quantization` "
            "to dequantize weights at load for training."
        )
    if getattr(linear, "_veomni_self_dispatched", False):
        return linear._eager_forward(x)
    return linear(x)
