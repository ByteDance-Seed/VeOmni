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

"""Grouped block-diagonal linear blockwise-FP8 QAT implementation."""

from __future__ import annotations

from torch import Tensor

from ....qat import fp8_fake_quant_act, fp8_fake_quant_weight
from . import eager as grouped_eager


def wrapper(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    n_groups: int,
    block_size: int = 128,
) -> Tensor:
    """Run block-diagonal grouped linear with blockwise FP8 fake-quantized operands.

    ``x`` has shape ``[..., n_groups, in_per_group]`` and ``weight`` has shape
    ``[n_groups * out_per_group, in_per_group]``. An empty ``bias`` tensor
    denotes an unused bias. ``block_size`` sets the activation blocks and the
    weight tiles together, as :mod:`linear.standard.qat` documents.
    """
    if x.numel() == 0:
        return grouped_eager.wrapper(x, weight, bias, n_groups=n_groups, block_size=block_size)
    x_q = fp8_fake_quant_act(x, block_size=block_size)
    w_q = fp8_fake_quant_weight(weight, block_size=block_size)
    return grouped_eager.wrapper(x_q, w_q, bias, n_groups=n_groups, block_size=block_size)
