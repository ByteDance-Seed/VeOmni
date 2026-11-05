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

"""Standard dense linear MXFP4 QAT implementation.

The operands of ``F.linear`` are fake-quantized to MXFP4 and the GEMM itself runs
in the model dtype. That is what deployment does with an emulated MXFP4 kernel:
the quantized values are read back and multiplied in high precision, so the
training forward matches what the deployment kernel computes rather than the
accumulator width it uses.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from ....qat.torchao_mxfp4 import MXFP4_BLOCK_SIZE, mxfp4_fake_quant, replicated_local


def wrapper(x: Tensor, weight: Tensor, bias: Tensor, *, block_size: int = MXFP4_BLOCK_SIZE) -> Tensor:
    """Run ``F.linear`` with MXFP4 fake-quantized activation and weight.

    An empty ``bias`` tensor denotes an unused bias. ``block_size`` is the MX
    group width and defaults to the OCP MXFP4 width of 32.
    """
    if x.numel() == 0:
        return F.linear(x, weight, None if bias.numel() == 0 else bias)
    return F.linear(
        mxfp4_fake_quant(x, block_size),
        mxfp4_fake_quant(weight, block_size),
        None if bias.numel() == 0 else replicated_local(bias, "bias"),
    )
