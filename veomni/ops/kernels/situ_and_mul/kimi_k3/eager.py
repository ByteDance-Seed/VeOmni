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

"""Kimi K3 SituGLU eager reference (fp32 compute).

``situ_eager`` is the single definition of the SituGLU math on pre-split
(gate, up) halves: the registered ``eager`` row of the ``situ_and_mul``
op and the Kimi K3 MoE expert loops (``moe_experts/kimi_k3``) both call
it, so the single-device loop, the EP dispatch, and the op registry stay
bit-identical by construction:

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    up     = linear_beta * tanh(up / linear_beta)   (when linear_beta is set)
    y      = situ_a * up
"""

import torch
from torch import Tensor


def situ_eager(gate: Tensor, up: Tensor, beta: float = 1.0, linear_beta: float | None = None) -> Tensor:
    """Compute the SituGLU core in fp32 on pre-split (gate, up) halves.

    Returns the fp32 product ``situ_a * up``; callers cast to their storage
    dtype.
    """
    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return situ_a * up


def situ_and_mul(x: Tensor, beta: float = 1.0, linear_beta: float | None = None) -> Tensor:
    """Run SituGLU on packed ``[..., 2d]`` input (gate left half, up right).

    Registered ``eager`` implementation of the ``situ_and_mul`` op: computes
    in fp32 and casts back to the input dtype.
    """
    d = x.shape[-1] // 2
    return situ_eager(x[..., :d], x[..., d:], beta=beta, linear_beta=linear_beta).to(x.dtype)
