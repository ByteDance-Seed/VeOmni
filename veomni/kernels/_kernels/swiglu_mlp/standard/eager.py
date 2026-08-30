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

"""standard SwiGLU eager math (``silu(gate) * up``)."""

from __future__ import annotations

import torch
from torch import Tensor

from ....registry import SavedState


def forward(gate: Tensor, up: Tensor) -> tuple[Tensor, SavedState]:
    """Elementwise SwiGLU activation used by HF MLP bodies.

    Matches ``act_fn(gate) * up`` with ``hidden_act="silu"``. Gate / up / down
    projections stay in the module, same as the ops Liger factory.
    """
    return torch.nn.functional.silu(gate) * up, SavedState((gate, up))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, Tensor]:
    """Return ``(grad_gate, grad_up)`` matching the positional tensors."""
    gate, up = saved.tensors
    sig = torch.sigmoid(gate)
    silu_gate = gate * sig
    grad_up = grad_output * silu_gate
    grad_gate = grad_output * up * (silu_gate * (1 - sig) + sig)
    return grad_gate, grad_up
