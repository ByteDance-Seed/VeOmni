# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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

"""Ascend NPU optimised rmsnorm gated kernel, used as the
``npu`` backend for the ``rms_norm_gated`` op in the kernel registry.
"""

import torch
import torch.nn as nn
import torch_npu
from transformers.activations import ACT2FN


# ``npu_swiglu`` hard-codes a SiLU gate, so it can only stand in for the gate
# activation when the model asks for SiLU. Qwen3.5 does (``hidden_act``), but
# Qwen4-Exp sets ``output_gate_type="sigmoid"`` — fusing there would silently
# compute the wrong thing.
_FUSED_GATE_ACTIVATIONS = frozenset({"silu", "swish"})


class NPUFusedRMSNormGated(nn.Module):
    """Ascend NPU RMSNorm with an activation-gated output projection."""

    def __init__(self, hidden_size, eps=1e-6, activation: str = "silu", **kwargs):
        """Initialize the norm weight and gate activation."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states, gate=None):
        """Apply RMSNorm to ``hidden_states`` and multiply by the activated ``gate``."""
        normed = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        if self.activation in _FUSED_GATE_ACTIVATIONS:
            return torch_npu.npu_swiglu(torch.cat([gate, normed], dim=-1), dim=-1)

        # Still fuses the norm; only the gate falls back to the eager formula
        # (float32 activation, then cast back) that the reference modules use.
        return (normed * ACT2FN[self.activation](gate.to(torch.float32))).to(hidden_states.dtype)
