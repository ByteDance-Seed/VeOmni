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


class NPUFusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, activation="silu", device=None, dtype=None, **kwargs):
        super().__init__()
        if activation not in {"silu", "sigmoid"}:
            raise ValueError(f"Unsupported NPU gated RMSNorm activation: {activation!r}")
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states, gate=None):
        if gate is None or gate.shape != hidden_states.shape:
            raise ValueError("NPU gated RMSNorm requires a gate with the input shape")
        if self.activation == "sigmoid":
            # Match Qwen4-Exp's rounding boundary: normalize, cast to the input
            # dtype, then apply the learned weight. Fusing the learned weight
            # into RMSNorm would move that boundary for BF16 compute weights.
            unit_weight = torch.ones(hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
            normalized = torch_npu.npu_rms_norm(hidden_states, unit_weight, self.variance_epsilon)[0]
            weighted = self.weight * normalized.to(hidden_states.dtype)
            return (weighted * gate.float().sigmoid()).to(hidden_states.dtype)
        # Preserve the existing SiLU route for models such as Qwen3.6.
        normalized = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return torch_npu.npu_swiglu(torch.cat([gate, normalized], dim=-1), dim=-1)
