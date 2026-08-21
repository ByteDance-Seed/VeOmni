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
    # Capability marker only. The Qwen GDN patch opts an *instance* into the
    # grouped nested FSDP2 leaf exclusively when KCP is selected. Keeping the
    # collector marker off the class preserves CP-off and state-passing FSDP
    # behavior. Do not replace the KCP leaf with ``to_local()``: that would feed
    # the fused op a rank-local shard instead of the complete RMSNorm weight.
    _veomni_kcp_requires_grouped_nested_fsdp_leaf = True

    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        hidden_states = torch.cat([gate, hidden_states], dim=-1)
        hidden_states = torch_npu.npu_swiglu(hidden_states, dim=-1)

        return hidden_states
