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

"""Ascend NPU optimised RMSNorm forward, used as the ``npu`` backend for the
``rms_norm`` op in the kernel registry.
"""

import torch_npu


def rms_norm_npu(hidden_states, weight, eps):
    """NPU optimized RMSNorm implementation for OpSlot dispatch."""
    if hidden_states.dtype != weight.dtype:
        hidden_states = hidden_states.to(weight.dtype)
    return torch_npu.npu_rms_norm(hidden_states, weight, epsilon=eps)[0]


def rms_norm_forward_npu(self, x):
    """NPU optimized implementation for RMSNorm."""
    return rms_norm_npu(x, self.weight, self.variance_epsilon)


__all__ = ["rms_norm_forward_npu", "rms_norm_npu"]
