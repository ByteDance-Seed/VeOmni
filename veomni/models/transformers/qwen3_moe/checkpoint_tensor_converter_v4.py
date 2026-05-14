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

"""
Runtime per-expert -> stacked converter for transformers v4 Qwen3-MoE.

The v4 patch (`PatchQwen3MoeExperts` in modeling_qwen3_moe.py) stores expert
weights as three separate 3-D nn.Parameters in both eager and fused modes;
HuggingFace ships per-expert keys. This converter stacks them at load time.
Mirrors v5's `Qwen3MoeCheckpointTensorConverter` minus the gate/up cat step.
"""

import re

from .._moe_v4_converter import MoEV4StackingConverter


_EXPERT_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def create_qwen3_moe_v4_checkpoint_tensor_converter(model):
    """Factory registered on v4 Qwen3-MoE model classes via `_create_checkpoint_tensor_converter`."""
    num_experts = model.config.num_experts
    return MoEV4StackingConverter(
        pattern=_EXPERT_PATTERN,
        num_experts_for=lambda _prefix: num_experts,
    )
