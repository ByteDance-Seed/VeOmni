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
Runtime per-expert -> stacked converter for transformers v4 DeepSeek V3.

The v4 patch (`PatchDeepseekV3NaiveMoe`) stores expert weights as three 3-D
nn.Parameters; HF ships per-expert keys. DeepSeek V3 differs from Qwen3-MoE
in two ways:
- Number of experts is `config.n_routed_experts` (not `num_experts`).
- The first `config.first_k_dense_replace` layers are dense MLPs with no
  `experts.*` keys; the regex naturally won't match those layers.
"""

import re

from .._moe_v4_converter import MoEV4StackingConverter


_EXPERT_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def create_deepseek_v3_v4_checkpoint_tensor_converter(model):
    """Factory registered on v4 DeepSeek V3 model classes via `_create_checkpoint_tensor_converter`."""
    num_experts = model.config.n_routed_experts
    return MoEV4StackingConverter(
        pattern=_EXPERT_PATTERN,
        num_experts_for=lambda _prefix: num_experts,
    )
