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

"""Runtime checkpoint tensor conversion for GLM-MoE-DSA experts."""

from typing import Dict, List, Optional, Tuple

import torch

from veomni.models_kernel.checkpoint.convert import ConvertedCheckpointTensor
from veomni.models_kernel.checkpoint.moe_map import (
    PER_EXPERT_SPLIT_TO_FUSED_PATTERN,
    convert_per_expert_fqn_mapping_to_fused,
)


_EXPERT_PATTERN = PER_EXPERT_SPLIT_TO_FUSED_PATTERN


class GlmMoeDsaCheckpointTensorConverter:
    """Stack split HF expert tensors into the fused v5 expert layout."""

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        self._stacked_buffer: Dict[str, Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(_EXPERT_PATTERN.match(name))

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        match = _EXPERT_PATTERN.match(name)
        if not match:
            return None

        prefix, expert_id_str, projection = match.groups()
        buffer_key = (prefix, projection)
        expert_buffer = self._expert_buffer.setdefault(buffer_key, {})
        expert_buffer[int(expert_id_str)] = tensor
        if len(expert_buffer) < self.num_experts:
            return None

        stacked = torch.stack([expert_buffer[expert_id] for expert_id in range(self.num_experts)])
        del self._expert_buffer[buffer_key]

        if projection == "down_proj":
            return ConvertedCheckpointTensor(f"{prefix}.experts.down_proj", stacked)

        stacked_buffer = self._stacked_buffer.setdefault(prefix, {})
        stacked_buffer[projection] = stacked
        if "gate_proj" not in stacked_buffer or "up_proj" not in stacked_buffer:
            return None

        gate = stacked_buffer.pop("gate_proj")
        up = stacked_buffer.pop("up_proj")
        if not stacked_buffer:
            del self._stacked_buffer[prefix]
        return ConvertedCheckpointTensor(f"{prefix}.experts.gate_up_proj", torch.cat([gate, up], dim=1))

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        errors: List[str] = []
        if self._expert_buffer:
            unflushed = {key: len(value) for key, value in self._expert_buffer.items()}
            errors.append(
                f"unflushed per-expert buffer (incomplete experts, expected {self.num_experts}): {unflushed}"
            )
        if self._stacked_buffer:
            unflushed = {key: list(value) for key, value in self._stacked_buffer.items()}
            errors.append(f"unflushed stacked buffer (missing gate/up pair): {unflushed}")
        if errors:
            raise RuntimeError(
                "GLM-MoE-DSA checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors)
            )
        return []


def create_glm_moe_dsa_checkpoint_tensor_converter(model):
    """Create a converter using the model's routed-expert count."""
    return GlmMoeDsaCheckpointTensorConverter(num_experts=model.config.n_routed_experts)


def convert_glm_moe_dsa_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align split HF checkpoint index keys with fused expert parameters."""
    return convert_per_expert_fqn_mapping_to_fused(fqn_to_index_mapping, _EXPERT_PATTERN)
