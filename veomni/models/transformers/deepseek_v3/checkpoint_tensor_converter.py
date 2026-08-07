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
Runtime checkpoint tensor converter for DeepseekV3 MoE models.

Converts HuggingFace per-expert checkpoint format to v5 fused format
at load time, eliminating the need for offline checkpoint merging.

    HF checkpoint format (per-expert):
        model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]
        model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
        model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]

    Target v5 format (direct layout, no transpose):
        model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        model.layers.{i}.mlp.experts.down_proj     [E, H, I]
"""

import re
from typing import Dict, List, Optional, Tuple

import torch

from ....utils import logging
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


logger = logging.get_logger(__name__)

# Matches per-expert split keys like: model.layers.0.mlp.experts.3.gate_proj.weight
_EXPERT_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_MTP_SHARED_PATTERN = re.compile(r"^model\.layers\.\d+\.(embed_tokens|shared_head\.(?:norm|head))\.weight$")
_MODEL_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
_MTP_SHARED_TARGETS = {
    "embed_tokens": "model.embed_tokens.weight",
    "shared_head.norm": "model.norm.weight",
    "shared_head.head": "lm_head.weight",
}


class DeepseekV3CheckpointTensorConverter:
    """Converts per-expert split checkpoint keys to stacked & merged v5 format.

    Buffers per-expert tensors as they stream from safetensor files, and emits
    merged tensors once all experts for a given (layer, projection) are collected.

    Args:
        num_experts: Number of routed experts per MoE layer
            (``config.n_routed_experts`` — equal to ``config.num_local_experts``
            in v5 DeepseekV3).
    """

    def __init__(self, num_experts: int, num_hidden_layers: Optional[int] = None, mtp_enabled: bool = True):
        self.num_experts = num_experts
        self.num_hidden_layers = num_hidden_layers
        self.mtp_enabled = mtp_enabled
        # {(prefix, proj_name): {expert_id: tensor}}
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        # {prefix: {proj_name: stacked_tensor}} for gate/up merge waiting
        self._stacked_buffer: Dict[str, Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return self._is_disabled_mtp_key(name) or bool(_EXPERT_PATTERN.match(name) or _MTP_SHARED_PATTERN.match(name))

    def _is_disabled_mtp_key(self, name: str) -> bool:
        layer_match = _MODEL_LAYER_PATTERN.match(name)
        return (
            not self.mtp_enabled
            and self.num_hidden_layers is not None
            and layer_match is not None
            and int(layer_match.group(1)) >= self.num_hidden_layers
        )

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        if self._is_disabled_mtp_key(name):
            return None
        # Official DeepSeek-V3 checkpoints repeat the shared embedding, final
        # norm, and LM-head tensors under every MTP layer. VeOmni keeps one
        # registered owner for each shared parameter so FSDP2 never sees the
        # same parameter across nested shard boundaries. Redirect each alias
        # to that canonical owner; loading it twice is harmless when canonical
        # keys are also present and remains safe for alias-only checkpoints.
        shared_match = _MTP_SHARED_PATTERN.match(name)
        if shared_match:
            return ConvertedCheckpointTensor(_MTP_SHARED_TARGETS[shared_match.group(1)], tensor)

        match = _EXPERT_PATTERN.match(name)
        if not match:
            return None

        prefix, expert_id_str, proj_name = match.groups()
        expert_id = int(expert_id_str)
        buf_key = (prefix, proj_name)

        if buf_key not in self._expert_buffer:
            self._expert_buffer[buf_key] = {}
        self._expert_buffer[buf_key][expert_id] = tensor

        if len(self._expert_buffer[buf_key]) < self.num_experts:
            return None

        # Stack all experts: [E, I, H] or [E, H, I]
        stacked = torch.stack([self._expert_buffer[buf_key][i] for i in range(self.num_experts)])
        del self._expert_buffer[buf_key]

        if proj_name == "down_proj":
            return ConvertedCheckpointTensor(f"{prefix}.experts.down_proj", stacked)

        # gate_proj or up_proj — buffer for merging with the other
        if prefix not in self._stacked_buffer:
            self._stacked_buffer[prefix] = {}
        self._stacked_buffer[prefix][proj_name] = stacked

        if "gate_proj" in self._stacked_buffer[prefix] and "up_proj" in self._stacked_buffer[prefix]:
            gate = self._stacked_buffer[prefix].pop("gate_proj")
            up = self._stacked_buffer[prefix].pop("up_proj")
            if not self._stacked_buffer[prefix]:
                del self._stacked_buffer[prefix]
            merged = torch.cat([gate, up], dim=1)  # [E, 2*I, H]
            return ConvertedCheckpointTensor(f"{prefix}.experts.gate_up_proj", merged)

        return None

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        """Validate that all buffers were flushed.

        Raises RuntimeError if any buffers remain unflushed, since incomplete
        expert tensors cannot be merged into valid fused format and indicate
        a corrupted or incomplete checkpoint.
        """
        errors: List[str] = []
        if self._expert_buffer:
            unflushed = {k: len(v) for k, v in self._expert_buffer.items()}
            errors.append(
                f"unflushed per-expert buffer (incomplete experts, expected {self.num_experts}): {unflushed}"
            )
        if self._stacked_buffer:
            unflushed = {k: list(v.keys()) for k, v in self._stacked_buffer.items()}
            errors.append(f"unflushed stacked buffer (missing gate/up pair): {unflushed}")
        if errors:
            raise RuntimeError("DeepseekV3 checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors))
        return []


def create_deepseek_v3_checkpoint_tensor_converter(model):
    """Factory function registered on model classes via _create_checkpoint_tensor_converter."""
    decoder = getattr(model, "model", model)
    return DeepseekV3CheckpointTensorConverter(
        num_experts=model.config.n_routed_experts,
        num_hidden_layers=model.config.num_hidden_layers,
        mtp_enabled=len(decoder.layers) > model.config.num_hidden_layers,
    )


def convert_deepseek_v3_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align HF safetensors index keys with fused expert parameter names."""
    from ..._moe_fused_weight_map import convert_per_expert_fqn_mapping_to_fused

    converted = convert_per_expert_fqn_mapping_to_fused(fqn_to_index_mapping, _EXPERT_PATTERN)
    for name, index in list(converted.items()):
        match = _MTP_SHARED_PATTERN.match(name)
        if match:
            converted.pop(name)
            converted.setdefault(_MTP_SHARED_TARGETS[match.group(1)], index)
    return converted
