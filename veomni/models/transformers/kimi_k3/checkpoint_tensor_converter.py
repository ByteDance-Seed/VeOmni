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
Runtime checkpoint tensor converter for Kimi-K3 models.

Converts the HuggingFace per-expert checkpoint format to the v5 fused format
at load time, eliminating the need for offline checkpoint merging.

    HF checkpoint format (per-expert, ``KimiBlockSparseMLP`` naming):
        {prefix}.block_sparse_moe.experts.{j}.w1.weight   [I, H]  (gate)
        {prefix}.block_sparse_moe.experts.{j}.w3.weight   [I, H]  (up)
        {prefix}.block_sparse_moe.experts.{j}.w2.weight   [H, I]  (down)

    Target v5 format (F.linear layout, matches ``PatchedKimiK3Experts``):
        {prefix}.block_sparse_moe.experts.gate_up_proj    [E, 2*I, H]
        {prefix}.block_sparse_moe.experts.down_proj       [E, H, I]

where ``H`` is the routed-expert hidden size (``routed_expert_hidden_size``
for the latent MoE) and ``I`` is ``moe_intermediate_size``.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch

from ....utils import logging
from ...checkpoint import ConvertedCheckpointTensor


logger = logging.get_logger(__name__)

# Matches per-expert split keys like:
#   language_model.model.layers.0.block_sparse_moe.experts.3.w1.weight
_EXPERT_PATTERN = re.compile(r"^(.+)\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


class KimiK3CheckpointTensorConverter:
    """Converts per-expert split checkpoint keys to stacked & merged v5 format.

    Buffers per-expert tensors as they stream from safetensor files, and emits
    merged tensors once all experts for a given (prefix, projection) are collected.

    Args:
        num_experts: Number of routed experts per MoE layer.
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        # {(prefix, proj_name): {expert_id: tensor}}
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        # {prefix: {proj_name: stacked_tensor}} for w1/w3 merge waiting
        self._stacked_buffer: Dict[str, Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(_EXPERT_PATTERN.match(name))

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        match = _EXPERT_PATTERN.match(name)
        if not match:
            return None

        prefix, expert_id_str, proj_name = match.groups()
        expert_id = int(expert_id_str)
        buf_key = (prefix, proj_name)

        if buf_key not in self._expert_buffer:
            self._expert_buffer[buf_key] = {}
        self._expert_buffer[buf_key][expert_id] = tensor

        # Check if all experts collected for this (prefix, proj)
        if len(self._expert_buffer[buf_key]) < self.num_experts:
            return None

        # Stack all experts: [E, I, H] (w1/w3) or [E, H, I] (w2)
        stacked = torch.stack([self._expert_buffer[buf_key][i] for i in range(self.num_experts)])
        del self._expert_buffer[buf_key]

        if proj_name == "w2":  # down projection
            return ConvertedCheckpointTensor(f"{prefix}.experts.down_proj", stacked)

        # w1 (gate) or w3 (up) — buffer for merging with the other
        if prefix not in self._stacked_buffer:
            self._stacked_buffer[prefix] = {}
        self._stacked_buffer[prefix][proj_name] = stacked

        if "w1" in self._stacked_buffer[prefix] and "w3" in self._stacked_buffer[prefix]:
            gate = self._stacked_buffer[prefix].pop("w1")
            up = self._stacked_buffer[prefix].pop("w3")
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
            errors.append(f"unflushed stacked buffer (missing w1/w3 pair): {unflushed}")
        if errors:
            raise RuntimeError("KimiK3 checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors))
        return []


def _get_text_config(config):
    if hasattr(config, "text_config") and config.text_config is not None:
        return config.text_config
    return config


def create_kimi_k3_checkpoint_tensor_converter(model):
    """Factory function registered on model classes via _create_checkpoint_tensor_converter."""
    text_config = _get_text_config(model.config)
    return KimiK3CheckpointTensorConverter(
        num_experts=text_config.num_experts,
    )


def convert_kimi_k3_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Align HF safetensors index keys with fused expert parameter names.

    Per-expert ``w1``/``w3`` shards map to ``{prefix}.experts.gate_up_proj`` and
    ``w2`` shards to ``{prefix}.experts.down_proj`` (keeping the minimum shard
    index so every fused key resolves to the shard file holding its first slice).
    Non-matching keys are copied unchanged.
    """
    gate_up_shard_indices: Dict[str, List[int]] = {}
    down_shard_indices: Dict[str, List[int]] = {}
    converted: Dict[str, int] = {}

    for fqn, shard_idx in fqn_to_index_mapping.items():
        match = _EXPERT_PATTERN.match(fqn)
        if not match:
            converted[fqn] = shard_idx
            continue

        prefix, _expert_id, proj_name = match.groups()
        if proj_name == "w2":
            down_shard_indices.setdefault(prefix, []).append(shard_idx)
        else:
            gate_up_shard_indices.setdefault(prefix, []).append(shard_idx)

    for prefix, indices in down_shard_indices.items():
        converted[f"{prefix}.experts.down_proj"] = min(indices)

    for prefix, indices in gate_up_shard_indices.items():
        converted[f"{prefix}.experts.gate_up_proj"] = min(indices)

    return converted
