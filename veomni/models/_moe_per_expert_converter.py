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

"""Shared runtime converter: per-expert HF MoE checkpoint keys -> v5 fused experts.

HuggingFace's own loader fuses these keys through ``conversion_mapping.py``
(``MergeModulelist(dim=0) + Concatenate(dim=1)``). VeOmni reads safetensors
directly and never runs that recipe, so this converter reproduces it::

    HF per-expert:                                VeOmni fused:
      {prefix}.experts.{j}.gate_proj.weight [I, H]  ->  {prefix}.experts.gate_up_proj [E, 2*I, H]
      {prefix}.experts.{j}.up_proj.weight   [I, H]  ->    (gate / up concatenated on dim 1)
      {prefix}.experts.{j}.down_proj.weight [H, I]  ->  {prefix}.experts.down_proj    [E, H, I]

Fused keys (e.g. a VeOmni ``save_original_format=False`` export) do not match
the pattern and pass through to dispatch untouched.

The converter also implements the optional ``fused_expert_target`` /
``for_expert_range`` capabilities, so ``ep_sharded_stream_load`` streams these
checkpoints with each EP rank reading only its own experts.
"""

from typing import Dict, List, Optional, Tuple

import torch

from ._moe_fused_weight_map import PER_EXPERT_SPLIT_TO_FUSED_PATTERN
from .checkpoint_tensor_loading import ConvertedCheckpointTensor


class PerExpertFusedCheckpointTensorConverter:
    """Stack per-expert gate/up/down tensors into the v5 fused expert layout.

    Buffers per-expert tensors keyed by ``(prefix, proj_name)`` as they stream
    from safetensors, stacks along dim-0 once all ``num_experts`` are collected,
    then merges ``gate_proj`` + ``up_proj`` along dim-1 to form ``gate_up_proj``.

    Args:
        num_experts: Number of experts per MoE layer this converter stacks.
        expert_offset: Checkpoint index of the first expert it stacks; experts
            ``[expert_offset, expert_offset + num_experts)`` become rows ``0..num_experts-1``.
    """

    #: Prefix of the ``finalize`` error, naming the model family.
    model_name = "MoE"

    def __init__(self, num_experts: int, expert_offset: int = 0):
        self.num_experts = num_experts
        self.expert_offset = expert_offset
        # {(prefix, proj_name): {local_expert_id: tensor}}
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        # {prefix: {proj_name: stacked_tensor}} for gate/up merge waiting
        self._stacked_buffer: Dict[str, Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(PER_EXPERT_SPLIT_TO_FUSED_PATTERN.match(name))

    def fused_expert_target(self, name: str) -> Optional[Tuple[str, int]]:
        match = PER_EXPERT_SPLIT_TO_FUSED_PATTERN.match(name)
        if not match:
            return None
        prefix, expert_id_str, proj_name = match.groups()
        fused_proj = "down_proj" if proj_name == "down_proj" else "gate_up_proj"
        return f"{prefix}.experts.{fused_proj}", int(expert_id_str)

    def for_expert_range(self, start: int, num_local: int) -> "PerExpertFusedCheckpointTensorConverter":
        if start < 0 or num_local <= 0 or start + num_local > self.num_experts:
            raise ValueError(
                f"Expert range [{start}, {start + num_local}) is outside this converter's {self.num_experts} experts."
            )
        return type(self)(num_experts=num_local, expert_offset=self.expert_offset + start)

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        match = PER_EXPERT_SPLIT_TO_FUSED_PATTERN.match(name)
        if not match:
            return None

        prefix, expert_id_str, proj_name = match.groups()
        expert_id = int(expert_id_str) - self.expert_offset
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(
                f"{name}: expert {int(expert_id_str)} is outside this converter's range "
                f"[{self.expert_offset}, {self.expert_offset + self.num_experts})."
            )
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
            raise RuntimeError(
                f"{self.model_name} checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors)
            )
        return []
