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
Runtime checkpoint tensor converter for Qwen3-Omni-MoE (thinker) models.

HuggingFace's own loader would convert the on-disk per-expert keys into the
fused modeling layout via `conversion_mapping.py` (the `qwen2_moe` recipe:
`MergeModulelist(dim=0) + Concatenate(dim=1)`). VeOmni's loader reads
safetensors directly and never invokes that recipe, so we reproduce it here.

    HF on-disk layout (per-expert split):
        thinker.model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]
        thinker.model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
        thinker.model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]

    Target v5 modeling layout (fused):
        thinker.model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        thinker.model.layers.{i}.mlp.experts.down_proj     [E, H, I]

VeOmni's training save path can emit the v5 fused layout directly (e.g.
`save_pretrained(save_original_format=False)`). Those keys do not match the
per-expert regex here, so `maybe_convert_checkpoint_tensor` passes them
through untouched — the fused shape already equals the modeling layout, so
dispatch copies them into place.

The talker tower (`talker.model.layers.{i}.mlp.experts.*`) uses the same
per-expert convention; the regex matches both tower prefixes so standalone
talker tensors (if ever loaded through this converter) are handled uniformly.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch

from ....utils import logging
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


logger = logging.get_logger(__name__)

# Matches per-expert split keys like:
#   thinker.model.layers.0.mlp.experts.3.gate_proj.weight
_EXPERT_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class Qwen3OmniMoeCheckpointTensorConverter:
    """Stack per-expert gate/up/down tensors into v5 fused layout at load time.

    Buffers per-expert tensors keyed by ``(prefix, proj_name)`` as they stream
    from safetensors, stacks along dim-0 once all ``num_experts`` are collected,
    then merges ``gate_proj`` + ``up_proj`` along dim-1 to form ``gate_up_proj``.

    Args:
        num_experts: Number of experts per MoE layer.
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        # {(prefix, proj_name): {expert_id: tensor}}
        self._expert_buffer: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        # {prefix: {proj_name: stacked_tensor}} — waiting for gate/up pair
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

        if len(self._expert_buffer[buf_key]) < self.num_experts:
            return None

        stacked = torch.stack([self._expert_buffer[buf_key][i] for i in range(self.num_experts)])
        del self._expert_buffer[buf_key]

        if proj_name == "down_proj":
            # v5 modeling layout is [E, H, I] — already matches stacked shape.
            return ConvertedCheckpointTensor(f"{prefix}.experts.down_proj", stacked)

        # gate_proj or up_proj — buffer for gate/up merge.
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
        """Raise if any buffers remain — incomplete experts indicate a bad checkpoint."""
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
                "Qwen3OmniMoe checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors)
            )
        return []


def create_qwen3_omni_moe_checkpoint_tensor_converter(model):
    """Factory registered on model classes via `_create_checkpoint_tensor_converter`.

    Resolves the text config from whichever top-level config is attached to the
    model:
    - ``Qwen3OmniMoeConfig`` (top) → ``config.thinker_config.text_config``
    - ``Qwen3OmniMoeThinkerConfig`` → ``config.text_config``
    - ``Qwen3OmniMoeThinkerTextConfig`` (the inner text submodel) → ``config``
    """
    config = model.config
    thinker_config = getattr(config, "thinker_config", config)
    text_config = getattr(thinker_config, "text_config", thinker_config)
    return Qwen3OmniMoeCheckpointTensorConverter(num_experts=text_config.num_experts)
