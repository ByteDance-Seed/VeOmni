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
Runtime checkpoint tensor converter for Qwen3-VL-MoE models.

HuggingFace already ships Qwen3-VL-MoE checkpoints with fused expert tensors,
but stored in a layout that is *transposed* relative to the transformers v5
modeling. This converter fixes the axis order in-place at load time; no
per-expert stacking is needed (unlike `qwen3_moe`).

    HF checkpoint layout:
        model.language_model.layers.{i}.mlp.experts.gate_up_proj  [E, H, 2*I]
        model.language_model.layers.{i}.mlp.experts.down_proj     [E, I, H]

    Target v5 modeling layout:
        model.language_model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        model.language_model.layers.{i}.mlp.experts.down_proj     [E, H, I]

Because VeOmni's training save path can also emit the v5 layout directly (e.g.
`save_pretrained(save_original_format=False)`), the converter uses the dim-1
shape to distinguish HF layout from v5 layout and only transposes when needed;
v5-layout tensors pass through untouched. Shapes matching neither layout raise
loudly rather than silently corrupting weights.
"""

import re
from typing import List, Optional

import torch

from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


# Matches fused-expert keys like: ...mlp.experts.{gate_up_proj|down_proj}
_EXPERT_PATTERN = re.compile(r"^(.+\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj))$")


class Qwen3VLMoeCheckpointTensorConverter:
    """Normalize fused expert tensors to the v5 modeling layout.

    The incoming tensor is always 3-D with dim-0 == ``num_experts``. For each
    projection we know both the HF layout and the v5 layout shapes exactly, so
    we dispatch on dim-1:

    - ``gate_up_proj``: HF has dim-1 == ``hidden_size``, v5 has dim-1 == ``2 * intermediate_size``.
    - ``down_proj``:    HF has dim-1 == ``intermediate_size``, v5 has dim-1 == ``hidden_size``.

    Hidden size, intermediate size and (for Qwen3-VL-MoE) their doubled variants
    are all distinct integers for any realistic model, so the dispatch is
    unambiguous.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def can_handle(self, name: str) -> bool:
        return bool(_EXPERT_PATTERN.match(name))

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        match = _EXPERT_PATTERN.match(name)
        if not match:
            return None

        proj = match.group("proj")

        if tensor.ndim != 3 or tensor.shape[0] != self.num_experts:
            raise RuntimeError(
                f"Qwen3VLMoe checkpoint converter: unexpected shape {tuple(tensor.shape)} for {name} "
                f"(expected 3-D with dim-0 == num_experts={self.num_experts})"
            )

        if proj == "gate_up_proj":
            hf_mid, v5_mid = self.hidden_size, 2 * self.intermediate_size
        else:  # down_proj
            hf_mid, v5_mid = self.intermediate_size, self.hidden_size

        if tensor.shape[1] == hf_mid:
            converted = tensor.transpose(1, 2).contiguous()
        elif tensor.shape[1] == v5_mid:
            converted = tensor
        else:
            raise RuntimeError(
                f"Qwen3VLMoe checkpoint converter: unrecognized layout for {name} "
                f"(shape={tuple(tensor.shape)}; expected dim-1 == {hf_mid} (HF) or {v5_mid} (v5))"
            )
        return ConvertedCheckpointTensor(name, converted)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        return []


def create_qwen3_vl_moe_checkpoint_tensor_converter(model):
    """Factory registered on model classes via `_create_checkpoint_tensor_converter`.

    Resolves the text config from either the top-level `Qwen3VLMoeConfig`
    (has nested `text_config`) or the inner `Qwen3VLMoeTextConfig` directly
    (when the converter is attached to `Qwen3VLMoeTextModel`).
    """
    config = model.config
    text_config = getattr(config, "text_config", config)
    return Qwen3VLMoeCheckpointTensorConverter(
        num_experts=text_config.num_experts,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.moe_intermediate_size,
    )
