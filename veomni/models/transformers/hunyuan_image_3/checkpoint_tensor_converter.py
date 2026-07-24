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

"""Streaming import conversion for the official HunyuanImage 3 checkpoint."""

import re
from collections import defaultdict
from typing import Optional

import torch

from ....utils import logging
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor
from .component_policy import HunyuanImage3ComponentPolicy


logger = logging.get_logger(__name__)

_EXPERT_PATTERN = re.compile(r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_and_up_proj|down_proj)\.weight$")
_DIRECT_RENAMES = {
    "model.wte.weight": "model.embed_tokens.weight",
    "model.ln_f.weight": "model.norm.weight",
}


class HunyuanImage3CheckpointTensorConverter:
    """Convert the official split-expert layout into the VeOmni runtime layout.

    The official expert ``gate_and_up_proj`` stores ``[up, gate]`` halves. The
    Transformers 5.9 fused expert parameter stores ``[gate, up]`` halves, so
    stacking must also swap the two halves. The official group-interleaved QKV
    tensor is preserved unchanged by the patched runtime attention class.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        component_policy: HunyuanImage3ComponentPolicy,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.component_policy = component_policy
        self._expert_buffer: dict[tuple[str, str], dict[int, torch.Tensor]] = defaultdict(dict)
        self._skipped_prefix_counts: dict[str, int] = defaultdict(int)

    def can_handle(self, name: str) -> bool:
        return (
            name in _DIRECT_RENAMES
            or bool(_EXPERT_PATTERN.match(name))
            or self.component_policy.checkpoint_prefix_is_absent(name)
        )

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        if self.component_policy.checkpoint_prefix_is_absent(name):
            prefix = name.split(".", maxsplit=1)[0]
            self._skipped_prefix_counts[prefix] += 1
            return None

        renamed = _DIRECT_RENAMES.get(name)
        if renamed is not None:
            return ConvertedCheckpointTensor(name=renamed, tensor=tensor)

        match = _EXPERT_PATTERN.match(name)
        if match is None:
            return None
        prefix, expert_id_text, projection = match.groups()
        expert_id = int(expert_id_text)
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(f"Expert id {expert_id} is outside [0, {self.num_experts}) for checkpoint key {name}.")

        key = (prefix, projection)
        if expert_id in self._expert_buffer[key]:
            raise ValueError(f"Duplicate expert tensor in checkpoint: {name}.")
        self._validate_expert_shape(name, projection, tensor)
        self._expert_buffer[key][expert_id] = tensor
        if len(self._expert_buffer[key]) != self.num_experts:
            return None

        experts = self._expert_buffer.pop(key)
        if projection == "down_proj":
            stacked = experts[0].new_empty((self.num_experts, self.hidden_size, self.intermediate_size))
            for index in range(self.num_experts):
                stacked[index].copy_(experts[index])
            return ConvertedCheckpointTensor(name=f"{prefix}.experts.down_proj", tensor=stacked)

        stacked = experts[0].new_empty((self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        for index in range(self.num_experts):
            source = experts[index]
            stacked[index, : self.intermediate_size].copy_(source[self.intermediate_size :])
            stacked[index, self.intermediate_size :].copy_(source[: self.intermediate_size])
        return ConvertedCheckpointTensor(name=f"{prefix}.experts.gate_up_proj", tensor=stacked)

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        if self._expert_buffer:
            incomplete = {
                f"{prefix}.{projection}": sorted(experts)
                for (prefix, projection), experts in self._expert_buffer.items()
            }
            raise RuntimeError(
                "HunyuanImage 3 checkpoint converter found incomplete expert sets; "
                f"expected expert ids 0..{self.num_experts - 1}: {incomplete}."
            )
        if self._skipped_prefix_counts:
            logger.info_rank0(
                "Skipped official checkpoint tensors for absent HunyuanImage 3 components: "
                f"{dict(sorted(self._skipped_prefix_counts.items()))}."
            )
        return []

    def is_dim0_zero_pad(self, name: str) -> bool:
        del name
        return False

    def _validate_expert_shape(self, name: str, projection: str, tensor: torch.Tensor) -> None:
        expected = (
            (self.hidden_size, self.intermediate_size)
            if projection == "down_proj"
            else (2 * self.intermediate_size, self.hidden_size)
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Unexpected shape for {name}: got {tuple(tensor.shape)}, expected {expected}.")


def create_hunyuan_image_3_checkpoint_tensor_converter(model) -> HunyuanImage3CheckpointTensorConverter:
    config = model.config
    if not isinstance(config.num_experts, int):
        raise ValueError("The initial HunyuanImage 3 checkpoint converter requires one num_experts value.")
    policy = HunyuanImage3ComponentPolicy.from_dict(config.component_policy)
    return HunyuanImage3CheckpointTensorConverter(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        component_policy=policy,
    )


__all__ = [
    "HunyuanImage3CheckpointTensorConverter",
    "create_hunyuan_image_3_checkpoint_tensor_converter",
]
