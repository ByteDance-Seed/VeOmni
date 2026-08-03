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

"""Load-time conversion from the official HunyuanImage 3 layout into VeOmni's.

Registered on the model class as ``_create_checkpoint_tensor_converter``, so
``build_foundation_model`` streams the official checkpoint through it. Three
facts separate the two layouts:

* two key renames (``model.wte`` / ``model.ln_f``),
* the expert fusion: ``E`` official ``experts.{e}.{proj}.weight`` tensors stack
  into one ``experts.{proj}`` parameter,
* the half-order swap inside ``gate_and_up_proj``: official stores ``[up, gate]``,
  the Transformers 5.9 fused parameter stores ``[gate, up]``.

Everything else maps identically -- attention keeps the official
group-interleaved fused QKV (the patched runtime attention class reads it as
laid out), shared MLP / image projector / timestep / head / ``vae.encoder.*``
are untouched. Components the runtime drops (per
:class:`HunyuanImage3ComponentPolicy`) are skipped on import; reassembling an
official checkpoint means restoring their ``CHECKPOINT_PREFIXES`` entries from
the pinned official Base.
"""

import re
from collections import defaultdict
from typing import Optional

import torch

from ....utils import logging
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor
from .component_policy import HunyuanImage3ComponentPolicy


logger = logging.get_logger(__name__)

#: Official per-expert key, e.g. ``model.layers.3.mlp.experts.17.down_proj.weight``.
_SPLIT_EXPERT_PATTERN = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_and_up_proj|down_proj)\.weight$"
)

_OFFICIAL_TO_RUNTIME_RENAMES = {
    "model.wte.weight": "model.embed_tokens.weight",
    "model.ln_f.weight": "model.norm.weight",
}


class _ExpertLayout:
    """Expert tensor geometry shared by the official and fused layouts."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def official_shape(self, projection: str) -> tuple[int, ...]:
        """Per-expert shape; ``down_proj`` is ``[H, I]``, either gate/up name is ``[2I, H]``."""
        if projection == "down_proj":
            return (self.hidden_size, self.intermediate_size)
        return (2 * self.intermediate_size, self.hidden_size)

    def fused_shape(self, projection: str) -> tuple[int, ...]:
        return (self.num_experts, *self.official_shape(projection))

    @staticmethod
    def check_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Unexpected shape for {name}: got {tuple(tensor.shape)}, expected {expected}.")


class HunyuanImage3CheckpointTensorConverter(_ExpertLayout):
    """Stream the official split-expert layout into the runtime layout."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        component_policy: HunyuanImage3ComponentPolicy,
    ) -> None:
        super().__init__(num_experts, hidden_size, intermediate_size)
        self.component_policy = component_policy
        self._expert_buffer: dict[tuple[str, str], dict[int, torch.Tensor]] = defaultdict(dict)
        self._skipped_prefix_counts: dict[str, int] = defaultdict(int)

    def can_handle(self, name: str) -> bool:
        return (
            name in _OFFICIAL_TO_RUNTIME_RENAMES
            or bool(_SPLIT_EXPERT_PATTERN.match(name))
            or self.component_policy.checkpoint_prefix_is_absent(name)
        )

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        if self.component_policy.checkpoint_prefix_is_absent(name):
            self._skipped_prefix_counts[name.split(".", maxsplit=1)[0]] += 1
            return None

        renamed = _OFFICIAL_TO_RUNTIME_RENAMES.get(name)
        if renamed is not None:
            return ConvertedCheckpointTensor(name=renamed, tensor=tensor)

        match = _SPLIT_EXPERT_PATTERN.match(name)
        if match is None:
            return None
        prefix, expert_id_text, projection = match.groups()
        expert_id = int(expert_id_text)
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(f"Expert id {expert_id} is outside [0, {self.num_experts}) for checkpoint key {name}.")

        key = (prefix, projection)
        if expert_id in self._expert_buffer[key]:
            raise ValueError(f"Duplicate expert tensor in checkpoint: {name}.")
        self.check_shape(name, tensor, self.official_shape(projection))
        self._expert_buffer[key][expert_id] = tensor
        if len(self._expert_buffer[key]) != self.num_experts:
            return None

        experts = self._expert_buffer.pop(key)
        if projection == "down_proj":
            stacked = experts[0].new_empty(self.fused_shape("down_proj"))
            for index in range(self.num_experts):
                stacked[index].copy_(experts[index])
            return ConvertedCheckpointTensor(name=f"{prefix}.experts.down_proj", tensor=stacked)

        stacked = experts[0].new_empty(self.fused_shape("gate_up_proj"))
        for index in range(self.num_experts):
            source = experts[index]  # official [up, gate] -> runtime [gate, up]
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


def create_hunyuan_image_3_checkpoint_tensor_converter(model) -> HunyuanImage3CheckpointTensorConverter:
    config = model.config
    if not isinstance(config.num_experts, int):
        raise ValueError("The HunyuanImage 3 checkpoint converter requires one num_experts value.")
    return HunyuanImage3CheckpointTensorConverter(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        component_policy=HunyuanImage3ComponentPolicy.from_dict(config.component_policy),
    )


__all__ = [
    "HunyuanImage3CheckpointTensorConverter",
    "create_hunyuan_image_3_checkpoint_tensor_converter",
]
