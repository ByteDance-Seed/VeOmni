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

import re
from typing import TYPE_CHECKING, Dict, Optional

import torch

from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...hf_checkpoint_tensor_converter import HfConvertedCheckpointTensor


if TYPE_CHECKING:
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


_QWEN3_MOE_EXPERT_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


class Qwen3MoeV5CheckpointTensorConverter:
    def __init__(self, config: "Qwen3MoeConfig"):
        """Initialize a qwen3_moe checkpoint tensor converter for transformers v5 layouts.

        Args:
            config: Qwen3MoeConfig used to validate layer/expert index ranges and tensor shapes.
        """
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_hidden_layers = config.num_hidden_layers

        # Per-layer temporary cache for per-expert gate tensors `[I, H]` before emitting merged gate_up.
        self._gate_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
        # Per-layer temporary cache for per-expert up tensors `[I, H]` before emitting merged gate_up.
        self._up_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
        # Per-layer temporary cache for per-expert down tensors `[H, I]` before emitting merged down.
        self._down_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
        # Tracks layers whose merged gate_up tensor has been emitted to avoid duplicate emission.
        self._gate_up_emitted_layers: set[int] = set()
        # Tracks layers whose merged down tensor has been emitted to avoid duplicate emission.
        self._down_emitted_layers: set[int] = set()

    def can_handle(self, name: str) -> bool:
        """Check if a safetensor key is a qwen3_moe per-expert tensor key.

        Args:
            name: Raw state dict key from safetensors.

        Returns:
            True if the key matches qwen3_moe per-expert gate/up/down format.
        """
        return _QWEN3_MOE_EXPERT_KEY.match(name) is not None

    @torch.no_grad()
    def convert(self, name: str, tensor: torch.Tensor) -> Optional[HfConvertedCheckpointTensor]:
        """Consume one safetensor tensor and optionally emit a merged transformers-v5 tensor.

        Args:
            name: State dict key after key-conversion mapping.
            tensor: Tensor loaded for `name`.

        Returns:
            HfConvertedCheckpointTensor when a full merged tensor is ready, `None` when still accumulating.
        """
        match = _QWEN3_MOE_EXPERT_KEY.match(name)
        if match is None:
            return HfConvertedCheckpointTensor(name=name, tensor=tensor)

        layer_idx = int(match.group("layer"))
        expert_idx = int(match.group("expert"))
        proj_name = match.group("proj")
        if expert_idx >= self.num_experts:
            raise ValueError(
                f"qwen3_moe converter got expert index {expert_idx} >= num_experts {self.num_experts} for key: {name}"
            )
        if layer_idx >= self.num_hidden_layers:
            raise ValueError(
                f"qwen3_moe converter got layer index {layer_idx} >= num_hidden_layers {self.num_hidden_layers} for key: {name}"
            )

        self._validate_input_tensor_shape(name, proj_name, tensor)

        if proj_name == "gate_proj":
            self._gate_by_layer.setdefault(layer_idx, {})[expert_idx] = tensor
            gate_up = self._try_build_gate_up(layer_idx)
            if gate_up is not None:
                return HfConvertedCheckpointTensor(
                    name=f"model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                    tensor=gate_up,
                )
        elif proj_name == "up_proj":
            self._up_by_layer.setdefault(layer_idx, {})[expert_idx] = tensor
            gate_up = self._try_build_gate_up(layer_idx)
            if gate_up is not None:
                return HfConvertedCheckpointTensor(
                    name=f"model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                    tensor=gate_up,
                )
        elif proj_name == "down_proj":
            self._down_by_layer.setdefault(layer_idx, {})[expert_idx] = tensor
            down = self._try_build_down(layer_idx)
            if down is not None:
                return HfConvertedCheckpointTensor(
                    name=f"model.layers.{layer_idx}.mlp.experts.down_proj",
                    tensor=down,
                )

        return None

    def _validate_input_tensor_shape(self, name: str, proj_name: str, tensor: torch.Tensor) -> None:
        """Validate the incoming per-expert tensor shape against Qwen3MoeConfig."""
        expected_shape = {
            "gate_proj": (self.moe_intermediate_size, self.hidden_size),
            "up_proj": (self.moe_intermediate_size, self.hidden_size),
            "down_proj": (self.hidden_size, self.moe_intermediate_size),
        }[proj_name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"qwen3_moe converter got unexpected shape for key {name}: "
                f"got {tuple(tensor.shape)}, expected {expected_shape} from config."
            )

    def _try_build_gate_up(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Emit merged gate_up tensor `[E, 2*I, H]` once all experts are accumulated.

        This also drops per-layer gate/up accumulation buffers immediately to save CPU memory.
        """
        if layer_idx in self._gate_up_emitted_layers:
            return None
        gate_by_expert = self._gate_by_layer.get(layer_idx, {})
        up_by_expert = self._up_by_layer.get(layer_idx, {})
        if len(gate_by_expert) < self.num_experts or len(up_by_expert) < self.num_experts:
            return None

        gate = torch.stack([gate_by_expert[expert_idx] for expert_idx in range(self.num_experts)], dim=0)
        up = torch.stack([up_by_expert[expert_idx] for expert_idx in range(self.num_experts)], dim=0)
        self._gate_up_emitted_layers.add(layer_idx)
        self._gate_by_layer.pop(layer_idx, None)
        self._up_by_layer.pop(layer_idx, None)
        return torch.cat([gate, up], dim=1)

    def _try_build_down(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Emit merged down tensor `[E, H, I]` once all experts are accumulated.

        This also drops per-layer down accumulation buffers immediately to save CPU memory.
        """
        if layer_idx in self._down_emitted_layers:
            return None
        down_by_expert = self._down_by_layer.get(layer_idx, {})
        if len(down_by_expert) < self.num_experts:
            return None

        self._down_emitted_layers.add(layer_idx)
        down = torch.stack([down_by_expert[expert_idx] for expert_idx in range(self.num_experts)], dim=0)
        self._down_by_layer.pop(layer_idx, None)
        return down


def create_qwen3_moe_checkpoint_tensor_converter(model) -> Optional[Qwen3MoeV5CheckpointTensorConverter]:
    """Create qwen3_moe checkpoint converter for transformers v5 runtime only.

    Args:
        model: Model instance being loaded from checkpoint.

    Returns:
        Qwen3MoeV5CheckpointTensorConverter when model/runtime matches expected conditions, else None.
    """
    if not is_transformers_version_greater_or_equal_to("5.0.0"):
        return None
    if getattr(model.config, "model_type", None) != "qwen3_moe":
        return None

    required_attrs = ("num_experts", "hidden_size", "moe_intermediate_size", "num_hidden_layers")
    missing_attrs = [attr for attr in required_attrs if not hasattr(model.config, attr)]
    if missing_attrs:
        raise ValueError(f"qwen3_moe converter requires config attrs: {required_attrs}, missing={missing_attrs}.")
    return Qwen3MoeV5CheckpointTensorConverter(config=model.config)
