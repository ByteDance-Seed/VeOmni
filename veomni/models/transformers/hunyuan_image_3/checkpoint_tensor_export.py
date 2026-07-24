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

"""Export conversion: VeOmni runtime layout -> official HunyuanImage 3 layout.

Exact inverse of :mod:`.checkpoint_tensor_converter` (the import path):

* ``model.embed_tokens.weight`` -> ``model.wte.weight``,
  ``model.norm.weight`` -> ``model.ln_f.weight`` (inverse of the two renames).
* The fused expert ``experts.gate_up_proj`` ``[E, 2I, H]`` (runtime ``[gate, up]``
  halves) splits into ``E`` official ``experts.{e}.gate_and_up_proj.weight`` ``[2I, H]``
  tensors with the halves swapped back to the official ``[up, gate]`` order;
  ``experts.down_proj`` ``[E, H, I]`` splits into ``E`` ``experts.{e}.down_proj.weight``
  ``[H, I]`` tensors unchanged.
* Every other runtime key (attention group-interleaved QKV, shared MLP, image
  projector / timestep / head, ``vae.encoder.*``) maps identity.

Components absent from the runtime model (``lm_head`` / ``vae.decoder`` / vision, per
component policy) are NOT produced here; :func:`absent_official_prefixes` lists them so
the caller can restore them byte-for-byte from the pinned official Base.
"""

import re
from typing import Iterable

import torch

from .component_policy import HunyuanImage3ComponentPolicy


_INVERSE_RENAMES = {
    "model.embed_tokens.weight": "model.wte.weight",
    "model.norm.weight": "model.ln_f.weight",
}
_FUSED_EXPERT_PATTERN = re.compile(r"^(model\.layers\.\d+\.mlp)\.experts\.(gate_up_proj|down_proj)$")

# Official checkpoint prefixes for components the initial T2I runtime may drop.
_ABSENT_PREFIXES = {
    "lm_head": "lm_head.",
    "vae_decoder": "vae.decoder.",
    "vision_model": "vision_model.",
    "vision_aligner": "vision_aligner.",
}


class HunyuanImage3CheckpointExporter:
    """Stream runtime tensors back to the official split-expert layout."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def export_tensor(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Return the official ``(name, tensor)`` pair(s) for one runtime tensor."""
        renamed = _INVERSE_RENAMES.get(name)
        if renamed is not None:
            return [(renamed, tensor)]

        match = _FUSED_EXPERT_PATTERN.match(name)
        if match is None:
            return [(name, tensor)]  # identity for every non-expert, non-renamed key

        prefix, projection = match.groups()
        if projection == "down_proj":
            self._validate_fused_shape(name, tensor, (self.num_experts, self.hidden_size, self.intermediate_size))
            return [
                (f"{prefix}.experts.{e}.down_proj.weight", tensor[e].contiguous()) for e in range(self.num_experts)
            ]

        self._validate_fused_shape(name, tensor, (self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        pairs = []
        for e in range(self.num_experts):
            official = tensor.new_empty((2 * self.intermediate_size, self.hidden_size))
            # runtime [gate, up] -> official [up, gate]
            official[: self.intermediate_size].copy_(tensor[e, self.intermediate_size :])
            official[self.intermediate_size :].copy_(tensor[e, : self.intermediate_size])
            pairs.append((f"{prefix}.experts.{e}.gate_and_up_proj.weight", official))
        return pairs

    def export_state(self, items: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in items:
            yield from self.export_tensor(name, tensor)

    def _validate_fused_shape(self, name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected fused expert shape for {name}: got {tuple(tensor.shape)}, expected {expected}."
            )


def absent_official_prefixes(component_policy: HunyuanImage3ComponentPolicy) -> list[str]:
    """Official key prefixes to restore byte-for-byte from the pinned Base on export."""
    prefixes = [prefix for name, prefix in _ABSENT_PREFIXES.items() if component_policy.state(name) == "absent"]
    if component_policy.vae_encoder == "absent":
        prefixes.append("vae.encoder.")
    return sorted(prefixes)


def create_hunyuan_image_3_checkpoint_exporter(config) -> HunyuanImage3CheckpointExporter:
    if not isinstance(config.num_experts, int):
        raise ValueError("The HunyuanImage 3 exporter requires one num_experts value.")
    return HunyuanImage3CheckpointExporter(
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
    )


__all__ = [
    "HunyuanImage3CheckpointExporter",
    "absent_official_prefixes",
    "create_hunyuan_image_3_checkpoint_exporter",
]
