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

"""
Runtime checkpoint tensor converter for DeepseekV4 MoE models.

DeepSeek-V4 upstream checkpoints ship per-expert split keys with the
``w1`` / ``w2`` / ``w3`` naming convention (HF's
``conversion_mapping.py`` recipe is ``MergeModulelist(dim=0) +
Concatenate(dim=1)`` for ``w1`` / ``w3`` -> ``gate_up_proj``, plus
``MergeModulelist(dim=0)`` for ``w2`` -> ``down_proj``). VeOmni's runtime
loader needs the v5 fused layout directly, so we stack + merge per-expert
tensors as they stream in from safetensors.

    HF checkpoint format (per-expert):
        model.layers.{i}.mlp.experts.{j}.w1.weight  [I, H]   # gate
        model.layers.{i}.mlp.experts.{j}.w2.weight  [H, I]   # down
        model.layers.{i}.mlp.experts.{j}.w3.weight  [I, H]   # up

    Target v5 format (direct layout, matches DeepseekV4Experts.__init__):
        model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        model.layers.{i}.mlp.experts.down_proj     [E, H, I]

VeOmni's runtime loader reads safetensor keys directly and does not run HF's
``WeightConverter`` objects, so this converter accepts both the raw on-disk
``w1`` / ``w2`` / ``w3`` keys and the already-renamed
``gate_proj`` / ``down_proj`` / ``up_proj`` form.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

import torch
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming, rename_source_key

from ....utils import logging
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


logger = logging.get_logger(__name__)

# Matches raw DeepSeek-V4 checkpoint keys (w1/w2/w3) and post-rename
# keys (gate_proj/up_proj/down_proj), e.g.:
#   model.layers.0.mlp.experts.3.w1.weight
#   model.layers.0.mlp.experts.3.gate_proj.weight
_EXPERT_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)\.weight$")
_RAW_FP4_EXPERT_WEIGHT_PATTERN = re.compile(r"^(?:model\.)?layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
_PROJ_NAME_ALIASES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}
_DEEPSEEK_V4_WEIGHT_RENAMINGS = [
    transform
    for transform in (get_checkpoint_conversion_mapping("deepseek_v4") or [])
    if isinstance(transform, WeightRenaming)
]
_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_BASE_MODEL_KEY_PREFIXES = ("embed_tokens.", "hc_head.", "layers.", "norm.")
_FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, name, None)
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu")
    )
    if dtype
)
_QUANTIZED_WEIGHT_DTYPES = _FLOAT8_DTYPES + (torch.int8,)


def _is_quantized_weight(tensor: torch.Tensor) -> bool:
    return tensor.dtype in _QUANTIZED_WEIGHT_DTYPES


def _is_mtp_checkpoint_key(name: str) -> bool:
    return name.removeprefix("model.").startswith("mtp.")


def convert_deepseek_v4_checkpoint_key(name: str, *, target_model_prefix: str = "model.") -> str:
    """Convert DeepSeek's inference checkpoint names to Transformers v5 names."""
    if _is_mtp_checkpoint_key(name):
        return name
    has_model_prefix = name.startswith("model.")
    source_name = name.removeprefix("model.") if has_model_prefix else name
    converted_name, _ = rename_source_key(source_name, _DEEPSEEK_V4_WEIGHT_RENAMINGS, [])
    if converted_name.startswith(_BASE_MODEL_KEY_PREFIXES):
        return f"{target_model_prefix}{converted_name}"
    if has_model_prefix and converted_name == source_name:
        return name
    return converted_name


@lru_cache(maxsize=None)
def _get_fp4_e2m1_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32, device=device)


def _is_block_scale_tensor(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 or tensor.numel() == 1


def _weight_name_from_scale_name(name: str) -> str | None:
    if name.endswith(".scale"):
        return f"{name.removesuffix('.scale')}.weight"
    if name.endswith(".weight_scale_inv"):
        return f"{name.removesuffix('.weight_scale_inv')}.weight"
    return None


def _dequantize_scaled_weight(weight: torch.Tensor, scale: torch.Tensor, *, packed_fp4: bool = False) -> torch.Tensor:
    """Apply DeepSeek block scales to an FP8 or packed E2M1 FP4 weight."""
    scale = scale.to(device=weight.device)
    if packed_fp4:
        if weight.dtype != torch.int8:
            raise TypeError(f"Packed E2M1 FP4 weights must use int8 storage, got {weight.dtype}.")
        # DeepSeek-V4 stores two E2M1 FP4 expert values in every int8 byte.
        # Preserve the bit pattern when splitting the low and high nibbles.
        packed = weight.contiguous().view(torch.uint8)
        table = _get_fp4_e2m1_table(weight.device)
        low = table[(packed & 0x0F).long()]
        high = table[((packed >> 4) & 0x0F).long()]
        weight = torch.stack((low, high), dim=-1).flatten(-2)
    else:
        weight = weight.float()

    if scale.numel() == 1:
        return weight * scale.float()
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "DeepseekV4 quantized checkpoint weight scales must be scalar or 2D block scales; "
            f"got weight shape {tuple(weight.shape)} and scale shape {tuple(scale.shape)}."
        )
    if weight.shape[0] % scale.shape[0] != 0 or weight.shape[1] % scale.shape[1] != 0:
        raise ValueError(
            "DeepseekV4 quantized checkpoint weight shape must be divisible by scale shape; "
            f"got weight shape {tuple(weight.shape)} and scale shape {tuple(scale.shape)}."
        )

    block_rows = weight.shape[0] // scale.shape[0]
    block_cols = weight.shape[1] // scale.shape[1]
    return (
        weight.view(scale.shape[0], block_rows, scale.shape[1], block_cols)
        .mul(scale.float().view(scale.shape[0], 1, scale.shape[1], 1))
        .reshape(weight.shape)
    )


class DeepseekV4CheckpointTensorConverter:
    """Converts per-expert split checkpoint keys to stacked & merged v5 format.

    Buffers per-expert tensors as they stream from safetensor files, and emits
    merged tensors once all experts for a given (layer, projection) are collected.

    Args:
        num_experts: Number of routed experts per MoE layer
            (``config.n_routed_experts`` — equal to ``config.num_local_experts``
            via ``DeepseekV4Config.attribute_map``).
    """

    def __init__(self, num_experts: int, *, target_model_prefix: str = "model."):
        self.num_experts = num_experts
        self.target_model_prefix = target_model_prefix
        # {(prefix, proj_name): {expert_id: tensor}}
        self._expert_buffer: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
        # {prefix: {proj_name: stacked_tensor}} for gate/up merge waiting
        self._stacked_buffer: dict[str, dict[str, torch.Tensor]] = {}
        # {weight_name: {"weight": tensor, "scale": tensor}} for FP8/int8 checkpoint pairs.
        self._scaled_weight_buffer: dict[str, dict[str, torch.Tensor | str | bool]] = {}

    def can_handle(self, name: str) -> bool:
        if _is_mtp_checkpoint_key(name):
            return False
        converted_name = convert_deepseek_v4_checkpoint_key(name, target_model_prefix=self.target_model_prefix)
        return (
            converted_name != name
            or bool(_EXPERT_PATTERN.match(converted_name))
            or _weight_name_from_scale_name(converted_name) is not None
        )

    def can_handle_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        if _is_mtp_checkpoint_key(name):
            return False
        converted_name = convert_deepseek_v4_checkpoint_key(name, target_model_prefix=self.target_model_prefix)
        return self.can_handle(name) or (converted_name.endswith(".weight") and _is_quantized_weight(tensor))

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        if _is_mtp_checkpoint_key(name):
            return ConvertedCheckpointTensor(name, tensor)
        packed_fp4 = tensor.dtype == torch.int8 and bool(_RAW_FP4_EXPERT_WEIGHT_PATTERN.fullmatch(name))
        name = convert_deepseek_v4_checkpoint_key(name, target_model_prefix=self.target_model_prefix)
        weight_name = _weight_name_from_scale_name(name)
        if weight_name is not None:
            if not _is_block_scale_tensor(tensor):
                return ConvertedCheckpointTensor(name, tensor)
            return self._convert_scaled_weight_part(weight_name, "scale", tensor, name)

        if name.endswith(".weight") and _is_quantized_weight(tensor):
            return self._convert_scaled_weight_part(name, "weight", tensor, None, packed_fp4=packed_fp4)

        return self._convert_weight(name, tensor)

    def _convert_scaled_weight_part(
        self,
        weight_name: str,
        part_name: str,
        tensor: torch.Tensor,
        scale_name: str | None,
        *,
        packed_fp4: bool = False,
    ) -> ConvertedCheckpointTensor | None:
        parts = self._scaled_weight_buffer.setdefault(weight_name, {})
        parts[part_name] = tensor
        if part_name == "weight":
            parts["packed_fp4"] = packed_fp4
        if scale_name is not None:
            parts["scale_name"] = scale_name
        if "weight" not in parts or "scale" not in parts:
            return None

        weight = parts["weight"]
        scale = parts["scale"]
        del self._scaled_weight_buffer[weight_name]
        assert isinstance(weight, torch.Tensor)
        assert isinstance(scale, torch.Tensor)
        return self._convert_weight(
            weight_name,
            _dequantize_scaled_weight(weight, scale, packed_fp4=bool(parts.get("packed_fp4", False))),
        )

    def _convert_weight(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        match = _EXPERT_PATTERN.match(name)
        if not match:
            return ConvertedCheckpointTensor(name, tensor)

        prefix, expert_id_str, proj_name = match.groups()
        proj_name = _PROJ_NAME_ALIASES.get(proj_name, proj_name)
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

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        """Validate that all buffers were flushed.

        Raises RuntimeError if any buffers remain unflushed, since incomplete
        expert tensors cannot be merged into valid fused format and indicate
        a corrupted or incomplete checkpoint.
        """
        errors: list[str] = []
        if self._expert_buffer:
            unflushed = {k: len(v) for k, v in self._expert_buffer.items()}
            errors.append(
                f"unflushed per-expert buffer (incomplete experts, expected {self.num_experts}): {unflushed}"
            )
        if self._stacked_buffer:
            unflushed = {k: list(v.keys()) for k, v in self._stacked_buffer.items()}
            errors.append(f"unflushed stacked buffer (missing gate/up pair): {unflushed}")
        finalized: list[ConvertedCheckpointTensor] = []
        for weight_name, parts in self._scaled_weight_buffer.items():
            if "weight" in parts:
                errors.append(f"unflushed scaled weight buffer (missing scale for {weight_name})")
                continue
            scale = parts.get("scale")
            scale_name = parts.get("scale_name")
            if isinstance(scale, torch.Tensor) and isinstance(scale_name, str):
                finalized.append(ConvertedCheckpointTensor(scale_name, scale))
        if errors:
            raise RuntimeError("DeepseekV4 checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors))
        self._scaled_weight_buffer.clear()
        return finalized


def create_deepseek_v4_checkpoint_tensor_converter(model):
    """Factory function registered on model classes via _create_checkpoint_tensor_converter."""
    return DeepseekV4CheckpointTensorConverter(
        num_experts=model.config.n_routed_experts,
        target_model_prefix=_get_deepseek_v4_target_model_prefix(model),
    )


def _get_deepseek_v4_target_model_prefix(model) -> str:
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        for name, _ in named_parameters():
            if name.startswith("model."):
                return "model."
            if name.startswith(_BASE_MODEL_KEY_PREFIXES):
                return ""

    if type(model).__name__.endswith("ForCausalLM"):
        return "model."
    return ""


def convert_deepseek_v4_fqn_to_index_mapping(
    fqn_to_index_mapping: dict[str, int], *, target_model_prefix: str = "model."
) -> dict[str, int]:
    """Align HF safetensors index keys with fused expert parameter names."""
    gate_up_shard_indices: dict[str, list[int]] = defaultdict(list)
    down_shard_indices: dict[str, list[int]] = defaultdict(list)
    converted: dict[str, int] = {}

    for fqn, shard_idx in fqn_to_index_mapping.items():
        fqn = convert_deepseek_v4_checkpoint_key(fqn, target_model_prefix=target_model_prefix)
        match = _EXPERT_PATTERN.match(fqn)
        if not match:
            converted[fqn] = shard_idx
            continue

        prefix, _expert_id, proj_name = match.groups()
        proj_name = _PROJ_NAME_ALIASES.get(proj_name, proj_name)
        if proj_name == "down_proj":
            down_shard_indices[prefix].append(shard_idx)
        else:
            gate_up_shard_indices[prefix].append(shard_idx)

    for prefix, indices in down_shard_indices.items():
        converted[f"{prefix}.experts.down_proj"] = min(indices)

    for prefix, indices in gate_up_shard_indices.items():
        converted[f"{prefix}.experts.gate_up_proj"] = min(indices)

    return converted
