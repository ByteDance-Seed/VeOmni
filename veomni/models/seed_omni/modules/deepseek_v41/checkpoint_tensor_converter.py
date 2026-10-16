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

"""Load-time dequant for the split DeepSeek-V4.1 checkpoint.

The converter writes the released dtypes (fp8 dense, packed fp4 experts, fp8
n-gram tables) and carries each scale as ``<param>_scale``. Training builds
ordinary bf16 parameters -- the same graph DeepSeek-V4 uses -- so this hook
spends the scale while the tensors stream in, the way
``DeepseekV4CheckpointTensorConverter`` does. Inference keeps
``config.weight_quantization`` and does not install the hook: the quantized
modules already match the file.

The split checkpoint has already fused experts and renamed keys, so this
converter only pairs a weight with its scale. It does not restack ``w1``/``w2``/``w3``.
"""

from __future__ import annotations

import re

import torch

from ....checkpoint_tensor_loading import ConvertedCheckpointTensor
from .dequant import dequant_fp4_blockwise, dequant_fp8_blockwise
from .llm.configuration import (
    RELEASED_ENGRAM_LAYER_IDS,
    RELEASED_NUM_HIDDEN_LAYERS,
    shrink_layer_schedule,
)
from .quant_scheme import quantized_storage_dtype


_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.*)$")
_ENGRAM_KEY = re.compile(r"^model\.engram\.(\d+)\.(.*)$")


def _shrink_maps(config) -> tuple[dict[int, int], dict[int, int], bool] | None:
    """Layer / Engram old->new maps, plus whether draft stages are dropped.

    ``None`` means the live module is the released 40-layer stack: every
    checkpoint key is already the live FQN. A shorter stack keeps the CED
    layers ``shrink_layer_schedule`` selected (released 2 and 20 when N=2),
    not ``layers.0`` / ``layers.1`` which would be the first two encoder
    layers and would break the decoder-start candidate source.
    """
    n_layers = getattr(config, "num_hidden_layers", RELEASED_NUM_HIDDEN_LAYERS)
    n_draft = getattr(config, "num_nextn_predict_layers", 0)
    if n_layers == RELEASED_NUM_HIDDEN_LAYERS:
        return None
    schedule = shrink_layer_schedule(n_layers, n_draft=n_draft)
    layer_map = {orig: new for new, orig in enumerate(schedule["kept_layer_ids"])}
    engram_map = dict(zip(RELEASED_ENGRAM_LAYER_IDS, schedule["engram_layer_ids"], strict=False))
    return layer_map, engram_map, n_draft == 0


def remap_shrunk_checkpoint_key(
    name: str,
    layer_map: dict[int, int],
    engram_map: dict[int, int],
    *,
    drop_mtp: bool,
) -> str | None:
    """Rewrite a released FQN onto the shrunk module, or ``None`` to drop it."""
    if drop_mtp and (name == "model.mtp" or name.startswith("model.mtp.")):
        return None
    match = _LAYER_KEY.match(name)
    if match is not None:
        orig = int(match.group(1))
        if orig not in layer_map:
            return None
        return f"model.layers.{layer_map[orig]}.{match.group(2)}"
    match = _ENGRAM_KEY.match(name)
    if match is not None:
        orig = int(match.group(1))
        if orig not in engram_map:
            return None
        return f"model.engram.{engram_map[orig]}.{match.group(2)}"
    return name


_FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, name, None)
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu")
    )
    if dtype is not None
)
_QUANTIZED_WEIGHT_DTYPES = _FLOAT8_DTYPES + (torch.int8, torch.uint8)


def scale_key(weight_name: str) -> str:
    """Scale sibling written by the split converter: ``P`` -> ``P_scale``."""
    return f"{weight_name}_scale"


def weight_name_from_scale_name(name: str) -> str | None:
    if name.endswith("_scale"):
        return name[: -len("_scale")]
    return None


def is_quantized_weight(tensor: torch.Tensor) -> bool:
    return tensor.dtype in _QUANTIZED_WEIGHT_DTYPES


def dequant_converted_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Undo one converted tensor's quantization, including stacked expert slabs."""
    packed = weight.dtype in (torch.int8, torch.uint8)
    if weight.ndim == 3:
        rows = [
            dequant_fp4_blockwise(weight[i], scale[i]) if packed else dequant_fp8_blockwise(weight[i], scale[i])
            for i in range(weight.shape[0])
        ]
        return torch.stack(rows)
    if packed:
        return dequant_fp4_blockwise(weight, scale)
    return dequant_fp8_blockwise(weight, scale)


class DeepseekV41DequantConverter:
    """Pair ``P`` with ``P_scale`` and emit the bf16 weight training expects."""

    def __init__(self, config=None) -> None:
        self._parts: dict[str, dict[str, torch.Tensor]] = {}
        maps = _shrink_maps(config) if config is not None else None
        if maps is None:
            self._layer_map: dict[int, int] | None = None
            self._engram_map: dict[int, int] | None = None
            self._drop_mtp = False
        else:
            self._layer_map, self._engram_map, self._drop_mtp = maps

    def _rewrite(self, name: str) -> str | None:
        if self._layer_map is None:
            if self._drop_mtp and (name == "model.mtp" or name.startswith("model.mtp.")):
                return None
            return name
        return remap_shrunk_checkpoint_key(name, self._layer_map, self._engram_map, drop_mtp=self._drop_mtp)

    def can_handle(self, name: str) -> bool:
        # A shrunk stack renames every key of a kept layer, unquantized ones included.
        if self._rewrite(name) != name:
            return True
        return weight_name_from_scale_name(name) is not None

    def can_handle_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        if self._rewrite(name) != name:
            return True
        if weight_name_from_scale_name(name) is not None:
            return True
        return is_quantized_weight(tensor)

    def dim0_rowwise_companion(self, name: str) -> str | None:
        """Each weight's scale rows follow its own dim-0 rows (experts, or output rows).

        The streaming loader asks this for every key and uses the answer only
        when the checkpoint holds the scale, so a bf16 tensor reads as is. A
        shrunk stack renames its layers, so it offers no companion: every
        renamed key is converted whole, and the streaming loader refuses one
        that lands on an ExtraParallel parameter (a fused expert slab).
        """
        if self._layer_map is not None or self._rewrite(name) is None:
            return None
        if weight_name_from_scale_name(name) is not None:
            return None
        return scale_key(name)

    def convert_rows(self, name: str, rows: torch.Tensor, companion_rows: torch.Tensor) -> torch.Tensor:
        if rows.shape[0] == 0:
            return torch.empty(rows.shape, dtype=torch.bfloat16)
        return dequant_converted_weight(rows, companion_rows)

    def should_skip_without_loading(self, name: str) -> bool:
        return self._rewrite(name) is None

    def record_skip_without_loading(self, name: str) -> None:
        pass

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        live = self._rewrite(name)
        if live is None:
            return None
        weight_name = weight_name_from_scale_name(live)
        if weight_name is not None:
            return self._take(weight_name, "scale", tensor)
        if is_quantized_weight(tensor):
            return self._take(live, "weight", tensor)
        return ConvertedCheckpointTensor(live, tensor)

    def _take(self, weight_name: str, part: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        parts = self._parts.setdefault(weight_name, {})
        parts[part] = tensor
        if "weight" not in parts or "scale" not in parts:
            return None
        weight = parts["weight"]
        scale = parts["scale"]
        del self._parts[weight_name]
        return ConvertedCheckpointTensor(weight_name, dequant_converted_weight(weight, scale))

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        if not self._parts:
            return []
        leftover = ", ".join(sorted(self._parts))
        raise RuntimeError(
            "DeepseekV41 dequant converter: incomplete checkpoint, unpaired weight/scale for " + leftover
        )


#: One released expert tensor, after the HF layout's prefix rename:
#: ``model.layers.3.mlp.experts.17.w1.weight`` / ``...w1.scale``.
_RELEASED_EXPERT_KEY = re.compile(
    r"^(?P<prefix>model\.(?P<kind>layers|mtp)\.\d+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>w[123])\.(?P<part>weight|scale)$"
)
_FUSED_PROJ = {"w1": "gate_up_proj", "w3": "gate_up_proj", "w2": "down_proj"}


class DeepseekV41LlmConverter:
    """The llm's load converter: dense dequant plus the release's per-expert stacking.

    A split checkpoint already holds fused experts and only needs the dense
    weight/scale pairing (:class:`DeepseekV41DequantConverter`, training only).
    The release, read in place through the HF layout, holds each expert's
    ``w1``/``w2``/``w3`` and its ``scale`` separately; this stacks them into
    ``gate_up_proj`` (``w1`` rows then ``w3`` rows, per expert) and
    ``down_proj``:

    * ``dequantize=True`` (training): each expert is dequantized to bf16 as
      soon as its weight and scale have both arrived, then stacked;
    * ``dequantize=False`` (quantized inference): the packed fp4 bytes and the
      scales are stacked separately, into ``<fused>`` and ``<fused>_scale``.

    Main layers have ``n_routed_experts`` experts and DSpark stages
    ``dspark_n_routed_experts``. :meth:`for_expert_range` restricts a copy to
    experts ``[start, start + num_local)`` of every fused target it is given,
    so the per-rank streaming loader reads only its own experts.
    """

    def __init__(
        self,
        config,
        *,
        dequantize: bool,
        expert_range: tuple[int, int] | None = None,
    ) -> None:
        self.config = config
        self.dequantize = dequantize
        self.expert_range = expert_range
        self.dense = DeepseekV41DequantConverter(config) if dequantize else None
        # The shrunk-stack remap applies to a quantized module too, which dequantizes nothing.
        self._remap = self.dense if self.dense is not None else DeepseekV41DequantConverter(config)
        self._num_experts = {
            "layers": config.n_routed_experts,
            "mtp": getattr(config, "dspark_n_routed_experts", None) or config.n_routed_experts,
        }
        # {(prefix, proj, part): {local_expert: tensor}}; part is "value" once dequantized.
        self._stacks: dict[tuple[str, str, str], dict[int, torch.Tensor]] = {}
        # {(prefix, proj, local_expert): {"weight"|"scale": tensor}} awaiting their pair.
        self._pairs: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}
        # {(prefix, fused, part): {proj: stacked}} awaiting the other half of gate_up.
        self._fused: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}

    def _rewrite(self, name: str) -> str | None:
        return self._remap._rewrite(name)

    def _expert(self, name: str) -> re.Match | None:
        live = self._rewrite(name)
        return None if live is None else _RELEASED_EXPERT_KEY.match(live)

    def _local_count(self, kind: str) -> int:
        return self.expert_range[1] if self.expert_range is not None else self._num_experts[kind]

    def can_handle(self, name: str) -> bool:
        if self._expert(name) is not None:
            return True
        if self.dense is None:
            return self._rewrite(name) != name
        return self.dense.can_handle(name)

    def can_handle_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        if self._expert(name) is not None:
            return True
        return self.dense is not None and self.dense.can_handle_tensor(name, tensor)

    def dim0_rowwise_companion(self, name: str) -> str | None:
        if self.dense is None or self._expert(name) is not None:
            return None
        return self.dense.dim0_rowwise_companion(name)

    def convert_rows(self, name: str, rows: torch.Tensor, companion_rows: torch.Tensor) -> torch.Tensor:
        return self.dense.convert_rows(name, rows, companion_rows)

    def should_skip_without_loading(self, name: str) -> bool:
        return self._rewrite(name) is None

    def record_skip_without_loading(self, name: str) -> None:
        pass

    def fused_expert_target(self, name: str) -> tuple[str, int] | None:
        match = self._expert(name)
        if match is None:
            return None
        fused = f"{match['prefix']}.{_FUSED_PROJ[match['proj']]}"
        if not self.dequantize and match["part"] == "scale":
            fused = scale_key(fused)
        return fused, int(match["expert"])

    def for_expert_range(self, start: int, num_local: int) -> DeepseekV41LlmConverter:
        offset = self.expert_range[0] if self.expert_range is not None else 0
        return DeepseekV41LlmConverter(
            self.config, dequantize=self.dequantize, expert_range=(offset + start, num_local)
        )

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        match = self._expert(name)
        if match is None:
            if self.dense is None:
                live = self._rewrite(name)
                return None if live is None else ConvertedCheckpointTensor(live, tensor)
            return self.dense.convert(name, tensor)

        prefix, kind, proj, part = match["prefix"], match["kind"], match["proj"], match["part"]
        expert = int(match["expert"])
        start = self.expert_range[0] if self.expert_range is not None else 0
        local = expert - start
        if not 0 <= local < self._local_count(kind):
            raise ValueError(
                f"{name}: expert {expert} is outside this converter's range [{start}, {start + self._local_count(kind)})."
            )
        if not self.dequantize:
            return self._stack(prefix, kind, proj, part, local, tensor)
        pair = self._pairs.setdefault((prefix, proj, local), {})
        pair[part] = tensor
        if len(pair) < 2:
            return None
        del self._pairs[(prefix, proj, local)]
        return self._stack(prefix, kind, proj, "value", local, dequant_converted_weight(pair["weight"], pair["scale"]))

    def _stack(
        self, prefix: str, kind: str, proj: str, part: str, local: int, tensor: torch.Tensor
    ) -> ConvertedCheckpointTensor | None:
        experts = self._stacks.setdefault((prefix, proj, part), {})
        experts[local] = tensor
        count = self._local_count(kind)
        if len(experts) < count:
            return None
        del self._stacks[(prefix, proj, part)]
        stacked = torch.stack([experts[i] for i in range(count)])
        fused = _FUSED_PROJ[proj]
        name = f"{prefix}.{fused}" if part != "scale" else scale_key(f"{prefix}.{fused}")
        if fused == "down_proj":
            return ConvertedCheckpointTensor(name, stacked)
        halves = self._fused.setdefault((prefix, fused, part), {})
        halves[proj] = stacked
        if len(halves) < 2:
            return None
        del self._fused[(prefix, fused, part)]
        return ConvertedCheckpointTensor(name, torch.cat([halves["w1"], halves["w3"]], dim=1))

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        problems = []
        if self._pairs:
            problems.append(f"experts missing their weight or scale: {sorted(self._pairs)}")
        if self._stacks:
            problems.append(f"incomplete expert stacks: { {key: len(v) for key, v in self._stacks.items()} }")
        if self._fused:
            problems.append(
                f"gate_up halves without their pair: { {key: sorted(v) for key, v in self._fused.items()} }"
            )
        if problems:
            raise RuntimeError("DeepseekV41 llm converter: incomplete checkpoint, " + "; ".join(problems))
        return self.dense.finalize() if self.dense is not None else []


def create_deepseek_v41_llm_checkpoint_tensor_converter(model) -> DeepseekV41LlmConverter:
    """Dequantizes for the bf16 training graph; a quantized module only stacks released experts."""
    return DeepseekV41LlmConverter(model.config, dequantize=quantized_storage_dtype(model.config) is None)


__all__ = [
    "DeepseekV41DequantConverter",
    "DeepseekV41LlmConverter",
    "create_deepseek_v41_llm_checkpoint_tensor_converter",
    "dequant_converted_weight",
    "is_quantized_weight",
    "remap_shrunk_checkpoint_key",
    "scale_key",
    "weight_name_from_scale_name",
]
