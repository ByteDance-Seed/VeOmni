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
        if self._rewrite(name) is None:
            return True
        return weight_name_from_scale_name(name) is not None

    def can_handle_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        if self._rewrite(name) is None:
            return True
        if weight_name_from_scale_name(name) is not None:
            return True
        return is_quantized_weight(tensor)

    def dim0_rowwise_companion(self, name: str) -> str | None:
        """Each weight's scale rows follow its own dim-0 rows (experts, or output rows).

        The streaming loader asks this for every key and uses the answer only
        when the checkpoint holds the scale, so a bf16 tensor reads as is. A
        shrunk stack renames its layers, which only the whole-tensor loader
        applies; offering no companion there leaves its scales unclaimed, and
        the loader refuses them.
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
        if self._rewrite(name) is None:
            return True
        return self._layer_map is None and weight_name_from_scale_name(name) is not None

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


def create_deepseek_v41_llm_checkpoint_tensor_converter(model) -> DeepseekV41DequantConverter | None:
    """Installed on the bf16 training graph; skipped when the module is quantized."""
    if quantized_storage_dtype(model.config) is not None:
        return None
    return DeepseekV41DequantConverter(model.config)


__all__ = [
    "DeepseekV41DequantConverter",
    "create_deepseek_v41_llm_checkpoint_tensor_converter",
    "dequant_converted_weight",
    "is_quantized_weight",
    "remap_shrunk_checkpoint_key",
    "scale_key",
    "weight_name_from_scale_name",
]
