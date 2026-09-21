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

from typing import Dict, List, Optional

import torch

from ....checkpoint_tensor_loading import ConvertedCheckpointTensor
from .dequant import dequant_fp4_blockwise, dequant_fp8_blockwise
from .quant_scheme import quantized_storage_dtype


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


def weight_name_from_scale_name(name: str) -> Optional[str]:
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

    def __init__(self) -> None:
        self._parts: Dict[str, Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return weight_name_from_scale_name(name) is not None

    def can_handle_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        if weight_name_from_scale_name(name) is not None:
            return True
        return is_quantized_weight(tensor)

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        weight_name = weight_name_from_scale_name(name)
        if weight_name is not None:
            return self._take(weight_name, "scale", tensor)
        if is_quantized_weight(tensor):
            return self._take(name, "weight", tensor)
        return ConvertedCheckpointTensor(name, tensor)

    def _take(self, weight_name: str, part: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        parts = self._parts.setdefault(weight_name, {})
        parts[part] = tensor
        if "weight" not in parts or "scale" not in parts:
            return None
        weight = parts["weight"]
        scale = parts["scale"]
        del self._parts[weight_name]
        return ConvertedCheckpointTensor(weight_name, dequant_converted_weight(weight, scale))

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        if not self._parts:
            return []
        leftover = ", ".join(sorted(self._parts))
        raise RuntimeError(
            "DeepseekV41 dequant converter: incomplete checkpoint, unpaired weight/scale for " + leftover
        )


def create_deepseek_v41_llm_checkpoint_tensor_converter(model) -> Optional[DeepseekV41DequantConverter]:
    """Installed on the bf16 training graph; skipped when the module is quantized."""
    if quantized_storage_dtype(model.config) is not None:
        return None
    return DeepseekV41DequantConverter()


__all__ = [
    "DeepseekV41DequantConverter",
    "create_deepseek_v41_llm_checkpoint_tensor_converter",
    "dequant_converted_weight",
    "is_quantized_weight",
    "scale_key",
    "weight_name_from_scale_name",
]
