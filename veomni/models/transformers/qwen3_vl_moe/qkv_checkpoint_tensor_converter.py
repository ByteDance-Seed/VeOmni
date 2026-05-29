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
Composite checkpoint tensor converter for Qwen3-VL-MoE.

Qwen3-VL-MoE needs **two** independent runtime conversions when loading an HF
checkpoint:

1. Expert layout normalization (HF ``[E, H, 2*I]`` → v5 ``[E, 2*I, H]`` etc.).
   Lives in the sibling ``checkpoint_tensor_converter.py`` and is kept here
   verbatim by reusing its converter class.
2. QKV projection fusion (HF three-Linear ``q/k/v_proj`` → fused
   ``qkv_proj.weight``) that mirrors the qwen3_vl converter exactly.

``CheckpointTensorConverter`` (the Protocol the loader expects) allows only a
single object to be registered per model. To compose both without disturbing
the existing expert converter, we wrap them in a thin composite that
dispatches each tensor to whichever child claims it via ``can_handle`` and
forwards ``finalize`` to both.
"""

from typing import List, Optional

import torch

from ...checkpoint_tensor_loading import ConvertedCheckpointTensor
from ..qwen3_vl.checkpoint_tensor_converter import Qwen3VLAttentionCheckpointTensorConverter
from .checkpoint_tensor_converter import (
    Qwen3VLMoeCheckpointTensorConverter,
    create_qwen3_vl_moe_checkpoint_tensor_converter,
)


class Qwen3VLMoeFullCheckpointTensorConverter:
    """Routes per-tensor to either the MoE expert converter or the QKV converter.

    Both children implement the ``CheckpointTensorConverter`` Protocol and are
    mutually exclusive in their ``can_handle`` domains (expert keys live under
    ``mlp.experts.``, QKV keys under ``self_attn.``), so the dispatch is
    unambiguous.
    """

    def __init__(
        self,
        expert_converter: "Qwen3VLMoeCheckpointTensorConverter",
        qkv_converter: "Qwen3VLAttentionCheckpointTensorConverter",
    ) -> None:
        self._expert = expert_converter
        self._qkv = qkv_converter

    def can_handle(self, name: str) -> bool:
        return self._expert.can_handle(name) or self._qkv.can_handle(name)

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        if self._expert.can_handle(name):
            return self._expert.convert(name, tensor)
        if self._qkv.can_handle(name):
            return self._qkv.convert(name, tensor)
        return None

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        # Children raise loudly on incomplete buffers; concatenate the (likely
        # empty) flushed-tensor lists so the loader sees a single sequence.
        return list(self._expert.finalize()) + list(self._qkv.finalize())


def create_qwen3_vl_moe_full_checkpoint_tensor_converter(model):
    """Factory: pair the existing MoE expert converter with a fresh QKV converter.

    Registered on ``Qwen3VLMoe*`` model classes via
    ``_create_checkpoint_tensor_converter`` from ``__init__.py``.
    """
    return Qwen3VLMoeFullCheckpointTensorConverter(
        expert_converter=create_qwen3_vl_moe_checkpoint_tensor_converter(model),
        qkv_converter=Qwen3VLAttentionCheckpointTensorConverter(),
    )
