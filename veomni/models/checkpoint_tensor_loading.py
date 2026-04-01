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
Checkpoint tensor converter utilities for runtime weight format conversion.

Models that need to convert HuggingFace checkpoint tensors at load time (e.g. MoE
per-expert weights -> fused format) register a ``_create_checkpoint_tensor_converter``
class attribute.  The helpers here retrieve and apply such converters during weight
loading in ``module_utils.py``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Protocol, Union

import torch
from torch import nn

from ..utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


@dataclass
class ConvertedCheckpointTensor:
    """One converted checkpoint tensor ready for dispatch."""

    name: str
    tensor: "torch.Tensor"


class CheckpointTensorConverter(Protocol):
    """Per-tensor converter protocol for runtime checkpoint format conversion.

    Models register a converter via ``_create_checkpoint_tensor_converter`` to handle
    mismatches between HF safetensor layout and modeling layout (e.g. MoE per-expert
    keys -> fused format).

    Implementations consume one tensor at a time and can choose to:
    - emit a converted tensor immediately, or
    - return ``None`` to keep accumulating internal state until ready.
    """

    def can_handle(self, name: str) -> bool:
        """Whether this converter should consume the incoming checkpoint key."""
        ...

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional["ConvertedCheckpointTensor"]:
        """Consume a tensor and optionally emit a converted result.

        Returns ``None`` when still accumulating (e.g. collecting per-expert tensors).
        """
        ...

    def finalize(self) -> List["ConvertedCheckpointTensor"]:
        """Flush remaining buffered tensors after all shards are consumed.

        Called after all checkpoint shards have been iterated. Implementations should
        warn about any unexpected unflushed state.
        """
        ...


def get_checkpoint_tensor_converter(
    model: Union["nn.Module", "PreTrainedModel"],
) -> Optional["CheckpointTensorConverter"]:
    """Return the checkpoint tensor converter registered on *model*, or ``None``."""
    factory = getattr(model, "_create_checkpoint_tensor_converter", None)
    if factory is None:
        return None
    if not callable(factory):
        logger.warning_rank0("Ignore invalid `_create_checkpoint_tensor_converter`: not callable.")
        return None

    converter = factory(model)
    if converter is None:
        return None
    if not hasattr(converter, "can_handle") or not hasattr(converter, "convert"):
        logger.warning_rank0("Ignore invalid checkpoint tensor converter: missing can_handle/convert.")
        return None
    return converter


def maybe_convert_checkpoint_tensor(
    name: str,
    tensor: "torch.Tensor",
    converter: Optional["CheckpointTensorConverter"],
) -> Optional["ConvertedCheckpointTensor"]:
    """Apply converter if applicable, otherwise pass through.

    Returns:
        ``ConvertedCheckpointTensor`` if tensor is ready for dispatch (pass-through or converted).
        ``None`` if converter consumed the tensor but is still accumulating.
    """
    if converter is None or not converter.can_handle(name):
        return ConvertedCheckpointTensor(name=name, tensor=tensor)
    return converter.convert(name, tensor)
