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

Models that need to rename per-expert HF ``weight_map`` keys for sharded HF export
register a ``_convert_fqn_to_index_mapping`` class attribute (same module as the
tensor converter). Conversion runs at save time (and is cached on the model after
load when an index mapping is supplied).
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Union

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


FqnToIndexMappingConverter = Callable[[Dict[str, int]], Dict[str, int]]


def shard_index_from_filename(filename: str) -> int:
    """Parse shard index from ``model-00003-of-00014.safetensors`` style names."""
    return int(filename.split("-")[1])


def parse_fqn_to_index_mapping_from_json(safetensor_idx_path: str) -> Dict[str, int]:
    """Load ``weight_map`` from a HuggingFace ``model.safetensors.index.json`` file."""
    with open(safetensor_idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    return {fqn: shard_index_from_filename(filename) for fqn, filename in weight_map.items()}


def _unwrap_model_for_fqn_mapping(
    model: Union["nn.Module", "PreTrainedModel"],
) -> Union["nn.Module", "PreTrainedModel"]:
    """Unwrap PEFT wrappers so class-level converters registered on the base model are found."""
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def get_fqn_to_index_mapping_converter(
    model: Union["nn.Module", "PreTrainedModel"],
) -> Optional[FqnToIndexMappingConverter]:
    """Return ``_convert_fqn_to_index_mapping`` registered on the model class, if any."""
    model = _unwrap_model_for_fqn_mapping(model)
    for model_cls in type(model).__mro__:
        converter = getattr(model_cls, "_convert_fqn_to_index_mapping", None)
        if converter is not None and callable(converter):
            return converter
    return None


def maybe_convert_fqn_to_index_mapping(
    fqn_to_index_mapping: Optional[Dict[str, int]],
    model: Optional[Union["nn.Module", "PreTrainedModel"]],
) -> Optional[Dict[str, int]]:
    """Convert HF index keys to match the live model's parameter names when needed."""
    if not fqn_to_index_mapping or model is None:
        return fqn_to_index_mapping

    converter = get_fqn_to_index_mapping_converter(model)
    if converter is None:
        return fqn_to_index_mapping

    original_size = len(fqn_to_index_mapping)
    converted = converter(fqn_to_index_mapping)
    if len(converted) != original_size:
        logger.info_rank0(
            f"Converted fqn_to_index_mapping for {type(model).__name__} ({original_size} -> {len(converted)} entries)."
        )
    return converted


def prepare_fqn_to_index_mapping_for_model(
    model: Union["nn.Module", "PreTrainedModel"],
    fqn_to_index_mapping: Optional[Dict[str, int]],
) -> Optional[Dict[str, int]]:
    """Convert and cache ``fqn_to_index_mapping`` on *model* for later HF export."""
    prepared = maybe_convert_fqn_to_index_mapping(fqn_to_index_mapping, model)
    if prepared is not None:
        model._veomni_prepared_fqn_to_index_mapping = prepared
    return prepared


def resolve_fqn_to_index_mapping_for_save(
    model: Optional[Union["nn.Module", "PreTrainedModel"]],
    fqn_to_index_mapping: Optional[Dict[str, int]],
) -> Optional[Dict[str, int]]:
    """Resolve mapping for HF safetensor save: use cache from load, else convert now."""
    if model is not None:
        cached = getattr(model, "_veomni_prepared_fqn_to_index_mapping", None)
        if cached is not None:
            return cached
    return maybe_convert_fqn_to_index_mapping(fqn_to_index_mapping, model)


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
