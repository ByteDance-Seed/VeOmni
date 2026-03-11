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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Union

import torch
from torch import nn

from ..utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


@dataclass
class HfConvertedCheckpointTensor:
    """One converted checkpoint tensor ready for normal loading dispatch.

    Attributes:
        name: Final state-dict key after model-specific conversion.
        tensor: Tensor payload for ``name``.
    """

    name: str
    tensor: "torch.Tensor"


class HfCheckpointTensorConverter(Protocol):
    """Model-specific incremental converter for HF checkpoint tensors.

    Implementations consume one tensor at a time from checkpoint iteration and can
    choose to emit:
    - one converted tensor now, or
    - ``None`` to keep accumulating internal state until ready.

    Example (qwen3_moe on transformers v5):
    - Safetensor checkpoints are often stored in per-expert split keys, e.g.
      ``model.layers.0.mlp.experts.0.gate_proj.weight`` with shape ``[I, H]``,
      ``model.layers.0.mlp.experts.0.up_proj.weight`` with shape ``[I, H]``,
      and ``model.layers.0.mlp.experts.0.down_proj.weight`` with shape ``[H, I]``.
    - The modeling code expects merged expert tensors:
      ``model.layers.0.mlp.experts.gate_up_proj`` with shape ``[E, 2*I, H]``
      and ``model.layers.0.mlp.experts.down_proj`` with shape ``[E, H, I]``.
    - A converter accumulates incoming per-expert tensors across keys/shards,
      and emits merged tensors only when enough inputs have been collected.
    """

    def can_handle(self, name: str) -> bool:
        """Whether this converter should consume the incoming checkpoint key.

        Args:
            name: Checkpoint tensor key after key-mapping conversion.

        Returns:
            ``True`` when ``convert`` should be called for this key, otherwise ``False``.
        """

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[HfConvertedCheckpointTensor]:
        """Consume an input checkpoint tensor and optionally emit a converted tensor.

        Args:
            name: Checkpoint tensor key after key-mapping conversion.
            tensor: Tensor payload associated with ``name``.

        Returns:
            ``HfConvertedCheckpointTensor`` when a tensor is ready to dispatch/load.
            ``None`` when converter needs more input tensors before emitting.
        """


def get_hf_checkpoint_tensor_converter(
    model: Union["nn.Module", "PreTrainedModel"],
) -> Optional[HfCheckpointTensorConverter]:
    """Resolve model-registered HF checkpoint tensor converter.

    The model can optionally expose a factory hook:
    ``_create_checkpoint_tensor_converter()``.

    Args:
        model: Model instance currently loading checkpoint weights.

    Returns:
        A converter instance if available and valid, otherwise ``None``.
    """

    factory = getattr(model, "_create_checkpoint_tensor_converter", None)
    if factory is None:
        return None
    if not callable(factory):
        logger.warning_rank0("Ignore invalid `_create_checkpoint_tensor_converter` because it is not callable.")
        return None

    converter = factory()
    if converter is None:
        return None
    if not hasattr(converter, "can_handle") or not hasattr(converter, "convert"):
        logger.warning_rank0("Ignore invalid checkpoint tensor converter because it has no `can_handle/convert`.")
        return None
    return converter


def maybe_convert_hf_checkpoint_tensor(
    name: str,
    tensor: "torch.Tensor",
    converter: Optional[HfCheckpointTensorConverter],
) -> Optional[HfConvertedCheckpointTensor]:
    """Optionally convert a checkpoint tensor using model-specific converter.

    Args:
        name: Checkpoint tensor key after key-mapping conversion.
        tensor: Tensor payload associated with ``name``.
        converter: Converter instance registered by model, or ``None``.

    Returns:
        - pass-through tensor as ``HfConvertedCheckpointTensor`` if no converter or key not handled
        - converted tensor when converter emits one
        - ``None`` when converter handled key but still accumulates state
    """

    if converter is None or not converter.can_handle(name):
        return HfConvertedCheckpointTensor(name=name, tensor=tensor)
    return converter.convert(name, tensor)
