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

"""Wan patch-embedding layout. Not a config-selected VeomniOp."""

from __future__ import annotations

from torch import Tensor


def confusion_transpose(hidden_states: Tensor) -> Tensor:
    """Flatten trailing spatial axes and swap channel with sequence.

    NPU tensors use ``npu_confusion_transpose``. Every other device uses
    ``flatten(2).transpose(1, 2)``. Host availability of ``torch_npu`` does
    not change the CPU path.
    """
    if hidden_states.device.type == "npu":
        return _npu_confusion_transpose(hidden_states)
    return hidden_states.flatten(2).transpose(1, 2)


def _npu_confusion_transpose(hidden_states: Tensor) -> Tensor:
    import torch_npu

    batch, channels = hidden_states.shape[:2]
    sequence = hidden_states.shape[2:].numel()
    return torch_npu.npu_confusion_transpose(
        hidden_states,
        [0, 2, 1],
        [batch, channels, sequence],
        transpose_first=False,
    )
