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
"""Hardware-specific launch geometry for Ascend gated delta-rule kernels."""

from __future__ import annotations

from functools import cache


_ASCEND_910B4_DEVICE_NAMES = frozenset({"Ascend910B4", "Ascend910B4-1"})
_DEFAULT_HIDDEN_STATE_BLOCK_VALUE = 128
_ASCEND_910B4_HIDDEN_STATE_BLOCK_VALUE = 64


def select_hidden_state_block_value(device_name: str) -> int:
    """Select the GDR hidden-state value tile for a raw Ascend device name.

    The recurrence kernel keeps its ``[64, BV]`` hidden-state tiles in the
    unified buffer. ``BV=128`` exceeds the 910B4-1 limit at Qwen3.5 training
    shapes, while ``BV=64`` partitions only the independent value dimension
    and preserves the kernel's math.
    """

    if device_name in _ASCEND_910B4_DEVICE_NAMES:
        return _ASCEND_910B4_HIDDEN_STATE_BLOCK_VALUE
    return _DEFAULT_HIDDEN_STATE_BLOCK_VALUE


@cache
def get_hidden_state_block_value(device_index: int) -> int:
    """Return the GDR hidden-state value tile for one explicit NPU."""

    import torch_npu

    return select_hidden_state_block_value(torch_npu.npu.get_device_name(device_index))
