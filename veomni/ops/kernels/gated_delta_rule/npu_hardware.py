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


_ASCEND_910B4_DEVICE_NAMES = frozenset({"Ascend910B4-1"})
_DEFAULT_HIDDEN_STATE_BLOCK_VALUE = 128
_ASCEND_910B4_HIDDEN_STATE_BLOCK_VALUE = 64


def select_hidden_state_block_value(device_name: str) -> int:
    """Select the GDR hidden-state value tile for a raw Ascend device name.

    This conservative candidate changes only the independent value tile on
    the exact B4-1 SKU. It is not a measured unified-buffer diagnosis; public
    Triton forward/backward parity still requires a matched hardware run.
    Mainstream B4 and unmeasured devices retain the upstream default.
    """

    if device_name in _ASCEND_910B4_DEVICE_NAMES:
        return _ASCEND_910B4_HIDDEN_STATE_BLOCK_VALUE
    return _DEFAULT_HIDDEN_STATE_BLOCK_VALUE


def get_hidden_state_block_value(device_index: int | None) -> int:
    """Resolve the current NPU before caching its hardware-specific launch tile."""
    import torch_npu

    if device_index is None:
        device_index = torch_npu.npu.current_device()
    return _get_hidden_state_block_value(device_index)


@cache
def _get_hidden_state_block_value(device_index: int) -> int:
    import torch_npu

    return select_hidden_state_block_value(torch_npu.npu.get_device_name(device_index))
