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
# See the License for the specific language governing limitations
# under the License.

"""Ops-facing aliases for the kernel-impl config singleton.

Storage lives in ``veomni.kernels.config``. These names stay for
``apply_ops_config`` and unconsumed OpSlot models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...kernels.config import get_kernels_config, set_kernels_config


if TYPE_CHECKING:
    from ...arguments.arguments_types import OpsImplementationConfig


def set_ops_config(config: OpsImplementationConfig | None) -> None:
    """Write the shared kernel-impl config."""
    set_kernels_config(config)


def get_ops_config() -> Any:
    """Read the shared kernel-impl config."""
    return get_kernels_config()
