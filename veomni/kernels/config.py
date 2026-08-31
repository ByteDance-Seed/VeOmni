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

"""Installed kernel-impl config.

One process-global object. ``apply_ops_config`` still writes it. Modeling
reads ``get_kernels_config``. ``get_ops_config`` is the same store for
unconsumed OpSlot models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig


_kernels_config: OpsImplementationConfig | None = None


def set_kernels_config(config: Any) -> None:
    """Install the process-global kernel-impl config."""
    global _kernels_config
    _kernels_config = config


def get_kernels_config() -> Any:
    """Return the installed kernel-impl config, or ``None``."""
    return _kernels_config
