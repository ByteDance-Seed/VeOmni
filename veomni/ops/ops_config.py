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

"""Global ops implementation config singleton.

This module stores the resolved ``OpsImplementationConfig`` so that model
``gpu_patch.py`` and ``npu_patch.py`` files can query per-operation kernel
selections without relying on environment variables.

Typical lifecycle:
1. ``OpsImplementationConfig.__post_init__`` validates requested backends.
2. ``set_ops_config(config)`` is called (from trainer or test harness).
3. ``gpu_patch.py`` / ``npu_patch.py`` call ``get_ops_config()`` to decide
   which kernels to apply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig


_ops_config: OpsImplementationConfig | None = None


def set_ops_config(config: OpsImplementationConfig) -> None:
    """Set the global ops implementation config singleton."""
    global _ops_config
    _ops_config = config


def get_ops_config() -> OpsImplementationConfig | None:
    """Return the global ops implementation config, or ``None`` if not yet set."""
    return _ops_config
